"""K' = G' + Mac SAM2 refine.

G'의 bbox 후보를 Mac local SAM2로 refine — 면적·중심점 정보 추가.
"""
import os, json, base64, time, unicodedata, sys, subprocess
from datetime import datetime
from collections import Counter, defaultdict
from openai import OpenAI
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'framework'))
import v6_voting_debate_4 as v6m
from common_utils import parse_json_v2, get_bbox_prompt
from g_prime import (ENDPOINTS, TEST_IMAGES_FULL, BASE_IMG, BASE_LBL, TS_MAP,
                     fp, load_gt, call_vlm, iou, bbox_to_9region,
                     union_with_bbox_merge, eval_final)
import english_prompts as enp

SAM2_VENV = "python3"


def sam2_predict_bbox(img_path, bbox_orig):
    """Mac SAM2 — 주어진 bbox에서 정확한 mask 추출."""
    tmp_out = f"/tmp/sam2_{os.getpid()}.json"
    x1, y1, x2, y2 = [int(v) for v in bbox_orig]
    script = f"""
import json, numpy as np, torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
device = "mps" if torch.backends.mps.is_available() else "cpu"
sam2 = build_sam2(
    config_file="configs/sam2.1/sam2.1_hiera_b+.yaml",
    ckpt_path="./sam2_models/sam2.1_hiera_base_plus.pt",
    device=device, apply_postprocessing=False,
)
predictor = SAM2ImagePredictor(sam2)
image = np.array(Image.open({img_path!r}).convert("RGB"))
predictor.set_image(image)
masks, scores, _ = predictor.predict(box=np.array([{x1},{y1},{x2},{y2}]), multimask_output=False)
mask = masks[0]
h, w = image.shape[:2]
ys, xs = np.where(mask)
area_ratio = float(mask.sum()) / (h*w)
cx = float(xs.mean()) if len(xs) else (x1+x2)/2
cy = float(ys.mean()) if len(ys) else (y1+y2)/2
with open({tmp_out!r}, "w") as f:
    json.dump({{"area_ratio": area_ratio, "cx": cx, "cy": cy, "h": h, "w": w, "score": float(scores[0])}}, f)
"""
    try:
        r = subprocess.run([SAM2_VENV, "-c", script], capture_output=True, text=True, timeout=30)
        if r.returncode != 0: return None
        with open(tmp_out) as f:
            return json.load(f)
    except Exception:
        return None


def make_judge_with_sam_en(cat, merged_with_sam):
    desc = enp.CATEGORY_DESC_EN.get(cat, "HTP drawing")
    items_str = ""
    for c in merged_with_sam:
        sam = c.get('sam', {})
        area = sam.get('area_ratio', 0) * 100 if sam else 0
        pos = c.get('sam_pos9') or c.get('pos_9', '')
        items_str += (f"  - '{c['obj']}' position={pos} area(SAM2)={area:.1f}% models={c['models']}\n")
    return f"""You are a Bbox+SAM2 Judge for {desc}.

Three VLMs output bounding boxes. SAM2 refined each bbox to provide precise area·position.

Candidates:
{items_str}

For each (image visible):
- Confirm if actually present
- Area < 0.3% → likely hallucination (small region)
- Area > 50% with overlapping smaller objects → parent object
- Same object variants unified
- Object names in **Korean**

```json
[
  {{
    "객체": "(Korean noun)",
    "최종_판정": "있음" or "없음",
    "위치": "(9-region)",
    "근거": "..."
  }}
]
```

Position: top-left|top-center|top-right|middle-left|middle-center|middle-right|bottom-left|bottom-center|bottom-right"""


def main():
    ORCH = "qwen"
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    TEST = TEST_IMAGES_FULL[:N] if N != 4 else [
        ("TL_나무","나무_8_남_01445.jpg"),
        ("TL_집","집_12_여_08971.jpg"),
        ("TL_남자사람","남자사람_13_남_02804.jpg"),
        ("TL_여자사람","여자사람_10_남_02125.jpg"),
    ]
    MODELS = ["qwen", "exaone", "gemma"]
    print(f"=== K' (G' + Mac SAM2 refine) — Judge: {ORCH}, N={len(TEST)} ===")
    print(f"시작: {datetime.now().strftime('%H:%M:%S')}")

    results = []
    for idx, (cat, img) in enumerate(TEST):
        print(f"\n[{idx+1}/{len(TEST)}] {cat}/{img}", flush=True)
        img_path = fp(BASE_IMG, [TS_MAP[cat], img])
        gt = load_gt(cat, img); gt_n = len(gt)
        from PIL import Image as PILImage
        im = PILImage.open(img_path); W, H = im.size

        # 1. 3 모델 bbox grounding (모델별 prompt)
        bbox_outs = {}
        for m in MODELS:
            t0 = time.time()
            raw = call_vlm(m, get_bbox_prompt(cat, m), img_path=img_path, temp=0.2)
            parsed = parse_json_v2(raw)
            n_obj = len(parsed) if isinstance(parsed, list) else 0
            print(f"  Bbox {m}: {time.time()-t0:.1f}s, {n_obj} 객체", flush=True)
            bbox_outs[m] = parsed if isinstance(parsed, list) else []

        # 2. Union + IoU merge
        merged = union_with_bbox_merge(bbox_outs)
        print(f"  Union+IoU: {len(merged)} 후보", flush=True)

        # 3. Mac SAM2 refine — 각 후보 bbox
        t0 = time.time()
        sam_ok = 0
        for c in merged:
            # bbox 1000 → 원본 좌표
            bb = c['bbox']
            x1 = int(bb[0] * W / 1000); y1 = int(bb[1] * H / 1000)
            x2 = int(bb[2] * W / 1000); y2 = int(bb[3] * H / 1000)
            sam_res = sam2_predict_bbox(img_path, [x1, y1, x2, y2])
            if sam_res:
                c['sam'] = sam_res
                cx_n = sam_res['cx'] / W; cy_n = sam_res['cy'] / H
                col = "left" if cx_n < 1/3 else ("right" if cx_n > 2/3 else "center")
                row = "top" if cy_n < 1/3 else ("bottom" if cy_n > 2/3 else "middle")
                c['sam_pos9'] = f"{row}-{col}"
                sam_ok += 1
            else:
                c['sam'] = {}
        print(f"  SAM2 refine: {time.time()-t0:.1f}s ({sam_ok}/{len(merged)} 성공)", flush=True)

        # 4. Judge
        t0 = time.time()
        j_raw = call_vlm(ORCH, make_judge_with_sam_en(cat, merged), img_path=img_path, temp=0.2)
        j_parsed = parse_json_v2(j_raw)
        print(f"  Judge: {time.time()-t0:.1f}s, {len(j_parsed) if isinstance(j_parsed, list) else 0}", flush=True)

        final = {}
        if isinstance(j_parsed, list):
            for it in j_parsed:
                if not isinstance(it, dict): continue
                if it.get("최종_판정") != "있음": continue
                o = it.get("객체","").strip()
                if o and o not in final: final[o] = {"위치": it.get("위치","")}

        ev = eval_final(final, gt)
        rec = ev['tp']/gt_n if gt_n else 0
        prec = ev['tp']/(ev['tp']+ev['fp']) if (ev['tp']+ev['fp']) else 0
        f1 = 2*rec*prec/(rec+prec) if (rec+prec) else 0
        pacc = ev['pc']/ev['pt'] if ev['pt'] else 0
        acc = ev['tp']/(ev['tp']+ev['fn']+ev['fp']) if (ev['tp']+ev['fn']+ev['fp']) else 0
        print(f"  📊 TP {ev['tp']}/{gt_n}, Acc {acc*100:.1f}%, F1 {f1*100:.1f}%, 위치 {pacc*100:.1f}%, 환각 {ev['fp']}", flush=True)
        results.append({"cat":cat,"img":img,"gt_n":gt_n,"bbox_outs":bbox_outs,
                        "merged":merged,"judge":j_parsed,"final":final,"eval":ev})
        with open(f"./kprime_orch-{ORCH}_{len(TEST)}img.json","w",encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    tot = Counter()
    for r in results:
        for k in ['tp','fn','fp','pc','pt']: tot[k] += r['eval'][k]
        tot['gtn'] += r['gt_n']
    rec = tot['tp']/tot['gtn']
    prec = tot['tp']/(tot['tp']+tot['fp']) if (tot['tp']+tot['fp']) else 0
    f1 = 2*rec*prec/(rec+prec) if (rec+prec) else 0
    pacc = tot['pc']/tot['pt'] if tot['pt'] else 0
    acc = tot['tp']/(tot['tp']+tot['fn']+tot['fp'])
    print(f"\n=== K' 완료 {datetime.now().strftime('%H:%M:%S')} N={len(results)} ===")
    print(f"  Acc {acc*100:.1f}%, F1 {f1*100:.1f}%, Recall {rec*100:.1f}%, Prec {prec*100:.1f}%, 9-Pos {pacc*100:.1f}%, 환각 {tot['fp']}")


if __name__ == "__main__":
    main()
