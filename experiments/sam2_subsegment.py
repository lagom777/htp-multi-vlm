"""SAM2-SubSegment-Judge — SAM2의 진짜 강점 활용.

[1] Qwen bbox grounding (영어, 큰 객체 위주)
[2] 각 큰 bbox 내부에서 SAM2 sub-segment (부위 자동 분리)
[3] 각 sub-mask를 VLM에 라벨링 ("이 sub-region이 무엇?")
[4] 전체 bbox 라벨 + sub-mask 라벨 합쳐서 Judge

SAM2가 hierarchical sub-segmentation으로 부위 분리에 기여.
"""
import os, json, base64, time, unicodedata, sys, subprocess, io
import requests, numpy as np
from PIL import Image, ImageDraw
from datetime import datetime
from collections import Counter, defaultdict
from openai import OpenAI

sys.path.insert(0, '/Users/kg/nonmoon/htp_thesis')
import v6_voting_debate_4 as v6m
from common_utils import parse_json_v2
import english_prompts_v2 as enp2
from en_to_kr import normalize_en_to_kr
from g_prime import (ENDPOINTS, TEST_IMAGES_FULL, BASE_IMG, BASE_LBL, TS_MAP,
                     fp, load_gt, call_vlm)
from fullen_freeunion import eval_with_kr_mapping

SAM2_VENV = "/Users/kg/nonmoon/htp_thesis/.venv_sam2/bin/python"
HOST = "192.168.200.138"


def sam2_subsegment(img_path, parent_bbox, out_json):
    """parent bbox 내부에서 SAM2 sub-region 자동 추출."""
    x1, y1, x2, y2 = [int(v) for v in parent_bbox]
    script = f"""
import json, numpy as np, torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from PIL import Image
device = "mps" if torch.backends.mps.is_available() else "cpu"
sam2 = build_sam2(
    config_file="configs/sam2.1/sam2.1_hiera_b+.yaml",
    ckpt_path="/Users/kg/nonmoon/htp_thesis/sam2_models/sam2.1_hiera_base_plus.pt",
    device=device, apply_postprocessing=False,
)
mask_gen = SAM2AutomaticMaskGenerator(
    sam2, points_per_side=16,
    pred_iou_thresh=0.7, stability_score_thresh=0.85,
    min_mask_region_area=50,
)
full_img = np.array(Image.open({img_path!r}).convert("RGB"))
# parent bbox로 crop
crop = full_img[{y1}:{y2}, {x1}:{x2}]
masks = mask_gen.generate(crop) if crop.size > 0 else []
H, W = crop.shape[:2] if crop.size > 0 else (1, 1)
sub_regions = []
for m in masks:
    bb = m["bbox"]  # [x, y, w, h] in crop coordinates
    # 원본 좌표로 변환
    orig_bb = [bb[0] + {x1}, bb[1] + {y1}, bb[2], bb[3]]
    sub_regions.append({{
        "orig_bbox_xywh": orig_bb,
        "area_ratio_in_parent": m["area"] / (H*W),
        "score": float(m["predicted_iou"]),
    }})
with open({out_json!r}, "w") as f:
    json.dump({{"sub_regions": sub_regions, "parent_h": H, "parent_w": W}}, f)
"""
    try:
        r = subprocess.run([SAM2_VENV, "-c", script], capture_output=True, text=True, timeout=60)
        if r.returncode != 0: return []
        with open(out_json) as f:
            return json.load(f).get("sub_regions", [])
    except Exception:
        return []


def crop_with_marker(img_path, bbox_xywh, padding=0.15):
    im = Image.open(img_path).convert("RGB").copy()
    w, h = im.size
    x, y, bw, bh = bbox_xywh
    x1, y1, x2, y2 = x, y, x+bw, y+bh
    pw = (x2-x1) * padding; ph = (y2-y1) * padding
    cx1 = max(0, int(x1-pw)); cy1 = max(0, int(y1-ph))
    cx2 = min(w, int(x2+pw)); cy2 = min(h, int(y2+ph))
    marked = im.copy()
    draw = ImageDraw.Draw(marked)
    draw.rectangle([x1, y1, x2, y2], outline="red", width=6)
    buf = io.BytesIO()
    marked.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def call_qwen_with_b64(prompt, img_b64, temp=0.2):
    cli = OpenAI(base_url=f"http://{HOST}:8005/v1", api_key="local", timeout=300)
    r = cli.chat.completions.create(
        model="RedHatAI/Qwen3.6-35B-A3B-NVFP4",
        messages=[{"role":"user","content":[
            {"type":"text","text":prompt},
            {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{img_b64}"}}
        ]}],
        temperature=temp, max_tokens=4000,
        extra_body={"chat_template_kwargs":{"enable_thinking":False}},
    )
    return r.choices[0].message.content or ""


def bbox_xywh_to_9region(bbox_xywh, W, H):
    cx = bbox_xywh[0] + bbox_xywh[2]/2
    cy = bbox_xywh[1] + bbox_xywh[3]/2
    cx_n = cx/W; cy_n = cy/H
    col = "left" if cx_n < 1/3 else ("right" if cx_n > 2/3 else "center")
    row = "top" if cy_n < 1/3 else ("bottom" if cy_n > 2/3 else "middle")
    return f"{row}-{col}"


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    TEST = TEST_IMAGES_FULL[:N] if N != 4 else [
        ("TL_나무","나무_8_남_01445.jpg"),
        ("TL_집","집_12_여_08971.jpg"),
        ("TL_남자사람","남자사람_13_남_02804.jpg"),
        ("TL_여자사람","여자사람_10_남_02125.jpg"),
    ]
    print(f"=== SAM2-SubSegment-Judge — Qwen Judge, N={len(TEST)} ===")
    print(f"시작: {datetime.now().strftime('%H:%M:%S')}")

    results = []
    for idx, (cat, img) in enumerate(TEST):
        print(f"\n[{idx+1}/{len(TEST)}] {cat}/{img}", flush=True)
        img_path = fp(BASE_IMG, [TS_MAP[cat], img])
        gt = load_gt(cat, img); gt_n = len(gt)
        im = Image.open(img_path); W, H = im.size

        # 1. Qwen이 큰 객체 bbox 식별 (영어)
        t0 = time.time()
        bbox_raw = call_vlm("qwen", enp2.make_bbox_prompt_full_en(cat),
                            img_path=img_path, temp=0.2)
        bbox_parsed = parse_json_v2(bbox_raw)
        n_main = len(bbox_parsed) if isinstance(bbox_parsed, list) else 0
        print(f"  Qwen bbox: {time.time()-t0:.1f}s, {n_main} 큰 객체", flush=True)

        # 2. 각 큰 bbox에 SAM2 sub-segment
        all_labels = []
        t_sam = 0
        if isinstance(bbox_parsed, list):
            for bi, b_item in enumerate(bbox_parsed[:15]):  # 최대 15 객체
                obj = b_item.get("object","").strip().lower()
                bbox_1000 = b_item.get("bbox",[])
                if len(bbox_1000) < 4: continue
                # 1000 → 원본 좌표
                x1 = int(bbox_1000[0] * W / 1000); y1 = int(bbox_1000[1] * H / 1000)
                x2 = int(bbox_1000[2] * W / 1000); y2 = int(bbox_1000[3] * H / 1000)
                # bbox 자체 라벨 (parent)
                pos9 = bbox_xywh_to_9region([x1,y1,x2-x1,y2-y1], W, H)
                all_labels.append({
                    "object_en": obj,
                    "object_kr": normalize_en_to_kr(obj),
                    "bbox": [x1,y1,x2,y2],
                    "pos9": pos9,
                    "source": "qwen_bbox",
                })
                # SAM2 sub-segment
                sam_out = f"/tmp/sam2_sub_{os.getpid()}_{bi}.json"
                t0 = time.time()
                subs = sam2_subsegment(img_path, [x1,y1,x2,y2], sam_out)
                t_sam += time.time() - t0
                # sub region 라벨링 (Qwen에게)
                for si, sub in enumerate(subs[:8]):  # 최대 8 sub per parent
                    if sub['area_ratio_in_parent'] < 0.02: continue
                    sub_bb = sub['orig_bbox_xywh']
                    sub_pos9 = bbox_xywh_to_9region(sub_bb, W, H)
                    crop_b64 = crop_with_marker(img_path, sub_bb, padding=0.1)
                    sub_prompt = f"""In this HTP {enp2.CATEGORY_DESC_EN[cat]}, a sub-region is marked with RED box (inside larger object '{obj}').
What specific part is in the RED box? English single noun only.
```json
{{"object": "english_noun", "evidence": "..."}}
```"""
                    sub_raw = call_qwen_with_b64(sub_prompt, crop_b64, temp=0.2)
                    sub_parsed = parse_json_v2(sub_raw)
                    if isinstance(sub_parsed, dict):
                        sub_obj = sub_parsed.get("object","").strip().lower()
                        if sub_obj and sub_obj != obj:  # 부모와 다른 객체
                            all_labels.append({
                                "object_en": sub_obj,
                                "object_kr": normalize_en_to_kr(sub_obj),
                                "bbox": [sub_bb[0], sub_bb[1], sub_bb[0]+sub_bb[2], sub_bb[1]+sub_bb[3]],
                                "pos9": sub_pos9,
                                "source": f"sam2_sub_{obj}",
                            })
        print(f"  SAM2 sub-segment + label: {t_sam:.1f}s, 총 {len(all_labels)} 객체 (parent+sub)", flush=True)

        # 3. Aggregator — 중복 제거 (한국어 정규화 기준)
        final_en = {}
        for label in all_labels:
            kr = label['object_kr']
            if kr and kr not in final_en:
                final_en[label['object_en']] = {"위치": label['pos9']}

        ev = eval_with_kr_mapping(final_en, gt)
        rec = ev['tp']/gt_n if gt_n else 0
        prec = ev['tp']/(ev['tp']+ev['fp']) if (ev['tp']+ev['fp']) else 0
        f1 = 2*rec*prec/(rec+prec) if (rec+prec) else 0
        pacc = ev['pc']/ev['pt'] if ev['pt'] else 0
        acc = ev['tp']/(ev['tp']+ev['fn']+ev['fp']) if (ev['tp']+ev['fn']+ev['fp']) else 0
        print(f"  📊 TP {ev['tp']}/{gt_n}, Acc {acc*100:.1f}%, F1 {f1*100:.1f}%, 위치 {pacc*100:.1f}%, 환각 {ev['fp']}", flush=True)
        results.append({"cat":cat,"img":img,"gt_n":gt_n,"all_labels":all_labels,
                        "final_en":final_en,"eval":ev})
        with open(f"/Users/kg/nonmoon/htp_thesis/sam2_subsegment_{N}img.json","w",encoding="utf-8") as f:
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
    print(f"\n=== SAM2-SubSegment-Judge 완료 {datetime.now().strftime('%H:%M:%S')} N={N} ===")
    print(f"  Acc {acc*100:.1f}%, F1 {f1*100:.1f}%, Recall {rec*100:.1f}%, Prec {prec*100:.1f}%, 9-Pos {pacc*100:.1f}%, 환각 {tot['fp']}")


if __name__ == "__main__":
    main()
