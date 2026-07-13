"""12장 BboxUnion + Zoom-R2 + Judge.

R1 → Union → 단독 답 객체의 bbox만 crop → 다른 모델에게 확인 → Judge
"""
import os, json, time, sys, base64, io
from datetime import datetime
from collections import Counter
from PIL import Image
sys.path.insert(0, '/Users/kg/nonmoon/htp_thesis')
import v6_voting_debate_4 as v6m
from common_utils import parse_json_v2
import clean_prompts_v2 as cp2
from en_to_kr import normalize_en_to_kr
from g_prime import (ENDPOINTS, BASE_IMG, BASE_LBL, TS_MAP, fp, load_gt, call_vlm,
                     iou, bbox_to_9region)
from test12_voting_judge import TEST_12
from openai import OpenAI

NAME = "BboxUnion+Zoom-R2+Judge"
OUTFILE = "/Users/kg/nonmoon/htp_thesis/test12_zoom_r2.json"


def crop_bbox(img_path, bbox_1000, padding_px=30):
    """bbox 1000 좌표 → 원본 crop → b64."""
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    x1 = int(bbox_1000[0] * W / 1000); y1 = int(bbox_1000[1] * H / 1000)
    x2 = int(bbox_1000[2] * W / 1000); y2 = int(bbox_1000[3] * H / 1000)
    cx1 = max(0, x1 - padding_px); cy1 = max(0, y1 - padding_px)
    cx2 = min(W, x2 + padding_px); cy2 = min(H, y2 + padding_px)
    crop = im.crop((cx1, cy1, cx2, cy2))
    # 작으면 upscale
    if min(crop.size) < 150:
        scale = 200 / min(crop.size)
        crop = crop.resize((int(crop.size[0]*scale), int(crop.size[1]*scale)))
    buf = io.BytesIO(); crop.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def call_vlm_with_b64(model, prompt, b64, temp=0.2):
    ep = ENDPOINTS[model]
    cli = OpenAI(base_url=f"http://192.168.200.138:{ep['port']}/v1", api_key="local", timeout=300)
    kw = {"model": ep["model"], "messages":[{"role":"user","content":[
            {"type":"text","text":prompt},
            {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}]}],
        "temperature":temp, "max_tokens":4000}
    if ep["thinking_off"]:
        kw["extra_body"] = {"chat_template_kwargs":{"enable_thinking":False}}
    r = cli.chat.completions.create(**kw)
    return r.choices[0].message.content or ""


def make_zoom_r2_prompt(cat, claimed_obj):
    desc = cp2.CATEGORY_EN.get(cat, "drawing")
    return f"""This is a ZOOMED region from an HTP '{desc}' drawing.
Another model claimed there is a '{claimed_obj}' in this region.

Verify carefully — is this region actually showing '{claimed_obj}'?
Or is it something else / nothing?

JSON only:
```json
{{
  "확인": "있음" or "없음",
  "실제_객체": "(if 없음, what is it actually? Korean noun)",
  "근거": "..."
}}
```"""


def main():
    JUDGE = "qwen"
    MODELS = ["qwen", "exaone", "gemma"]
    print(f"=== {NAME} 12장 ===")
    print(f"시작: {datetime.now().strftime('%H:%M:%S')}")
    results = []
    for idx, (cat, img) in enumerate(TEST_12):
        print(f"\n[{idx+1}/12] {cat}/{img}", flush=True)
        img_path = fp(BASE_IMG, [TS_MAP[cat], img])
        gt = load_gt(cat, img); gt_n = len(gt)

        # R1: bbox
        bbox_outs = {}
        prompt = cp2.make_bbox_sizehint_en(cat)
        for m in MODELS:
            t0 = time.time()
            try:
                raw = call_vlm(m, prompt, img_path=img_path, temp=0.2)
            except: raw = ""
            parsed = parse_json_v2(raw)
            n_obj = len(parsed) if isinstance(parsed, list) else 0
            print(f"  R1 {m}: {time.time()-t0:.1f}s, {n_obj}", flush=True)
            bbox_outs[m] = parsed if isinstance(parsed, list) else []

        # IoU 0.3 merge
        all_items = []
        for m, ans in bbox_outs.items():
            if not isinstance(ans, list): continue
            for it in ans:
                if not isinstance(it, dict): continue
                obj = it.get("객체","").strip()
                bb = it.get("bbox", [])
                if obj and isinstance(bb, list) and len(bb) >= 4:
                    all_items.append({"obj":obj, "bbox":bb, "model":m})
        merged = []
        used = [False]*len(all_items)
        for i, it in enumerate(all_items):
            if used[i]: continue
            cluster = [it]; used[i] = True
            norm_i = v6m.normalize_label(it['obj'])
            for j in range(i+1, len(all_items)):
                if used[j]: continue
                norm_j = v6m.normalize_label(all_items[j]['obj'])
                if norm_i == norm_j and iou(it['bbox'], all_items[j]['bbox']) >= 0.3:
                    cluster.append(all_items[j]); used[j] = True
            avg_bb = [sum(c['bbox'][k] for c in cluster)/len(cluster) for k in range(4)]
            merged.append({"obj":cluster[0]['obj'],"bbox":avg_bb,
                           "models":list(set(c['model'] for c in cluster)),
                           "pos_9":bbox_to_9region(avg_bb)})
        print(f"  Union+IoU: {len(merged)}", flush=True)

        # Zoom-R2: 단독 답 객체의 bbox crop → 다른 모델 확인
        single = [c for c in merged if len(c['models']) == 1]
        print(f"  단독 답: {len(single)}", flush=True)
        zoom_results = []
        for c in single:
            obj = c['obj']
            bbox = c['bbox']
            original_model = c['models'][0]
            # bbox crop
            try:
                crop_b64 = crop_bbox(img_path, bbox)
            except Exception as e:
                continue
            # 다른 두 모델에게 zoom 확인
            verifications = {}
            for m in MODELS:
                if m == original_model: continue
                t0 = time.time()
                try:
                    raw = call_vlm_with_b64(m, make_zoom_r2_prompt(cat, obj), crop_b64, temp=0.2)
                except: raw = ""
                parsed = parse_json_v2(raw)
                verifications[m] = parsed if isinstance(parsed, dict) else {"확인":"파싱실패"}
            zoom_results.append({"obj":obj, "bbox":bbox, "original_model":original_model,
                                 "verifications":verifications})
            confirms = sum(1 for v in verifications.values() if v.get("확인")=="있음")
            print(f"    Zoom '{obj}' (orig: {original_model}, 다른모델 확인 {confirms}/2)", flush=True)

        # Judge: union + zoom 결과 종합
        judge_input_str = ""
        for c in merged:
            if len(c['models']) >= 2:
                judge_input_str += f"  - '{c['obj']}' (CONSENSUS {c['models']}, bbox={[int(x) for x in c['bbox']]}, pos: {c['pos_9']})\n"
            else:
                z = next((z for z in zoom_results if z['obj'] == c['obj']), None)
                if z:
                    confirms = sum(1 for v in z['verifications'].values() if v.get("확인")=="있음")
                    other_objs = [v.get("실제_객체","") for v in z['verifications'].values() if v.get("실제_객체")]
                    judge_input_str += f"  - '{c['obj']}' (only {c['models']}, Zoom-R2: {confirms}/2 confirmed, other suggestions: {other_objs}, bbox={[int(x) for x in c['bbox']]}, pos: {c['pos_9']})\n"
                else:
                    judge_input_str += f"  - '{c['obj']}' (only {c['models']}, bbox={[int(x) for x in c['bbox']]}, pos: {c['pos_9']})\n"

        desc = cp2.CATEGORY_EN.get(cat,"drawing")
        judge_prompt = f"""HTP '{desc}' Judge. Candidates:
{judge_input_str}

For each (verify with image):
- "있음" or "없음"
- Use Zoom-R2 verification as supplementary
- Object names in **Korean**

```json
[{{"객체":"(Korean noun)","최종_판정":"있음" or "없음","위치":"(9-region)","근거":"..."}}]
```

Position: top-left|top-center|top-right|middle-left|middle-center|middle-right|bottom-left|bottom-center|bottom-right"""

        t0 = time.time()
        j_raw = call_vlm(JUDGE, judge_prompt, img_path=img_path, temp=0.2)
        j_parsed = parse_json_v2(j_raw)
        print(f"  Judge: {time.time()-t0:.1f}s, {len(j_parsed) if isinstance(j_parsed, list) else 0}", flush=True)

        final_norm = {}
        if isinstance(j_parsed, list):
            for it in j_parsed:
                if not isinstance(it, dict): continue
                if it.get("최종_판정") not in ["있음","present"]: continue
                o = it.get("객체","").strip()
                if not o: continue
                o_kr = normalize_en_to_kr(o) if any('a' <= c.lower() <= 'z' for c in o) else o
                n = v6m.normalize_label(o_kr)
                if n and n not in final_norm:
                    final_norm[n] = {"위치": it.get("위치","")}

        c_tp=c_fn=c_fp=c_pc=c_pt=0
        for gl, gpos in gt.items():
            if gl in final_norm:
                c_tp+=1; c_pt+=1
                fpos = final_norm[gl].get("위치","")
                if fpos and fpos in gpos: c_pc+=1
            else: c_fn+=1
        for fo in final_norm:
            if fo not in gt: c_fp+=1
        ev = dict(tp=c_tp,fn=c_fn,fp=c_fp,pc=c_pc,pt=c_pt)

        rec = ev['tp']/gt_n if gt_n else 0
        prec = ev['tp']/(ev['tp']+ev['fp']) if (ev['tp']+ev['fp']) else 0
        f1 = 2*rec*prec/(rec+prec) if (rec+prec) else 0
        pacc = ev['pc']/ev['pt'] if ev['pt'] else 0
        acc = ev['tp']/(ev['tp']+ev['fn']+ev['fp']) if (ev['tp']+ev['fn']+ev['fp']) else 0
        print(f"  📊 TP {ev['tp']}/{gt_n}, Acc {acc*100:.1f}%, F1 {f1*100:.1f}%, 위치 {pacc*100:.1f}%, 환각 {ev['fp']}", flush=True)
        results.append({"cat":cat,"img":img,"gt_n":gt_n,"bbox_outs":bbox_outs,
                        "merged":merged,"zoom_results":zoom_results,
                        "judge":j_parsed,"final":final_norm,"eval":ev})
        with open(OUTFILE,"w",encoding="utf-8") as f:
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
    print(f"\n=== {NAME} 완료 {datetime.now().strftime('%H:%M:%S')} ===")
    print(f"  Acc {acc*100:.1f}%, F1 {f1*100:.1f}%, Recall {rec*100:.1f}%, Prec {prec*100:.1f}%, 9-Pos {pacc*100:.1f}%, 환각 {tot['fp']}")


if __name__ == "__main__":
    main()
