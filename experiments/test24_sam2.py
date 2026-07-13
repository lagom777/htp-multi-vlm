"""24장 SAM2-SubSegment + Clean prompt (영어, hint X)."""
import os, json, time, sys, base64, subprocess, io
from datetime import datetime
from collections import Counter
from PIL import Image, ImageDraw
sys.path.insert(0, '/Users/kg/nonmoon/htp_thesis')
import v6_voting_debate_4 as v6m
from common_utils import parse_json_v2
import clean_prompts_v2 as cp2
from en_to_kr import normalize_en_to_kr
from g_prime import (ENDPOINTS, BASE_IMG, BASE_LBL, TS_MAP, fp, load_gt, call_vlm)
from sam2_subsegment import sam2_subsegment, crop_with_marker, call_qwen_with_b64, bbox_xywh_to_9region

NAME = "SAM2-SubSegment-Clean"
OUTFILE = "/Users/kg/nonmoon/htp_thesis/test24_sam2.json"


def main():
    JUDGE = "qwen"
    TEST = cp2.TEST_IMAGES_24
    print(f"=== {NAME} 24장 ===")
    print(f"시작: {datetime.now().strftime('%H:%M:%S')}")
    results = []
    for idx, (cat, img) in enumerate(TEST):
        print(f"\n[{idx+1}/24] {cat}/{img}", flush=True)
        img_path = fp(BASE_IMG, [TS_MAP[cat], img])
        gt = load_gt(cat, img); gt_n = len(gt)
        im = Image.open(img_path); W, H = im.size

        # 1. Qwen이 큰 객체 bbox 출력 (Clean prompt — hint X)
        t0 = time.time()
        bbox_raw = call_vlm("qwen", cp2.make_bbox_simple_en(cat),
                            img_path=img_path, temp=0.2)
        bbox_parsed = parse_json_v2(bbox_raw)
        n_main = len(bbox_parsed) if isinstance(bbox_parsed, list) else 0
        print(f"  Qwen bbox: {time.time()-t0:.1f}s, {n_main} parent", flush=True)

        all_labels = []
        t_sam = 0
        if isinstance(bbox_parsed, list):
            for bi, b_item in enumerate(bbox_parsed[:10]):
                obj = b_item.get("객체","").strip()
                bbox_1000 = b_item.get("bbox",[])
                if len(bbox_1000) < 4: continue
                x1 = int(bbox_1000[0] * W / 1000); y1 = int(bbox_1000[1] * H / 1000)
                x2 = int(bbox_1000[2] * W / 1000); y2 = int(bbox_1000[3] * H / 1000)
                pos9 = bbox_xywh_to_9region([x1,y1,x2-x1,y2-y1], W, H)
                all_labels.append({"object_kr": v6m.normalize_label(obj),
                                   "object_raw": obj, "pos9": pos9, "source": "parent_bbox"})
                # SAM2 sub
                sam_out = f"/tmp/sam2_sub_24_{os.getpid()}_{bi}.json"
                t0 = time.time()
                subs = sam2_subsegment(img_path, [x1,y1,x2,y2], sam_out)
                t_sam += time.time() - t0
                subs_filtered = [s for s in subs if 0.05 <= s['area_ratio_in_parent'] <= 0.5]
                subs_filtered.sort(key=lambda x: -x['area_ratio_in_parent'])
                for si, sub in enumerate(subs_filtered[:5]):
                    sub_bb = sub['orig_bbox_xywh']
                    sub_pos9 = bbox_xywh_to_9region(sub_bb, W, H)
                    crop_b64 = crop_with_marker(img_path, sub_bb, padding=0.05)
                    sub_prompt = f"""HTP '{cp2.CATEGORY_EN[cat]}' drawing. RED-marked region inside larger '{obj}'.
What specific part is in RED box? Korean single noun.
```json
{{"객체": "(Korean noun)"}}
```"""
                    sub_raw = call_qwen_with_b64(sub_prompt, crop_b64, temp=0.2)
                    sub_parsed = parse_json_v2(sub_raw)
                    if isinstance(sub_parsed, dict):
                        sub_obj = sub_parsed.get("객체","").strip()
                        if sub_obj and sub_obj != obj and sub_obj not in ["없음","없"]:
                            n_sub = v6m.normalize_label(sub_obj)
                            if n_sub:
                                all_labels.append({"object_kr": n_sub, "object_raw": sub_obj,
                                                   "pos9": sub_pos9, "source": f"sam2_sub_of_{obj}"})
        print(f"  SAM2 sub+label: {t_sam:.1f}s, {len(all_labels)}", flush=True)

        # Judge
        desc = cp2.CATEGORY_EN.get(cat,"drawing")
        items_str = "".join([f"  - '{lbl['object_kr']}' at {lbl['pos9']} ({lbl['source']})\n" for lbl in all_labels])
        judge_prompt = f"""HTP '{desc}' analysis. SAM2 + VLM identified:
{items_str}

For each (verify with image):
- "있음" if present, "없음" if hallucination
- Same objects unified
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
        results.append({"cat":cat,"img":img,"gt_n":gt_n,"all_labels":all_labels,
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
