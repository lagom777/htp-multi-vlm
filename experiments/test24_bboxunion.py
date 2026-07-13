"""24장 BboxUnion — simple or sizehint."""
import os, json, time, sys
from datetime import datetime
from collections import Counter
sys.path.insert(0, '/Users/kg/nonmoon/htp_thesis')
import v6_voting_debate_4 as v6m
from common_utils import parse_json_v2
import clean_prompts_v2 as cp2
from en_to_kr import normalize_en_to_kr
from g_prime import (ENDPOINTS, BASE_IMG, BASE_LBL, TS_MAP, fp, load_gt, call_vlm,
                     iou, bbox_to_9region)

MODE = sys.argv[1] if len(sys.argv) > 1 else "simple"
PROMPT_FN = cp2.make_bbox_simple_en if MODE == "simple" else cp2.make_bbox_sizehint_en
NAME = f"BboxUnion-{MODE}"
OUTFILE = f"/Users/kg/nonmoon/htp_thesis/test24_bboxunion_{MODE}.json"


def main():
    JUDGE = "qwen"
    MODELS = ["qwen", "exaone", "gemma"]
    TEST = cp2.TEST_IMAGES_24
    print(f"=== {NAME} 24장 ===")
    print(f"시작: {datetime.now().strftime('%H:%M:%S')}")
    results = []
    for idx, (cat, img) in enumerate(TEST):
        print(f"\n[{idx+1}/24] {cat}/{img}", flush=True)
        img_path = fp(BASE_IMG, [TS_MAP[cat], img])
        gt = load_gt(cat, img); gt_n = len(gt)

        bbox_outs = {}
        prompt = PROMPT_FN(cat)
        for m in MODELS:
            t0 = time.time()
            try:
                raw = call_vlm(m, prompt, img_path=img_path, temp=0.2)
            except: raw = ""
            parsed = parse_json_v2(raw)
            n_obj = len(parsed) if isinstance(parsed, list) else 0
            print(f"  Bbox {m}: {time.time()-t0:.1f}s, {n_obj}", flush=True)
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

        desc = cp2.CATEGORY_EN.get(cat,"drawing")
        items_str = "".join([f"  - '{c['obj']}' bbox={[int(x) for x in c['bbox']]} pos={c['pos_9']} models={c['models']}\n" for c in merged])
        judge_prompt = f"""You are a Judge for HTP '{desc}'.

Candidates from 3 VLMs (bbox + IoU>0.3 merged):
{items_str}

For each (verify with image):
- "있음" if visible, "없음" if hallucination
- Same object variants unified
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
                        "merged":merged,"judge":j_parsed,"final":final_norm,"eval":ev})
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
