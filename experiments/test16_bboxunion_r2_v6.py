"""16장 BboxUnion + R2 + Judge — v6 prompt (English output + anatomy hint).

R2: 단독 답(1 모델만) 객체에 대해 다른 모델들에게 재확인.
24장 BboxUnion+Judge (R2 없는 버전) 와 직접 비교용.

16장 = 4 카테고리 × 4장 (TEST_IMAGES_24의 첫 4장씩).
"""
import os, json, time, sys
from datetime import datetime
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'framework'))
import v6_voting_debate_4 as v6m
from common_utils import parse_json_v2
import clean_prompts_v2 as cp2
import clean_prompts_v6 as cp6
from en_to_kr import normalize_en_to_kr
from g_prime import (ENDPOINTS, BASE_IMG, BASE_LBL, TS_MAP, fp, load_gt, call_vlm,
                     iou, bbox_to_9region)

NAME = "BboxUnion-R2-v6"
OUTFILE = "./test16_bboxunion_r2_v6.json"

# 4 카테고리 × 4장 = 16장 (24장 셋의 앞 4장씩)
TEST_16 = [
    # 나무 4
    ("TL_나무", "나무_8_남_01445.jpg"),
    ("TL_나무", "나무_10_여_00019.jpg"),
    ("TL_나무", "나무_11_남_00004.jpg"),
    ("TL_나무", "나무_12_여_00007.jpg"),
    # 집 4
    ("TL_집", "집_12_여_08971.jpg"),
    ("TL_집", "집_10_여_00006.jpg"),
    ("TL_집", "집_11_남_00013.jpg"),
    ("TL_집", "집_12_여_00007.jpg"),
    # 남자사람 4
    ("TL_남자사람", "남자사람_13_남_02804.jpg"),
    ("TL_남자사람", "남자사람_10_여_00023.jpg"),
    ("TL_남자사람", "남자사람_11_남_00000.jpg"),
    ("TL_남자사람", "남자사람_12_여_00005.jpg"),
    # 여자사람 4
    ("TL_여자사람", "여자사람_10_남_02125.jpg"),
    ("TL_여자사람", "여자사람_10_여_00018.jpg"),
    ("TL_여자사람", "여자사람_11_남_00002.jpg"),
    ("TL_여자사람", "여자사람_12_여_00008.jpg"),
]


def make_r2_prompt_v6(cat, single_objects):
    """R2: 객체 재확인 — **중립 prompt** (negative bias 제거).

    옛 Phase 3 prompt는 'others did NOT mention' + 'verify carefully' 편향 → reject 경향.
    v6 R2는 '독립적으로 examine, 다른 모델의 미언급은 증거 아님' 명시.
    """
    desc = cp6.CATEGORY_EN.get(cat, "drawing")
    objs_str = "\n".join([
        f"  - '{o['obj']}' (~{o['obj_kr']}) at bbox {[int(x) for x in o['bbox']]}"
        for o in single_objects
    ])
    return f"""This is a HTP '{desc}' drawing.

For each candidate object, examine the image at the given bbox location independently:
{objs_str}

Decision rule: report "yes" if the object is visible at that location; "no" only if clearly absent.
**Important**: do NOT assume that other models' lack of mention is evidence of absence —
judge purely from what you see in the image.

For animals, name the specific species (e.g., squirrel, bird, butterfly).
Object names in English single noun (lowercase).

JSON only:
```json
[
  {{"object": "(English noun)", "confirmation": "yes" or "no", "evidence": "..."}}
]
```"""


def make_judge_prompt_v6(cat, merged, r2_results):
    desc = cp6.CATEGORY_EN.get(cat, "drawing")
    items_str = ""
    for c in merged:
        models = c['models']
        r2_status = ""
        if c.get('is_single'):
            confirms = r2_results.get(c['obj_kr'], [])
            r2_status = f", R2 confirms={confirms}"
        items_str += (
            f"  - '{c['obj']}' (~{c['obj_kr']}) bbox={[int(x) for x in c['bbox']]} "
            f"pos={c['pos_9']} models={models}{r2_status}\n"
        )
    return f"""You are a Judge for HTP '{desc}' drawing.

Object candidates from 3 VLMs (Union + R2 verification):
{items_str}

For each candidate (verify with image):
- "present" if visible, "absent" if hallucination
- Use R2 info as supplementary signal
- Same object variants unified
- Object names in **English** single noun (lowercase)
- For animals, use specific species

JSON only:
```json
[{{"object":"(English noun)","final_judgment":"present" or "absent","position":"(9-region)","evidence":"..."}}]
```

Position: top-left|top-center|top-right|middle-left|middle-center|middle-right|bottom-left|bottom-center|bottom-right"""


def main():
    JUDGE = "qwen"
    MODELS = ["qwen", "exaone", "gemma"]
    TEST = TEST_16
    print(f"=== {NAME} 16장 (R2 포함) ===")
    print(f"시작: {datetime.now().strftime('%H:%M:%S')}")
    # Resume
    results = []
    done_imgs = set()
    if os.path.exists(OUTFILE):
        try:
            with open(OUTFILE) as f: results = json.load(f)
            done_imgs = {(r['cat'], r['img']) for r in results}
            print(f"Resume: {len(results)} 이미지 이미 완료")
        except Exception: results = []
    for idx, (cat, img) in enumerate(TEST):
        if (cat, img) in done_imgs:
            print(f"\n[{idx+1}/16] skip (이미 완료): {cat}/{img}", flush=True)
            continue
        print(f"\n[{idx+1}/16] {cat}/{img}", flush=True)
        img_path = fp(BASE_IMG, [TS_MAP[cat], img])
        gt = load_gt(cat, img); gt_n = len(gt)

        # === Bbox grounding ===
        bbox_outs = {}
        prompt = cp6.make_bbox_v6_en(cat)
        for m in MODELS:
            t0 = time.time()
            try:
                raw = call_vlm(m, prompt, img_path=img_path, temp=0.2)
            except Exception:
                raw = ""
            parsed = parse_json_v2(raw)
            n_obj = len(parsed) if isinstance(parsed, list) else 0
            print(f"  Bbox {m}: {time.time()-t0:.1f}s, {n_obj}", flush=True)
            bbox_outs[m] = parsed if isinstance(parsed, list) else []

        # === Union + IoU 0.3 merge ===
        all_items = []
        for m, ans in bbox_outs.items():
            if not isinstance(ans, list): continue
            for it in ans:
                if not isinstance(it, dict): continue
                obj = (it.get("object") or it.get("객체","")).strip()
                bb = it.get("bbox", [])
                if obj and isinstance(bb, list) and len(bb) >= 4:
                    obj_kr = normalize_en_to_kr(obj)
                    all_items.append({"obj":obj, "obj_kr":obj_kr, "bbox":bb, "model":m})

        merged = []
        used = [False]*len(all_items)
        for i, it in enumerate(all_items):
            if used[i]: continue
            cluster = [it]; used[i] = True
            norm_i = v6m.normalize_label(it['obj_kr'])
            for j in range(i+1, len(all_items)):
                if used[j]: continue
                norm_j = v6m.normalize_label(all_items[j]['obj_kr'])
                if norm_i == norm_j and iou(it['bbox'], all_items[j]['bbox']) >= 0.3:
                    cluster.append(all_items[j]); used[j] = True
            avg_bb = [sum(c['bbox'][k] for c in cluster)/len(cluster) for k in range(4)]
            cluster_models = list(set(c['model'] for c in cluster))
            merged.append({
                "obj":cluster[0]['obj'], "obj_kr":cluster[0]['obj_kr'],
                "bbox":avg_bb,
                "models":cluster_models,
                "is_single": len(cluster_models) == 1,
                "pos_9":bbox_to_9region(avg_bb),
            })
        print(f"  Union+IoU: {len(merged)}", flush=True)

        # === R2 — 단독 답 객체에 대해 다른 모델 재확인 ===
        single_objs = [c for c in merged if c['is_single']]
        print(f"  단독 답: {len(single_objs)}", flush=True)
        r2_results = {}  # {obj_kr: [confirms_from_models]}
        if single_objs:
            for m in MODELS:
                # 이 m이 답 안 한 단독 객체만 verify
                m_single_objs = [s for s in single_objs if m not in s['models']]
                if not m_single_objs: continue
                t0 = time.time()
                try:
                    r2_raw = call_vlm(m, make_r2_prompt_v6(cat, m_single_objs),
                                      img_path=img_path, temp=0.2)
                except Exception:
                    r2_raw = ""
                r2_parsed = parse_json_v2(r2_raw)
                if isinstance(r2_parsed, list):
                    for it in r2_parsed:
                        if not isinstance(it, dict): continue
                        o = (it.get("object") or "").strip().lower()
                        conf = (it.get("confirmation") or "").strip().lower()
                        if conf in ("yes", "있음", "present"):
                            o_kr = normalize_en_to_kr(o)
                            o_norm = v6m.normalize_label(o_kr)
                            r2_results.setdefault(o_norm, []).append(m)
                print(f"  R2 {m}: {time.time()-t0:.1f}s, {len(m_single_objs)} verified", flush=True)

        # === Judge — Union + R2 결과 종합 ===
        t0 = time.time()
        j_raw = call_vlm(JUDGE, make_judge_prompt_v6(cat, merged, r2_results),
                         img_path=img_path, temp=0.2)
        j_parsed = parse_json_v2(j_raw)
        print(f"  Judge: {time.time()-t0:.1f}s, {len(j_parsed) if isinstance(j_parsed, list) else 0}", flush=True)

        final_norm = {}
        if isinstance(j_parsed, list):
            for it in j_parsed:
                if not isinstance(it, dict): continue
                judg = it.get("final_judgment") or it.get("최종_판정")
                if judg not in ["present", "있음"]: continue
                o = (it.get("object") or it.get("객체","")).strip()
                if not o: continue
                o_kr = normalize_en_to_kr(o) if any('a' <= c.lower() <= 'z' for c in o) else o
                n = v6m.normalize_label(o_kr)
                if n and n not in final_norm:
                    final_norm[n] = {"위치": it.get("position") or it.get("위치","")}

        c_tp=c_fn=c_fp=c_pc=c_pt=0
        for gl, gpos in gt.items():
            if gl in final_norm:
                c_tp+=1; c_pt+=1
                fpos = final_norm[gl].get("위치","")
                if fpos and fpos in gpos: c_pc+=1
            else: c_fn+=1
        for fo in final_norm:
            if fo not in gt: c_fp+=1
        ev = dict(tp=c_tp, fn=c_fn, fp=c_fp, pc=c_pc, pt=c_pt)

        rec = ev['tp']/gt_n if gt_n else 0
        prec = ev['tp']/(ev['tp']+ev['fp']) if (ev['tp']+ev['fp']) else 0
        f1 = 2*rec*prec/(rec+prec) if (rec+prec) else 0
        pacc = ev['pc']/ev['pt'] if ev['pt'] else 0
        acc = ev['tp']/(ev['tp']+ev['fn']+ev['fp']) if (ev['tp']+ev['fn']+ev['fp']) else 0
        print(f"  📊 TP {ev['tp']}/{gt_n}, Acc {acc*100:.1f}%, F1 {f1*100:.1f}%, 위치 {pacc*100:.1f}%, 환각 {ev['fp']}", flush=True)
        results.append({"cat":cat,"img":img,"gt_n":gt_n,"bbox_outs":bbox_outs,
                       "merged":merged,"r2_results":r2_results,
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
