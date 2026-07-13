"""FullEN-FreeUnion = E' 영어 통일 + 영어 출력 + 영어→한국어 매핑.

3 VLM 영어 자유 분석 (영어 출력) → Union → Judge (영어) → 영어→한국어 변환 → 채점.
"""
import os, json, base64, time, unicodedata, sys
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


def eval_with_kr_mapping(final_en, gt):
    """final_en: {english_name: {...}} → kr 변환 후 GT 비교."""
    final_kr = {}
    for k_en, v in final_en.items():
        # 영어 → 한국어 변환
        k_kr_raw = normalize_en_to_kr(k_en)
        # 한국어 → 정규화 (GT 라벨 형식)
        k_kr_norm = v6m.normalize_label(k_kr_raw)
        if k_kr_norm and k_kr_norm not in final_kr:
            final_kr[k_kr_norm] = v
    tp=fn=fp=pc=pt=0
    for gl, gpos in gt.items():
        if gl in final_kr:
            tp+=1; pt+=1
            fpos = final_kr[gl].get("위치","")
            if fpos and fpos in gpos: pc+=1
        else: fn+=1
    for fo in final_kr:
        if fo not in gt: fp+=1
    return dict(tp=tp,fn=fn,fp=fp,pc=pc,pt=pt)


def main():
    ORCH = os.environ.get("JUDGE", "qwen")
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    TEST = TEST_IMAGES_FULL[:N]
    MODELS = ["qwen", "exaone", "gemma"]
    print(f"=== FullEN-FreeUnion-Judge ({ORCH}) — N={N} ===")
    print(f"시작: {datetime.now().strftime('%H:%M:%S')}")

    results = []
    for idx, (cat, img) in enumerate(TEST):
        print(f"\n[{idx+1}/{N}] {cat}/{img}", flush=True)
        img_path = fp(BASE_IMG, [TS_MAP[cat], img])
        gt = load_gt(cat, img); gt_n = len(gt)

        r1 = {}
        prompt = enp2.make_r1_prompt_full_en(cat)
        for m in MODELS:
            t0 = time.time()
            try:
                raw = call_vlm(m, prompt, img_path=img_path, temp=0.2)
            except Exception as e:
                raw = ""
            parsed = parse_json_v2(raw)
            n_obj = len(parsed) if isinstance(parsed, list) else 0
            print(f"  R1 {m}: {time.time()-t0:.1f}s, {n_obj} 객체", flush=True)
            r1[m] = parsed if isinstance(parsed, list) else []

        # Union — 영어 객체명
        union = {}
        for m, ans in r1.items():
            for it in ans:
                if not isinstance(it, dict): continue
                # english key field
                obj = (it.get("object") or it.get("객체","")).strip().lower()
                judg = it.get("judgment") or it.get("판정")
                if judg not in ["present", "있음", None]: continue
                if not obj: continue
                if obj not in union:
                    union[obj] = {"pos": it.get("position") or it.get("위치",""),
                                  "ev": it.get("evidence") or it.get("근거",""),
                                  "models": [m]}
                else:
                    union[obj]["models"].append(m)
        print(f"  Union: {len(union)} 후보", flush=True)

        t0 = time.time()
        j_raw = call_vlm(ORCH, enp2.make_union_judge_prompt_full_en(cat, union),
                         img_path=img_path, temp=0.2)
        j_parsed = parse_json_v2(j_raw)
        print(f"  Judge: {time.time()-t0:.1f}s, {len(j_parsed) if isinstance(j_parsed, list) else 0}", flush=True)

        final_en = {}
        if isinstance(j_parsed, list):
            for it in j_parsed:
                if not isinstance(it, dict): continue
                if (it.get("final_judgment") or it.get("최종_판정")) not in ["present", "있음"]: continue
                o = (it.get("object") or it.get("객체","")).strip().lower()
                if o and o not in final_en:
                    final_en[o] = {"위치": it.get("position") or it.get("위치","")}

        ev = eval_with_kr_mapping(final_en, gt)
        rec = ev['tp']/gt_n if gt_n else 0
        prec = ev['tp']/(ev['tp']+ev['fp']) if (ev['tp']+ev['fp']) else 0
        f1 = 2*rec*prec/(rec+prec) if (rec+prec) else 0
        pacc = ev['pc']/ev['pt'] if ev['pt'] else 0
        acc = ev['tp']/(ev['tp']+ev['fn']+ev['fp']) if (ev['tp']+ev['fn']+ev['fp']) else 0
        print(f"  📊 TP {ev['tp']}/{gt_n}, Acc {acc*100:.1f}%, F1 {f1*100:.1f}%, 위치 {pacc*100:.1f}%, 환각 {ev['fp']}", flush=True)
        results.append({"cat":cat,"img":img,"gt_n":gt_n,"r1":r1,"union":union,
                        "judge":j_parsed,"final_en":final_en,"eval":ev})
        with open(f"/Users/kg/nonmoon/htp_thesis/fullen_freeunion_{ORCH}_{N}img.json","w",encoding="utf-8") as f:
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
    print(f"\n=== FullEN-FreeUnion({ORCH}) 완료 {datetime.now().strftime('%H:%M:%S')} N={N} ===")
    print(f"  Acc {acc*100:.1f}%, F1 {f1*100:.1f}%, Recall {rec*100:.1f}%, Prec {prec*100:.1f}%, 9-Pos {pacc*100:.1f}%, 환각 {tot['fp']}")


if __name__ == "__main__":
    main()
