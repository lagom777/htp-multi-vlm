"""단일 모델 R1 비교 — 한국어 vs 영어 prompt (Qwen, EXAONE, Gemma) × N장."""
import os, json, base64, time, unicodedata, sys
from datetime import datetime
from collections import Counter, defaultdict
sys.path.insert(0, '/Users/kg/nonmoon/htp_thesis')
import v6_voting_debate_4 as v6m
import english_prompts as enp
from common_utils import parse_json_v2
from g_prime import (ENDPOINTS, TEST_IMAGES_FULL, BASE_IMG, BASE_LBL, TS_MAP,
                     fp, load_gt, call_vlm)


def evalv(parsed, gt):
    if not isinstance(parsed,list): return dict(tp=0,fn=len(gt),fp=0,pc=0,pt=0)
    final={}
    for it in parsed:
        if not isinstance(it,dict): continue
        if it.get("판정") not in ["있음",None]: continue
        n=v6m.normalize_label(it.get("객체",""))
        if n and n not in final: final[n]={"위치":it.get("위치","")}
    tp=fn=fp=pc=pt=0
    for gl,gpos in gt.items():
        if gl in final:
            tp+=1; pt+=1
            f=final[gl].get("위치","")
            if f and f in gpos: pc+=1
        else: fn+=1
    for fo in final:
        if fo not in gt: fp+=1
    return dict(tp=tp,fn=fn,fp=fp,pc=pc,pt=pt)


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    TEST = TEST_IMAGES_FULL[:N]
    MODELS = ["qwen", "exaone", "gemma"]

    print(f"=== 단일 R1 비교 — 한국어 vs 영어 × 3 모델 × {len(TEST)}장 ===")
    print(f"시작: {datetime.now().strftime('%H:%M:%S')}")

    all_results = {}
    for model in MODELS:
        for label, prompt_fn in [("kr", v6m.make_r1_prompt), ("en", enp.make_r1_prompt_en)]:
            print(f"\n[{model}/{label}]")
            tot = Counter()
            per_img = []
            for idx, (cat, img) in enumerate(TEST):
                img_path = fp(BASE_IMG, [TS_MAP[cat], img])
                gt = load_gt(cat, img); gt_n = len(gt)
                prompt = prompt_fn(cat)
                t0 = time.time()
                try:
                    raw = call_vlm(model, prompt, img_path=img_path, temp=0.2)
                except Exception as e:
                    raw = ""
                el = time.time()-t0
                parsed = parse_json_v2(raw)
                n_obj = len(parsed) if isinstance(parsed, list) else 0
                ev = evalv(parsed, gt)
                for k in ['tp','fn','fp','pc','pt']: tot[k] += ev[k]
                tot['gtn'] += gt_n
                per_img.append({"cat":cat,"img":img,"gt_n":gt_n,"n_obj":n_obj,"eval":ev})
                if (idx+1) % 4 == 0:
                    print(f"    {idx+1}/{len(TEST)} 진행", flush=True)
            rec = tot['tp']/tot['gtn'] if tot['gtn'] else 0
            prec = tot['tp']/(tot['tp']+tot['fp']) if (tot['tp']+tot['fp']) else 0
            f1 = 2*rec*prec/(rec+prec) if (rec+prec) else 0
            pacc = tot['pc']/tot['pt'] if tot['pt'] else 0
            acc = tot['tp']/(tot['tp']+tot['fn']+tot['fp'])
            print(f"  Acc {acc*100:.1f}% F1 {f1*100:.1f}% Recall {rec*100:.1f}% Prec {prec*100:.1f}% 9-Pos {pacc*100:.1f}% 환각 {tot['fp']}")
            all_results[f"{model}_{label}"] = {"per_img":per_img,
                "acc":acc,"f1":f1,"recall":rec,"precision":prec,"pacc":pacc,"fp":tot['fp']}

    with open(f"/Users/kg/nonmoon/htp_thesis/single_compare_{len(TEST)}img.json","w",encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # 최종 표
    print(f"\n=== 최종 비교 표 ({len(TEST)}장) ===")
    print(f"{'모델':10s} {'lang':5s} {'Acc':>6s} {'F1':>6s} {'Recall':>7s} {'Prec':>6s} {'9-Pos':>6s} {'환각':>5s}")
    for k, v in all_results.items():
        m, lang = k.rsplit("_", 1)
        print(f"{m:10s} {lang:5s} {v['acc']*100:>5.1f}% {v['f1']*100:>5.1f}% {v['recall']*100:>6.1f}% {v['precision']*100:>5.1f}% {v['pacc']*100:>5.1f}% {v['fp']:>5d}")


if __name__ == "__main__":
    main()
