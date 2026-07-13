"""체크리스트 없는 12장에서 임계값 × Judge 전체 ablation

입력: v3_state_nocheck_12.json
출력: csv/judge_full_ablation.csv

실험 1: 1/4, 2/4, 3/4, 4/4 + Gemini Judge (baseline)
실험 2: 2/4 고정 + 5 Judge 모델 비교
"""
import os, json, time, sys
from openai import OpenAI
import google.generativeai as genai

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(PROJECT_DIR, "config.json"), 'r') as f:
    config = json.load(f)
genai.configure(api_key=config.get("GOOGLE_API_KEY", ""))

sys.path.insert(0, '.')
from judge_ablation import normalize, match, build_judge_prompt, parse_final, call_judge, JUDGES

STATE_FILE = "v3_state_nocheck_12.json"

def reclassify(models_data_parsed, threshold):
    all_objects = {}
    for model_name, parsed in models_data_parsed.items():
        if not parsed: continue
        for entry in parsed:
            obj = entry.get("객체", "")
            판정 = entry.get("판정", "있음")
            if "없음" in str(판정): continue
            obj_norm = normalize(obj)
            if obj_norm not in all_objects:
                all_objects[obj_norm] = {}
            all_objects[obj_norm][model_name] = {
                "위치": entry.get("위치", ""),
                "형태": entry.get("형태", ""),
                "크기": entry.get("크기", ""),
                "배치": entry.get("배치", ""),
                "선_특성": entry.get("선_특성", ""),
                "주변_맥락": entry.get("주변_맥락", ""),
                "확신도": entry.get("확신도", 0),
                "근거": entry.get("근거", ""),
            }
    agreed, disagreed = [], []
    for obj, md in all_objects.items():
        count = len(md)
        if count >= threshold:
            confs = [d["확신도"] for d in md.values() if isinstance(d["확신도"], (int, float))]
            avg_conf = sum(confs) / len(confs) if confs else 0
            agreed.append({"객체": obj, "동의수": count, "확신도": round(avg_conf, 2)})
        else:
            disagreed.append({"객체": obj, "동의수": count, "모델": md})
    return agreed, disagreed

def run_condition(states, threshold, judge_name, judge_config):
    results = []
    for state in states:
        agreed, disagreed = reclassify(state["models_data_parsed"], threshold)
        if disagreed:
            prompt = build_judge_prompt(agreed, disagreed)
            judge_result = call_judge(prompt, judge_config)
            pred = parse_final(judge_result)
        else:
            pred = [a["객체"] for a in agreed]
        acc, prec, f1, m, mi, fp = match(pred, state['gt'])
        results.append({
            "image": state["image"], "category": state["category"],
            "acc": acc, "prec": prec, "f1": f1,
            "fp": len(fp), "missed": len(mi),
            "n_agreed": len(agreed), "n_disputed": len(disagreed),
        })
    return results

def summarize(results):
    n = len(results)
    return {
        "acc": sum(x["acc"] for x in results) / n,
        "prec": sum(x["prec"] for x in results) / n,
        "f1": sum(x["f1"] for x in results) / n,
        "fp": sum(x["fp"] for x in results) / n,
        "n_disputed": sum(x["n_disputed"] for x in results) / n,
    }

def main():
    with open(STATE_FILE) as f:
        states = json.load(f)
    print(f"로드: {len(states)}장 (체크리스트 없이)")

    # Investigate model response status
    print("\n각 이미지 모델 응답 상태:")
    for s in states:
        cnt = sum(1 for m, p in s["models_data_parsed"].items() if p)
        print(f"  {s['image']}: {cnt}/4 모델 응답")

    summary = {}

    # 실험 1: 4개 임계값 × Gemini Judge
    print("\n" + "="*60)
    print("  실험 1: 임계값 1/4 ~ 4/4 (Judge = Gemini Pro)")
    print("="*60)
    for threshold in [1, 2, 3, 4]:
        print(f"\n  [임계값 {threshold}/4]", flush=True)
        results = run_condition(states, threshold, "gemini-pro", JUDGES["gemini-pro"])
        s = summarize(results)
        summary[f"t{threshold}_gemini"] = s
        print(f"  Acc={s['acc']:.1f}% Prec={s['prec']:.1f}% F1={s['f1']:.1f} FP={s['fp']:.2f}/img 불일치평균={s['n_disputed']:.1f}")

    # 실험 2: 2/4 고정 × 5 Judges
    print("\n" + "="*60)
    print("  실험 2: 임계값 2/4 × Judge 모델 5개")
    print("="*60)
    for judge_name, judge_config in JUDGES.items():
        if judge_name == "gemini-pro":
            summary[f"t2_{judge_name}"] = summary["t2_gemini"]  # 이미 했음
            continue
        print(f"\n  [Judge = {judge_name}]", flush=True)
        results = run_condition(states, 2, judge_name, judge_config)
        s = summarize(results)
        summary[f"t2_{judge_name}"] = s
        print(f"  Acc={s['acc']:.1f}% Prec={s['prec']:.1f}% F1={s['f1']:.1f} FP={s['fp']:.2f}/img")

    # 종합 출력
    print("\n" + "="*60)
    print("  종합 요약")
    print("="*60)
    print(f"\n[임계값 ablation (Gemini Judge)]")
    print(f"  {'임계값':<10} {'Acc':>7} {'Prec':>7} {'F1':>7} {'FP/img':>8}")
    for t in [1, 2, 3, 4]:
        s = summary[f"t{t}_gemini"]
        marker = " ★" if t == 3 else ""
        print(f"  {t}/4{marker:<8} {s['acc']:>6.1f}% {s['prec']:>6.1f}% {s['f1']:>6.1f}% {s['fp']:>8.2f}")

    print(f"\n[Judge 모델 비교 (임계값 2/4)]")
    print(f"  {'Judge':<20} {'Acc':>7} {'Prec':>7} {'F1':>7} {'FP/img':>8}")
    for j in JUDGES.keys():
        key = f"t2_gemini" if j == "gemini-pro" else f"t2_{j}"
        s = summary[key]
        print(f"  {j:<20} {s['acc']:>6.1f}% {s['prec']:>6.1f}% {s['f1']:>6.1f}% {s['fp']:>8.2f}")

    # CSV 저장
    import csv
    os.makedirs("csv", exist_ok=True)
    rows = []
    for key, s in summary.items():
        rows.append({
            "condition": key,
            "acc": f"{s['acc']:.1f}%",
            "prec": f"{s['prec']:.1f}%",
            "f1": f"{s['f1']:.1f}",
            "fp_per_img": f"{s['fp']:.2f}",
        })
    with open("csv/judge_full_ablation.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\n저장: csv/judge_full_ablation.csv")

if __name__ == "__main__":
    main()
