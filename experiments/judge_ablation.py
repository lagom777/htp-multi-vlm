"""Judge Ablation — 저장된 v3_state_all.json 로드 후 Step 4만 5개 모델로 반복

실행: python3 judge_ablation.py
결과: csv/judge_ablation_results.csv
"""
import os, json, time, sys
from openai import OpenAI
import google.generativeai as genai

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'framework'))
from project_paths import load_api_config

config = load_api_config()

genai.configure(api_key=config.get("GOOGLE_API_KEY", ""))
gpt_client = OpenAI(api_key=config.get("OPENAI_API_KEY", ""))
grok_client = OpenAI(api_key=config.get("GROK_API_KEY", ""), base_url="https://api.x.ai/v1")
openrouter_client = OpenAI(api_key=config.get("OPENROUTER_API_KEY", ""), base_url="https://openrouter.ai/api/v1")

# 테스트할 Judge 모델 5개
JUDGES = {
    "gemini-pro": {"provider": "gemini", "model": "gemini-3.1-pro-preview"},
    "claude-sonnet": {"provider": "openrouter", "model": "anthropic/claude-sonnet-4-6"},
    "claude-opus-4.7": {"provider": "openrouter", "model": "anthropic/claude-opus-4.7"},
    "gpt-5.4": {"provider": "openai", "model": "gpt-5.4"},
    "grok-reasoning": {"provider": "grok", "model": "grok-4.20-beta-0309-reasoning"},
}

SYNONYMS = {
    "해": ["태양", "sun"], "풀": ["잔디", "grass", "풀밭"],
    "구름": ["cloud"], "별": ["star"], "달": ["moon", "초승달"],
    "산": ["mountain"], "꽃": ["flower"], "새": ["bird"],
    "나무": ["나무전체", "tree"], "수관": ["나무수관", "treecrown", "canopy"],
    "기둥": ["나무줄기", "trunk"], "가지": ["branch"],
    "뿌리": ["나무뿌리", "root"], "나뭇잎": ["잎", "leaf", "leaves"],
    "열매": ["fruit", "사과", "도토리"], "다람쥐": ["squirrel", "동물"],
    "그네": ["swing"], "집": ["집전체", "house"], "사람": ["사람전체", "person"],
    "신발": ["운동화", "구두", "shoe", "하이힐"],
    "손": ["hand"], "발": ["foot"],
    "창문": ["window"], "문": ["door"], "지붕": ["roof"],
    "굴뚝": ["chimney"], "연기": ["smoke"], "길": ["path", "road"],
    "연못": ["pond"], "울타리": ["fence"], "머리": ["head"],
    "머리카락": ["hair"], "눈": ["eye"], "코": ["nose"],
    "입": ["mouth"], "귀": ["ear"], "팔": ["arm"],
    "다리": ["leg"], "상체": ["body", "몸통"],
    "목": ["neck"], "얼굴": ["face"], "집벽": ["벽", "wall"],
    "단추": ["button"], "주머니": ["pocket"],
    "태양": ["sun"], "잔디": ["grass"], "집전체": ["house"],
    "사람전체": ["person"],
}

def normalize(name):
    name = name.strip().replace(" ", "").lower()
    for key, vals in SYNONYMS.items():
        if name == key.lower(): return key
        for v in vals:
            if name == v.lower().replace(" ", ""): return key
    for key, vals in SYNONYMS.items():
        if key.lower() in name or name in key.lower(): return key
        for v in vals:
            v_clean = v.lower().replace(" ", "")
            if v_clean in name or name in v_clean: return key
    return name

def infer_meta_labels(pred_norm):
    extra = set()
    if len(pred_norm & {"수관", "기둥", "가지", "뿌리", "나뭇잎"}) >= 2: extra.add("나무")
    if len(pred_norm & {"머리", "얼굴", "눈", "코", "입", "상체", "팔", "다리"}) >= 2: extra.add("사람")
    if len(pred_norm & {"지붕", "집벽", "문", "창문"}) >= 2: extra.add("집")
    return extra

def match(pred_list, gt_labels):
    pred_norm = set(normalize(p) for p in pred_list if p.strip())
    pred_norm |= infer_meta_labels(pred_norm)
    gt_norm = set(normalize(g) for g in gt_labels)
    matched = gt_norm & pred_norm
    missed = gt_norm - pred_norm
    fp = pred_norm - gt_norm
    acc = len(matched) / len(gt_norm) * 100 if gt_norm else 0
    prec = len(matched) / len(pred_norm) * 100 if pred_norm else 0
    rec = acc
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return acc, prec, f1, matched, missed, fp

def call_judge(prompt, judge_config):
    provider = judge_config["provider"]
    model_id = judge_config["model"]
    for attempt in range(3):
        try:
            if provider == "gemini":
                m = genai.GenerativeModel(model_id)
                resp = m.generate_content(prompt)
                if not resp.candidates: raise RuntimeError("빈 응답")
                return resp.candidates[0].content.parts[0].text.strip()
            elif provider == "openai":
                resp = gpt_client.chat.completions.create(
                    model=model_id, max_completion_tokens=4000,
                    messages=[{"role": "user", "content": prompt}])
                return resp.choices[0].message.content.strip()
            elif provider == "openrouter":
                resp = openrouter_client.chat.completions.create(
                    model=model_id, max_tokens=4000,
                    messages=[{"role": "user", "content": prompt}])
                return resp.choices[0].message.content.strip()
            elif provider == "grok":
                resp = grok_client.chat.completions.create(
                    model=model_id, max_tokens=4000,
                    messages=[{"role": "user", "content": prompt}])
                return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"      Judge 에러 ({attempt+1}): {e}", flush=True)
            time.sleep(3)
    return "Error"

def build_judge_prompt(agreed, disagreed):
    comparison_text = ""
    for d in disagreed:
        comparison_text += f"\n[{d['객체']}] — {d['동의수']}개 모델만 주장\n"
        for mn, attrs in d["모델"].items():
            comparison_text += (
                f"  {mn}: 위치:{attrs['위치']} 형태:{attrs['형태']} 크기:{attrs['크기']} "
                f"배치:{attrs['배치']} 맥락:{attrs['주변_맥락']} "
                f"확신도:{attrs['확신도']} 근거:{attrs['근거']}\n"
            )
    agreed_list = ", ".join(a["객체"] for a in agreed)
    return (
        f"당신은 HTP 심리검사 그림 분석의 Judge입니다.\n"
        f"이미지는 볼 수 없으며, 오직 각 모델이 기록한 세세한 속성만 보고 판정합니다.\n\n"
        f"[합의된 객체 ({len(agreed)}개)]: {agreed_list}\n\n"
        f"[불일치 항목 — 속성 비교]:\n{comparison_text}\n\n"
        f"각 불일치 항목의 형태/크기/배치/맥락이 구체적이고 일관성 있으면 승인, 아니면 거절.\n\n"
        f"마지막 줄에: 최종: (합의 객체 + 승인한 객체) 쉼표로 구분"
    )

def parse_final(judge_result):
    final = judge_result
    for line in reversed(judge_result.split("\n")):
        if "최종:" in line or "최종 :" in line:
            final = line.split(":", 1)[1].strip()
            break
    return [p.strip() for p in final.split(",") if p.strip()]

def main():
    STATE_FILE = "v3_state_all.json"
    if not os.path.exists(STATE_FILE):
        print(f"오류: {STATE_FILE} 없음. save_v3_state.py 먼저 실행 필요.")
        return

    with open(STATE_FILE, 'r', encoding='utf-8') as f:
        all_states = json.load(f)
    print(f"상태 로드: {len(all_states)}장")

    # Judge별 결과 수집
    judge_results = {}

    for judge_name, judge_config in JUDGES.items():
        print(f"\n{'='*60}")
        print(f"  Judge: {judge_name} ({judge_config['model']})")
        print(f"{'='*60}")

        per_image = []
        for idx, state in enumerate(all_states, 1):
            img_name = state["image"]
            gt = state["gt"]
            agreed = state["agreed"]
            disagreed = state["disagreed"]

            if disagreed:
                judge_prompt = build_judge_prompt(agreed, disagreed)
                print(f"  [{idx}/16] {img_name} — 불일치 {len(disagreed)}개, Judge 호출...", end=" ", flush=True)
                t0 = time.time()
                judge_result = call_judge(judge_prompt, judge_config)
                elapsed = time.time() - t0
                pred = parse_final(judge_result)
                print(f"{elapsed:.0f}초", flush=True)
            else:
                pred = [a["객체"] for a in agreed]
                print(f"  [{idx}/16] {img_name} — 불일치 없음", flush=True)

            acc, prec, f1, matched, missed, fp = match(pred, gt)
            per_image.append({
                "image": img_name, "category": state["category"],
                "acc": acc, "prec": prec, "f1": f1,
                "fp_count": len(fp), "missed_count": len(missed),
                "n_disputed": len(disagreed),
            })
            time.sleep(0.5)

        judge_results[judge_name] = per_image

        avg_acc = sum(r["acc"] for r in per_image) / len(per_image)
        avg_prec = sum(r["prec"] for r in per_image) / len(per_image)
        avg_f1 = sum(r["f1"] for r in per_image) / len(per_image)
        avg_fp = sum(r["fp_count"] for r in per_image) / len(per_image)
        print(f"\n  {judge_name}: Acc={avg_acc:.1f}% Prec={avg_prec:.1f}% F1={avg_f1:.1f} FP={avg_fp:.2f}/img", flush=True)

    # 종합 테이블
    print(f"\n{'='*60}")
    print("  Judge Ablation 전체 비교")
    print(f"{'='*60}\n")
    print(f"  {'Judge':<20} {'Acc':>7} {'Prec':>7} {'F1':>7} {'FP/img':>8}")
    print(f"  {'-'*20} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")

    summary = []
    for judge_name, per_image in judge_results.items():
        n = len(per_image)
        avg_acc = sum(r["acc"] for r in per_image) / n
        avg_prec = sum(r["prec"] for r in per_image) / n
        avg_f1 = sum(r["f1"] for r in per_image) / n
        avg_fp = sum(r["fp_count"] for r in per_image) / n
        marker = " ★ (현재)" if judge_name == "gemini-pro" else ""
        print(f"  {judge_name:<20} {avg_acc:>6.1f}% {avg_prec:>6.1f}% {avg_f1:>6.1f}% {avg_fp:>8.2f}{marker}")
        summary.append({
            "judge": judge_name,
            "avg_acc": f"{avg_acc:.1f}%",
            "avg_prec": f"{avg_prec:.1f}%",
            "avg_f1": f"{avg_f1:.1f}%",
            "avg_fp": f"{avg_fp:.2f}",
        })

    # CSV 저장
    import csv
    os.makedirs("csv", exist_ok=True)
    with open("csv/judge_ablation_results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=summary[0].keys())
        w.writeheader()
        w.writerows(summary)
    print(f"\n  요약 저장: csv/judge_ablation_results.csv")

    # 이미지별 상세
    detail_rows = []
    for judge_name, per_image in judge_results.items():
        for r in per_image:
            detail_rows.append({
                "judge": judge_name,
                "category": r["category"], "image": r["image"],
                "acc": f"{r['acc']:.1f}", "prec": f"{r['prec']:.1f}",
                "f1": f"{r['f1']:.1f}", "fp": r["fp_count"],
                "missed": r["missed_count"], "n_disputed": r["n_disputed"],
            })
    with open("csv/judge_ablation_per_image.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=detail_rows[0].keys())
        w.writeheader()
        w.writerows(detail_rows)
    print(f"  이미지별 저장: csv/judge_ablation_per_image.csv")

if __name__ == "__main__":
    main()
