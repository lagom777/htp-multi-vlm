"""v7 — 로컬 3 VLM + Judge rotate (Qwen/EXAONE/Gemma) on 32장.

R1, R2: 3 로컬 모델 (Qwen 8005, EXAONE 8003, Gemma 8004)
Judge: 같은 3 모델 중 1개 rotate (3 실험)

호출 비용: $0
"""
import os, json, base64, time, unicodedata, re, sys
from datetime import datetime
from collections import Counter, defaultdict
from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'framework'))
import v6_voting_debate_4 as v6m
from project_paths import IMAGE_DIR, LABEL_DIR

HOST = "192.168.200.138"
ENDPOINTS = {
    "qwen": {"port": 8005, "model": "RedHatAI/Qwen3.6-35B-A3B-NVFP4", "thinking_off": True},
    "exaone": {"port": 8003, "model": "LGAI-EXAONE/EXAONE-4.5-33B-AWQ", "thinking_off": True},
    "gemma": {"port": 8004, "model": "RedHatAI/gemma-4-26B-A4B-it-NVFP4", "thinking_off": False},
}

TEST_IMAGES = [
    ("TL_나무", "나무_8_남_01445.jpg"),
    ("TL_나무", "나무_10_여_00019.jpg"),
    ("TL_나무", "나무_11_남_00004.jpg"),
    ("TL_나무", "나무_12_여_00007.jpg"),
    ("TL_나무", "나무_8_여_00041.jpg"),
    ("TL_나무", "나무_10_남_00013.jpg"),
    ("TL_나무", "나무_11_여_00000.jpg"),
    ("TL_나무", "나무_13_남_00001.jpg"),
    ("TL_집", "집_12_여_08971.jpg"),
    ("TL_집", "집_10_여_00006.jpg"),
    ("TL_집", "집_11_남_00013.jpg"),
    ("TL_집", "집_12_여_00007.jpg"),
    ("TL_집", "집_8_여_00066.jpg"),
    ("TL_집", "집_10_남_00015.jpg"),
    ("TL_집", "집_11_여_00009.jpg"),
    ("TL_집", "집_13_남_00002.jpg"),
    ("TL_남자사람", "남자사람_13_남_02804.jpg"),
    ("TL_남자사람", "남자사람_10_여_00023.jpg"),
    ("TL_남자사람", "남자사람_11_남_00000.jpg"),
    ("TL_남자사람", "남자사람_12_여_00005.jpg"),
    ("TL_남자사람", "남자사람_8_여_00016.jpg"),
    ("TL_남자사람", "남자사람_10_남_00022.jpg"),
    ("TL_남자사람", "남자사람_11_여_00002.jpg"),
    ("TL_남자사람", "남자사람_13_남_00051.jpg"),
    ("TL_여자사람", "여자사람_10_남_02125.jpg"),
    ("TL_여자사람", "여자사람_10_여_00018.jpg"),
    ("TL_여자사람", "여자사람_11_남_00002.jpg"),
    ("TL_여자사람", "여자사람_12_여_00008.jpg"),
    ("TL_여자사람", "여자사람_8_여_00081.jpg"),
    ("TL_여자사람", "여자사람_10_남_00010.jpg"),
    ("TL_여자사람", "여자사람_11_여_00001.jpg"),
    ("TL_여자사람", "여자사람_13_남_00009.jpg"),
]

BASE_IMG = str(IMAGE_DIR)
TS_MAP = {"TL_나무": "TS_나무", "TL_집": "TS_집",
          "TL_남자사람": "TS_남자사람", "TL_여자사람": "TS_여자사람"}

TEMP_R1 = 0.2
TEMP_R2 = 0.5
TEMP_JUDGE = 0.2
MAX_TOKENS = 16000


def find_path(base, parts):
    for variants in [parts, [unicodedata.normalize('NFD', p) for p in parts]]:
        cand = os.path.join(base, *variants)
        if os.path.exists(cand):
            return cand
    for i in range(len(parts)):
        v = list(parts)
        v[i] = unicodedata.normalize('NFD', v[i])
        c = os.path.join(base, *v)
        if os.path.exists(c):
            return c
    return None


def client_for(name):
    ep = ENDPOINTS[name]
    return OpenAI(base_url=f"http://{HOST}:{ep['port']}/v1", api_key="local", timeout=300)


def call_vlm(name, prompt, img_path=None, temp=0.2, retries=2):
    ep = ENDPOINTS[name]
    cli = client_for(name)
    if img_path:
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]
    else:
        content = prompt

    kwargs = {
        "model": ep["model"],
        "messages": [{"role": "user", "content": content}],
        "temperature": temp,
        "max_tokens": MAX_TOKENS,
    }
    if ep["thinking_off"]:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    for attempt in range(retries):
        try:
            r = cli.chat.completions.create(**kwargs)
            return r.choices[0].message.content or ""
        except Exception as e:
            print(f"    {name} try{attempt+1} 에러: {str(e)[:100]}", flush=True)
            time.sleep(3)
    return ""


def parse_json_block(text):
    if not text or text == "Error":
        return None
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    raw = m.group(1) if m else text.strip()
    try:
        return json.loads(raw)
    except Exception:
        m2 = re.search(r"\[\s*\{.*\}\s*\]", raw, re.DOTALL)
        if m2:
            try:
                return json.loads(m2.group(0))
            except Exception:
                pass
    return None


def majority_vote(round_answers):
    """Raw 객체명 정확 매치 (Strict)."""
    votes = defaultdict(dict)
    for model, ans_list in round_answers.items():
        if not isinstance(ans_list, list):
            continue
        for item in ans_list:
            if not isinstance(item, dict):
                continue
            if item.get("판정") != "있음":
                continue
            raw = item.get("객체", "").strip()
            if not raw:
                continue
            votes[raw][model] = item

    consensus, disagreed = {}, {}
    for obj, mi in votes.items():
        if len(mi) >= 2:
            pos_c = Counter([it.get("위치", "") for it in mi.values() if it.get("위치")])
            best = pos_c.most_common(1)[0][0] if pos_c else ""
            consensus[obj] = {"판정": "있음", "위치": best, "참여_모델": list(mi.keys())}
            if len(pos_c) > 1:
                disagreed[obj] = list(mi.items())
        else:
            disagreed[obj] = list(mi.items())
    return consensus, disagreed


def reload_gt(cat, img):
    p = find_path(str(LABEL_DIR), [cat, img.replace(".jpg", ".json")])
    if not p:
        return {}
    with open(p) as f:
        d = json.load(f)
    res_w = res_h = 1280
    if "meta" in d and "img_resolution" in d["meta"]:
        try:
            wh = d["meta"]["img_resolution"].split("x")
            res_w, res_h = int(wh[0]), int(wh[1])
        except:
            pass
    by = defaultdict(list)
    for bb in d.get("annotations", {}).get("bbox", []):
        n = v6m.normalize_label(bb["label"])
        cx = (bb["x"] + bb["w"] / 2) / res_w
        cy = (bb["y"] + bb["h"] / 2) / res_h
        col = "left" if cx < 1/3 else ("right" if cx > 2/3 else "center")
        row = "top" if cy < 1/3 else ("bottom" if cy > 2/3 else "middle")
        by[n].append(f"{row}-{col}")
    return dict(by)


def eval_final(final, gt):
    """매핑은 채점 시에만 — Strict."""
    final_norm = {}
    for k, v in final.items():
        n = v6m.normalize_label(k)
        if n and n not in final_norm:
            final_norm[n] = v
    tp = fn = fp = pc = pt = 0
    halluc = []
    for gl, gpos in gt.items():
        if gl in final_norm:
            tp += 1
            pt += 1
            fpos = final_norm[gl].get("위치", "")
            if fpos and fpos in gpos:
                pc += 1
        else:
            fn += 1
    for fo in final_norm:
        if fo not in gt:
            fp += 1
            halluc.append(fo)
    return dict(tp=tp, fn=fn, fp=fp, pc=pc, pt=pt, halluc=halluc)


# ===== 카테고리 필터 (병렬 실행용) =====
CAT_FILTER = sys.argv[1] if len(sys.argv) > 1 else None
JUDGE_NAME = sys.argv[2] if len(sys.argv) > 2 else "qwen"  # qwen/exaone/gemma

if CAT_FILTER and CAT_FILTER.startswith("TL_"):
    TEST_IMAGES = [x for x in TEST_IMAGES if x[0] == CAT_FILTER]


def make_judge_prompt(cat, disagreed_objects, all_answers):
    obj_summary = "\n".join([f"  - {obj}" for obj in disagreed_objects])
    detail = ""
    for obj in disagreed_objects:
        detail += f"\n[객체: {obj}]\n"
        for model_name, ans_list in all_answers.items():
            obj_ans = next((a for a in ans_list if isinstance(a, dict) and a.get("객체") == obj), None)
            if obj_ans:
                detail += f"  {model_name}: 판정={obj_ans.get('판정')}, 위치={obj_ans.get('위치')}, 근거={obj_ans.get('근거','')}\n"
            else:
                detail += f"  {model_name}: (답에 없음)\n"
    return f"""당신은 HTP 심리검사 그림 분석의 Judge입니다.
**이미지는 볼 수 없습니다.** 오직 다른 세 모델이 제공한 텍스트 근거만 보고 판정하세요.

카테고리: {v6m.CATEGORY_DESC[cat]}

다음 객체들에 대해 세 모델이 의견이 갈렸습니다:
{obj_summary}

세 모델의 답변 상세:
{detail}

각 객체에 대해 다음 기준으로 판정:
1. 위치·근거가 구체적이고 일관된 모델 다수가 있다면 채택
2. 한 모델만 주장하더라도 근거가 매우 구체적이고 반대 근거가 모호하면 채택
3. 모든 근거가 모호하면 거절

JSON 배열로만 답:
```json
[
  {{
    "객체": "(객체명)",
    "최종_판정": "있음" 또는 "없음",
    "최종_위치": "(9-region 라벨, 있음일 때만)",
    "Judge_근거": "(왜 이 판정인지)"
  }}
]
```"""


def main():
    suffix = f"_{CAT_FILTER}_judge-{JUDGE_NAME}" if CAT_FILTER else f"_all_judge-{JUDGE_NAME}"
    out_path = f"./v7_local_state{suffix}.json"
    log_path = f"./v7_local_state{suffix}.log"

    print(f"=== v7 로컬 — {CAT_FILTER or 'ALL'} / Judge {JUDGE_NAME} ===")
    print(f"시작: {datetime.now().strftime('%H:%M:%S')}, 이미지 {len(TEST_IMAGES)}장")
    print(f"R1·R2 VLM: qwen + exaone + gemma (로컬)")
    print(f"Judge: {JUDGE_NAME} ({ENDPOINTS[JUDGE_NAME]['model']})")

    results = []
    VLM_NAMES = ["qwen", "exaone", "gemma"]

    for idx, (cat, img_name) in enumerate(TEST_IMAGES):
        print(f"\n[{idx+1}/{len(TEST_IMAGES)}] {cat}/{img_name}", flush=True)
        img_path = find_path(BASE_IMG, [TS_MAP[cat], img_name])
        if not img_path:
            print(f"  ❌ 이미지 없음")
            continue
        gt = reload_gt(cat, img_name)
        gt_n = len(gt)

        # ===== Round 1 =====
        round1 = {}
        prompt_r1 = v6m.make_r1_prompt(cat)
        for name in VLM_NAMES:
            t0 = time.time()
            raw = call_vlm(name, prompt_r1, img_path, TEMP_R1)
            elapsed = time.time() - t0
            parsed = parse_json_block(raw)
            n_obj = len(parsed) if isinstance(parsed, list) else 0
            status = "✅" if n_obj else "⚠"
            print(f"  R1 {status} {name} {elapsed:.1f}s, 객체 {n_obj}개", flush=True)
            round1[name] = parsed

        # ===== Voting 1차 =====
        r1_cons, r1_dis = majority_vote(round1)
        print(f"  R1 합의 {len(r1_cons)}, 불일치 {len(r1_dis)}", flush=True)

        # ===== Round 2 (불일치 객체만) =====
        round2 = {}
        r2_cons, r2_dis = {}, {}
        if r1_dis:
            disagreed_names = list(r1_dis.keys())
            for name in VLM_NAMES:
                own = round1.get(name, []) or []
                others = {n: round1.get(n, []) for n in VLM_NAMES if n != name}
                prompt_r2 = v6m.make_r2_prompt(cat, own, others, disagreed_names)
                t0 = time.time()
                raw = call_vlm(name, prompt_r2, img_path, TEMP_R2)
                elapsed = time.time() - t0
                parsed = parse_json_block(raw)
                n_obj = len(parsed) if isinstance(parsed, list) else 0
                status = "✅" if n_obj else "⚠"
                print(f"  R2 {status} {name} {elapsed:.1f}s, 객체 {n_obj}개", flush=True)
                round2[name] = parsed
            r2_cons, r2_dis = majority_vote(round2)
            print(f"  R2 후 합의 {len(r2_cons)}, 잔여 {len(r2_dis)}", flush=True)

        # ===== Judge (잔여 불일치) =====
        judge_parsed = None
        if r2_dis:
            evidence = {}
            for n in VLM_NAMES:
                r1 = round1.get(n) or []
                r2 = round2.get(n) or []
                evidence[n] = (r1 + r2) if isinstance(r1, list) and isinstance(r2, list) else (r1 or r2 or [])
            j_prompt = make_judge_prompt(cat, list(r2_dis.keys()), evidence)
            t0 = time.time()
            j_raw = call_vlm(JUDGE_NAME, j_prompt, img_path=None, temp=TEMP_JUDGE)
            elapsed = time.time() - t0
            judge_parsed = parse_json_block(j_raw)
            n_j = len(judge_parsed) if isinstance(judge_parsed, list) else 0
            print(f"  Judge ({JUDGE_NAME}) {elapsed:.1f}s, {n_j}건", flush=True)

        # ===== 최종 통합 =====
        final = {}
        final.update(r1_cons)
        final.update(r2_cons)
        if isinstance(judge_parsed, list):
            for jit in judge_parsed:
                if not isinstance(jit, dict):
                    continue
                obj = jit.get("객체", "").strip()
                if jit.get("최종_판정") == "있음" and obj:
                    final[obj] = {"판정": "있음", "위치": jit.get("최종_위치", "")}

        # ===== 평가 =====
        ev = eval_final(final, gt)
        rec = ev['tp'] / gt_n if gt_n else 0
        prec = ev['tp'] / (ev['tp']+ev['fp']) if (ev['tp']+ev['fp']) else 0
        f1 = 2*rec*prec/(rec+prec) if (rec+prec) else 0
        pacc = ev['pc']/ev['pt'] if ev['pt'] else 0
        print(f"  📊 TP {ev['tp']}/{gt_n}, F1 {f1*100:.1f}%, 위치 {pacc*100:.1f}%, 환각 {ev['fp']}", flush=True)

        results.append({
            "cat": cat, "img": img_name, "gt_n": gt_n,
            "round1": {n: round1.get(n) for n in VLM_NAMES},
            "round2": {n: round2.get(n) for n in VLM_NAMES},
            "judge_parsed": judge_parsed,
            "final": final,
            "eval": ev,
            "recall": rec, "f1": f1, "pos_acc": pacc,
        })

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # ===== 합산 =====
    tot = Counter()
    for r in results:
        for k in ['tp', 'fn', 'fp', 'pc', 'pt']:
            tot[k] += r['eval'][k]
        tot['gt_n'] += r['gt_n']
    rec = tot['tp']/tot['gt_n'] if tot['gt_n'] else 0
    prec = tot['tp']/(tot['tp']+tot['fp']) if (tot['tp']+tot['fp']) else 0
    f1 = 2*rec*prec/(rec+prec) if (rec+prec) else 0
    pacc = tot['pc']/tot['pt'] if tot['pt'] else 0
    avg_hall = tot['fp']/len(results) if results else 0
    print(f"\n=== 완료: {datetime.now().strftime('%H:%M:%S')} ===")
    print(f"  Judge: {JUDGE_NAME}, N={len(results)}장")
    print(f"  Recall {rec*100:.1f}%, Prec {prec*100:.1f}%, F1 {f1*100:.1f}%, "
          f"9-Pos {pacc*100:.1f}%, 환각/img {avg_hall:.2f}")
    print(f"  저장: {out_path}")


if __name__ == "__main__":
    main()
