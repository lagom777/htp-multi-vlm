"""Y 모델 = MACT (Multi-Agent Cognitive Tree).

4단계 인지 흐름:
  Planner (오케스트레이터) → 무엇을 봐야 할지 계획
  Executor (3 로컬 모델) → 자유 분석
  Judgment (오케스트레이터) → 통합·평가
  Answer (오케스트레이터) → 최종 5-field JSON

오케스트레이터: qwen / exaone / gemma 중 1개 rotate.
"""
import os, json, base64, time, unicodedata, re, sys
from datetime import datetime
from collections import Counter, defaultdict
from openai import OpenAI
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'framework'))
import v6_voting_debate_4 as v6m

HOST = "192.168.200.138"
ENDPOINTS = {
    "qwen":   {"port":8005, "model":"RedHatAI/Qwen3.6-35B-A3B-NVFP4",   "thinking_off":True},
    "exaone": {"port":8003, "model":"LGAI-EXAONE/EXAONE-4.5-33B-AWQ",   "thinking_off":True},
    "gemma":  {"port":8004, "model":"RedHatAI/gemma-4-26B-A4B-it-NVFP4","thinking_off":False},
}

# 8장 (카테고리당 2장)
TEST_IMAGES = [
    ("TL_나무","나무_8_남_01445.jpg"),
    ("TL_집","집_12_여_08971.jpg"),
    ("TL_남자사람","남자사람_13_남_02804.jpg"),
    ("TL_여자사람","여자사람_10_남_02125.jpg"),
]

BASE_IMG = ("./"
            "266.AI 기반 아동 미술심리 진단을 위한 그림 데이터 구축/"
            "01-1.정식개방데이터/Training/01.원천데이터")
BASE_LBL = ("./"
            "266.AI 기반 아동 미술심리 진단을 위한 그림 데이터 구축/"
            "01-1.정식개방데이터/Training/02.라벨링데이터")
TS_MAP = {"TL_나무":"TS_나무","TL_집":"TS_집","TL_남자사람":"TS_남자사람","TL_여자사람":"TS_여자사람"}


def fp(base, parts):
    for v in [parts, [unicodedata.normalize('NFD', p) for p in parts]]:
        c = os.path.join(base, *v)
        if os.path.exists(c): return c
    for i in range(len(parts)):
        v = list(parts); v[i] = unicodedata.normalize('NFD', v[i])
        c = os.path.join(base, *v)
        if os.path.exists(c): return c
    return None


def load_gt(cat, img):
    p = fp(BASE_LBL, [cat, img.replace(".jpg",".json")])
    if not p: return {}
    with open(p) as f: d = json.load(f)
    rw=rh=1280
    if "meta" in d and "img_resolution" in d["meta"]:
        try:
            wh = d["meta"]["img_resolution"].split("x"); rw=int(wh[0]); rh=int(wh[1])
        except: pass
    by=defaultdict(list)
    for bb in d.get("annotations",{}).get("bbox",[]):
        n=v6m.normalize_label(bb["label"])
        cx=(bb["x"]+bb["w"]/2)/rw; cy=(bb["y"]+bb["h"]/2)/rh
        col="left" if cx<1/3 else ("right" if cx>2/3 else "center")
        row="top" if cy<1/3 else ("bottom" if cy>2/3 else "middle")
        by[n].append(f"{row}-{col}")
    return dict(by)


def call_vlm(name, prompt, img_path=None, temp=0.2, retries=2):
    ep = ENDPOINTS[name]
    cli = OpenAI(base_url=f"http://{HOST}:{ep['port']}/v1", api_key="local", timeout=300)
    if img_path:
        with open(img_path,"rb") as f: b64=base64.b64encode(f.read()).decode("utf-8")
        content=[{"type":"text","text":prompt},
                 {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}]
    else:
        content=prompt
    kw={"model":ep["model"],"messages":[{"role":"user","content":content}],
        "temperature":temp,"max_tokens":16000}
    if ep["thinking_off"]:
        kw["extra_body"]={"chat_template_kwargs":{"enable_thinking":False}}
    for a in range(retries):
        try:
            r = cli.chat.completions.create(**kw)
            return r.choices[0].message.content or ""
        except Exception as e:
            print(f"    {name} try{a+1}: {str(e)[:100]}", flush=True)
            time.sleep(3)
    return ""


def parse_json(text):
    if not text: return None
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    raw = m.group(1) if m else text.strip()
    try: return json.loads(raw)
    except:
        m2 = re.search(r"\[\s*\{.*\}\s*\]|\{.*\}", raw, re.DOTALL)
        if m2:
            try: return json.loads(m2.group(0))
            except: pass
    return None


def make_planner_prompt(cat):
    desc = v6m.CATEGORY_DESC[cat]
    return f"""당신은 HTP 심리검사 그림 분석의 Planner입니다.
카테고리: {desc}

이 그림을 분석하기 위해 어떤 시각적 요소를 봐야 하는지 계획을 세우세요.
이미지는 보지 않고 카테고리 지식만으로 계획.

JSON으로 출력:
```json
{{
  "주요_관찰_포인트": ["...", "...", ...],
  "위치_평가_기준": "...",
  "환각_위험_신호": ["...", ...]
}}
```"""


def make_executor_prompt(cat, plan):
    base_prompt = v6m.make_r1_prompt(cat)
    plan_text = json.dumps(plan, ensure_ascii=False, indent=2) if plan else "(계획 없음)"
    return (
        f"[Planner의 계획]\n{plan_text}\n\n"
        f"위 계획을 참고하되, 그림에 보이는 객체를 자유롭게 식별하세요.\n\n"
        + base_prompt
    )


def make_judgment_prompt(cat, plan, exec_outputs):
    desc = v6m.CATEGORY_DESC[cat]
    detail = ""
    for model, ans in exec_outputs.items():
        s = json.dumps(ans, ensure_ascii=False)[:1500] if ans else "(빈 답)"
        detail += f"\n[{model}]\n{s}\n"
    plan_text = json.dumps(plan, ensure_ascii=False) if plan else ""
    return f"""당신은 HTP 심리검사 그림 분석의 Judgment 모듈입니다.
**이미지는 볼 수 없습니다.** 세 Executor 모델의 답과 Planner 계획만 보고 통합합니다.

카테고리: {desc}

[Planner 계획]
{plan_text}

[3 Executor 답]
{detail}

다음 기준으로 통합:
1. 2/3 이상 합의 객체 → 채택
2. 1 모델 주장이지만 근거가 매우 구체적 + Planner 계획에 부합 → 채택
3. 모든 근거가 모호하거나 한 모델의 주관적 해석으로만 보이는 경우 → 거절

JSON 배열로만 출력:
```json
[
  {{
    "객체": "(객체명, 한국어 단일 명사)",
    "판정": "있음",
    "위치": "(9-region 라벨)",
    "크기": "(설명)",
    "근거": "(통합 근거)"
  }}
]
```

위치 9개 라벨:
  top-left | top-center | top-right
  middle-left | middle-center | middle-right
  bottom-left | bottom-center | bottom-right"""


def eval_final(final, gt):
    final_norm = {}
    for k, v in final.items():
        n = v6m.normalize_label(k)
        if n and n not in final_norm: final_norm[n] = v
    tp=fn=fp=pc=pt=0
    for gl, gpos in gt.items():
        if gl in final_norm:
            tp+=1; pt+=1
            fpos = final_norm[gl].get("위치","")
            if fpos and fpos in gpos: pc+=1
        else: fn+=1
    for fo in final_norm:
        if fo not in gt: fp+=1
    return dict(tp=tp,fn=fn,fp=fp,pc=pc,pt=pt)


def main():
    ORCH = sys.argv[1] if len(sys.argv) > 1 else "qwen"
    print(f"=== Y 모델 (MACT) — 오케스트레이터: {ORCH} ===")
    print(f"시작: {datetime.now().strftime('%H:%M:%S')}")

    results = []
    EXECS = ["qwen", "exaone", "gemma"]

    for idx, (cat, img) in enumerate(TEST_IMAGES):
        print(f"\n[{idx+1}/8] {cat}/{img}", flush=True)
        img_path = fp(BASE_IMG, [TS_MAP[cat], img])
        gt = load_gt(cat, img)
        gt_n = len(gt)

        # 1. Planner
        plan_raw = call_vlm(ORCH, make_planner_prompt(cat), img_path=None, temp=0.2)
        plan = parse_json(plan_raw) or {}
        print(f"  Planner({ORCH}): {len(json.dumps(plan))} chars", flush=True)

        # 2. Executor (3 모델)
        executor_outputs = {}
        for ex in EXECS:
            t0 = time.time()
            raw = call_vlm(ex, make_executor_prompt(cat, plan), img_path=img_path, temp=0.2)
            el = time.time()-t0
            parsed = parse_json(raw)
            n_obj = len(parsed) if isinstance(parsed, list) else 0
            print(f"  Executor {ex}: {el:.1f}s, {n_obj} 객체", flush=True)
            executor_outputs[ex] = parsed if isinstance(parsed, list) else []

        # 3. Judgment (오케스트레이터)
        j_raw = call_vlm(ORCH, make_judgment_prompt(cat, plan, executor_outputs),
                         img_path=None, temp=0.2)
        j_parsed = parse_json(j_raw)
        print(f"  Judgment({ORCH}): {len(j_parsed) if isinstance(j_parsed, list) else 0} 객체", flush=True)

        # 4. Answer = Judgment 결과
        final = {}
        if isinstance(j_parsed, list):
            for it in j_parsed:
                if not isinstance(it, dict): continue
                if it.get("판정") != "있음": continue
                obj = it.get("객체","").strip()
                if obj and obj not in final:
                    final[obj] = {"위치": it.get("위치",""), "크기": it.get("크기","")}

        ev = eval_final(final, gt)
        rec = ev['tp']/gt_n if gt_n else 0
        prec = ev['tp']/(ev['tp']+ev['fp']) if (ev['tp']+ev['fp']) else 0
        f1 = 2*rec*prec/(rec+prec) if (rec+prec) else 0
        pacc = ev['pc']/ev['pt'] if ev['pt'] else 0
        print(f"  📊 TP {ev['tp']}/{gt_n}, F1 {f1*100:.1f}%, 위치 {pacc*100:.1f}%, 환각 {ev['fp']}", flush=True)
        results.append({"cat":cat,"img":img,"gt_n":gt_n,"plan":plan,
                        "executor":executor_outputs,"judgment":j_parsed,
                        "final":final,"eval":ev})

        with open(f"./y_mact_orch-{ORCH}.json","w",encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # 합산
    tot = Counter()
    for r in results:
        for k in ['tp','fn','fp','pc','pt']: tot[k] += r['eval'][k]
        tot['gtn'] += r['gt_n']
    rec = tot['tp']/tot['gtn'] if tot['gtn'] else 0
    prec = tot['tp']/(tot['tp']+tot['fp']) if (tot['tp']+tot['fp']) else 0
    f1 = 2*rec*prec/(rec+prec) if (rec+prec) else 0
    pacc = tot['pc']/tot['pt'] if tot['pt'] else 0
    print(f"\n=== Y(MACT) 완료 {datetime.now().strftime('%H:%M:%S')} ===")
    print(f"  오케스트레이터: {ORCH}, N={len(results)}장")
    print(f"  Recall {rec*100:.1f}%, Prec {prec*100:.1f}%, F1 {f1*100:.1f}%, "
          f"9-Pos {pacc*100:.1f}%, 환각 {tot['fp']} ({tot['fp']/len(results):.2f}/img)")


if __name__ == "__main__":
    main()
