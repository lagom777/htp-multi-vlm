"""Z 모델 = Tree-of-Debate.

분기 토의 구조:
  Initial: 3 모델 독립 답 (3 branches)
  Refinement: 각 모델이 다른 두 모델 답 보고 revision (3 refined branches)
  Judge: 오케스트레이터가 트리 평가 후 최적 객체 집합 선택

오케스트레이터: qwen / exaone / gemma 중 1개 rotate.
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
    "qwen":   {"port":8005, "model":"RedHatAI/Qwen3.6-35B-A3B-NVFP4",   "thinking_off":True},
    "exaone": {"port":8003, "model":"LGAI-EXAONE/EXAONE-4.5-33B-AWQ",   "thinking_off":True},
    "gemma":  {"port":8004, "model":"RedHatAI/gemma-4-26B-A4B-it-NVFP4","thinking_off":False},
}

TEST_IMAGES = [
    ("TL_나무","나무_8_남_01445.jpg"),
    ("TL_집","집_12_여_08971.jpg"),
    ("TL_남자사람","남자사람_13_남_02804.jpg"),
    ("TL_여자사람","여자사람_10_남_02125.jpg"),
]

BASE_IMG = str(IMAGE_DIR)
BASE_LBL = str(LABEL_DIR)
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
        m2 = re.search(r"\[\s*\{.*\}\s*\]", raw, re.DOTALL)
        if m2:
            try: return json.loads(m2.group(0))
            except: pass
    return None


def make_refinement_prompt(cat, own_initial, others_initial):
    desc = v6m.CATEGORY_DESC[cat]
    own_str = json.dumps(own_initial, ensure_ascii=False)[:1500] if own_initial else "(빈 답)"
    others_str = ""
    for m_name, m_ans in others_initial.items():
        s = json.dumps(m_ans, ensure_ascii=False)[:1000] if m_ans else "(빈 답)"
        others_str += f"\n[{m_name}]\n{s}\n"
    return f"""당신은 HTP 심리검사 그림 분석 중입니다. 카테고리: {desc}

당신의 초기 답:
{own_str}

다른 두 모델의 초기 답:
{others_str}

다른 모델의 답을 참고하여 당신의 답을 개선하세요.
- 다른 모델이 본 객체가 당신 답에 없고 합리적이면 추가
- 당신 답에만 있는 객체가 약한 근거라면 제거 또는 유지 결정
- 위치·근거가 다른 경우, 그림 다시 본 뒤 최종 결정

5-field JSON 배열로 출력:
```json
[
  {{
    "객체": "(객체명, 한국어 단일 명사)",
    "판정": "있음",
    "위치": "(9-region 라벨)",
    "크기": "(설명)",
    "근거": "(시각적 근거)"
  }}
]
```

위치 9개 라벨:
  top-left | top-center | top-right
  middle-left | middle-center | middle-right
  bottom-left | bottom-center | bottom-right"""


def make_tree_judge_prompt(cat, initial_branches, refined_branches):
    desc = v6m.CATEGORY_DESC[cat]
    tree_str = ""
    for m in initial_branches.keys():
        init = json.dumps(initial_branches.get(m), ensure_ascii=False)[:800] if initial_branches.get(m) else "(빈)"
        ref = json.dumps(refined_branches.get(m), ensure_ascii=False)[:1200] if refined_branches.get(m) else "(빈)"
        tree_str += f"\n=== Branch [{m}] ===\nInitial:\n{init}\nRefined:\n{ref}\n"
    return f"""당신은 HTP 심리검사 그림 분석의 Tree Judge입니다.
**이미지는 볼 수 없습니다.** 세 모델의 분기 트리(Initial → Refined)를 모두 평가하여 최적 객체 집합을 선택합니다.

카테고리: {desc}

[분기 트리]
{tree_str}

평가 기준:
1. Refined 단계에서 안정적으로 등장한 객체(2/3 이상) → 채택
2. Refinement에서 새로 등장하거나 빠진 객체는 근거 평가 후 결정
3. Initial에는 있었으나 모든 Refined에서 빠진 객체 → 제거 (수렴 신호)
4. 한 branch만 주장하더라도 근거가 매우 구체적이면 채택

JSON 배열로만 출력:
```json
[
  {{
    "객체": "(객체명, 한국어 단일 명사)",
    "판정": "있음",
    "위치": "(9-region 라벨)",
    "크기": "(설명)",
    "근거": "(어느 branch에서 어떻게 수렴했는지)"
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
    print(f"=== Z 모델 (Tree-of-Debate) — 오케스트레이터: {ORCH} ===")
    print(f"시작: {datetime.now().strftime('%H:%M:%S')}")

    results = []
    MODELS = ["qwen", "exaone", "gemma"]

    for idx, (cat, img) in enumerate(TEST_IMAGES):
        print(f"\n[{idx+1}/8] {cat}/{img}", flush=True)
        img_path = fp(BASE_IMG, [TS_MAP[cat], img])
        gt = load_gt(cat, img)
        gt_n = len(gt)

        # 1. Initial branches — 3 모델 독립 답
        initial = {}
        prompt_init = v6m.make_r1_prompt(cat)
        for m in MODELS:
            t0 = time.time()
            raw = call_vlm(m, prompt_init, img_path=img_path, temp=0.2)
            el = time.time()-t0
            parsed = parse_json(raw)
            n_obj = len(parsed) if isinstance(parsed, list) else 0
            print(f"  Initial {m}: {el:.1f}s, {n_obj} 객체", flush=True)
            initial[m] = parsed if isinstance(parsed, list) else []

        # 2. Refinement — 각 모델이 다른 두 모델 답 보고 revision
        refined = {}
        for m in MODELS:
            own = initial.get(m, [])
            others = {n: initial.get(n, []) for n in MODELS if n != m}
            t0 = time.time()
            raw = call_vlm(m, make_refinement_prompt(cat, own, others),
                           img_path=img_path, temp=0.5)
            el = time.time()-t0
            parsed = parse_json(raw)
            n_obj = len(parsed) if isinstance(parsed, list) else 0
            print(f"  Refined {m}: {el:.1f}s, {n_obj} 객체", flush=True)
            refined[m] = parsed if isinstance(parsed, list) else []

        # 3. Tree Judge — 오케스트레이터가 트리 평가
        j_raw = call_vlm(ORCH, make_tree_judge_prompt(cat, initial, refined),
                         img_path=None, temp=0.2)
        j_parsed = parse_json(j_raw)
        print(f"  TreeJudge({ORCH}): {len(j_parsed) if isinstance(j_parsed, list) else 0} 객체", flush=True)

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
        results.append({"cat":cat,"img":img,"gt_n":gt_n,
                        "initial":initial,"refined":refined,
                        "tree_judge":j_parsed,"final":final,"eval":ev})

        with open(f"./z_treedebate_orch-{ORCH}.json","w",encoding="utf-8") as f:
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
    print(f"\n=== Z(Tree-of-Debate) 완료 {datetime.now().strftime('%H:%M:%S')} ===")
    print(f"  오케스트레이터: {ORCH}, N={len(results)}장")
    print(f"  Recall {rec*100:.1f}%, Prec {prec*100:.1f}%, F1 {f1*100:.1f}%, "
          f"9-Pos {pacc*100:.1f}%, 환각 {tot['fp']} ({tot['fp']/len(results):.2f}/img)")


if __name__ == "__main__":
    main()
