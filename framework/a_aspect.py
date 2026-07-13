"""A 모델 = Aspect-based Multi-Agent.

3 전문가 + 1 Coordinator:
  객체 식별 전문가: 무엇이 그려졌나
  위치 전문가: 각 객체의 9-region 위치
  세부 전문가: 크기·형태·표현 세부
  Coordinator (오케스트레이터): 3 전문가 답 통합 → 5-field JSON

전문가 = 3 로컬 모델 분담 (qwen 객체식별, exaone 위치, gemma 세부).
"""
import os, json, base64, time, unicodedata, re, sys
from datetime import datetime
from collections import Counter, defaultdict
from openai import OpenAI
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v6_voting_debate_4 as v6m

HOST = "192.168.200.138"
# 저사양 로컬 (현재 활성 — 64장 측정): qwen 8006 Qwen3VL-4B / exaone 8007 InternVL3.5-2B / gemma 8008 gemma-3-4b
# 강한 로컬 (비활성): qwen 8005 Qwen3.6-35B(T) · exaone 8003 EXAONE-4.5-33B(T) · gemma 8004 gemma-4-26B(F)
ENDPOINTS = {
    "qwen":   {"port":8006, "model":"Qwen/Qwen3-VL-4B-Instruct-FP8", "thinking_off":False},
    "exaone": {"port":8007, "model":"OpenGVLab/InternVL3_5-2B",       "thinking_off":False},
    "gemma":  {"port":8008, "model":"unsloth/gemma-3-4b-it",          "thinking_off":False},
}

TEST_IMAGES = [
    ("TL_나무","나무_8_남_01445.jpg"),
    ("TL_집","집_12_여_08971.jpg"),
    ("TL_남자사람","남자사람_13_남_02804.jpg"),
    ("TL_여자사람","여자사람_10_남_02125.jpg"),
]

BASE_IMG = os.environ.get("HTP266_IMG_DIR", "./data/01.원천데이터")  # AI Hub 266 원천데이터 경로
BASE_LBL = os.environ.get("HTP266_LBL_DIR", "./data/02.라벨링데이터")  # AI Hub 266 라벨링데이터 경로
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
        try: wh = d["meta"]["img_resolution"].split("x"); rw=int(wh[0]); rh=int(wh[1])
        except: pass
    by=defaultdict(list)
    for bb in d.get("annotations",{}).get("bbox",[]):
        n=v6m.normalize_label(bb["label"])
        cx=(bb["x"]+bb["w"]/2)/rw; cy=(bb["y"]+bb["h"]/2)/rh
        col="left" if cx<1/3 else ("right" if cx>2/3 else "center")
        row="top" if cy<1/3 else ("bottom" if cy>2/3 else "middle")
        by[n].append(f"{row}-{col}")
    return dict(by)


class _VLMTimeout(Exception): pass
def _vlm_alarm(signum, frame): raise _VLMTimeout()

def call_vlm(name, prompt, img_path=None, temp=0.2, retries=2, hard_timeout=150, return_meta=False):
    import signal
    ep = ENDPOINTS[name]
    # max_retries=0: SDK 내부 재시도 끔(아래 루프에서만 재시도). timeout=read 보호용.
    cli = OpenAI(base_url=f"http://{HOST}:{ep['port']}/v1", api_key="local",
                 timeout=hard_timeout, max_retries=0)
    if img_path:
        with open(img_path,"rb") as f: b64=base64.b64encode(f.read()).decode("utf-8")
        content=[{"type":"text","text":prompt},
                 {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}]
    else:
        content=prompt
    kw={"model":ep["model"],"messages":[{"role":"user","content":content}],
        "temperature":temp,"max_tokens":4096}
    if ep["thinking_off"]:
        kw["extra_body"]={"chat_template_kwargs":{"enable_thinking":False}}
    # 메인스레드면 SIGALRM 하드 타임아웃(read-timeout 미작동 hang 방지)
    use_alarm = signal.getsignal is not None
    for a in range(retries):
        try:
            try:
                signal.signal(signal.SIGALRM, _vlm_alarm); signal.alarm(hard_timeout)
            except (ValueError, AttributeError):
                use_alarm = False
            r = cli.chat.completions.create(**kw)
            if use_alarm: signal.alarm(0)
            content = r.choices[0].message.content or ""
            if return_meta:
                return content, (r.choices[0].finish_reason or "")
            return content
        except Exception as e:
            try: signal.alarm(0)
            except Exception: pass
            print(f"    {name} try{a+1}: {str(e)[:100]}", flush=True)
            time.sleep(3)
    return ("", "no_response") if return_meta else ""


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


def make_object_prompt(cat):
    desc = v6m.CATEGORY_DESC[cat]
    return f"""당신은 '객체 식별 전문가'입니다. 카테고리: {desc}

이 그림에 어떤 객체들이 보이는지 가능한 한 자세히 식별하세요.
부위까지 분리 식별 (나무 → 기둥/가지/잎/뿌리 등, 사람 → 머리/눈/코/입/팔/다리 등).

JSON 배열로만 출력:
```json
["객체명1", "객체명2", "객체명3", ...]
```"""


def make_position_prompt(cat, object_list):
    desc = v6m.CATEGORY_DESC[cat]
    objs = ", ".join(object_list) if object_list else "(없음)"
    return f"""당신은 '위치 전문가'입니다. 카테고리: {desc}

다음 객체들이 그림에서 9-region 중 어느 위치에 있는지 판정하세요:
{objs}

위치 9개 라벨:
  top-left | top-center | top-right
  middle-left | middle-center | middle-right
  bottom-left | bottom-center | bottom-right

JSON 객체로만 출력:
```json
{{"객체명1": "위치", "객체명2": "위치", ...}}
```"""


def make_detail_prompt(cat, object_list):
    desc = v6m.CATEGORY_DESC[cat]
    objs = ", ".join(object_list) if object_list else "(없음)"
    return f"""당신은 '세부 묘사 전문가'입니다. 카테고리: {desc}

다음 객체들의 크기와 형태·표현 특징을 설명하세요:
{objs}

JSON 객체로만 출력:
```json
{{"객체명1": {{"크기": "...", "근거": "..."}}, "객체명2": {{...}}, ...}}
```"""


def make_coordinator_prompt(cat, obj_ans, pos_ans, det_ans):
    desc = v6m.CATEGORY_DESC[cat]
    o = json.dumps(obj_ans, ensure_ascii=False)[:1500]
    p = json.dumps(pos_ans, ensure_ascii=False)[:1500]
    d = json.dumps(det_ans, ensure_ascii=False)[:1500]
    return f"""당신은 Aspect Coordinator입니다. 세 전문가의 답을 통합합니다.
**이미지는 볼 수 없습니다.** 텍스트만 보고 통합하세요.

카테고리: {desc}

[객체 식별 전문가 답]
{o}

[위치 전문가 답]
{p}

[세부 묘사 전문가 답]
{d}

세 전문가 답을 5-field JSON으로 통합 (이상하거나 모호한 객체는 제외):
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
    # 전문가 분담: 객체 = qwen, 위치 = exaone, 세부 = gemma (기본)
    # 오케스트레이터(coordinator)가 통합
    OBJ_EXPERT = "qwen"
    POS_EXPERT = "exaone"
    DET_EXPERT = "gemma"
    print(f"=== A 모델 (Aspect-based) — Coordinator: {ORCH} ===")
    print(f"  객체={OBJ_EXPERT}, 위치={POS_EXPERT}, 세부={DET_EXPERT}")
    print(f"시작: {datetime.now().strftime('%H:%M:%S')}")

    results = []
    for idx, (cat, img) in enumerate(TEST_IMAGES):
        print(f"\n[{idx+1}/8] {cat}/{img}", flush=True)
        img_path = fp(BASE_IMG, [TS_MAP[cat], img])
        gt = load_gt(cat, img)
        gt_n = len(gt)

        # 1. 객체 식별 전문가
        t0=time.time()
        obj_raw = call_vlm(OBJ_EXPERT, make_object_prompt(cat), img_path=img_path, temp=0.2)
        obj_ans = parse_json(obj_raw)
        n_obj = len(obj_ans) if isinstance(obj_ans, list) else 0
        print(f"  ObjExpert({OBJ_EXPERT}): {time.time()-t0:.1f}s, {n_obj} 객체", flush=True)
        obj_list = obj_ans if isinstance(obj_ans, list) else []

        # 2. 위치 전문가
        t0=time.time()
        pos_raw = call_vlm(POS_EXPERT, make_position_prompt(cat, obj_list), img_path=img_path, temp=0.2)
        pos_ans = parse_json(pos_raw)
        n_pos = len(pos_ans) if isinstance(pos_ans, dict) else 0
        print(f"  PosExpert({POS_EXPERT}): {time.time()-t0:.1f}s, {n_pos} 위치", flush=True)

        # 3. 세부 묘사 전문가
        t0=time.time()
        det_raw = call_vlm(DET_EXPERT, make_detail_prompt(cat, obj_list), img_path=img_path, temp=0.2)
        det_ans = parse_json(det_raw)
        n_det = len(det_ans) if isinstance(det_ans, dict) else 0
        print(f"  DetExpert({DET_EXPERT}): {time.time()-t0:.1f}s, {n_det} 세부", flush=True)

        # 4. Coordinator
        t0=time.time()
        c_raw = call_vlm(ORCH, make_coordinator_prompt(cat, obj_list, pos_ans, det_ans),
                         img_path=None, temp=0.2)
        c_parsed = parse_json(c_raw)
        print(f"  Coordinator({ORCH}): {time.time()-t0:.1f}s, {len(c_parsed) if isinstance(c_parsed, list) else 0} 객체", flush=True)

        final = {}
        if isinstance(c_parsed, list):
            for it in c_parsed:
                if not isinstance(it, dict): continue
                if it.get("판정") != "있음": continue
                o = it.get("객체","").strip()
                if o and o not in final:
                    final[o] = {"위치": it.get("위치","")}

        ev = eval_final(final, gt)
        rec = ev['tp']/gt_n if gt_n else 0
        prec = ev['tp']/(ev['tp']+ev['fp']) if (ev['tp']+ev['fp']) else 0
        f1 = 2*rec*prec/(rec+prec) if (rec+prec) else 0
        pacc = ev['pc']/ev['pt'] if ev['pt'] else 0
        print(f"  📊 TP {ev['tp']}/{gt_n}, F1 {f1*100:.1f}%, 위치 {pacc*100:.1f}%, 환각 {ev['fp']}", flush=True)
        results.append({"cat":cat,"img":img,"gt_n":gt_n,
                        "obj_ans":obj_ans,"pos_ans":pos_ans,"det_ans":det_ans,
                        "coordinator":c_parsed,"final":final,"eval":ev})
        with open(f"./a_aspect_orch-{ORCH}.json","w",encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    tot = Counter()
    for r in results:
        for k in ['tp','fn','fp','pc','pt']: tot[k] += r['eval'][k]
        tot['gtn'] += r['gt_n']
    rec = tot['tp']/tot['gtn'] if tot['gtn'] else 0
    prec = tot['tp']/(tot['tp']+tot['fp']) if (tot['tp']+tot['fp']) else 0
    f1 = 2*rec*prec/(rec+prec) if (rec+prec) else 0
    pacc = tot['pc']/tot['pt'] if tot['pt'] else 0
    print(f"\n=== A(Aspect) 완료 {datetime.now().strftime('%H:%M:%S')} ===")
    print(f"  Coordinator: {ORCH}, N={len(results)}장")
    print(f"  Recall {rec*100:.1f}%, Prec {prec*100:.1f}%, F1 {f1*100:.1f}%, "
          f"9-Pos {pacc*100:.1f}%, 환각 {tot['fp']} ({tot['fp']/len(results):.2f}/img)")


if __name__ == "__main__":
    main()
