"""64장 — V5 50개 체크리스트(정답+해석항목+있을법한 교란) 측정.

3모델 신규 R1(bbox) → 투표 → Qwen 재검증.
집계: 모델별 단독(F1·환각·9pos) + 투표 + 투표+재검증.
3등급 채점: 정답=정량 / 해석항목=참고(F1 안 깎음) / 교란=환각(FP).
9-pos: 단독은 자기 bbox, 최종은 단독우선(qwen→exaone→gemma).
chunk 16×4 + cooldown 15분(발열). resume 지원. bbox·raw 전부 저장.
"""
import os, sys, json, time
from datetime import datetime
from collections import Counter, defaultdict
from project_paths import RESULTS_DIR, ensure_output_dir
from a_aspect import call_vlm, parse_json, load_gt, fp, BASE_IMG, TS_MAP
from test_64_files import TEST_64
import v6_voting_debate_4 as v6m

# plans_v5_data.json(50개) 로드
with (RESULTS_DIR / "plans_v5_data.json").open(encoding="utf-8") as f:
    V5 = json.load(f)
PLANS = {c: ", ".join(d["gt"] + d["interp"] + d["distractor"]) for c, d in V5.items()}
META  = {c: {"gt": d["gt"], "interp": d["interp"], "distractor": d["distractor"]} for c, d in V5.items()}

MODELS = ["qwen", "exaone", "gemma"]
JUDGE = "qwen"
POS_PRIORITY = ["qwen", "exaone", "gemma"]
OUTFILE = str(ensure_output_dir() / "test64_v5.json")
CAT_KR = {"TL_나무":"나무","TL_집":"집","TL_남자사람":"남자 사람","TL_여자사람":"여자 사람"}
CHUNK = 16
COOLDOWN = 0   # 연속 측정(휴식 없음)

def r1_prompt(cat):
    return f"""이 그림은 HTP 검사의 '{CAT_KR[cat]}' 그림입니다. 어린이/청소년이 손으로 그린 그림입니다.

아래 50개 객체 목록 중 그림에 그려진 것을 bbox와 함께 답하세요. 의심스러운 것도 포함.

객체 목록: {PLANS[cat]}

좌표: 1000x1000 정규화 [x1, y1, x2, y2].

JSON만:
```json
[{{"객체": "(50개 목록 중)", "bbox": [x1, y1, x2, y2]}}]
```"""

def judge_prompt(cat, r1, disputed):
    js = {m: json.dumps([{"객체":o,"bbox":i.get("bbox")} for o,i in r1[m].items()], ensure_ascii=False) for m in MODELS}
    dl = "\n".join(f'- "{o}"' for o in disputed)
    return f"""HTP '{CAT_KR[cat]}' 그림. 어린이/청소년 손그림.

**Qwen:** {js['qwen']}

**EXAONE:** {js['exaone']}

**Gemma:** {js['gemma']}

1개 모델만 찾은 객체:
{dl}

3개 답을 참고하고 이미지를 다시 보며 갈린 객체가 진짜 있는지 판단.
JSON만:
```json
[{{"객체":"이름","있음":true 또는 false}}]
```"""

def bb9(bb):
    if not isinstance(bb,(list,tuple)) or len(bb)<4: return ""
    try:
        x1,y1,x2,y2=[float(v) for v in bb[:4]]
        cx,cy=(x1+x2)/2,(y1+y2)/2
        col="left" if cx<333 else ("right" if cx>667 else "center")
        row="top" if cy<333 else ("bottom" if cy>667 else "middle")
        return f"{row}-{col}"
    except: return ""

def parse_r1(raw):
    ans=parse_json(raw); items=ans if isinstance(ans,list) else []
    pred={}
    for it in items:
        if not isinstance(it,dict): continue
        o=it.get("객체","").strip(); bb=it.get("bbox")
        if not o: continue
        ok=isinstance(bb,(list,tuple)) and len(bb)>=4
        pred[o]={"bbox":list(bb) if ok else None,"위치":bb9(bb) if ok else ""}
    return pred

def parse_r2(raw):
    ans=parse_json(raw); items=ans if isinstance(ans,list) else []
    v={}
    for it in items:
        if not isinstance(it,dict): continue
        o=it.get("객체","").strip(); ex=it.get("있음")
        if o: v[o]=bool(ex)
    return v

def ev(objset, gt, cat):
    """3등급: 정답=정량(TP/FN), 교란=환각(FP), 해석항목=참고(F1 안 깎음)."""
    interp_norm = {v6m.normalize_label(x) for x in META[cat]["interp"]}
    norm={}
    for o in objset:
        n=v6m.normalize_label(o)
        if n and n not in norm: norm[n]=o
    tp=sum(1 for g in gt if g in norm)
    fn=len(gt)-tp
    fpset=[]; interp_hit=[]
    for n,orig in norm.items():
        if n in gt: continue
        if n in interp_norm: interp_hit.append(orig)
        else: fpset.append(orig)
    fp=len(fpset)
    rec=tp/(tp+fn) if tp+fn else 0
    prec=tp/(tp+fp) if tp+fp else 0
    f1=2*rec*prec/(rec+prec) if rec+prec else 0
    return dict(tp=tp,fn=fn,fp=fp,f1=f1,rec=rec,prec=prec,fp_objs=fpset,interp_hit=interp_hit)

def ev_pos(pos_map, gt):
    """9-pos: 식별 TP 중 위치 일치 비율 (pc/pt)."""
    pc=pt=0
    for gl,gpos in gt.items():
        if gl in pos_map:
            pt+=1
            if pos_map[gl] and pos_map[gl] in gpos: pc+=1
    return dict(pc=pc,pt=pt)

def process(idx,cat,img):
    print(f"\n[{idx}/64] {cat}/{img}",flush=True)
    img_path=fp(BASE_IMG,[TS_MAP[cat],img]); gt=load_gt(cat,img)
    r1={}
    for m in MODELS:
        try:
            raw=call_vlm(m,r1_prompt(cat),img_path=img_path,temp=0.2)
            r1[m]=parse_r1(raw)
        except Exception as e:
            r1[m]={}; print(f"  R1 {m} ERR {e}",flush=True)
        nb=sum(1 for v in r1[m].values() if v["bbox"])
        print(f"  R1 {m}: {len(r1[m])}개, bbox {nb}",flush=True)

    counts=Counter(); by_model=defaultdict(dict)
    for m in MODELS:
        for o,info in r1[m].items():
            counts[o]+=1; by_model[o][m]=info
    passed=[o for o in counts if counts[o]>=2]
    disputed=[o for o in counts if counts[o]==1]
    r2={}
    if disputed:
        try:
            raw=call_vlm(JUDGE,judge_prompt(cat,r1,disputed),img_path=img_path,temp=0.2)
            r2=parse_r2(raw)
        except Exception as e: print(f"  judge ERR {e}",flush=True)
    accepted=[o for o in disputed if r2.get(o,False)]
    final=passed+accepted

    # 식별 채점
    e_q=ev(list(r1["qwen"].keys()),gt,cat)
    e_ex=ev(list(r1["exaone"].keys()),gt,cat)
    e_ge=ev(list(r1["gemma"].keys()),gt,cat)
    e_vote=ev(passed,gt,cat)
    e_final=ev(final,gt,cat)

    # 9-pos: 단독은 자기 bbox 위치, 최종은 단독우선
    def solo_pos(m):
        pm={}
        for o,info in r1[m].items():
            n=v6m.normalize_label(o)
            if n and n not in pm: pm[n]=info.get("위치","")
        return pm
    def final_pos():
        pm={}
        for o in final:
            n=v6m.normalize_label(o)
            if not n or n in pm: continue
            for pr in POS_PRIORITY:
                if pr in by_model[o] and by_model[o][pr]["bbox"]:
                    pm[n]=by_model[o][pr]["위치"]; break
            pm.setdefault(n,"")
        return pm
    p_q =ev_pos(solo_pos("qwen"),gt)
    p_ex=ev_pos(solo_pos("exaone"),gt)
    p_ge=ev_pos(solo_pos("gemma"),gt)
    p_fin=ev_pos(final_pos(),gt)

    print(f"  단독F1 q{e_q['f1']*100:.0f} e{e_ex['f1']*100:.0f} g{e_ge['f1']*100:.0f} / 투표{e_vote['f1']*100:.0f} 최종{e_final['f1']*100:.0f}  | 환각q{e_q['fp']} e{e_ex['fp']} g{e_ge['fp']}→최종{e_final['fp']}",flush=True)
    return {"cat":cat,"img":img,"gt_n":len(gt),"gt":dict(gt),
            "r1":{m:dict(r1[m]) for m in MODELS},
            "passed":passed,"disputed":disputed,"accepted":accepted,"r2":r2,"final":final,
            "eval":{"qwen":e_q,"exaone":e_ex,"gemma":e_ge,"vote":e_vote,"final":e_final},
            "pos":{"qwen":p_q,"exaone":p_ex,"gemma":p_ge,"final":p_fin}}

def summarize(results):
    n=len(results)
    print(f"\n=== 종합 ({n}장, V5 50개) ===")
    print(f"  {'방식':14s} {'F1':>6} {'R':>6} {'P':>6} {'환각/img':>9} {'9-pos':>7}")
    rows=[("qwen","Qwen단독"),("exaone","EXAONE단독"),("gemma","Gemma단독"),
          ("vote","투표만"),("final","투표+재검증")]
    for k,nm in rows:
        t=Counter()
        for r in results:
            for kk in ['tp','fn','fp']: t[kk]+=r["eval"][k][kk]
        rec=t['tp']/(t['tp']+t['fn']) if t['tp']+t['fn'] else 0
        prec=t['tp']/(t['tp']+t['fp']) if t['tp']+t['fp'] else 0
        f1=2*rec*prec/(rec+prec)*100 if rec+prec else 0
        pk = k if k in ("qwen","exaone","gemma","final") else None
        if pk:
            pc=sum(r["pos"][pk]["pc"] for r in results); pt=sum(r["pos"][pk]["pt"] for r in results)
            p9=f"{pc/pt*100:.1f}%" if pt else "-"
        else: p9="-"
        print(f"  {nm:14s} {f1:6.1f} {rec*100:6.1f} {prec*100:6.1f} {t['fp']/n:9.2f} {p9:>7}")
    # 환각으로 자주 잡힌 교란
    for k,nm in [("qwen","Qwen단독"),("final","최종")]:
        fpc=Counter()
        for r in results:
            for o in r["eval"][k]["fp_objs"]: fpc[v6m.normalize_label(o)]+=1
        print(f"  [{nm}] 환각 교란 빈도순: {dict(fpc.most_common(12))}")

def main():
    print(f"=== 64장 V5 50개 {datetime.now():%H:%M} ===",flush=True)
    results=[]
    if os.path.exists(OUTFILE):
        results=json.load(open(OUTFILE)); print(f"이전 {len(results)}장 로드",flush=True)
    start=len(results)
    chunks=[(i,min(i+CHUNK,64)) for i in range(start,64,CHUNK)]
    for (s,e) in chunks:
        print(f"\n{'='*50}\n=== Chunk {s+1}~{e} {datetime.now():%H:%M} ===\n{'='*50}",flush=True)
        for i in range(s,e):
            cat,img=TEST_64[i]
            try:
                results.append(process(i+1,cat,img))
                json.dump(results,open(OUTFILE,"w"),ensure_ascii=False,indent=2)
            except Exception as ex:
                print(f"  ERR {ex}",flush=True)
        summarize(results)
        if e<64:
            print(f"\n=== Cooldown {COOLDOWN//60}분 {datetime.now():%H:%M} ===",flush=True)
            time.sleep(COOLDOWN)
    print(f"\n=== 완료 {datetime.now():%H:%M} ===")
    summarize(results)

if __name__=="__main__": main()
