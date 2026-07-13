"""Phase 2 — 새 prompt v6 (Caption-then-Detect 2-pass) × 4장 단일."""
import sys, time
sys.path.insert(0, '/Users/kg/nonmoon/htp_thesis')
from openai import OpenAI
import v6_voting_debate_4 as v6m
from common_utils import parse_json_v2
from g_prime import (BASE_IMG, BASE_LBL, TS_MAP, fp, load_gt, call_vlm)
from collections import Counter
import clean_prompts_v2 as cp2

CATEGORY_EN = cp2.CATEGORY_EN


def make_caption_prompt(cat):
    desc = CATEGORY_EN.get(cat, "drawing")
    return f"""Describe this HTP '{desc}' drawing in detail.
What objects are visible? Where are they positioned?
2-3 sentences in Korean."""


def make_detect_prompt(cat, caption):
    desc = CATEGORY_EN.get(cat, "drawing")
    return f"""Based on the description below and the image, identify all distinct objects.

Description: {caption}

Now extract a structured list. Object names in Korean single noun.

JSON only:
```json
[{{"객체":"(Korean noun)","위치":"(9-region)","근거":"..."}}]
```"""


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


TEST = [
    ("TL_나무","나무_8_남_01445.jpg"),
    ("TL_집","집_12_여_08971.jpg"),
    ("TL_남자사람","남자사람_13_남_02804.jpg"),
    ("TL_여자사람","여자사람_10_남_02125.jpg"),
]
print("=== Qwen 단일 v6 (Caption-then-Detect) 4장 ===")
tot = Counter()
for cat, img in TEST:
    img_path = fp(BASE_IMG, [TS_MAP[cat], img])
    gt = load_gt(cat, img); gtn = len(gt)
    try:
        caption = call_vlm("qwen", make_caption_prompt(cat), img_path=img_path, temp=0.2)
        raw = call_vlm("qwen", make_detect_prompt(cat, caption[:500]), img_path=img_path, temp=0.2)
    except: raw = ""
    parsed = parse_json_v2(raw)
    n_obj = len(parsed) if isinstance(parsed, list) else 0
    ev = evalv(parsed, gt)
    print(f"  {cat}: {n_obj}, TP {ev['tp']}/{gtn}, 환각 {ev['fp']}")
    for k in ['tp','fn','fp','pc','pt']: tot[k] += ev[k]
    tot['gtn'] += gtn
rec = tot['tp']/tot['gtn']
prec = tot['tp']/(tot['tp']+tot['fp']) if (tot['tp']+tot['fp']) else 0
f1 = 2*rec*prec/(rec+prec) if (rec+prec) else 0
pacc = tot['pc']/tot['pt'] if tot['pt'] else 0
acc = tot['tp']/(tot['tp']+tot['fn']+tot['fp'])
print(f"=== v6 완료 Acc {acc*100:.1f}% F1 {f1*100:.1f}% Recall {rec*100:.1f}% Prec {prec*100:.1f}% 9-Pos {pacc*100:.1f}% 환각 {tot['fp']} ===")
