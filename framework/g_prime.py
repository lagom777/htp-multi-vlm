"""G' = G + 모델별 prompt + NFC normalize + 객체 많이 식별 강조.

업그레이드:
- Qwen: 영어 bbox prompt
- EXAONE/Gemma: 한국어 bbox prompt
- 모든 prompt에 "객체 많이 식별" 강조
- NFC normalize parse
- Judge: Qwen 영어 prompt
"""
import os, json, base64, time, unicodedata, re, sys
from datetime import datetime
from collections import Counter, defaultdict
from openai import OpenAI
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v6_voting_debate_4 as v6m
from common_utils import parse_json_v2, get_bbox_prompt
import english_prompts as enp

HOST = "192.168.200.138"
ENDPOINTS = {
    "qwen":   {"port":8005, "model":"RedHatAI/Qwen3.6-35B-A3B-NVFP4",   "thinking_off":True},
    "exaone": {"port":8003, "model":"LGAI-EXAONE/EXAONE-4.5-33B-AWQ",   "thinking_off":True},
    "gemma":  {"port":8004, "model":"RedHatAI/gemma-4-26B-A4B-it-NVFP4","thinking_off":False},
}

# 32장 풀 셋 (8장 × 4 카테고리, v7과 동일)
TEST_IMAGES_FULL = [
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
BASE_IMG = "./data/01.원천데이터"
BASE_LBL = "./data/02.라벨링데이터"
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


def iou(b1, b2):
    if not b1 or not b2 or len(b1) < 4 or len(b2) < 4: return 0
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    if x2 < x1 or y2 < y1: return 0
    inter = (x2-x1)*(y2-y1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter / (a1+a2-inter) if (a1+a2-inter) else 0


def bbox_to_9region(bbox, w=1000, h=1000):
    if not bbox or len(bbox) < 4: return ""
    cx = (bbox[0]+bbox[2])/2/w; cy = (bbox[1]+bbox[3])/2/h
    col = "left" if cx < 1/3 else ("right" if cx > 2/3 else "center")
    row = "top" if cy < 1/3 else ("bottom" if cy > 2/3 else "middle")
    return f"{row}-{col}"


def union_with_bbox_merge(bbox_outs):
    all_items = []
    for m, ans in bbox_outs.items():
        if not isinstance(ans, list): continue
        for it in ans:
            if not isinstance(it, dict): continue
            obj = it.get("객체","").strip()
            bb = it.get("bbox", [])
            ev = it.get("근거", "")
            if obj and isinstance(bb, list) and len(bb) >= 4:
                all_items.append({"obj":obj, "bbox":bb, "ev":ev, "model":m})
    merged = []
    used = [False] * len(all_items)
    for i, it in enumerate(all_items):
        if used[i]: continue
        cluster = [it]; used[i] = True
        norm_i = v6m.normalize_label(it['obj'])
        for j in range(i+1, len(all_items)):
            if used[j]: continue
            norm_j = v6m.normalize_label(all_items[j]['obj'])
            if norm_i == norm_j and iou(it['bbox'], all_items[j]['bbox']) >= 0.3:
                cluster.append(all_items[j]); used[j] = True
        avg_bb = [sum(c['bbox'][k] for c in cluster)/len(cluster) for k in range(4)]
        merged.append({
            "obj": cluster[0]['obj'],
            "obj_variants": list(set(c['obj'] for c in cluster)),
            "bbox": avg_bb,
            "evidence": "; ".join(set(c['ev'][:100] for c in cluster if c['ev']))[:300],
            "models": list(set(c['model'] for c in cluster)),
            "n_votes": len(cluster),
            "pos_9": bbox_to_9region(avg_bb),
        })
    return merged


def make_judge_prompt_en(cat, merged):
    return enp.make_judge_prompt_en(cat, merged)


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
    ORCH = "qwen"  # Judge 모델
    N_IMAGES = int(sys.argv[1]) if len(sys.argv) > 1 else 4  # 4, 16, 32
    TEST_IMAGES = TEST_IMAGES_FULL[:N_IMAGES] if N_IMAGES != 4 else [
        ("TL_나무","나무_8_남_01445.jpg"),
        ("TL_집","집_12_여_08971.jpg"),
        ("TL_남자사람","남자사람_13_남_02804.jpg"),
        ("TL_여자사람","여자사람_10_남_02125.jpg"),
    ]
    MODELS = ["qwen", "exaone", "gemma"]
    print(f"=== G' (G + 모델별 prompt + NFC + 강조) — Judge: {ORCH}, N={len(TEST_IMAGES)} ===")
    print(f"시작: {datetime.now().strftime('%H:%M:%S')}")

    results = []
    for idx, (cat, img) in enumerate(TEST_IMAGES):
        print(f"\n[{idx+1}/{len(TEST_IMAGES)}] {cat}/{img}", flush=True)
        img_path = fp(BASE_IMG, [TS_MAP[cat], img])
        gt = load_gt(cat, img); gt_n = len(gt)

        bbox_outs = {}
        for m in MODELS:
            t0 = time.time()
            prompt = get_bbox_prompt(cat, m)
            raw = call_vlm(m, prompt, img_path=img_path, temp=0.2)
            parsed = parse_json_v2(raw)
            n_obj = len(parsed) if isinstance(parsed, list) else 0
            print(f"  Bbox {m}: {time.time()-t0:.1f}s, {n_obj} 객체", flush=True)
            bbox_outs[m] = parsed if isinstance(parsed, list) else []

        merged = union_with_bbox_merge(bbox_outs)
        print(f"  Union+IoU: {len(merged)} 후보", flush=True)

        t0 = time.time()
        j_raw = call_vlm(ORCH, make_judge_prompt_en(cat, merged), img_path=img_path, temp=0.2)
        j_parsed = parse_json_v2(j_raw)
        print(f"  Judge({ORCH}): {time.time()-t0:.1f}s, {len(j_parsed) if isinstance(j_parsed, list) else 0} 판정", flush=True)

        final = {}
        if isinstance(j_parsed, list):
            for it in j_parsed:
                if not isinstance(it, dict): continue
                if it.get("최종_판정") != "있음": continue
                o = it.get("객체","").strip()
                if o and o not in final:
                    final[o] = {"위치": it.get("위치","")}

        ev = eval_final(final, gt)
        rec = ev['tp']/gt_n if gt_n else 0
        prec = ev['tp']/(ev['tp']+ev['fp']) if (ev['tp']+ev['fp']) else 0
        f1 = 2*rec*prec/(rec+prec) if (rec+prec) else 0
        pacc = ev['pc']/ev['pt'] if ev['pt'] else 0
        acc = ev['tp']/(ev['tp']+ev['fn']+ev['fp']) if (ev['tp']+ev['fn']+ev['fp']) else 0
        print(f"  📊 TP {ev['tp']}/{gt_n}, Acc {acc*100:.1f}%, F1 {f1*100:.1f}%, 위치 {pacc*100:.1f}%, 환각 {ev['fp']}", flush=True)
        results.append({"cat":cat,"img":img,"gt_n":gt_n,
                        "bbox_outs":bbox_outs,"merged":merged,
                        "judge":j_parsed,"final":final,"eval":ev})
        with open(f"./gprime_orch-{ORCH}_{len(TEST_IMAGES)}img.json","w",encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    tot = Counter()
    for r in results:
        for k in ['tp','fn','fp','pc','pt']: tot[k] += r['eval'][k]
        tot['gtn'] += r['gt_n']
    rec = tot['tp']/tot['gtn'] if tot['gtn'] else 0
    prec = tot['tp']/(tot['tp']+tot['fp']) if (tot['tp']+tot['fp']) else 0
    f1 = 2*rec*prec/(rec+prec) if (rec+prec) else 0
    pacc = tot['pc']/tot['pt'] if tot['pt'] else 0
    acc = tot['tp']/(tot['tp']+tot['fn']+tot['fp']) if (tot['tp']+tot['fn']+tot['fp']) else 0
    print(f"\n=== G' 완료 {datetime.now().strftime('%H:%M:%S')} ===")
    print(f"  Judge: {ORCH}, N={len(results)}장")
    print(f"  Acc {acc*100:.1f}%, Recall {rec*100:.1f}%, Prec {prec*100:.1f}%, F1 {f1*100:.1f}%, "
          f"9-Pos {pacc*100:.1f}%, 환각 {tot['fp']} ({tot['fp']/len(results):.2f}/img)")


if __name__ == "__main__":
    main()
