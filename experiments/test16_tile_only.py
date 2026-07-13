"""16장 타일 only — Qwen 1 모델 × 5 타일 (전체 + 4분할).

호출 = 5/장 (3축에서 타일만 추출)
union으로 합산 — 같은 객체 여러 타일에서 잡으면 한 번만 채택.

usage: python3 test16_tile_only.py [raw|closed]
"""
import os, sys, json, time, tempfile
from datetime import datetime
from collections import Counter, defaultdict
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'framework'))
from a_aspect import call_vlm, parse_json, load_gt, fp, BASE_IMG, TS_MAP, eval_final
from test_64_files import TEST_64
from plans_v2 import PLANS_V2 as PLANS
import v6_voting_debate_4 as v6m

MODE = sys.argv[1] if len(sys.argv) > 1 else "closed"  # 'raw' or 'closed'
assert MODE in ("raw", "closed")

CAT_KR = {"TL_나무":"나무","TL_집":"집","TL_남자사람":"남자 사람","TL_여자사람":"여자 사람"}
MODEL = "qwen"
OUTFILE = f"./test16_tile_only_{MODE}.json"
CHUNK_SIZE = 8
COOLDOWN_SEC = 15 * 60

TEST_16 = TEST_64[0:4] + TEST_64[16:20] + TEST_64[32:36] + TEST_64[48:52]

def closed_prompt(cat):
    return f"""이 그림은 HTP 검사의 '{CAT_KR[cat]}' 그림의 일부 (또는 전체)입니다. 어린이/청소년이 손으로 그린 그림입니다.

아래 50개 객체 목록 중 보이는 객체를 답하세요. 부분/조각만 보여도 포함.

객체 목록: {PLANS[cat]}

JSON만:
```json
[{{"객체": "(50개 목록 중)", "bbox": [x1, y1, x2, y2]}}]
```"""

def raw_prompt(cat):
    return f"""이 그림은 HTP 검사의 '{CAT_KR[cat]}' 그림의 일부 (또는 전체)입니다. 어린이/청소년이 손으로 그린 그림입니다.

보이는 모든 객체를 한국어 단어로 답하세요. bbox 좌표 (1000x1000 정규화)도 함께.

JSON만:
```json
[{{"객체": "...", "bbox": [x1, y1, x2, y2]}}]
```"""

def prompt_for(cat):
    return closed_prompt(cat) if MODE == "closed" else raw_prompt(cat)

def split_tiles(img_path):
    """5 source 임시 파일: 전체 + 4 타일."""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    tiles = {
        "full": img,
        "tl": img.crop((0, 0, w//2, h//2)),
        "tr": img.crop((w//2, 0, w, h//2)),
        "bl": img.crop((0, h//2, w//2, h)),
        "br": img.crop((w//2, h//2, w, h)),
    }
    paths = {}
    for name, tile in tiles.items():
        f = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tile.save(f.name, "JPEG", quality=92)
        f.close()
        paths[name] = f.name
    return paths

def bbox_to_9region(bb):
    if not isinstance(bb,(list,tuple)) or len(bb)<4: return ""
    try:
        x1,y1,x2,y2=[float(v) for v in bb[:4]]
        cx,cy=(x1+x2)/2,(y1+y2)/2
        col="left" if cx<333 else ("right" if cx>667 else "center")
        row="top" if cy<333 else ("bottom" if cy>667 else "middle")
        return f"{row}-{col}"
    except: return ""

def parse_answer(raw):
    ans = parse_json(raw)
    items = ans if isinstance(ans, list) else []
    objs = {}
    for it in items:
        if not isinstance(it, dict): continue
        obj_raw = it.get("객체","").strip()
        bb = it.get("bbox")
        if not obj_raw: continue
        obj = v6m.normalize_label(obj_raw) if MODE == "raw" else obj_raw
        if not obj or obj in objs: continue
        objs[obj] = {"bbox": bb, "raw_name": obj_raw}
    return objs

def process_image(idx, total, cat, img):
    print(f"\n[{idx}/{total}] {cat}/{img}", flush=True)
    img_path = fp(BASE_IMG, [TS_MAP[cat], img])
    gt = load_gt(cat, img)
    print(f"  GT {len(gt)}개", flush=True)

    tile_paths = split_tiles(img_path)
    try:
        per_tile = {}
        all_src = defaultdict(list)
        for tname, tpath in tile_paths.items():
            t0 = time.time()
            try:
                raw = call_vlm(MODEL, prompt_for(cat), img_path=tpath, temp=0.2)
                objs = parse_answer(raw)
                per_tile[tname] = list(objs.keys())
                for o, info in objs.items():
                    all_src[o].append((tname, info.get("bbox")))
                print(f"  {tname:<5} {time.time()-t0:.1f}s  {len(objs)}개", flush=True)
            except Exception as e:
                per_tile[tname] = []
                print(f"  {tname:<5} ERROR: {e}", flush=True)

        # union (src ≥ 1)
        union_pred = {}
        for obj, sources in all_src.items():
            bboxes = [bb for _, bb in sources if isinstance(bb,(list,tuple)) and len(bb)>=4]
            if bboxes:
                avg = [sum(b[i] for b in bboxes)/len(bboxes) for i in range(4)]
                union_pred[obj] = {"bbox": avg, "위치": bbox_to_9region(avg)}
            else:
                union_pred[obj] = {"위치":""}

        # src ≥ 2 (2+ 타일에서 잡힘)
        src2_pred = {o: union_pred[o] for o in all_src if len(all_src[o]) >= 2}

        # full-only (baseline)
        full_objs = set(per_tile.get("full", []))
        full_pred = {o: union_pred[o] for o in full_objs}

        e_full = eval_final(full_pred, gt)
        e_union = eval_final(union_pred, gt)
        e_src2 = eval_final(src2_pred, gt)
        print(f"  ⇒ full F1 {e_full.get('f1',0)*100:.1f}  union F1 {e_union.get('f1',0)*100:.1f}  src≥2 F1 {e_src2.get('f1',0)*100:.1f}", flush=True)

        return {"cat":cat, "img":img, "gt_n":len(gt),
                "per_tile": per_tile,
                "src_count": {o: len(v) for o, v in all_src.items()},
                "eval": {"full_only":e_full, "union":e_union, "src_ge_2":e_src2}}
    finally:
        for p in tile_paths.values():
            try: os.unlink(p)
            except: pass

def summarize(results):
    print(f"\n=== 종합 ({MODE.upper()}, {len(results)}장) ===")
    for label in ["full_only","union","src_ge_2"]:
        tot = Counter()
        for r in results:
            for k in ['tp','fn','fp','pc','pt']: tot[k] += r["eval"][label][k]
            tot['gtn'] += r["gt_n"]
        rec = tot['tp']/tot['gtn'] if tot['gtn'] else 0
        prec = tot['tp']/(tot['tp']+tot['fp']) if (tot['tp']+tot['fp']) else 0
        f1 = 2*rec*prec/(rec+prec) if (rec+prec) else 0
        pacc = tot['pc']/tot['pt'] if tot['pt'] else 0
        print(f"  {label:<15} F1 {f1*100:5.1f}%  R {rec*100:5.1f}%  P {prec*100:5.1f}%  9-Pos {pacc*100:5.1f}%  환각 {tot['fp']:3d}")

def main():
    print(f"=== 16장 타일 only ({MODE.upper()}) — Qwen × 5 타일 ===")
    print(f"시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n", flush=True)

    results = []
    if os.path.exists(OUTFILE):
        with open(OUTFILE) as f: results = json.load(f)
        print(f"이전 결과 {len(results)}장 로드", flush=True)

    start_idx = len(results)
    chunks = [(i, min(i+CHUNK_SIZE, 16)) for i in range(start_idx, 16, CHUNK_SIZE)]

    for chunk_i, (start, end) in enumerate(chunks):
        chunk_n = chunk_i + 1 + (start_idx // CHUNK_SIZE)
        n_chunks = (16 - start_idx + CHUNK_SIZE - 1) // CHUNK_SIZE
        print(f"\n=== Chunk {chunk_n}/{n_chunks}: 그림 {start+1}~{end} {datetime.now().strftime('%H:%M:%S')} ===", flush=True)

        for idx in range(start, end):
            cat, img = TEST_16[idx]
            try:
                r = process_image(idx+1, 16, cat, img)
                results.append(r)
                with open(OUTFILE, "w") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)

        summarize(results)
        if end < 16:
            print(f"\n=== Cooldown {COOLDOWN_SEC//60}분 ===", flush=True)
            time.sleep(COOLDOWN_SEC)

    print(f"\n=== 전체 완료 {datetime.now().strftime('%H:%M:%S')} ===", flush=True)

if __name__ == "__main__":
    main()
