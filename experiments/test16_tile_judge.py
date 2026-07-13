"""16장 타일 + Judge — Qwen × 5 타일 + Qwen Judge 재검증.

src ≥ 2 통과 + src = 1 갈림 → Qwen Judge로 재검증 → 살릴까/버릴까.

usage: python3 test16_tile_judge.py [raw|closed]
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

MODE = sys.argv[1] if len(sys.argv) > 1 else "closed"
assert MODE in ("raw", "closed")

CAT_KR = {"TL_나무":"나무","TL_집":"집","TL_남자사람":"남자 사람","TL_여자사람":"여자 사람"}
MODEL = "qwen"
OUTFILE = f"./test16_tile_judge_{MODE}.json"
CHUNK_SIZE = 8
COOLDOWN_SEC = 15 * 60

TEST_16 = TEST_64[0:4] + TEST_64[16:20] + TEST_64[32:36] + TEST_64[48:52]

def closed_prompt(cat):
    return f"""이 그림은 HTP 검사의 '{CAT_KR[cat]}' 그림의 일부 (또는 전체)입니다. 어린이/청소년이 손으로 그린 그림입니다.

아래 50개 객체 목록 중 보이는 객체를 답하세요. 부분만 보여도 포함.

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

def judge_prompt(cat, tile_results, disputed):
    """5 타일 답 전부 + 갈린 객체 → Judge."""
    tile_lines = []
    for tname in ["full","tl","tr","bl","br"]:
        objs = tile_results.get(tname, [])
        tile_lines.append(f"  {tname}: {objs}")
    tiles_text = "\n".join(tile_lines)
    disputed_list = "\n".join(f'- "{o}"' for o in disputed)
    return f"""이 그림은 HTP 검사의 '{CAT_KR[cat]}' 그림입니다. 어린이/청소년이 손으로 그린 그림입니다.

같은 그림을 5가지 시각으로 분석한 결과:
{tiles_text}

(full=전체, tl=왼쪽위, tr=오른쪽위, bl=왼쪽아래, br=오른쪽아래)

1개 시각에서만 잡힌 객체 (의견 갈림):
{disputed_list}

이미지를 다시 보면서 갈린 객체들이 진짜 있는지 판단하세요.

JSON만:
```json
[{{"객체": "이름", "있음": true 또는 false, "근거": "간단히"}}]
```"""

def prompt_for(cat):
    return closed_prompt(cat) if MODE == "closed" else raw_prompt(cat)

def split_tiles(img_path):
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
        objs[obj] = bb
    return objs

def parse_verdict(raw):
    ans = parse_json(raw)
    items = ans if isinstance(ans, list) else []
    v = {}
    for it in items:
        if not isinstance(it, dict): continue
        obj = v6m.normalize_label(it.get("객체","").strip()) if MODE == "raw" else it.get("객체","").strip()
        ex = it.get("있음")
        if obj:
            v[obj] = bool(ex)
    return v

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
                for o, bb in objs.items():
                    all_src[o].append((tname, bb))
                print(f"  {tname:<5} {time.time()-t0:.1f}s  {len(objs)}개", flush=True)
            except Exception as e:
                per_tile[tname] = []
                print(f"  {tname:<5} ERROR: {e}", flush=True)

        passed = {o for o, srcs in all_src.items() if len(srcs) >= 2}
        disputed = [o for o, srcs in all_src.items() if len(srcs) == 1]
        print(f"  Tile vote: 통과 {len(passed)} (src≥2), 갈림 {len(disputed)} (src=1)", flush=True)

        # Qwen Judge
        verdicts = {}
        if disputed:
            t0 = time.time()
            try:
                raw = call_vlm(MODEL, judge_prompt(cat, per_tile, disputed), img_path=img_path, temp=0.2)
                verdicts = parse_verdict(raw)
                print(f"  Judge {time.time()-t0:.1f}s  답 {len(verdicts)}, yes {sum(verdicts.values())}", flush=True)
            except Exception as e:
                print(f"  Judge ERROR: {e}", flush=True)

        def build(objs_set):
            out = {}
            for o in objs_set:
                bboxes = [bb for _, bb in all_src[o] if isinstance(bb,(list,tuple)) and len(bb)>=4]
                if bboxes:
                    avg = [sum(b[i] for b in bboxes)/len(bboxes) for i in range(4)]
                    out[o] = {"bbox": avg, "위치": bbox_to_9region(avg)}
                else:
                    out[o] = {"위치":""}
            return out

        union = build(set(all_src.keys()))
        src2 = build(passed)
        judge_final = build(passed | {o for o in disputed if verdicts.get(o, False)})

        e_full = eval_final(build(set(per_tile.get("full", []))), gt)
        e_union = eval_final(union, gt)
        e_src2 = eval_final(src2, gt)
        e_judge = eval_final(judge_final, gt)
        print(f"  ⇒ full F1 {e_full.get('f1',0)*100:.1f}  union F1 {e_union.get('f1',0)*100:.1f}  src≥2 F1 {e_src2.get('f1',0)*100:.1f}  src≥2+Judge F1 {e_judge.get('f1',0)*100:.1f}", flush=True)

        return {"cat":cat, "img":img, "gt_n":len(gt),
                "per_tile": per_tile,
                "src_count": {o: len(v) for o, v in all_src.items()},
                "disputed": disputed,
                "verdicts": verdicts,
                "eval": {"full_only":e_full, "union":e_union, "src_ge_2":e_src2, "src2_plus_judge":e_judge}}
    finally:
        for p in tile_paths.values():
            try: os.unlink(p)
            except: pass

def summarize(results):
    print(f"\n=== 종합 ({MODE.upper()}, {len(results)}장) ===")
    for label in ["full_only","union","src_ge_2","src2_plus_judge"]:
        tot = Counter()
        for r in results:
            for k in ['tp','fn','fp','pc','pt']: tot[k] += r["eval"][label][k]
            tot['gtn'] += r["gt_n"]
        rec = tot['tp']/tot['gtn'] if tot['gtn'] else 0
        prec = tot['tp']/(tot['tp']+tot['fp']) if (tot['tp']+tot['fp']) else 0
        f1 = 2*rec*prec/(rec+prec) if (rec+prec) else 0
        pacc = tot['pc']/tot['pt'] if tot['pt'] else 0
        print(f"  {label:<20} F1 {f1*100:5.1f}%  R {rec*100:5.1f}%  P {prec*100:5.1f}%  9-Pos {pacc*100:5.1f}%  환각 {tot['fp']:3d}")

def main():
    print(f"=== 16장 타일 + Judge ({MODE.upper()}) — Qwen × 5 타일 + Qwen Judge ===")
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
