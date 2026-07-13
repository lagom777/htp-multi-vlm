"""GB10 Gemma 단독 + 50 distractor checklist + bbox + 후하게 × 64장.

본 thesis Multi-Agent 진짜 가치 검증 — 모델별 outlier 분포.
cooldown 15분 (시간 한계, Gemma는 26B로 발열 EXAONE 33B보다 적음).
총 시간: 4 × ~10분 + 3 × 15분 = ~85분.
"""
import os, sys, json, time
from datetime import datetime
from collections import Counter
from project_paths import ensure_output_dir
from a_aspect import call_vlm, parse_json, load_gt, fp, BASE_IMG, TS_MAP, eval_final
from test_64_files import TEST_64
from plans_v2 import PLANS_V2 as PLANS

CHUNK_SIZE = 16
COOLDOWN_SEC = 15 * 60  # 15분 (시간 한계)
OUTFILE = str(ensure_output_dir() / "test64_gemma_solo_chunked.json")
MODEL_KEY = "gemma"

CAT_KR = {"TL_나무":"나무","TL_집":"집","TL_남자사람":"남자 사람","TL_여자사람":"여자 사람"}

def make_prompt(cat):
    return f"""이 그림은 HTP(House-Tree-Person) 검사의 '{CAT_KR[cat]}' 그림입니다. 어린이/청소년이 손으로 그린 그림입니다.

아래 50개 객체 목록 중 그림에 실제로 그려진 것만 bbox 좌표와 함께 답하세요.
없는 객체는 답에서 제외하세요.

객체 목록: {PLANS[cat]}

의심스럽지만 그림에 있을 가능성이 있는 객체도 포함해서 답해주세요.
보수적으로 답하지 마세요.

좌표는 1000x1000 정규화 공간에서 [x1, y1, x2, y2] 형식입니다.

JSON만 출력:
```json
[
  {{"객체": "(50개 목록 중)", "bbox": [x1, y1, x2, y2]}}
]
```"""

def bbox_to_9region(bb):
    if not isinstance(bb,(list,tuple)) or len(bb)<4: return ""
    try:
        x1,y1,x2,y2=[float(v) for v in bb[:4]]
        cx,cy=(x1+x2)/2,(y1+y2)/2
        col="left" if cx<333 else ("right" if cx>667 else "center")
        row="top" if cy<333 else ("bottom" if cy>667 else "middle")
        return f"{row}-{col}"
    except: return ""

def summarize(results):
    tot = Counter()
    for r in results:
        for k in ['tp','fn','fp','pc','pt']: tot[k] += r['eval'][k]
        tot['gtn'] += r['gt_n']
    rec = tot['tp']/tot['gtn'] if tot['gtn'] else 0
    prec = tot['tp']/(tot['tp']+tot['fp']) if (tot['tp']+tot['fp']) else 0
    f1 = 2*rec*prec/(rec+prec) if (rec+prec) else 0
    acc = tot['tp']/(tot['tp']+tot['fn']+tot['fp']) if (tot['tp']+tot['fn']+tot['fp']) else 0
    pacc = tot['pc']/tot['pt'] if tot['pt'] else 0
    n = len(results)
    return f"  Gemma 단독 {n}장: F1 {f1*100:.1f}%  Acc {acc*100:.1f}%  R {rec*100:.1f}%  P {prec*100:.1f}%  9-Pos {pacc*100:.1f}%  환각 {tot['fp']} ({tot['fp']/n:.2f}/img)"

def process_image(idx, total, cat, img):
    print(f"\n[{idx}/{total}] {cat}/{img}", flush=True)
    img_path = fp(BASE_IMG, [TS_MAP[cat], img])
    gt = load_gt(cat, img)
    gt_n = len(gt)
    print(f"  GT {gt_n}개", flush=True)

    t0 = time.time()
    raw = call_vlm(MODEL_KEY, make_prompt(cat), img_path=img_path, temp=0.2)
    ans = parse_json(raw)
    items = ans if isinstance(ans, list) else []
    print(f"  gemma {time.time()-t0:.1f}s  {len(items)}개", flush=True)

    pred = {}
    for it in items:
        if not isinstance(it, dict): continue
        obj = it.get("객체","").strip()
        bb = it.get("bbox")
        if not obj: continue
        if isinstance(bb,(list,tuple)) and len(bb)>=4:
            pred[obj] = {"bbox": bb, "위치": bbox_to_9region(bb)}
        else:
            pred[obj] = {"위치": ""}

    e = eval_final(pred, gt)
    print(f"  ⇒ TP {e['tp']}/{gt_n}, FP {e['fp']}, FN {e['fn']}", flush=True)
    return {"cat":cat, "img":img, "gt_n":gt_n, "raw":items, "pred":pred, "eval":e}

def main():
    print(f"=== GB10 Gemma 단독 64장 (4 chunk × 16장, cooldown 15분) ===")
    print(f"시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n", flush=True)

    results = []
    if os.path.exists(OUTFILE):
        with open(OUTFILE) as f: results = json.load(f)
        print(f"이전 결과 {len(results)}장 로드", flush=True)

    start_idx = len(results)
    chunks = [(i, min(i+CHUNK_SIZE, 64)) for i in range(start_idx, 64, CHUNK_SIZE)]

    for chunk_i, (start, end) in enumerate(chunks):
        chunk_n = chunk_i + 1 + (start_idx // CHUNK_SIZE)
        print(f"\n{'='*60}")
        print(f"=== Chunk {chunk_n}/4: 그림 {start+1}~{end} ===")
        print(f"=== 시작 {datetime.now().strftime('%H:%M:%S')} ===")
        print(f"{'='*60}", flush=True)

        for idx in range(start, end):
            cat, img = TEST_64[idx]
            try:
                result = process_image(idx+1, 64, cat, img)
                results.append(result)
                with open(OUTFILE, "w") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)

        print(f"\n{'='*60}")
        print(f"=== Chunk {chunk_n}/4 완료 {datetime.now().strftime('%H:%M:%S')} ===")
        print(summarize(results), flush=True)

        if chunk_i < len(chunks) - 1:
            print(f"\n=== Cooldown {COOLDOWN_SEC//60}분 시작 {datetime.now().strftime('%H:%M:%S')} ===", flush=True)
            time.sleep(COOLDOWN_SEC)
            print(f"=== Cooldown 끝 {datetime.now().strftime('%H:%M:%S')} ===", flush=True)

    print(f"\n{'='*60}")
    print(f"=== 전체 64장 완료 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(summarize(results), flush=True)

if __name__ == "__main__":
    main()
