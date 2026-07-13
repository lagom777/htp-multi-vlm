"""32장 V5 — robust parser + raw 통째 저장.
0%가 빈응답이 아니라 JSON 형식오류→파싱실패였음이 확인되어, 강건 parser로 재측정.
robust_parse: 같은 객체명 1개(원 parse_r1 규칙) + 깨진 JSON도 정규식으로 (객체,bbox) 추출.
정상 응답엔 기존과 동일, 형식오류만 복구(EXAONE 14장 검증: 정상 8 동일·깨진 6 복구).
원본 test64_v5.json 보존. 3모델 동일 parser 적용 → 공정 비교.
"""
import os, sys, json, time, re
from datetime import datetime
from collections import Counter, defaultdict
sys.path.insert(0, '/Users/kg/nonmoon/htp_thesis')
from a_aspect import call_vlm, load_gt, fp, BASE_IMG, TS_MAP
from test_64_files import TEST_64
import v6_voting_debate_4 as v6m
from test64_v5 import (r1_prompt, judge_prompt, parse_r2, ev, ev_pos, bb9,
                       MODELS, JUDGE, POS_PRIORITY, summarize)

OUTFILE = "/Users/kg/nonmoon/htp_thesis/test64_v5_lowspec.json"

def robust_parse(raw):
    """3모델 형식 통일 파싱. 객체 블록 단위로 (이름,bbox) 추출 — 키이름/순서/오타/깨진괄호 무관.
    - 이름 키: 객체 | label | 객체오타(객ate 등)   - bbox 키 무관(블록 내 첫 좌표4개)
    - 같은 객체명 1개(원 parse_r1 규칙). EXAONE 단일dict·좌우bbox중복·깨진JSON, Qwen label형식,
      Gemma 키오타 전부 복구. 정상 응답엔 기존 결과와 동일."""
    raw = raw or ""
    m = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL)
    s = m.group(1) if m else raw
    pred = {}
    for blk in re.findall(r'\{[^{}]*\}', s):
        nm = re.search(r'"(?:객[^"]*|label)"\s*:\s*"([^"]+)"', blk)
        if not nm:
            continue
        name = nm.group(1).strip()
        if not name or name in pred:
            continue
        bb = re.search(r'\[\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})', blk)
        if bb:
            box = [int(bb.group(i)) for i in range(1, 5)]
            if all(0 <= v <= 1000 for v in box):
                pred[name] = {"bbox": box, "위치": bb9(box)}
                continue
        pred[name] = {"bbox": None, "위치": ""}
    return pred

def pick_32():
    """카테고리별 앞 8장 = 32장 (TEST_64는 카테고리 16장 블록)."""
    bycat = defaultdict(list)
    for i, (cat, img) in enumerate(TEST_64):
        bycat[cat].append((i, cat, img))
    sel = []
    for cat in ["TL_나무", "TL_집", "TL_남자사람", "TL_여자사람"]:
        sel += bycat[cat][:8]    # 앞 8장 = 기존 test32_v5_robust.json 순서 (재활용)
    for cat in ["TL_나무", "TL_집", "TL_남자사람", "TL_여자사람"]:
        sel += bycat[cat][8:16]  # 뒤 8장 = 추가 32장 (32→64 이어서)
    return sel

def process(idx, total, src_idx, cat, img):
    print(f"\n[{idx}/{total}] (src{src_idx}) {cat}/{img}", flush=True)
    img_path = fp(BASE_IMG, [TS_MAP[cat], img]); gt = load_gt(cat, img)
    r1 = {}; r1_raw = {}; r1_fr = {}
    for m in MODELS:
        raw, fr = call_vlm(m, r1_prompt(cat), img_path=img_path, temp=0.2, return_meta=True)
        r1_raw[m] = raw; r1_fr[m] = fr
        r1[m] = robust_parse(raw)
        nb = sum(1 for v in r1[m].values() if v["bbox"])
        print(f"  R1 {m}: {len(r1[m])}개 bbox{nb} fr={fr!r}", flush=True)

    counts = Counter(); by_model = defaultdict(dict)
    for m in MODELS:
        for o, info in r1[m].items():
            counts[o] += 1; by_model[o][m] = info
    passed = [o for o in counts if counts[o] >= 2]
    disputed = [o for o in counts if counts[o] == 1]
    r2 = {}; judge_raw = ""
    if disputed:
        judge_raw, _ = call_vlm(JUDGE, judge_prompt(cat, r1, disputed),
                                img_path=img_path, temp=0.2, return_meta=True)
        r2 = parse_r2(judge_raw)
    accepted = [o for o in disputed if r2.get(o, False)]
    final = passed + accepted

    e_q = ev(list(r1["qwen"].keys()), gt, cat); e_ex = ev(list(r1["exaone"].keys()), gt, cat)
    e_ge = ev(list(r1["gemma"].keys()), gt, cat); e_vote = ev(passed, gt, cat); e_final = ev(final, gt, cat)

    def solo_pos(m):
        pm = {}
        for o, info in r1[m].items():
            n = v6m.normalize_label(o)
            if n and n not in pm: pm[n] = info.get("위치", "")
        return pm
    def final_pos():
        pm = {}
        for o in final:
            n = v6m.normalize_label(o)
            if not n or n in pm: continue
            for pr in POS_PRIORITY:
                if pr in by_model[o] and by_model[o][pr]["bbox"]:
                    pm[n] = by_model[o][pr]["위치"]; break
            pm.setdefault(n, "")
        return pm
    p_q = ev_pos(solo_pos("qwen"), gt); p_ex = ev_pos(solo_pos("exaone"), gt)
    p_ge = ev_pos(solo_pos("gemma"), gt); p_fin = ev_pos(final_pos(), gt)

    print(f"  단독F1 q{e_q['f1']*100:.0f} e{e_ex['f1']*100:.0f} g{e_ge['f1']*100:.0f}"
          f" / 투표{e_vote['f1']*100:.0f} 최종{e_final['f1']*100:.0f}"
          f"  | 환각 q{e_q['fp']} e{e_ex['fp']} g{e_ge['fp']}→{e_final['fp']}", flush=True)
    return {"src_idx": src_idx, "cat": cat, "img": img, "gt_n": len(gt), "gt": dict(gt),
            "r1": {m: dict(r1[m]) for m in MODELS},
            "r1_raw": r1_raw, "r1_fr": r1_fr, "judge_raw": judge_raw,
            "passed": passed, "disputed": disputed, "accepted": accepted, "r2": r2, "final": final,
            "eval": {"qwen": e_q, "exaone": e_ex, "gemma": e_ge, "vote": e_vote, "final": e_final},
            "pos": {"qwen": p_q, "exaone": p_ex, "gemma": p_ge, "final": p_fin}}

def main():
    sel = pick_32(); total = len(sel)
    results = []
    if os.path.exists(OUTFILE):
        results = json.load(open(OUTFILE))
    start = len(results)
    print(f"=== 32장 robust+raw {datetime.now():%H:%M} (이전 {start}장 로드) ===", flush=True)
    for k in range(start, total):
        src_idx, cat, img = sel[k]
        try:
            results.append(process(k + 1, total, src_idx, cat, img))
            json.dump(results, open(OUTFILE, "w"), ensure_ascii=False, indent=2)
        except Exception as ex:
            print(f"  ERR {ex}", flush=True)
    print(f"\n=== 완료 {datetime.now():%H:%M} ===", flush=True)
    summarize(results)

if __name__ == "__main__":
    main()
