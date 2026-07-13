"""32장 raw (객체 목록 없이) + 통합 vs 분산 재검증 fair 비교."""
import os, sys, json, time
from datetime import datetime
from collections import Counter, defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'framework'))
from a_aspect import call_vlm, parse_json, load_gt, fp, BASE_IMG, TS_MAP, eval_final
from test_64_files import TEST_64
import v6_voting_debate_4 as v6m

CAT_KR = {"TL_나무":"나무","TL_집":"집","TL_남자사람":"남자 사람","TL_여자사람":"여자 사람"}
MODELS = ["qwen", "exaone", "gemma"]
JUDGE_MODEL = "qwen"
OUTFILE = "./test32_compare_r2_raw_part2.json"
CHUNK_SIZE = 16
COOLDOWN_SEC = 20 * 60

TEST_32 = TEST_64[8:16] + TEST_64[24:32] + TEST_64[40:48] + TEST_64[56:64]

def r1_prompt(cat):
    return f"""이 그림은 HTP 검사의 '{CAT_KR[cat]}' 그림입니다. 어린이/청소년이 손으로 그린 그림입니다.

그림에 그려진 모든 객체를 한국어 단어로 답하세요. bbox 좌표 (1000x1000 정규화)도 함께.
의심스럽거나 작게 그려진 객체도 포함하세요. 보수적으로 답하지 마세요.

JSON만:
```json
[{{"객체": "...", "bbox": [x1, y1, x2, y2]}}]
```"""

def r2a_integrated_prompt(cat, r1_jsons, disputed_objs):
    qwen_json = json.dumps([{"객체":o, "bbox":i.get("bbox")} for o,i in r1_jsons["qwen"].items()], ensure_ascii=False)
    exaone_json = json.dumps([{"객체":o, "bbox":i.get("bbox")} for o,i in r1_jsons["exaone"].items()], ensure_ascii=False)
    gemma_json = json.dumps([{"객체":o, "bbox":i.get("bbox")} for o,i in r1_jsons["gemma"].items()], ensure_ascii=False)
    disputed_list = "\n".join(f'- "{o}"' for o in disputed_objs)
    return f"""이 그림은 HTP 검사의 '{CAT_KR[cat]}' 그림입니다. 어린이/청소년이 손으로 그린 그림입니다.

3개 AI 모델 분석 결과:

**Qwen 답:** {qwen_json}

**EXAONE 답:** {exaone_json}

**Gemma 답:** {gemma_json}

1개 모델만 찾은 객체 (의견 갈림):
{disputed_list}

3개 모델의 답을 모두 참고하고 이미지를 다시 보면서 갈린 객체들이 진짜 있는지 종합 판단하세요.

JSON만:
```json
[{{"객체": "이름", "있음": true 또는 false, "근거": "간단히"}}]
```"""

def r2b_distributed_prompt(cat, to_recheck):
    lines = []
    for obj, info in to_recheck.items():
        ans_mods = ", ".join(info['answered_by'])
        not_mods = ", ".join(info['not_answered_by'])
        bb = info.get('bbox')
        bb_str = f"위치 {bb}" if bb else "위치 모름"
        lines.append(f'- "{obj}" — {ans_mods}는 있다고 답함 ({bb_str}). {not_mods}는 없다고 함.')
    items_text = "\n".join(lines)
    return f"""이 그림은 HTP 검사의 '{CAT_KR[cat]}' 그림입니다. 어린이/청소년이 손으로 그린 그림입니다.

3개 AI 모델이 분석했고 일부 객체에서 의견이 갈렸습니다. 이미지를 다시 보고 판단하세요.

의견이 갈린 객체:
{items_text}

이미지를 자세히 보고 위치 정보를 참고해서 그 자리를 집중적으로 보세요.

JSON만:
```json
[{{"객체": "이름", "있음": true 또는 false, "근거": "간단히"}}]
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

def parse_r1(raw):
    ans = parse_json(raw)
    items = ans if isinstance(ans, list) else []
    pred = {}
    for it in items:
        if not isinstance(it, dict): continue
        obj_raw = it.get("객체","").strip()
        bb = it.get("bbox")
        if not obj_raw: continue
        obj = v6m.normalize_label(obj_raw)
        if not obj or obj in pred: continue
        if isinstance(bb,(list,tuple)) and len(bb)>=4:
            pred[obj] = {"bbox": bb, "위치": bbox_to_9region(bb)}
        else:
            pred[obj] = {"위치": ""}
    return pred

def parse_r2(raw):
    ans = parse_json(raw)
    items = ans if isinstance(ans, list) else []
    verdicts = {}
    for it in items:
        if not isinstance(it, dict): continue
        obj_raw = it.get("객체","").strip()
        ex = it.get("있음")
        if not obj_raw: continue
        obj = v6m.normalize_label(obj_raw)
        if not obj: continue
        verdicts[obj] = bool(ex)
    return verdicts

def build_final(passed_2of3, disputed_1of3, bboxes, r2_yes):
    final = {}
    for obj in passed_2of3:
        if bboxes[obj]:
            avg = [sum(b[i] for b in bboxes[obj])/len(bboxes[obj]) for i in range(4)]
            final[obj] = {"bbox": avg, "위치": bbox_to_9region(avg)}
        else:
            final[obj] = {"위치":""}
    for obj in disputed_1of3:
        if r2_yes.get(obj, False):
            bb = bboxes[obj][0] if bboxes[obj] else None
            if bb:
                final[obj] = {"bbox": bb, "위치": bbox_to_9region(bb)}
            else:
                final[obj] = {"위치":""}
    return final

def process_image(idx, total, cat, img):
    print(f"\n[{idx}/{total}] {cat}/{img}", flush=True)
    img_path = fp(BASE_IMG, [TS_MAP[cat], img])
    gt = load_gt(cat, img)
    print(f"  GT {len(gt)}개", flush=True)

    r1_results = {}
    for m in MODELS:
        t0 = time.time()
        try:
            raw = call_vlm(m, r1_prompt(cat), img_path=img_path, temp=0.2)
            r1_results[m] = parse_r1(raw)
            print(f"  R1 {m} {time.time()-t0:.1f}s  매핑후 {len(r1_results[m])}개", flush=True)
        except Exception as e:
            r1_results[m] = {}
            print(f"  R1 {m} ERROR: {e}", flush=True)

    counts = Counter()
    bboxes = defaultdict(list)
    answered_by = defaultdict(list)
    for m, pred in r1_results.items():
        for obj, info in pred.items():
            counts[obj] += 1
            if "bbox" in info: bboxes[obj].append(info["bbox"])
            answered_by[obj].append(m)

    passed_2of3 = {o:counts[o] for o in counts if counts[o] >= 2}
    disputed_1of3 = [o for o in counts if counts[o] == 1]
    disputed_info = {o: {
        "answered_by": answered_by[o],
        "not_answered_by": [m for m in MODELS if m not in answered_by[o]],
        "bbox": bboxes[o][0] if bboxes[o] else None,
    } for o in disputed_1of3}
    print(f"  Vote: 통과 {len(passed_2of3)}, 갈림 {len(disputed_1of3)}", flush=True)

    r2a_yes = {}
    if disputed_1of3:
        t0 = time.time()
        try:
            raw = call_vlm(JUDGE_MODEL, r2a_integrated_prompt(cat, r1_results, disputed_1of3), img_path=img_path, temp=0.2)
            r2a_yes = parse_r2(raw)
            print(f"  통합 (Qwen) {time.time()-t0:.1f}s  답 {len(r2a_yes)}, yes {sum(r2a_yes.values())}", flush=True)
        except Exception as e:
            print(f"  통합 ERROR: {e}", flush=True)

    r2b_votes = defaultdict(list)
    if disputed_info:
        for model in MODELS:
            to_recheck = {o: info for o, info in disputed_info.items() if model not in info["answered_by"]}
            if not to_recheck: continue
            t0 = time.time()
            try:
                raw = call_vlm(model, r2b_distributed_prompt(cat, to_recheck), img_path=img_path, temp=0.2)
                verdicts = parse_r2(raw)
                print(f"  분산 {model} {time.time()-t0:.1f}s  재검 {len(to_recheck)}, yes {sum(verdicts.values())}", flush=True)
                for obj, ex in verdicts.items():
                    r2b_votes[obj].append(ex)
            except Exception as e:
                print(f"  분산 {model} ERROR: {e}", flush=True)
    r2b_yes = {o: (sum(votes) >= 1) for o, votes in r2b_votes.items()}

    final_A = build_final(passed_2of3, disputed_1of3, bboxes, r2a_yes)
    final_B = build_final(passed_2of3, disputed_1of3, bboxes, r2b_yes)

    e_r1 = {m: eval_final(p, gt) for m, p in r1_results.items()}
    e_vote = eval_final({o:{"위치":""} for o in passed_2of3}, gt)
    e_A = eval_final(final_A, gt)
    e_B = eval_final(final_B, gt)
    print(f"  ⇒ R1: Q {e_r1['qwen'].get('f1',0)*100:.1f} E {e_r1['exaone'].get('f1',0)*100:.1f} G {e_r1['gemma'].get('f1',0)*100:.1f}")
    print(f"  ⇒ Vote F1 {e_vote.get('f1',0)*100:.1f}  통합 F1 {e_A.get('f1',0)*100:.1f}  분산 F1 {e_B.get('f1',0)*100:.1f}", flush=True)

    return {"cat":cat, "img":img, "gt_n":len(gt),
            "r1": {m: list(r1_results[m].keys()) for m in MODELS},
            "passed_2of3": list(passed_2of3.keys()),
            "disputed_1of3": disputed_1of3,
            "r2a_integrated": r2a_yes,
            "r2b_distributed": r2b_yes,
            "eval": {"r1_qwen":e_r1["qwen"], "r1_exaone":e_r1["exaone"], "r1_gemma":e_r1["gemma"],
                     "vote_2of3":e_vote, "A_integrated":e_A, "B_distributed":e_B}}

def summarize(results):
    print(f"\n=== 종합 ({len(results)}장) ===")
    for label in ["r1_qwen","r1_exaone","r1_gemma","vote_2of3","A_integrated","B_distributed"]:
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
    print(f"=== 32장 (카테고리당 8장) raw + 통합 vs 분산 재검증 ===")
    print(f"시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n", flush=True)

    results = []
    if os.path.exists(OUTFILE):
        with open(OUTFILE) as f: results = json.load(f)
        print(f"이전 결과 {len(results)}장 로드", flush=True)

    start_idx = len(results)
    chunks = [(i, min(i+CHUNK_SIZE, 32)) for i in range(start_idx, 32, CHUNK_SIZE)]

    for chunk_i, (start, end) in enumerate(chunks):
        chunk_n = chunk_i + 1 + (start_idx // CHUNK_SIZE)
        n_chunks = (32 - start_idx + CHUNK_SIZE - 1) // CHUNK_SIZE
        print(f"\n{'='*60}")
        print(f"=== Chunk {chunk_n}/{n_chunks}: 그림 {start+1}~{end} {datetime.now().strftime('%H:%M:%S')} ===")
        print(f"{'='*60}", flush=True)

        for idx in range(start, end):
            cat, img = TEST_32[idx]
            try:
                r = process_image(idx+1, 32, cat, img)
                results.append(r)
                with open(OUTFILE, "w") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)

        summarize(results)
        if end < 32:
            print(f"\n=== Cooldown {COOLDOWN_SEC//60}분 시작 ===", flush=True)
            time.sleep(COOLDOWN_SEC)
            print(f"=== Cooldown 끝 ===", flush=True)

    summarize(results)
    print(f"\n=== 전체 32장 완료 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)

if __name__ == "__main__":
    main()
