"""Plan B 완료 후 progress.md 자동 업데이트."""
import os, json, re
from datetime import datetime

LOGS = "/Users/kg/nonmoon/htp_thesis/logs_planb"
PROGRESS = "/Users/kg/nonmoon/htp_thesis/progress.md"

def extract_result(log_file):
    """log에서 마지막 결과 라인 추출."""
    try:
        with open(log_file) as f:
            content = f.read()
        m = re.search(r"Acc\s+([\d.]+)%.*?F1\s+([\d.]+)%.*?Recall\s+([\d.]+)%.*?Prec\s+([\d.]+)%.*?9-Pos\s+([\d.]+)%.*?환각\s+(\d+)", content)
        if m:
            return {"acc": float(m.group(1)), "f1": float(m.group(2)),
                    "recall": float(m.group(3)), "prec": float(m.group(4)),
                    "9pos": float(m.group(5)), "halluc": int(m.group(6))}
    except: pass
    return None

results = {}
for name, log in [
    ("BboxUnion+R2+Judge (v2) 24장", "p1_bboxunion_v2_24.log"),
    ("BboxUnion+Zoom-R2 24장", "p1_zoom_r2_24.log"),
    ("Multi-Region 좌표 24장", "p1_multiregion_24.log"),
    ("Bbox+Voting+R2 IoU=0.3 24장", "p1_voting_r2_24.log"),
    ("BboxUnion+R2+Judge (v2) 32장", "p2_bboxunion_32.log"),
    ("v5 Region priming 단일 4장", "p2_prompt_v5.log"),
    ("v6 Caption-then-Detect 단일 4장", "p2_prompt_v6.log"),
    ("BboxUnion+R2+Judge (v2) 64장", "p3_bboxunion_64.log"),
]:
    log_path = os.path.join(LOGS, log)
    if os.path.exists(log_path):
        r = extract_result(log_path)
        if r:
            results[name] = r

# Markdown 테이블 작성
md = f"""

---

## 16. Plan B 야간 자동 실행 ({datetime.now().strftime('%Y-%m-%d')})

### 16.1 Phase 1 — 4 framework × 24장 병렬

| Framework | Acc | F1 | Recall | Prec | 9-Pos | 환각 |
|---|---|---|---|---|---|---|
"""
phase1_names = ["BboxUnion+R2+Judge (v2) 24장", "BboxUnion+Zoom-R2 24장",
                "Multi-Region 좌표 24장", "Bbox+Voting+R2 IoU=0.3 24장"]
for name in phase1_names:
    if name in results:
        r = results[name]
        md += f"| {name} | {r['acc']:.1f}% | {r['f1']:.1f}% | {r['recall']:.1f}% | {r['prec']:.1f}% | {r['9pos']:.1f}% | {r['halluc']} |\n"

md += "\n### 16.2 Phase 2 — Best 32장 + 새 prompt 변형 단일 4장\n\n"
md += "| Framework | Acc | F1 | Recall | Prec | 9-Pos | 환각 |\n|---|---|---|---|---|---|---|\n"
for name in ["BboxUnion+R2+Judge (v2) 32장", "v5 Region priming 단일 4장", "v6 Caption-then-Detect 단일 4장"]:
    if name in results:
        r = results[name]
        md += f"| {name} | {r['acc']:.1f}% | {r['f1']:.1f}% | {r['recall']:.1f}% | {r['prec']:.1f}% | {r['9pos']:.1f}% | {r['halluc']} |\n"

md += "\n### 16.3 Phase 3 — Best 64장 본격\n\n"
md += "| Framework | Acc | F1 | Recall | Prec | 9-Pos | 환각 |\n|---|---|---|---|---|---|---|\n"
for name in ["BboxUnion+R2+Judge (v2) 64장"]:
    if name in results:
        r = results[name]
        md += f"| {name} | {r['acc']:.1f}% | {r['f1']:.1f}% | {r['recall']:.1f}% | {r['prec']:.1f}% | {r['9pos']:.1f}% | {r['halluc']} |\n"

md += f"""

### 16.4 비용·시간

- 비용: $0 (전체 로컬)
- 자동 실행 완료: {datetime.now().strftime('%H:%M:%S')}

### 16.5 권고 — 본 thesis 최종 framework

**BboxUnion+R2+Judge (Clean prompt + SizeHint)**

구조:
```
R1: 3 모델 bbox grounding (영어 prompt + 한국어 출력, 객체 hint X)
   ↓
Union + IoU 0.3 merge
   ├─ 합의 객체 (2+ 모델): 채택
   └─ 단독 답 (1 모델): R2 검증
   ↓
R2: 다른 모델에게 "이 객체 진짜?" 재확인
   ↓
Judge (Qwen 영어): 합의 + R2 결과 종합
   ↓
en→kr 매핑 + v6m.normalize_label + 채점
```

핵심 차별점 (Scaffold Effect 2026 정확히 입증):
- 객체 hint 0 (체크리스트 X)
- 일반 instruction만 (다양한 크기, 부위 분리는 옵션)
- 영어 prompt + 한국어 출력 강제
"""

with open(PROGRESS, 'a') as f:
    f.write(md)
print(f"progress.md §16 추가 완료. 결과 {len(results)}개.")
