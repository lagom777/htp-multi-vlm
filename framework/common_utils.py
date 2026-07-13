"""공통 utils — NFC normalize + 모델별 prompt 선택 + 객체 강조."""
import json, re, unicodedata
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v6_voting_debate_4 as v6m
import english_prompts as enp


def parse_json_v2(text):
    """NFC normalize + JSON parsing (Gemma 안정화)."""
    if not text: return None
    text = unicodedata.normalize('NFC', text)
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    raw = m.group(1) if m else text.strip()
    raw = unicodedata.normalize('NFC', raw)
    try: return json.loads(raw)
    except Exception:
        m2 = re.search(r"\[\s*\{.*\}\s*\]|\{.*\}", raw, re.DOTALL)
        if m2:
            try: return json.loads(m2.group(0))
            except: pass
    return None


def get_r1_prompt(cat, model_name):
    """모델별 R1 prompt — Qwen 영어, 나머지 한국어 (객체 많이 식별 강조)."""
    if model_name == "qwen":
        return _enhance_en(enp.make_r1_prompt_en(cat))
    else:
        return _enhance_kr(v6m.make_r1_prompt(cat))


def _enhance_kr(prompt):
    """한국어 prompt에 '객체 많이 식별' 강조 추가."""
    emphasis = """

⚠ 중요 — Recall 우선:
- 보이는 모든 객체를 누락 없이 식별 (의심스러워도 포함)
- 부위까지 분리 (나무 → 기둥/가지/잎/뿌리/열매 별도)
- 사람 → 머리/얼굴/눈/코/입/팔/다리 별도
- 작은 객체도 놓치지 말 것
- 정원 객체 (잔디·돌·꽃) 포함"""
    # JSON 형식 부분 앞에 삽입
    if "JSON" in prompt:
        return prompt.replace("JSON 배열로만 출력", emphasis + "\n\nJSON 배열로만 출력")
    return prompt + emphasis


def _enhance_en(prompt):
    """영어 prompt에 '객체 많이 식별' 강조 추가."""
    emphasis = """

⚠ Important — Recall priority:
- Identify ALL visible objects (include if uncertain)
- Separate parts (tree → trunk/branches/leaves/roots/fruits)
- Person → head/face/eyes/nose/mouth/arms/legs separately
- Include small objects
- Include garden elements (grass/stones/flowers) if present"""
    if "```json" in prompt:
        # JSON 형식 직전에 삽입
        return prompt.replace("```json", emphasis + "\n\n```json", 1)
    return prompt + emphasis


def get_bbox_prompt(cat, model_name):
    """모델별 bbox grounding prompt."""
    if model_name == "qwen":
        base = enp.make_bbox_prompt_en(cat)
    else:
        # 한국어 bbox prompt (v6에는 없으므로 인라인 작성)
        desc = v6m.CATEGORY_DESC[cat]
        base = f"""이 HTP 그림 ({desc})에 보이는 모든 객체를 식별하고 bbox 출력.
이미지 크기 1000x1000 normalize, [x1,y1,x2,y2].
부위 분리 (나무→기둥/가지/잎/뿌리/열매, 사람→머리/눈/코/입/팔/다리 등).

JSON으로만:
```json
[
  {{"객체": "객체명", "bbox": [x1,y1,x2,y2], "근거": "..."}},
  ...
]
```"""
    return _enhance_en(base) if model_name == "qwen" else _enhance_kr(base)


def get_r2_prompt(cat, model_name, own_r1, others_r1, disagreed_objs):
    """모델별 R2 (debate) prompt — 현재는 v6m 사용 (영어 변형은 향후 추가 가능)."""
    return v6m.make_r2_prompt(cat, own_r1, others_r1, disagreed_objs)
