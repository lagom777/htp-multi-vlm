"""영어 prompt 버전 — VLM 이해도 향상, 한국어 출력 강제.

GT 라벨은 한국어이므로 출력은 반드시 한국어로 강제.
정규화 사전은 그대로 사용 (한국어 → 한국어).
"""

CATEGORY_DESC_EN = {
    "TL_나무": "Tree drawing from House-Tree-Person (HTP) test by a Korean child",
    "TL_집": "House drawing from House-Tree-Person (HTP) test by a Korean child",
    "TL_남자사람": "Male person drawing from House-Tree-Person (HTP) test by a Korean child",
    "TL_여자사람": "Female person drawing from House-Tree-Person (HTP) test by a Korean child",
}

CATEGORY_PARTS_HINT = {
    "TL_나무": "tree (나무전체), trunk (기둥), branches (가지), crown/canopy (수관), leaves (나뭇잎), roots (뿌리), fruits (열매), flowers (꽃), sun (해), cloud (구름), moon (달), star (별), bird (새), squirrel (다람쥐), swing (그네)",
    "TL_집": "house (집전체), roof (지붕), wall (벽), door (문), window (창문), chimney (굴뚝), garage (차고), fence (울타리), tree (나무), path (길), sun (해), cloud (구름), garden (정원), flower (꽃)",
    "TL_남자사람": "whole person (사람전체), head (머리), face (얼굴), hair (머리카락), eyes (눈), nose (코), mouth (입), ears (귀), neck (목), body (몸), arms (팔), hands (손), legs (다리), feet (발), clothes (옷), shirt (셔츠), pants (바지), shoes (신발)",
    "TL_여자사람": "whole person (사람전체), head (머리), face (얼굴), hair (머리카락), eyes (눈), nose (코), mouth (입), ears (귀), neck (목), body (몸), arms (팔), hands (손), legs (다리), feet (발), clothes (옷), skirt (치마), dress (원피스), shoes (신발), accessories (액세서리)",
}


def make_r1_prompt_en(cat):
    """영어 prompt — 자유 분석. 출력은 한국어 강제."""
    desc = CATEGORY_DESC_EN.get(cat, "HTP drawing")
    parts_hint = CATEGORY_PARTS_HINT.get(cat, "")
    return f"""You are analyzing an HTP (House-Tree-Person) psychological projective drawing test.

Category: {desc}

Task: Identify ALL visible objects in the drawing. Separate parts as distinct objects.
Example parts (use Korean output names): {parts_hint}

⚠ Strict output rules:
1. Output ONLY a JSON array. No other text.
2. No markdown numbering.
3. No prose explanation.
4. Response MUST start with `[` and end with `]`.
5. **Object names MUST be in Korean (한국어, e.g., 나무전체, 기둥, 가지).**

For each object, output 5 fields:
```json
[
  {{
    "객체": "(object name in Korean, single noun)",
    "판정": "있음",
    "위치": "(one of 9 region labels)",
    "크기": "(size relative to image, e.g., 작음/보통/큼)",
    "근거": "(brief visual evidence)"
  }}
]
```

9 position labels:
  top-left | top-center | top-right
  middle-left | middle-center | middle-right
  bottom-left | bottom-center | bottom-right"""


def make_bbox_prompt_en(cat):
    """영어 prompt — bbox grounding. 출력 객체명은 한국어."""
    desc = CATEGORY_DESC_EN.get(cat, "HTP drawing")
    parts_hint = CATEGORY_PARTS_HINT.get(cat, "")
    return f"""Analyze this HTP {desc}.

Identify ALL visible objects with their bounding boxes.
Image coordinates are normalized to 1000x1000.
Output format: [x1, y1, x2, y2].
Separate parts (tree → trunk/branches/leaves/roots/fruits; person → head/eyes/arms/legs).

Example object names (use Korean): {parts_hint}

⚠ Output rules:
1. Object names in **Korean** (e.g., 기둥, 가지, 나뭇잎)
2. JSON array only, start with `[`, end with `]`

```json
[
  {{"객체": "(Korean noun)", "bbox": [x1, y1, x2, y2], "근거": "(brief reason)"}},
  ...
]
```"""


def make_judge_prompt_en(cat, merged_candidates):
    """영어 prompt — Judge."""
    desc = CATEGORY_DESC_EN.get(cat, "HTP drawing")
    items_str = ""
    for c in merged_candidates:
        items_str += (f"  - '{c['obj']}' bbox={[int(x) for x in c['bbox']]} "
                      f"position={c.get('pos_9', '')} models={c['models']} votes={c['n_votes']}\n"
                      f"    evidence: {c.get('evidence', '')[:200]}\n")
    return f"""You are a Bbox+Union Judge for HTP drawing analysis.

Category: {desc}

Three VLM models output bounding boxes. Same objects (IoU>0.3) were merged.
Candidate objects:
{items_str}

For each candidate, judge if it actually exists in the drawing:
- Verify with image
- Remove suspected hallucinations
- Keep detailed part separations
- Same object variants (나무전체 = 나무 = tree) should be unified
- Object names in Korean

```json
[
  {{
    "객체": "(Korean noun, single object name)",
    "최종_판정": "있음" or "없음",
    "위치": "(9-region label)",
    "근거": "(brief reason)"
  }}
]
```

9 position labels:
  top-left | top-center | top-right
  middle-left | middle-center | middle-right
  bottom-left | bottom-center | bottom-right"""
