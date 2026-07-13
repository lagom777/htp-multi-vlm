"""영어 prompt v2 — 영어 자연 출력 (한국어 강제 X).

EN_TO_KR 매핑으로 채점 단계에서 한국어 GT 변환.
"""

CATEGORY_DESC_EN = {
    "TL_나무": "Tree drawing from House-Tree-Person (HTP) test by a Korean child",
    "TL_집": "House drawing from House-Tree-Person (HTP) test by a Korean child",
    "TL_남자사람": "Male person drawing from House-Tree-Person (HTP) test by a Korean child",
    "TL_여자사람": "Female person drawing from House-Tree-Person (HTP) test by a Korean child",
}

CATEGORY_PARTS_HINT_EN = {
    "TL_나무": "tree, trunk, branches, crown, leaves, roots, fruits, flowers, sun, cloud, moon, star, bird, squirrel, swing",
    "TL_집": "house, roof, wall, door, window, chimney, garage, fence, tree, path, sun, cloud, garden, flower, grass",
    "TL_남자사람": "person, head, face, hair, eyes, eyebrows, nose, mouth, ears, neck, body, arms, hands, fingers, legs, feet, shirt, pants, shoes, hat, glasses",
    "TL_여자사람": "person, head, face, hair, eyes, eyebrows, nose, mouth, ears, neck, body, arms, hands, fingers, legs, feet, skirt, dress, shoes, earrings, accessories",
}


def make_r1_prompt_full_en(cat):
    """영어 자유 분석 — 영어 출력 (한국어 강제 X)."""
    desc = CATEGORY_DESC_EN.get(cat, "HTP drawing")
    parts = CATEGORY_PARTS_HINT_EN.get(cat, "")
    return f"""Analyze this HTP (House-Tree-Person) psychological projective drawing.

Category: {desc}

Identify ALL visible objects. Separate parts as distinct objects.
Common objects: {parts}

⚠ Recall priority — identify ALL objects (include uncertain ones).
Object names in **English** (single noun, lowercase).

JSON only:
```json
[
  {{
    "object": "(English single noun, lowercase)",
    "judgment": "present",
    "position": "(9-region label)",
    "size": "(small/medium/large)",
    "evidence": "(brief visual evidence)"
  }}
]
```

9 position labels:
  top-left | top-center | top-right
  middle-left | middle-center | middle-right
  bottom-left | bottom-center | bottom-right"""


def make_bbox_prompt_full_en(cat):
    """영어 bbox grounding — 영어 출력."""
    desc = CATEGORY_DESC_EN.get(cat, "HTP drawing")
    parts = CATEGORY_PARTS_HINT_EN.get(cat, "")
    return f"""Detect ALL objects in this HTP {desc}.

Output bounding boxes (1000x1000 normalized) [x1, y1, x2, y2].
Separate parts: tree → trunk/branches/leaves/roots/fruits; person → head/eyes/arms/legs/etc.
Common objects: {parts}

⚠ Recall priority — detect ALL objects.
Object names in **English** (single noun, lowercase).

JSON only:
```json
[
  {{"object": "english_noun", "bbox": [x1, y1, x2, y2], "evidence": "brief reason"}},
  ...
]
```"""


def make_judge_prompt_full_en(cat, merged):
    desc = CATEGORY_DESC_EN.get(cat, "HTP drawing")
    items_str = ""
    for c in merged:
        items_str += (f"  - '{c['obj']}' bbox={[int(x) for x in c['bbox']]} "
                      f"position={c.get('pos_9', '')} models={c['models']}\n")
    return f"""You are a Judge for {desc}.

VLMs proposed these objects (with bboxes):
{items_str}

For each (verify with image):
- Confirm "present" if visible, "absent" if hallucination
- Same object variants (tree/whole tree) unified
- Object names in **English** (lowercase, single noun)

```json
[
  {{
    "object": "english_noun",
    "final_judgment": "present" or "absent",
    "position": "(9-region)",
    "evidence": "..."
  }}
]
```

Position: top-left|top-center|top-right|middle-left|middle-center|middle-right|bottom-left|bottom-center|bottom-right"""


def make_union_judge_prompt_full_en(cat, union_items):
    desc = CATEGORY_DESC_EN.get(cat, "HTP drawing")
    items_str = ""
    for obj, evid in union_items.items():
        items_str += f"  - '{obj}' (position: {evid['pos']}, models: {evid['models']})\n"
    return f"""You are a Union Judge for {desc}.

VLMs identified these objects:
{items_str}

For each (verify with image):
- Confirm "present" if visible, "absent" if hallucination
- Same object variants unified
- Object names in **English** (lowercase, single noun)

```json
[
  {{
    "object": "english_noun",
    "final_judgment": "present" or "absent",
    "position": "(9-region)",
    "evidence": "..."
  }}
]
```

Position: top-left|top-center|top-right|middle-left|middle-center|middle-right|bottom-left|bottom-center|bottom-right"""
