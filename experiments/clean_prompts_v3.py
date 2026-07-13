"""Clean prompts v3 — SizeHint + PartHint (객체명 hint X, 부위 분리 일반 instruction)."""

CATEGORY_EN = {
    "TL_나무": "tree",
    "TL_집": "house",
    "TL_남자사람": "male person",
    "TL_여자사람": "female person",
}


def make_r1_v3_en(cat):
    """SizeHint + PartHint — 객체명 X, 부위 분리 일반 지시."""
    desc = CATEGORY_EN.get(cat, "drawing")
    return f"""This is a '{desc}' drawing from the HTP (House-Tree-Person) projective psychological test.

Identify all visible objects in the drawing.
- Include objects of **various sizes** (large prominent and smaller details)
- Identify **whole objects AND their visible parts as separate items**
  (e.g., if you see a complex object, identify both the whole and its visible components)
- Include overlapping objects as separate
- Only report what you actually see (no guessing)

No predefined object list — discover all objects freely.
Object names in **Korean** single noun.

JSON only:
```json
[
  {{
    "객체": "(Korean noun)",
    "위치": "(9-region label)",
    "근거": "(brief visual evidence)"
  }}
]
```

9 position labels:
  top-left | top-center | top-right
  middle-left | middle-center | middle-right
  bottom-left | bottom-center | bottom-right"""


def make_bbox_v3_en(cat):
    """SizeHint + PartHint bbox."""
    desc = CATEGORY_EN.get(cat, "drawing")
    return f"""This is a '{desc}' drawing from the HTP psychological projective test.

Detect all visible objects with bounding boxes:
- Include objects of **various sizes** (large prominent and smaller details)
- Detect **whole objects AND their visible parts as separate items**
- Include overlapping objects as separate
- Bounding boxes in 1000x1000 normalized space [x1, y1, x2, y2]

No predefined object list — discover all objects freely.
Object names in **Korean** single noun.

JSON only:
```json
[
  {{"객체": "(Korean noun)", "bbox": [x1, y1, x2, y2], "근거": "..."}}
]
```"""
