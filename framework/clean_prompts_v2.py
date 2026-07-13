"""Clean prompts v2 — 영어 + JSON only + 옵션 (단순 / 크기 명시).

원칙:
- 카테고리만 명시 (집·나무·사람)
- 객체 목록 X (컨닝페이퍼 X)
- 영어 prompt + 한국어 출력
- JSON only
"""

CATEGORY_EN = {
    "TL_나무": "tree",
    "TL_집": "house",
    "TL_남자사람": "male person",
    "TL_여자사람": "female person",
}


def make_r1_simple_en(cat):
    """Simple — 옵션 1, 단순."""
    desc = CATEGORY_EN.get(cat, "drawing")
    return f"""This is a '{desc}' drawing from the HTP (House-Tree-Person) psychological projective test.

Identify all visible objects.
No predefined object list.

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

9 positions: top-left | top-center | top-right | middle-left | middle-center | middle-right | bottom-left | bottom-center | bottom-right"""


def make_r1_sizehint_en(cat):
    """크기 hint 옵션 2 — Recall 향상 시도, 객체명 hint X."""
    desc = CATEGORY_EN.get(cat, "drawing")
    return f"""This is a '{desc}' drawing from the HTP (House-Tree-Person) psychological projective test.

Identify all visible objects in the drawing.
- Include objects of **various sizes** (large prominent and smaller details)
- Include overlapping objects as separate
- Only report what you actually see (no guessing)

No predefined object list.
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

9 positions: top-left | top-center | top-right | middle-left | middle-center | middle-right | bottom-left | bottom-center | bottom-right"""


def make_bbox_simple_en(cat):
    """Bbox simple."""
    desc = CATEGORY_EN.get(cat, "drawing")
    return f"""This is a '{desc}' drawing from the HTP psychological projective test.

Detect all visible objects with bounding boxes.
Coordinates in 1000x1000 normalized space [x1, y1, x2, y2].

No predefined object list.
Object names in **Korean** single noun.

JSON only:
```json
[
  {{"객체": "(Korean noun)", "bbox": [x1, y1, x2, y2], "근거": "..."}}
]
```"""


def make_bbox_sizehint_en(cat):
    """Bbox + 크기 hint."""
    desc = CATEGORY_EN.get(cat, "drawing")
    return f"""This is a '{desc}' drawing from the HTP psychological projective test.

Detect all visible objects with bounding boxes:
- Include objects of **various sizes** (large prominent and smaller details)
- Include overlapping objects as separate
- Bounding boxes in 1000x1000 normalized space [x1, y1, x2, y2]

No predefined object list.
Object names in **Korean** single noun.

JSON only:
```json
[
  {{"객체": "(Korean noun)", "bbox": [x1, y1, x2, y2], "근거": "..."}}
]
```"""


# 24장 — 카테고리당 6장 (균등)
TEST_IMAGES_24 = [
    # 나무 6
    ("TL_나무", "나무_8_남_01445.jpg"),
    ("TL_나무", "나무_10_여_00019.jpg"),
    ("TL_나무", "나무_11_남_00004.jpg"),
    ("TL_나무", "나무_12_여_00007.jpg"),
    ("TL_나무", "나무_8_여_00041.jpg"),
    ("TL_나무", "나무_10_남_00013.jpg"),
    # 집 6
    ("TL_집", "집_12_여_08971.jpg"),
    ("TL_집", "집_10_여_00006.jpg"),
    ("TL_집", "집_11_남_00013.jpg"),
    ("TL_집", "집_12_여_00007.jpg"),
    ("TL_집", "집_8_여_00066.jpg"),
    ("TL_집", "집_10_남_00015.jpg"),
    # 남자사람 6
    ("TL_남자사람", "남자사람_13_남_02804.jpg"),
    ("TL_남자사람", "남자사람_10_여_00023.jpg"),
    ("TL_남자사람", "남자사람_11_남_00000.jpg"),
    ("TL_남자사람", "남자사람_12_여_00005.jpg"),
    ("TL_남자사람", "남자사람_8_여_00016.jpg"),
    ("TL_남자사람", "남자사람_10_남_00022.jpg"),
    # 여자사람 6
    ("TL_여자사람", "여자사람_10_남_02125.jpg"),
    ("TL_여자사람", "여자사람_10_여_00018.jpg"),
    ("TL_여자사람", "여자사람_11_남_00002.jpg"),
    ("TL_여자사람", "여자사람_12_여_00008.jpg"),
    ("TL_여자사람", "여자사람_8_여_00081.jpg"),
    ("TL_여자사람", "여자사람_10_남_00010.jpg"),
]
