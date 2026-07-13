"""Clean prompts — 카테고리만 명시, 객체·부위 hint 0 (체크리스트 X 원칙 준수).

본 thesis 핵심 원칙: 자유 분석. VLM이 객체를 알지 못한 상태로 식별.
"""

CATEGORY_KR = {
    "TL_나무": "나무",
    "TL_집": "집",
    "TL_남자사람": "남자사람",
    "TL_여자사람": "여자사람",
}

CATEGORY_EN = {
    "TL_나무": "tree",
    "TL_집": "house",
    "TL_남자사람": "male person",
    "TL_여자사람": "female person",
}


def make_r1_clean_kr(cat):
    """한국어 — 카테고리만, hint 0."""
    desc = CATEGORY_KR.get(cat, cat)
    return f"""이 그림은 HTP(House-Tree-Person) 심리검사의 '{desc}' 그림입니다.

그림에 보이는 모든 객체를 자유롭게 식별하세요.
미리 주어진 객체 목록은 없습니다.

JSON 배열로만:
```json
[
  {{
    "객체": "(객체명, 한국어 단일 명사)",
    "판정": "있음",
    "위치": "(9-region 라벨)",
    "크기": "(작음/보통/큼)",
    "근거": "(시각적 근거)"
  }}
]
```

위치 9개 라벨:
  top-left | top-center | top-right
  middle-left | middle-center | middle-right
  bottom-left | bottom-center | bottom-right"""


def make_r1_clean_en(cat):
    """영어 — 카테고리만, hint 0."""
    desc = CATEGORY_EN.get(cat, "drawing")
    return f"""This is a '{desc}' drawing from the HTP (House-Tree-Person) projective psychological test.

Identify all visible objects in the drawing.
No predefined object list is given.

Output object names in **Korean** (single noun).

JSON only:
```json
[
  {{
    "객체": "(Korean noun)",
    "판정": "있음",
    "위치": "(9-region label)",
    "크기": "(small/medium/large)",
    "근거": "(brief visual evidence)"
  }}
]
```

9 position labels:
  top-left | top-center | top-right
  middle-left | middle-center | middle-right
  bottom-left | bottom-center | bottom-right"""


def make_bbox_clean_kr(cat):
    """한국어 bbox — hint 0."""
    desc = CATEGORY_KR.get(cat, cat)
    return f"""이 HTP '{desc}' 그림에 보이는 모든 객체를 식별하고 bbox 좌표 출력.
이미지 크기 1000x1000 normalize, [x1, y1, x2, y2].
미리 주어진 객체 목록 없음.

JSON으로만:
```json
[
  {{"객체": "객체명", "bbox": [x1, y1, x2, y2], "근거": "..."}},
  ...
]
```"""


def make_bbox_clean_en(cat):
    """영어 bbox — hint 0."""
    desc = CATEGORY_EN.get(cat, "drawing")
    return f"""Detect all objects in this HTP '{desc}' drawing.
Bounding boxes in 1000x1000 normalized coordinates [x1, y1, x2, y2].
No predefined object list.
Object names in **Korean** single noun.

JSON only:
```json
[
  {{"객체": "(Korean noun)", "bbox": [x1, y1, x2, y2], "근거": "..."}},
  ...
]
```"""
