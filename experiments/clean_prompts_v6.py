"""Clean prompts v6 — English output + 매핑 (v5의 영어 변형).

v5 vs v6:
- v5: Korean output (모델이 "수관" 직접 답)
- v6: **English output** (모델이 "crown/canopy" 답 → 매핑이 한국어로 변환)

원칙 (v5와 동일):
1. 카테고리별 anatomy hint (전체 + 기본 부위)
2. 구체적 종 명명 (squirrel not animal)
3. 그 외 객체 자유 발견
4. 영어 prompt + **영어 출력**
5. en_to_kr.py 매핑으로 한국어 GT와 매칭

영어 anatomy 용어 (VLM이 잘 알 것):
- Tree: tree (whole), trunk, branches, crown/canopy, leaves, roots
- House: house (whole), walls, roof, door, windows
- Person: person (whole), head, face, eyes, nose, mouth, ears, hair, neck, body/torso, arms, hands, legs, feet
"""

CATEGORY_EN = {
    "TL_나무": "tree",
    "TL_집": "house",
    "TL_남자사람": "male person",
    "TL_여자사람": "female person",
}

ANATOMY_HINT_EN = {
    "TL_나무": (
        "If a tree is visible, decompose into anatomical parts: "
        "**tree** (whole tree outline as single object), **trunk** (vertical wooden stem), "
        "**branches** (main wooden arms extending from trunk), **crown** or **canopy** "
        "(dense leafy mass as a single object — use this for the whole foliage area), "
        "**leaves** (individual visible leaves), **roots** (if visible at the base)."
    ),
    "TL_집": (
        "If a house is visible, decompose into structural parts: "
        "**house** (whole house outline), **walls** (exterior walls), **roof** (top covering), "
        "**door** (entrance), **windows** (window panes)."
    ),
    "TL_남자사람": (
        "If a person is visible, decompose into body parts: "
        "**person** (whole figure outline), **head** (the head outline/skull shape), "
        "**face** (the whole facial area as a single region — even if details inside are simple), "
        "**eyes**, **nose**, **mouth**, **ears**, **hair**, **neck**, "
        "**body** (torso), **arms**, **hands**, **legs**, **feet**. "
        "Note: 'head' and 'face' are separate — head is the outline, face is the facial region within."
    ),
    "TL_여자사람": (
        "If a person is visible, decompose into body parts: "
        "**person** (whole figure outline), **head** (the head outline/skull shape), "
        "**face** (the whole facial area as a single region — even if details inside are simple), "
        "**eyes**, **nose**, **mouth**, **ears**, **hair**, **neck**, "
        "**body** (torso), **arms**, **hands**, **legs**, **feet**. "
        "Note: 'head' and 'face' are separate — head is the outline, face is the facial region within."
    ),
}

SPECIFIC_NAMING_EN = (
    "If you identify an animal, name the specific species "
    "(e.g., 'squirrel' not 'animal', 'bird' not 'fowl', 'butterfly' not 'insect'). "
    "If you identify plants, name them specifically "
    "(e.g., 'flower' not 'plant', 'grass' not 'vegetation')."
)


def make_r1_v6_en(cat):
    """v6 — English output + anatomy hint + specific naming."""
    desc = CATEGORY_EN.get(cat, "drawing")
    hint = ANATOMY_HINT_EN.get(cat, "")
    return f"""This is a '{desc}' drawing from the HTP (House-Tree-Person) psychological projective test.

Identify all visible objects in the drawing.

Decomposition guidance:
- {hint}
- **Beyond the anatomical parts above, also identify any other visible objects** (background elements, accessories, scene elements, animals, decorations, etc.) — discover them freely.

Specific naming:
- {SPECIFIC_NAMING_EN}

General instructions:
- Include objects of **various sizes** (large prominent and smaller details)
- Include overlapping objects as separate items
- Only report what you actually see (no guessing, no inference of hidden objects)

No exhaustive object list — discover all other objects on your own.
Object names in **English** single noun (lowercase).

JSON only:
```json
[
  {{
    "object": "(English noun, lowercase)",
    "position": "(9-region label)",
    "evidence": "(brief visual evidence)"
  }}
]
```

9 position labels:
  top-left | top-center | top-right
  middle-left | middle-center | middle-right
  bottom-left | bottom-center | bottom-right"""


def make_bbox_v6_en(cat):
    """Bbox v6 — English output."""
    desc = CATEGORY_EN.get(cat, "drawing")
    hint = ANATOMY_HINT_EN.get(cat, "")
    return f"""This is a '{desc}' drawing from the HTP psychological projective test.

Detect all visible objects with bounding boxes.

Decomposition guidance:
- {hint}
- **Beyond the anatomical parts above, also detect any other visible objects** (background, accessories, scene elements, animals, decorations) — discover them freely.

Specific naming:
- {SPECIFIC_NAMING_EN}

General instructions:
- Include objects of **various sizes** (large prominent and smaller details)
- Include overlapping objects as separate items
- Bounding boxes in 1000x1000 normalized space [x1, y1, x2, y2]
- Only report what you actually see (no guessing)

No exhaustive object list — discover all other objects on your own.
Object names in **English** single noun (lowercase).

JSON only:
```json
[
  {{"object": "(English noun, lowercase)", "bbox": [x1, y1, x2, y2], "evidence": "..."}}
]
```"""
