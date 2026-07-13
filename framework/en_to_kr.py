"""영어/한국어 객체명 → 한국어 GT 라벨 정규화 사전.

VLM이 영어 또는 한국어로 답해도 한국어 GT와 매칭 위해 변환.

2026-05-26 업데이트:
- 한국어 변형 (줄기/낙엽 등) 추가
- 신발/구두 통합
- foliage 정밀화 (작은 잎 → 나뭇잎, 큰 군집 → 수관)
- 동물 specific naming 가정 (다람쥐/새/나비 등 직접 매칭)
"""

EN_TO_KR = {
    # === 나무 (Tree) ===
    "tree": "나무전체", "trees": "나무전체", "whole tree": "나무전체",
    "tree outline": "나무전체",
    "trunk": "기둥", "tree trunk": "기둥", "stem": "기둥",
    "branch": "가지", "branches": "가지", "tree branch": "가지",
    "main branch": "가지", "main branches": "가지", "limb": "가지",
    # 수관 (dense foliage mass) vs 나뭇잎 (individual leaves)
    "crown": "수관", "canopy": "수관", "tree crown": "수관",
    "treetop": "수관", "tree top": "수관", "foliage mass": "수관",
    "leaf": "나뭇잎", "leaves": "나뭇잎", "tree leaves": "나뭇잎",
    "foliage": "나뭇잎", "leaves cluster": "나뭇잎",
    "root": "뿌리", "roots": "뿌리", "tree roots": "뿌리",
    "fruit": "열매", "fruits": "열매", "apple": "열매", "apples": "열매",
    "berry": "열매", "berries": "열매", "tree fruit": "열매",
    "flower": "꽃", "flowers": "꽃", "blossom": "꽃", "tree flower": "꽃",
    "knot": "옹이", "tree knot": "옹이",
    # 한국어 변형 (model이 가끔 한국어 답)
    "줄기": "기둥",       # stem → 기둥
    "낙엽": "나뭇잎",     # fallen leaf → 나뭇잎
    "잎": "나뭇잎",       # leaf default
    "잎사귀": "나뭇잎",
    "이파리": "나뭇잎",
    "수관": "수관", "기둥": "기둥", "가지": "가지", "뿌리": "뿌리",
    "나뭇잎": "나뭇잎", "열매": "열매", "꽃": "꽃", "옹이": "옹이",
    "나무": "나무전체", "나무전체": "나무전체",
    # === 자연·하늘 ===
    "sun": "해", "moon": "달", "crescent moon": "달", "crescent": "달",
    "star": "별", "stars": "별",
    "cloud": "구름", "clouds": "구름",
    "rain": "비", "rainbow": "무지개", "sky": "하늘",
    "mountain": "산", "mountains": "산", "hill": "산",
    # 한국어 자연
    "해": "해", "태양": "해", "달": "달", "별": "별", "구름": "구름",
    "비": "비", "무지개": "무지개", "하늘": "하늘", "산": "산",
    # === 동물 (구체적 종 매핑) ===
    "bird": "새", "birds": "새",
    "squirrel": "다람쥐", "squirrels": "다람쥐",
    "monkey": "다람쥐",  # 손그림 모호 — 다람쥐로 통합
    # 일반어 fallback (specific naming 안 됐을 때)
    "animal": "동물", "creature": "동물",
    # 한국어 동물
    "새": "새", "다람쥐": "다람쥐", "토끼": "토끼", "고양이": "고양이",
    "강아지": "강아지", "쥐": "다람쥐", "동물": "동물",
    "butterfly": "나비", "butterflies": "나비", "나비": "나비",
    "bug": "벌레", "insect": "벌레", "벌레": "벌레",
    # === 놀이 ===
    "swing": "그네", "그네": "그네",
    # === 집 (House) ===
    "house": "집전체", "whole house": "집전체", "house outline": "집전체",
    "roof": "지붕", "house roof": "지붕", "rooftop": "지붕",
    "wall": "집벽", "walls": "집벽", "house wall": "집벽", "exterior wall": "집벽",
    "door": "문", "house door": "문", "entrance": "문", "front door": "문",
    "window": "창문", "windows": "창문", "window pane": "창문",
    "chimney": "굴뚝",
    "smoke": "연기", "chimney smoke": "연기",
    "garage": "차고",
    "fence": "울타리", "gate": "울타리",
    "path": "길", "road": "길", "pathway": "길", "walkway": "길",
    "garden": "정원", "yard": "정원",
    "grass": "잔디", "lawn": "잔디",
    "pond": "연못", "pool": "연못",
    "stone": "돌", "rock": "돌", "rocks": "돌",
    "balcony": "발코니",
    "stairs": "계단", "step": "계단", "steps": "계단",
    "mailbox": "우편함",
    # 추가 환경 객체 (관측된 미매핑 단어 → 가장 가까운 GT)
    "river": "연못", "stream": "연못", "lake": "연못", "water": "연못",
    "ground": "잔디", "land": "잔디", "field": "잔디",
    "bush": "잔디", "shrub": "잔디", "vegetation": "잔디",
    "plant": "꽃", "plants": "꽃",
    "leaves cluster": "수관",
    # 한국어 집
    "집": "집전체", "집전체": "집전체",
    "지붕": "지붕", "벽": "집벽", "집벽": "집벽",
    "문": "문", "창문": "창문", "굴뚝": "굴뚝", "연기": "연기",
    "차고": "차고", "울타리": "울타리", "길": "길",
    "정원": "정원", "잔디": "잔디", "연못": "연못",
    "돌": "돌", "발코니": "발코니", "계단": "계단", "우편함": "우편함",
    "덤불": "잔디", "풀": "잔디",
    # === 사람 — 신체 ===
    "person": "사람전체", "people": "사람전체", "whole person": "사람전체",
    "figure": "사람전체", "child": "사람전체", "human": "사람전체",
    "person outline": "사람전체",
    "head": "머리",
    "face": "얼굴",
    "hair": "머리카락",
    "eye": "눈", "eyes": "눈",
    "eyebrow": "눈썹", "eyebrows": "눈썹",
    "nose": "코",
    "mouth": "입", "lips": "입", "smile": "입",
    "ear": "귀", "ears": "귀",
    "neck": "목",
    "body": "상체", "torso": "상체", "upper body": "상체",
    "arm": "팔", "arms": "팔",
    "hand": "손", "hands": "손",
    "finger": "손가락", "fingers": "손가락",
    "leg": "다리", "legs": "다리",
    "foot": "발", "feet": "발",
    # 한국어 신체
    "사람": "사람전체", "사람전체": "사람전체",
    "머리": "머리", "얼굴": "얼굴", "머리카락": "머리카락",
    "눈": "눈", "눈썹": "눈썹", "코": "코", "입": "입", "귀": "귀",
    "목": "목", "몸": "상체", "상체": "상체", "몸통": "상체",
    "팔": "팔", "손": "손", "손가락": "손가락", "다리": "다리", "발": "발",
    # === 사람 — 옷 ===
    "clothes": "옷", "clothing": "옷", "outfit": "옷",
    "shirt": "셔츠", "t-shirt": "셔츠", "top": "상의", "blouse": "상의",
    "pants": "바지", "trousers": "바지", "shorts": "반바지",
    "skirt": "치마", "dress": "원피스", "gown": "원피스",
    # 신발 — 운동화 vs 구두 type 분리 (gender 제거: 남자/여자구두 통합)
    # 운동화 = sneakers/athletic
    # 구두 = 그 외 footwear (남자구두 + 여자구두 + boots/sandals/loafers 등)
    "sneaker": "운동화", "sneakers": "운동화", "athletic shoes": "운동화",
    "running shoes": "운동화", "trainers": "운동화",
    "shoe": "구두", "shoes": "구두",  # generic은 구두 default
    "boots": "구두", "boot": "구두", "sandals": "구두", "sandal": "구두",
    "slippers": "구두", "slipper": "구두",
    "dress shoes": "구두", "loafers": "구두", "loafer": "구두", "heels": "구두", "heel": "구두",
    # 한국어 옷·신발
    "옷": "옷", "셔츠": "셔츠", "상의": "상의", "바지": "바지",
    "반바지": "반바지", "치마": "치마", "원피스": "원피스",
    "신발": "구두", "신": "구두",
    "운동화": "운동화",
    "구두": "구두", "남자구두": "구두", "여자구두": "구두",  # gender 제거
    "샌들": "구두", "슬리퍼": "구두", "부츠": "구두", "하이힐": "구두",
    # === 액세서리·기타 ===
    "hat": "모자", "cap": "모자",
    "glasses": "안경", "sunglasses": "선글라스",
    "necklace": "목걸이", "earrings": "귀걸이",
    "watch": "시계", "ring": "반지", "belt": "벨트",
    "bag": "가방", "backpack": "가방",
    "button": "단추", "buttons": "단추",
    "collar": "옷깃",
    "pocket": "주머니", "pockets": "주머니",
    # 한국어 액세서리
    "모자": "모자", "안경": "안경", "선글라스": "선글라스",
    "목걸이": "목걸이", "귀걸이": "귀걸이", "시계": "시계",
    "반지": "반지", "벨트": "벨트", "가방": "가방",
    "단추": "단추", "옷깃": "옷깃", "주머니": "주머니",
}


def normalize_en_to_kr(name):
    """영어/한국어 객체명 → 한국어 GT 라벨 정규화.

    매칭 순서:
    1. 직접 매칭 (소문자·trim)
    2. 단어 끝부분 매칭 ('tree branch' → 'branch')
    3. 단어 시작 부분 매칭 ('squirrel-like' → 'squirrel')
    4. 한국어 fallthrough
    """
    if not name: return name
    n = name.strip().lower()
    # 1. 직접 매칭
    if n in EN_TO_KR:
        return EN_TO_KR[n]
    # 2. 단어 끝부분 매칭 (예: "tree's branches" → "branches")
    for k, v in EN_TO_KR.items():
        if n.endswith(" " + k) or n.endswith("-" + k) or n.endswith("_" + k):
            return v
    # 3. 단어 시작 부분 매칭 (예: "squirrel-like" → "squirrel")
    for k, v in EN_TO_KR.items():
        if " " in k or "-" in k: continue  # 단일 단어만
        if n.startswith(k + " ") or n.startswith(k + "-") or n.startswith(k + "_"):
            return v
    # 4. 한국어이면 정규화 시도
    if any('가' <= c <= '힣' for c in name):
        # 한국어 매핑도 dict에 있음 (예: "줄기" → "기둥")
        if name.strip() in EN_TO_KR:
            return EN_TO_KR[name.strip()]
        return name
    return name  # 매핑 안 되면 원본


def normalize_with_context(name, bbox=None, image_size=None, category=None):
    """문맥 기반 매핑 (bbox 크기, 카테고리).

    추가 규칙:
    - "leaves"/"잎"이면서 bbox 면적 >= 15% → 수관 (큰 군집)
    - "leaves"/"잎"이면서 bbox 면적 < 15% → 나뭇잎 (개별)
    - "cloud"/"구름"이면서 집 카테고리 → 검토 필요 (연기 가능성)

    bbox: [x1, y1, x2, y2] 원본 좌표
    image_size: (W, H) 이미지 크기
    category: TL_나무 등 카테고리
    """
    n_lower = name.strip().lower()
    # 잎 size-based mapping (나뭇잎 vs 수관)
    if n_lower in ("leaf", "leaves", "foliage", "잎", "잎사귀", "이파리"):
        if bbox and image_size:
            x1, y1, x2, y2 = bbox
            W, H = image_size
            area_ratio = abs((x2-x1) * (y2-y1)) / (W * H)
            if area_ratio >= 0.15:
                return "수관"  # 큰 군집
            else:
                return "나뭇잎"  # 개별
        return "나뭇잎"  # bbox 정보 없으면 기본값
    # cloud near chimney (집 + 굴뚝 근처) — Judge가 별도 처리 권장
    if n_lower in ("cloud", "clouds", "구름") and category == "TL_집":
        # 굴뚝 근처면 연기일 가능성, 위쪽이면 구름
        if bbox and image_size:
            _, y1, _, y2 = bbox
            H = image_size[1]
            cy = (y1 + y2) / 2 / H
            # 상단 1/3은 구름, 중단/하단(굴뚝 근처)은 연기 가능
            if cy >= 1/3:
                return "연기"
        return "구름"
    # 기본 fallback
    return normalize_en_to_kr(name)


def is_shoe(name):
    """신발 alias 체크 — 운동화 + 구두 모두 footwear."""
    n = name.strip().lower()
    shoe_keys = {
        "shoe", "shoes", "sneaker", "sneakers", "boot", "boots",
        "sandal", "sandals", "slipper", "slippers", "trainers",
        "athletic shoes", "running shoes", "dress shoes", "loafers", "heels",
        "신발", "운동화", "구두", "남자구두", "여자구두", "샌들", "슬리퍼", "부츠",
    }
    return n in shoe_keys


def normalize_gt_shoe(gt_label):
    """GT 라벨의 신발 통합 — 남자구두/여자구두 gender 편향 제거.

    GT side:
    - "남자구두" → "구두"
    - "여자구두" → "구두"
    - "운동화" → "운동화" (그대로)

    §17.7 GT 카테고리 편향 보정용 — 채점 시 양쪽 통합 후 비교.
    """
    gl = gt_label.strip()
    if gl in ("남자구두", "여자구두"):
        return "구두"
    return gl
