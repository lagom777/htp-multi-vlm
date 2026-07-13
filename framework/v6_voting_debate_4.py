"""HTP v6 — Voting + Debate + Judge (C 버전) on 4장 (카테고리별 1장).

파이프라인:
  Round 1: 3 VLM 자유 분석 (temp 0.2)
  → 다수결 (2/3) → 합의 객체 채택, 불일치 객체 분리
  Round 2: 불일치 객체에 대해 3 VLM revision (temp 0.5)
  → 다수결 → 신규 합의 추가, 여전히 불일치는 Judge로
  Blind Judge: Claude Opus 4.7 (텍스트 근거만, 이미지 차단)
  → 최종 객체 + 위치 (9-region)

모델 (전부 OpenRouter):
  google/gemini-3.5-flash
  openai/gpt-5.5
  anthropic/claude-sonnet-4-6
  anthropic/claude-opus-4.7  (Judge)

출력: v6_state_4.json (partial save 매 이미지마다)
로그: v6_state_4.log
"""
import os, json, base64, time, re
from datetime import datetime
from collections import Counter, defaultdict
from openai import OpenAI

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(PROJECT_DIR, "config.json")) as f:
    config = json.load(f)

client = OpenAI(
    api_key=config.get("OPENROUTER_API_KEY", ""),
    base_url="https://openrouter.ai/api/v1",
)

# ---------- 설정 ----------
DATASET_ROOT = os.path.join(
    PROJECT_DIR,
    "266.AI 기반 아동 미술심리 진단을 위한 그림 데이터 구축",
    "01-1.정식개방데이터",
)
TRAIN_ORIGIN = os.path.join(DATASET_ROOT, "Training", "01.원천데이터")
TRAIN_LABEL = os.path.join(DATASET_ROOT, "Training", "02.라벨링데이터")
CAT_MAP = {
    "TL_나무": "TS_나무", "TL_남자사람": "TS_남자사람",
    "TL_여자사람": "TS_여자사람", "TL_집": "TS_집",
}

# 4장 — 카테고리별 1장씩 (기존 평가셋 첫번째)
TEST_IMAGES = [
    ("TL_나무", "나무_8_남_01445.jpg"),
    ("TL_집", "집_12_여_08971.jpg"),
    ("TL_남자사람", "남자사람_13_남_02804.jpg"),
    ("TL_여자사람", "여자사람_10_남_02125.jpg"),
]

MODELS_R1 = [
    ("gemini", "google/gemini-3.1-pro-preview"),  # Pro for JSON instruction following
    ("gpt", "openai/gpt-5.5"),
    ("claude", "anthropic/claude-sonnet-4-6"),
]
JUDGE_MODEL = "anthropic/claude-opus-4.7"

TEMP_R1 = 0.2
TEMP_R2 = 0.5
TEMP_JUDGE = 0.2
MAX_TOKENS = 4000

# ---------- 카테고리별 프롬프트 ----------
CATEGORY_HEADER = {
    "TL_집": "[HOUSE — House-Tree-Person 검사 중 '집' 그림]",
    "TL_나무": "[TREE — House-Tree-Person 검사 중 '나무' 그림]",
    "TL_남자사람": "[PERSON (male) — House-Tree-Person 검사 중 '남자사람' 그림]",
    "TL_여자사람": "[PERSON (female) — House-Tree-Person 검사 중 '여자사람' 그림]",
}
CATEGORY_DESC = {
    "TL_집": "이 그림은 HTP 심리검사의 '집(House)' 그림입니다.",
    "TL_나무": "이 그림은 HTP 심리검사의 '나무(Tree)' 그림입니다.",
    "TL_남자사람": "이 그림은 HTP 심리검사의 '남자사람(Person, male)' 그림입니다.",
    "TL_여자사람": "이 그림은 HTP 심리검사의 '여자사람(Person, female)' 그림입니다.",
}
COMMON_BODY = """
그림에 보이는 모든 객체를 자유롭게 식별해주세요. 미리 주어진 객체 목록은 없으니, 보이는 그대로 적어주세요.

⚠ 응답 규칙 (필수 준수):
1. **반드시 JSON 배열만 출력하세요.** 다른 어떤 텍스트도 절대 포함하지 마세요.
2. **마크다운 번호 매기기 금지** (1. 2. 3. 같은 형식).
3. **자연어 설명 금지** ("이 그림은..." 같은 도입 X).
4. 응답은 `[`로 시작해서 `]`로 끝나야 합니다.
5. 객체명은 **간결한 단일 명사**로만 작성. 방향 prefix(왼쪽/오른쪽/위쪽 등) 없이, 괄호 안 부가 설명 없이, "전체"·"부분" 같은 수식어 없이.
6. 같은 종류 객체가 여러 개 있어도 **하나의 객체명으로 묶어 1개 항목으로** 답하세요.

각 객체는 다음 5개 필드로 답:
```json
[
  {
    "객체": "(객체명, 한국어 단일 명사)",
    "판정": "있음",
    "위치": "(아래 9개 라벨 중 하나)",
    "크기": "(이미지 대비 비율)",
    "근거": "(왜 이 객체로 판단했는지)"
  }
]
```

**위치 필드는 반드시 다음 9개 라벨 중 하나로만 선택**하세요 (객체 중심이 들어가는 영역):

  top-left    | top-center    | top-right
  middle-left | middle-center | middle-right
  bottom-left | bottom-center | bottom-right

종이를 가로 3등분 × 세로 3등분으로 나눈 격자에서, 객체 중심점이 들어가는 한 영역을 고르면 됩니다.

다시 강조: **JSON 배열만 응답.** 응답이 `[`로 시작하지 않으면 파싱 실패하니 절대 다른 텍스트 포함 금지."""


def make_r1_prompt(cat):
    return f"{CATEGORY_HEADER[cat]}\n\n{CATEGORY_DESC[cat]}\n{COMMON_BODY}"


def make_r2_prompt(cat, own_answer, others_answers, disagreed_objects):
    """Round 2 revision prompt — 자기 답 + 다른 모델 답 + 불일치 객체 명시."""
    header = CATEGORY_HEADER[cat]
    desc = CATEGORY_DESC[cat]
    disagreed_str = ", ".join(disagreed_objects)

    others_str = ""
    for i, (model_name, ans) in enumerate(others_answers.items()):
        others_str += f"\n[다른 모델 {i+1} ({model_name})의 답변]:\n{json.dumps(ans, ensure_ascii=False, indent=2)}\n"

    return f"""{header}

{desc}

이전 Round 1에서 당신과 다른 두 모델이 답했습니다. 다음 객체들에 대해 의견이 갈렸습니다:
**{disagreed_str}**

[당신(Round 1)의 답변]:
{json.dumps(own_answer, ensure_ascii=False, indent=2)}
{others_str}

그림을 다시 자세히 보고, 위 의견 차이가 있는 객체들에 대해 재검토한 최종 답변을 주세요.

⚠ 응답 규칙 (필수 준수):
1. **반드시 JSON 배열만 출력하세요.** 다른 어떤 텍스트도 절대 포함하지 마세요.
2. **마크다운 번호 매기기 금지** (1. 2. 3. 같은 형식).
3. **자연어 설명 금지** ("이 그림은..." 같은 도입 X).
4. 응답은 `[`로 시작해서 `]`로 끝나야 합니다.
5. 객체명은 **간결한 단일 명사**로 작성. 방향 prefix(왼쪽/오른쪽 등) 없이, 괄호 안 부가 설명 없이, 수식어 없이.

답변 형식 (불일치 객체들에 대해서만):
```json
[
  {{
    "객체": "(객체명, 한국어 단일 명사)",
    "판정": "있음 또는 없음",
    "위치": "(9-region 라벨, 판정이 있음일 때만, 없음이면 빈 문자열)",
    "크기": "(이미지 대비 비율, 판정이 있음일 때만, 없음이면 빈 문자열)",
    "근거": "(재검토 이유 간결히)"
  }}
]
```

위치 9개 라벨 (이 중 하나만):
  top-left | top-center | top-right
  middle-left | middle-center | middle-right
  bottom-left | bottom-center | bottom-right

다시 강조: **JSON 배열만 응답.** 응답 시작이 `[`가 아니면 파싱 실패하니 주의."""


def make_judge_prompt(cat, disagreed_objects, all_answers):
    """Blind Judge prompt — 이미지 없이 텍스트 근거만으로 판정."""
    obj_summary = "\n".join([f"  - {obj}" for obj in disagreed_objects])

    detail = ""
    for obj in disagreed_objects:
        detail += f"\n[객체: {obj}]\n"
        for model_name, ans_list in all_answers.items():
            obj_ans = next((a for a in ans_list if a.get("객체") == obj or normalize_label(a.get("객체", "")) == normalize_label(obj)), None)
            if obj_ans:
                detail += f"  {model_name}: 판정={obj_ans.get('판정')}, 위치={obj_ans.get('위치')}, 근거={obj_ans.get('근거','')}\n"
            else:
                detail += f"  {model_name}: (답에 없음)\n"

    return f"""당신은 HTP 심리검사 그림 분석의 Judge입니다.
**이미지는 볼 수 없습니다.** 오직 다른 세 모델이 제공한 텍스트 근거만 보고 판정하세요.

카테고리: {CATEGORY_DESC[cat]}

다음 객체들에 대해 세 모델이 의견이 갈렸습니다 (Round 1·2 모두 토론):
{obj_summary}

세 모델의 답변 상세:
{detail}

각 객체에 대해 다음 기준으로 판정하세요:
1. 위치·근거가 구체적이고 일관된 모델 다수가 있다면 그 답 채택
2. 한 모델만 주장하더라도 근거가 매우 구체적이고 다른 모델의 반대 근거가 모호하면 채택
3. 모든 근거가 모호하면 거절

JSON 배열로만 답:
```json
[
  {{
    "객체": "(객체명)",
    "최종_판정": "있음" 또는 "없음",
    "최종_위치": "(9-region 라벨, 있음일 때만)",
    "Judge_근거": "(왜 이 판정인지)"
  }}
]
```"""


# ---------- 유의어 정규화 (GT 기반) ----------
GT_SYNONYMS = {
    "태양": ["태양", "해", "sun", "햇빛", "햇살"],
    "잔디": ["잔디", "풀", "풀밭", "잔디밭", "grass", "땅", "지면"],
    # 신발 — 운동화 vs 구두 type 분리 유지 (gender만 제거: 남자/여자구두 통합)
    # §17.7 GT 카테고리 편향 보정 (gender)
    "운동화": ["운동화", "sneaker", "sneakers", "athletic shoes",
              "running shoes", "trainers"],
    "구두": ["구두", "남자구두", "여자구두",
            "남자 구두", "여자 구두",
            "남자신발", "여자신발", "남자 신발", "여자 신발",
            "신발", "신", "shoe", "shoes",
            "boots", "boot", "부츠",
            "sandals", "sandal", "샌들",
            "slippers", "slipper", "슬리퍼",
            "loafers", "loafer", "heels", "heel",
            "dress shoes", "하이힐"],
    "머리카락": ["머리카락", "hair", "앞머리", "땋은머리", "헤어", "두발"],
    "머리": ["머리", "head", "두상", "두부"],
    "수관": ["수관", "나무수관", "canopy", "treecrown", "잎부분", "수간",
             "잎무리", "나뭇잎무리", "잎사귀무리"],
    "기둥": ["기둥", "나무줄기", "줄기", "trunk", "나무기둥", "나무 줄기",
             "나무 기둥"],
    "가지": ["가지", "나뭇가지", "branch", "나무가지"],
    "뿌리": ["뿌리", "나무뿌리", "root", "나무 뿌리"],
    "나뭇잎": ["나뭇잎", "잎", "leaf", "leaves", "잎사귀"],
    "열매": ["열매", "사과", "체리", "도토리", "버찌", "fruit", "apple",
             "cherry", "과일", "열매들", "열매군집"],
    "꽃": ["꽃", "튤립", "들꽃", "flower", "꽃잎", "꽃봉오리"],
    "구름": ["구름", "cloud", "먹구름"],
    "달": ["달", "moon", "초승달", "보름달"],
    "별": ["별", "star", "별빛"],
    "새": ["새", "bird", "참새", "비둘기"],
    "다람쥐": ["다람쥐", "squirrel"],
    "그네": ["그네", "swing"],
    "지붕": ["지붕", "roof", "기와지붕"],
    "집벽": ["집벽", "벽", "wall", "외벽", "벽면"],
    "문": ["문", "door", "현관문"],
    "창문": ["창문", "window", "유리창", "창"],
    "굴뚝": ["굴뚝", "chimney"],
    "연기": ["연기", "smoke"],
    "길": ["길", "path", "road", "도로", "보도"],
    "울타리": ["울타리", "fence", "담장", "담"],
    "산": ["산", "mountain", "언덕"],
    # '강', '물가', '호수'도 연못과 같은 '물' 객체로 통합 (GT 라벨이 '연못')
    "연못": ["연못", "pond", "물가", "호수", "웅덩이", "강", "river", "물"],
    # '나무'는 '나무전체'에 통합 — 두 카테고리 GT 모두 매칭되도록
    "나무전체": ["나무전체", "나무 전체", "나무", "tree", "작은나무",
                  "큰나무", "작은 나무", "큰 나무", "어린나무"],
    "집전체": ["집전체", "집 전체", "집", "house", "건물", "home"],
    "사람전체": ["사람전체", "사람 전체", "남자사람", "남자 사람",
                  "여자사람", "여자 사람", "사람", "인물", "person",
                  "사람그림", "인물그림"],
    "얼굴": ["얼굴", "face"],
    # '속눈썹'도 눈 영역의 일부로 통합
    "눈": ["눈", "eye", "eyes", "양 눈", "두 눈", "속눈썹", "눈동자",
            "눈썹"],
    "코": ["코", "nose"],
    "입": ["입", "mouth"],
    "귀": ["귀", "ear", "ears", "양 귀", "두 귀"],
    "목": ["목", "neck"],
    "상체": ["상체", "몸통", "body", "상의", "티셔츠", "옷", "셔츠",
             "상체옷", "윗옷"],
    "팔": ["팔", "arm", "arms", "양 팔", "두 팔", "왼팔", "오른팔"],
    # '손가락'도 손 영역의 일부로 통합
    "손": ["손", "hand", "hands", "양 손", "두 손", "왼손", "오른손",
            "손가락"],
    "다리": ["다리", "leg", "legs", "양 다리", "두 다리", "왼다리", "오른다리",
              "맨다리"],
    "발": ["발", "foot", "feet", "양 발", "두 발", "왼발", "오른발",
            "발가락", "맨발"],
    "단추": ["단추", "button", "buttons", "단추들"],
    "주머니": ["주머니", "pocket", "pockets", "주머니들"],
    # 신발 — AI Hub GT가 운동화 + 남자/여자구두 둘 다 라벨링 (그림에 진짜 2종류 그려짐)
    # 모델 출력이 단순 "신발"이면 카테고리에 따라 다르게 매핑되지만 충돌 가능 — alias만 추가
    "운동화": ["운동화", "스니커즈", "sneakers", "운동화한쌍"],
    "남자구두": ["남자구두", "남성구두", "남자 구두", "남성 구두"],
    "여자구두": ["여자구두", "여성구두", "여자 구두", "여성 구두", "구두"],
    # '바지'·'반바지'·'치마'·'하의'는 GT '다리' 영역(다리 옷)으로 통합
    # (GT가 옷 세부 라벨 없이 '다리'에 옷+다리를 함께 처리)
    # 단 본 thesis는 환각 측정 목적상 별도 처리 — 매핑 X (canonical 외 진짜 객체)
    # '소매'·'끈' 같은 옷 부분도 별개로 둠
    # 추가 변형 정규화 — 같은 객체의 다른 표현만 매핑
    # (옹이는 수관과 다른 객체 → 별도 환각으로 카운트)
    "손잡이": ["손잡이", "문손잡이", "현관손잡이", "handle"],  # 같은 객체, canonical 외
    "나선": ["나선", "나선무늬", "spiral"],  # 같은 장식 표현
}


_PAREN_RE = re.compile(r"\s*\([^)]*\)")
# 단글자 prefix(상/하/좌/우 등)는 정상 단어 자르므로 제거.
# 다글자 prefix만 + 공백 1+ 강제 (예: "상체"는 잘리지 않고, "왼쪽 꽃"은 잘림)
_DIRECTION_PREFIX_RE = re.compile(
    r"^(왼쪽|오른쪽|위쪽|아래쪽|상단|하단|좌측|우측|중앙|중간|"
    r"좌상|우상|좌하|우하|중심|"
    r"첫번째|두번째|세번째|네번째|첫|두|세|네|"
    r"1번|2번|3번|4번|5번|6번|7번|8번|9번|10번)\s+"
)


def preprocess_label(name):
    """객체명 변형을 정규화 전 전처리.

    - 괄호 안 설명 제거: '나무 (중앙 큰 나무)' -> '나무'
    - 방향/순서 prefix 반복 제거: '왼쪽 꽃' -> '꽃', '오른쪽 위 신발' -> '신발'
    - 공백·구두점 정리
    """
    if not name:
        return name
    s = name.strip()
    # 1. 괄호 안 설명 제거
    s = _PAREN_RE.sub("", s).strip()
    # 2. 방향·순서 prefix 반복 제거
    for _ in range(5):  # 최대 5번 (안전 한계)
        new_s = _DIRECTION_PREFIX_RE.sub("", s).strip()
        if new_s == s or not new_s:
            break
        s = new_s
    # 3. 끝부분 공백
    s = s.strip(" -·,/")
    return s


def normalize_label(name):
    """객체명 → 정규 라벨. 전처리 + 정확 매칭 + 부분 매칭 (긴 라벨 우선)."""
    if not name:
        return name
    s = preprocess_label(name)
    s_norm = s.strip().lower().replace(" ", "")

    # 1) 정확 매칭
    for canonical, aliases in GT_SYNONYMS.items():
        for a in aliases:
            if s_norm == a.lower().replace(" ", ""):
                return canonical

    # 2) 부분 매칭 — 양쪽 모두 길이 ≥2 강제 + 긴 alias 우선
    candidates = []
    if len(s_norm) >= 2:
        for canonical, aliases in GT_SYNONYMS.items():
            for a in aliases:
                a_norm = a.lower().replace(" ", "")
                if len(a_norm) >= 2 and (a_norm in s_norm or s_norm in a_norm):
                    strength = min(len(a_norm), len(s_norm))
                    candidates.append((canonical, strength, len(a_norm)))
    if candidates:
        # alias 길이 긴 것 우선 (머리카락 우선, 머리 후순위)
        candidates.sort(key=lambda x: (-x[2], -x[1]))
        return candidates[0][0]

    return s  # 매핑 실패 — 전처리된 원본 반환


# ---------- API 호출 ----------
def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_vlm(model_id, prompt, image_path=None, temperature=0.2, max_tokens=MAX_TOKENS, retries=3):
    """OpenRouter 통해 VLM 호출. 이미지 있으면 vision, 없으면 텍스트."""
    if image_path:
        b64 = encode_image(image_path)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }]
    else:
        messages = [{"role": "user", "content": prompt}]

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content
            if text and len(text.strip()) > 5:
                return text.strip()
            raise RuntimeError(f"empty/short response: {text!r}")
        except Exception as e:
            msg = str(e)[:200]
            print(f"      try{attempt+1} 실패 ({model_id}): {msg}", flush=True)
            time.sleep(2 * (attempt + 1))
    return "Error"


def parse_json_block(text):
    if not text or text == "Error":
        return None
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    raw = m.group(1) if m else text.strip()
    try:
        return json.loads(raw)
    except Exception:
        m2 = re.search(r"\[\s*\{.*\}\s*\]", raw, re.DOTALL)
        if m2:
            try:
                return json.loads(m2.group(0))
            except Exception:
                pass
    return None


# ---------- 다수결 ----------
def majority_vote(round_answers):
    """Round 답들로부터 객체별 다수결.

    round_answers: {model: [{객체:, 판정:, 위치:, ...}]}
    Returns: (consensus, disagreed)
      consensus = {정규라벨: {판정, 위치, 근거_list, 모델_list}}
      disagreed = {정규라벨: [(model, parsed_item), ...]}
    """
    # 객체별 votes 수집 (정규화 후)
    votes = defaultdict(dict)  # {정규라벨: {model: parsed_item}}
    for model, ans_list in round_answers.items():
        if not isinstance(ans_list, list):
            continue
        for item in ans_list:
            if not isinstance(item, dict):
                continue
            if item.get("판정") != "있음":
                continue
            norm = normalize_label(item.get("객체", ""))
            if not norm:
                continue
            votes[norm][model] = item

    consensus = {}
    disagreed = {}
    num_models = len(round_answers)
    threshold = 2 if num_models >= 3 else 2  # 2/3 또는 2/2

    for obj, model_items in votes.items():
        if len(model_items) >= threshold:
            # 합의 — 위치는 다수결, 충돌 시 첫 모델 것
            position_votes = Counter([it.get("위치", "") for it in model_items.values() if it.get("위치")])
            best_pos = position_votes.most_common(1)[0][0] if position_votes else ""
            consensus[obj] = {
                "판정": "있음",
                "위치": best_pos,
                "투표": len(model_items),
                "참여_모델": list(model_items.keys()),
                "위치_분포": dict(position_votes),
            }
            # 위치 만장일치인지 체크 — 위치 자체가 갈리면 disagreed로 보내야 함
            if len(position_votes) > 1:
                # 위치 갈렸으면 disagreed로 추가 (식별은 합의, 위치는 토의)
                disagreed[obj] = list(model_items.items())
                # consensus에서 빼지 않고 함께 둔다 (debate에서 위치 다시 묻기)
        else:
            disagreed[obj] = list(model_items.items())

    return consensus, disagreed


# ---------- 메인 ----------
def get_gt(cat, img_name):
    p = os.path.join(TRAIN_LABEL, cat, img_name.replace(".jpg", ".json"))
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def gt_objects_with_region(gt_dict, img_w=1280, img_h=1280):
    """GT JSON에서 객체별 9-region 라벨 추출."""
    if gt_dict and "meta" in gt_dict and "img_resolution" in gt_dict["meta"]:
        try:
            wh = gt_dict["meta"]["img_resolution"].split("x")
            img_w, img_h = int(wh[0]), int(wh[1])
        except Exception:
            pass

    by_label = defaultdict(list)
    if not gt_dict or "annotations" not in gt_dict:
        return {}
    for bb in gt_dict["annotations"]["bbox"]:
        lbl = normalize_label(bb["label"])
        cx = (bb["x"] + bb["w"] / 2) / img_w
        cy = (bb["y"] + bb["h"] / 2) / img_h
        col = "left" if cx < 1/3 else ("right" if cx > 2/3 else "center")
        row = "top" if cy < 1/3 else ("bottom" if cy > 2/3 else "middle")
        by_label[lbl].append(f"{row}-{col}")
    return dict(by_label)


def main():
    import sys
    suffix = sys.argv[1] if len(sys.argv) > 1 else ""
    suf = f"_{suffix}" if suffix else ""
    out_path = f"/Users/kg/nonmoon/htp_thesis/v6_state_4{suf}.json"
    log_path = f"/Users/kg/nonmoon/htp_thesis/v6_state_4{suf}.log"

    print(f"=== HTP v6 — Voting + Debate + Judge (C 버전) 4장 ===")
    print(f"시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"모델: {[m[1] for m in MODELS_R1]} + Judge {JUDGE_MODEL}")
    print(f"Temperature: R1={TEMP_R1}, R2={TEMP_R2}, Judge={TEMP_JUDGE}")
    print()

    results = []

    for idx, (cat, img_name) in enumerate(TEST_IMAGES):
        print(f"\n[{idx+1}/{len(TEST_IMAGES)}] {cat}/{img_name}")
        print(f"  시작 {datetime.now().strftime('%H:%M:%S')}")
        img_path = os.path.join(TRAIN_ORIGIN, CAT_MAP[cat], img_name)
        if not os.path.exists(img_path):
            print(f"  ❌ 이미지 없음")
            continue

        gt_dict = get_gt(cat, img_name)
        gt_regions = gt_objects_with_region(gt_dict)

        # ----- Round 1 -----
        print(f"  → Round 1 (3 VLM, temp={TEMP_R1})")
        round1_raw = {}
        round1_parsed = {}
        for name, model_id in MODELS_R1:
            t0 = time.time()
            prompt = make_r1_prompt(cat)
            raw = call_vlm(model_id, prompt, image_path=img_path, temperature=TEMP_R1)
            elapsed = time.time() - t0
            parsed = parse_json_block(raw)
            n = len(parsed) if isinstance(parsed, list) else 0
            status = "✅" if n else "⚠"
            print(f"    {status} {name} {elapsed:.1f}s, 객체 {n}개, raw {len(raw)}자", flush=True)
            round1_raw[name] = raw
            round1_parsed[name] = parsed
            time.sleep(1)

        # ----- Round 1 다수결 -----
        r1_consensus, r1_disagreed = majority_vote(round1_parsed)
        print(f"  Round 1 합의 {len(r1_consensus)}개 / 불일치 {len(r1_disagreed)}개")
        if r1_disagreed:
            print(f"    불일치 객체: {list(r1_disagreed.keys())[:8]}{'...' if len(r1_disagreed)>8 else ''}")

        # ----- Round 2 (불일치 있으면) -----
        round2_raw = {}
        round2_parsed = {}
        r2_consensus = {}
        r2_disagreed_final = {}

        if r1_disagreed:
            print(f"  → Round 2 revision (3 VLM, temp={TEMP_R2})")
            disagreed_obj_names = list(r1_disagreed.keys())
            for name, model_id in MODELS_R1:
                own_ans = round1_parsed.get(name, [])
                others_ans = {n: round1_parsed.get(n, []) for n, _ in MODELS_R1 if n != name}
                t0 = time.time()
                prompt = make_r2_prompt(cat, own_ans, others_ans, disagreed_obj_names)
                raw = call_vlm(model_id, prompt, image_path=img_path, temperature=TEMP_R2)
                elapsed = time.time() - t0
                parsed = parse_json_block(raw)
                n = len(parsed) if isinstance(parsed, list) else 0
                status = "✅" if n else "⚠"
                print(f"    {status} {name} {elapsed:.1f}s, 객체 {n}개", flush=True)
                round2_raw[name] = raw
                round2_parsed[name] = parsed
                time.sleep(1)

            # ----- Round 2 다수결 -----
            r2_consensus, r2_disagreed_final = majority_vote(round2_parsed)
            print(f"  Round 2 후 추가 합의 {len(r2_consensus)}개 / 잔여 불일치 {len(r2_disagreed_final)}개")
        else:
            print(f"  Round 1만으로 모든 객체 합의 — Round 2 건너뜀")

        # ----- Blind Judge (잔여 불일치) -----
        judge_raw = ""
        judge_parsed = None
        if r2_disagreed_final:
            print(f"  → Blind Judge ({JUDGE_MODEL}, 텍스트만, temp={TEMP_JUDGE})")
            t0 = time.time()
            # 모든 round의 답 합쳐 evidence 생성
            all_evidence = {}
            for name, _ in MODELS_R1:
                r1 = round1_parsed.get(name, []) or []
                r2 = round2_parsed.get(name, []) or []
                all_evidence[name] = (r1 + r2) if isinstance(r1, list) and isinstance(r2, list) else (r1 or r2 or [])
            j_prompt = make_judge_prompt(cat, list(r2_disagreed_final.keys()), all_evidence)
            judge_raw = call_vlm(JUDGE_MODEL, j_prompt, image_path=None, temperature=TEMP_JUDGE)
            elapsed = time.time() - t0
            judge_parsed = parse_json_block(judge_raw)
            n = len(judge_parsed) if isinstance(judge_parsed, list) else 0
            print(f"    ✅ Judge {elapsed:.1f}s, 판정 {n}건")

        # ----- 최종 통합 -----
        final = {}
        final.update(r1_consensus)
        final.update(r2_consensus)
        if judge_parsed and isinstance(judge_parsed, list):
            for item in judge_parsed:
                obj = normalize_label(item.get("객체", ""))
                if item.get("최종_판정") == "있음" and obj:
                    final[obj] = {
                        "판정": "있음",
                        "위치": item.get("최종_위치", ""),
                        "근거_judge": item.get("Judge_근거", ""),
                        "from": "judge",
                    }

        # ----- 9-region 평가 -----
        eval_result = {
            "tp": 0, "fn": 0,
            "pos_correct": 0, "pos_total": 0,
            "fp_hall": 0,
            "details": [],
        }
        if gt_regions:
            # GT에 있는데 final에 잡힌 객체 = TP
            for gt_label, gt_pos_list in gt_regions.items():
                if gt_label in final:
                    eval_result["tp"] += 1
                    final_pos = final[gt_label].get("위치", "")
                    eval_result["pos_total"] += 1
                    if final_pos and final_pos in gt_pos_list:
                        eval_result["pos_correct"] += 1
                        eval_result["details"].append({"obj": gt_label, "ok_pos": True, "vlm": final_pos, "gt": gt_pos_list})
                    else:
                        eval_result["details"].append({"obj": gt_label, "ok_pos": False, "vlm": final_pos, "gt": gt_pos_list})
                else:
                    eval_result["fn"] += 1
                    eval_result["details"].append({"obj": gt_label, "missed": True})
            # final에 있는데 GT에 없는 객체 = 환각
            for f_obj in final:
                if f_obj not in gt_regions:
                    eval_result["fp_hall"] += 1
                    eval_result["details"].append({"obj": f_obj, "halluc": True})

        gt_count = len(gt_regions) if gt_regions else 0
        print(f"  📊 식별 TP={eval_result['tp']}/{gt_count}, "
              f"위치 {eval_result['pos_correct']}/{eval_result['pos_total']}, "
              f"환각 {eval_result['fp_hall']}")

        # ----- 결과 저장 -----
        results.append({
            "category": cat,
            "image": img_name,
            "gt_unique_labels": list(gt_regions.keys()) if gt_regions else [],
            "gt_regions": gt_regions,
            "round1": {
                "raw": round1_raw,
                "parsed": round1_parsed,
            },
            "round1_consensus": r1_consensus,
            "round1_disagreed": list(r1_disagreed.keys()),
            "round2": {
                "raw": round2_raw,
                "parsed": round2_parsed,
            },
            "round2_consensus": r2_consensus,
            "round2_disagreed_final": list(r2_disagreed_final.keys()),
            "judge_raw": judge_raw,
            "judge_parsed": judge_parsed,
            "final": final,
            "eval": eval_result,
        })

        # partial save
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  💾 partial save → {out_path}")

    # 요약
    print(f"\n=== 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"\n간이 요약:")
    print(f"{'카테고리':10s} {'이미지':35s} {'TP/GT':>10s} {'위치':>10s} {'환각':>5s} {'R1합의':>6s} {'R2합의':>6s} {'Judge':>5s}")
    print('-' * 100)
    for r in results:
        ev = r['eval']
        gt_n = len(r['gt_unique_labels'])
        r1_n = len(r['round1_consensus'])
        r2_n = len(r['round2_consensus'])
        j_n = len(r['judge_parsed']) if isinstance(r.get('judge_parsed'), list) else 0
        print(f"{r['category']:10s} {r['image'][:35]:35s} {ev['tp']}/{gt_n:>4d} "
              f"{ev['pos_correct']}/{ev['pos_total']:>4d} {ev['fp_hall']:>5d} "
              f"{r1_n:>6d} {r2_n:>6d} {j_n:>5d}")


if __name__ == "__main__":
    main()
