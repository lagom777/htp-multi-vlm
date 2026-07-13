"""Agent v3 Step 1-3 병렬 실행 버전

4 VLM 호출을 concurrent.futures로 동시 실행 → 4배 빠름.
예상 시간: 20~25분 (기존 65~90분 대비)
"""
import os, json, time, re, base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import google.generativeai as genai
from openai import OpenAI

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(PROJECT_DIR, "config.json"), 'r') as f:
    config = json.load(f)

genai.configure(api_key=config.get("GOOGLE_API_KEY", ""))
gpt_client = OpenAI(api_key=config.get("OPENAI_API_KEY", ""))
grok_client = OpenAI(api_key=config.get("GROK_API_KEY", ""), base_url="https://api.x.ai/v1")
openrouter_client = OpenAI(api_key=config.get("OPENROUTER_API_KEY", ""), base_url="https://openrouter.ai/api/v1")

DATASET_ROOT = os.path.join(PROJECT_DIR, "266.AI 기반 아동 미술심리 진단을 위한 그림 데이터 구축", "01-1.정식개방데이터")
TRAIN_ORIGIN = os.path.join(DATASET_ROOT, "Training", "01.원천데이터")
TRAIN_LABEL = os.path.join(DATASET_ROOT, "Training", "02.라벨링데이터")
CAT_MAP = {"TL_나무": "TS_나무", "TL_남자사람": "TS_남자사람", "TL_여자사람": "TS_여자사람", "TL_집": "TS_집"}

SYNONYMS = {
    "해": ["태양", "sun"], "풀": ["잔디", "grass", "풀밭"],
    "구름": ["cloud"], "별": ["star"], "달": ["moon", "초승달"],
    "산": ["mountain"], "꽃": ["flower"], "새": ["bird"],
    "나무": ["나무전체", "tree"], "수관": ["나무수관", "treecrown", "canopy"],
    "기둥": ["나무줄기", "trunk"], "가지": ["branch"],
    "뿌리": ["나무뿌리", "root"], "나뭇잎": ["잎", "leaf", "leaves"],
    "열매": ["fruit", "사과", "도토리"], "다람쥐": ["squirrel", "동물"],
    "그네": ["swing"], "집": ["집전체", "house"], "사람": ["사람전체", "person"],
    "신발": ["운동화", "구두", "shoe", "하이힐"],
    "손": ["hand"], "발": ["foot"],
    "창문": ["window"], "문": ["door"], "지붕": ["roof"],
    "굴뚝": ["chimney"], "연기": ["smoke"], "길": ["path", "road"],
    "연못": ["pond"], "울타리": ["fence"], "머리": ["head"],
    "머리카락": ["hair"], "눈": ["eye"], "코": ["nose"],
    "입": ["mouth"], "귀": ["ear"], "팔": ["arm"],
    "다리": ["leg"], "상체": ["body", "몸통"],
    "목": ["neck"], "얼굴": ["face"], "집벽": ["벽", "wall"],
    "단추": ["button"], "주머니": ["pocket"],
    "태양": ["sun"], "잔디": ["grass"], "집전체": ["house"],
    "사람전체": ["person"],
}

PLANS = {
    "TL_나무": {
        "dino_query": "tree crown. trunk. branch. root. leaves. cloud. moon. star. bird. small animal. flower.",
        "checklist": "수관, 기둥, 가지, 뿌리, 나뭇잎, 열매, 꽃, 구름, 달, 별, 새, 다람쥐, 그네",
    },
    "TL_남자사람": {
        "dino_query": "head. hair. face. eyes. nose. mouth. ears. neck. body. arms. hands. legs. feet. buttons. pockets. shoes.",
        "checklist": "머리, 머리카락, 얼굴, 눈, 코, 입, 귀, 목, 상체, 팔, 손, 다리, 발, 단추, 주머니, 신발",
    },
    "TL_여자사람": {
        "dino_query": "head. hair. face. eyes. nose. mouth. ears. neck. body. arms. hands. legs. feet. buttons. shoes. skirt.",
        "checklist": "머리, 머리카락, 얼굴, 눈, 코, 입, 귀, 목, 상체, 팔, 손, 다리, 발, 단추, 주머니, 신발",
    },
    "TL_집": {
        "dino_query": "house roof. house wall. door. window. chimney. smoke. path. fence. mountain. sun. tree. flower. pond.",
        "checklist": "지붕, 집벽, 문, 창문, 굴뚝, 연기, 길, 울타리, 산, 해, 나무, 꽃, 연못, 풀",
    },
}

TEST_IMAGES = [
    ("TL_나무", "나무_12_여_06908.jpg"),
    ("TL_나무", "나무_13_남_09955.jpg"),
    ("TL_나무", "나무_8_남_09618.jpg"),
    ("TL_나무", "나무_11_여_08538.jpg"),
    ("TL_남자사람", "남자사람_9_남_05962.jpg"),
    ("TL_남자사람", "남자사람_13_남_00712.jpg"),
    ("TL_남자사람", "남자사람_11_여_02562.jpg"),
    ("TL_남자사람", "남자사람_12_남_05052.jpg"),
    ("TL_여자사람", "여자사람_13_남_11853.jpg"),
    ("TL_여자사람", "여자사람_10_여_09489.jpg"),
    ("TL_여자사람", "여자사람_11_여_07408.jpg"),
    ("TL_여자사람", "여자사람_10_남_08555.jpg"),
    ("TL_집", "집_13_여_09914.jpg"),
    ("TL_집", "집_8_남_06189.jpg"),
    ("TL_집", "집_8_남_01735.jpg"),
    ("TL_집", "집_10_여_06585.jpg"),
]

def normalize(name):
    name = name.strip().replace(" ", "").lower()
    for key, vals in SYNONYMS.items():
        if name == key.lower(): return key
        for v in vals:
            if name == v.lower().replace(" ", ""): return key
    for key, vals in SYNONYMS.items():
        if key.lower() in name or name in key.lower(): return key
        for v in vals:
            v_clean = v.lower().replace(" ", "")
            if v_clean in name or name in v_clean: return key
    return name

def get_gt(cat, img_name):
    json_path = os.path.join(TRAIN_LABEL, cat, img_name.replace(".jpg", ".json"))
    if not os.path.exists(json_path): return []
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    labels = set()
    for b in data.get("annotations", {}).get("bbox", []):
        if "label" in b: labels.add(b["label"])
    return list(labels)

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def call_gemini(image, prompt):
    for attempt in range(3):
        try:
            m = genai.GenerativeModel("gemini-3.1-pro-preview")
            resp = m.generate_content([prompt, image])
            if not resp.candidates: raise RuntimeError("빈 응답")
            text = resp.candidates[0].content.parts[0].text.strip()
            if len(text) < 5 or "retry_delay" in text: raise RuntimeError(f"비정상: {text[:50]}")
            return text
        except Exception as e:
            print(f"    gemini 에러 ({attempt+1}): {str(e)[:100]}", flush=True)
            time.sleep(3)
    return "Error"

def call_gpt(img_path, prompt):
    for attempt in range(3):
        try:
            b64 = encode_image(img_path)
            resp = gpt_client.chat.completions.create(
                model="gpt-5.4", max_completion_tokens=6000,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]}])
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"    gpt 에러 ({attempt+1}): {str(e)[:100]}", flush=True)
            time.sleep(3)
    return "Error"

def call_grok(img_path, prompt):
    for attempt in range(3):
        try:
            b64 = encode_image(img_path)
            resp = grok_client.chat.completions.create(
                model="grok-4.20-beta-0309-reasoning", max_tokens=4000,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]}])
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"    grok 에러 ({attempt+1}): {str(e)[:100]}", flush=True)
            time.sleep(3)
    return "Error"

def call_claude(img_path, prompt):
    for attempt in range(3):
        try:
            b64 = encode_image(img_path)
            resp = openrouter_client.chat.completions.create(
                model="anthropic/claude-sonnet-4-6", max_tokens=4000,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]}])
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"    claude 에러 ({attempt+1}): {str(e)[:100]}", flush=True)
            time.sleep(3)
    return "Error"

def run_dino(image, query, processor, model, threshold=0.2):
    inputs = processor(images=image, text=query, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=threshold, text_threshold=threshold,
        target_sizes=[image.size[::-1]]
    )[0]
    seen = {}
    for label, score, box in zip(results["labels"], results["scores"], results["boxes"]):
        first = label.split()[0]
        s = round(score.item(), 3)
        if first not in seen or s > seen[first]["score"]:
            seen[first] = {"label": label, "score": s, "box": [round(b.item()) for b in box]}
    return list(seen.values())

def parse_json_arr(text):
    m = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', text, re.DOTALL)
    if m:
        try: return json.loads(m.group(1))
        except: pass
    m = re.search(r'\[.*\]', text, re.DOTALL)
    if m:
        try: return json.loads(m.group(0))
        except: pass
    return None

def format_dino(detections, w, h):
    lines = []
    for d in detections:
        box = d["box"]
        x = round((box[0] + box[2]) / 2 / w * 100)
        y = round((box[1] + box[3]) / 2 / h * 100)
        xp = "왼쪽" if x < 33 else ("중앙" if x < 67 else "오른쪽")
        yp = "상단" if y < 33 else ("중단" if y < 67 else "하단")
        lines.append(f"  - '{d['label']}' ({d['score']}) — {yp} {xp} (x{x}%, y{y}%)")
    return "\n".join(lines)

def process_image(cat, img_name, processor, dino_model):
    img_path = os.path.join(TRAIN_ORIGIN, CAT_MAP[cat], img_name)
    image = Image.open(img_path).convert("RGB")
    gt = get_gt(cat, img_name)
    w, h = image.size
    plan = PLANS[cat]

    # Step 1: DINO (CPU, 빠름)
    detections = run_dino(image, plan["dino_query"], processor, dino_model)
    dino_text = format_dino(detections, w, h)

    # Step 2: 4 VLM 병렬 호출
    analysis_prompt = (
        f"이 HTP 심리검사 그림을 분석해주세요.\n\n"
        f"[DINO 탐지 결과]:\n{dino_text}\n\n"
        f"[체크리스트]: {plan['checklist']}\n\n"
        f"발견한 모든 객체에 대해 **세세한 속성**을 기록해주세요.\n"
        f"JSON 배열로 답해주세요:\n"
        f"```json\n[\n  {{\n"
        f"    \"객체\": \"수관\",\n"
        f"    \"판정\": \"있음\",\n"
        f"    \"위치\": \"상단 중앙 (x50%, y20%)\",\n"
        f"    \"형태\": \"둥근 곡선\",\n"
        f"    \"크기\": \"이미지 너비의 약 30%\",\n"
        f"    \"배치\": \"기둥 바로 위\",\n"
        f"    \"선_특성\": \"진한 연속선\",\n"
        f"    \"주변_맥락\": \"기둥과 연결\",\n"
        f"    \"확신도\": 0.85,\n"
        f"    \"근거\": \"기둥 위의 큰 둥근 형태\"\n"
        f"  }}\n]\n```\n"
        f"체크리스트에 있지만 그림에 없는 것은 \"판정\": \"없음\"으로 포함.\n"
        f"반드시 JSON 형식으로만."
    )

    t0 = time.time()
    models_data = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {
            ex.submit(call_gemini, image, analysis_prompt): "gemini",
            ex.submit(call_gpt, img_path, analysis_prompt): "gpt",
            ex.submit(call_grok, img_path, analysis_prompt): "grok",
            ex.submit(call_claude, img_path, analysis_prompt): "claude",
        }
        for fut in as_completed(futs):
            model_name = futs[fut]
            raw = fut.result()
            parsed = parse_json_arr(raw)
            n = len(parsed) if parsed else 0
            print(f"    [{model_name}] {n}개 객체", flush=True)
            models_data[model_name] = {"raw": raw, "parsed": parsed}
    elapsed = time.time() - t0
    print(f"  4모델 병렬 완료: {elapsed:.0f}초", flush=True)

    # Step 3: 합의/불일치 분류
    all_objects = {}
    for model_name, data in models_data.items():
        parsed = data["parsed"]
        if not parsed: continue
        for entry in parsed:
            obj = entry.get("객체", "")
            판정 = entry.get("판정", "있음")
            if "없음" in str(판정): continue
            obj_norm = normalize(obj)
            if obj_norm not in all_objects:
                all_objects[obj_norm] = {}
            all_objects[obj_norm][model_name] = {
                "위치": entry.get("위치", ""),
                "형태": entry.get("형태", ""),
                "크기": entry.get("크기", ""),
                "배치": entry.get("배치", ""),
                "선_특성": entry.get("선_특성", ""),
                "주변_맥락": entry.get("주변_맥락", ""),
                "확신도": entry.get("확신도", 0),
                "근거": entry.get("근거", ""),
            }

    agreed, disagreed = [], []
    for obj, md in all_objects.items():
        count = len(md)
        if count >= 3:
            confs = [d["확신도"] for d in md.values() if isinstance(d["확신도"], (int, float))]
            avg_conf = sum(confs) / len(confs) if confs else 0
            agreed.append({"객체": obj, "동의수": count, "확신도": round(avg_conf, 2)})
        else:
            disagreed.append({"객체": obj, "동의수": count, "모델": md})

    return {
        "category": cat,
        "image": img_name,
        "gt": gt,
        "dino_text": dino_text,
        "dino_detections": detections,
        "models_data_parsed": {k: v["parsed"] for k, v in models_data.items()},
        "agreed": agreed,
        "disagreed": disagreed,
    }

def main():
    print("DINO 로딩...", flush=True)
    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")
    print("완료!\n", flush=True)

    all_states = []
    t_start = time.time()

    for idx, (cat, img_name) in enumerate(TEST_IMAGES, 1):
        print(f"\n{'='*60}", flush=True)
        print(f"[{idx}/16] {img_name}", flush=True)
        state = process_image(cat, img_name, processor, dino_model)
        print(f"  합의: {len(state['agreed'])}개, 불일치: {len(state['disagreed'])}개", flush=True)
        all_states.append(state)
        with open("v3_state_all.json", "w", encoding="utf-8") as f:
            json.dump(all_states, f, ensure_ascii=False, indent=2, default=str)
        elapsed = time.time() - t_start
        print(f"  누적 {elapsed/60:.1f}분, 저장 완료 ({len(all_states)}/16)", flush=True)

    print(f"\n{'='*60}")
    print(f"  완료! 총 {len(all_states)}장, {(time.time()-t_start)/60:.1f}분 소요")

if __name__ == "__main__":
    main()
