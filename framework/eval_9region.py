"""
9-region position 평가 파이프라인.

GT bbox 중심 → 9-region 라벨
VLM 텍스트 위치 → 9-region 라벨 (파서)
비교해서 Acc / F1 / FP 계산.
"""

import json
import os
import re
from collections import defaultdict, Counter

# ============================================================
# 9-region 정의
# ============================================================
# row × col grid (한국어 키워드)
#
#   ┌───────┬───────┬───────┐
#   │ TL    │  TC   │  TR   │     (상단)
#   ├───────┼───────┼───────┤
#   │ ML    │  MC   │  MR   │     (중단)
#   ├───────┼───────┼───────┤
#   │ BL    │  BC   │  BR   │     (하단)
#   └───────┴───────┴───────┘

ROWS = ['top', 'middle', 'bottom']
COLS = ['left', 'center', 'right']

# 한국어 → row/col 매핑
ROW_KW = {
    'top':    ['상단', '상부', '위', '맨위', '상측', '꼭대기', '윗부분', '윗쪽', '상',  '하늘'],
    'middle': ['중단', '중부', '중간', '중앙', '가운데', '중',  '중심'],
    'bottom': ['하단', '하부', '아래', '맨아래', '하측', '바닥', '아랫부분', '아랫쪽', '하', '지면'],
}
COL_KW = {
    'left':   ['좌측', '왼쪽', '좌', '왼편', '좌단'],
    'center': ['중앙', '가운데', '중심', '중', '중부'],
    'right':  ['우측', '오른쪽', '우', '오른편', '우단'],
}


def bbox_to_region(x, y, w, h, img_w=1280, img_h=1280):
    """GT bbox 중심점을 9-region 라벨로 변환."""
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    col = 'left' if cx < 1/3 else ('right' if cx > 2/3 else 'center')
    row = 'top' if cy < 1/3 else ('bottom' if cy > 2/3 else 'middle')
    return f'{row}-{col}'


def bbox_to_quadrant(x, y, w, h, img_w=1280, img_h=1280):
    """GT bbox 중심점을 4분면 라벨로 변환 (Buck 임상 전통)."""
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    col = 'left' if cx < 0.5 else 'right'
    row = 'top' if cy < 0.5 else 'bottom'
    return f'{row}-{col}'


def region_to_quadrant(region):
    """9-region 라벨을 4분면 라벨로 변환.

    middle-* 와 *-center는 어느 사분면? — 보수적으로 더 가까운 방향 사용.
    중앙 그대로면 None 반환 (사분면 미정).
    """
    if not region:
        return None
    parts = region.split('-')
    if len(parts) != 2:
        return None
    row, col = parts
    new_row = 'top' if row == 'top' else ('bottom' if row == 'bottom' else None)
    new_col = 'left' if col == 'left' else ('right' if col == 'right' else None)
    if new_row and new_col:
        return f'{new_row}-{new_col}'
    return None  # 중앙 셀은 사분면 미정


def parse_vlm_position(text):
    """VLM 텍스트 위치 답을 (9-region, 4-quadrant) 튜플로 변환.

    우선순위: 좌표 표기 (x50%, y20%) > 키워드
    실패 시 (None, None)
    """
    if not text:
        return (None, None)
    s = text.strip()

    # 1. 좌표 형식 (가장 정확)
    m_x = re.search(r'x\s*[=:]?\s*(\d+\.?\d*)\s*%', s, re.IGNORECASE)
    m_y = re.search(r'y\s*[=:]?\s*(\d+\.?\d*)\s*%', s, re.IGNORECASE)
    if m_x and m_y:
        cx = float(m_x.group(1)) / 100
        cy = float(m_y.group(1)) / 100
        # 9-region (3분할)
        col9 = 'left' if cx < 1/3 else ('right' if cx > 2/3 else 'center')
        row9 = 'top' if cy < 1/3 else ('bottom' if cy > 2/3 else 'middle')
        # 4-quadrant (2분할, 0.5 기준)
        col4 = 'left' if cx < 0.5 else 'right'
        row4 = 'top' if cy < 0.5 else 'bottom'
        return (f'{row9}-{col9}', f'{row4}-{col4}')

    # 2. 키워드 매칭
    row_found, col_found = None, None
    for r, kws in ROW_KW.items():
        for kw in kws:
            if kw in s:
                row_found = r
                break
        if row_found:
            break
    for c, kws in COL_KW.items():
        for kw in kws:
            if kw in s:
                col_found = c
                break
        if col_found:
            break

    if row_found or col_found:
        r9 = (row_found or 'middle') + '-' + (col_found or 'center')
        # 4-quadrant: 중앙은 미정
        if row_found in ('top', 'bottom') and col_found in ('left', 'right'):
            r4 = f'{row_found}-{col_found}'
        else:
            r4 = None
        return (r9, r4)

    return (None, None)


# ============================================================
# 유의어 정규화
# ============================================================
SYNONYMS = {
    '해': ['태양', 'sun', '햇빛'],
    '풀': ['잔디', 'grass', '풀밭'],
    '신발': ['운동화', '구두', '남자구두', '여자구두', 'shoe', 'sneaker'],
    '머리': ['head', '두상'],
    '얼굴': ['face'],
    '눈': ['eye', 'eyes'],
    '코': ['nose'],
    '입': ['mouth'],
    '귀': ['ear', 'ears'],
    '목': ['neck'],
    '상체': ['몸통', 'body', '상의', '티셔츠'],
    '팔': ['arm', 'arms', '소매'],
    '손': ['hand', 'hands'],
    '다리': ['leg', 'legs'],
    '발': ['foot', 'feet'],
    '단추': ['button', 'buttons'],
    '주머니': ['pocket', 'pockets'],
    '머리카락': ['hair', '앞머리', '땋은머리'],
    '수관': ['나무수관', 'canopy', 'treecrown', '잎부분'],
    '기둥': ['나무줄기', '줄기', 'trunk', '나무기둥'],
    '가지': ['나뭇가지', 'branch'],
    '뿌리': ['나무뿌리', 'root'],
    '나뭇잎': ['잎', 'leaf', 'leaves', '잎사귀'],
    '열매': ['fruit', '사과', '도토리'],
    '꽃': ['flower'],
    '구름': ['cloud'],
    '달': ['moon'],
    '별': ['star'],
    '새': ['bird'],
    '다람쥐': ['squirrel'],
    '그네': ['swing'],
    '지붕': ['roof'],
    '집벽': ['벽', 'wall', '외벽'],
    '문': ['door'],
    '창문': ['window'],
    '굴뚝': ['chimney'],
    '연기': ['smoke'],
    '길': ['path', 'road'],
    '울타리': ['fence', '담장'],
    '산': ['mountain', '언덕'],
    '연못': ['pond'],
    '나무': ['tree'],
    '나무전체': ['tree'],
    '집전체': ['house', 'building'],
    '사람전체': ['person'],
}


def normalize_label(name):
    """객체 이름을 정규 라벨로 매핑."""
    if not name:
        return name
    s = name.strip().lower().replace(' ', '')
    for canonical, aliases in SYNONYMS.items():
        if s == canonical.lower():
            return canonical
        for a in aliases:
            if s == a.lower():
                return canonical
        # 부분 매칭
        if canonical.lower() in s or s in canonical.lower():
            return canonical
    return name  # 매칭 실패 시 원래 이름


# ============================================================
# GT 처리 — 카테고리별 객체별 region 라벨
# ============================================================
def gt_objects_with_region(gt_json_path):
    """GT JSON에서 객체별 region을 추출.

    Returns: dict[label] = list of (9-region, 4-quadrant) 튜플
    """
    with open(gt_json_path) as f:
        d = json.load(f)
    res_w, res_h = 1280, 1280
    if 'meta' in d and 'img_resolution' in d['meta']:
        wh = d['meta']['img_resolution'].split('x')
        if len(wh) == 2:
            res_w, res_h = int(wh[0]), int(wh[1])

    by_label = defaultdict(list)
    for bb in d['annotations']['bbox']:
        lbl = normalize_label(bb['label'])
        r9 = bbox_to_region(bb['x'], bb['y'], bb['w'], bb['h'], res_w, res_h)
        r4 = bbox_to_quadrant(bb['x'], bb['y'], bb['w'], bb['h'], res_w, res_h)
        by_label[lbl].append((r9, r4))
    return dict(by_label)


# ============================================================
# 평가 — VLM 답 vs GT
# ============================================================
def evaluate_image(vlm_parsed, gt_by_label, canonical_set):
    """한 이미지에 대해 한 모델의 답을 GT와 비교.

    9-region과 4-quadrant 둘 다 평가.
    Returns: dict with TP, FP, FN, pos9_correct, pos4_correct, totals, hallucinations
    """
    fp_hall = 0
    pos9_correct = 0
    pos4_correct = 0
    pos_total = 0  # 9-region 평가 대상
    pos4_total = 0  # 4-quadrant 평가 대상 (중앙 셀은 제외)
    detail = []

    found_labels = set()
    for item in vlm_parsed or []:
        if item.get('판정') != '있음':
            continue
        raw_lbl = item.get('객체', '')
        norm = normalize_label(raw_lbl)

        if norm not in canonical_set:
            fp_hall += 1
            detail.append({'type': 'fp_hall', 'label': raw_lbl, 'norm': norm})
            continue

        found_labels.add(norm)
        pos_total += 1
        vlm_r9, vlm_r4 = parse_vlm_position(item.get('위치', ''))
        gt_pairs = gt_by_label.get(norm, [])

        if vlm_r9 and gt_pairs:
            gt_r9_list = [p[0] for p in gt_pairs]
            gt_r4_list = [p[1] for p in gt_pairs]

            r9_ok = vlm_r9 in gt_r9_list
            if r9_ok:
                pos9_correct += 1

            if vlm_r4:
                pos4_total += 1
                r4_ok = vlm_r4 in gt_r4_list
                if r4_ok: pos4_correct += 1
            else:
                r4_ok = None

            detail.append({
                'type': 'pos',
                'label': norm,
                'vlm9': vlm_r9, 'vlm4': vlm_r4,
                'gt9': gt_r9_list, 'gt4': gt_r4_list,
                'r9_ok': r9_ok, 'r4_ok': r4_ok,
            })
        else:
            detail.append({'type': 'pos_unparseable', 'label': norm,
                           'vlm_text': item.get('위치', '')})

    gt_labels = set(gt_by_label.keys()) & canonical_set
    tp = len(found_labels & gt_labels)
    fn = len(gt_labels - found_labels)

    return {
        'tp': tp,
        'fn': fn,
        'fp_hall': fp_hall,
        'pos9_correct': pos9_correct,
        'pos4_correct': pos4_correct,
        'pos_total': pos_total,
        'pos4_total': pos4_total,
        'detail': detail,
    }


# ============================================================
# 카테고리별 canonical 셋
# ============================================================
CANONICAL = {
    'TL_나무': {'수관','기둥','가지','뿌리','나뭇잎','열매','꽃','구름','달','별','새','다람쥐','그네','나무전체'},
    'TL_집':   {'지붕','집벽','문','창문','굴뚝','연기','길','울타리','산','해','나무','꽃','연못','풀','집전체'},
    'TL_남자사람': {'머리','머리카락','얼굴','눈','코','입','귀','목','상체','팔','손','다리','발','단추','주머니','신발','사람전체'},
    'TL_여자사람': {'머리','머리카락','얼굴','눈','코','입','귀','목','상체','팔','손','다리','발','단추','주머니','신발','사람전체'},
}


# ============================================================
# 메인
# ============================================================
def main():
    # 8장 선정 — 카테고리별 첫 2장
    with open('/Users/kg/nonmoon/htp_thesis/v3_state_3vlm_distractor_16.json') as f:
        all_data = json.load(f)

    by_cat = defaultdict(list)
    for it in all_data:
        by_cat[it['category']].append(it)

    selected = []
    for cat in ['TL_나무', 'TL_집', 'TL_남자사람', 'TL_여자사람']:
        selected.extend(by_cat[cat][:2])

    # GT region 라벨 (각 이미지)
    GT_BASE = "/Users/kg/nonmoon/htp_thesis/266.AI 기반 아동 미술심리 진단을 위한 그림 데이터 구축/01-1.정식개방데이터/Training/02.라벨링데이터"

    results = []
    for item in selected:
        cat = item['category']
        img = item['image']
        gt_path = os.path.join(GT_BASE, cat, img.replace('.jpg', '.json'))
        gt_by_label = gt_objects_with_region(gt_path)
        canon = CANONICAL[cat]

        per_model = {}
        for model in ['gemini', 'gpt', 'claude']:
            parsed = item['models'].get(model, {}).get('parsed')
            r = evaluate_image(parsed, gt_by_label, canon)
            per_model[model] = r

        results.append({
            'category': cat,
            'image': img,
            'gt_labels': sorted(gt_by_label.keys()),
            'gt_regions': {k: v for k, v in gt_by_label.items()},
            'per_model': per_model,
        })

    # 요약 출력
    print('=' * 80)
    print('9-region 위치 평가 결과 — 8장 (카테고리별 2장)')
    print('=' * 80)

    overall = {'gemini': Counter(), 'gpt': Counter(), 'claude': Counter()}
    for r in results:
        print(f'\n[{r["category"]}] {r["image"]}')
        for m, met in r['per_model'].items():
            tp, fn, fph = met['tp'], met['fn'], met['fp_hall']
            pc, pt = met['pos_correct'], met['pos_total']
            acc_id = tp / (tp + fn) if (tp + fn) else 0
            acc_pos = pc / pt if pt else 0
            print(f'  {m:8s}: 식별 {tp}/{tp+fn} (Recall={acc_id:.1%}) | '
                  f'위치 {pc}/{pt} (Acc={acc_pos:.1%}) | 환각 {fph}')
            for k in ('tp','fn','fp_hall','pos_correct','pos_total'):
                overall[m][k] += met[k]

    print()
    print('=' * 80)
    print('전체 합산')
    print('=' * 80)
    print(f'{"모델":8s} {"식별 Recall":>15s} {"위치 Acc":>15s} {"환각/이미지":>15s}')
    print('-' * 60)
    for m in ['gemini', 'gpt', 'claude']:
        o = overall[m]
        rec = o['tp'] / (o['tp'] + o['fn']) if (o['tp'] + o['fn']) else 0
        pacc = o['pos_correct'] / o['pos_total'] if o['pos_total'] else 0
        print(f'{m:8s} {rec:>14.1%}  {pacc:>14.1%}  {o["fp_hall"]/8:>14.2f}')

    # 상세 결과 저장
    with open('/Users/kg/nonmoon/htp_thesis/eval_9region_8images_results.json', 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('\n상세 결과 → eval_9region_8images_results.json')


if __name__ == '__main__':
    main()
