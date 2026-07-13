"""Phase 1 P1 — BboxUnion+R2+Judge (v2 SizeHint) × 24장."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'framework'))

# TEST 24장
TEST_24 = [
    ("TL_나무","나무_8_남_01445.jpg"),("TL_나무","나무_10_여_00019.jpg"),
    ("TL_나무","나무_11_남_00004.jpg"),("TL_나무","나무_12_여_00007.jpg"),
    ("TL_나무","나무_8_여_00041.jpg"),("TL_나무","나무_10_남_00013.jpg"),
    ("TL_집","집_12_여_08971.jpg"),("TL_집","집_10_여_00006.jpg"),
    ("TL_집","집_11_남_00013.jpg"),("TL_집","집_12_여_00007.jpg"),
    ("TL_집","집_8_여_00066.jpg"),("TL_집","집_10_남_00015.jpg"),
    ("TL_남자사람","남자사람_13_남_02804.jpg"),("TL_남자사람","남자사람_10_여_00023.jpg"),
    ("TL_남자사람","남자사람_11_남_00000.jpg"),("TL_남자사람","남자사람_12_여_00005.jpg"),
    ("TL_남자사람","남자사람_8_여_00016.jpg"),("TL_남자사람","남자사람_10_남_00022.jpg"),
    ("TL_여자사람","여자사람_10_남_02125.jpg"),("TL_여자사람","여자사람_10_여_00018.jpg"),
    ("TL_여자사람","여자사람_11_남_00002.jpg"),("TL_여자사람","여자사람_12_여_00008.jpg"),
    ("TL_여자사람","여자사람_8_여_00081.jpg"),("TL_여자사람","여자사람_10_남_00010.jpg"),
]

# Monkey-patch import 시 사용할 TEST_12를 TEST_24로 override
import test12_voting_judge
test12_voting_judge.TEST_12 = TEST_24

import test12_bboxunion_r2_judge as mod
mod.TEST_12 = TEST_24
mod.OUTFILE = "./phase1_p1_bboxunion_v2_24.json"

# main 호출
mod.main()
