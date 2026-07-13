"""Phase 2 — BboxUnion+R2+Judge (v2) × 32장."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'framework'))
from g_prime import TEST_IMAGES_FULL

TEST_32 = TEST_IMAGES_FULL[:32]  # 32장

import test12_voting_judge
test12_voting_judge.TEST_12 = TEST_32

import test12_bboxunion_r2_judge as mod
mod.TEST_12 = TEST_32
mod.OUTFILE = "./phase2_bboxunion_v2_32.json"

mod.main()
