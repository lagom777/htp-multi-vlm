"""Phase 3 — BboxUnion+R2+Judge × 64장."""
import sys
sys.path.insert(0, '/Users/kg/nonmoon/htp_thesis')
from test_64_files import TEST_64

import test12_voting_judge
test12_voting_judge.TEST_12 = TEST_64

import test12_bboxunion_r2_judge as mod
mod.TEST_12 = TEST_64
mod.OUTFILE = "/Users/kg/nonmoon/htp_thesis/phase3_bboxunion_v2_64.json"

mod.main()
