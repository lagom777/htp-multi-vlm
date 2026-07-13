"""Phase 1 P3 — Multi-Region 좌표 텍스트 × 24장."""
import sys
sys.path.insert(0, '/Users/kg/nonmoon/htp_thesis')
from test24_phase1_bboxunion_v2 import TEST_24

import test12_voting_judge
test12_voting_judge.TEST_12 = TEST_24

import test12_multiregion_text as mod
mod.TEST_12 = TEST_24
mod.OUTFILE = "/Users/kg/nonmoon/htp_thesis/phase1_p3_multiregion_24.json"

mod.main()
