"""Phase 1 P4 — Bbox+Voting+R2+Judge IoU=0.3 × 24장 (paper 정합)."""
import sys
sys.path.insert(0, '/Users/kg/nonmoon/htp_thesis')
from test24_phase1_bboxunion_v2 import TEST_24

import test12_voting_judge
test12_voting_judge.TEST_12 = TEST_24

import test12_bbox_voting_r2 as mod
mod.TEST_12 = TEST_24
sys.argv = ['test12_bbox_voting_r2.py', '0.3']
mod.IOU_THRESH = 0.3
mod.NAME = "Bbox+Voting+R2+Judge IoU=0.3 24장"
mod.OUTFILE = "/Users/kg/nonmoon/htp_thesis/phase1_p4_voting_r2_24.json"

mod.main()
