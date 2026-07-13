"""Phase 1 P2 — BboxUnion+Zoom-R2+Judge × 24장."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'framework'))
from test24_phase1_bboxunion_v2 import TEST_24

import test12_voting_judge
test12_voting_judge.TEST_12 = TEST_24

import test12_zoom_r2 as mod
mod.TEST_12 = TEST_24
mod.OUTFILE = "./phase1_p2_zoom_r2_24.json"

mod.main()
