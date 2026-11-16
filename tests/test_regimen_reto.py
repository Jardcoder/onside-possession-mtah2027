"""bancos/regimen_reto.py reads the challenge tracking only: no label, ground truth or Smart Tagging field is touched."""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("bancos", "src"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import regimen_reto as rr  # noqa: E402

FORBIDDEN = re.compile(r"label|ball_status|is_touch|uncertain|video_missing|ball_visible|ball_z|players_visible|attack_right"
                       r"|\bgt\b|_gt\b|gt_|ground.?truth|smart|tagging|possession_metrics|possession_gt|make_gt|clip_periods"
                       r"|read_label|event_f", re.I)


def _code_without_docstring():
    src = open(os.path.join(ROOT, "bancos", "regimen_reto.py")).read()
    return re.sub(r'^""".*?"""', "", src, count=1, flags=re.S)


def test_script_reads_no_labels_or_ground_truth():
    hit = FORBIDDEN.search(_code_without_docstring())
    assert hit is None, hit and hit.group(0)


def test_docstring_declares_the_rule_and_script_has_no_file_reads():
    code = _code_without_docstring()
    assert "open(" not in code.replace('open(OUT, "w")', "")
    assert "read_csv" not in code and "np.load" not in code


def test_owner_changes_and_possession_lengths():
    per = np.array([1] * 6 + [2] * 3)
    pred = np.array(list("HHDAAA") + list("AAH"))
    m = np.ones(9, bool)
    assert rr.owner_changes(pred, per, m) == 1 + 1                 # H->A in period 1 (D ignored), A->H in period 2
    assert sorted(rr.possession_lengths(pred, per, m)) == sorted(np.array([2, 3, 2, 1]) / rr.FPS)
