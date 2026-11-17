"""Synthetic tests for src/reglas.py (no real data)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
import reglas  # noqa: E402


def mk(pattern, per=None):
    """pattern: string of 'H','A' (ball visible, nearest team) or '.' (ball absent)."""
    n = len(pattern)
    ball = np.full((n, 2), np.nan)
    H = np.full((n, 24, 2), np.nan)
    A = np.full((n, 22, 2), np.nan)
    for i, c in enumerate(pattern):
        if c == ".":
            H[i, 0] = (0.9, 0.9)
            A[i, 0] = (0.1, 0.1)
            continue
        ball[i] = (0.5, 0.5)
        near, far = (H, A) if c == "H" else (A, H)
        near[i, 0] = (0.51, 0.5)
        far[i, 0] = (0.9, 0.9)
    period = np.ones(n, int) if per is None else np.asarray(per)
    return ball, {"H": H, "A": A}, period


PITCH = (100.0, 100.0)


def run(variant, pattern, per=None, **kw):
    b, p, pe = mk(pattern, per)
    return "".join(reglas.predict(variant, b, p, pe, PITCH, window=0, **kw))


def test_nearest_codes():
    b, p, _ = mk("HA.")
    assert list(reglas.nearest_codes(b, p, PITCH)) == [reglas.HOME, reglas.AWAY, reglas.NOBODY]


def test_gap_runs_split_by_period():
    present = np.array([1, 0, 0, 0, 0, 1, 0], bool)
    period = np.array([1, 1, 1, 2, 2, 2, 2])
    r = reglas.gap_runs(present, period)
    assert [(x["start"], x["end"], x["pre_idx"], x["post_idx"]) for x in r] == [
        (1, 3, 0, None), (3, 5, None, 5), (6, 7, 5, None)]


def test_same_owner_all_variants():
    for v in ("F", "T_back", "T_comb_D", "T_comb_mid"):
        assert run(v, "HH...HH") == "HHHHHHH"


def test_discrepant_owner():
    assert run("F", "HH....AA") == "HHHHHHAA"
    assert run("T_back", "HH....AA") == "HHAAAAAA"
    assert run("T_comb_D", "HH....AA") == "HHDDDDAA"
    assert run("T_comb_mid", "HH....AA") == "HHHHAAAA"


def test_period_border():
    per = [1, 1, 1, 2, 2, 2]
    # gap at the end of period 1: no post -> pre; gap at start of period 2: no pre -> post
    assert run("T_back", "HH.A.A", per) == "HHHAAA"
    assert run("F", "HH.A.A", per) == "HHHAAA"
    assert run("T_comb_D", "HH.A.A", per) == "HHHAAA"
    # a gap does not cross the period change
    assert run("T_comb_D", "H..A", [1, 1, 2, 2]) == "HHAA"
    # no owner at all -> D
    assert run("F", "...") == "DDD"


def test_max_gap():
    s = "H" + "." * 50 + "H"          # 50 frames = 2 s
    assert run("F", s, max_gap_s=5) == "H" * 52
    assert run("F", s, max_gap_s=1) == "H" + "D" * 50 + "H"
    assert run("T_comb_mid", s, max_gap_s=1) == "H" + "D" * 50 + "H"


def test_b0_has_no_filling():
    assert run("B0", "HH..AA") == "HHDDAA"


def test_smooth_does_not_cross_periods():
    # 10 A frames end period 1, 10 H frames start period 2; a window of 50 spans both.
    b, p, pe = mk("A" * 10 + "H" * 10, per=[1] * 10 + [2] * 10)
    for v in ("B0", "F"):
        out = "".join(reglas.predict(v, b, p, pe, PITCH, window=50))
        assert out == "A" * 10 + "H" * 10, (v, out)
