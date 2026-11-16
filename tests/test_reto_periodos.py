"""src/reto_periodos.py: the challenge periods follow the PFF layout (kick-off to last frame, no gaps, clock from the kick-off)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("src",):
    sys.path.insert(0, os.path.join(ROOT, sub))
import modelo as mo  # noqa: E402
import reto_periodos as rpd  # noqa: E402


def test_blocks_match_the_published_frames():
    assert rpd.blocks() == [(1, 11185, 90651), (2, 113742, 190197)]
    assert rpd.pre_kickoff_2nd() == (110808, 113741)
    assert rpd.pre_kickoff_2nd()[1] - rpd.pre_kickoff_2nd()[0] + 1 == 2934


def test_period_array_and_counts():
    per = rpd.periods()
    assert len(per) == 199319
    assert (per == 1).sum() == 79467 and (per == 2).sum() == 76456
    assert (per == 0).sum() == 40462 + 2934                                    # not graded + graded-but-before-kick-off
    f = rpd.play_frames()
    assert len(f) == 79467 + 76456 and f[0] == 11185 and f[-1] == 190197 and (np.diff(f) >= 1).all()
    assert per[11184] == 0 and per[11185] == 1 and per[90651] == 1 and per[90652] == 0 and per[113741] == 0 and per[113742] == 2 and per[190198] == 0


def test_features_clock_starts_at_kickoff_of_each_period():
    f = rpd.play_frames()
    per = rpd.periods()[f]
    n = len(f)
    nan = np.full((n, 2), np.nan)
    X, names = mo.build_features(nan, {"H": np.full((n, 2, 2), np.nan), "A": np.full((n, 2, 2), np.nan)}, per, (105.0, 68.0),
                                 np.array(["D"] * n), np.array(["D"] * n), np.array([], int))
    t_in, t_end = X[:, names.index("t_in_period_s")], X[:, names.index("t_to_period_end_s")]
    i2 = int(np.flatnonzero(per == 2)[0])
    assert t_in[0] == 0 and t_in[i2] == 0 and t_end[i2 - 1] == 0 and t_end[-1] == 0
    assert abs(t_in[i2 - 1] - (90651 - 11185) / 25) < 1e-3 and abs(t_in[-1] - (190197 - 113742) / 25) < 1e-3


def test_to_full_round_trip():
    f = rpd.play_frames()
    v = np.array(["H"] * len(f))
    full = rpd.to_full(v, f, rpd.n_frames())
    assert len(full) == rpd.n_frames() and (full[f] == "H").all() and (full[rpd.periods() == 0] == "X").all()
