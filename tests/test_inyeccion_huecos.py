"""Variant h of the PFF bank (bancos/inyeccion_huecos.py): holes only inside live play, deterministic, per period, existing NaN untouched."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("bancos", "src"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import inyeccion_huecos as ih  # noqa: E402


def draw_short(rng):
    return int(rng.integers(40, 160))


def toy(seed=0):
    """Two periods of 12000 frames: live runs of 1500 and dead runs of 500; the ball is missing on 80 % of the dead frames and on two live stretches."""
    rng = np.random.default_rng(seed)
    n = 24000
    per = np.repeat([1, 2], n // 2)
    label = np.where((np.arange(n) % 2000) < 1500, "H", "D").astype("<U1")
    label[(np.arange(n) % 2000 < 1500) & (np.arange(n) // 2000 % 2 == 1)] = "A"
    ball = rng.random((n, 2))
    dead = label == "D"
    ball[dead & (rng.random(n) < 0.8)] = np.nan
    ball[300:340] = np.nan                                  # an existing gap in live play
    ball[13000:13050] = np.nan
    return ball, label, per


def runs(mask):
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def test_deterministic_per_match_and_seed_depends_on_the_match():
    ball, label, per = toy()
    a, ha = ih.inject_holes(ball, label, per, "3812", draw=draw_short)
    b, hb = ih.inject_holes(ball, label, per, "3812", draw=draw_short)
    assert np.array_equal(ha, hb) and np.array_equal(np.nan_to_num(a, nan=-1), np.nan_to_num(b, nan=-1))
    assert ha.any()
    _, hc = ih.inject_holes(ball, label, per, "3813", draw=draw_short)
    assert not np.array_equal(ha, hc)


def test_holes_open_only_on_live_frames_that_had_a_ball():
    ball, label, per = toy()
    for rho in (0.0, 0.5, 1.0):
        out, hole = ih.inject_holes(ball, label, per, "3812", draw=draw_short, anchor_p=rho)
        assert hole.any()
        assert (label[hole] != "D").all()
        assert np.isfinite(ball[hole, 0]).all()


def test_existing_nan_are_kept_and_nothing_else_changes():
    ball, label, per = toy()
    snap = ball.copy()
    out, hole = ih.inject_holes(ball, label, per, "3812", draw=draw_short)
    assert np.array_equal(ball, snap, equal_nan=True)                    # the input is not modified
    assert np.isnan(out[np.isnan(ball[:, 0]), 0]).all()                   # existing NaN stay NaN
    assert np.array_equal(np.isnan(out[:, 0]), np.isnan(ball[:, 0]) | hole)
    keep = ~hole
    assert np.array_equal(out[keep], ball[keep], equal_nan=True)


def test_holes_respect_the_periods():
    ball, label, per = toy()
    label[11000:13000] = "H"                                            # live play across the period break
    ball[11000:13000] = 0.5
    for rho in (0.0, 1.0):
        out, hole = ih.inject_holes(ball, label, per, "3812", draw=lambda rng: 900, anchor_p=rho, target=0.6)
        for a, b in runs(np.isnan(out[:, 0])):
            assert per[a] == per[b - 1], (a, b)                          # no gap of the result crosses the break
        for a, b in runs(hole):
            assert per[a] == per[b - 1]


def test_free_holes_keep_a_visible_frame_on_each_side_and_anchored_ones_touch_an_existing_gap():
    ball, label, per = toy()
    present0 = np.isfinite(ball[:, 0])
    out, hole = ih.inject_holes(ball, label, per, "3812", draw=draw_short, anchor_p=0.0)
    for a, b in runs(hole):
        assert present0[a - 1] and present0[b]
    out, hole = ih.inject_holes(ball, label, per, "3812", draw=draw_short, anchor_p=1.0)
    n = len(ball)
    for a, b in runs(hole):
        touches = (a > 0 and not present0[a - 1]) or (b < n and not present0[b])
        assert touches, (a, b)


def test_target_fraction_is_reached_and_a_match_above_it_gets_nothing():
    ball, label, per = toy()
    out, hole = ih.inject_holes(ball, label, per, "3812", draw=draw_short, target=0.5)
    frac = np.isnan(out[:, 0]).mean()
    assert 0.5 <= frac < 0.5 + 160 / len(ball) + 1e-9
    base = np.isnan(ball[:, 0]).mean()
    out2, hole2 = ih.inject_holes(ball, label, per, "3812", draw=draw_short, target=base - 0.01)
    assert not hole2.any() and np.array_equal(out2, ball, equal_nan=True)


@pytest.mark.skipif(not (os.path.exists(ih.REGIMEN_CSV) and os.path.exists(ih.dd.CAL_LARGO)),
                    reason="calibration aggregates not built (./run.sh regimen, ./run.sh calibrar-largo)")
def test_length_model_comes_from_the_challenge_classes():
    draw = ih.length_model()
    rng = np.random.default_rng(0)
    d = np.array([draw(rng) for _ in range(3000)])
    assert d.min() >= 1 and d.max() > 25 * 20 and np.median(d) > 25 * 5          # classes up to >= 20 s exist and most runs are >= 5 s
    reto_abs, reto_gaps = ih.challenge_targets()
    assert abs(reto_abs - ih.TARGET_ABSENT) < 1e-4 and abs(reto_gaps - ih.TARGET_HUECOS_H) < 1e-3
