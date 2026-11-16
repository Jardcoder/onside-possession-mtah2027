"""Synthetic tests for src/toques.py (no real data). Pitch 100 x 100 m, so metres = normalised * 100."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
import reglas  # noqa: E402
import toques  # noqa: E402

PITCH = (100.0, 100.0)
KICK = 40


def scene(ball, h_paths, a_paths, period=None):
    """ball (n,2) metres (NaN = absent); h_paths / a_paths: lists of (n,2) metre paths."""
    n = len(ball)
    H = np.full((n, 24, 2), np.nan)
    A = np.full((n, 22, 2), np.nan)
    for j, p in enumerate(h_paths):
        H[:, j] = np.asarray(p) / 100
    for j, p in enumerate(a_paths):
        A[:, j] = np.asarray(p) / 100
    per = np.ones(n, int) if period is None else np.asarray(period)
    return np.asarray(ball, float) / 100, {"H": H, "A": A}, per


def const(n, x, y):
    return np.tile([x, y], (n, 1)).astype(float)


def kick_line(n, x0, speed, t0=KICK, x_stop=None):
    """Ball at rest at x0 until t0, then moving along +x at `speed` m/s (stops at x_stop if given)."""
    t = np.arange(n)
    x = x0 + np.clip(t - t0, 0, None) * speed / 25
    if x_stop is not None:
        x = np.minimum(x, x_stop)
    return np.stack([x, np.full(n, 50.0)], 1)


def changes(lab):
    return int((lab[1:] != lab[:-1]).sum())


def test_clean_pass_between_teams_changes_owner_once():
    n = 140
    ball = kick_line(n, 30.0, 15.0, x_stop=60.0)         # kick at 40, arrives at 60 at frame 90, stays
    b, p, per = scene(ball, [const(n, 29.5, 50), const(n, 10, 10)], [const(n, 60.3, 50), const(n, 90, 90)])
    lab = toques.predict(b, p, per, PITCH)
    assert set(lab[:KICK]) == {"H"}
    assert set(lab[100:]) == {"A"}
    assert changes(lab) == 1
    flip = int(np.flatnonzero(lab == "A")[0])
    assert 80 <= flip <= 95                              # at the reception, not when the ball is mid-way
    # the plain rule flips earlier (nearest player changes half way)
    base = reglas.predict("F", b, p, per, PITCH, window=0, max_gap_s=10)
    assert int(np.flatnonzero(base == "A")[0]) < flip - 10


def test_dribbling_without_deflection_never_changes_owner():
    n = 120
    t = np.arange(n)
    run_x = 20 + t * 0.2                                  # 5 m/s
    ball = np.stack([run_x + 0.5, np.full(n, 50.0)], 1)
    h = np.stack([run_x, np.full(n, 50.0)], 1)
    a = np.stack([run_x, np.full(n, 52.5)], 1)            # defender running alongside, 2 m from the ball
    b, p, per = scene(ball, [h, const(n, 10, 10)], [a, const(n, 90, 90)])
    lab = toques.predict(b, p, per, PITCH)
    assert set(lab) == {"H"}
    near = toques.prepare(b, p, PITCH)
    _, tt = toques.detect_touches(near, [(0, n)])
    assert set(tt.tolist()) <= {toques.HOME}


def test_ball_passing_close_to_a_rival_without_deviation_keeps_the_owner():
    n = 140
    ball = kick_line(n, 30.0, 15.0)                       # never stops: x reaches 30 + 100*0.6 = 90
    ball[:, 0] = np.minimum(ball[:, 0], 95)
    rival = const(n, 55.0, 51.0)                          # 1 m from the ball line
    b, p, per = scene(ball, [const(n, 29.5, 50), const(n, 10, 10)], [rival, const(n, 90, 5)])
    lab = toques.predict(b, p, per, PITCH)
    assert set(lab) == {"H"}
    base = reglas.predict("F", b, p, per, PITCH, window=0, max_gap_s=10)
    assert "A" in set(base)                               # without the rule the pass-by flips the label


def test_owner_is_not_carried_across_the_period_break():
    n = 80
    t = np.arange(n)
    # period 1: H touch at the start (control), period 2: ball visible near an A player, no touch event
    ball = np.concatenate([const(40, 30, 50), const(40, 70, 50)])
    h = np.concatenate([const(40, 30.3, 50), const(40, 10, 10)])
    a = np.concatenate([const(40, 90, 90), const(40, 70 + 4, 50)])
    per = np.concatenate([np.ones(40, int), np.full(40, 2)])
    b, p, per = scene(ball, [h], [a], per)
    lab = toques.predict(b, p, per, PITCH, k=12, r_ctrl=1.0)
    assert set(lab[:40]) == {"H"}
    assert set(lab[40:]) == {"A"}


def test_holes_follow_the_base_rule():
    n = 600
    ball = const(n, 30, 50)
    ball[100:200] = np.nan                                # 4 s: carry the owner
    ball[300:600] = np.nan                                # 12 s: D
    b, p, per = scene(ball, [const(n, 30.3, 50)], [const(n, 90, 90)])
    lab = toques.predict(b, p, per, PITCH, max_gap_s=10)
    assert set(lab[:300]) == {"H"}
    assert set(lab[300:]) == {"D"}


def test_no_touch_detected_on_empty_or_too_short_input():
    b, p, per = scene(np.full((5, 2), np.nan), [const(5, 1, 1)], [const(5, 2, 2)])
    assert set(toques.predict(b, p, per, PITCH)) == {"D"}
    b, p, per = scene(const(5, 50, 50), [const(5, 50.2, 50)], [const(5, 90, 90)])
    assert toques.predict(b, p, per, PITCH, return_touches=True)[1].size == 0


def _hole_scene(hole):
    """H controls the ball at (30, 50), the ball disappears for `hole` frames and reappears at (70, 50),
    at rest, with only an A player nearby (4 m away: no touch can be detected in the second stretch)."""
    n1, n2 = 60, 60
    n = n1 + hole + n2
    ball = np.concatenate([const(n1, 30, 50), np.full((hole, 2), np.nan), const(n2, 70, 50)])
    h = np.concatenate([const(n1, 30.3, 50), const(hole + n2, 10, 10)])
    a = np.concatenate([const(n1, 90, 90), const(hole + n2, 74, 50)])
    return scene(ball, [h], [a]), n1, n2


def test_owner_survives_a_short_hole_without_a_new_touch():
    (b, p, per), n1, n2 = _hole_scene(100)                 # 4 s hole
    lab = toques.predict(b, p, per, PITCH, max_gap_s=10)
    assert set(lab[:n1 + 100]) == {"H"}
    assert set(lab[n1 + 100:]) == {"H"}                    # same owner after the hole: no touch, no change
    old = toques.predict(b, p, per, PITCH, max_gap_s=10, keep_owner=False)
    assert set(old[n1 + 100:]) == {"A"}                    # the earlier behaviour reset to the nearest player


def test_owner_is_invalidated_after_a_hole_longer_than_the_cap():
    (b, p, per), n1, n2 = _hole_scene(300)                 # 12 s hole > 10 s cap
    lab = toques.predict(b, p, per, PITCH, max_gap_s=10)
    assert set(lab[:n1]) == {"H"}
    assert set(lab[n1:n1 + 300]) == {"D"}
    assert set(lab[n1 + 300:]) == {"A"}                    # back to the base until the first touch
    lab = toques.predict(b, p, per, PITCH, max_gap_s=20)   # a larger cap keeps the owner through it
    assert set(lab[n1:]) == {"H"}


def test_owner_is_invalidated_by_a_hole_that_spans_the_period_break():
    (b, p, per), n1, n2 = _hole_scene(100)
    per = np.concatenate([np.ones(n1 + 50, int), np.full(50 + n2, 2)])
    lab = toques.predict(b, p, per, PITCH, max_gap_s=10)
    assert set(lab[n1 + 100:]) == {"A"}                    # short hole, but the period changed inside it
