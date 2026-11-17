"""Synthetic tests for src/muerto.py (no real data). Pitch 100 x 60 m, so metres = normalised * (100, 60)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
import muerto  # noqa: E402

PITCH = (100.0, 60.0)
FPS = muerto.FPS


def scene(ball_m, players_speed, n=None):
    """ball_m (n,2) metres (NaN = absent). Two players (one per team) walking along x at `players_speed` m/s."""
    ball_m = np.asarray(ball_m, float)
    n = len(ball_m)
    t = np.arange(n) / FPS
    H = np.full((n, 24, 2), np.nan)
    A = np.full((n, 22, 2), np.nan)
    H[:, 0] = np.stack([20 + players_speed * t, np.full(n, 20.0)], 1) / np.array(PITCH)
    A[:, 0] = np.stack([30 + players_speed * t, np.full(n, 40.0)], 1) / np.array(PITCH)
    return ball_m / np.array(PITCH), {"H": H, "A": A}, np.ones(n, int)


def still_ball(n, x=50.0, y=30.0):
    return np.tile([x, y], (n, 1)).astype(float)


def moving_ball(n, speed):
    t = np.arange(n) / FPS
    return np.stack([10 + speed * t, np.full(n, 30.0)], 1)


def mask(ball, players_speed, **kw):
    b, p, per = scene(ball, players_speed)
    return muerto.dead_visible_mask(muerto.features(b, p, per, PITCH), **kw)


def test_ball_outside_pitch_needs_margin_and_duration():
    n = 400
    ball = still_ball(n)
    ball[100:200, 0] = 103.0                                   # 3 m beyond the right line for 4 s
    ball[300:330, 0] = 103.0                                   # 1.2 s only
    m = mask(ball, 4.0, t_min=3.0, margin=1.0)
    assert m[100:200].all() and not m[:100].any() and not m[200:].any()
    assert not mask(ball, 4.0, t_min=3.0, margin=5.0).any()    # not far enough out
    assert mask(ball, 4.0, t_min=1.0, margin=1.0)[300:330].all()


def test_still_ball_and_calm_players_but_not_one_without_the_other():
    n = 400
    kw = dict(t_min=3.0, v_bal=1.0, v_jug=2.0)
    assert mask(still_ball(n), 1.0, **kw)[50:350].all()        # still ball, players walking
    assert not mask(still_ball(n), 5.0, **kw).any()            # players running
    assert not mask(moving_ball(n, 6.0), 1.0, **kw).any()      # ball moving
    assert not mask(still_ball(n), 1.0, t_min=3.0).any()       # both conditions off


def test_holes_do_not_count_and_speeds_are_not_taken_across_them():
    n = 400
    ball = still_ball(n)
    ball[150:250] = np.nan                                     # 4 s hole in the middle of a still ball
    m = mask(ball, 1.0, t_min=3.0, v_bal=1.0, v_jug=2.0)
    assert not m[150:250].any()
    # 6 s hole with the ball jumping 40 m across it: the jump must not create a fast ball nor a fake still run
    ball = np.vstack([still_ball(60, 10, 30), np.full((150, 2), np.nan), still_ball(60, 50, 30)])
    m = mask(ball, 1.0, t_min=1.0, v_bal=1.0, v_jug=2.0)
    assert not m[60:210].any() and m[5:55].all() and m[215:265].all()


def test_players_tracking_jumps_are_ignored():
    n = 300
    b, p, per = scene(still_ball(n), 1.0)
    p["H"][100, 0] += 0.5                                      # 50 m jump for one frame (slot swap)
    feat = muerto.features(b, p, per, PITCH)
    m = muerto.dead_visible_mask(feat, t_min=3.0, v_bal=1.0, v_jug=2.0)
    assert m[50:250].all()


def test_p75_of_players_not_the_mean():
    n = 300
    b, p, per = scene(still_ball(n), 1.0)
    t = np.arange(n) / FPS
    for j in range(1, 4):                                      # three fast players among 5: p75 is fast
        p["H"][:, j] = np.stack([20 + 6.0 * t, np.full(n, 10.0 + j)], 1) / np.array(PITCH)
    feat = muerto.features(b, p, per, PITCH)
    assert not muerto.dead_visible_mask(feat, t_min=3.0, v_bal=1.0, v_jug=2.0).any()


def test_split_long_hole_gives_the_tail_to_the_owner_at_reappearance():
    n = 1000
    present = np.ones(n, bool)
    present[100:600] = False                                   # 20 s hole
    lab = np.array(["H"] * 100 + ["D"] * 500 + ["A"] * 400, dtype="<U1")
    per = np.ones(n, int)
    out = muerto.split_long_holes(lab, present, per, 10, 4)
    assert (out[100:500] == "D").all() and (out[500:600] == "A").all() and (out[600:] == "A").all()
    assert (muerto.split_long_holes(lab, present, per, 10, None) == lab).all()
    assert (muerto.split_long_holes(lab, present, per, 30, 4) == lab).all()      # hole not longer than the cap


def test_split_long_hole_keeps_d_when_nothing_follows():
    n = 800
    present = np.ones(n, bool)
    present[300:] = False                                      # hole until the end
    lab = np.array(["H"] * 300 + ["D"] * 500, dtype="<U1")
    assert (muerto.split_long_holes(lab, present, np.ones(n, int), 10, 4) == lab).all()
    present = np.ones(n, bool)
    present[300:500] = False
    lab = np.array(["H"] * 300 + ["D"] * 500, dtype="<U1")                       # D at the first visible frame too
    assert (muerto.split_long_holes(lab, present, np.ones(n, int), 5, 4) == lab).all()


def test_predict_off_is_identity_and_c1_runs_before_c2():
    n = 1000
    ball = still_ball(n)
    ball[100:600] = np.nan
    b, p, per = scene(ball, 1.0)
    lab = np.array(["H"] * 100 + ["D"] * 500 + ["A"] * 400, dtype="<U1")
    assert (muerto.predict(lab, b, p, per, PITCH) == lab).all()
    out = muerto.predict(lab, b, p, per, PITCH, v_bal=1.0, v_jug=2.0, t_min=3.0, s_rest=4)
    # the ball is still after the hole: C1 turns those frames to D, so C2 hangs no owner tail on the hole
    assert (out[100:600] == "D").all()
    out = muerto.predict(lab, b, p, per, PITCH, s_rest=4)
    assert (out[500:600] == "A").all()


def test_ball_features_is_the_ball_part_of_features():
    b, p, per = scene(moving_ball(200, 3.0), 1.0)
    f = muerto.features(b, p, per, PITCH)
    out_m, v_ball = muerto.ball_features(b, per, PITCH)
    assert np.array_equal(f["out_m"], out_m) and np.array_equal(f["v_ball"], v_ball, equal_nan=True)
