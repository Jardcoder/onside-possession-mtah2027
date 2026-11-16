"""Synthetic tests for the ball smoothing of the bank (bancos/dfl_degradado.smooth_ball)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "bancos"))
sys.path.insert(0, os.path.join(ROOT, "src"))
import dfl_degradado as dd  # noqa: E402


def test_smoothing_never_crosses_a_hole_or_the_period_break():
    x = np.ones((100, 2))
    x[40:50] = np.nan
    x[70:] = 5.0
    per = np.concatenate([np.ones(70, int), np.full(30, 2)])
    y = dd.smooth_ball(x, per, "g6")
    assert np.array_equal(np.isnan(y[:, 0]), np.isnan(x[:, 0]))
    assert np.allclose(y[:70][~np.isnan(y[:70, 0])], 1.0)       # nothing from period 2 leaks into period 1
    assert np.allclose(y[70:], 5.0)


def test_causal_filter_does_not_anticipate_and_reports_its_delay():
    x = np.zeros((300, 2))
    x[150:] = 1.0
    per = np.ones(300, int)
    y = dd.smooth_ball(x, per, "causal")
    assert np.all(y[:150] == 0.0)                               # no response before the step
    assert y[160, 0] > 0 and y[299, 0] > 0.99
    assert dd.kernel_delay_frames("causal") > 5
    assert dd.kernel_delay_frames("g6") == 0.0
    c = dd.smooth_ball(x, per, "g6")
    assert c[149, 0] > 0.3                                      # the centred filter does anticipate


def test_load_degraded_default_is_untouched_by_the_smoothing_argument():
    import inspect
    assert inspect.signature(dd.load_degraded).parameters["ball_smooth"].default is None


# ---------------------------------------------------------------- dead-ball motion (dfl_degradado.apply_dead_motion)
MID = dd.dfl.DESARROLLO[0]
PARAMS = {"q_quieto": 0.0, "v_med": 3.0, "sigma": 1.0}


def _scene():
    """Period 1: 40 live frames, 200 dead, 30 live, hole of 20, 150 dead, 30 live. Period 2: 100 dead."""
    n1 = 40 + 200 + 30 + 20 + 150 + 30
    status = np.ones(n1 + 100, int)
    status[40:240] = 0
    status[320:470] = 0
    status[n1:] = 0
    per = np.concatenate([np.ones(n1, int), np.full(100, 2)])
    rng = np.random.default_rng(0)
    ball = np.cumsum(rng.normal(0, 1e-4, (len(per), 2)), axis=0) + 0.5
    ball[270:290] = np.nan
    return ball, status, per


def test_dead_motion_bridge_is_zero_at_both_ends_and_moves_the_ball():
    ball, st, per = _scene()
    out = dd.apply_dead_motion(ball, st, per, PARAMS, MID, 0)
    for a, b in ((40, 240), (320, 470), (470, 570)):
        d = (out[a:b] - ball[a:b]) * np.asarray(dd.ATD_PITCH)
        assert np.allclose(d[0], 0, atol=1e-9) and np.allclose(d[-1], 0, atol=1e-9)
        assert np.abs(d).max() > 0.1                           # metres: the ball really moves


def test_dead_motion_keeps_live_frames_holes_and_periods_apart():
    ball, st, per = _scene()
    out = dd.apply_dead_motion(ball, st, per, PARAMS, MID, 0)
    assert np.array_equal(np.isnan(out[:, 0]), np.isnan(ball[:, 0]))          # no new NaN, holes intact
    assert np.array_equal(out[st == 1], ball[st == 1], equal_nan=True)                        # live frames unchanged
    assert not np.array_equal(out[st == 0], ball[st == 0], equal_nan=True)
    # the period break cuts a stretch: dead frames 320..470 (period 1) and 470..570 (period 2) are separate bridges
    assert np.allclose(out[469] - ball[469], 0) and np.allclose(out[470] - ball[470], 0)


def test_dead_motion_stretch_cut_by_a_hole_is_two_bridges():
    ball, st, per = _scene()
    st[:] = 0
    out = dd.apply_dead_motion(ball, st, per, PARAMS, MID, 0)
    for edge in (269, 290):                                                   # frames next to the hole
        assert np.allclose(out[edge] - ball[edge], 0, atol=1e-9)
    assert np.abs(out[200] - ball[200]).max() > 1e-4                          # and the ball does move in between


def test_dead_motion_is_deterministic_and_depends_on_match_and_seed():
    ball, st, per = _scene()
    a = dd.apply_dead_motion(ball, st, per, PARAMS, MID, 0)
    assert np.array_equal(a, dd.apply_dead_motion(ball, st, per, PARAMS, MID, 0), equal_nan=True)
    assert not np.array_equal(a, dd.apply_dead_motion(ball, st, per, PARAMS, MID, 1), equal_nan=True)
    assert not np.array_equal(a, dd.apply_dead_motion(ball, st, per, PARAMS, dd.dfl.DESARROLLO[1], 0), equal_nan=True)
    assert np.array_equal(ball, _scene()[0], equal_nan=True)                                # the input is never modified


def test_dead_motion_quiet_stretches_barely_move_and_none_is_identity():
    ball, st, per = _scene()
    q = dd.apply_dead_motion(ball, st, per, {**PARAMS, "q_quieto": 1.0}, MID, 0)
    m = dd.apply_dead_motion(ball, st, per, PARAMS, MID, 0)
    assert np.nanmax(np.abs(q - ball)) < np.nanmax(np.abs(m - ball)) / 5
    assert np.array_equal(dd.apply_dead_motion(ball, st, per, None, MID, 0), ball, equal_nan=True)


def test_dead_motion_keys_and_load_degraded_signature():
    import inspect
    assert inspect.signature(dd.load_degraded).parameters["dead_motion"].default is None
    assert set(dd.DEAD_MOTION["m1"]) == {"q_quieto", "v_med", "sigma"}
    assert dd.resolve_dead_motion("m1")["tau"] == dd.DEAD_MOTION_DEFAULTS["tau"]


def test_pff_calibration_modes_are_centred_gaussians_with_no_delay():
    for name, sigma in (("g2", 2.0), ("g3", 3.0), ("g4", 4.0), ("g10", 10.0)):
        assert dd.BALL_SMOOTH[name] == ("gauss", sigma)
        assert dd.kernel_delay_frames(name) == 0.0
    assert dd.BALL_SMOOTH["g6"] == ("gauss", 6.0) and dd.BALL_SMOOTH["g8"] == ("gauss", 8.0)
    x = np.zeros((200, 2))
    x[100:] = 1.0
    per = np.ones(200, int)
    y2, y10 = dd.smooth_ball(x, per, "g2"), dd.smooth_ball(x, per, "g10")
    assert abs(y2[99, 0] + y2[100, 0] - 1.0) < 1e-9                     # symmetric around the step
    assert y10[89, 0] > y2[89, 0] > 0.0 or y10[89, 0] > 0.0             # wider sigma leaks further
    assert y2[89, 0] < 1e-6 < y10[89, 0]
