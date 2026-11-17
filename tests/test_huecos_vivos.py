"""Synthetic tests for src/huecos_vivos.py (no real data). Pitch 100 x 60 m, so metres = normalised * (100, 60)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
import huecos_vivos as hv  # noqa: E402

PITCH = (100.0, 60.0)
FPS = hv.FPS
V_OUT = 2.0


def scene(n, holes, v_in, period=None, ball_start=(50.0, 30.0), edge_pre=None, edge_post=None, v_out=V_OUT):
    """n frames. `holes` [(a, b)] without ball. Two players (one per team) run along x at `v_out` m/s, at `v_in` (scalar or
    per hole) inside the holes. The ball sits at the centre; edge_pre / edge_post = (x, y) metres of the last visible frame
    before / first after every hole (the frame just outside it) when given."""
    v_in = np.broadcast_to(np.asarray(v_in, float), (len(holes),))
    v = np.full(n, v_out)
    for (a, b), vi in zip(holes, v_in):
        v[a:b] = vi
    x = np.cumsum(v) / FPS
    ball = np.tile(ball_start, (n, 1)).astype(float)
    for a, b in holes:
        if edge_pre is not None:
            ball[a - 1] = edge_pre
        if edge_post is not None and b < n:
            ball[b] = edge_post
        ball[a:b] = np.nan
    H = np.full((n, 24, 2), np.nan)
    A = np.full((n, 22, 2), np.nan)
    H[:, 0] = np.stack([20 + x, np.full(n, 20.0)], 1) / np.array(PITCH)
    A[:, 0] = np.stack([30 + x, np.full(n, 40.0)], 1) / np.array(PITCH)
    per = np.ones(n, int) if period is None else np.asarray(period)
    return ball / np.array(PITCH), {"H": H, "A": A}, per


def base_labels(n, split, pre="H", post="A"):
    lab = np.full(n, "D", dtype="<U1")
    lab[:split] = pre
    lab[split:] = post
    return lab


def table(ball, slots, per, **kw):
    return hv.hole_table(ball, slots, per, PITCH, **kw)


def test_only_holes_longer_than_the_cap_enter_the_table():
    ball, slots, per = scene(1500, [(200, 300), (500, 700)], 2.0)          # 4 s and 8 s
    t = table(ball, slots, per)
    assert t["start"].tolist() == [500] and t["end"].tolist() == [700] and t["length"].tolist() == [200]
    assert t["pre"].tolist() == [499] and t["post"].tolist() == [700]


def test_v_rel_is_hole_speed_over_the_same_period_median_with_borders_dropped():
    ball, slots, per = scene(1500, [(500, 700)], 3.0)
    t = table(ball, slots, per)
    assert np.isclose(t["v_ref"][0], V_OUT, atol=1e-6)
    assert np.isclose(t["v_abs"][0], 3.0, atol=1e-6)
    assert np.isclose(t["v_rel"][0], 1.5, atol=1e-6)


def test_alive_and_dead_by_theta_and_nan_is_dead():
    ball, slots, per = scene(2200, [(500, 700), (1200, 1400)], [3.0, 0.8])         # v_rel 1.5 and 0.4
    t = table(ball, slots, per)
    assert hv.decide(t, 1.0).tolist() == [True, False]
    assert hv.decide(t, 0.3).tolist() == [True, True]
    assert hv.decide(t, 1.6).tolist() == [False, False]
    t["v_rel"] = np.array([np.nan, 1.5])
    assert hv.decide(t, 0.0).tolist() == [False, True]


def test_l_max_leaves_longer_holes_dead():
    ball, slots, per = scene(2600, [(500, 700), (1200, 1700)], 3.0)               # 8 s and 20 s
    t = table(ball, slots, per)
    assert hv.decide(t, 1.0, l_max_s=10).tolist() == [True, False]
    assert hv.decide(t, 1.0, l_max_s=20).tolist() == [True, True]
    assert hv.decide(t, 1.0, l_max_s=np.inf).tolist() == [True, True]


def test_edge_veto_on_either_side_and_for_a_ball_outside():
    far = (50.0, 30.0)
    cases = {"pre near the touchline": ((0.5, 30.0), far, True), "post near the byline": (far, (99.4, 30.0), True),
             "pre outside": ((-0.5, 30.0), far, True), "both inside": ((10.0, 30.0), (60.0, 20.0), False),
             "corner flag side": (far, (99.5, 59.7), True)}
    for name, (pre, post, vetoed) in cases.items():
        ball, slots, per = scene(1500, [(500, 700)], 3.0, edge_pre=pre, edge_post=post)
        t = table(ball, slots, per)
        for e in (1.0, 2.0):
            assert hv.decide(t, 1.0, edge_m=e).tolist() == [not vetoed], (name, e)
        assert hv.decide(t, 1.0, edge_m=None).tolist() == [True], name                  # veto off


def test_edge_threshold_is_in_metres():
    ball, slots, per = scene(1500, [(500, 700)], 3.0, edge_pre=(1.5, 30.0), edge_post=(50.0, 30.0))
    t = table(ball, slots, per)
    assert hv.decide(t, 1.0, edge_m=1.0).tolist() == [True]
    assert hv.decide(t, 1.0, edge_m=2.0).tolist() == [False]


def test_the_three_fills_and_dead_holes_stay_d():
    ball, slots, per = scene(2400, [(500, 700), (1200, 1400)], [3.0, 0.8])
    t = table(ball, slots, per)
    alive = hv.decide(t, 1.0)
    base = base_labels(2400, 600)
    base[500:700] = "D"
    base[1200:1400] = "D"
    pre, post, split = (hv.fill(base, t, alive, m) for m in hv.MODES)
    assert (pre[500:700] == "H").all() and (post[500:700] == "A").all()
    assert (split[500:600] == "H").all() and (split[600:700] == "A").all()
    for out in (pre, post, split):
        assert (out[1200:1400] == "D").all()                    # dead hole untouched
        assert (out[:500] == base[:500]).all() and (out[700:1200] == base[700:1200]).all() and (out[1400:] == base[1400:]).all()
    assert (base[500:700] == "D").all()                         # input not modified


def test_missing_owner_leaves_d_in_the_fill():
    ball, slots, per = scene(1500, [(500, 700)], 3.0)
    t = table(ball, slots, per)
    alive = np.array([True])
    base = np.full(1500, "H", dtype="<U1")
    base[500:700] = "D"
    base[699 + 1] = "D"                                         # no owner on the first visible frame after the hole
    assert (hv.fill(base, t, alive, "post")[500:700] == "D").all()
    assert (hv.fill(base, t, alive, "pre")[500:700] == "H").all()
    out = hv.fill(base, t, alive, "split")
    assert (out[500:600] == "H").all() and (out[600:700] == "D").all()


def test_period_border_holes():
    n = 3000
    per = np.where(np.arange(n) < 1500, 1, 2)
    # hole at the start of period 2 (pre is None) and hole at the end of period 1 (post is None)
    ball, slots, _ = scene(n, [(1300, 1500), (1500, 1700), (2500, 2700)], 3.0, period=per)
    t = table(ball, slots, per)
    assert t["start"].tolist() == [1300, 1500, 2500]            # the absent run across the break is split in two
    assert t["period"].tolist() == [1, 2, 2]
    assert t["post"].tolist() == [-1, 1700, 2700] and t["pre"].tolist() == [1299, -1, 2499]
    alive = np.array([True, True, True])
    base = np.full(n, "H", dtype="<U1")
    base[1500:] = "A"
    for a, b in ((1300, 1500), (1500, 1700), (2500, 2700)):
        base[a:b] = "D"
    pre = hv.fill(base, t, alive, "pre")
    post = hv.fill(base, t, alive, "post")
    split = hv.fill(base, t, alive, "split")
    assert (pre[1300:1500] == "H").all() and (post[1300:1500] == "D").all()          # end of period 1: no post owner
    assert (pre[1500:1700] == "D").all() and (post[1500:1700] == "A").all()          # start of period 2: no pre owner
    assert (split[1500:1600] == "D").all() and (split[1600:1700] == "A").all()
    assert (pre[1300:1500] != "A").all()                                              # the owner never crosses the break


def test_reference_speed_is_per_period():
    n = 3000
    per = np.where(np.arange(n) < 1500, 1, 2)
    ball, slots, _ = scene(n, [(500, 700)], 3.0, period=per)                         # period 1: v_ref 2.0
    ball2, slots2, _ = scene(n, [(2000, 2200)], 3.0, period=per, v_out=1.0)           # period 2 slower: v_ref 1.0
    t = table(ball, slots, per)
    t2 = table(ball2, slots2, per)
    assert np.isclose(t["v_ref"][0], 2.0, atol=1e-6) and np.isclose(t2["v_ref"][0], 1.0, atol=1e-6)
    assert np.isclose(t["v_rel"][0], 1.5, atol=1e-6) and np.isclose(t2["v_rel"][0], 3.0, atol=1e-6)


def test_ref_mask_restricts_the_reference_frames():
    ball, slots, per = scene(1500, [(500, 700)], 3.0)
    mask = np.zeros(1500, bool)
    mask[800:] = True
    t = table(ball, slots, per, ref_mask=mask)
    assert np.isclose(t["v_ref"][0], V_OUT, atol=1e-6)
    mask[:] = False
    assert np.isnan(table(ball, slots, per, ref_mask=mask)["v_rel"][0])


def test_nothing_alive_returns_the_base_and_predict_matches_the_pieces():
    ball, slots, per = scene(1500, [(500, 700)], 3.0)
    base = base_labels(1500, 600)
    base[500:700] = "D"
    assert (hv.predict(base, ball, slots, per, PITCH, theta=99.0) == base).all()
    out = hv.predict(base, ball, slots, per, PITCH, theta=1.0, mode="post")
    assert (out[500:700] == "A").all() and (out[:500] == base[:500]).all()


def test_grid_is_small_and_unique():
    g = hv.grid()
    assert len(g) == len(hv.THETAS) * len(hv.L_MAX_S) * len(hv.EDGES_M) * len(hv.MODES) <= 150
    assert len({t for t, _ in g}) == len(g)
