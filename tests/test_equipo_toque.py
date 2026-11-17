"""Tests for src/equipo_toque.py: the team given to each v2 deflection (synthetic scenes, no real data)."""
import inspect
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)
import equipo_toque as eq  # noqa: E402
import toques  # noqa: E402
from baseline import HOME, AWAY  # noqa: E402

PITCH = (100.0, 100.0)
V2 = {"r": 3.5, "dv": 2.0, "win": 5, "r_ctrl": 1.0, "k": 12}


def random_bank(seed, n=1500):
    """Ball with a new random velocity every 15 to 40 frames (kicks), 4 + 4 players that stay 1 to 4 m from it and
    jitter, with ball gaps, one NaN player and a period break."""
    rng = np.random.default_rng(seed)
    vel = np.zeros((n, 2))
    t = 0
    while t < n:
        step = int(rng.integers(15, 40))
        ang = rng.uniform(0, 2 * np.pi)
        vel[t:t + step] = rng.uniform(0.5, 9.0) / 25 * np.array([np.cos(ang), np.sin(ang)])
        t += step
    ball = np.clip(50 + np.cumsum(vel, axis=0) * 0.5, 5, 95)
    ball_n = ball.copy() / 100
    H = np.full((n, 24, 2), np.nan)
    A = np.full((n, 22, 2), np.nan)
    for k in range(4):
        for arr in (H, A):
            off = rng.normal(0, 2.0, 2)
            arr[:, k] = (ball + off + np.cumsum(rng.normal(0, 0.1, (n, 2)), axis=0)) / 100
    H[300:400, 1] = np.nan
    for a, b in ((200, 260), (700, 900), (1100, 1130)):
        ball_n[a:b] = np.nan
    per = np.where(np.arange(n) < 900, 1, 2)
    return ball_n, {"H": H, "A": A}, per


def bank(n, ball, h_paths, a_paths):
    """ball (n,2) metres; h_paths / a_paths: lists of (n,2) metre paths. Returns (ball_n, slots, near)."""
    H = np.full((n, 24, 2), np.nan)
    A = np.full((n, 22, 2), np.nan)
    for j, p in enumerate(h_paths):
        H[:, j] = np.asarray(p, float) / 100
    for j, p in enumerate(a_paths):
        A[:, j] = np.asarray(p, float) / 100
    ball_n = np.asarray(ball, float) / 100
    slots = {"H": H, "A": A}
    return ball_n, slots, toques.prepare(ball_n, slots, PITCH)


def const(n, x, y):
    return np.tile([x, y], (n, 1)).astype(float)


def cand_at(near, peak, team, vp=(0.0, 0.0), vn=(1.0, 0.0), n=None):
    """One hand-made deflection candidate (the fields equipo_toque reads)."""
    n = len(near["dmin"]) if n is None else n
    return {"frame": np.array([peak]), "team": np.array([team], np.int8), "peak": np.array([peak]),
            "start": np.array([0]), "stop": np.array([n]), "vp": np.array([vp], float), "vn": np.array([vn], float),
            "dv": np.array([np.hypot(*(np.subtract(vn, vp)))])}


def team_of(mode, cand, near, slots, r=3.5, win=5):
    return int(eq.deflection_teams(mode, cand, near, slots, PITCH, r=r, win=win)[0])


# ------------------------------------------------------------------------------------------------ the modes

def test_dir_picks_the_player_the_ball_leaves_from():
    """Ball at rest kicked along +x. H is slightly nearer but off to the side (the ball does not go away from it);
    A is a bit further, behind the ball. T0 says H, dir says A."""
    n = 140
    t = np.arange(n)
    ball = np.stack([50.0 + np.clip(t - 40, 0, None) * 15.0 / 25, np.full(n, 50.0)], 1)
    b, slots, near = bank(n, ball, [const(n, 50.0, 51.5)], [const(n, 48.2, 50.0)])
    runs = toques.visible_runs(~np.isnan(b[:, 0]), np.ones(n, int))
    d = toques.deflection_candidates(near, runs, r=3.5, dv=2.0, win=5)
    assert len(d["frame"]) == 1
    assert team_of("T0", d, near, slots) == HOME
    assert team_of("dir", d, near, slots) == AWAY
    # the tie rule only lets dir decide when the two distances are closer than D
    dh = np.linalg.norm(near["ball_m"][d["peak"][0]] - [50.0, 51.5])
    da = np.linalg.norm(near["ball_m"][d["peak"][0]] - [48.2, 50.0])
    gap = abs(dh - da)
    assert team_of(f"dir+tie{gap + 0.5:.1f}", d, near, slots) == AWAY
    assert team_of(f"dir+tie{max(gap - 0.2, 0.05):.1f}", d, near, slots) == HOME


def test_dir_with_one_player_inside_r_takes_that_one_and_with_none_falls_back_to_t0():
    n = 100
    b, slots, near = bank(n, const(n, 50.0, 50.0), [const(n, 50.0, 53.0)], [const(n, 50.0, 46.0)])   # H 3 m, A 4 m
    c = cand_at(near, 50, AWAY)                              # pretend T0 said AWAY
    assert team_of("dir", c, near, slots) == HOME            # only H is closer than r = 3.5
    assert team_of("kink", c, near, slots) == HOME
    assert team_of("dir", c, near, slots, r=2.0) == AWAY     # nobody inside r: T0
    assert team_of("kink", c, near, slots, r=2.0) == AWAY


def _runner_scene(n=100, jump_h=False, nan_h_before=False):
    """Ball at rest. H (1.5 m away) runs at 5 m/s along x the whole time; A (1.8 m away) runs the same way and stops
    dead at frame 50. Optional glitches on H: a one-frame teleport before the peak, or no data before the peak."""
    t = np.arange(n)
    h = np.stack([50.0 + (t - 50) * 0.2, np.full(n, 51.5)], 1)
    a = np.stack([50.0 + np.minimum(t - 50, 0) * 0.2, np.full(n, 48.2)], 1)
    if jump_h:
        h[47, 0] += 10.0
    if nan_h_before:
        h[45:50] = np.nan
    return bank(n, const(n, 50.0, 50.0), [h], [a])


def test_kink_picks_the_player_who_brakes_dead():
    b, slots, near = _runner_scene()
    c = cand_at(near, 50, HOME)                              # H is the nearest (1.5 < 1.8): T0 = H
    assert team_of("T0", c, near, slots) == HOME
    assert team_of("kink", c, near, slots) == AWAY           # A goes from 5 m/s to 0, H does not change
    assert team_of("kink+tie1.0", c, near, slots) == AWAY    # distances differ by 0.3 m < 1
    assert team_of("kink+tie0.1", c, near, slots) == HOME    # 0.3 m >= 0.1: tie rule does not fire, T0


def test_kink_drops_slot_jumps_and_falls_back_to_t0_without_valid_data():
    b, slots, near = _runner_scene(jump_h=True)
    c = cand_at(near, 50, HOME)
    assert team_of("kink", c, near, slots) == AWAY           # the teleport (250 m/s) is not a velocity change of H
    b, slots, near = _runner_scene(nan_h_before=True)
    c = cand_at(near, 50, HOME)
    assert team_of("kink", c, near, slots) == HOME           # H has no valid step before the peak: T0
    assert team_of("kink+tie1.0", c, near, slots) == HOME


def _passer_scene(n=100):
    """Ball at rest. A stands 1.9 m away. H runs past at 12 m/s along x, 0.5 m off the ball, closest at frame 32."""
    t = np.arange(n)
    h = np.stack([50.0 + (t - 32) * 0.48, np.full(n, 50.5)], 1)
    return bank(n, const(n, 50.0, 50.0), [h], [const(n, 50.0, 51.9)])


def test_lag_and_minwin_look_back_in_time():
    b, slots, near = _passer_scene()
    c = cand_at(near, 40, int(near["team"][40]))
    assert c["team"][0] == AWAY                              # at the peak H is already far: T0 = A
    for mode, want in (("T0", AWAY), ("lag2", AWAY), ("lag4", AWAY), ("lag6", HOME),
                       ("minwin3", AWAY), ("minwin6", HOME)):
        assert team_of(mode, c, near, slots) == want, mode


def test_lag_and_minwin_are_clamped_to_the_visible_stretch():
    b, slots, near = _passer_scene()
    c = cand_at(near, 3, AWAY)                               # peak 3 frames after the start: lag 6 clamps to frame 0
    for mode in ("lag6", "minwin6"):
        assert team_of(mode, c, near, slots) in (HOME, AWAY)
    c = cand_at(near, 3, AWAY, n=10)
    assert team_of("minwin6", c, near, slots) in (HOME, AWAY)


def test_no_candidates_and_unknown_mode():
    b, slots, near = _passer_scene()
    runs = toques.visible_runs(np.zeros(len(b), bool), np.ones(len(b), int))
    d = toques.deflection_candidates(near, runs, r=3.5, dv=2.0, win=5)
    for mode in eq.MODES:
        assert eq.deflection_teams(mode, d, near, slots, PITCH).shape == (0,)
    with pytest.raises(ValueError):
        eq.parse_mode("nearest")


def test_modes_are_the_twelve_of_the_grid():
    assert len(eq.MODES) == len(set(eq.MODES)) == 12
    assert [eq.parse_mode(m) for m in ("T0", "lag4", "minwin6", "dir", "kink+tie2.0")] == [
        ("T0", None, None), ("lag", 4, None), ("minwin", 6, None), ("dir", None, None), ("kink", None, 2.0)]


# ----------------------------------------------------------------------------- v2 identity and invariants

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_t0_is_exactly_the_v2_touches(seed):
    ball, slots, per = random_bank(seed)
    near = toques.prepare(ball, slots, PITCH)
    runs = toques.visible_runs(~np.isnan(ball[:, 0]), per)
    f0, t0 = toques.detect_touches(near, runs, **V2)
    f1, t1 = eq.detect_touches("T0", near, slots, PITCH, runs, **V2)
    assert len(f0) > 5
    assert np.array_equal(f0, f1) and np.array_equal(t0, t1)
    # and the whole prediction with the injected touches is the v2 one
    p0 = toques.predict(ball, slots, per, PITCH, max_gap_s=5, **V2)
    p1 = toques.predict(ball, slots, per, PITCH, max_gap_s=5, touches=(f1, t1), **V2)
    assert np.array_equal(p0, p1)


@pytest.mark.parametrize("seed", [0, 1])
def test_every_mode_keeps_frames_and_control_touches(seed):
    ball, slots, per = random_bank(seed)
    near = toques.prepare(ball, slots, PITCH)
    runs = toques.visible_runs(~np.isnan(ball[:, 0]), per)
    d = toques.deflection_candidates(near, runs, r=V2["r"], dv=V2["dv"], win=V2["win"])
    ctrl = toques.control_touches(near, runs, r_ctrl=V2["r_ctrl"], k=V2["k"])
    f0, _ = toques.detect_touches(near, runs, **V2)
    changed = 0
    for mode in eq.MODES:
        teams = eq.deflection_teams(mode, d, near, slots, PITCH, r=V2["r"], win=V2["win"])
        assert teams.dtype == np.int8 and set(np.unique(teams)) <= {HOME, AWAY} and len(teams) == len(d["frame"])
        f, t = eq.detect_touches(mode, near, slots, PITCH, runs, **V2)
        assert np.array_equal(f, f0)
        fm, tm = toques.merge_touches((d["frame"], teams), ctrl)
        assert np.array_equal(f, fm) and np.array_equal(t, tm)
        changed += int((teams != d["team"]).sum())
    assert changed > 0                                       # the modes do something on this bank


@pytest.mark.parametrize("mode", eq.MODES)
def test_no_leak_and_slot_ids_are_not_an_identity(mode):
    """The functions take arrays only (no label, status, event or frame ids), and permuting the slots of a team
    leaves the result unchanged."""
    for fn in (eq.deflection_teams, eq.detect_touches, eq._decide, eq._nearest, eq._slot_kink):
        for p in inspect.signature(fn).parameters:
            assert not any(w in p.lower() for w in ("label", "status", "event", "gt")), (fn.__name__, p)
    src = inspect.getsource(eq).lower()
    code = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("#"))
    body = code.split('"""', 2)[2]                           # the code after the module docstring
    for word in ("label", "ball_status", "event", "ball_present", "n_present"):
        assert word not in body, word
    ball, slots, per = random_bank(3)
    rng = np.random.default_rng(0)
    perm = {"H": slots["H"][:, rng.permutation(24)], "A": slots["A"][:, rng.permutation(22)]}
    runs = toques.visible_runs(~np.isnan(ball[:, 0]), per)
    a = eq.detect_touches(mode, toques.prepare(ball, slots, PITCH), slots, PITCH, runs, **V2)
    b = eq.detect_touches(mode, toques.prepare(ball, perm, PITCH), perm, PITCH, runs, **V2)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
