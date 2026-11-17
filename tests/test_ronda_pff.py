"""Round 1 on PFF (bancos/ronda_pff.py, src/veto_dueno.py): folds, nested selection, C2 veto, no event-derived input to a method."""
import os
import re
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("bancos", "src", "tests"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import evaluar_pff as ep  # noqa: E402
import pff_split  # noqa: E402
import ronda_pff as rp  # noqa: E402
import veto_dueno as vd  # noqa: E402
from baseline import HOME, AWAY  # noqa: E402

FORBIDDEN = re.compile(r"ball_status|is_touch|uncertain|video_missing|ball_visible|ball_z|players_visible|label_poss|label_ut"
                       r"|attack_right|\[\"label\"\]|\['label'\]")
needs_meta = pytest.mark.skipif(not os.path.isdir(pff_split.META_DIR), reason="PFF metadata not available (set ONSIDE_PFF_DIR)")


# ------------------------------------------------------------------------------------------------ folds

def test_folds_partition_the_development_set_by_phase():
    allm = [m for f in rp.FOLDS for m in f]
    assert sorted(allm) == sorted(pff_split.DESARROLLO) and len(set(allm)) == 40
    assert all(len(f) == 8 for f in rp.FOLDS) and not set(allm) & set(pff_split.RESERVADOS)
    assert all(sum(int(m) >= 10000 for m in f) == 2 for f in rp.FOLDS)      # knockout ids are 105xx: 2 per fold
    assert [rp.fold_of(m) for m in rp.FOLDS[3]] == [3] * 8


@needs_meta
def test_folds_regenerate_to_the_literals():
    assert rp.make_folds(pff_split.read_phases()) == rp.FOLDS


# ------------------------------------------------------------------------------------------------ nested selection

def synth_f1(seed=0, cand="C1"):
    rng = np.random.default_rng(seed)
    f1 = {}
    for b in rp.BALLS:
        f1[("B3", "B3", b, "label")] = {m: 0.30 + 0.02 * rng.standard_normal() for m in pff_split.DESARROLLO}
        for tag, _ in rp.GRIDS[cand]:
            f1[(cand, tag, b, "label")] = {m: f1[("B3", "B3", b, "label")][m] + 0.01 * rng.standard_normal()
                                           for m in pff_split.DESARROLLO}
    return f1


def test_selection_never_sees_the_test_fold():
    f1 = synth_f1()
    chosen, per_fold = rp.nested(f1, "C1")
    g = synth_f1()
    for b in rp.MAIN_BALLS:                                  # make a different configuration look perfect on fold 0's test matches
        for tag, _ in rp.GRIDS["C1"]:
            for m in rp.FOLDS[0]:
                g[("C1", tag, b, "label")][m] += 1.0 if tag == rp.GRIDS["C1"][5][0] else -1.0
    chosen2, per_fold2 = rp.nested(g, "C1")
    assert per_fold2[0] == per_fold[0]                       # fold 0 trains on the other folds only
    assert all(chosen2[m] == chosen[m] for m in rp.FOLDS[0])


def test_selection_objective_is_min_of_g6_and_g8_not_g6_alone():
    f1 = synth_f1()
    tags = [t for t, _ in rp.GRIDS["C1"]]
    train = [m for f in rp.FOLDS[1:] for m in f]
    for tag in tags:                                         # neutral everywhere ...
        for b in rp.BALLS:
            f1[("C1", tag, b, "label")] = dict(f1[("B3", "B3", b, "label")])
    for m in train:                                          # ... except two configurations
        f1[("C1", tags[3], "g6", "label")][m] += 0.05        # huge in g6, zero in g8: min = 0
        f1[("C1", tags[7], "g6", "label")][m] += 0.01
        f1[("C1", tags[7], "g8", "label")][m] += 0.01
    assert rp.select(f1, "C1", train)[0] == tags[7]


def test_tie_break_fewer_parameters_then_wins_then_grid_order():
    tags = [t for t, _ in rp.GRIDS["C2"]]
    train = [m for f in rp.FOLDS[1:] for m in f]
    f1 = {}
    for b in rp.BALLS:
        f1[("B3", "B3", b, "label")] = {m: 0.3 for m in pff_split.DESARROLLO}
        for t in tags:
            f1[("C2", t, b, "label")] = {m: 0.3 for m in pff_split.DESARROLLO}
    for b in rp.MAIN_BALLS:                                  # all configurations tie at +0.01 mean
        for t in tags:
            for m in train:
                f1[("C2", t, b, "label")][m] = 0.31
    assert rp.select(f1, "C2", train)[0] == "rchg1.5_drinf"  # infinite d_rel has one parameter less; first of the grid among them
    assert rp.n_free("C2", (1.5, np.inf)) == 2 and rp.n_free("C2", (1.5, 0.5)) == 3


def test_tie_break_more_matches_won_then_grid_order():
    tags = [t for t, _ in rp.GRIDS["C1"]]
    train = [m for f in rp.FOLDS[1:] for m in f]
    f1 = {}
    for b in rp.BALLS:
        f1[("B3", "B3", b, "label")] = {m: 0.3 for m in pff_split.DESARROLLO}
        for t in tags:
            f1[("C1", t, b, "label")] = {m: 0.3 for m in pff_split.DESARROLLO}
    for b in rp.MAIN_BALLS:
        for m in train:                                      # tags[4]: +0.01 on every match; tags[9]: same mean, 16 wins only
            f1[("C1", tags[4], b, "label")][m] = 0.31
            f1[("C1", tags[9], b, "label")][m] = 0.3 + (0.02 if train.index(m) < 16 else 0.0)
            f1[("C1", tags[2], b, "label")][m] = 0.3 + (0.02 if train.index(m) < 16 else 0.0)
    assert rp.select(f1, "C1", train)[0] == tags[4]          # same free parameters: more wins beats an earlier grid position
    for b in rp.MAIN_BALLS:
        for m in train:
            f1[("C1", tags[9], b, "label")][m] = 0.31
    assert rp.select(f1, "C1", train)[0] == tags[4]          # full tie: the earlier of the grid
    assert rp.n_free("C1", None) == 3 and rp.n_free("C3", (0.65, np.inf, "pre")) == 2 and rp.n_free("C3", (0.65, 10.0, "pre")) == 3


def test_verdict_applies_8_20_literally():
    ok = {"mean": 0.007, "lo": 0.001, "median": 0.002}
    assert rp.verdict(ok, ok)[1]
    assert not rp.verdict(ok, {**ok, "mean": 0.0059})[1]
    assert not rp.verdict(ok, {**ok, "lo": 0.0})[1]
    assert not rp.verdict({**ok, "median": 0.0}, ok)[1]


# ------------------------------------------------------------------------------------------------ C2 veto

N = 100
RUNS = [(0, N)]
PER = np.ones(N, int)


def scene(a_series, h=1.0):
    d = np.full((N, 2), 5.0)
    d[:, HOME] = h
    d[:, AWAY] = a_series
    return d


def valley(depth=1.2, at=50):
    s = np.full(N, 3.0)
    s[at - 2:at + 3] = [2.0, 1.6, depth, 1.6, 2.0]
    return s


def run_veto(dists, frames, teams, defl, base="H", r_chg=1.5, d_rel=np.inf, gap=125, runs=RUNS, per=PER):
    b = np.full(len(dists), base, dtype="<U1")
    return vd.veto_owner_change(np.array(frames), np.array(teams, np.int8), np.array(defl, bool), dists, runs, per, b, r_chg, d_rel, gap)


def test_veto_conditions_one_by_one():
    d = scene(valley())
    assert run_veto(d, [50], [AWAY], [True])[0]                              # passes all three
    assert not run_veto(d, [50], [AWAY], [True], r_chg=1.1)[0]               # 1: rival too far (1.2 m)
    assert not run_veto(d, [50], [AWAY], [True], d_rel=0.0)[0]               # 2: farther than the previous owner (1.0 m)
    assert run_veto(d, [50], [AWAY], [True], d_rel=0.2 + 1e-9)[0]
    mono = np.concatenate([np.linspace(3.0, 1.2, 51), np.linspace(1.2, 0.5, 49)])   # 3: still approaching, no valley at t
    assert not run_veto(scene(mono), [50], [AWAY], [True])[0]
    assert run_veto(scene(valley(1.2, 50)), [50], [AWAY], [True], r_chg=2.5)[0]


def test_veto_only_acts_on_owner_changes_of_deflections():
    d = scene(np.full(N, 9.0))                                               # rival far away: every condition fails
    assert not run_veto(d, [50], [AWAY], [True])[0]                          # H owns (base) -> A is a change: vetoed
    assert run_veto(d, [50], [AWAY], [False])[0]                             # control touches are never vetoed
    assert run_veto(d, [50], [AWAY], [True], base="A")[0]                    # A already owns: not a change
    assert run_veto(d, [50], [HOME], [True])[0]                              # H keeps the ball
    assert run_veto(d, [50], [AWAY], [True], base="D")[0]                    # previous owner D: no veto


def test_vetoed_touch_does_not_become_the_owner_and_kept_one_does():
    far, near = np.full(N, 9.0), valley()
    d = scene(far)
    assert list(run_veto(d, [40, 60], [AWAY, AWAY], [True, True])) == [False, False]   # H stays: both are changes
    d2 = scene(near)
    assert list(run_veto(d2, [50, 70], [AWAY, AWAY], [True, True], r_chg=1.5)) == [True, True]
    s = valley(1.2, 50)
    s[68:73] = 9.0                                                           # second A touch would fail, but A already owns
    assert list(run_veto(scene(s), [50, 70], [AWAY, AWAY], [True, True])) == [True, True]
    # a kept A touch then an H deflection that fails its own conditions is vetoed (A owns now)
    h = np.full(N, 1.0)
    h[75:] = 9.0
    dd_ = scene(valley(), h=1.0)
    dd_[70:, HOME] = 8.0
    assert list(run_veto(dd_, [50, 72], [AWAY, HOME], [True, True])) == [True, False]


def test_veto_owner_across_holes_and_periods():
    d = scene(np.full(N, 9.0))
    runs = [(0, 40), (60, 100)]                                              # hole of 20 frames
    base = np.full(N, "H", dtype="<U1")
    k = vd.veto_owner_change(np.array([60]), np.array([AWAY], np.int8), np.array([True]), d, runs, PER, base, 1.5, np.inf, 125)
    assert not k[0]                                                          # first frame of a linked run: previous owner H (label of the hole)
    k = vd.veto_owner_change(np.array([60]), np.array([AWAY], np.int8), np.array([True]), d, runs, PER, base, 1.5, np.inf, 10)
    assert k[0]                                                              # hole longer than the cap: invalidated, no previous owner
    per = np.where(np.arange(N) < 50, 1, 2)
    k = vd.veto_owner_change(np.array([60]), np.array([AWAY], np.int8), np.array([True]), d, runs, per, base, 1.5, np.inf, 125)
    assert k[0]                                                              # period change: no previous owner
    k = vd.veto_owner_change(np.array([20, 70]), np.array([AWAY, AWAY], np.int8), np.array([False, True]), d, runs, PER, base, 1.5, np.inf, 125)
    assert k[1]                                                              # A owns (control touch) and carries over the hole: not a change


def test_team_distances():
    n = 3
    H = np.full((n, 24, 2), np.nan)
    A = np.full((n, 22, 2), np.nan)
    H[:, 0] = [0.5, 0.5]
    H[:, 1] = [0.6, 0.5]
    A[0, 0] = [0.5, 0.52]
    ball = np.full((n, 2), 0.5)
    ball[2] = np.nan
    out = vd.team_distances(ball, {"H": H, "A": A}, (100.0, 100.0))
    assert out[0, HOME] == 0 and abs(out[0, AWAY] - 2.0) < 1e-9
    assert np.isinf(out[1, AWAY]) and np.isinf(out[2]).all()


# ------------------------------------------------------------------------------------------------ pipeline

def synth_match(n=1500, seed=3):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    ball = np.stack([0.5 + 0.2 * np.sin(t / 40.0), 0.5 + 0.1 * np.cos(t / 55.0)], 1) + rng.normal(0, 0.0015, (n, 2))
    ball[400:520] = np.nan
    H = np.full((n, 24, 2), np.nan)
    A = np.full((n, 22, 2), np.nan)
    for j in range(11):
        H[:, j] = ball + rng.normal(0, 0.03, (n, 2)) + rng.normal(0, 0.003, (n, 2)).cumsum(0) * 0.1
        A[:, j] = ball + rng.normal(0, 0.03, (n, 2)) + rng.normal(0, 0.003, (n, 2)).cumsum(0) * 0.1
    return ball, {"H": H, "A": A}, np.ones(n, int), (105.0, 68.0)


def test_pipeline_base_is_b3_and_c1_null_point_is_b3():
    ball, slots, per, pitch = synth_match()
    m = rp.build_method_ctx(ball, slots, per, pitch)
    base, tf, _ = rp.pipeline(m)
    assert np.array_equal(base, ep.ladder(ball, slots, per, pitch)["B3"])
    assert dict(rp.C1_GRID)["dv2_w5_r3.5"] == rp.B3_C1
    assert np.array_equal(rp.pipeline(m, c1=dict(rp.C1_GRID)["dv2_w5_r3.5"])[0], base)
    pr, tf2, _ = rp.pipeline(m, c2=(1.5, 0.0))
    assert len(tf2) <= len(tf) and set(np.unique(pr)) <= {"H", "A", "D"}
    pr3, _, alive = rp.pipeline(m, c3=(0.65, np.inf, "pre"))
    assert len(alive) == len(m["tab"]["start"])
    assert (pr3[:400] == base[:400]).all()                                   # C3 only touches the holes


def test_method_modules_of_the_round_read_nothing_derived_from_events():
    for name in ("veto_dueno.py", "huecos_vivos.py", "toques.py", "equipo_toque.py"):
        assert not FORBIDDEN.search(open(os.path.join(ROOT, "src", name)).read()), name
    import inspect
    assert not FORBIDDEN.search(inspect.getsource(rp.pipeline)) and not FORBIDDEN.search(inspect.getsource(rp.build_method_ctx))
    code = "\n".join(l for l in open(os.path.join(ROOT, "bancos", "ronda_pff.py")).read().splitlines() if not l.lstrip().startswith("#"))
    assert "pff_split.load_dev(" in code and "smoothed=True" not in code
    assert not re.search(r"\bpff\.load\(|from pff import|^\s*import pff\b", code, re.M)
    assert not set(pff_split.RESERVADOS) & {m for f in rp.FOLDS for m in f}


def test_round3_grid_and_cache_sharing():
    """R3 = same mechanism as R2 with the round 3 grid; it is cached as R2 so the taus shared with round 2 reuse their files."""
    assert [p[0] for _, p in rp.GRIDS["R3"]] == [0.0, 0.8, 1.0, 1.2, 1.6, 2.0]
    assert [p[0] for _, p in rp.R2_GRID] == [0.0, 0.2, 0.4, 0.6, 0.8]
    cfgs = rp.all_configs(3)
    assert {c for c, _, _ in cfgs} == {"B3", "R2"} and len(cfgs) == 7
    assert rp.n_free("R3", (0.0,)) == 0 and rp.n_free("R3", (1.6,)) == 1


# ------------------------------------------------------------------------------------------------ round 4

def test_round4_grid_base_and_cache_layout():
    assert [(t, m) for _, (t, m) in rp.GRIDS["R4"]] == [(t, m) for t in (1.2, 1.6, 2.0, 3.0) for m in ("any", "own")]
    assert rp.BASE_OF["R4"] == ("R2", "tau1.6") and ("tau1.6", (1.6,)) in rp.R3_GRID
    cfgs = rp.all_configs(4)
    assert [c[:2] for c in cfgs[:2]] == [("B3", "B3"), ("R2", "tau1.6")] and len(cfgs) == 10
    assert rp.GTS4 == ("label", "label_poss", "label_aerial") and rp.GTS == ("label", "label_poss")          # rounds 1-3 rows unchanged
    assert rp.cache_file("3826", "B3", "B3", 4) != rp.cache_file("3826", "B3", "B3")
    assert rp.fingerprint4("R4", "tau1.6_any", (1.6, "any")) != rp.fingerprint("R4", "tau1.6_any", (1.6, "any"))
    assert rp.fingerprint4("R4", "tau1.6_any", (1.6, "any")) != rp.fingerprint4("R4", "tau1.6_own", (1.6, "own"))
    assert rp.n_free("R4", (1.2, "any")) == 2


def test_round4_deltas_and_selection_are_measured_against_the_base_not_against_b3():
    mids = list(pff_split.DESARROLLO)
    f1 = {}
    for g in rp.GTS4:
        for b in rp.BALLS:
            f1[("B3", "B3", b, g)] = {m: 0.30 for m in mids}
            f1[("R2", "tau1.6", b, g)] = {m: 0.40 for m in mids}                       # the base is 0.10 above B3
            for tag, (t, mode) in rp.GRIDS["R4"]:
                f1[("R4", tag, b, g)] = {m: 0.40 + (0.01 if (t, mode) == (2.0, "own") else -0.01) for m in mids}
    assert np.allclose(rp.deltas(f1, "R4", "tau2_own", "g6", mids), 0.01)
    assert np.allclose(rp.deltas(f1, "R4", "tau3_any", "g8", mids, "label_aerial"), -0.01)
    assert rp.select(f1, "R4", mids)[0] == "tau2_own"
    chosen, per_fold = rp.nested(f1, "R4")
    assert set(per_fold) == {"tau2_own"} and np.allclose(rp.oof_deltas(f1, "R4", chosen, "g6", "label_poss"), 0.01)
    assert np.allclose(rp.deltas(f1, "R2", "tau1.6", "g6", mids), 0.10)               # other candidates keep B3 as the base


def test_round4_verdict_follows_the_rule_literally():
    ok = lambda *a: rp.verdict4(*a)[1]                                                  # noqa: E731
    assert ok(-0.004, -0.002, 0.010, 0.020) == "ADOPTA"
    assert ok(-0.005, 0.001, 0.010, 0.020) == "ADOPTA"                                  # loses exactly 0.005: allowed
    assert ok(-0.006, 0.001, 0.010, 0.020) == "NO ADOPTA"                               # loses more than 0.005, up to 0.01: not adopted
    assert ok(-0.010, 0.001, 0.010, 0.020) == "NO ADOPTA"
    assert ok(-0.011, 0.001, 0.010, 0.020) == "REVISION MANUAL"                         # loses more than 0.01: manual review
    assert ok(-0.011, 0.001, -0.5, -0.5) == "REVISION MANUAL"                           # whatever label_aerial says
    assert ok(0.000, 0.002, 0.010, -0.001) == "NO ADOPTA"                               # must win in g6 and in g8
    assert ok(0.000, 0.002, 0.0, 0.020) == "NO ADOPTA"                                  # a tie is not a win


def test_round4_prediction_is_a_postprocess_of_b3_and_matches_the_functions_of_absorcion():
    ball, slots, per, pitch = synth_match()
    m = rp.build_method_ctx(ball, slots, per, pitch)
    b3, tf, _ = rp.pipeline(m)
    assert rp.pred_r4(m, b3, "B3", None) is b3
    assert np.array_equal(rp.pred_r4(m, b3, "R2", (1.6,)), rp.pipeline(m, r2=1.6)[0])
    cf, ct = m["ctrl"]
    r4 = rp.pred_r4(m, b3, "R4", (1.6, "any"))
    assert np.array_equal(r4, ab_cond(b3, cf, ct, m["per"]))
    assert (r4 != b3).sum() <= (rp.pred_r4(m, b3, "R2", (1.6,)) != b3).sum()


def ab_cond(b3, cf, ct, per):
    import absorcion
    return absorcion.absorb_conditional(b3, 1.6, cf, np.where(ct == HOME, "H", "A"), "any", per)
