"""Synthetic tests for the gap-length calibration and the coupled_len gap generator (bancos/dfl_degradado.py)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "bancos"))
sys.path.insert(0, os.path.join(ROOT, "src"))
import dfl_degradado as dd  # noqa: E402


def test_gap_records_classes_and_borders():
    absent = np.zeros(2000, bool)
    absent[100:200] = True            # 4 s -> "<5 s", too short to keep frames (> 2 s of borders are dropped: 50 kept)
    absent[300:550] = True            # 10 s -> "10-20 s"
    absent[1000:1500] = True          # 20 s -> ">20 s"
    spd = np.full(2000, 1.5)
    spd[300:550] = 3.0
    r = dd.gap_records(absent, [(0, 2000)], spd, label=np.where(absent, "D", "H"))
    assert r["cls"].tolist() == [0, 2, 3]
    assert r["kept"].tolist() == [50, 200, 450]
    assert r["hist"].sum(1).tolist() == [50, 200, 450]
    assert np.isclose(r["ssum"][1], 3.0 * 200)
    assert r["d_kept"].tolist() == [50, 200, 450] and r["d_all"].tolist() == [100, 250, 500]


def test_mix_dead_recovers_the_share():
    f_dead = np.zeros(50); f_dead[10] = 1.0
    f_alive = np.zeros(50); f_alive[30] = 1.0
    assert dd.mix_dead(0.25 * f_dead + 0.75 * f_alive, f_dead, f_alive) == 0.25


def _largo(pd):
    run = [0] * len(dd.EDGES)
    cl = []
    for c, (n, f) in enumerate(zip((10, 50, 30, 40), (0.001, 0.06, 0.07, 0.27))):
        h = list(run)
        h[[12, 14, 16, 22][c]] = n          # one bin per class (EDGES 87, 173, 344, 1920 frames)
        cl.append({"n_runs": n, "frac_live": f, "run_hist": h, "lo_frames": ([0] + list(dd.LEN_CUTS))[c],
                   "hi_frames": (list(dd.LEN_CUTS) + [None])[c], dd.LEN_WIN: {"pd": pd[c]}})
    return {"classes": cl}


def test_len_model_global_p_and_class_weights():
    pd = (1.0, 0.5, 0.9, 1.0)
    p, draw = dd._len_gap_model(_largo(pd))
    f = np.array([0.001, 0.06, 0.07, 0.27])
    assert np.isclose(p, (f * np.array(pd)).sum() / f.sum())
    rng = np.random.default_rng(0)
    cls = np.digitize([draw(rng) for _ in range(4000)], dd.LEN_CUTS)
    assert set(cls) == {1, 2}                                       # pd = 1 classes never draw
    assert abs((cls == 1).mean() - 50 * 0.5 / (50 * 0.5 + 30 * 0.1)) < 0.03


def test_coupled_len_hides_dead_stretches_and_adds_in_play_gaps():
    gaps = {"ball_absent_frac_live": 0.3, "n_absent_runs": 0, "live_frames": 100000, "absent_run_hist": [1] + [0] * (len(dd.EDGES) - 1),
            "present_run_hist": [0] * len(dd.EDGES), "dependence": {"p_dead_given_absent": 0.985}}
    dead = np.zeros(60000, bool)
    dead[::3000] = True
    dead = np.convolve(dead, np.ones(500), "same") > 0                 # 20 s dead stretches every 2 min
    ab = dd.ball_gaps(dead, np.random.default_rng(1), gaps, "coupled_len", largo=_largo((1.0, 0.5, 0.9, 1.0)))
    assert (ab & dead).any() and (ab & ~dead).any()                   # hides dead stretches and adds in-play gaps


def _runs_of(mask):
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    a, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return a, e - a


def test_coupled_len_in_play_gaps_do_not_cross_dead_stretches_nor_touch_other_gaps():
    # live stretches of 40 s between dead stretches of 10 s; gaps drawn only 5-10 s and 10-20 s
    gaps = {"ball_absent_frac_live": 0.35, "n_absent_runs": 0, "live_frames": 100000, "absent_run_hist": [1] + [0] * (len(dd.EDGES) - 1),
            "present_run_hist": [0] * len(dd.EDGES), "dependence": {"p_dead_given_absent": 0.985}}
    dead = np.zeros(80000, bool)
    for a in range(1000, 80000, 1250):
        dead[a:a + 250] = True
    for seed in range(5):
        info = {}
        ab = dd.ball_gaps(dead, np.random.default_rng(seed), gaps, "coupled_len", largo=_largo((1.0, 0.5, 0.9, 1.0)), info=info)
        assert info["placed"] >= 5 and info["out_of_class"] == 0
        st, ln = _runs_of(ab)
        live_runs = [(a, n) for a, n in zip(st, ln) if not dead[a:a + n].any()]     # gaps with no dead frame: the in-play ones
        assert len(live_runs) == info["placed"]
        assert all(np.digitize(n, dd.LEN_CUTS) in (1, 2) for _, n in live_runs)    # none merged into another length
        assert all(dead[a:a + n].all() or not dead[a:a + n].any() for a, n in zip(st, ln))   # no gap mixes dead and live frames


def test_coupled_len_rejects_forced_pd_when_all_classes_are_dead():
    import pytest
    gaps = {"ball_absent_frac_live": 0.3, "n_absent_runs": 0, "live_frames": 100000, "absent_run_hist": [1] + [0] * (len(dd.EDGES) - 1),
            "present_run_hist": [0] * len(dd.EDGES), "dependence": {"p_dead_given_absent": 0.985}}
    dead = np.zeros(20000, bool)
    dead[5000:5500] = True
    with pytest.raises(ValueError, match="P\\(dead \\| absent\\) = 1"):
        dd.ball_gaps(dead, np.random.default_rng(0), gaps, "coupled_len", p_dead_given_absent=0.9, largo=_largo((1.0, 1.0, 1.0, 1.0)))
