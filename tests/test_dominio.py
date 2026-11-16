"""bancos/dominio.py: the reference calculation (univariate AUC from ranks, folds, rule, risk list) on synthetic data."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("bancos", "src"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import dominio as dm  # noqa: E402
import pff_split  # noqa: E402
import ronda_pff as rp  # noqa: E402


def test_univariate_auc_matches_sklearn_with_ties_and_nan():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    X = rng.integers(0, 5, size=(400, 3)).astype(float)
    X[rng.random(X.shape) < 0.2] = np.nan
    gid = np.repeat(np.array(["a", "b", "c", "d"]), 100)
    got = dm.univariate_vs_rest(X, gid, ["a", "c"])
    for t in ("a", "c"):
        y = (gid == t).astype(int)
        for j in range(3):
            col = np.where(np.isfinite(X[:, j]), X[:, j], -1e9)                   # NaN below every finite value
            a = roc_auc_score(y, col)
            assert got[t][j] == pytest.approx(max(a, 1 - a), abs=1e-12)


def test_reference_separates_a_shifted_match_and_is_direction_free():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(600, 2))
    gid = np.repeat(np.array(["a", "b", "c"]), 200)
    X[gid == "c", 0] -= 3.0                                                       # the match c is shifted down on feature 0
    r = dm.univariate_vs_rest(X, gid, ["a", "c"])
    assert r["c"][0] > 0.95 and r["c"][1] < 0.6                                   # the sign does not matter, the null feature stays near 0.5
    assert r["a"][1] < 0.6 and 0.6 < r["a"][0] < 0.85                            # a sits above the pool that contains the shifted c; its null feature is near 0.5
    Xt = X[gid == "c"]
    assert dm.univariate_vs_pool(Xt, X[gid != "c"])[0] > 0.95


def test_time_blocks_interleaved_and_balanced():
    f = dm.time_blocks(5 * 3 * dm.BLOCK)
    assert set(f.tolist()) == set(range(dm.K))
    assert (np.bincount(f) == 3 * dm.BLOCK).all()
    assert (f[:dm.BLOCK] == 0).all() and (f[dm.BLOCK:2 * dm.BLOCK] == 1).all()


def test_group_folds_follow_rp_folds_and_drop_the_target():
    mids = np.array(pff_split.DESARROLLO)
    f = dm.group_folds(mids)
    assert (f >= 0).all() and sorted(set(f.tolist())) == list(range(dm.K))
    assert all(f[list(mids).index(m)] == k for k, fo in enumerate(rp.FOLDS) for m in fo)
    assert (dm.group_folds(mids, target=mids[3])[3] == -1) and (dm.group_folds(mids, target=mids[3]) >= 0).sum() == len(mids) - 1


def test_rule_and_verdict():
    auc = np.array([0.90, 0.90, 0.58, 0.70])
    p90 = np.array([0.70, 0.85, 0.50, 0.55])
    exc, mk = dm.exceed_marks(auc, p90, important=[True, True, True, False], risk=[False, False, False, False])
    assert exc.round(2).tolist() == [0.20, 0.05, 0.08, 0.15]
    assert mk.tolist() == [True, False, False, False]                             # the last one exceeds but is neither important nor risk
    mk2 = dm.exceed_marks(auc, p90, [False] * 4, [False, False, False, True])[1]
    assert mk2.tolist() == [False, False, False, True]
    assert dm.verdict(0, 0.80, 0.85)[0] == "transferible"
    assert dm.verdict(0, 0.90, 0.85)[0] == "con desplazamiento"
    assert dm.verdict(2, 0.80, 0.85)[0] == "con desplazamiento"


def test_risk_list_takes_the_windows_of_the_risk_bases():
    names = ["ball_present", "ball_present@p0.2", "ball_since_visible_s", "ball_until_visible_s", "b3_dead", "b3_dead@f2", "b3abs_dead@p1",
             "ball_speed_ms", "ball_speed@f2", "sgn__b3", "touch_since_s"]
    assert dm.risk_features(names) == names[:7]
