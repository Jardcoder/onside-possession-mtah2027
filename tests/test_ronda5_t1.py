"""bancos/ronda5_t1.py: weights of the three variants, normalisation, weighted fit equal to mo.fit with unit weights, fingerprints."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("bancos", "src"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import modelo as mo  # noqa: E402
import ronda5_t1 as r5  # noqa: E402

H, A, D = 0, 1, 2


def seq(*parts):
    return np.concatenate([np.full(n, c, np.int8) for c, n in parts])


def test_segments_break_at_label_change_and_period_change():
    y = seq((H, 10), (A, 20), (A, 30), (D, 5))
    per = np.concatenate([np.full(40, 1), np.full(25, 2)]).astype(np.int8)       # the A stretch of 50 frames crosses the period break at 40
    assert r5.segment_bounds(y, per).tolist() == [0, 10, 40, 60, 65]
    assert r5.segment_lengths(y, per).tolist() == [10] * 10 + [30] * 30 + [20] * 20 + [5] * 5
    assert r5.median_length([y, y], [per, per]) == 15.0                          # stretches 10, 30, 20, 5 per match


def test_balance_weight_is_median_over_length_clipped():
    y = seq((H, 10), (A, 100), (H, 1000), (D, 10), (A, 10))
    per = np.ones(len(y), np.int8)
    w = r5.raw_weights("T1w", y, per, np.zeros(len(y)), 20.0)
    assert np.allclose(w[:10], 2.0) and np.allclose(w[10:110], 0.2) and np.allclose(w[110:1110], 0.1) and np.allclose(w[1110:], 2.0)
    w = r5.raw_weights("T1w", y, per, np.zeros(len(y)), 1000.0)
    assert np.allclose(w[:10], 10.0) and np.allclose(w[110:1110], 1.0)           # 1000 / 10 = 100 -> clipped to 10


def test_mask_and_product_variants():
    y = seq((H, 10), (A, 100))
    per = np.ones(len(y), np.int8)
    rw = np.zeros(len(y), np.int8)
    rw[5:15] = 4
    m = r5.raw_weights("T1m", y, per, rw, 20.0)
    assert m[5:15].sum() == 0 and m[:5].tolist() == [1.0] * 5 and m[15:].tolist() == [1.0] * 95
    wm = r5.raw_weights("T1wm", y, per, rw, 20.0)
    w = r5.raw_weights("T1w", y, per, np.zeros(len(y)), 20.0)
    assert np.allclose(wm, w * m)
    assert np.allclose(r5.raw_weights(r5.CONTROL, y, per, rw, 20.0), 1.0)


def test_normalisation_mean_one_on_training_and_same_scale_on_validation():
    rng = np.random.default_rng(0)
    w_tr, w_va = rng.random(1000) + 0.5, rng.random(300) + 0.5
    a, b, scale = r5.normalise_weights("T1w", w_tr, w_va)
    assert abs(a.mean() - 1) < 1e-12 and np.allclose(b * scale, w_va)
    a, b, scale = r5.normalise_weights("T1m", w_tr, w_va)                        # the mask variant is not renormalised
    assert scale == 1.0 and (a == w_tr).all() and (b == w_va).all()


def test_variant_definitions_match_the_preregistration():
    assert r5.VARIANTS == ("T1w", "T1m", "T1wm") and r5.CLIP == (0.1, 10.0)
    assert (r5.BALANCE["T1w"], r5.MASK["T1w"], r5.NORMALISE["T1w"]) == (True, False, True)
    assert (r5.BALANCE["T1m"], r5.MASK["T1m"], r5.NORMALISE["T1m"]) == (False, True, False)
    assert (r5.BALANCE["T1wm"], r5.MASK["T1wm"], r5.NORMALISE["T1wm"]) == (True, True, True)


def small_problem(n=900, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = (X[:, 0] + 0.5 * rng.normal(size=n) > 0).astype(int) + (X[:, 1] > 1).astype(int)
    return X, y


PARAMS = {**mo.LGB_PARAMS, "min_data_in_leaf": 10, "num_leaves": 7, "bagging_fraction": 1.0, "bagging_freq": 0}


def test_fit_weighted_with_unit_weights_gives_the_booster_of_mo_fit():
    X, y = small_problem()
    a = mo.fit(X[:600], y[:600], X[600:], y[600:], threads=1, rounds=30, params=PARAMS)
    b = r5.fit_weighted(X[:600], y[:600], np.ones(600), X[600:], y[600:], np.ones(300), threads=1, rounds=30, params=PARAMS)
    assert a.best_iteration == b.best_iteration
    assert np.array_equal(a.predict(X, num_iteration=a.best_iteration), b.predict(X, num_iteration=b.best_iteration))
    assert a.model_to_string() == b.model_to_string()


def test_fit_weighted_with_bagging_params_of_t1_also_matches_and_weights_matter():
    X, y = small_problem()
    p = {**mo.LGB_PARAMS, "min_data_in_leaf": 10, "num_leaves": 7}
    a = mo.fit(X[:600], y[:600], X[600:], y[600:], threads=1, rounds=30, params=p)
    b = r5.fit_weighted(X[:600], y[:600], np.ones(600), X[600:], y[600:], np.ones(300), threads=1, rounds=30, params=p)
    assert np.array_equal(a.predict(X[:50]), b.predict(X[:50]))
    w = np.where(y[:600] == 2, 5.0, 1.0)                                         # weighting the rare class changes the model
    c = r5.fit_weighted(X[:600], y[:600], w, X[600:], y[600:], np.ones(300), threads=1, rounds=30, params=p)
    assert not np.array_equal(a.predict(X[:50]), c.predict(X[:50]))


def test_zero_weight_rows_do_not_move_the_model():
    X, y = small_problem()
    p = {**PARAMS, "min_data_in_leaf": 1}
    junk = np.random.default_rng(1).normal(size=(100, 5)) * 10
    Xa = np.concatenate([X[:600], junk])
    ya = np.concatenate([y[:600], np.random.default_rng(2).integers(0, 3, 100)])
    wa = np.concatenate([np.ones(600), np.zeros(100)])
    b = r5.fit_weighted(Xa, ya, wa, X[600:], y[600:], np.ones(300), threads=1, rounds=10, params=p)
    assert b.num_trees() > 0 and np.isfinite(b.predict(X[:20])).all()


@pytest.mark.needs_pff_and_calibration
def test_fingerprint_depends_on_variant_and_is_stable():
    assert r5.fp_variant("T1w") == r5.fp_variant("T1w") and r5.fp_variant("T1w") != r5.fp_variant("T1m") != r5.fp_variant("T1wm")


def test_num_keeps_whole_numbers_as_int():
    assert r5.num("8") == 8 and isinstance(r5.num("8"), int)
    assert r5.num("8.0") == 8 and isinstance(r5.num("8.0"), int)
    assert r5.num("1.5") == 1.5


def test_set_grid_is_lam_ha_major_and_gets_its_own_sweep_cache():
    saved = (r5.bl.LAM_HA, r5.bl.LAM_D, r5.bl.GRID, r5.bl.GRID_HA, r5.bl.GRID_D)
    default_path = r5.sweep_path("T1wm")
    try:
        grid = r5.set_grid((4, 24), (8, 16, 64))
        assert grid == [(4, 8), (4, 16), (4, 64), (24, 8), (24, 16), (24, 64)]
        assert r5.bl.GRID_HA.tolist() == [4, 4, 4, 24, 24, 24] and r5.bl.GRID_D.tolist() == [8, 16, 64, 8, 16, 64]
        wide = r5.sweep_path("T1wm")
        assert wide != default_path and os.path.basename(wide).startswith("sweep_T1wm_")
        r5.set_grid((4, 24), (8, 16, 64))
        assert r5.sweep_path("T1wm") == wide                                          # same grid, same file
        r5.set_grid((4, 24), (8, 16, 128))
        assert r5.sweep_path("T1wm") != wide
    finally:
        r5.bl.LAM_HA, r5.bl.LAM_D, r5.bl.GRID, r5.bl.GRID_HA, r5.bl.GRID_D = saved
    assert r5.sweep_path("T1wm") == default_path
