"""Border detector and variable-price Viterbi (src/borde.py, bancos/ronda_bordes.py, src/predecir_reto_m2.py), on synthetic data: the path with a unit price is
the one of `modelo.viterbi`, what the evaluation scores is what the prediction decodes, a detector never sees the fold it is scored on, a cheaper change near a peak lets a
short possession through, and the committed weights are checked against their SHA-256 and load from a checkout with CRLF line endings."""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("bancos", "src", "tests"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import borde  # noqa: E402
import modelo as mo  # noqa: E402
import pff_split  # noqa: E402
import predecir_reto_m2 as m2  # noqa: E402
import ronda_bordes as rb  # noqa: E402
import ronda_pff as rp  # noqa: E402
from test_modelo import toy_world  # noqa: E402


def probs_from_log(logp):
    return np.exp(np.asarray(logp, float))


# ------------------------------------------------------------------------------------------------ columns

def test_border_columns_drop_the_presence_family_and_its_windows():
    names = ["ball_x_m", "ball_present", "ball_present@p0.2", "ball_present@f2", "ball_since_visible_s", "ball_until_visible_s", "b3_dead",
             "b3_dead@p1", "b3abs_dead", "b3abs_dead@f0.5", "ball_speed_ms", "ball_speed_ms@p0.2", "sgn__nearest", "d1__H", "d1__A"]
    kept = [names[i] for i in borde.border_columns(names)]
    assert kept == ["ball_x_m", "ball_speed_ms", "ball_speed_ms@p0.2", "sgn__nearest", "d1__H", "d1__A"]


def test_the_presence_family_exists_under_those_names_in_the_real_feature_matrix():
    X, names = mo.build_features(*toy_world())
    for base in borde.PRESENCE:
        assert base in names, f"{base} is not a column of modelo.build_features any more: the exclusion of the presence family would silently stop working"
    dropped = [nm for i, nm in enumerate(names) if i not in set(borde.border_columns(names).tolist())]
    assert len(dropped) == 29 and all(nm.split("@")[0] in borde.PRESENCE for nm in dropped)       # 5 bases + their windows
    assert len(borde.border_columns(names)) == len(names) - 29


def test_column_index_keeps_the_order_asked_and_fails_loudly():
    names = ["a", "b", "c", "d"]
    assert borde.column_index(names, ["c", "a"]).tolist() == [2, 0]
    with pytest.raises(ValueError, match="not in the feature matrix"):
        borde.column_index(names, ["a", "z"])


# ------------------------------------------------------------------------------------------------ helpers

def test_transitions_are_found_inside_each_run_only():
    codes = np.array([0, 0, 1, 1, 2, 3, 3, 0, 1, 1])
    assert borde.transitions(codes, [(0, 5), (7, 10)]).tolist() == [2, 4, 8]       # the 3 -> 0 jump between the runs is not a transition
    assert borde.transitions(codes, []).tolist() == []


def test_near_marks_the_frames_within_k_of_an_event():
    assert borde.near(np.array([2, 9]), 1, 10).tolist() == [False, True, True, True, False, False, False, False, True, True]
    assert not borde.near(np.array([], np.int64), 3, 5).any()
    assert borde.near(np.array([-5, 50]), 2, 10).sum() == 0                             # events outside the array are ignored


def naive_peaks(p, threshold, w):
    n = len(p)
    cand = [p[t] >= threshold and p[t] >= p[max(0, t - w):min(n, t + w + 1)].max() for t in range(n)]
    return [t for t in range(n) if cand[t] and not (t > 0 and cand[t - 1] and p[t] == p[t - 1])]


def test_peaks_match_a_naive_definition():
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = np.round(rng.random(300), 1)                                                # coarse values make plateaus
        for w in (1, 3, 6):
            assert borde.peaks(p, 0.5, w).tolist() == naive_peaks(p, 0.5, w)


def test_peaks_take_the_first_frame_of_a_plateau_and_respect_the_threshold():
    p = np.array([0.1, 0.5, 0.5, 0.2, 0.9, 0.1, 0.3])
    assert borde.peaks(p, 0.4, 1).tolist() == [1, 4]
    assert borde.peaks(p, 0.95, 1).tolist() == []
    assert borde.peaks(np.zeros(0), 0.1, 3).tolist() == []
    assert borde.peaks(np.full(10, 0.5), 0.4, 3).tolist() == [0]                         # a constant probability is one plateau: one peak
    assert borde.peaks(np.full(10, 0.5), 0.6, 3).tolist() == []


def test_multipliers_follow_the_setting_and_never_cross_a_period():
    per = np.array([1] * 20 + [2] * 20)
    p = np.zeros(40)
    p[19] = 0.9                                                                         # last frame of period 1
    cfg = {"threshold": 0.5, "f": 0.25, "k": 3, "g": 2.5}
    m = borde.multipliers(p, [cfg, dict(cfg, f=0.1, k=1)], per)
    assert m.shape == (2, 40)
    assert m[0, 16:20].tolist() == [0.25] * 4 and (m[0, :16] == 2.5).all()
    assert (m[0, 20:] == 2.5).all()                                                     # the neighbourhood stops at the end of the period
    assert m[1, 18:20].tolist() == [0.1, 0.1] and m[1, 17] == 2.5
    with pytest.raises(ValueError, match="rows"):
        borde.multipliers(p, [cfg], per[:-1])


def test_the_config_is_json_serialisable():
    config = {"threshold": 0.15, "f": 0.25, "k": 3, "g": 2.5}
    assert json.loads(json.dumps(config)) == config
    assert all(json.loads(json.dumps(c)) == c for c in rb.GRID)


# ------------------------------------------------------------------------------------------------ the Viterbi

def test_viterbi_variable_with_a_unit_multiplier_is_the_path_of_modelo_viterbi():
    rng = np.random.default_rng(1)
    n = 900
    per = np.array([1] * 500 + [2] * 400)
    logp = np.log(rng.dirichlet([1, 1, 1], n))
    for lam in ((8.0, 16.0), (2.0, 3.0), (0.0, 0.0)):
        ref = mo.viterbi(logp, per, [lam[0]], [lam[1]])[0]
        out = borde.viterbi_variable(logp, per, lam[0], lam[1], np.ones((3, n)))
        assert all(np.array_equal(o, ref) for o in out)
    half = borde.viterbi_variable(logp, per, 8.0, 16.0, np.full((1, n), 0.25))[0]       # a uniform multiplier is a smaller lambda
    assert np.array_equal(half, mo.viterbi(logp, per, [2.0], [4.0])[0])


def test_viterbi_variable_edge_cases():
    assert borde.viterbi_variable(np.zeros((0, 3)), np.zeros(0, int), 1.0, 1.0, np.ones((2, 0))).shape == (2, 0)
    rng = np.random.default_rng(5)
    logp = np.log(rng.dirichlet([1, 1, 1], 6))
    per = np.array([1, 2, 2, 2, 2, 2])                                                  # a period of a single frame
    out = borde.viterbi_variable(logp, per, 3.0, 3.0, np.ones((1, 6)))[0]
    assert out[0] == logp[0].argmax() and np.array_equal(out, mo.viterbi(logp, per, [3.0], [3.0])[0])


def test_a_batch_of_multipliers_gives_the_same_paths_as_one_by_one():
    rng = np.random.default_rng(2)
    n = 400
    per = np.array([1] * 400)
    logp = np.log(rng.dirichlet([1, 1, 1], n))
    mult = rng.choice([0.25, 1.0, 2.0], size=(5, n))
    batch = borde.viterbi_variable(logp, per, 3.0, 5.0, mult)
    for c in range(5):
        assert np.array_equal(batch[c], borde.viterbi_variable(logp, per, 3.0, 5.0, mult[c:c + 1])[0])


def blip_case():
    """H for 110 frames with a 10 frame stretch (50..59) where A has a small advantage: 0.3 per frame, 3.0 in total."""
    n = 110
    logp = np.full((n, 3), -5.0)
    logp[:, 0] = 0.0
    logp[50:60, 0], logp[50:60, 1] = -0.3, 0.0
    return n, np.ones(n, np.int8), logp


def test_a_cheaper_change_near_the_peaks_lets_the_short_possession_through():
    n, per, logp = blip_case()
    P = probs_from_log(logp)
    lam = (2.0, 100.0)
    config = {"threshold": 0.5, "f": 0.25, "k": 1, "g": 1.0}
    p_border = np.zeros(n)
    p_border[[50, 60]] = 0.9                                                            # peaks where the two changes happen
    assert (borde.decode(P, per, lam, np.zeros(n), config) == 0).all()                  # no peaks: two changes cost 4 > 3, the blip is smoothed away
    path = borde.decode(P, per, lam, p_border, config)                                   # two changes cost 1 < 3: the blip stays
    assert (path[50:60] == 1).all() and (path[:50] == 0).all() and (path[60:] == 0).all()
    p_wrong = np.zeros(n)
    p_wrong[[20, 80]] = 0.9                                                             # peaks in the wrong place do nothing here
    assert (borde.decode(P, per, lam, p_wrong, config) == 0).all()


def test_decode_with_f_and_g_one_is_the_t1w_path():
    rng = np.random.default_rng(3)
    n = 700
    per = np.array([1] * 300 + [2] * 400)
    P = rng.dirichlet([1, 1, 1], n)
    p_border = rng.random(n)
    lam = (6.0, 12.0)
    ref = mo.viterbi(np.log(np.clip(P, 1e-9, 1.0)), per, [lam[0]], [lam[1]])[0]
    assert np.array_equal(borde.decode(P, per, lam, p_border, {"threshold": 0.3, "f": 1.0, "k": 3, "g": 1.0}), ref)


def test_decode_refuses_inputs_of_different_length():
    P = np.full((10, 3), 1 / 3)
    with pytest.raises(ValueError, match="rows"):
        borde.decode(P, np.ones(10, int), (1.0, 1.0), np.zeros(9), {"threshold": 0.5, "f": 1.0, "k": 1, "g": 1.0})


def test_the_periods_are_decoded_independently():
    n, per1, logp = blip_case()
    per = np.array([1] * 55 + [2] * 55)                                                 # the blip is cut by a period change
    P = probs_from_log(logp)
    p_border = np.zeros(n)
    p_border[[50]] = 0.9
    path = borde.decode(P, per, (2.0, 100.0), p_border, {"threshold": 0.5, "f": 0.0, "k": 1, "g": 1.0})
    assert (path[50:55] == 1).all()                                                     # free change at 50, the rest of the period stays A (no return inside period 1)
    assert path[55] == 0                                                                # period 2 starts again from its own evidence


# ------------------------------------------------------------------------------------------------ what is scored is what is deployed

def test_the_grid_of_the_evaluation_decodes_like_the_prediction():
    rng = np.random.default_rng(6)
    n = 600
    per = np.array([1] * 250 + [2] * 350)
    P = rng.dirichlet([1, 1, 1], n)
    logp = np.log(np.clip(P, 1e-9, 1.0))
    p = rng.random(n)
    lam = (5.0, 9.0)
    paths = rb.paths_of_grid(p, logp, per, lam)
    assert len(rb.GRID) == 720 and paths.shape == (720, n)
    for i in (0, 17, 95, 130, 191, 333, 479, 600, 719):
        assert np.array_equal(paths[i], borde.decode(P, per, lam, p, rb.GRID[i])), rb.GRID[i]


def test_a_detector_never_sees_the_fold_it_is_scored_on():
    mids = list(pff_split.DESARROLLO)
    assert not set(mids) & set(pff_split.RESERVADOS)
    for k, fold in enumerate(rp.FOLDS):
        keys = rb.keys_without_fold(mids, fold)
        assert not {m for m, _ in keys} & set(fold)
        assert {m for m, _ in keys} == set(mids) - set(fold) and len(keys) == 2 * (len(mids) - len(fold))      # both balls of every other match


def test_every_match_is_scored_with_the_detector_of_its_own_fold(monkeypatch):
    monkeypatch.setattr(rb, "_DETECTORS", {k: f"detector of fold {k}" for k in range(len(rp.FOLDS))})
    for k, fold in enumerate(rp.FOLDS):
        for mid in fold:
            assert rb.fold_detector(mid) == f"detector of fold {k}"
            assert mid not in {m for m, _ in rb.keys_without_fold(list(pff_split.DESARROLLO), rp.FOLDS[rp.fold_of(mid)])}


# ------------------------------------------------------------------------------------------------ the committed weights

def tiny_detector(tmp_path, names=("a", "b", "c", "d"), **meta_overrides):
    lgb = pytest.importorskip("lightgbm")
    rng = np.random.default_rng(4)
    X = rng.random((500, 4))
    y = (X[:, 0] > 0.5).astype(int)
    booster = lgb.train({"objective": "binary", "verbosity": -1, "num_leaves": 4}, lgb.Dataset(X, label=y), 5)
    model = tmp_path / borde.MODEL_FILE
    booster.save_model(str(model))
    meta = {"model": borde.MODEL_FILE, "model_sha256": borde.sha256_text(str(model)), "feature_names": list(names), "config": {},
            "peak_window": borde.PEAK_WINDOW, "target_tolerance": borde.TARGET_TOL, "entry_rule_passes": True, "reference_peaks_per_min": 1.0}
    meta.update(meta_overrides)
    (tmp_path / borde.META_FILE).write_text(json.dumps(meta))
    return X, booster, meta, model


def test_the_committed_weights_are_checked_against_their_sha256(tmp_path):
    X, booster, meta, model = tiny_detector(tmp_path)
    loaded, got = borde.load_detector(str(tmp_path))
    assert np.allclose(loaded.predict(X), booster.predict(X)) and got["feature_names"] == list("abcd")
    model.write_text(model.read_text() + "\n# edited")
    with pytest.raises(ValueError, match="SHA-256"):
        borde.load_detector(str(tmp_path))


def test_the_weights_load_from_a_checkout_with_crlf_line_endings(tmp_path):
    X, booster, meta, model = tiny_detector(tmp_path)
    model.write_bytes(model.read_bytes().replace(b"\n", b"\r\n"))                       # what `core.autocrlf=true` leaves on Windows; LightGBM cannot read this file itself
    loaded, _ = borde.load_detector(str(tmp_path))
    assert np.allclose(loaded.predict(X), booster.predict(X))


def test_weights_trained_with_other_settings_are_refused(tmp_path):
    tiny_detector(tmp_path, peak_window=borde.PEAK_WINDOW + 1)
    with pytest.raises(ValueError, match="peak_window"):
        borde.load_detector(str(tmp_path))


def test_the_prediction_feeds_the_detector_the_columns_it_was_trained_on(tmp_path):
    X, booster, meta, model = tiny_detector(tmp_path, names=("c", "a", "b", "d"))
    shuffled = ["a", "b", "c", "d", "extra"]                                             # a feature matrix whose layout is not the one of the training
    XX = np.column_stack([X[:, 1], X[:, 2], X[:, 0], X[:, 3], np.zeros(len(X))])
    p = m2.border_probability(booster, meta, XX, shuffled)
    assert np.allclose(p, booster.predict(np.column_stack([XX[:, 2], XX[:, 0], XX[:, 1], XX[:, 3]]), num_threads=1))
    with pytest.raises(ValueError, match="not in the feature matrix"):
        m2.border_probability(booster, meta, XX, ["a", "b", "c", "z", "extra"])


def test_changed_fraction_counts_only_the_masked_frames():
    a, b = np.array(list("HHAAD")), np.array(list("HAAAA"))
    assert m2.changed_fraction(a, b, np.array([1, 1, 1, 0, 0], bool)) == pytest.approx(1 / 3)
    assert m2.changed_fraction(a, b, np.ones(5, bool)) == pytest.approx(2 / 5)


def test_provenance_uses_the_committed_sums_for_committed_boosters():
    assert m2.provenance_warnings({"t1w_boosters_sha256": ["other machine"], "lambda_file_sha256": "x"}, False) == []
    w = m2.provenance_warnings({}, False, cache=True)     # the cache files are absent or differ from empty metadata
    assert all(x.startswith("WARNING") for x in w)
