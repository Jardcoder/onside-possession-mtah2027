"""M1 (src/modelo.py, bancos/modelo_pff.py): Viterbi on toy cases, team symmetry, and no ground truth on the method side."""
import inspect
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("bancos", "src"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import modelo as mo  # noqa: E402
import modelo_pff as mp_  # noqa: E402
import possession_metrics as pm  # noqa: E402

FORBIDDEN = re.compile(r"ball_status|is_touch|uncertain|video_missing|ball_visible|ball_z|players_visible|label_poss|label_ut"
                       r"|attack_right|\[\"label\"\]|\['label'\]|gt_label|gt_poss")


def probs(*parts):
    """(N, 3) probabilities from [(state, n, confidence)]: the state gets `confidence`, the others share the rest."""
    rows = []
    for s, n, c in parts:
        p = np.full(3, (1 - c) / 2)
        p[s] = c
        rows += [p] * n
    return np.array(rows)


def test_viterbi_zero_penalty_is_the_argmax_per_frame():
    rng = np.random.default_rng(1)
    P = rng.dirichlet([1, 1, 1], 3000)
    per = np.repeat([1, 2], 1500)
    path = mo.viterbi(np.log(P), per, 0, 0)
    assert path.shape == (1, 3000)
    assert (path[0] == P.argmax(1)).all()


def test_high_penalty_suppresses_short_changes_and_keeps_long_ones():
    P = probs((0, 60, 0.7), (1, 3, 0.7), (0, 60, 0.7), (1, 80, 0.7))
    per = np.ones(len(P), int)
    raw = mo.viterbi(np.log(P), per, 0, 0)[0]
    smooth = mo.viterbi(np.log(P), per, 8, 8)[0]
    assert (raw[60:63] == 1).all()
    assert (smooth[:123] == 0).all()                       # the 3-frame A is gone
    assert (smooth[123:] == 1).all()                       # the 80-frame A stays
    assert (np.diff(smooth) != 0).sum() == 1


def test_lam_d_acts_only_on_entering_and_leaving_dead():
    P = probs((0, 50, 0.7), (2, 3, 0.7), (0, 50, 0.7))
    per = np.ones(len(P), int)
    assert (mo.viterbi(np.log(P), per, 0, 0)[0][50:53] == 2).all()
    assert (mo.viterbi(np.log(P), per, 0, 8)[0] == 0).all()
    P = probs((0, 50, 0.7), (1, 3, 0.7), (0, 50, 0.7))
    assert (mo.viterbi(np.log(P), per[:len(P)], 0, 8)[0][50:53] == 1).all()       # lam_d does not hold back H <-> A


def test_viterbi_grid_equals_one_by_one_and_periods_are_independent():
    rng = np.random.default_rng(2)
    P = rng.dirichlet([2, 2, 1], 800)
    per = np.repeat([1, 2], 400)
    grid = mo.viterbi(np.log(P), per, [0, 4, 32], [2, 0, 16])
    for q, (a, b) in enumerate([(0, 2), (4, 0), (32, 16)]):
        assert (grid[q] == mo.viterbi(np.log(P), per, a, b)[0]).all()
    both = mo.viterbi(np.log(P), per, 8, 8)[0]
    assert (both[:400] == mo.viterbi(np.log(P[:400]), np.ones(400, int), 8, 8)[0]).all()
    assert (both[400:] == mo.viterbi(np.log(P[400:]), np.ones(400, int), 8, 8)[0]).all()


def toy_world(n=900, swap=False, seed=0):
    """A synthetic tracking with the same number of slots in both teams; `swap` exchanges the teams (slots, B3 labels, touches)."""
    rng = np.random.default_rng(seed)
    per = np.where(np.arange(n) < n // 2, 1, 2)
    ball = np.cumsum(rng.normal(0, 0.003, (n, 2)), 0) + 0.5
    ball[300:340] = np.nan
    H = np.clip(0.5 + rng.normal(0, 0.15, (n, 14, 2)), 0, 1)
    A = np.clip(0.5 + rng.normal(0, 0.15, (n, 14, 2)), 0, 1)
    H[:, 12:] = np.nan
    b3 = np.array(rng.choice(["H", "A", "D"], n, p=[0.4, 0.4, 0.2]), dtype="<U1")
    b3a = b3.copy()
    b3a[::7] = "D"
    tf = np.sort(rng.choice(np.arange(n), 40, replace=False))
    if swap:
        H, A = A, H
        sw = {"H": "A", "A": "H", "D": "D"}
        b3, b3a = (np.array([sw[x] for x in b3], dtype="<U1"), np.array([sw[x] for x in b3a], dtype="<U1"))
    return ball, {"H": H, "A": A}, per, (105.0, 68.0), b3, b3a, tf


def same(a, b):
    return np.allclose(np.nan_to_num(a, nan=-999), np.nan_to_num(b, nan=-999), atol=1e-4, rtol=1e-5)


def test_swap_twice_is_the_identity_and_names_are_paired():
    X, names = mo.build_features(*toy_world())
    assert X.shape == (900, len(names)) and len(set(names)) == len(names)
    assert same(mo.swap_features(mo.swap_features(X, names), names), X)
    perm, sign = mo.swap_spec(names)
    assert sorted(perm) == list(range(len(names))) and (perm[perm] == np.arange(len(names))).all()
    assert (sign < 0).sum() > 0 and (perm != np.arange(len(names))).sum() > 0


def test_features_of_the_swapped_world_are_the_swapped_features():
    X, names = mo.build_features(*toy_world())
    Xs, _ = mo.build_features(*toy_world(swap=True))
    assert same(Xs, mo.swap_features(X, names))


def test_labels_and_probabilities_swap():
    y = np.array([0, 1, 2, 1, 0])
    assert (mo.swap_labels(y) == [1, 0, 2, 0, 1]).all()
    assert (mo.swap_labels(mo.swap_labels(y)) == y).all()
    P = np.array([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6]])
    assert (mo.swap_probs(mo.swap_probs(P)) == P).all() and (mo.swap_probs(P)[0] == [0.3, 0.6, 0.1]).all()


def test_symmetric_inference_is_equivariant():
    """predict_proba(swap(x)) = swap(predict_proba(x)) for any model, because it averages the two views."""
    X, names = mo.build_features(*toy_world())

    class Fake:                                           # an asymmetric "model" that is not invariant by itself
        best_iteration = 0

        def predict(self, A, num_iteration=None):
            z = np.nan_to_num(A[:, :3], nan=0.0)
            e = np.exp(np.stack([z[:, 0], 0.3 * z[:, 1], z[:, 2] - z[:, 0]], 1))
            return e / e.sum(1, keepdims=True)
    f = Fake()
    p = mo.predict_proba(f, X, names)
    ps = mo.predict_proba(f, mo.swap_features(X, names), names)
    assert np.allclose(ps, mo.swap_probs(p))
    assert np.allclose(p.sum(1), 1)


def test_augment_doubles_the_rows_with_swapped_labels():
    X, names = mo.build_features(*toy_world())
    y = np.arange(len(X)) % 3
    Xa, ya = mo.augment(X, y, names)
    assert len(Xa) == 2 * len(X) and (ya[len(X):] == mo.swap_labels(y)).all() and same(Xa[len(X):], mo.swap_features(X, names))


def test_features_do_not_read_the_ground_truth():
    assert list(inspect.signature(mo.build_features).parameters) == ["ball", "slots", "per", "pitch", "b3", "b3a", "tf"]
    assert not FORBIDDEN.search(open(os.path.join(ROOT, "src", "modelo.py")).read())
    for fn in (mo.build_features, mo.viterbi, mo.predict_proba, mo.fit, mo.augment):
        assert not FORBIDDEN.search(inspect.getsource(fn))


def test_features_depend_only_on_their_arguments():
    a, b = mo.build_features(*toy_world(seed=3)), mo.build_features(*toy_world(seed=3))
    assert same(a[0], b[0])
    args = list(toy_world(seed=3))
    args[4] = np.array(["D"] * len(args[4]), dtype="<U1")                 # other B3 labels -> other features: they are inputs
    assert not same(mo.build_features(*args)[0], a[0])


def test_the_evaluation_loads_matches_only_through_the_guard():
    code = "\n".join(l for l in inspect.getsource(mp_).splitlines() if not l.lstrip().startswith("#"))
    assert "RESERVADOS" not in code.replace("set(pff_split.RESERVADOS)", "")
    assert "load_match(" in code and "pff_split.load_dev(" not in code.replace("rp.load_match", "")
    assert not re.search(r"^\s*import pff\b|\bpff\.load\(|from pff import|CHALLENGE_GT|read_label_csv", code, re.M)


def test_scorer_used_in_the_workers_equals_evaluate():
    rng = np.random.default_rng(4)
    gt = np.repeat(rng.integers(0, 3, 40), 50).astype(np.int8)
    gt[0] = 3
    pred = np.repeat(rng.integers(0, 3, 100), 20).astype(np.int8)
    runs = pm.valid_runs(gt)
    f1 = mp_.eval_stats(gt, pred, runs)[0]
    assert abs(f1 - pm.evaluate(gt, pred, bootstrap_n=0)["spell_f1_mean"]) < 1e-12


def test_pick_lambda_maximises_the_min_of_the_two_balls_and_breaks_ties_by_grid_order():
    f1 = np.zeros((3, 2, len(mp_.GRID)))
    f1[:, 0, 7] = 0.5
    f1[:, 1, 7] = 0.4
    f1[:, 0, 9] = 0.9
    f1[:, 1, 9] = 0.3                                     # best g6 but min = 0.3 < 0.4
    idx, obj = mp_.pick_lambda(f1)
    assert idx == 7 and abs(obj[7] - 0.4) < 1e-12
    f1[:, :, 12] = f1[:, :, 7]
    assert mp_.pick_lambda(f1)[0] == 7
