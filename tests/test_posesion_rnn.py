"""Synthetic checks of bancos/posesion_rnn.py (approach A30): features, targets, network, prediction windows, decoding."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("bancos", "src"):
    sys.path.insert(0, os.path.join(ROOT, sub))
torch = pytest.importorskip("torch")
import posesion_rnn as pr  # noqa: E402

CFG = {"seed": 1, "hidden": 8, "layers": 2, "dropout": 0.0, "stride": 4, "chunk": 40, "batch": 2, "steps": 1, "epochs": 1, "patience": 1,
       "lr": 1e-3, "weight_decay": 0.0, "touch_weight": 5.0, "win": 20, "margin": 5, "infer_batch": 4, "short_ref": 4.0, "short_max": 4.0}


def synthetic(n, seed):
    rng = np.random.default_rng(seed)
    per = np.repeat([1, 2], n // 2)
    H = np.clip(rng.normal(0.35, 0.1, (n, 24, 2)), 0, 1)
    A = np.clip(rng.normal(0.65, 0.1, (n, 22, 2)), 0, 1)
    H[:, 11:] = np.nan
    A[:, 11:] = np.nan
    H[n // 2:, :, 0] = 1 - H[n // 2:, :, 0]
    A[n // 2:, :, 0] = 1 - A[n // 2:, :, 0]
    ball = rng.uniform(0, 1, (n, 2))
    ball[rng.random(n) < 0.3] = np.nan
    return ball, {"H": H, "A": A}, per


def test_orientation_follows_half_time_switch():
    _, slots, per = synthetic(400, 0)
    s = pr.orientation(slots, per)
    assert (s[per == 1] == 1).all() and (s[per == 2] == -1).all()


def test_features_shape_finite_and_team_swap_mirrors():
    ball, slots, per = synthetic(300, 1)
    X = pr.build_features(ball, slots, per, (105.0, 68.0))
    Xs = pr.build_features(ball, pr.swapped(slots), per, (105.0, 68.0))
    assert X.shape == (300, 38) and np.isfinite(X).all()
    np.testing.assert_allclose(Xs[:, 10 + 5:10 + 8], X[:, 24 + 5:24 + 8], atol=1e-6)
    np.testing.assert_allclose(Xs[:, 2], -X[:, 2], atol=1e-6)


def test_last_seen_resets_at_period_start():
    present = np.array([1, 0, 0, 0, 1, 0], bool)
    per = np.array([1, 1, 1, 2, 2, 2])
    np.testing.assert_array_equal(pr.last_seen(present, per), [0, 0, 0, -1, 4, 4])


def test_touch_targets_and_swaps():
    label = np.array(list("DDHHHAAADD"))
    f, t = pr.touch_teams(label, np.array([1, 5, 8]))
    np.testing.assert_array_equal(f, [1, 5])
    y = pr.touch_targets(10, f, t)
    np.testing.assert_array_equal(y, [1, 1, 1, 2, 2, 2, 2, 2, 0, 0])
    np.testing.assert_array_equal(pr.swap_touch(y)[:4], [2, 2, 2, 1])
    np.testing.assert_array_equal(pr.swap_codes(np.array([0, 1, 2])), [1, 0, 2])


def test_network_prediction_is_a_distribution_per_head():
    net = pr.Net(5, CFG)
    X = np.random.default_rng(0).normal(size=(130, 5)).astype(np.float32)
    per = np.repeat([1, 2], 65)
    P = pr.predict_tta(net, X, X, per, CFG, torch.device("cpu"))
    assert P.shape == (130, 6)
    np.testing.assert_allclose(P[:, :3].sum(1), 1, atol=1e-5)
    np.testing.assert_allclose(P[:, 3:].sum(1), 1, atol=1e-5)


def test_decode_follows_clear_probabilities():
    P = np.tile([0.9, 0.05, 0.05], (50, 1))
    P[25:] = [0.05, 0.05, 0.9]
    codes = pr.decode(P, np.ones(50, int), (4.0, 16.0))
    assert (codes[:20] == 0).all() and (codes[30:] == 2).all()


def test_short_spell_weights_favour_short_runs():
    y = np.array([0] * 25 + [1] * 200)                    # a 1 s run then an 8 s run at 25 fps
    w = pr.short_spell_weights(y, np.ones(len(y), int), CFG)
    assert w[0] == 4.0 and w[-1] == 1.0
