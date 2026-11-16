"""bancos/modelo_transfer.py: preregistered rule, control, no ground truth in the method inputs, features of a no-hole ball equal the M1 cache."""
import inspect
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("bancos", "src"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import modelo as mo  # noqa: E402
import modelo_pff as mp_  # noqa: E402
import modelo_transfer as mt  # noqa: E402
import pff_split  # noqa: E402
import ronda_pff as rp  # noqa: E402


def st(mean, lo):
    return {"mean": mean, "lo": lo, "hi": mean + 0.02, "median": mean, "wins": 30, "losses": 5, "ties": 5, "sd": 0.02}


def test_preregistered_rule():
    assert mt.transfer_verdict(st(0.08, 0.05), st(0.06, 0.03))[0] == "TRANSFERIBLE"
    assert mt.transfer_verdict(st(0.08, 0.05), st(0.04, 0.03))[0] == "GANANCIA REAL PERO FRAGIL"
    assert mt.transfer_verdict(st(0.08, 0.05), st(0.06, -0.01))[0] == "GANANCIA REAL PERO FRAGIL"
    assert mt.transfer_verdict(st(0.08, 0.05), st(0.0, -0.02))[0] == "EL SALTO ERA EL ATAJO"
    assert mt.transfer_verdict(st(-0.01, -0.03), st(0.1, 0.05))[0] == "EL SALTO ERA EL ATAJO"


def test_control_tolerance():
    rows = [{"modelo": "T0", "balon": b, "partido": m, "spell_f1": v} for b, v in (("g6", 0.6160), ("g8", 0.6125 + 0.02)) for m in pff_split.DESARROLLO]
    ok = {b: o for b, _, _, _, o in mt.control(rows)}
    assert ok == {"g6": True, "g8": False}


def test_build_one_h_passes_no_label_to_the_methods():
    src = inspect.getsource(mt.build_one_h)
    assert src.count('d["label"]') == 2                      # inject_holes (where a hole may open) and the meta (training target)
    assert "gt_" not in inspect.getsource(mo.build_features) and '["label"]' not in inspect.getsource(mo.build_features)


def test_only_development_matches_and_own_cache_dir():
    assert mt.CACHE.endswith(os.path.join("modelo_cache", "transfer")) and mt.CACHE != mp_.CACHE
    assert not set(rp.FOLDS[0]) & set(pff_split.RESERVADOS)


def test_features_of_a_ball_without_holes_equal_the_m1_cache():
    mid = "3812"
    if not (os.path.exists(mp_.path_x(mid, "g6")) and os.path.exists(mp_.path_meta(mid, "g6"))):
        pytest.skip("M1 cache not present")
    import absorcion as ab
    d = rp.load_match(mid)
    ball = rp.ball_variant(d, "g6")
    m = rp.build_method_ctx(ball, d["players_slots"], d["period"], d["pitch"])
    b3, tf, _ = rp.pipeline(m)
    b3a = [ab.absorb(b3, t, d["period"]) for t in mp_.TAUS]                      # what build_one_h does
    for t, x in zip(mp_.TAUS, b3a):
        assert (x == rp.pipeline(m, r2=t)[0]).all()                              # same as running the whole pipeline with r2
    X, names = mo.build_features(ball, d["players_slots"], d["period"], d["pitch"], b3, b3a[0], tf)
    Xc = np.load(mp_.path_x(mid, "g6"))
    assert list(mp_.load_meta(mid, "g6")["names"]) == names
    assert np.array_equal(np.nan_to_num(X, nan=-9e9), np.nan_to_num(Xc, nan=-9e9))
