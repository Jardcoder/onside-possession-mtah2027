"""PFF ball calibration (fixed metric set, common margin), label noise and extra touches: synthetic checks."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("bancos", "src"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import calibrar_balon as cb  # noqa: E402
import diagnostico_pff_sobrantes as ds  # noqa: E402
import evaluar_pff as ep  # noqa: E402
import evaluar_toques as ev  # noqa: E402
import ruido_gt_pff as rg  # noqa: E402


def _m(vals, lag1):
    k = cb.LOG_KEYS
    return {k[0]: np.array(vals[0:3]), k[1]: np.array(vals[3:6]), k[2]: vals[6], k[3]: vals[7], k[4]: vals[8],
            cb.DIFF_KEYS[0]: lag1}


def test_distance_uses_the_same_ten_terms_whatever_the_values():
    ref = _m([5, 14, 22, 1, 3, 5, 0.03, 1.2, 3.3], 0.2)
    zero_turns = _m([5, 14, 22, 1, 3, 5, 0.03, 1.2, 0.0], -0.4)       # zero count and negative lag1: nothing is dropped
    e = cb.errores(zero_turns, ref)
    assert len(e) == len(cb.TERMS) == 10
    assert np.isclose(e[-1], -0.6)                                     # lag1 is a plain difference
    assert np.isclose(e[8], np.log(cb.FLOOR / 3.3))                    # explicit floor for the zero
    assert np.isclose(cb.distancia(ref, ref), 0.0)
    assert np.isclose(cb.distancia(zero_turns, ref), np.abs(e).mean())


def test_common_margin_is_the_most_demanding_and_applies_to_every_mode():
    cm = cb.common_margin()
    for mode in (None,) + cb.PFF_MODES:
        l, r = cb.margins(mode)
        assert l <= cm[0] and r <= cm[1]
    assert cm == (cb.margins("causal")[0], cb.margins("g10")[1])
    rng = np.random.default_rng(1)
    n = 400
    ball = np.cumsum(rng.normal(0, 0.001, (n, 2)), 0) + 0.5
    ball[150:160] = np.nan
    data = {"m": (ball, np.ones(n, int), (105.0, 68.0))}
    runs = {mode: cb.banco_pff(data, mode, cm)["m"][1] for mode in (None, "g4", "g8", "causal")}
    assert all(r == runs[None] for r in runs.values())                 # same stretches in every mode


def test_delta_stats_adds_median_and_the_wider_interval():
    s = ep.delta_stats([0.01, 0.03, -0.02, 0.0, 0.02, 0.05], n_boot=3000)
    assert abs(s["median"] - 0.015) < 1e-12
    assert s["lo98"] <= s["lo"] <= s["mean"] <= s["hi"] <= s["hi98"]
    assert ("B3", "B1") in ep.PAIRS


def test_dfl_reference_uses_the_last_touch_metric():
    assert ep.DFL_KEY == "spell_f1_ut"


def test_shift_edges_keeps_order_period_breaks_and_the_shift_bound():
    rng = np.random.default_rng(0)
    n = 600
    per = np.where(np.arange(n) < 300, 1, 2)
    lab = np.array(list(("H" * 7 + "A" * 40 + "D" * 3 + "H" * 90 + "A" * 2 + "H" * 158)) + list("A" * 150 + "H" * 150), dtype="<U1")
    out, n_edges = rg.shift_edges(lab, per, rng, 4)
    assert len(out) == n and n_edges == 5 + 1
    assert out[299] == lab[299] and out[300] == lab[300]               # the period change is not an edge
    for seg in (slice(0, 300), slice(300, 600)):
        e0 = np.flatnonzero(lab[seg][1:] != lab[seg][:-1])
        e1 = np.flatnonzero(out[seg][1:] != out[seg][:-1])
        assert len(e0) == len(e1) and (np.abs(e1 - e0) <= 4).all()     # no run disappears, bounded shift
        assert (np.diff(e1) > 0).all()
    again, _ = rg.shift_edges(lab, per, np.random.default_rng(0), 4)
    assert np.array_equal(out, again)
    assert not np.array_equal(out, lab)


def test_extra_touch_counts_on_a_synthetic_match():
    n = 200
    gt = np.array(["H"] * 100 + ["A"] * 100, dtype="<U1")
    b1 = np.array(["H"] * n, dtype="<U1")
    b2 = b1.copy()
    b2[30:] = "A"                                                      # a touch at 30 flips H -> A inside the GT H possession
    tf = np.array([30, 98])
    event_f = np.array([99])                                           # matches the touch at 98 (|98 - 99| <= TOL)
    r = ds.extras(tf, event_f, b1, b2, gt)
    assert (r["n_det"], r["n_ev"], r["emparejados"], r["sobrantes"]) == (2, 1, 1, 1)
    assert (r["cambia"], r["dentro"], r["parte"]) == (1, 1, 1)
    assert ds.inside_gt_possession(gt, 30) and not ds.inside_gt_possession(gt, 100)       # a GT edge within TOL
    assert ev.TOL == 12
