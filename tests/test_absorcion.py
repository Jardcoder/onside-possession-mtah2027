"""R2 (src/absorcion.py): null configuration, contiguity, period cut, input untouched; cache fingerprint of bancos/ronda_pff.py."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("bancos", "src"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import absorcion as ab  # noqa: E402
import diagnostico_pff_sobrantes as dps  # noqa: E402
import ronda_pff as rp  # noqa: E402


def seq(*parts):
    return np.array(list("".join(k * n for k, n in parts)), dtype=object)


def test_tau_zero_is_the_identity():
    rng = np.random.default_rng(0)
    x = np.array(rng.choice(["H", "A", "D"], 5000, p=[0.45, 0.45, 0.1]), dtype=object)
    for per in (None, np.repeat([1, 2], 2500)):
        assert (ab.absorb(x, 0.0, per) == x).all()


def test_short_stretch_between_rival_stretches_is_absorbed():
    x = seq(("A", 50), ("H", 4), ("A", 50))                    # 4 frames = 0.16 s
    assert (ab.absorb(x, 0.2) == seq(("A", 104))).all()
    assert (ab.absorb(x, 0.16) == x).all()                     # strict: duration < tau


def test_not_absorbed_when_a_dead_stretch_or_a_same_team_neighbour_is_adjacent():
    assert (ab.absorb(seq(("A", 50), ("D", 1), ("H", 4), ("A", 50)), 0.8) == seq(("A", 50), ("D", 1), ("H", 4), ("A", 50))).all()
    assert (ab.absorb(seq(("A", 50), ("H", 4), ("D", 1), ("A", 50)), 0.8) == seq(("A", 50), ("H", 4), ("D", 1), ("A", 50))).all()
    x = seq(("H", 50), ("A", 4), ("H", 50), ("D", 3))
    assert (ab.absorb(x, 0.8) == seq(("H", 104), ("D", 3))).all()
    assert (ab.absorb(seq(("A", 50), ("H", 4)), 0.8) == seq(("A", 50), ("H", 4))).all()     # no following stretch: edge


def test_a_dead_stretch_is_never_absorbed_and_the_period_cuts_stretches():
    x = seq(("A", 50), ("D", 2), ("A", 50))
    assert (ab.absorb(x, 0.8) == x).all()
    y = seq(("A", 50), ("H", 4), ("A", 50))
    per = np.array([1] * 50 + [2] * 54)                        # the period changes where the micro-stretch starts: not contiguous
    assert (ab.absorb(y, 0.8, per) == y).all()
    per = np.array([1] * 104)
    assert (ab.absorb(y, 0.8, per) == seq(("A", 104))).all()


def test_input_is_not_modified_and_single_pass_on_the_original_labels():
    x = seq(("H", 50), ("A", 3), ("H", 3), ("A", 50))
    before = x.copy()
    ab.absorb(x, 0.8)
    assert (x == before).all()
    # H50 a3 h3 A50: both micro-stretches are judged on the original labels (each has the other as one rival neighbour)
    assert (ab.absorb(x, 0.8) == seq(("H", 53), ("A", 53))).all()


# ------------------------------------------------------------------------------------------------ cache fingerprint

def test_fingerprint_changes_with_the_parameters_and_the_smoothing(monkeypatch):
    a = rp.fingerprint("R2", "tau0.4", (0.4,))
    assert a == rp.fingerprint("R2", "tau0.4", (0.4,))
    assert a != rp.fingerprint("R2", "tau0.6", (0.6,))
    assert a != rp.fingerprint("C1", "tau0.4", (0.4,))
    monkeypatch.setattr(rp.dd, "BALL_SMOOTH", dict(rp.dd.BALL_SMOOTH, g6=0.123))
    assert a != rp.fingerprint("R2", "tau0.4", (0.4,))


def test_fingerprint_changes_with_the_source_of_a_module(monkeypatch):
    base = rp.fingerprint("B3", "B3", None)
    real = rp.source_of
    monkeypatch.setattr(rp, "source_of", lambda mod: b"edited" if mod is rp.toques else real(mod))
    assert rp.fingerprint("B3", "B3", None) != base


def good_rows(mid="3826", cand="B3", tag="B3"):
    fp = rp.fingerprint(cand, tag, None)
    rows = []
    for balon in rp.BALLS:
        for g in rp.GTS:
            r = {k: "" for k in rp.FIELDS}
            r.update(partido=mid, balon=balon, gt=g, cand=cand, tag=tag, spell_f1="0.3", n_gt="10", n_pred="12", huella=fp)
            if balon == "g6" and g == "label":
                r.update({k: "1" for k in rp.COUNTERS})
            rows.append(r)
    return rows


def write(path, rows):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, rp.FIELDS)
        w.writeheader()
        w.writerows(rows)


def test_cache_file_is_accepted_only_when_identity_completeness_and_fingerprint_match(tmp_path, monkeypatch):
    monkeypatch.setattr(rp, "CACHE", str(tmp_path))
    fp = rp.fingerprint("B3", "B3", None)
    path = rp.cache_file("3826", "B3", "B3")
    write(path, good_rows())
    assert rp.cache_ok("3826", "B3", "B3", None)
    write(path, good_rows()[:-1])                                           # a row missing
    assert not rp.cache_ok("3826", "B3", "B3", None)
    bad = good_rows(); bad[2]["huella"] = "0" * 64                          # stale code
    write(path, bad)
    assert not rp.cache_ok("3826", "B3", "B3", None)
    bad = good_rows(); bad[0]["partido"] = "3827"                           # another match's file
    write(path, bad)
    assert not rp.cache_ok("3826", "B3", "B3", None)
    bad = good_rows(); bad[0]["spell_f1"] = "nan"
    write(path, bad)
    assert not rp.cache_ok("3826", "B3", "B3", None)
    bad = good_rows(); bad[1]["gt"] = bad[0]["gt"]; bad[1]["balon"] = bad[0]["balon"]   # duplicated key
    write(path, bad)
    assert not rp.cache_ok("3826", "B3", "B3", None)
    bad = good_rows()
    for r in bad:
        r["sobrantes"] = ""                                                  # counters missing on the counter row
    write(path, bad)
    assert not rp.cache_ok("3826", "B3", "B3", None)
    assert not rp.cache_ok("3827", "B3", "B3", None)                        # file absent
    assert fp


# ------------------------------------------------------------------------------------------------ corrected counter

def test_split_possessions_counts_unique_gt_possessions_not_touches():
    gt = seq(("H", 100), ("D", 5), ("A", 100))
    b1 = seq(("H", 205))
    b2 = np.array(["H"] * 30 + ["A"] * 70 + ["H"] * 40 + ["D"] * 20 + ["H"] * 45, dtype=object)
    tf = np.array([40, 60, 80, 150])                                        # 3 touches inside the first GT possession, 1 in the second
    ex = dps.extras(tf, np.array([], int), b1, b2, gt)
    assert ex["cambia"] == 4 and ex["dentro"] == 4
    assert ex["parte_toques"] == 4 and ex["parte"] == 2


# ------------------------------------------------------------------------------------------------ R4: conditional absorption

def cond(x, tau, cf, ct, mode, per=None):
    return ab.absorb_conditional(x, tau, np.array(cf, int), np.array(ct, dtype=object), mode, per)


def test_conditional_tau_zero_changes_nothing_in_either_mode():
    rng = np.random.default_rng(1)
    x = np.array(rng.choice(["H", "A", "D"], 4000, p=[0.45, 0.45, 0.1]), dtype=object)
    cf = rng.integers(0, 4000, 300)
    ct = np.array(rng.choice(["H", "A"], 300), dtype=object)
    for mode in ("any", "own"):
        assert (ab.absorb_conditional(x, 0.0, cf, ct, mode) == x).all()
        assert (ab.absorb_conditional(x, 0.0, cf, ct, mode, np.repeat([1, 2], 2000)) == x).all()


def test_conditional_stretch_with_a_control_touch_inside_is_not_absorbed_in_any():
    x = seq(("A", 50), ("H", 20), ("A", 50))                    # 20 frames = 0.8 s < 1.6 s
    assert (cond(x, 1.6, [], [], "any") == seq(("A", 120))).all()                     # no touch: absorbed
    for team in ("H", "A"):                                                          # the touch of either team protects in `any`
        assert (cond(x, 1.6, [60], [team], "any") == x).all()


def test_conditional_rival_control_touch_does_not_protect_in_own_but_own_one_does():
    x = seq(("A", 50), ("H", 20), ("A", 50))
    assert (cond(x, 1.6, [60], ["A"], "own") == seq(("A", 120))).all()                # a rival touch inside: absorbed
    assert (cond(x, 1.6, [60], ["H"], "own") == x).all()                              # a touch of the stretch's team: protected


def test_conditional_touch_window_is_start_inclusive_stop_exclusive_and_order_free():
    x = seq(("A", 50), ("H", 20), ("A", 50))
    assert (cond(x, 1.6, [50], ["H"], "any") == x).all()                              # first frame of the stretch
    assert (cond(x, 1.6, [70], ["H"], "any") == seq(("A", 120))).all()                # first frame after it
    assert (cond(x, 1.6, [49], ["H"], "any") == seq(("A", 120))).all()                # last frame before it
    assert (cond(x, 1.6, [200, 60, 5], ["A", "H", "A"], "own") == x).all()            # unsorted touches
    assert (cond(x, 1.6, [200, 5], ["H", "H"], "any") == seq(("A", 120))).all()


def test_conditional_without_control_touches_equals_absorb_and_never_absorbs_more():
    rng = np.random.default_rng(2)
    x = np.array(rng.choice(["H", "A", "D"], 6000, p=[0.45, 0.45, 0.1]), dtype=object)
    per = np.repeat([1, 2], 3000)
    for tau in (0.8, 1.6, 3.0):
        for mode in ("any", "own"):
            assert (ab.absorb_conditional(x, tau, [], [], mode, per) == ab.absorb(x, tau, per)).all()
    cf = rng.integers(0, 6000, 400)
    ct = np.array(rng.choice(["H", "A"], 400), dtype=object)
    full, anyc, ownc = ab.absorb(x, 1.6, per), cond(x, 1.6, cf, ct, "any", per), cond(x, 1.6, cf, ct, "own", per)
    assert (anyc != x).sum() <= (ownc != x).sum() <= (full != x).sum()                # any protects the most, absorb the least


def test_conditional_keeps_the_contiguity_rules_and_does_not_modify_the_input():
    x = seq(("A", 50), ("D", 1), ("H", 4), ("A", 50))
    before = x.copy()
    assert (cond(x, 1.6, [], [], "any") == x).all() and (x == before).all()
    y = seq(("A", 50), ("H", 4), ("A", 50))
    assert (cond(y, 0.8, [], [], "own", np.array([1] * 50 + [2] * 54)) == y).all()    # period cut
    with pytest.raises(ValueError):
        cond(y, 0.8, [], [], "both")
