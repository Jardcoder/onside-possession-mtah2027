"""Contract of challenge/evaluation/possession_metrics.py, probed with SYNTHETIC sequences only.

No BAR-ATH data is read here. Every sequence is built in the test. The asserts document what the
evaluator DOES (measured), not what we think it should do. If the evaluator changes, these fail
and every method that depends on its scoring must be revisited.

Run: ./run.sh test
"""
import numpy as np
import pytest

import possession_metrics as pm  # noqa: E402

H, A, D, X = pm.H, pm.A, pm.D, pm.X
PAD = 100  # ungraded X frames either side, like pre-match / post-match
TOL = 2e-3


def seq(*parts):
    """seq((H, 350), (D, 100), ...) -> int8 array of codes."""
    return np.concatenate([np.full(n, c, dtype=np.int8) for c, n in parts])


def padded(s):
    return np.concatenate([np.full(PAD, X, np.int8), s, np.full(PAD, X, np.int8)])


def score(gt, pred):
    """(spellF1, macroF1, n_gt_spells, n_pred_spells) with the real evaluate()."""
    r = pm.evaluate(padded(gt), padded(pred))
    return r["spell_f1_mean"], r["macro_f1"], r["n_gt_spells"], r["n_pred_spells"]


def shifted(base, k):
    """Every boundary moved k frames later (k>0) or earlier (k<0); edges re-filled."""
    out = np.roll(base, k)
    if k > 0:
        out[:k] = base[0]
    else:
        out[k:] = base[-1]
    return out


# Reference match: 5 possessions of 14 s, two stoppages.
BASE = seq((H, 350), (D, 100), (A, 350), (H, 350), (D, 100), (A, 350), (H, 350))


def test_perfect_prediction():
    s, m, ng, npred = score(BASE, BASE)
    assert s == pytest.approx(1.0) and m == pytest.approx(1.0)
    assert ng == npred == 5


def test_constant_prediction_scores_zero_on_spells():
    s, m, _, npred = score(BASE, np.full(len(BASE), H, np.int8))
    assert s == 0.0 and npred == 1
    assert m < 0.3


@pytest.mark.parametrize("m", [1, 5, 25])
def test_micro_possession_is_a_coin_flip(m):
    """A-B-A with a micro B of m frames. Cost is the SAME for 1, 5 or 25 frames and is paid in
    both directions: spellF1 counts every spell once, however short."""
    with_micro = seq((H, 350), (D, 100), (A, 350), (H, 350), (A, m), (H, 350), (D, 100), (A, 350))
    fused = seq((H, 350), (D, 100), (A, 350), (H, 350 + m + 350), (D, 100), (A, 350))

    s, _, ng, npred = score(with_micro, with_micro)  # GT has it, predicted
    assert s == pytest.approx(1.0) and ng == npred == 6

    s, mf, ng, npred = score(with_micro, fused)  # GT has it, we fuse (miss it)
    assert (ng, npred) == (6, 4)
    assert s == pytest.approx(0.68, abs=TOL)
    assert mf > 0.98  # frame metric barely notices

    s, mf, ng, npred = score(fused, with_micro)  # GT has none, we invent it (false positive)
    assert (ng, npred) == (4, 6)
    assert s == pytest.approx(0.68, abs=TOL)
    assert mf > 0.98


@pytest.mark.parametrize("d", [1, 5, 25, 100])
def test_d_cut_inside_possession(d):
    """H-D-H counts as two possessions (EVALUATION.md); H continuous as one."""
    continuous = seq((A, 350), (H, 600), (A, 350))
    cut = seq((A, 350), (H, 300), (D, d), (H, 300), (A, 350))
    cut_pred = seq((A, 350), (H, 300), (D, d), (H, 300 - d), (A, 350))

    assert score(cut, cut)[0] == pytest.approx(1.0)  # perfect cut
    # GT continuous, we cut: one possession becomes two, even for a 1-frame D.
    s, _, ng, npred = score(continuous, cut_pred)
    assert (ng, npred) == (3, 4) and s == pytest.approx(0.743, abs=TOL)
    # GT cut, we stay continuous: the symmetric miss.
    s, _, ng, npred = score(cut, seq((A, 350), (H, 600 + d), (A, 350)))
    assert (ng, npred) == (4, 3) and s == pytest.approx(0.686, abs=TOL)


def test_x_in_gt_is_ignored_and_splits_the_run():
    """GT X is never scored, whatever is predicted there, and a GT X hole splits the temporal
    runs: H-X-H is two possession spells for BOTH gt and prediction (no phantom merge)."""
    gt = seq((A, 350), (H, 300), (X, 50), (H, 300), (A, 350))
    results = []
    for lab in (H, A, D, X):
        pred = seq((A, 350), (H, 300), (lab, 50), (H, 300), (A, 350))
        results.append(score(gt, pred))
    assert all(r == results[0] for r in results)  # prediction on GT X changes nothing
    s, _, ng, npred = results[0]
    assert s == pytest.approx(1.0) and ng == npred == 4  # 4 spells, not 3
    # Same labels with no X hole: 3 spells.
    assert score(seq((A, 350), (H, 650), (A, 350)), seq((A, 350), (H, 650), (A, 350)))[2] == 3


def test_x_in_gt_does_not_enter_frame_metrics():
    r = pm.evaluate(padded(BASE), padded(BASE))
    assert r["n_graded"] == len(BASE)


def test_abstention_inside_possession_is_free_at_handover_is_not():
    """Pred X is forward-filled with the previous label for spellF1."""
    gt = seq((H, 350), (A, 350), (H, 350))
    inside = seq((H, 150), (X, 100), (H, 100), (A, 350), (H, 350))
    s, m, _, _ = score(gt, inside)
    assert s == pytest.approx(1.0)  # free on the ranking metric
    assert m < 1.0  # but charged on macroF1
    # Going quiet over the H->A hand-over: the old H is carried across it.
    handover = seq((H, 350), (X, 200), (A, 150), (H, 350))
    s, _, _, _ = score(gt, handover)
    assert s == pytest.approx(0.733, abs=TOL)


@pytest.mark.parametrize("k,expected_spell,min_macro", [
    (1, 1.0, 0.99),
    (5, 1.0, 0.97),
    (25, 1.0, 0.87),
])
def test_boundary_shift(k, expected_spell, min_macro):
    """All boundaries moved by +-k frames. Spell IoU on 14 s possessions is almost insensitive:
    even 25 frames (1 s) late keeps every possession matched at every IoU threshold here."""
    for sign in (1, -1):
        s, m, _, _ = score(BASE, shifted(BASE, sign * k))
        assert s == pytest.approx(expected_spell, abs=TOL)
        assert min_macro < m < 1.0


def test_boundary_shift_hurts_short_possessions():
    """The same 1 s shift on SHORT possessions (3 s) does cost spellF1, so tolerance scales with
    possession length."""
    short = seq((H, 75), (A, 75), (H, 75), (A, 75), (H, 75), (A, 75))
    s0, _, _, _ = score(short, short)
    s25, _, _, _ = score(short, shifted(short, 25))
    s5, _, _, _ = score(short, shifted(short, 5))
    assert s0 == pytest.approx(1.0)
    assert s5 > 0.95
    assert s25 < 0.9


@pytest.mark.parametrize("d", [25, 100, 250])
def test_dragging_label_through_gt_d(d):
    """GT has H-D-A; the prediction keeps H (or starts A early) through the D. Spells only lose
    once the drag is long enough to push IoU below the thresholds."""
    gt = seq((H, 350), (D, d), (A, 350))
    for pred in (seq((H, 350 + d), (A, 350)), seq((H, 350), (A, d + 350))):
        s, m, ng, npred = score(gt, pred)
        assert ng == npred == 2
        if d <= 100:
            assert s == pytest.approx(1.0, abs=TOL)
        else:
            assert s == pytest.approx(0.8, abs=TOL)
        assert m < 1.0


def test_dragging_label_through_d_between_same_team_merges_possessions():
    """GT H-D-H (two possessions), prediction drags H through the D (one). Loses regardless of
    the D length: the dead-ball stoppage is what separates the possessions."""
    gt = seq((A, 350), (H, 350), (D, 100), (H, 350), (A, 350))
    pred = seq((A, 350), (H, 800), (A, 350))
    s, _, ng, npred = score(gt, pred)
    assert (ng, npred) == (4, 3)
    assert s == pytest.approx(0.686, abs=TOL)
