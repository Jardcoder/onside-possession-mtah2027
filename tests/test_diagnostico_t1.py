"""bancos/diagnostico_t1.py: pure functions on small synthetic label arrays (no match data)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("bancos", "src"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import diagnostico_t1 as dg  # noqa: E402

H, A, D, X = 0, 1, 2, 3


def seq(*parts):
    """Codes from (code, length) pairs."""
    return np.concatenate([np.full(n, c, np.int8) for c, n in parts])


def test_edge_shifts_split_by_what_surrounds_the_gt_possession():
    gt = seq((D, 10), (H, 100), (A, 100), (D, 10))
    pred = seq((D, 14), (H, 100), (A, 96), (D, 10))            # H opens 4 late and closes 4 late; A opens 4 late and closes on time
    by = {(c_s, c_e): (ds, de) for ds, de, c_s, c_e in dg.edge_shifts(gt, pred)}
    assert by == {("D", "otro"): (4, 4), ("otro", "D"): (4, 0)}


def test_edge_shifts_mark_the_run_border_and_skip_unmatched_possessions():
    gt = seq((X, 5), (H, 100), (D, 10), (A, 100), (X, 5))      # the H opens the valid run and the A closes it: no context there
    pred = seq((H, 105), (D, 10), (A, 5), (H, 100))            # predictions are cut at the run border, so the H start is on time; the predicted A (5 frames) has IoU < 0.5 with the GT A: no match
    e = dg.edge_shifts(gt, pred)
    assert [(ds, de, c_s, c_e) for ds, de, c_s, c_e in e] == [(0, 0, None, "D")]


def test_direct_transitions_skip_the_changes_that_go_through_d():
    codes = seq((H, 50), (A, 30), (D, 5), (H, 20), (A, 10), (H, 5))
    assert dg.direct_transitions(codes, [(0, len(codes))]) == [(50, H, A, 30), (105, H, A, 10), (115, A, H, 5)]


def test_transition_recall_by_tolerance_and_direction():
    gt = seq((H, 100), (A, 100), (H, 100))                     # H -> A at 100, A -> H at 200
    pred = seq((H, 110), (A, 130), (H, 60))                    # H -> A at 110 (10 away), A -> H at 240 (40 away)
    dur, hit = dg.transition_recall(gt, pred, tols=(25, 50))
    assert dur.tolist() == [100, 100]
    assert hit.tolist() == [[True, True], [False, True]]
    dur, hit = dg.transition_recall(seq((H, 100), (A, 100)), seq((A, 100), (H, 100)))   # same frame, opposite direction: not a hit
    assert dur.tolist() == [100] and not hit.any()


def test_spurious_transitions_are_the_model_changes_far_from_a_gt_one_of_the_same_direction():
    gt = seq((H, 100), (A, 100))
    pred = seq((H, 100), (A, 40), (H, 30), (A, 30))            # 100 matches; 140 (A -> H) has no GT one; 170 (H -> A) is 70 from the GT one
    assert dg.spurious_transitions(gt, pred, tol=50).tolist() == [30, 30]


def test_ceiling_dead_forces_gt_d_and_fills_the_live_frames_the_model_called_d():
    gt = np.array([H, H, D, A, A, A], np.int8)
    pred = np.array([H, D, D, D, D, A], np.int8)
    P = np.array([[.8, .1, .1], [.3, .4, .3], [.2, .2, .6], [.5, .1, .4], [.1, .6, .3], [.1, .8, .1]])
    assert dg.ceiling_dead(gt, pred, P).tolist() == [H, A, D, H, A, A]                       # per-frame argmax of H / A
    assert dg.ceiling_dead(gt, pred, P, per_stretch=True).tolist() == [H, A, D, A, A, A]     # frames 3-4: summed H .6 < A .7
    assert dg.ceiling_dead(np.array([X, H], np.int8), np.array([D, D], np.int8), P[:2]).tolist() == [D, A]   # a GT X frame is never filled


def test_ceiling_team_takes_the_gt_team_only_where_both_are_live():
    gt = np.array([H, H, D, A, A, X], np.int8)
    pred = np.array([A, D, H, D, H, A], np.int8)
    assert dg.ceiling_team(gt, pred).tolist() == [H, D, H, D, A, A]


def test_duration_bins_are_half_open():
    d = [10, 50, 124, 125, 749, 750, 3000]
    assert dg.duration_bins(d, dg.BINS_D).tolist() == [1, 2, 1, 1, 2]


def test_shift_stats():
    s = dg.shift_stats([-20, 0, 5, 13, 30])
    assert (s["n"], s["median"], s["big"]) == (5, 5.0, 60.0) and abs(s["mean"] - 5.6) < 1e-9
    assert dg.shift_stats([])["n"] == 0 and np.isnan(dg.shift_stats([])["median"])


def test_barrido_row_reads_the_lambda_point_of_the_informe(tmp_path):
    f = tmp_path / "informe.txt"
    f.write_text("title\n        ha / d         8        16\n"
                 "             4    0.5203 (0.5232/0.5203)    0.5283 (0.5311/0.5283)\n"
                 "             8    0.5314 (0.5356/0.5314)    0.5401 (0.5444/0.5401)\n")
    assert dg.barrido_row((8, 16), str(f)) == (0.5444, 0.5401)
    assert dg.barrido_row((8, 64), str(f)) is None and dg.barrido_row((8, 16), str(tmp_path / "missing.txt")) is None


# ------------------------------------------------------------------------------------------------ extension 8 to 11

def random_codes(rng, n, p_d=0.2):
    """A label array of stretches of H / A / D of random length (no X)."""
    out, c = [], 0
    while sum(len(x) for x in out) < n:
        out.append(np.full(int(rng.integers(3, 60)), rng.choice([H, A, D], p=[(1 - p_d) / 2, (1 - p_d) / 2, p_d]), np.int8))
    return np.concatenate(out)[:n]


def test_classify_gt_spells_counts_equal_classify_spells():
    import oraculo_pff as op
    rng = np.random.default_rng(7)
    for _ in range(40):
        gt = random_codes(rng, 1500)
        pred = np.where(rng.random(1500) < 0.7, gt, random_codes(rng, 1500))     # a noisy copy: every category shows up over the draws
        got = dg.classify_gt_spells(gt, pred)
        fn = op.classify_spells(gt, pred)[0]
        assert {k: sum(c == k for *_, c in got) for k in op.CATS_FN} == fn
        assert sum(c is None for *_, c in got) + sum(fn.values()) == len(got)


def test_classify_gt_spells_known_cases():
    gt = seq((H, 100), (A, 100), (H, 100))
    assert [c for *_, c in dg.classify_gt_spells(gt, gt)] == [None, None, None]
    pred = seq((H, 100), (H, 100), (H, 100))                                      # the A possession is absorbed by the H around it
    cats = [c for *_, c in dg.classify_gt_spells(gt, pred)]
    assert cats[1] == "equipo_equivocado" or cats[1] == "fusion"
    pred = seq((H, 100), (D, 100), (H, 100))
    assert [c for *_, c in dg.classify_gt_spells(gt, pred)][1] == "borde_D"


def test_restart_start_records_shift_of_matched_and_nan_of_unmatched():
    gt = seq((H, 50), (D, 30), (A, 100), (D, 20), (H, 100))
    rw = np.zeros(len(gt), np.int8)
    rw[80:84] = 4
    pred = seq((H, 50), (D, 36), (A, 94), (D, 20), (A, 100))                      # the A possession opens 6 late; the last H is predicted A: no match
    rec = dg.restart_start_records(gt, pred, rw, [80], [4])
    assert [(k, h) for k, h, _, _ in rec] == [(0, ""), (4, "in"), (0, "")]
    assert rec[1][2] == 6 and np.isnan(rec[2][2]) and rec[0][2] == 0
    assert rec[1][3] == 0                                                         # the possession opens on the event frame


def test_recall_by_window_marks_the_changes_near_a_window():
    gt = seq((H, 100), (A, 100), (H, 100), (A, 100))                              # changes at 100, 200, 300
    rw = np.zeros(len(gt), np.int8)
    rw[310:320] = 1
    pred = gt.copy()
    hit, near = dg.recall_by_window(gt, pred, rw)
    assert hit.all() and near.tolist() == [False, False, True]
    pred = seq((H, 100), (A, 100), (H, 140), (A, 60))                             # the change at 300 is 40 frames late: hit at +-50 only
    hit, near = dg.recall_by_window(gt, pred, rw)
    assert hit.tolist() == [[True, True], [True, True], [False, True]]
