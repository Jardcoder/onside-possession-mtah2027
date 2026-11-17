"""bancos/reinicios.py: label_rw, window margin, restart-opened possessions (synthetic arrays) and the per-match alignment with rp.load_match (real data, cheap)."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("bancos", "src"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import modelo_transfer as mt  # noqa: E402
import pff_split  # noqa: E402
import reinicios as rt  # noqa: E402

H, A, D, X = 0, 1, 2, 3


def seq(*parts):
    return np.concatenate([np.full(n, c, np.int8) for c, n in parts])


def test_dead_where_restart_keeps_x():
    gt = seq((X, 1), (H, 20), (D, 10), (A, 20))
    rw = np.zeros(len(gt), np.int8)
    rw[0:5] = 4                                               # over the X frame and four H frames
    rw[35:40] = 1                                             # inside the A possession
    out = rt.dead_where_restart(gt, rw)
    assert out[0] == X and (out[1:5] == D).all() and (out[5:21] == H).all() and (out[35:40] == D).all() and (out[40:] == A).all()
    assert (gt == seq((X, 1), (H, 20), (D, 10), (A, 20))).all()   # the input is not modified


def test_near_window_margin_is_inclusive():
    rw = np.zeros(100, np.int8)
    rw[50:53] = 5
    near = rt.near_window(rw, [10, 24, 25, 26, 52, 77, 78, 99], margin=25)
    assert near.tolist() == [False, False, True, True, True, True, False, False]
    assert rt.near_window(rw, [0, 99], 25).tolist() == [False, False]
    assert rt.near_window(np.zeros(10, np.int8), [3], 25).tolist() == [False]


def test_restart_started_in_window_and_after_d():
    # H possession 0..40, D 40..60 (stop, restart event at 45), A possession 60..100 (opens after the D stretch),
    # H possession 100..140 opens inside a window (event at 100), A possession 140..180 is a steal (no restart)
    gt = seq((H, 40), (D, 20), (A, 40), (H, 40), (A, 40))
    rw = np.zeros(len(gt), np.int8)
    rw[100:104] = 4
    ev_frame, ev_kind = [45, 100], [5, 4]
    out = rt.restart_started(gt, rw, ev_frame, ev_kind)
    assert [(a, kind, how) for a, _, _, kind, how in out] == [(0, 0, ""), (60, 5, "after"), (100, 4, "in"), (140, 0, "")]


def test_restart_started_takes_the_last_event_of_the_d_stretch_and_stops_at_the_run_border():
    gt = seq((H, 10), (D, 30), (A, 20))
    out = rt.restart_started(gt, np.zeros(len(gt), np.int8), [12, 30, 45], [2, 7, 1])    # the event at 45 is after the possession starts: not in the stretch
    assert [(kind, how) for *_, kind, how in out] == [(0, ""), (7, "after")]
    gt = seq((X, 5), (D, 10), (X, 5), (A, 10), (X, 1))                                    # a D before the run border does not open anything across it
    out = rt.restart_started(gt, np.zeros(len(gt), np.int8), [6], [3])
    assert [(kind, how) for *_, kind, how in out] == [(0, "")]


@pytest.mark.needs_pff
def test_alignment_with_rp_load_match_on_two_matches():
    for mid in pff_split.DESARROLLO[:2]:
        r = rt.load_restart(mid)                              # asserts length, label of both routes and the feature cache inside
        assert len(r["rw"]) == len(mt.meta_of(mid, "g6h")["y"])
        assert r["rw"].dtype == np.int8 and set(np.unique(r["rw"])) <= set(range(8))
        assert (r["ev_frame"] < len(r["rw"])).all()
        assert (r["rw"][r["ev_frame"]] > 0).all()             # every event frame is inside a window
