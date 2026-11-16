"""Restart windows of the PFF loader (bancos/pff.py 3.1-restart-window) aligned to the rows of the match dict of ronda_pff.

    from reinicios import load_restart, dead_where_restart, restart_started, near_window

`rp.load_match` (ronda_pff, not touched: its source is in the cache fingerprints) clips the match to periods 1..2 through `ep.clip_periods`
and does not carry `restart_window`. `load_restart` reads the same match through `pff_split.load_dev`, applies the same clip and returns the
window array with the rows of the cached features, probabilities and GT codes (25 fps, one entry per row). The alignment is asserted per match:
same length and the same `label` as `rp.load_match`, and the same codes as the `y` of the feature cache.

DIAGNOSTIC / EVALUATION SIDE ONLY: the windows come from the PFF events, so they are never a feature of a method. Only the training weights of
bancos/ronda5_t1.py (variants T1m, T1wm) and the diagnostics read them.

Pure helpers (tested in tests/test_reinicios.py):
  dead_where_restart   `label_rw`: the GT with D on every frame inside a restart window
  near_window          per frame: is there a window frame within +-margin
  restart_started      per GT possession: the restart (kind) that opens it, if any, and how it was found
"""
import functools
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import onside_paths  # noqa: E402
onside_paths.add_challenge_to_path()
import evaluar_pff as ep  # noqa: E402
import modelo_pff as mp_  # noqa: E402
import modelo_transfer as mt  # noqa: E402
import pff_split  # noqa: E402
import possession_metrics as pm  # noqa: E402
import ronda_pff as rp  # noqa: E402

# restart_window code -> name (pff.py RESTART_CODES; the order of the report)
KINDS = {1: "falta", 2: "saque de puerta", 3: "saque inicial", 4: "lateral", 5: "corner", 6: "penal", 7: "drop ball"}
MARGIN = 25                                     # frames of margin around a window for the transition recall (diagnostic 10)


# ------------------------------------------------------------------------------------------------ pure functions (tested)

def dead_where_restart(gt, rw):
    """`label_rw`: GT codes with D on every frame whose restart_window > 0 (X, the frame that closes a period, stays X)."""
    out = np.asarray(gt).copy()
    out[(np.asarray(rw) > 0) & (out != pm.X)] = pm.D
    return out


def near_window(rw, frames, margin=MARGIN):
    """bool per entry of `frames`: some frame with restart_window > 0 within +-margin (inclusive)."""
    inside = np.concatenate(([0], np.cumsum(np.asarray(rw) > 0)))
    frames = np.asarray(frames, int)
    lo = np.clip(frames - margin, 0, len(rw))
    hi = np.clip(frames + margin + 1, 0, len(rw))
    return (inside[hi] - inside[lo]) > 0


def restart_started(gt, rw, ev_frame, ev_kind):
    """[(start, stop, team, kind, how)] one per GT possession, in the order of pm.possession_spells(gt, valid_runs(gt)).
    A possession is opened by a restart when
      how = "in":    its first frame is inside a restart window (kind = the window code), or
      how = "after": its first frame follows a D stretch (maximal, inside the valid run) that contains the event frame of a restart
                     (kind = the last such restart); PFF dates the event before the ball rolls, so the label usually leaves D after the window ends.
    Otherwise kind = 0, how = ""."""
    gt, rw = np.asarray(gt), np.asarray(rw)
    ev_frame, ev_kind = np.asarray(ev_frame, int), np.asarray(ev_kind, int)
    runs = pm.valid_runs(gt)
    out = []
    for a, b, k in pm.possession_spells(gt, runs):
        run_start = next(s for s, e in runs if s <= a < e)
        kind, how = 0, ""
        if rw[a] > 0:
            kind, how = int(rw[a]), "in"
        elif a > run_start and gt[a - 1] == pm.D:
            s = a - 1
            while s > run_start and gt[s - 1] == pm.D:
                s -= 1
            hit = np.flatnonzero((ev_frame >= s) & (ev_frame < a))
            if len(hit):
                kind, how = int(ev_kind[hit[-1]]), "after"
        out.append((a, b, k, kind, how))
    return out


# ------------------------------------------------------------------------------------------------ loader

@functools.lru_cache(maxsize=None)
def load_restart(mid):
    """{rw (N,) int8, ev_frame, ev_kind, next_touch (one entry per restart of periods 1-2)} aligned to the rows of rp.load_match(mid). Memoised per process: do not mutate."""
    assert mid in pff_split.DESARROLLO and mid not in pff_split.RESERVADOS
    raw = pff_split.load_dev(mid)
    keep = np.asarray(raw["period"]) <= ep.LAST_PERIOD
    n = int(keep.sum())
    assert keep[:n].all() and not keep[n:].any(), "periods 1..2 must be a prefix of the rows (the event frames are row indices)"
    rw = np.asarray(raw["restart_window"])[keep]
    lab = np.asarray(raw["label"])[keep]
    t = raw["restart_windows"]
    sel = np.asarray(t["event_frame"]) < n
    d = rp.load_match(mid)
    assert len(rw) == len(d["label"]) == len(d["period"]), f"{mid}: restart_window length {len(rw)} != rows of rp.load_match {len(d['label'])}"
    assert (lab == np.asarray(d["label"])).all(), f"{mid}: label of the two routes differs"
    meta = mt.meta_of(mid, "g6h")
    assert len(rw) == len(meta["y"]) and (mp_.label_codes(lab) == meta["y"]).all(), f"{mid}: rows do not match the feature cache"
    return {"rw": rw.astype(np.int8), "ev_frame": np.asarray(t["event_frame"])[sel].astype(np.int64),
            "ev_kind": np.asarray(t["kind"])[sel].astype(np.int8), "next_touch": np.asarray(t["next_touch_frame"])[sel].astype(np.int64)}


_HASH = {}


def restart_hash():
    """SHA-256 of the restart_window arrays of the 40 development matches (what the ronda 5 weights and the label_rw GT read); memoised per process."""
    if "h" not in _HASH:
        h = hashlib.sha256()
        for mid in pff_split.DESARROLLO:
            r = load_restart(mid)
            h.update(mid.encode() + b"\0" + r["rw"].tobytes() + r["ev_frame"].tobytes() + r["ev_kind"].tobytes())
        _HASH["h"] = h.hexdigest()
    return _HASH["h"]
