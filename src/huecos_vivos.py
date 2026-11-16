"""Decide, hole by hole, whether a long ball-less gap is live play or dead ball (pure array functions, no I/O).

The base (toques.predict with `max_gap_s` = CAP_S) labels every hole longer than the cap as D. Here a hole longer than the
cap and at most `l_max_s` long can instead keep an owner when the players say the game was running. Nothing here reads a
label, a ball status or an event: ball, player slots, period and pitch only.

  v_rel      mean player speed inside the hole (1 s dropped at each border; `dfl_degradado.smoothed_player_speed`, 2 s
             window, metres) divided by the median of the same speed over the frames with a visible ball of the SAME match
             and period (`ref_mask` limits those frames, e.g. to the live halves of the challenge). Relative on purpose: the
             banks and the challenge differ in absolute speed.
  edge veto  (optional) the hole is dead when the ball disappears or reappears less than `edge_m` m from a line (a visible
             ball outside the pitch counts): throw-in, corner, goal kick. A side with no visible frame (period border) says
             nothing.
  alive      length <= l_max_s and v_rel >= theta and not vetoed. A NaN v_rel is dead.
  fill       an alive hole takes `pre` (owner at the last visible frame before it), `post` (owner at the first one after
             it) or `split` (first half `pre`, second half `post`); a missing owner (period border, or D there) leaves D.
             A dead hole stays D.

Inputs as in toques.py: ball_xy (N,2) normalised 0-1 with NaN, players_slots {"H": (N,24,2), "A": (N,22,2)} with NaN,
period (N,), pitch = (length_m, width_m). `labels` is the output of toques.predict with max_gap_s=CAP_S, where every hole
longer than the cap is already D.
"""
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "bancos"))
import dfl_degradado as dd  # noqa: E402
import muerto  # noqa: E402
import reglas  # noqa: E402

FPS = reglas.FPS
CAP_S = 5                  # holes up to this long are filled by the base; longer ones are the ones decided here
WIN_S = 2.0                # player speed window (s)
EDGE_S = 1.0               # seconds dropped at each border of the hole when averaging the speed
MODES = ("pre", "post", "split")

# Grid (135 combinations). THETAS were placed from the distribution of v_rel by real state in the main bank
# (evaluar_huecos.py --solo-distribucion), before any spellF1 was computed with them: dead holes p75 0.71-0.76 and p90 0.85-0.89,
# alive holes p10 0.75-0.76 and p25 0.94, so the frontier sits near 0.82 and the grid is centred a step above it.
THETAS = (0.65, 0.75, 0.85, 0.95, 1.05)
L_MAX_S = (10.0, 20.0, np.inf)
EDGES_M = (None, 1.0, 2.0)


def combo_tag(theta, l_max_s, edge_m, mode):
    e = "off" if edge_m is None else f"{edge_m:g}"
    return f"th{theta:g}_L{'inf' if np.isinf(l_max_s) else format(l_max_s, 'g')}_e{e}_{mode}"


def grid():
    """[(tag, (theta, l_max_s, edge_m, mode))] in a fixed order (ties in a selection go to the earlier one)."""
    return [(combo_tag(*c), c) for c in itertools.product(THETAS, L_MAX_S, EDGES_M, MODES)]


def player_speed(players_slots, period, pitch, win_s=WIN_S):
    """(N,) mean speed (m/s) of the present players, `win_s` moving average, computed on each period on its own so the
    smoothing never crosses the half-time break."""
    scale = np.asarray(pitch, float)
    slots_m = np.concatenate([np.asarray(players_slots["H"], float), np.asarray(players_slots["A"], float)], axis=1) * scale
    period = np.asarray(period)
    out = np.full(len(period), np.nan)
    cut = np.flatnonzero(period[1:] != period[:-1]) + 1
    edges = np.concatenate(([0], cut, [len(period)]))
    for lo, hi in zip(edges[:-1], edges[1:]):
        out[lo:hi] = dd.smoothed_player_speed(slots_m[lo:hi], win=int(round(win_s * FPS)))
    return out


def hole_table(ball_xy, players_slots, period, pitch, cap_s=CAP_S, edge_s=EDGE_S, win_s=WIN_S, ref_mask=None, speed=None):
    """Features of every hole (run of absent ball inside a period) longer than `cap_s`, as aligned arrays:
    start / end (end exclusive), length (frames), period, pre / post (frame of the last visible frame before / first after
    the hole in the same period, -1 when there is none), v_abs (mean speed in the interior, NaN when empty), v_ref (median
    speed over the visible-ball frames of the same period, restricted by `ref_mask`), v_rel = v_abs / v_ref, edge_pre /
    edge_post (metres from the ball to the nearest line at those frames, negative outside; NaN when there is none) and
    edge_min (their nanmin). `speed` (player_speed output) can be passed to reuse it."""
    ball = np.asarray(ball_xy, float)
    period = np.asarray(period)
    present = ~np.isnan(ball[:, 0])
    spd = speed if speed is not None else player_speed(players_slots, period, pitch, win_s)
    edge_dist = -muerto.ball_features(ball, period, pitch)[0]                 # + inside, - outside, NaN with no ball
    keep = np.ones(len(period), bool) if ref_mask is None else np.asarray(ref_mask, bool)
    v_ref = {}
    for p in np.unique(period):
        sel = present & keep & (period == p)
        v = spd[sel]
        v_ref[p] = float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan
    runs = [r for r in reglas.gap_runs(present, period) if (r["end"] - r["start"]) > cap_s * FPS]
    n, e = len(runs), int(round(edge_s * FPS))
    cols = {k: np.zeros(n, int) for k in ("start", "end", "length", "period", "pre", "post")}
    cols.update({k: np.full(n, np.nan) for k in ("v_abs", "v_ref", "v_rel", "edge_pre", "edge_post", "edge_min")})
    for i, r in enumerate(runs):
        a, b = r["start"], r["end"]
        cols["start"][i], cols["end"][i], cols["length"][i], cols["period"][i] = a, b, b - a, period[a]
        cols["pre"][i] = -1 if r["pre_idx"] is None else r["pre_idx"]
        cols["post"][i] = -1 if r["post_idx"] is None else r["post_idx"]
        seg = spd[a + e:b - e]
        if len(seg) and np.isfinite(seg).any():
            cols["v_abs"][i] = np.nanmean(seg)
        cols["v_ref"][i] = v_ref[period[a]]
        if r["pre_idx"] is not None:
            cols["edge_pre"][i] = edge_dist[r["pre_idx"]]
        if r["post_idx"] is not None:
            cols["edge_post"][i] = edge_dist[r["post_idx"]]
    with np.errstate(invalid="ignore", divide="ignore"):
        cols["v_rel"] = cols["v_abs"] / cols["v_ref"]
        cols["edge_min"] = np.fmin(cols["edge_pre"], cols["edge_post"])               # fmin ignores a NaN side
    return cols


def decide(table, theta, l_max_s=np.inf, edge_m=None):
    """(H,) bool per hole of `table`: True = alive (see the module docstring)."""
    alive = (table["length"] <= l_max_s * FPS) & (table["v_rel"] >= theta)         # NaN compares False
    if edge_m is not None:
        alive &= ~(table["edge_min"] < edge_m)
    return alive


def fill(labels, table, alive, mode):
    """`labels` with the alive holes filled with the owner of `mode` ('pre', 'post', 'split'); the rest untouched."""
    if mode not in MODES:
        raise ValueError(mode)
    out = np.array(labels, dtype="<U1")
    for i in np.flatnonzero(alive):
        a, b = int(table["start"][i]), int(table["end"][i])
        pre = None if table["pre"][i] < 0 or out[table["pre"][i]] == "D" else str(out[table["pre"][i]])
        post = None if table["post"][i] < 0 or out[table["post"][i]] == "D" else str(out[table["post"][i]])
        if mode == "pre":
            out[a:b] = pre or "D"
        elif mode == "post":
            out[a:b] = post or "D"
        else:
            mid = a + (b - a) // 2
            out[a:mid] = pre or "D"
            out[mid:b] = post or "D"
    return out


def predict(labels, ball_xy, players_slots, period, pitch, theta, l_max_s=np.inf, edge_m=None, mode="pre",
            table=None, ref_mask=None):
    """Base labels with the alive holes filled. `table` (hole_table output) can be passed to sweep a grid."""
    table = table if table is not None else hole_table(ball_xy, players_slots, period, pitch, ref_mask=ref_mask)
    return fill(labels, table, decide(table, theta, l_max_s, edge_m), mode)
