"""Persistent possession owner between touches (pure array functions, no I/O, same shapes as reglas.py).

The ground truth is "last touch": possession belongs to the team that touched the ball last, until the
next touch. Here a touch is detected from the ball track alone, inside the stretches where the ball is
visible, and the owner is held from touch to touch instead of re-voting every frame.

Touch candidates (per visible stretch, never across a period break):
  deflection  |dv| >= dv_thr, where dv is the vector change of the ball velocity between the `win` frames
              before and after a frame (ball smoothed with a `win`-frame moving average first). The peak of
              each run of candidate frames counts as a touch when a player is within `r` metres of the
              ball around that frame; the touch goes to the team of the nearest player.
  control     the same team is the nearest one, closer than `r_ctrl`, for >= `k` consecutive frames
              (reception or dribbling without deflection). The touch is the first frame of the run.
              Disabled when k is None or 0.

Owner = team of the last touch, held until the next one. The owner is kept across a hole of at most
`max_gap_s` inside the same period (the next stretch continues with it until its own first touch); after
a longer hole (which is D) or a period change it is invalidated and the frames of the next stretch use the
base (nearest player, +-`base_window` majority) until its first touch. Frames with no ball follow the base
rule: carry the label of the last visible frame, and a hole longer than `max_gap_s` is D.
keep_owner=False restores the earlier behaviour (owner reset at every stretch), kept only to measure the fix.

Inputs: ball_xy (N,2) normalised 0-1 with NaN, players_slots {"H": (N,24,2), "A": (N,22,2)} with NaN,
period (N,), pitch = (length_m, width_m). Nothing here reads a ground truth.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import onside_paths  # noqa: E402
onside_paths.add_challenge_to_path()
import reglas  # noqa: E402
from baseline import HOME, AWAY, NOBODY  # noqa: E402

FPS = reglas.FPS
DEFAULTS = {"r": 2.0, "dv": 4.0, "win": 3, "r_ctrl": 1.0, "k": 12}


def prepare(ball_xy, players_slots, pitch):
    """Per-frame distance (m) to the nearest player and the team of that player (HOME/AWAY/NOBODY).
    Independent of the detector parameters, so a caller sweeping a grid computes it once."""
    scale = np.asarray(pitch, float)
    ball = np.asarray(ball_xy, float) * scale
    best, dist = {}, {}
    for key in ("H", "A"):
        p = np.asarray(players_slots[key], float) * scale
        d = np.linalg.norm(p - ball[:, None, :], axis=2)
        dist[key] = np.where(np.isnan(d), np.inf, d).min(axis=1)
    h, a = dist["H"], dist["A"]
    dmin = np.minimum(h, a)
    team = np.where(h <= a, HOME, AWAY).astype(np.int8)
    team[~np.isfinite(dmin)] = NOBODY
    return {"ball_m": ball, "dmin": dmin, "team": team}


def visible_runs(present, period):
    """[(start, stop)] maximal runs of visible ball, split at period changes."""
    present = np.asarray(present, bool)
    period = np.asarray(period)
    n = len(present)
    if n == 0:
        return []
    cut = np.flatnonzero(period[1:] != period[:-1]) + 1
    edges = np.concatenate(([0], cut, [n]))
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        d = np.diff(np.concatenate(([0], present[lo:hi].astype(np.int8), [0])))
        out += [(lo + int(s), lo + int(e)) for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]
    return out


def _moving_average(x, w):
    """Centred moving average over axis 0 with the window shrunk at the borders."""
    if w <= 1:
        return x
    h = w // 2
    c = np.concatenate((np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)))
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - h), np.minimum(len(x), i + h + 1)
    return (c[hi] - c[lo]) / (hi - lo).reshape((-1,) + (1,) * (x.ndim - 1))


def _runs_of_true(m):
    d = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def deflection_candidates(near, runs, r=2.0, dv=4.0, win=3):
    """Deflection touches inside the visible stretches `runs`, with what the detector saw at each one.
    Returns a dict of aligned arrays: frame (touch frame), team (HOME/AWAY), peak (frame of the |dv| peak),
    start / stop (the visible stretch that contains it), vp / vn ((C, 2) ball velocity in m/s before and
    after the peak) and dv (|vn - vp|). `near` comes from prepare()."""
    ball, dmin, team = near["ball_m"], near["dmin"], near["team"]
    q = max(2, win // 2)
    cols = {k: [] for k in ("frame", "team", "peak", "start", "stop", "vp", "vn", "dv")}
    for a, b in runs:
        L = b - a
        if L <= 2 * win:
            continue
        ps = _moving_average(ball[a:b], win)
        vp = (ps[win:L - win] - ps[:L - 2 * win]) * FPS / win           # frame i in [win, L-win)
        vn = (ps[2 * win:] - ps[win:L - win]) * FPS / win
        dvm = np.linalg.norm(vn - vp, axis=1)
        s_, e_ = _runs_of_true(dvm >= dv)
        for s, e in zip(s_, e_):
            k = int(np.argmax(dvm[s:e]))
            i = win + s + k                                             # peak, run-relative frame
            lo, hi = max(0, i - q), min(L, i + q + 1)
            j = lo + int(np.argmin(dmin[a + lo:a + hi]))
            if dmin[a + j] < r and team[a + j] != NOBODY:
                for key, v in zip(cols, (a + j, team[a + j], a + i, a, b, vp[s + k], vn[s + k], dvm[s + k])):
                    cols[key].append(v)
    out = {k: np.asarray(v) for k, v in cols.items()}
    out["frame"], out["team"] = out["frame"].astype(int), out["team"].astype(np.int8)
    for k in ("peak", "start", "stop"):
        out[k] = out[k].astype(int)
    for k in ("vp", "vn"):
        out[k] = out[k].reshape(-1, 2).astype(float)
    out["dv"] = out["dv"].astype(float)
    return out


def control_touches(near, runs, r_ctrl=1.0, k=12):
    """Control touches: the same team is the nearest one, closer than `r_ctrl`, for >= `k` consecutive frames;
    the touch is the first frame of the run. Returns (frames, teams) in the order of runs, HOME before AWAY.
    Empty when k is 0/None."""
    dmin, team = near["dmin"], near["team"]
    frames, teams = [], []
    if k:
        for a, b in runs:
            close = (dmin[a:b] < r_ctrl) & (team[a:b] != NOBODY)
            for t in (HOME, AWAY):
                s_, e_ = _runs_of_true(close & (team[a:b] == t))
                for s, e in zip(s_, e_):
                    if e - s >= k:
                        frames.append(a + int(s))
                        teams.append(t)
    return np.asarray(frames, int), np.asarray(teams, np.int8)


def merge_touches(*parts):
    """Sorted union of (frames, teams) lists (stable: on equal frames the earlier part comes first)."""
    frames = np.concatenate([np.asarray(f, int) for f, _ in parts]) if parts else np.zeros(0, int)
    if not len(frames):
        return np.zeros(0, int), np.zeros(0, np.int8)
    teams = np.concatenate([np.asarray(t, np.int8) for _, t in parts])
    order = np.argsort(frames, kind="stable")
    return frames[order], teams[order]


def detect_touches(near, runs, r=2.0, dv=4.0, win=3, r_ctrl=1.0, k=12):
    """Touches inside the visible stretches `runs`. Returns (frames, teams): sorted frame indices and
    HOME/AWAY codes (int8). `near` comes from prepare()."""
    d = deflection_candidates(near, runs, r=r, dv=dv, win=win)
    return merge_touches((d["frame"], d["team"]), control_touches(near, runs, r_ctrl=r_ctrl, k=k))


def owner_labels(n, runs, t_frames, t_teams, period=None, max_gap_frames=None):
    """(N,) "H"/"A"/"" : team of the last touch, "" where there is no valid owner (before the first touch
    after an invalidation, and outside the stretches).

    Without `period` / `max_gap_frames` the owner restarts in every visible stretch. With both, it is carried
    from one stretch to the next when they are in the same period and the hole between them is at most
    `max_gap_frames`; otherwise it is invalidated."""
    out = np.full(n, "", dtype="<U1")
    letter = np.array(["H", "A"])
    keep = period is not None and max_gap_frames is not None
    carry, prev_end = None, None
    for a, b in runs:
        if keep:
            same = prev_end is not None and period[a] == period[prev_end - 1] and a - prev_end <= max_gap_frames
            if not same:
                carry = None
            prev_end = b
        lo, hi = np.searchsorted(t_frames, a), np.searchsorted(t_frames, b)
        seg = out[a:b]
        if carry is not None:
            seg[:] = letter[carry]
        if hi > lo:
            ev, tm = t_frames[lo:hi], t_teams[lo:hi]
            idx = np.searchsorted(ev, np.arange(a, b), side="right") - 1
            ok = idx >= 0
            seg[ok] = letter[tm[idx[ok]]]
            carry = int(tm[-1]) if keep else None
    return out


def predict(ball_xy, players_slots, period, pitch, r=2.0, dv=4.0, win=3, r_ctrl=1.0, k=12,
            max_gap_s=10, base_window=50, near=None, base=None, touches=None, keep_owner=True,
            return_touches=False):
    """(N,) array of "H"/"A"/"D". `near` (prepare output), `base` (reglas "B0" labels) and `touches`
    ((frames, teams) from detect_touches, which does not depend on max_gap_s) can be passed to avoid
    recomputing them over a parameter grid."""
    period = np.asarray(period)
    present = ~np.isnan(np.asarray(ball_xy, float)[:, 0])
    near = near if near is not None else prepare(ball_xy, players_slots, pitch)
    base = base if base is not None else reglas.predict("B0", ball_xy, players_slots, period, pitch,
                                                        window=base_window)
    runs = visible_runs(present, period)
    tf, tt = touches if touches is not None else detect_touches(near, runs, r=r, dv=dv, win=win, r_ctrl=r_ctrl, k=k)
    own = owner_labels(len(present), runs, tf, tt, *((period, int(max_gap_s * FPS)) if keep_owner else ()))
    s = np.where(own != "", own, base).astype("<U1")
    out = reglas.fill_gaps(s, present, period, "F", max_gap_s)
    return (out, tf, tt) if return_touches else out
