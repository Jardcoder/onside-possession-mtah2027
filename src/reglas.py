"""Pure possession rules over arrays (no I/O), same data shape for DFL banks and the challenge.

Inputs: ball_xy (N,2) normalised 0-1 with NaN, players_slots {"H": (N,24,2), "A": (N,22,2)} with NaN,
period (N,), pitch = (length_m, width_m).
"""

import numpy as np

import onside_paths  # noqa: E402
onside_paths.add_challenge_to_path()
from baseline import HOME, AWAY, NOBODY, smooth  # noqa: E402

FPS = 25
VARIANTS = ("B0", "F", "T_back", "T_comb_D", "T_comb_mid")


def nearest_codes(ball_xy, players_slots, pitch):
    """(N,) int8: HOME/AWAY of the player closest to the ball (metres, NaN slots ignored); NOBODY if
    the ball is missing or no player is present."""
    scale = np.asarray(pitch, float)
    ball = np.asarray(ball_xy, float) * scale
    best = {}
    for key in ("H", "A"):
        p = np.asarray(players_slots[key], float) * scale
        d = np.linalg.norm(p - ball[:, None, :], axis=2)
        d = np.where(np.isnan(d), np.inf, d)
        best[key] = d.min(axis=1)
    h, a = best["H"], best["A"]
    codes = np.where(h <= a, HOME, AWAY).astype(np.int8)
    codes[~np.isfinite(np.minimum(h, a))] = NOBODY
    return codes


def gap_runs(ball_present, period):
    """Maximal runs of absent ball, split at period changes.

    Returns a list of dicts: start (inclusive), end (exclusive), pre_idx / post_idx = index of the
    last visible frame before / first visible frame after the run in the same period, or None when
    the run touches a period border (or the array border)."""
    absent = ~np.asarray(ball_present, bool)
    period = np.asarray(period)
    n = len(absent)
    if n == 0:
        return []
    brk = np.flatnonzero(period[1:] != period[:-1]) + 1
    edges = np.concatenate(([0], brk, [n]))
    runs = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = absent[lo:hi].astype(np.int8)
        d = np.diff(np.concatenate(([0], m, [0])))
        for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            s, e = lo + int(s), lo + int(e)
            runs.append({"start": s, "end": e,
                         "pre_idx": s - 1 if s > lo else None,
                         "post_idx": e if e < hi else None})
    return runs


def _valid(s, idx):
    if idx is None:
        return None
    v = s[idx]
    return None if v == "D" else str(v)


def fill_gaps(s, ball_present, period, variant, max_gap_s=None):
    """Fill the absent-ball frames of the smoothed labels `s` according to `variant`."""
    out = np.array(s, dtype="<U1")
    for r in gap_runs(ball_present, period):
        a, b = r["start"], r["end"]
        pre, post = _valid(s, r["pre_idx"]), _valid(s, r["post_idx"])
        if max_gap_s is not None and (b - a) > max_gap_s * FPS:
            out[a:b] = "D"
            continue
        if variant == "F":
            out[a:b] = pre or post or "D"
        elif variant == "T_back":
            out[a:b] = post or pre or "D"
        elif variant in ("T_comb_D", "T_comb_mid"):
            if pre and post and pre != post:
                if variant == "T_comb_D":
                    out[a:b] = "D"
                else:
                    mid = a + (b - a) // 2
                    out[a:mid] = pre
                    out[mid:b] = post
            else:
                out[a:b] = pre or post or "D"
        else:
            raise ValueError(variant)
    return out


def predict(variant, ball_xy, players_slots, period, pitch, window=50, max_gap_s=None):
    """(N,) array of "H"/"A"/"D". B0 is the official baseline; the others fill ball gaps."""
    codes = nearest_codes(ball_xy, players_slots, pitch)
    # Smooth each period on its own: a centred window must not vote across the half-time break.
    period = np.asarray(period)
    s = np.empty(len(codes), dtype="<U1")
    for p in np.unique(period):
        m = period == p
        s[m] = smooth(codes[m], window)
    if variant == "B0":
        return s
    present = ~np.isnan(np.asarray(ball_xy, float)[:, 0])
    return fill_gaps(s, present, period, variant, max_gap_s)
