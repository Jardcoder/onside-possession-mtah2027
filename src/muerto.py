"""Dead-ball (`D`) evidence with a visible ball, and long holes split in two (pure array functions, no I/O).

Two pieces applied on top of the labels of toques.predict (or any (N,) array of "H"/"A"/"D"). Nothing here reads a
ground truth, a player id, an absolute coordinate as a feature on its own, or the length of a hole as a predictor
of anything but the (fixed) cap that already exists.

C1  dead_visible_mask(): frames where the ball is visible and, for at least `t_min` seconds in a row, either
      * the ball is outside the pitch by more than `margin` metres (pitch in metres, origin at a corner), or
      * the ball is almost still (speed averaged over one second < `v_bal` m/s) AND the collective activity is low
        (p75 over the present players of each player's speed averaged over +-2 s < `v_jug` m/s).
    Speeds are never taken across a hole: a step needs both frames visible / present, and a period change is a cut.
    apply_dead() turns those frames into "D".
C2  split_long_holes(): a hole longer than the cap is D in the base. Here it stays D until `s_rest` seconds before the
    ball reappears; those last `s_rest` seconds take the label of the first visible frame after the hole (the owner
    when the ball reappears), because the restart usually happens inside the hole. If that label is D, or the hole
    has no visible frame after it in the same period, the hole stays D.

Inputs as in toques.py: ball_xy (N,2) normalised 0-1 with NaN, players_slots {"H": (N,24,2), "A": (N,22,2)} with
NaN, period (N,), pitch = (length_m, width_m).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import reglas  # noqa: E402

FPS = reglas.FPS
BALL_WIN_S = 1.0          # window (s) of the ball speed average
CROWD_WIN_S = 2.0         # half window (s) of the player speed average
STEP_MAX = 12.0           # m/s: faster steps are tracking jumps (slot swaps), ignored
MIN_VALID_S = 1.0         # a player needs this many seconds of valid steps in the window to count


def _centered_speed(xy_m, period):
    """(N,) or (N,K) speed (m/s) from the centred difference over frames i-1 and i+1; NaN when either is missing or the
    period changes. Works on (N,2) and (N,K,2) arrays."""
    v = np.full(xy_m.shape[:-1], np.nan)
    d = np.linalg.norm(xy_m[2:] - xy_m[:-2], axis=-1) * FPS / 2
    cut = (np.asarray(period)[2:] != np.asarray(period)[:-2])
    d = np.where(cut.reshape((-1,) + (1,) * (d.ndim - 1)), np.nan, d)
    v[1:-1] = d
    return v


def _nan_window_mean(x, half, min_valid=1):
    """Mean of the non-NaN values of x within +-half frames along axis 0 (NaN where fewer than `min_valid`)."""
    ok = ~np.isnan(x)
    c = np.concatenate((np.zeros((1,) + x.shape[1:]), np.cumsum(np.where(ok, x, 0.0), axis=0)))
    n = np.concatenate((np.zeros((1,) + x.shape[1:]), np.cumsum(ok, axis=0)))
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - half), np.minimum(len(x), i + half + 1)
    cnt = n[hi] - n[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (c[hi] - c[lo]) / cnt
    out[cnt < min_valid] = np.nan
    return out


def ball_features(ball_xy, period, pitch):
    """(out_m, v_ball) of `features`: the ball part, so a ball-only change (calibration of the dead-ball motion) does
    not recompute the player speeds."""
    scale = np.asarray(pitch, float)
    ball = np.asarray(ball_xy, float) * scale
    out_m = np.maximum.reduce([-ball[:, 0], ball[:, 0] - scale[0], -ball[:, 1], ball[:, 1] - scale[1]])
    v_ball = _nan_window_mean(_centered_speed(ball, period), int(BALL_WIN_S * FPS) // 2, min_valid=FPS // 2)
    return out_m, v_ball


def features(ball_xy, players_slots, period, pitch):
    """Per-frame signals that do not depend on the thresholds, computed once per match:
    out_m    metres the ball is beyond the pitch (max over the four lines; <= 0 inside), NaN with no ball
    v_ball   ball speed (m/s) averaged over one second of visible frames, NaN where there is no valid speed
    v_crowd  p75 over the present players of their speed averaged over +-2 s (m/s), NaN with no player"""
    scale = np.asarray(pitch, float)
    out_m, v_ball = ball_features(ball_xy, period, pitch)
    sp = []
    for key in ("H", "A"):
        v = _centered_speed(np.asarray(players_slots[key], float) * scale, period)
        sp.append(np.where(v > STEP_MAX, np.nan, v))
    sp = np.concatenate(sp, axis=1)
    per_player = _nan_window_mean(sp, int(CROWD_WIN_S * FPS), min_valid=int(MIN_VALID_S * FPS))
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v_crowd = np.nanpercentile(per_player, 75, axis=1)
    return {"out_m": out_m, "v_ball": v_ball, "v_crowd": v_crowd}


def _runs_at_least(mask, min_len):
    """Boolean mask keeping only the runs of True of length >= min_len."""
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    keep = np.zeros(len(mask), bool)
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if e - s >= min_len:
            keep[s:e] = True
    return keep


def dead_visible_mask(feat, t_min=3.0, v_bal=None, v_jug=None, margin=None):
    """(N,) bool: frames inside a run of at least `t_min` seconds where the ball is outside the pitch by more than
    `margin` m (margin=None: off) or is still and the players calm (v_bal / v_jug None: off). Both off gives an empty mask."""
    cond = np.zeros(len(feat["out_m"]), bool)
    if margin is not None:
        cond |= feat["out_m"] > margin                                   # NaN compares False
    if v_bal is not None and v_jug is not None:
        cond |= (feat["v_ball"] < v_bal) & (feat["v_crowd"] < v_jug)
    return _runs_at_least(cond, int(round(t_min * FPS)))


def apply_dead(labels, mask):
    out = np.array(labels, dtype="<U1")
    out[mask] = "D"
    return out


def split_long_holes(labels, present, period, max_gap_s, s_rest):
    """Last `s_rest` seconds of every hole longer than `max_gap_s` take the label of the first visible frame after it
    (see the module docstring). s_rest None or 0: unchanged. The labels of the hole must already be D."""
    out = np.array(labels, dtype="<U1")
    if not s_rest:
        return out
    n_rest = int(round(s_rest * FPS))
    for r in reglas.gap_runs(present, period):
        a, b = r["start"], r["end"]
        if (b - a) <= max_gap_s * FPS or r["post_idx"] is None:
            continue
        post = out[r["post_idx"]]
        if post == "D":
            continue
        out[max(a, b - n_rest):b] = post
    return out


def predict(labels, ball_xy, players_slots, period, pitch, max_gap_s=10, t_min=3.0, v_bal=None, v_jug=None,
            margin=None, s_rest=None, feat=None):
    """Labels of toques.predict (or the base) with C1 and C2 applied; both off returns them unchanged. C1 first, so that
    C2 does not hang an owner tail in front of frames that C1 declares dead. `feat` (features output) can be passed
    to avoid recomputing it over a parameter grid."""
    present = ~np.isnan(np.asarray(ball_xy, float)[:, 0])
    out = np.array(labels, dtype="<U1")
    if margin is not None or (v_bal is not None and v_jug is not None):
        feat = feat if feat is not None else features(ball_xy, players_slots, period, pitch)
        out = apply_dead(out, dead_visible_mask(feat, t_min, v_bal, v_jug, margin))
    return split_long_holes(out, present, period, max_gap_s, s_rest)
