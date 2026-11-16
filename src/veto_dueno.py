"""Geometric veto of the owner change at a deflection (candidate C2; pure array functions, no I/O).

toques.py hands the possession to the team of every deflection touch (`dir` picks that team). Here a deflection that WOULD
CHANGE the owner is dropped (the previous owner keeps the ball; the touch is not reassigned) unless the rival j that `dir`
chose passes all three conditions at the touch frame t:
  1. d(j, ball) <= r_chg                       the rival is really on the ball
  2. d(j, ball) - d(previous owner, ball) <= d_rel   and it is not clearly farther than the previous owner
  3. d(j, ball) has a local minimum in [t-2, t+2]     the rival is at its closest around the touch (a valley, not a
                                                    monotone approach or retreat)
d(team, ball) = distance (m) of the nearest player of that team to the ball. Previous owner = the team that owns the frame
t-1 in the prediction being built (the owner of the last KEPT touch, carried across holes as toques.owner_labels does,
else the base label of that frame; across a hole of at most `max_gap_frames` the label of the last visible frame). If the
previous owner is D / nobody, or the touch does not change it, there is nothing to veto. Control touches are never vetoed.

Condition 3 is "some frame of the window, other than its two ends, reaches the window minimum" (ties count); a window
clipped by the visible stretch to fewer than five frames cannot be judged and passes. d_rel = inf switches condition 2 off.

Inputs: ball_xy (N,2) normalised, players_slots {"H","A"}, pitch, the base labels ("H"/"A"/"D") and the touches of
toques.deflection_candidates / control_touches. Nothing here reads a ground truth.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import onside_paths  # noqa: E402
onside_paths.add_challenge_to_path()
from baseline import HOME, AWAY  # noqa: E402

HALF_WINDOW = 2


def team_distances(ball_xy, players_slots, pitch):
    """(N, 2) distance in metres from the ball to the nearest HOME / AWAY player (inf where nobody or no ball)."""
    scale = np.asarray(pitch, float)
    ball = np.asarray(ball_xy, float) * scale
    out = np.empty((len(ball), 2))
    for col, key in ((HOME, "H"), (AWAY, "A")):
        d = np.linalg.norm(np.asarray(players_slots[key], float) * scale - ball[:, None, :], axis=2)
        out[:, col] = np.where(np.isnan(d), np.inf, d).min(axis=1)
    return out


def passes_veto(dists, a, b, f, j, prev, r_chg, d_rel, half=HALF_WINDOW):
    """True when the change of owner prev -> j at frame f (inside the visible stretch [a, b)) is allowed."""
    dj = dists[f, j]
    if not dj <= r_chg:
        return False
    if not dj - dists[f, prev] <= d_rel:
        return False
    w = dists[max(a, f - half):min(b, f + half + 1), j]
    if len(w) < 2 * half + 1:
        return True
    return bool(w[1:-1].min() <= w.min())


def veto_owner_change(frames, teams, is_defl, dists, runs, period, base, r_chg, d_rel, max_gap_frames):
    """(T,) bool: touches kept. `frames` sorted, `teams` HOME/AWAY, `is_defl` marks the deflections (the only ones that can
    be dropped), `runs` the visible stretches, `base` the (N,) "H"/"A"/"D" labels of reglas "B0"."""
    frames = np.asarray(frames, int)
    keep = np.ones(len(frames), bool)
    period = np.asarray(period)
    code = {"H": HOME, "A": AWAY}
    ti, prev_end, carry, end_label = 0, None, None, None
    for a, b in runs:
        linked = prev_end is not None and period[a] == period[prev_end - 1] and a - prev_end <= max_gap_frames
        if not linked:
            carry, entry = None, None
        else:
            entry = end_label
        owner = carry
        while ti < len(frames) and frames[ti] < b:
            f, t = int(frames[ti]), int(teams[ti])
            if is_defl[ti]:
                if f > a:
                    prev = owner if owner is not None else code.get(str(base[f - 1]))
                else:
                    prev = entry
                if prev is not None and t != prev and not passes_veto(dists, a, b, f, t, prev, r_chg, d_rel):
                    keep[ti] = False
            if keep[ti]:
                owner = t
            ti += 1
        carry = owner
        end_label = owner if owner is not None else code.get(str(base[b - 1]))
        prev_end = b
    return keep
