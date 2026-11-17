"""Absorption of micro-possessions (candidate R2; pure function over the H / A / D labels of a prediction).

A stretch of H or A (maximal constant-label run, never crossing a period change) that lasts LESS than tau seconds and whose
immediately preceding and immediately following stretches are both of the opposite team (contiguous: no D in between, same
period) is handed to that team. Decided on the labels as they come in, one pass: a stretch absorbed does not change the
neighbours another decision looks at. tau = 0 absorbs nothing (null configuration: the output equals the input).

Inputs: the label array and, optionally, the period array only to cut stretches at the half-time. Nothing else is read.
`absorb_conditional` (candidate R4) adds the frames and teams of the control touches the detector found, so that a short
stretch with a control touch inside is left alone.
"""
import numpy as np

FPS = 25                                       # frames per second of the tracking (same as toques.FPS)


def absorb(labels, tau_s, per=None, fps=FPS):
    """Copy of `labels` ("H"/"A"/"D", (N,)) with the micro-possessions shorter than `tau_s` seconds given to the rival."""
    out = np.array(labels, copy=True)
    n = len(out)
    if n == 0 or tau_s <= 0:
        return out
    change = out[1:] != out[:-1]
    if per is not None:
        change |= np.asarray(per)[1:] != np.asarray(per)[:-1]
    starts = np.concatenate(([0], np.flatnonzero(change) + 1))
    stops = np.concatenate((starts[1:], [n]))
    lab = out[starts]                           # label of each stretch (the input, before any absorption)
    same_period = np.ones(len(starts), bool)    # stretch i and i + 1 share a period (always true if `per` is None)
    if per is not None:
        p = np.asarray(per)
        same_period = np.concatenate((p[starts[:-1]] == p[starts[1:]], [False]))
    for i in range(1, len(starts) - 1):
        if lab[i] not in ("H", "A") or (stops[i] - starts[i]) / fps >= tau_s:
            continue
        rival = "A" if lab[i] == "H" else "H"
        if lab[i - 1] == rival and lab[i + 1] == rival and same_period[i - 1] and same_period[i]:
            out[starts[i]:stops[i]] = rival
    return out


def absorb_conditional(labels, tau_s, ctrl_frames, ctrl_teams, mode, per=None, fps=FPS):
    """Candidate R4: `absorb`, except that a stretch holding a control touch is not absorbed.

    Same candidates as `absorb` (an H / A stretch shorter than `tau_s` seconds between two stretches of the rival, same period, no D
    in between), judged on the labels as they come in. A candidate is kept as it is when a control touch falls inside it, i.e. at a
    frame in [start, stop) of the stretch. `ctrl_frames` are the frames of the control touches (`ctx["ctrl"]`, any order) and
    `ctrl_teams` their team as "H" / "A" (same length). Mode "any": a control touch of either team protects the stretch; mode "own":
    only one of the team of the stretch does. With no control touch at all this is `absorb`; tau = 0 absorbs nothing."""
    if mode not in ("any", "own"):
        raise ValueError(f"mode must be 'any' or 'own', got {mode!r}")
    out = np.array(labels, copy=True)
    n = len(out)
    if n == 0 or tau_s <= 0:
        return out
    cf = np.asarray(ctrl_frames, int)
    ct = np.asarray(ctrl_teams)
    order = np.argsort(cf, kind="stable")
    cf, ct = cf[order], ct[order]
    change = out[1:] != out[:-1]
    if per is not None:
        change |= np.asarray(per)[1:] != np.asarray(per)[:-1]
    starts = np.concatenate(([0], np.flatnonzero(change) + 1))
    stops = np.concatenate((starts[1:], [n]))
    lab = out[starts]
    same_period = np.ones(len(starts), bool)
    if per is not None:
        p = np.asarray(per)
        same_period = np.concatenate((p[starts[:-1]] == p[starts[1:]], [False]))
    for i in range(1, len(starts) - 1):
        if lab[i] not in ("H", "A") or (stops[i] - starts[i]) / fps >= tau_s:
            continue
        rival = "A" if lab[i] == "H" else "H"
        if not (lab[i - 1] == rival and lab[i + 1] == rival and same_period[i - 1] and same_period[i]):
            continue
        lo, hi = np.searchsorted(cf, [starts[i], stops[i]], side="left")
        if hi > lo and (mode == "any" or (ct[lo:hi] == lab[i]).any()):
            continue
        out[starts[i]:stops[i]] = rival
    return out
