"""Border detector and variable-price Viterbi on top of T1w.

T1w decodes each period with a Viterbi whose price for changing state is the same everywhere (`lam_ha` for H<->A, `lam_d` for entering
or leaving D). On the PFF development matches the dominant loss is where each possession starts and ends, not which team owns it: with the
team of every T1w stretch set to the truth the gain is small, with the borders of the truth and the labels of T1w it is large. This module
keeps the probabilities and the lambda of T1w and makes a change cheaper only near a frame where a small LightGBM detector says a possession
edge is likely:

    price(t) = lam * f   if t is within k frames of a detector peak
             = lam * g   everywhere else

With f = g = 1 the path is exactly the one of `modelo.viterbi`. Peaks, and the frames near them, never cross a period.

The detector reads the T1w feature matrix without the ball-presence family (in the PFF training matches the injected holes are anchored to
real dead-ball stops, so those columns mark edges there and not in the challenge). The target of the detector is built, and the weights are
trained, in bancos/ronda_bordes.py, which reads the PFF labels to do so exactly as the T1w boosters do; nothing here reads the ground truth.

    border_columns(names)              indices of the T1w feature columns the detector reads
    column_index(names, wanted)        indices of `wanted` names in `names`, in that order (raises if one is missing)
    transitions(codes, runs)           first frame of every label change inside a run
    near(events, k, n)                 (n,) bool: within k frames of an event
    peaks(p, threshold, w)             frames where p >= threshold and p is the maximum of its +-w window (first frame of a plateau)
    multipliers(p, configs, per)       (C, N) price multiplier of each setting
    viterbi_variable(logp, per, lam_ha, lam_d, mult)   modelo.viterbi with the price of a change multiplied by mult[c, t]
    decode(P, per, lam, p_border, config)              the T1w path with the prices of the detector
    load_detector(directory)           (booster, meta) of models/detector_bordes.{txt,json}, SHA-256 and settings checked

Codes as in `possession_metrics`: 0 = H, 1 = A, 2 = D, 3 = X (not graded).
"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import onside_paths  # noqa: E402

MODELS_DIR = onside_paths.MODELS_DIR                          # ONSIDE_MODELS_DIR, default ./models: the committed weights of the detector
MODEL_FILE = "detector_bordes.txt"
META_FILE = "detector_bordes.json"

PRESENCE = ("ball_present", "ball_since_visible_s", "ball_until_visible_s", "b3_dead", "b3abs_dead")   # base names; windows are `<base>@p0.2`, `@f1`, ...
PEAK_WINDOW = 6                    # frames on each side that a peak has to dominate
TARGET_TOL = 3                     # a frame is a positive of the detector if a true edge is at most this many frames away
NEG_FRACTION = 0.04                # share of the negative frames kept for training (they get the weight 1 / NEG_FRACTION)
H, A, D, X = 0, 1, 2, 3


def border_columns(names):
    """Indices of the columns of the T1w feature matrix that the detector reads: all but the ball-presence family and its windows."""
    return np.array([i for i, nm in enumerate(names) if nm.split("@")[0] not in PRESENCE], dtype=np.int64)


def column_index(names, wanted):
    """Indices of the names `wanted` in `names`, in the order of `wanted`. Raises ValueError listing the missing ones, so a feature matrix
    whose layout drifted from the one the detector was trained on fails loudly instead of feeding the wrong columns."""
    pos = {nm: i for i, nm in enumerate(names)}
    missing = [nm for nm in wanted if nm not in pos]
    if missing:
        raise ValueError(f"{len(missing)} detector features not in the feature matrix, e.g. {missing[:3]}")
    return np.array([pos[nm] for nm in wanted], dtype=np.int64)


def transitions(codes, runs):
    """Frames t (a < t < b) of every run [a, b) where codes[t] != codes[t - 1]."""
    codes = np.asarray(codes)
    out = [a + np.flatnonzero(codes[a:b][1:] != codes[a:b][:-1]) + 1 for a, b in runs]
    return np.concatenate(out) if out else np.zeros(0, np.int64)


def near(events, k, n):
    """(n,) bool: frames at most k frames from some event."""
    m = np.zeros(n + 1, np.int64)
    e = np.asarray(events, np.int64)
    e = e[(e >= 0) & (e < n)]
    np.add.at(m, np.maximum(e - k, 0), 1)
    np.add.at(m, np.minimum(e + k + 1, n), -1)
    return np.cumsum(m)[:n] > 0


def peaks(p, threshold, w=PEAK_WINDOW):
    """Frames where p >= threshold and p is the maximum of [t - w, t + w] (the window is clipped at the ends by repeating the edge value);
    on a plateau only its first frame counts."""
    p = np.asarray(p, float)
    if not len(p):
        return np.zeros(0, np.int64)
    win = np.lib.stride_tricks.sliding_window_view(np.pad(p, w, mode="edge"), 2 * w + 1)
    cand = (p >= threshold) & (p >= win.max(axis=1))
    cand[1:] &= ~(cand[:-1] & (p[1:] == p[:-1]))
    return np.flatnonzero(cand)


def _period_slices(per):
    """[(first, one past the last)] of every period of the array `per`."""
    per = np.asarray(per)
    starts = np.concatenate(([0], np.flatnonzero(per[1:] != per[:-1]) + 1)) if len(per) else np.zeros(0, np.int64)
    return list(zip(starts.tolist(), np.concatenate((starts[1:], [len(per)])).astype(int).tolist()))


def multipliers(p_border, configs, per):
    """(len(configs), N) price multiplier of each setting {"threshold", "f", "k", "g"}: f within k frames of a peak of p_border, g elsewhere.
    The peaks and the frames near them are found inside each period, so nothing crosses the change of period (the Viterbi cuts there)."""
    p = np.asarray(p_border, float)
    if len(p) != len(per):
        raise ValueError(f"p_border has {len(p)} rows and per {len(per)}")
    out = np.empty((len(configs), len(p)))
    for a, b in _period_slices(per):
        events, around = {}, {}
        for i, c in enumerate(configs):
            u = c["threshold"]
            if u not in events:
                events[u] = peaks(p[a:b], u, PEAK_WINDOW)
            if (u, c["k"]) not in around:
                around[(u, c["k"])] = near(events[u], c["k"], b - a)
            out[i, a:b] = np.where(around[(u, c["k"])], c["f"], c["g"])
    return out


def viterbi_variable(logp, per, lam_ha, lam_d, mult):
    """Viterbi over the states 0 = H, 1 = A, 2 = D, decoded independently inside each period, like `modelo.viterbi` but the price of changing state
    on arriving at frame t is lam * mult[c, t] (lam_ha for H<->A, lam_d for entering or leaving D). mult is (C, N): C settings decoded in one pass.
    With mult = 1 it returns exactly the path of `modelo.viterbi`. Returns (C, N) int8."""
    logp = np.asarray(logp, float)
    mult = np.asarray(mult, float)
    c, n = mult.shape
    out = np.zeros((c, n), np.int8)
    if not n:
        return out
    price = np.zeros((3, 3))
    price[0, 1] = price[1, 0] = lam_ha
    for s in (0, 1):
        price[s, 2] = price[2, s] = lam_d
    rows = np.arange(c)
    for a, b in _period_slices(per):
        lp, m = logp[a:b], mult[:, a:b]
        score = np.repeat(lp[:1], c, axis=0)
        back = np.zeros((b - a, c, 3), np.int8)
        for t in range(1, b - a):
            cand = score[:, :, None] - price[None] * m[:, t][:, None, None]
            arg = cand.argmax(1)
            back[t] = arg
            score = np.take_along_axis(cand, arg[:, None, :], 1)[:, 0, :] + lp[t]
        path = np.zeros((c, b - a), np.int8)
        path[:, -1] = score.argmax(1)
        for t in range(b - a - 1, 0, -1):
            path[:, t - 1] = back[t, rows, path[:, t]]
        out[:, a:b] = path
    return out


def decode(P, per, lam, p_border, config):
    """(N,) int8 path (0 = H, 1 = A, 2 = D) from the T1w probabilities P (N, 3), the lambda (lam_ha, lam_d) of T1w and the detector probability p_border (N,)."""
    if not (len(P) == len(per) == len(p_border)):
        raise ValueError(f"P has {len(P)} rows, per {len(per)} and p_border {len(p_border)}")
    logp = np.log(np.clip(P, 1e-9, 1.0))
    return viterbi_variable(logp, per, lam[0], lam[1], multipliers(p_border, [config], per))[0]


def model_text(path):
    """Text of a LightGBM model file with line endings normalised: LightGBM cannot read a model file with CRLF, which is what a Windows checkout
    with `core.autocrlf` produces, so the weights are always loaded from this text."""
    return open(path, "rb").read().replace(b"\r\n", b"\n").decode("utf-8")


def sha256_text(path):
    """SHA-256 of a text file with line endings normalised, so a checkout with CRLF line endings gives the same digest."""
    return hashlib.sha256(model_text(path).encode("utf-8")).hexdigest()


def load_detector(directory=MODELS_DIR):
    """(lightgbm Booster, meta dict) of the committed detector. Raises ValueError if the weights do not match the SHA-256 recorded next to them or
    were trained with another peak window or target tolerance than the ones this module uses."""
    import lightgbm as lgb
    meta = json.load(open(os.path.join(directory, META_FILE)))
    path = os.path.join(directory, meta["model"])
    if sha256_text(path) != meta["model_sha256"]:
        raise ValueError(f"{path} does not match the SHA-256 in {META_FILE}")
    for key, current in (("peak_window", PEAK_WINDOW), ("target_tolerance", TARGET_TOL)):
        if meta[key] != current:
            raise ValueError(f"the weights were trained with {key} = {meta[key]} but borde.py uses {current}")
    return lgb.Booster(model_str=model_text(path)), meta
