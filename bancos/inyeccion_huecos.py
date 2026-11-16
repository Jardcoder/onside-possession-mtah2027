"""Variant `h` of the PFF bank: ball holes inside live play, calibrated on the challenge (transfer test of M1).

    ball_h, hole = inject_holes(ball, label, per, mid)        # ball (N, 2) already degraded (g6 / g8), NaN = missing

In PFF 89 % of the frames without ball are dead ball (bancos/pff.py), in the challenge the ball is missing 0.44 of the time and with live
play far more often. This opens extra holes (ball set to NaN) on frames the GT calls alive (`label` != "D") to imitate that. The GT only
decides WHERE a hole may open; it never enters a feature (the returned ball is the only thing a method sees).

Calibration, fixed before any model is measured (see the report src/out/modelo_transfer_informe.txt):
  * lengths: the run-length histograms of the absent ball of the challenge (bancos/data/cache/huecos_largo_atd.json, aggregates:
    classes < 5, 5-10, 10-20, >= 20 s, each with its histogram over dd.EDGES). A hole draws a class with probability proportional to
    the number of runs of the class, then a length from the class histogram (uniform inside the bin, clipped to the class bounds),
    as dd._len_gap_model does but WITHOUT the (1 - P(dead | absent)) weight: that weight is the share of in-play gaps under the
    old calibration, here the point is to see what a ball that is missing like the challenge's does to the model.
  * rate: per match, holes are opened until the share of frames without ball reaches TARGET_ABSENT (the challenge, `completo`
    variant of src/out/regimen_reto.csv: 0.4396); a match that already exceeds it gets no hole. The rate of holes > 5 s per hour
    is not fitted: it follows from the two items above and is checked against the challenge (regimen_reto.csv, huecos_h).
Rules of a hole [a, a + n): all its frames are alive in the GT and have a ball (existing NaN are never touched), it stays inside one
period, and the frames just before and after it are visible and in the same period (>= 1 visible frame from any other gap, as coupled_len).
Deterministic: the generator is seeded with SHA-256(SEED_SALT + match id).
"""
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import onside_paths  # noqa: E402
import dfl_degradado as dd  # noqa: E402

SEED_SALT = "onside-huecos-h-v1"
TARGET_ABSENT = 0.43962                        # challenge `completo`, sin_balon (src/out/regimen_reto.csv)
TARGET_HUECOS_H = 64.11832                     # challenge `completo`, huecos > 5 s per hour (same file); checked, not fitted
ANCHOR_P = 0.85                               # probability that a hole is anchored to an existing gap (calibration: see the report)
MAX_TRIES = 400000
REGIMEN_CSV = os.path.join(onside_paths.OUT_DIR, "regimen_reto.csv")


def challenge_targets(path=REGIMEN_CSV):
    """(sin_balon, huecos_h) of the challenge `completo` row of the aggregates file (the constants above must agree with it)."""
    got = {}
    for line in open(path).read().splitlines()[1:]:
        var, smooth, metric, reto = line.split(",")[:4]
        if var == "completo" and smooth == "g6" and metric in ("sin_balon", "huecos_h"):
            got[metric] = float(reto)
    return got["sin_balon"], got["huecos_h"]


def length_model(largo=None):
    """draw(rng) -> hole length in frames, from the challenge gap classes (see the module docstring)."""
    cl = (largo or dd.load_calibration_largo())["classes"]
    w = np.array([c["n_runs"] for c in cl], float)
    edges = dd.EDGES

    def draw(rng):
        c = cl[rng.choice(len(cl), p=w / w.sum())]
        h = np.asarray(c["run_hist"], float)
        i = rng.choice(len(h), p=h / h.sum())
        lo, hi = edges[i], (edges[i + 1] if i + 1 < len(edges) else edges[i] * 2)
        lo, hi = max(lo, c["lo_frames"]), min(hi, c["hi_frames"] or hi)
        return int(rng.integers(lo, max(hi, lo + 1)))

    return draw


def match_rng(mid):
    return np.random.default_rng(int.from_bytes(hashlib.sha256((SEED_SALT + str(mid)).encode()).digest()[:8], "little"))


def inject_holes(ball, label, per, mid, target=TARGET_ABSENT, draw=None, anchor_p=ANCHOR_P):
    """(ball with the holes set to NaN, bool (N,) mask of the injected frames). `ball` is not modified.

    A hole is free with probability 1 - anchor_p (anywhere in live play, a visible frame on each side) or anchored with probability
    anchor_p (it starts at the live border of an existing gap, so the gap grows into live play and the two merge: the ball lost
    around a stoppage and found again later; the far end has a visible frame)."""
    ball = np.asarray(ball, float)
    per = np.asarray(per)
    alive = np.asarray(label) != "D"
    n = len(ball)
    present = np.isfinite(ball[:, 0])
    out = ball.copy()
    hole = np.zeros(n, bool)
    need = int(round(target * n)) - int((~present).sum())
    if need <= 0 or not alive.any():
        return out, hole
    draw = draw or length_model()
    rng = match_rng(mid)
    free = np.flatnonzero(alive & present)
    # live borders of the gaps that exist before any hole: (first live visible frame, +1) gap on its left, (last live visible frame, -1) gap on its right
    d = np.diff(present.astype(np.int8))
    same = per[:-1] == per[1:]
    anchors = [(int(a), 1) for a in np.flatnonzero((d == 1) & same) + 1 if alive[a]] + \
              [(int(a), -1) for a in np.flatnonzero((d == -1) & same) if alive[a]]
    cur = present.copy()                       # visible and not yet opened
    added = 0
    for _ in range(MAX_TRIES):
        if added >= need:
            break
        m = draw(rng)
        if anchors and rng.random() < anchor_p:
            a0, side = anchors[int(rng.integers(len(anchors)))]
            a, b = (a0, a0 + m) if side == 1 else (a0 - m + 1, a0 + 1)
            v0, v1 = (a, b + 1) if side == 1 else (a - 1, b)      # frames that must stay visible, same period
        else:
            a = int(rng.choice(free))
            b = a + m
            v0, v1 = a - 1, b + 1
        if v0 < 0 or v1 > n or per[v0] != per[v1 - 1]:
            continue
        if not (alive[a:b].all() and cur[v0:v1].all()):
            continue                           # live frames only, never over an existing gap or another hole
        cur[a:b] = False
        hole[a:b] = True
        added += m
    out[hole] = np.nan
    return out, hole
