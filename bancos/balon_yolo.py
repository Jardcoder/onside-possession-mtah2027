"""Ball track from detector candidates (npz arrays: frame, cand, model_sha256, video_sha256) and the per-frame homography.

    .venv/bin/python bancos/balon_yolo.py            # writes ONSIDE_OUT/balon_yolo/track.npz and prints a comparison with the ATD ball

Steps (pure functions, method side: no ground truth is read):
  1. every candidate (pixel centre) goes to normalised pitch coordinates through the inverse of the per-frame homography (the CSV maps
     pitch 0-1 -> pixels). A ball in the air is projected to where its image meets the ground, so its position is biased away from
     the camera; nothing corrects that here.
  2. candidates outside the pitch plus MARGIN_M are dropped.
  3. tracklets: each candidate extends the active tracklet whose last point is nearest within GATE_M_PER_FRAME * gap (gap <= MAX_GAP
     frames), otherwise it starts a new one.
  4. a tracklet is kept if it lasts MIN_LEN frames, its mean confidence is at least MIN_MEAN_CONF and it is not an image-fixed object
     (pixel spread below STATIC_PX while its pitch spread exceeds STATIC_PITCH_M: something stuck to the image while the camera pans).
  5. per frame the kept tracklet with the largest summed confidence wins; holes up to BRIDGE frames inside a period are interpolated.
Output: `ball_xy` (n_video, 2) normalised pitch coordinates, NaN where no ball, `conf` (n_video,), plus the parameters.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import onside_paths  # noqa: E402
onside_paths.add_challenge_to_path()
import atd  # noqa: E402

OUT = os.path.join(onside_paths.OUT_DIR, "balon_yolo")
PARTS = ("p1.npz", "p2.npz")
HOMOGRAPHY = os.path.join(onside_paths.CHALLENGE_DIR, "data", "tactical", "BAR-ATH_tactical_homography.csv")
MARGIN_M = 3.0
GATE_M_PER_FRAME = 1.4                    # 35 m/s at 25 fps
GATE_MIN_M = 1.0
MAX_GAP = 5
MIN_LEN = 8
MIN_MEAN_CONF = 0.25
STATIC_PX = 4.0
STATIC_PITCH_M = 2.0
BRIDGE = 5
PARAMS = {k: globals()[k] for k in ("MARGIN_M", "GATE_M_PER_FRAME", "GATE_MIN_M", "MAX_GAP", "MIN_LEN", "MIN_MEAN_CONF",
                                     "STATIC_PX", "STATIC_PITCH_M", "BRIDGE")}


def load_homographies(path: str) -> np.ndarray:
    """(n_video, 3, 3) pitch 0-1 -> pixel matrices; frame column must be 0..n-1."""
    h = np.loadtxt(path, delimiter=",", skiprows=1)
    if not (h[:, 0] == np.arange(len(h))).all():
        raise ValueError(f"{path}: frame column is not 0..n-1")
    return h[:, 1:10].reshape(-1, 3, 3)


def to_pitch(cand: np.ndarray, H: np.ndarray) -> np.ndarray:
    """(F, K, 2) normalised pitch coordinates of pixel candidates (F, K, >=2) through the inverse of H (F, 3, 3)."""
    px = np.concatenate([cand[..., :2], np.ones(cand.shape[:2] + (1,))], -1)
    q = np.einsum("fij,fkj->fki", np.linalg.inv(H), px)
    return q[..., :2] / q[..., 2:3]


def inside(xy: np.ndarray, pitch: tuple, margin_m: float) -> np.ndarray:
    m = np.array([margin_m / pitch[0], margin_m / pitch[1]])
    return np.all((xy >= -m) & (xy <= 1 + m), -1)


def build_tracklets(frames: np.ndarray, xy_m: np.ndarray, valid: np.ndarray, p: dict) -> np.ndarray:
    """(F, K) tracklet id of every candidate (-1 for invalid ones). Greedy: candidates by confidence order within a frame extend the
    nearest active tracklet (last point within max(GATE_MIN_M, GATE_M_PER_FRAME * gap), gap <= MAX_GAP and a different frame)."""
    F, K = valid.shape
    tid = np.full((F, K), -1, np.int64)
    last_f, last_xy = [], []
    active = []
    for i in range(F):
        f = frames[i]
        active = [t for t in active if f - last_f[t] <= p["MAX_GAP"]]
        used = set()
        for k in range(K):
            if not valid[i, k]:
                continue
            best, bd = -1, np.inf
            for t in active:
                if t in used:
                    continue
                gap = f - last_f[t]
                d = float(np.hypot(*(xy_m[i, k] - last_xy[t])))
                if gap >= 1 and d <= max(p["GATE_MIN_M"], p["GATE_M_PER_FRAME"] * gap) and d < bd:
                    best, bd = t, d
            if best < 0:
                best = len(last_f)
                last_f.append(f)
                last_xy.append(xy_m[i, k])
                active.append(best)
            else:
                last_f[best], last_xy[best] = f, xy_m[i, k]
            used.add(best)
            tid[i, k] = best
    return tid


def tracklet_table(tid: np.ndarray, conf: np.ndarray, px: np.ndarray, xy_m: np.ndarray, frames: np.ndarray) -> dict:
    """Per tracklet: n, first / last frame, mean and summed confidence, pixel and pitch spread (max of the x / y standard deviations)."""
    ok = tid >= 0
    t, c = tid[ok], conf[ok]
    n_t = int(t.max()) + 1 if len(t) else 0
    n = np.bincount(t, minlength=n_t)
    s = np.bincount(t, c, minlength=n_t)

    def spread(v):
        m1 = [np.bincount(t, v[:, j], minlength=n_t) / np.maximum(n, 1) for j in (0, 1)]
        m2 = [np.bincount(t, v[:, j] ** 2, minlength=n_t) / np.maximum(n, 1) for j in (0, 1)]
        return np.maximum(*[np.sqrt(np.maximum(m2[j] - m1[j] ** 2, 0)) for j in (0, 1)])
    return {"n": n, "sum_conf": s, "mean_conf": s / np.maximum(n, 1), "px_spread": spread(px[ok]), "pitch_spread": spread(xy_m[ok])}


def keep_mask(tab: dict, p: dict) -> np.ndarray:
    static = (tab["px_spread"] < p["STATIC_PX"]) & (tab["pitch_spread"] > p["STATIC_PITCH_M"])
    return (tab["n"] >= p["MIN_LEN"]) & (tab["mean_conf"] >= p["MIN_MEAN_CONF"]) & ~static


def choose(tid: np.ndarray, conf: np.ndarray, xy: np.ndarray, tab: dict, keep: np.ndarray) -> tuple:
    """(F, 2) position and (F,) confidence of the winning kept tracklet per frame (NaN / 0 where none)."""
    F, K = tid.shape
    if not len(keep):
        return np.full((F, 2), np.nan), np.zeros(F)
    score = np.where(tid >= 0, np.where(keep[np.maximum(tid, 0)], tab["sum_conf"][np.maximum(tid, 0)], -1.0), -1.0)
    k = np.argmax(score, 1)
    rows = np.arange(F)
    win = score[rows, k] > 0
    out = np.full((F, 2), np.nan)
    out[win] = xy[rows[win], k[win]]
    return out, np.where(win, conf[rows, k], 0.0)


def bridge(xy: np.ndarray, max_gap: int) -> np.ndarray:
    """Linear interpolation over NaN runs of at most max_gap frames between two known positions."""
    out = xy.copy()
    known = np.flatnonzero(np.isfinite(xy[:, 0]))
    for a, b in zip(known[:-1], known[1:]):
        if 1 < b - a <= max_gap + 1:
            w = (np.arange(a + 1, b) - a) / (b - a)
            out[a + 1:b] = xy[a] + (xy[b] - xy[a]) * w[:, None]
    return out


def track_part(z: dict, H: np.ndarray, pitch: tuple, p: dict) -> tuple:
    frames, cand = z["frame"], z["cand"]
    xy = to_pitch(cand, H[frames])
    conf = np.nan_to_num(cand[..., 2], nan=0.0)
    valid = np.isfinite(cand[..., 2]) & inside(xy, pitch, p["MARGIN_M"])
    xy_m = xy * np.array(pitch)
    tid = build_tracklets(frames, xy_m, valid, p)
    tab = tracklet_table(tid, conf, cand[..., :2], xy_m, frames)
    keep = keep_mask(tab, p)
    pos, c = choose(tid, conf, xy, tab, keep)
    return frames, bridge(pos, p["BRIDGE"]), c, {"tracklets": int(len(keep)), "kept": int(keep.sum())}


def track_video(p: dict) -> tuple:
    """(ball_xy (n_video, 2), conf (n_video,), stats) of the whole tactical video with parameters p (keys of PARAMS)."""
    H = load_homographies(HOMOGRAPHY)
    ball = np.full((len(H), 2), np.nan)
    conf = np.zeros(len(H))
    stats = {}
    for part in PARTS:
        z = dict(np.load(os.path.join(OUT, part)))
        f, pos, c, st = track_part(z, H, atd.PITCH, p)
        ball[f], conf[f] = pos, c
        stats[part] = st
    return ball, conf, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--selection", type=str, required=True)
    args = parser.parse_args()
    with open(args.selection) as stream:
        selection = json.load(stream)
    if selection.get("source") != "PFF development videos only":
        raise ValueError("Challenge tracking requires a frozen PFF-only tracker selection")
    params = selection["tracker"]["params"]
    for part in PARTS:
        with np.load(os.path.join(OUT, part), allow_pickle=False) as source:
            if str(source["model_sha256"]) != selection["detector_sha256"]:
                raise ValueError("Challenge candidates and selected external detector differ")
    ball, conf, stats = track_video(params)
    tmp = os.path.join(OUT, "track.tmp.npz")
    np.savez_compressed(tmp, ball_xy=ball, conf=conf, params=json.dumps(params), stats=json.dumps(stats),
                        selection_sha256=hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest())
    os.replace(tmp, os.path.join(OUT, "track.npz"))
    print(json.dumps({"stage": "track_written", "frames": len(ball), "selection_source": selection["source"]}))


if __name__ == "__main__":
    main()
