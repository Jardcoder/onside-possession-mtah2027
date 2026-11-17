"""Select tracking and estimate ball gaps on reviewed PFF videos only.

The source directory contains <clip>.npz candidate arrays and <clip>_H.jsonl
automatic homographies. Images, video, tracking and frame labels stay local.
The published calibration contains aggregates and input hashes only.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import balon_yolo as by
import dfl_degradado as dd
import pff_split


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_clip(root: Path, reviews: Path, clip: dict[str, Any], detector_sha256: str,
              pitch_sha256: str) -> dict[str, Any]:
    key = clip["id"]
    if clip["match_id"] not in pff_split.DESARROLLO:
        raise ValueError(f"External calibration clip is not a PFF development match: {key}")
    candidate_path = root / f"{key}.npz"
    homography_path = root / f"{key}_H.jsonl"
    review_path = reviews / f"{key}.json"
    with np.load(candidate_path, allow_pickle=False) as archive:
        if "model_sha256" not in archive or str(archive["model_sha256"]) != detector_sha256:
            raise ValueError(f"Candidate detector provenance mismatch: {key}")
        candidates = {"frame": archive["frame"], "cand": archive["cand"]}
        video_sha256 = str(archive["video_sha256"])
    frames = candidates["frame"]
    if not np.array_equal(frames, np.arange(len(frames))):
        raise ValueError(f"External candidates must cover a complete contiguous clip: {key}")
    provenance_path = root / f"{key}_H.jsonl.provenance.json"
    provenance = json.loads(provenance_path.read_text())
    if (provenance["model_sha256"] != pitch_sha256 or provenance["video_sha256"] != video_sha256
            or provenance["homography_sha256"] != sha256(homography_path) or provenance["frames"] != len(frames)
            or abs(provenance["fps"] - clip["fps"]) > 1e-6):
        raise ValueError(f"External homography provenance mismatch: {key}")
    homography = np.full((len(frames), 3, 3), np.nan)
    with homography_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            frame = int(row["frame_index"])
            if 0 <= frame < len(frames):
                homography[frame] = np.asarray(row["homography"], float) @ np.diag([*clip["pitch"], 1.0])
    finite = np.isfinite(homography).all(axis=(1, 2))
    covered = finite & (np.abs(np.linalg.det(np.nan_to_num(homography))) > 1e-12)
    # NaN matrices invert to NaNs; singular finite matrices must also be excluded.
    homography[~covered] = np.nan
    if not covered.any():
        raise ValueError(f"No usable external homographies: {key}")
    return {"clip": clip, "candidates": candidates, "homography": homography,
            "covered": covered, "reviews": json.loads(review_path.read_text()),
            "hashes": {"candidates": sha256(candidate_path), "homography": sha256(homography_path),
                       "homography_provenance": sha256(provenance_path),
                       "reviews": sha256(review_path), "video": video_sha256},
            "homography_provenance": provenance}


def rescale_tracker(params: dict[str, float], fps: float) -> dict[str, float]:
    result = dict(params)
    result["GATE_M_PER_FRAME"] *= 25.0 / fps
    for key in ("MAX_GAP", "MIN_LEN", "BRIDGE"):
        result[key] = round(params[key] * fps / 25.0)
    return result


def external_track(source: dict[str, Any], params: dict[str, float]) -> np.ndarray:
    frame, positions, _, _ = by.track_part(source["candidates"], source["homography"],
                                          tuple(source["clip"]["pitch"]),
                                          rescale_tracker(params, source["clip"]["fps"]))
    ball = np.full((len(source["covered"]), 2), np.nan)
    ball[frame] = positions
    ball[~source["covered"]] = np.nan
    return ball


def point_f1(ball: np.ndarray, source: dict[str, Any]) -> dict[str, float | int]:
    """Score centers inside reviewed boxes; a wrong positive is both FP and FN."""
    tp, fp, fn, negatives, positives = 0, 0, 0, 0, 0
    for key, review in source["reviews"].items():
        frame = int(key)
        if frame >= len(ball) or not source["covered"][frame]:
            continue
        predicted = bool(np.isfinite(ball[frame]).all())
        if review["status"] == "absent":
            negatives += 1
            fp += int(predicted)
        elif review["status"] == "accepted":
            positives += 1
            inside = False
            if predicted:
                q = source["homography"][frame] @ np.r_[ball[frame], 1.0]
                x, y = q[:2] / q[2]
                x0, y0, x1, y1 = review["bbox"]
                inside = x0 <= x <= x1 and y0 <= y <= y1
            tp += int(inside)
            fn += int(not inside)
            fp += int(predicted and not inside)
    if positives == 0:
        raise ValueError("Tracker selection requires reviewed positives within homography coverage")
    return {"tp": tp, "fp": fp, "fn": fn, "positives": positives, "negatives": negatives,
            "f1": 2 * tp / (2 * tp + fp + fn)}


def gap_summary(tracks: list[tuple[np.ndarray, np.ndarray]], fps: float) -> dict[str, Any]:
    lengths, absences = [], []
    for ball, coverage in tracks:
        indices = np.minimum(np.rint(np.arange(int(len(ball) * 25.0 / fps)) * fps / 25.0).astype(int), len(ball) - 1)
        covered = coverage[indices]
        absent = ~np.isfinite(ball[indices, 0])
        edges = np.diff(np.r_[0, covered.astype(int), 0])
        segments = [(int(a), int(b)) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
                    if b - a >= 250]
        if not segments:
            raise ValueError("External calibration needs continuous homography stretches of at least 10 seconds")
        lengths.append(dd.run_lengths(absent, segments))
        absences.append(np.concatenate([absent[a:b] for a, b in segments]))
    runs = np.concatenate(lengths)
    if not len(runs):
        raise ValueError("External calibration contains no ball gaps")
    classes = np.digitize(runs, dd.LEN_CUTS)
    low, high = [0, *dd.LEN_CUTS], [*dd.LEN_CUTS, None]
    return {"share": float(np.concatenate(absences).mean()), "fps": 25,
            "classes": [{"name": name, "n_runs": int((classes == i).sum()),
                         "lo_frames": int(low[i]), "hi_frames": int(high[i]) if high[i] is not None else None,
                         "run_hist": dd.hist_edges(runs[classes == i])}
                        for i, name in enumerate(dd.LEN_NAMES)]}


def calibrate(root: Path, reviews: Path, protocol: dict[str, Any], detector_sha256: str,
              excluded: set[str]) -> dict[str, Any]:
    clips = [c for c in protocol["external_clips"] if c["match_id"] not in excluded]
    if not clips:
        raise ValueError("No external development clips remain after match exclusions")
    sources = [read_clip(root, reviews, clip, detector_sha256, protocol["pitch_model_sha256"]) for clip in clips]
    measurements, tracks = [], []
    for candidate in protocol["tracker_candidates"]:
        balls = [external_track(s, candidate["params"]) for s in sources]
        scores = [point_f1(b, s) for b, s in zip(balls, sources)]
        measurements.append({"name": candidate["name"], "params": candidate["params"],
                             "matches": {s["clip"]["match_id"]: score for s, score in zip(sources, scores)},
                             "objective": min(float(score["f1"]) for score in scores)})
        tracks.append(balls)
    selected = max(range(len(measurements)), key=lambda i: (measurements[i]["objective"], -i))
    fps = {s["clip"]["fps"] for s in sources}
    if len(fps) != 1:
        raise ValueError("External clip frame rates must agree")
    summary = gap_summary([(ball, source["covered"]) for ball, source in zip(tracks[selected], sources)], fps.pop())
    return {"schema_version": 1, "protocol_version": protocol["version"], "source": "PFF development videos only",
            "detector_sha256": detector_sha256, "excluded_matches": sorted(excluded),
            "sources": [{"clip": s["clip"], "sha256": s["hashes"],
                         "homography_provenance": s["homography_provenance"]} for s in sources],
            "tracker_candidates": measurements, "tracker": measurements[selected],
            "anchor_probability": protocol["anchor_probability"], **summary}
