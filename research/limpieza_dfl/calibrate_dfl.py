"""Calibration of the player-cleaning constants of src/limpieza.py on DFL tracking (CPU).

    python calibrate_dfl.py      # -> stats_dfl.json, decision_dfl.json

Preregistered (2026-11-12). Clean reference: DFL / Sportec open tracking (CC BY 4.0, native 25 fps), the 5 development
matches (dfl.RESERVADOS excluded), native metres, every frame, per period. In the submission limpieza cleans the player
slots and the challenge tracking ball; this script calibrates the player constants:
  * V_PLAYER (m/s): physical sprint ceiling 12 m/s; kept if the DFL 99.99th percentile of player step speed is below
    it, otherwise raised to that percentile rounded up to 0.5 m/s.
  * TOL_PLAYER (m): admissible when TOL >= 2 x the 99.9th percentile of DFL position noise (residual to the centred
    5-frame mean) and the jump rule dist > V_PLAYER * dt + TOL flags at most 1e-5 of the DFL links. Grid 0.05 ... 0.50.
  * DUP_M (m): admissible when frames with some same-team pair closer than DUP_M are at most 1 % of the DFL frames.
  * SPIKE_MAX_PLAYER (12 frames) is a design constant; HANDOVER_M = V_PLAYER * SPIKE_MAX_PLAYER / FPS rounded to the
    metre; K_MAX = 11 is the law of the game.
Reference values are kept when admissible. Nothing from BAR-ATH is read (training read guard on).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bancos"))


class c:  # noqa: N801  (constants of the calibration)
    FPS = 25
    V_PHYSICAL = 12.0
    TOL_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50)
    DUP_GRID = (0.10, 0.15, 0.20, 0.25, 0.30, 0.50)
    INCUMBENT = {"V_PLAYER": 12.0, "TOL_PLAYER": 0.15, "DUP_M": 0.25, "SPIKE_MAX_PLAYER": 12, "HANDOVER_M": 6.0}

    @staticmethod
    def guard() -> None:
        import onside_paths
        from training_data_guard import restrict_training_reads
        restrict_training_reads((Path(onside_paths.CHALLENGE_DIR) / "data", Path(onside_paths.EVALUATION_DIR) / "gt",
                                 Path(onside_paths.OUT_DIR) / "balon_yolo"))


def dfl_stats(mid: str) -> dict:
    import dfl
    df = dfl.load_match(mid)
    frame = df["frame"].values.astype(np.int64)
    speeds, noise, jump = [], [], []
    close = {g: np.zeros(len(df), bool) for g in c.DUP_GRID}
    for side in ("h", "a"):
        cols = sorted({k[:-2] for k in df.columns if k[0] == side and k[1:2].isdigit()})
        s = np.stack([np.stack([df[k + "_x"].values.astype(float), df[k + "_y"].values.astype(float)], -1)
                      for k in cols], 1)
        for j in range(s.shape[1]):
            p = s[:, j]
            ok = np.isfinite(p[:, 0])
            link = ok[:-1] & ok[1:] & (np.diff(frame) == 1)
            step = np.linalg.norm(np.diff(p, axis=0), axis=1)
            speeds.append(step[link] * c.FPS)
            jump.append(step[link])
            win = np.ones(len(p), bool)
            for o in range(-2, 3):
                win &= np.roll(ok, o) & (np.roll(frame, o) == frame - o)
            win[:2] = False
            win[-2:] = False
            mean = sum(np.roll(p, o, axis=0) for o in range(-2, 3)) / 5
            noise.append(np.linalg.norm(p - mean, axis=1)[win])
        for i in range(s.shape[1]):
            for j in range(i + 1, s.shape[1]):
                dist = np.nan_to_num(np.linalg.norm(s[:, i] - s[:, j], axis=1), nan=np.inf)
                for g in c.DUP_GRID:
                    close[g] |= dist < g
    return {"mid": mid, "frames": int(len(df)), "speeds": np.concatenate(speeds), "noise": np.concatenate(noise),
            "jump": np.concatenate(jump), "dup_frames": {g: int(close[g].sum()) for g in c.DUP_GRID}}


def main() -> None:
    c.guard()
    import dfl
    mids = [m for m in dfl.MATCHES if m not in dfl.RESERVADOS]
    rows = [dfl_stats(m) for m in mids]
    speeds = np.concatenate([r["speeds"] for r in rows])
    noise = np.concatenate([r["noise"] for r in rows])
    jump = np.concatenate([r["jump"] for r in rows])
    frames = sum(r["frames"] for r in rows)
    p_speed = float(np.percentile(speeds, 99.99))
    p_noise = float(np.percentile(noise, 99.9))
    v_player = c.V_PHYSICAL if p_speed < c.V_PHYSICAL else math.ceil(p_speed * 2) / 2
    fp = {t: float(np.mean(jump > v_player / c.FPS + t)) for t in c.TOL_GRID}
    tol_ok = [t for t in c.TOL_GRID if t >= 2 * p_noise and fp[t] <= 1e-5]
    tol = c.INCUMBENT["TOL_PLAYER"] if c.INCUMBENT["TOL_PLAYER"] in tol_ok else (min(tol_ok) if tol_ok else None)
    dup_rate = {g: sum(r["dup_frames"][g] for r in rows) / frames for g in c.DUP_GRID}
    dup_ok = [g for g in c.DUP_GRID if dup_rate[g] <= 0.01]
    dup = c.INCUMBENT["DUP_M"] if c.INCUMBENT["DUP_M"] in dup_ok else max(
        (g for g in dup_ok if g < c.INCUMBENT["DUP_M"]), default=None)
    handover = float(round(v_player * c.INCUMBENT["SPIKE_MAX_PLAYER"] / c.FPS))
    stats = {"matches": mids, "frames": frames, "links": int(len(jump)), "speed_p99_99": p_speed,
             "speed_p99_9": float(np.percentile(speeds, 99.9)), "speed_max": float(speeds.max()),
             "noise_p99_9": p_noise, "noise_p50": float(np.median(noise)),
             "jump_rule_false_positive_rate": fp, "dup_frame_rate": dup_rate}
    chosen = {"V_PLAYER": v_player, "TOL_PLAYER": tol, "DUP_M": dup,
              "SPIKE_MAX_PLAYER": c.INCUMBENT["SPIKE_MAX_PLAYER"], "HANDOVER_M": handover}
    decision = {"incumbent": c.INCUMBENT, "chosen": chosen, "unchanged": chosen == c.INCUMBENT}
    (HERE / "stats_dfl.json").write_text(json.dumps(stats, indent=1))
    (HERE / "decision_dfl.json").write_text(json.dumps(decision, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != "matches"} | decision, indent=1))


if __name__ == "__main__":
    main()
