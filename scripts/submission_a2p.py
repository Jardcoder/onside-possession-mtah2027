"""Regenerate the submitted possession file from the challenge data bundle.

    ./run.sh submission [--output PATH]

Pipeline, every component frozen in models/submission_a2p (SHA256SUMS) and fit on external data only:

1. Ball: the ball of the challenge tracking, cleaned by rules only (limpieza.clean_atd; holes stay NaN). No video, no detector.
2. Possession: short8 A30 GRU (5 seeds, mean with swap TTA) blended with the T1w_external LightGBM booster refit with
   the PFF-selected B3 constants (t1w_external_fold1.txt, b3.json), alpha 0.75, decoded by the two-lambda Viterbi of
   bancos/posesion_rnn.py with lambda (8, 32) from selection.json. Each model sees two smoothings of the ball (g6, g8), averaged.
3. CSV written with the starter's writer and checked by the official validate_submission.py.

Outputs: the CSV at --output (default $ONSIDE_OUT/submission_a2p/submission_a2p.csv). No ground truth is read.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch  # noqa: F401  (must load before lightgbm: both ship an OpenMP runtime and torch hangs or crashes on macOS otherwise)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bancos"))

import onside_paths  # noqa: E402

ASSETS = ROOT / "models" / "submission_a2p"
WORK = Path(onside_paths.OUT_DIR) / "submission_a2p"
os.environ["ONSIDE_OUT"] = str(WORK)   # caches and workers resolve their paths from the environment
onside_paths.OUT_DIR = str(WORK)
onside_paths.add_challenge_to_path()

import absorcion  # noqa: E402
import atd  # noqa: E402
import features_reto as fr  # noqa: E402
import limpieza  # noqa: E402
import modelo as mo  # noqa: E402
import posesion_rnn as pr  # noqa: E402
import predecir_reto_m1 as submission  # noqa: E402
import reto_periodos as rpd  # noqa: E402
import torch  # noqa: E402


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_assets() -> None:
    for line in (ASSETS / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        if sha256(ASSETS / name) != digest:
            raise ValueError(f"Asset {name} does not match models/submission_a2p/SHA256SUMS")


def tracking_ball() -> np.ndarray:
    """Ball of the challenge tracking, cleaned by rules only (no interpolation: holes stay NaN), video frame index."""
    ball = limpieza.clean_atd()["ball_xy"]
    print(json.dumps({"ball_frames": int(np.isfinite(ball[:, 0]).sum()), "ball": "tracking"}), flush=True)
    return ball


def possession(ball_video: np.ndarray) -> np.ndarray:
    protocol = json.loads((ASSETS / "short8/protocol.json").read_text())
    selection = json.loads((ASSETS / "short8/selection.json").read_text())["blend"]
    booster = pr.mp_pff.lightgbm.Booster(model_file=str(ASSETS / "t1w_external_fold1.txt"))
    b3_constants = json.loads((ASSETS / "b3.json").read_text())          # PFF-selected B3 constants (feed the T1w features)
    pr.rp.V2.update({k: b3_constants[k] for k in ("r", "dv", "win", "r_ctrl", "k")})
    pr.rp.B3_C1 = (b3_constants["dv"], b3_constants["win"], b3_constants["r"])
    pr.rp.ep.CAP = b3_constants["cap"]
    d = limpieza.clean_atd()
    n = len(d["ball_xy"])
    frames = rpd.play_frames(n)
    per = rpd.periods(n)[frames].astype(np.int64)
    slots = {k: v[frames] for k, v in d["players_slots"].items()}
    device = torch.device("cpu")
    views = []
    for smoothing in pr.CHALLENGE_SMOOTH:
        ball = pr.dd.smooth_ball(ball_video[frames], per, smoothing)
        X = pr.build_features(ball, slots, per, atd.PITCH)
        Xs = pr.build_features(ball, pr.swapped(slots), per, atd.PITCH)
        probs = []
        for seed in protocol["seeds"]:
            cfg = {**protocol["cfg"], "seed": seed}
            net = pr.Net(X.shape[1], cfg).to(device)
            net.load_state_dict(torch.load(ASSETS / "short8" / str(seed) / "weights.pt", map_location=device,
                                           weights_only=True))
            probs.append(pr.predict_tta(net, X, Xs, per, cfg, device)[:, :3])
        context = pr.rp.build_method_ctx(ball, slots, per, atd.PITCH)
        b3, touches, _ = pr.rp.pipeline(context)
        absorbed = absorcion.absorb(b3, pr.mp_pff.TAUS[0], per)
        features, names = mo.build_features(ball, slots, per, atd.PITCH, b3, absorbed, touches)
        t1w = mo.predict_proba(booster, features, names)
        views.append(selection["alpha"] * np.mean(probs, axis=0) + (1 - selection["alpha"]) * t1w)
    codes = pr.decode(np.mean(views, axis=0), per, tuple(selection["lambda"]))
    return fr.to_video(np.asarray(mo.STATES)[codes], {"frames": frames, "n_video": n})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, default=WORK / "submission_a2p.csv")
    parser.add_argument("--ball", choices=("tracking",), default="tracking",
                        help="ball source; the ball of the challenge tracking is the only one in this repository")
    args = parser.parse_args()
    verify_assets()
    labels = possession(tracking_ball())
    submission.check_labels(labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.write_csv(str(args.output), labels)
    subprocess.run([sys.executable, str(Path(onside_paths.EVALUATION_DIR) / "validate_submission.py"), str(args.output)],
                   check=True)
    print(json.dumps({"csv": str(args.output), "sha256": sha256(args.output)}))


if __name__ == "__main__":
    main()
