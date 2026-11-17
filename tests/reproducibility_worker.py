"""Exercise real A30 optimization and CSV prediction in an independent process."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for folder in ("src", "bancos"):
    sys.path.insert(0, str(ROOT / folder))
import posesion_rnn as pr
import predecir_reto_m1 as submission
from torch_reproducibility import ensure_reproducible_process, runtime_description, tensor_sha256


def main() -> None:
    ensure_reproducible_process()
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("device")
    parser.add_argument("seed", type=int)
    args = parser.parse_args()
    args.output.mkdir()
    device = torch.device(args.device)
    pr.limit_gpu(device)
    rng = np.random.default_rng(14)
    n = 160
    X = rng.normal(size=(n, 5)).astype(np.float32)
    y = rng.integers(0, 3, size=(n, 2), dtype=np.int64)
    entry = {"mid": "synthetic", "view": "test", "X": X, "Xs": -X, "y": y,
             "ys": np.stack([pr.swap_codes(y[:, 0]), pr.swap_touch(y[:, 1])], axis=1),
             "w": np.ones(n, np.float32), "spans": [(0, 80), (80, 160)]}
    cfg = {"seed": args.seed, "hidden": 8, "layers": 2, "dropout": 0.3, "stride": 4,
           "chunk": 40, "batch": 2, "steps": 3, "epochs": 3, "patience": 3,
           "lr": 0.001, "weight_decay": 0.0001, "touch_weight": 5.0,
           "win": 40, "margin": 8, "infer_batch": 2, "maximum_training_seconds": 60}
    model, history = pr.train([entry], [entry], cfg, device, print)
    checkpoint = args.output / "weights.pt"
    torch.save(model.state_dict(), checkpoint)
    period = np.repeat([1, 2], 80)
    probability = pr.predict_tta(model, X, -X, period, cfg, device)
    codes = pr.decode(probability[:, :3], period, (4.0, 16.0))
    csv = args.output / "prediction.csv"
    submission.write_csv(str(csv), np.asarray(pr.mo.STATES)[codes])
    result = {"tensor_sha256": tensor_sha256(model.state_dict()),
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "probability_sha256": hashlib.sha256(probability.tobytes()).hexdigest(),
              "csv_sha256": hashlib.sha256(csv.read_bytes()).hexdigest(),
              "history": history, "runtime": runtime_description(args.device)}
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
