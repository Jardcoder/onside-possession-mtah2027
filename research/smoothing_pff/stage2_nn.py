"""Stage 2a of the PFF-only smoothing selection: 5-seed short8 GRU on the chosen smoothing pair (GPU).

    SMOOTH_PAIR=g8,g10 python stage2_nn.py --device 0 --processes 6      # -> stage2/<seed>/, stage2/probs/*.npz

Stage 1 chose (g8, g10) by its preregistered rule (decision.json). Same recipe as ../promoted5_short8_20261111 (fit 28,
early stopping inner 4, 5 seeds, cfg unchanged); only the ball smoothing differs. The first seed reuses the stage-1
model (same cfg, same data). Saved probabilities (5-seed mean), for the later blend selection and outer report:
  inner_std / outer_std  : training-style balls (PFF ball smoothed, C00_release holes), as the existing protocol;
  inner_yolo / outer_yolo: the stage-1 evaluation balls (PFF + fj jumps + AR jitter -> holes -> smoothing).
Nothing from BAR-ATH is read (training read guard on).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import select_smoothing as ss  # noqa: E402

pr, ap, pj = ss.pr, ss.ap, ss.pj


def main() -> None:
    from torch_reproducibility import ensure_reproducible_process
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", choices=("cpu", "0"), required=True)
    parser.add_argument("--processes", type=int, required=True)
    args = parser.parse_args()
    decision = json.loads((HERE / "decision.json").read_text())
    pair = os.environ.get("SMOOTH_PAIR")
    if pair != decision["chosen"]:
        raise ValueError("SMOOTH_PAIR must equal the stage-1 decision")
    ensure_reproducible_process()
    ap.guard_training()
    short8 = json.loads(ss.SHORT8.read_text())
    mids = short8["fit_pff"] + short8["inner"] + short8["outer"]
    started = time.monotonic()
    built = ss.pool(ss.build_train, mids, args.processes)
    evaluation = ss.pool(ss.build_eval, short8["inner"] + short8["outer"], args.processes)
    print(json.dumps({"stage": "built", "seconds": round(time.monotonic() - started)}), flush=True)
    device = ap.device_of(args.device)
    sets = {"inner_std": (short8["inner"], built), "outer_std": (short8["outer"], built),
            "inner_yolo": (short8["inner"], evaluation), "outer_yolo": (short8["outer"], evaluation)}
    entries = {k: pr.entries(m, b, short8["cfg"]) for k, (m, b) in sets.items()}
    probs = {k: [] for k in sets}
    for seed in short8["seeds"]:
        cfg = {**short8["cfg"], "seed": seed}
        directory = HERE / pair.replace(",", "_") / "model" if seed == ss.SEED else HERE / "stage2" / str(seed)
        ap.fit_one(directory, built, short8["fit_pff"], short8["inner"], cfg, device)
        net = pr.Net(entries["inner_std"][0]["X"].shape[1], cfg).to(device)
        net.load_state_dict(ap.torch.load(directory / "weights.pt", map_location=device, weights_only=True))
        for k, es in entries.items():
            probs[k].append([pr.predict_tta(net, e["X"], e["Xs"], e["per"], cfg, device)[:, :3] for e in es])
        del net
        print(json.dumps({"stage": "trained", "seed": seed, "seconds": round(time.monotonic() - started)}), flush=True)
    for k, es in entries.items():
        avg = [np.mean([p[i] for p in probs[k]], axis=0) for i in range(len(es))]
        pj.save_probs(HERE / "stage2/probs" / k, es, avg)
    ap.save_json(HERE / "stage2/done.json", {"pair": pair, "views": dict(pr.VIEWS), "seeds": short8["seeds"],
                                             "seconds": round(time.monotonic() - started)})
    print(json.dumps({"stage": "done"}), flush=True)


if __name__ == "__main__":
    main()
