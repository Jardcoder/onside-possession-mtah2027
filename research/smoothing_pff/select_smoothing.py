"""Stage 1 of the PFF-only smoothing selection: which pair of ball smoothings the possession GRU should use (GPU).

    python select_smoothing.py run --pair g4,g6 --device 0 --processes 6     # one candidate -> <pair>/result.json
    python select_smoothing.py decide                                         # -> decision.json

Preregistered (2026-11-12, before any result of this study; nothing from BAR-ATH is read, the training read guard is on):
  * Candidates: smoothing pairs (g2,g4), (g4,g6), (g6,g8) [incumbent],
    (g8,g10). Gaussian sigma in frames at 25 fps (bancos/dfl_degradado.BALL_SMOOTH).
  * Training, identical to short8 except the smoothing: fit 28 PFF matches, early stopping on the inner 4, cfg of
    ../promoted5_short8_20261111/protocol.json, first seed 20261101 only, PFF ball smoothed then C00_release holes.
  * Evaluation ball on the inner 4 (simulates the video ball at inference, PFF data only): PFF ball + frozen fj jumps
    (C00 on PFF clips) + AR(1) jitter of jitter.json (C00 on PFF clips) -> C00_release holes -> candidate smoothing.
  * Objective: standalone GRU, pr.choose_lambda on those inner entries (min over the two views of the mean spellF1
    against PFF `label`).
  * Decision: the incumbent (g6,g8) stays unless another pair beats its objective by at least 0.003; among those, the
    highest objective wins. If the incumbent stays, A2 is unchanged and its smoothing is PFF-confirmed. Otherwise stage 2
    (5 seeds + T1w_external on the new views, entry rule against A2 on the outer 8) is required before any change.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bancos"))
sys.path.insert(0, str(OUT / "promoted5_jumps_20261111"))
sys.path.insert(0, str(HERE))

import promoted5_jumps as pj  # noqa: E402
from jitter_calibration import SALT, ar_noise  # noqa: E402

pr, ap = pj.pr, pj.ap
PAIRS = ("g2,g4", "g4,g6", "g6,g8", "g8,g10")
INCUMBENT = "g6,g8"
MARGIN = 0.003
SEED = 20261101
SHORT8 = OUT / "promoted5_short8_20261111/protocol.json"
FJ = OUT / "ball_continuity_20261111/calibration_fj.json"


def set_pair(pair: str) -> None:
    a, b = pair.split(",")
    pr.VIEWS.clear()
    pr.VIEWS.update({"g6h": a, "g8h": b})     # keys kept so pr.entries and the decoders work unchanged


if os.environ.get("SMOOTH_PAIR"):
    set_pair(os.environ["SMOOTH_PAIR"])


def eval_ball(d: dict, mid: str, view: str, std: dict, fj: dict, jitter: dict) -> np.ndarray:
    raw = pj.inject_jumps(np.asarray(d["ball_xy"], float), mid, view, fj["jumps"], tuple(d["pitch"]))
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"{SALT}eval:{mid}:{view}".encode()).digest()[:8],
                                               "little"))
    raw = raw + ar_noise(len(raw), jitter["sigma_m"], jitter["rho"], rng) / np.array(d["pitch"], float)
    holed = pr.ih.inject_holes(raw, d["label"], d["period"], mid, target=std["share"], draw=pr.ih.length_model(std),
                               anchor_p=std["anchor_probability"])[0]
    return pr.dd.smooth_ball(holed, d["period"], pr.VIEWS[view])


def build_train(mid: str) -> dict:
    return pr.build_match(mid, str(pj.STANDARD))


def build_eval(mid: str) -> dict:
    pr.restrict_training_reads((Path(pr.onside_paths.CHALLENGE_DIR) / "data",
                                Path(pr.onside_paths.EVALUATION_DIR) / "gt",
                                Path(pr.onside_paths.OUT_DIR) / "balon_yolo"))
    d = pr.rp.load_match(mid)
    per = np.asarray(d["period"])
    std = pr.external_gap_model(str(pj.STANDARD))
    fj = json.loads(FJ.read_text())
    jitter = json.loads((HERE / "jitter.json").read_text())["chosen"]
    out = {"mid": mid, "per": per,
           "targets": {k: pr.mp_pff.label_codes(d[k]).astype(np.int64) for k in ("label", "label_poss")},
           "unc": np.asarray(d["uncertain"], bool), "gt": pr.rp.ev.make_gt(d, "label")}
    ev_f, ev_t = pr.touch_teams(d["label"], d["event_f"])
    out["touch"] = pr.touch_targets(len(per), ev_f, ev_t).astype(np.int64)
    for view in pr.VIEWS:
        ball = eval_ball(d, mid, view, std, fj, jitter)
        out[view] = {"X": pr.build_features(ball, d["players_slots"], per, d["pitch"]).astype(np.float16),
                     "Xs": pr.build_features(ball, pr.swapped(d["players_slots"]), per, d["pitch"]).astype(np.float16)}
    return out


def pool(function, mids: list, processes: int) -> dict:
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    with ProcessPoolExecutor(processes, mp_context=mp.get_context("spawn")) as ex:
        return {r["mid"]: r for r in ex.map(function, mids)}


def run(pair: str, device_name: str, processes: int) -> None:
    from torch_reproducibility import ensure_reproducible_process
    if pair not in PAIRS:
        raise ValueError(f"pair not preregistered: {pair}")
    os.environ["SMOOTH_PAIR"] = pair
    set_pair(pair)
    ensure_reproducible_process()
    ap.guard_training()
    short8 = json.loads(SHORT8.read_text())
    cfg = {**short8["cfg"], "seed": SEED}
    out = HERE / pair.replace(",", "_")
    started = time.monotonic()
    built = pool(build_train, short8["fit_pff"] + short8["inner"], processes)
    evaluation = pool(build_eval, short8["inner"], processes)
    print(json.dumps({"stage": "built", "pair": pair, "seconds": round(time.monotonic() - started)}), flush=True)
    device = ap.device_of(device_name)
    ap.fit_one(out / "model", built, short8["fit_pff"], short8["inner"], cfg, device)
    net = pr.Net(built[short8["fit_pff"][0]]["g6h"]["X"].shape[1], cfg).to(device)
    net.load_state_dict(ap.torch.load(out / "model/weights.pt", map_location=device, weights_only=True))
    entries = pr.entries(short8["inner"], evaluation, cfg)
    probs = [pr.predict_tta(net, e["X"], e["Xs"], e["per"], cfg, device)[:, :3] for e in entries]
    lam, objective = pr.choose_lambda(probs, entries)
    result = {"pair": pair, "views": dict(pr.VIEWS), "seed": SEED, "standalone_lambda": list(lam),
              "objective": float(objective), "seconds": round(time.monotonic() - started),
              "jitter": json.loads((HERE / "jitter.json").read_text())["chosen"],
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    ap.save_json(out / "result.json", result)
    print(json.dumps(result), flush=True)


def decide() -> None:
    results = {p: json.loads((HERE / p.replace(",", "_") / "result.json").read_text())["objective"] for p in PAIRS}
    incumbent = results[INCUMBENT]
    better = {p: v for p, v in results.items() if p != INCUMBENT and v >= incumbent + MARGIN}
    chosen = max(better, key=better.get) if better else INCUMBENT
    decision = {"objectives": results, "incumbent": INCUMBENT, "margin": MARGIN, "chosen": chosen,
                "a2_unchanged": chosen == INCUMBENT}
    ap.save_json(HERE / "decision.json", decision)
    print(json.dumps(decision, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="action", required=True)
    r = sub.add_parser("run")
    r.add_argument("--pair", required=True)
    r.add_argument("--device", choices=("cpu", "0"), required=True)
    r.add_argument("--processes", type=int, required=True)
    sub.add_parser("decide")
    args = parser.parse_args()
    if args.action == "run":
        run(args.pair, args.device, args.processes)
    else:
        decide()


if __name__ == "__main__":
    main()
