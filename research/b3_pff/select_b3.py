"""PFF-only recalibration of the B3 touch-detector constants that feed T1w_external (CPU).

    python select_b3.py run --processes 10      # -> rows/<mid>__<view>.json
    python select_b3.py decide                 # -> decision.json

Preregistered (2026-11-12, before any result of this study). The constants are those of COMBO in bancos/evaluar_pff.py
(radius r, speed change dv, window win, control touches r_ctrl x k, hole cap g seconds; reference value
r3.5_dv2.0_w5_c1.0x12_g5). They are selected with PFF data only:
  * Grid: exactly the original grid of bancos/evaluar_toques.py (GRID_AXES x CAPS, 432 combinations).
  * Data: PFF training32 (fit 28 + inner 4 of ../promoted5_short8_20261111/protocol.json; the outer 8 are never read),
    both training views g6h / g8h (PFF ball smoothed, C00_release holes: posesion_rnn.training_ball, PFF-only gap
    calibration). Training read guard on: nothing from BAR-ATH is read.
  * Prediction: the rule base the T1w features are built from, B3 then absorption with mp_pff.TAUS[0]
    (ronda_pff.pipeline + absorcion.absorb), with the combination's constants.
  * Objective: min over the two views of the mean per-match spellF1 against PFF `label` (methods never see it).
  * Decision: the incumbent stays unless another combination beats its objective by at least 0.003; then the highest
    objective wins. Stage 2 refits T1w_external with the chosen constants (stage2_t1w.py) and selects the blend on
    the inner 4 (select_blend.py).
"""
import argparse
import itertools
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bancos"))
sys.path.insert(0, str(OUT / "promoted5_jumps_20261111"))

GRID_AXES = {"r": (2.0, 2.5, 3.0, 3.5), "dv": (1.5, 2.0, 2.5, 3.0), "win": (5, 7, 9),
             "ctrl": ((0.0, 0), (1.0, 12), (1.5, 12))}
CAPS = (5, 10, 20)
INCUMBENT = "r3.5_dv2.0_w5_c1.0x12_g5"
MARGIN = 0.003
SHORT8 = OUT / "promoted5_short8_20261111/protocol.json"


def tag(r, dv, win, ctrl, cap):
    return f"r{r}_dv{dv}_w{win}_c{ctrl[0]}x{ctrl[1]}_g{cap}"      # same format as evaluar_toques.tag


def job(item):
    mid, view = item
    import promoted5_jumps as pj
    pr = pj.pr
    pr.restrict_training_reads((Path(pr.onside_paths.CHALLENGE_DIR) / "data",
                                Path(pr.onside_paths.EVALUATION_DIR) / "gt",
                                Path(pr.onside_paths.OUT_DIR) / "balon_yolo"))
    rp, ep, toques = pr.rp, pr.rp.ep, pr.rp.toques
    d = rp.load_match(mid)
    per = np.asarray(d["period"])
    ball = pr.training_ball(d, mid, view, pr.external_gap_model(str(pj.STANDARD)))
    gt = rp.ev.make_gt(d, "label")
    started = time.monotonic()
    base = rp.build_method_ctx(ball, d["players_slots"], per, d["pitch"])
    out = {}
    for ctrl in GRID_AXES["ctrl"]:
        ctx = dict(base)
        ctx["ctrl"] = toques.control_touches(base["near"], base["runs"], r_ctrl=ctrl[0], k=ctrl[1])
        for r, dv, win, cap in itertools.product(GRID_AXES["r"], GRID_AXES["dv"], GRID_AXES["win"], CAPS):
            rp.V2.update({"r": r, "dv": dv, "win": win, "r_ctrl": ctrl[0], "k": ctrl[1]})
            ep.CAP = cap
            pred, _, _ = rp.pipeline(ctx, c1=(dv, win, r))
            pred = pj.absorcion.absorb(pred, pr.mp_pff.TAUS[0], per)
            out[tag(r, dv, win, ctrl, cap)] = rp.pm.evaluate(gt, rp.pm.label_codes(pred), bootstrap_n=0)["spell_f1_mean"]
    (HERE / "rows" / f"{mid}__{view}.json").write_text(json.dumps(out))
    return {"mid": mid, "view": view, "seconds": round(time.monotonic() - started)}


def run(processes: int) -> None:
    protocol = json.loads(SHORT8.read_text())
    items = [(m, v) for m in protocol["fit_pff"] + protocol["inner"] for v in ("g6h", "g8h")
             if not (HERE / "rows" / f"{m}__{v}.json").exists()]
    with ProcessPoolExecutor(processes, mp_context=mp.get_context("spawn")) as ex:
        for res in ex.map(job, items):
            print(json.dumps(res), flush=True)


def decide() -> None:
    protocol = json.loads(SHORT8.read_text())
    mids = protocol["fit_pff"] + protocol["inner"]
    rows = {v: [json.loads((HERE / "rows" / f"{m}__{v}.json").read_text()) for m in mids] for v in ("g6h", "g8h")}
    tags = list(rows["g6h"][0])
    obj = {t: min(float(np.mean([r[t] for r in rows[v]])) for v in rows) for t in tags}
    if INCUMBENT not in obj:
        raise ValueError(f"incumbent tag missing: {INCUMBENT}")
    best = max(obj, key=obj.get)
    chosen = best if obj[best] >= obj[INCUMBENT] + MARGIN else INCUMBENT
    top = sorted(obj.items(), key=lambda kv: -kv[1])[:10]
    decision = {"incumbent": INCUMBENT, "incumbent_objective": obj[INCUMBENT], "best": best, "best_objective": obj[best],
                "margin": MARGIN, "chosen": chosen, "unchanged": chosen == INCUMBENT, "top10": top,
                "rank_of_incumbent": sorted(obj, key=lambda t: -obj[t]).index(INCUMBENT) + 1, "n": len(obj)}
    (HERE / "decision.json").write_text(json.dumps(decision, indent=1))
    print(json.dumps(decision, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("run").add_argument("--processes", type=int, required=True)
    sub.add_parser("decide")
    args = parser.parse_args()
    run(args.processes) if args.action == "run" else decide()


if __name__ == "__main__":
    main()
