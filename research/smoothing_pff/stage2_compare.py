"""Smoothing stage 2, part b: (g8, g10) vs (g6, g8) with the submission's PFF B3 constants (CPU).

    SMOOTH_PAIR=g8,g10 python smooth_compare.py      # -> smooth_compare.json

Preregistered (2026-11-13, before any result): candidate S = (g8, g10) 5-seed GRU of smoothing_pff_20261112 + T1w
refit with (g8, g10) and the PFF B3 constants (t1w_g8_g10_b3), blend by pj.select on the inner 4 std balls. Reference
= the submission: short8 GRU + T1w refit with the PFF B3 constants, (g6, g8), alpha 0.75, lambda (8, 32). Outer 8, std
and detector-like (yolo) ball families. (g8, g10) replaces (g6, g8) only if the repo entry rule passes on the std
family (min mean delta >= +0.006, CI95 low > 0 and median > 0 in both views); yolo is reported.
"""
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
sys.path.insert(0, str(OUT / "b3_pff_20261112"))
sys.path.insert(0, str(OUT / "smoothing_pff_20261112"))
import select_smoothing as ss  # noqa: E402  (views from SMOOTH_PAIR)
import b3_patch  # noqa: E402

pr, ap, pj = ss.pr, ss.ap, ss.pj
SHORT8 = OUT / "promoted5_short8_20261111"


def t1w_probs(booster, ball, d) -> np.ndarray:
    context = pr.rp.build_method_ctx(ball, d["players_slots"], d["period"], d["pitch"])
    b3, touches, _ = pr.rp.pipeline(context)
    absorbed = pj.absorcion.absorb(b3, pr.mp_pff.TAUS[0], d["period"])
    features, names = pj.mo.build_features(ball, d["players_slots"], d["period"], d["pitch"], b3, absorbed, touches)
    return pj.mo.predict_proba(booster, features, names)
B3_NEW = "r3.5_dv2.5_w5_c1.5x12_g5"
T1W_REF = OUT / "b3_pff_20261112" / f"t1w_{B3_NEW}" / "T1w_external_fold1.txt"
T1W_S = HERE / "t1w_g8_g10_b3" / "T1w_external_fold1.txt"
SMOOTH = OUT / "smoothing_pff_20261112"
REF_BLEND = {"alpha": 0.75, "lambda": [8.0, 32.0]}


def rows_for(mid: str) -> list[dict]:
    pr.restrict_training_reads((Path(pr.onside_paths.CHALLENGE_DIR) / "data",
                                Path(pr.onside_paths.EVALUATION_DIR) / "gt",
                                Path(pr.onside_paths.OUT_DIR) / "balon_yolo"))
    b3_patch.apply(pr.rp, B3_NEW)
    d = pr.rp.load_match(mid)
    per = np.asarray(d["period"])
    std = pr.external_gap_model(str(pj.STANDARD))
    fj = json.loads(ss.FJ.read_text())
    jitter = json.loads((SMOOTH / "jitter.json").read_text())["chosen"]
    new_views = dict(pr.VIEWS)
    ref_views = {"g6h": "g6", "g8h": "g8"}
    boosters = {"REF": pr.mp_pff.lightgbm.Booster(model_file=str(T1W_REF)),
                "S": pr.mp_pff.lightgbm.Booster(model_file=str(T1W_S))}
    rows = []
    for family in ("std", "yolo"):
        for view in ("g6h", "g8h"):
            out = {"mid": mid, "view": view, "family": family, "per": per, "gt": pr.rp.ev.make_gt(d, "label")}
            for name, views in (("REF", ref_views), ("S", new_views)):
                pr.VIEWS.clear()
                pr.VIEWS.update(views)
                ball = pr.training_ball(d, mid, view, std) if family == "std" else ss.eval_ball(d, mid, view, std, fj, jitter)
                out[name] = {"X": pr.build_features(ball, d["players_slots"], per, d["pitch"]),
                             "Xs": pr.build_features(ball, pr.swapped(d["players_slots"]), per, d["pitch"]),
                             "t1w": t1w_probs(boosters[name], ball, d)}
            pr.VIEWS.clear()
            pr.VIEWS.update(new_views)
            rows.append(out)
    return rows


def nn_mean(directories, rows, key, cfg):
    """5-seed mean of the GRU probabilities (GPU when one is visible)."""
    device = ap.torch.device("cuda:0" if ap.torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and pr.GPU_BUDGET is None:
        pr.limit_gpu(device)                     # repo GPU budget (70 % duty cycle and device memory)
    out = [[] for _ in rows]
    for seed, directory in directories:
        c = {**cfg, "seed": seed}
        net = pr.Net(rows[0][key]["X"].shape[1], c).to(device)
        net.load_state_dict(ap.torch.load(directory / "weights.pt", map_location=device, weights_only=True))
        for i, r in enumerate(rows):
            out[i].append(pr.predict_tta(net, r[key]["X"], r[key]["Xs"], r["per"], c, device)[:, :3])
        del net
    return [np.mean(p, axis=0) for p in out]


def main() -> None:
    short8 = json.loads(ss.SHORT8.read_text())
    cfg = short8["cfg"]
    inner_entries, inner_nn, inner_t1 = [], [], []
    for mid in short8["inner"]:
        for view in ("g6h", "g8h"):
            z = pj.npz(SMOOTH / "stage2/probs/inner_std" / f"{mid}__{view}.npz")
            t = pj.npz(HERE / "t1w_g8_g10_b3" / "inner" / f"{mid}__{view}.npz")
            if not np.array_equal(z["gt"], t["gt"]):
                raise ValueError(f"inner ground truth differs: {mid} {view}")
            inner_entries.append({"mid": mid, "view": view, "gt": z["gt"], "per": z["period"]})
            inner_nn.append(z["probabilities"])
            inner_t1.append(t["probabilities"])
    selection = pj.select(inner_nn, inner_entries, inner_t1)
    with ProcessPoolExecutor(4, mp_context=mp.get_context("spawn")) as ex:
        rows = [r for rs in ex.map(rows_for, short8["outer"]) for r in rs]
    s_dirs = [(s, SMOOTH / "g8_g10/model" if s == ss.SEED else SMOOTH / "stage2" / str(s)) for s in short8["seeds"]]
    r_dirs = [(s, SHORT8 / "models" / str(s)) for s in short8["seeds"]]
    nn = {"S": nn_mean(s_dirs, rows, "S", cfg), "REF": nn_mean(r_dirs, rows, "REF", cfg)}
    blends = {"S": selection["blend"], "REF": REF_BLEND}
    table = []
    for i, r in enumerate(rows):
        row = {"match": r["mid"], "view": r["view"], "family": r["family"]}
        for name in ("REF", "S"):
            b = blends[name]
            p = b["alpha"] * nn[name][i] + (1 - b["alpha"]) * r[name]["t1w"]
            row[name] = pr.spell_f1(r["gt"], pr.decode(p, r["per"], tuple(b["lambda"])))
        table.append(row)
    summary = {}
    for family in ("std", "yolo"):
        for view in ("g6h", "g8h"):
            vr = [t for t in table if t["family"] == family and t["view"] == view]
            summary[f"{family}/{view}"] = {"REF": float(np.mean([t["REF"] for t in vr])),
                                           "S": float(np.mean([t["S"] for t in vr])),
                                           "delta": pr.rp.ep.delta_stats([t["S"] - t["REF"] for t in vr])}
    sd = [summary[f"std/{v}"]["delta"] for v in ("g6h", "g8h")]
    replace = min(d["mean"] for d in sd) >= 0.006 and all(d["lo"] > 0 and d["median"] > 0 for d in sd)
    (HERE / "smooth_compare.json").write_text(json.dumps({"selection": selection["blend"], "summary": summary,
                                                          "replace_g6_g8": replace, "rows": table}, indent=1))
    print(json.dumps({k: (round(v["REF"], 4), round(v["S"], 4), round(v["delta"]["mean"], 4), v["delta"]["wins"],
                          v["delta"]["losses"], round(v["delta"]["lo"], 4), round(v["delta"]["hi"], 4))
                      for k, v in summary.items()} | {"blend": selection["blend"], "replace_g6_g8": replace}, indent=1))


if __name__ == "__main__":
    main()
