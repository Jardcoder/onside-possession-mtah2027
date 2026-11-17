"""Adoption of short8 (CPU): short8 vs promoted5, both blended with the T1w_external refit with the
PFF B3 constants. PFF data only.

    python short8_clean.py --processes 2      # -> short8_clean.json

Preregistered (2026-11-13, before any result): for each GRU (promoted5 frozen, short8 5 seeds) the blend with the new
T1w is chosen by pj.select on the inner 4 std balls; outer 8 paired short8 - promoted5 per view. short8 stays adopted
if the repo entry rule passes (min mean >= +0.006, CI95 low > 0 and median > 0 on both views); otherwise the report
says so and the user decides.
"""
import argparse
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
sys.path.insert(0, str(OUT / "promoted5_jumps_20261111"))
sys.path.insert(0, str(OUT / "b3_pff_20261112"))
import promoted5_jumps as pj  # noqa: E402
import b3_patch  # noqa: E402

pr, ap = pj.pr, pj.ap


class cr:  # noqa: N801  (T1w_external probabilities with the PFF B3 constants)
    B3_NEW = "r3.5_dv2.5_w5_c1.5x12_g5"

    @staticmethod
    def t1w_job(item):
        mid, family = item
        pr.restrict_training_reads((Path(pr.onside_paths.CHALLENGE_DIR) / "data",
                                    Path(pr.onside_paths.EVALUATION_DIR) / "gt",
                                    Path(pr.onside_paths.OUT_DIR) / "balon_yolo"))
        b3_patch.apply(pr.rp, cr.B3_NEW)
        d = pr.rp.load_match(mid)
        std = pr.external_gap_model(str(pj.STANDARD))
        booster = pr.mp_pff.lightgbm.Booster(model_file=str(OUT / "b3_pff_20261112" / f"t1w_{cr.B3_NEW}" / "T1w_external_fold1.txt"))
        out = {}
        for view in ("g6h", "g8h"):
            ball = pr.training_ball(d, mid, view, std)
            ctx = pr.rp.build_method_ctx(ball, d["players_slots"], d["period"], d["pitch"])
            b3, touches, _ = pr.rp.pipeline(ctx)
            absorbed = pj.absorcion.absorb(b3, pr.mp_pff.TAUS[0], d["period"])
            f, names = pj.mo.build_features(ball, d["players_slots"], d["period"], d["pitch"], b3, absorbed, touches)
            out[view] = pj.mo.predict_proba(booster, f, names).astype(np.float32)
        return mid, family, out
SHORT8 = OUT / "promoted5_short8_20261111"
T1W_INNER = OUT / "b3_pff_20261112" / f"t1w_{cr.B3_NEW}" / "inner"


def promoted_job(item):
    mid, = item
    ap.torch.set_num_threads(2)
    pr.restrict_training_reads((Path(pr.onside_paths.CHALLENGE_DIR) / "data",
                                Path(pr.onside_paths.EVALUATION_DIR) / "gt",
                                Path(pr.onside_paths.OUT_DIR) / "balon_yolo"))
    manifest = json.loads((pj.PROMOTED / "a30.json").read_text())
    d = pr.rp.load_match(mid)
    per = np.asarray(d["period"])
    std = pr.external_gap_model(str(pj.STANDARD))
    out = {}
    for view in ("g6h", "g8h"):
        ball = pr.training_ball(d, mid, view, std)
        entry = {"X": pr.build_features(ball, d["players_slots"], per, d["pitch"]),
                 "Xs": pr.build_features(ball, pr.swapped(d["players_slots"]), per, d["pitch"]), "per": per}
        out[view] = ap.model_probabilities(pj.PROMOTED, entry, manifest, ap.torch.device("cpu"))
    return mid, out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, required=True)
    args = parser.parse_args()
    protocol = json.loads((SHORT8 / "protocol.json").read_text())
    inner, outer = protocol["inner"], protocol["outer"]
    cache = HERE / "short8_clean_cache.npz"          # intermediate probabilities (per match, local only)
    if cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            prom = {m: {v: z[f"p5|{m}|{v}"] for v in ("g6h", "g8h")} for m in inner + outer}
            t1 = {m: {v: z[f"t1|{m}|{v}"] for v in ("g6h", "g8h")} for m in outer}
    else:
        with ProcessPoolExecutor(args.processes, mp_context=mp.get_context("spawn")) as ex:
            prom = dict(ex.map(promoted_job, [(m,) for m in inner + outer]))
            t1 = {m: o for m, f, o in ex.map(cr.t1w_job, [(m, "std") for m in outer])}
        np.savez(cache, **{f"p5|{m}|{v}": a for m, o in prom.items() for v, a in o.items()},
                 **{f"t1|{m}|{v}": a for m, o in t1.items() for v, a in o.items()})
    def entries(mids, root):
        es, ps = [], []
        for m in mids:
            for v in ("g6h", "g8h"):
                z = pj.npz(root / f"{m}__{v}.npz")
                es.append({"mid": m, "view": v, "gt": z["gt"], "per": z["period"] if "period" in z else z["per"]})
                ps.append(z["probabilities"])
        return es, ps
    e_in, s8_in = entries(inner, SHORT8 / "inner")
    _, t_in = entries(inner, T1W_INNER)
    p5_in = [prom[e["mid"]][e["view"]] for e in e_in]
    sel = {"short8": pj.select(s8_in, e_in, t_in), "promoted5": pj.select(p5_in, e_in, t_in)}
    e_out, s8_out = entries(outer, SHORT8 / "outer")
    rows = []
    for i, e in enumerate(e_out):
        t = t1[e["mid"]][e["view"]]
        row = {"match": e["mid"], "view": e["view"]}
        for name, nn in (("short8", s8_out[i]), ("promoted5", prom[e["mid"]][e["view"]])):
            b = sel[name]["blend"]
            row[name] = pr.spell_f1(e["gt"], pr.decode(b["alpha"] * nn + (1 - b["alpha"]) * t, e["per"], tuple(b["lambda"])))
        rows.append(row)
    summary = {}
    for v in ("g6h", "g8h"):
        vr = [r for r in rows if r["view"] == v]
        summary[v] = {"promoted5": float(np.mean([r["promoted5"] for r in vr])), "short8": float(np.mean([r["short8"] for r in vr])),
                      "delta": pr.rp.ep.delta_stats([r["short8"] - r["promoted5"] for r in vr])}
    ds = [summary[v]["delta"] for v in ("g6h", "g8h")]
    passed = min(d["mean"] for d in ds) >= 0.006 and all(d["lo"] > 0 and d["median"] > 0 for d in ds)
    (HERE / "short8_clean.json").write_text(json.dumps({"selection": {k: v["blend"] for k, v in sel.items()},
                                                        "summary": summary, "entry_rule_passed": passed, "rows": rows}, indent=1))
    print(json.dumps({"selection": {k: v["blend"] for k, v in sel.items()},
                      **{v: (round(s["promoted5"], 4), round(s["short8"], 4), round(s["delta"]["mean"], 4), s["delta"]["wins"],
                             s["delta"]["losses"], round(s["delta"]["lo"], 4), round(s["delta"]["median"], 4)) for v, s in summary.items()},
                      "entry_rule_passed": passed}, indent=1))


if __name__ == "__main__":
    main()
