"""promoted5 with stronger short-possession weights (short_ref 8, short_max 8) on the standard balls, 5 seeds (GPU).

    python promoted5_short8.py run --device 0 --processes 4
    python promoted5_short8.py report

Motivation: ../blend_diagnosis_20261111 (the blend recalls 27 % of possessions under 1 s and 61 % of 1-2 s). promoted5
trains with frame weights min(4, max(1, 4 s / spell duration)); A30 only tried that setting against uniform weights.
Preregistered (before training):
  * Same partition (fit 28, inner 4, outer 8), seeds, early stopping, C00_release balls and every cfg field of promoted5
    except short_ref = 8 and short_max = 8 (spells < 1 s weigh 8, 1-8 s weigh 8 / duration). Reserved matches unused.
  * Selection exactly as promoted5 (inner 4: standalone lambda, blend alpha / lambda with the frozen T1w inner
    probabilities).
  * Outer 8 report paired against frozen promoted5 (blend and NN), both smoothings.
  * Adoption (fixed now): the AGENTS.md entry rule on the outer 8 blend (min of the two mean deltas >= +0.006, CI95
    lower bound > 0 and median > 0 on both smoothings); a smaller positive delta is reported as such.
  * BAR-ATH public window: report only, never selection.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src/out/promoted5_jumps_20261111"))

import promoted5_jumps as pj  # noqa: E402
from torch_reproducibility import ensure_reproducible_process  # noqa: E402

pr, ap = pj.pr, pj.ap


def protocol_record() -> dict:
    base = json.loads((pj.HERE / "protocol.json").read_text())
    return {"fit_pff": base["fit_pff"], "inner": base["inner"], "outer": base["outer"], "seeds": base["seeds"],
            "cfg": {**base["cfg"], "short_ref": 8.0, "short_max": 8.0}, "standard_sha256": base["standard_sha256"],
            "frozen_promoted5": base["frozen_promoted5"], "script_sha256": ap.external.sha256(Path(__file__))}


def reference_inner_t1w(entries: list[dict]) -> list[np.ndarray]:
    manifest = json.loads((pj.REFERENCE / "reference_manifest.json").read_text())
    rows = {(r["match"], r["view"]): r for r in manifest["inner_probabilities"]}
    return [np.load(pj.REFERENCE / rows[(e["mid"], e["view"])]["file"])["probabilities"] for e in entries]


def run(device_name: str, processes: int) -> None:
    ensure_reproducible_process()
    ap.guard_training()
    protocol = protocol_record()
    path = HERE / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Preregistered protocol changed; use a new namespace")
    ap.save_json(path, protocol)
    device = ap.device_of(device_name)
    started = time.monotonic()
    built = pr.build_matches(protocol["fit_pff"] + protocol["inner"] + protocol["outer"], processes, str(pj.STANDARD))
    inner = pr.entries(protocol["inner"], built, protocol["cfg"])
    outer = pr.entries(protocol["outer"], built, protocol["cfg"])
    inner_all, outer_all = [], []
    for seed in protocol["seeds"]:
        cfg = {**protocol["cfg"], "seed": seed}
        directory = HERE / "models" / str(seed)
        print(json.dumps({"stage": "train", "seed": seed}), flush=True)
        _, probs = ap.fit_one(directory, built, protocol["fit_pff"], protocol["inner"], cfg, device)
        inner_all.append(probs)
        net = pr.Net(outer[0]["X"].shape[1], cfg).to(device)
        net.load_state_dict(ap.torch.load(directory / "weights.pt", map_location=device, weights_only=True))
        outer_all.append([pr.predict_tta(net, e["X"], e["Xs"], e["per"], cfg, device)[:, :3] for e in outer])
        del net
        print(json.dumps({"stage": "trained", "seed": seed, "seconds": round(time.monotonic() - started)}), flush=True)
    inner_avg = [np.mean([p[i] for p in inner_all], axis=0) for i in range(len(inner))]
    outer_avg = [np.mean([p[i] for p in outer_all], axis=0) for i in range(len(outer))]
    pj.save_probs(HERE / "inner", inner, inner_avg)
    pj.save_probs(HERE / "outer", outer, outer_avg)
    selection = pj.select(inner_avg, inner, reference_inner_t1w(inner))
    ap.save_json(HERE / "selection.json", {"complete": True, **selection})
    print(json.dumps({"stage": "selected", "blend": selection["blend"]}), flush=True)


def report() -> None:
    protocol = json.loads((HERE / "protocol.json").read_text())
    frozen, selection = protocol["frozen_promoted5"], json.loads((HERE / "selection.json").read_text())
    rows = []
    for mid in protocol["outer"]:
        for view in pr.VIEWS:
            old = pj.npz(pj.RUN / "outer/promoted5" / f"{mid}__{view}.npz")
            new = pj.npz(HERE / "outer" / f"{mid}__{view}.npz")
            t = pj.reference_outer(mid, view)["probabilities"]
            gt, per = old["gt"], old["period"]
            if not np.array_equal(new["gt"], gt):
                raise ValueError(f"ground truth differs: {mid} {view}")
            f1 = lambda p, lam: pr.spell_f1(gt, pr.decode(p, per, tuple(lam)))  # noqa: E731
            a0, a1 = frozen["blend"]["alpha"], selection["blend"]["alpha"]
            rows.append({"match": mid, "view": view,
                         "frozen_blend": f1(a0 * old["probabilities"] + (1 - a0) * t, frozen["blend"]["lambda"]),
                         "new_blend": f1(a1 * new["probabilities"] + (1 - a1) * t, selection["blend"]["lambda"]),
                         "frozen_nn": f1(old["probabilities"], frozen["standalone_lambda"]),
                         "new_nn": f1(new["probabilities"], selection["standalone_lambda"])})
    out = {"selection": selection, "rows": rows, "views": {}}
    for view in pr.VIEWS:
        vr = [r for r in rows if r["view"] == view]
        out["views"][view] = {k: float(np.mean([r[k] for r in vr])) for k in vr[0] if k not in ("match", "view")}
        out["views"][view]["delta_blend"] = pr.rp.ep.delta_stats([r["new_blend"] - r["frozen_blend"] for r in vr])
        out["views"][view]["delta_nn"] = pr.rp.ep.delta_stats([r["new_nn"] - r["frozen_nn"] for r in vr])
    d = [out["views"][v]["delta_blend"] for v in pr.VIEWS]
    out["entry_rule"] = min(x["mean"] for x in d) >= 0.006 and all(x["lo"] > 0 and x["median"] > 0 for x in d)
    ap.save_json(HERE / "report.json", out)
    print(json.dumps({v: {k: (round(x, 4) if isinstance(x, float) else (round(x["mean"], 4), x["wins"], x["losses"]))
                          for k, x in out["views"][v].items()} for v in pr.VIEWS} | {"entry_rule": out["entry_rule"]},
                     indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="action", required=True)
    runner = sub.add_parser("run")
    runner.add_argument("--device", choices=("cpu", "0"), required=True)
    runner.add_argument("--processes", type=int, required=True)
    sub.add_parser("report")
    args = parser.parse_args()
    if args.action == "run":
        run(args.device, args.processes)
    else:
        report()


if __name__ == "__main__":
    main()
