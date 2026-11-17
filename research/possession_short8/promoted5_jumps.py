"""Path 2: promoted5 retrained on PFF balls with injected wrong-position episodes ("jumps"), two arms.

    python promoted5_jumps.py run --arm vj --device 0 --processes 4
    python promoted5_jumps.py run --arm fj --device 0 --processes 4
    python promoted5_jumps.py report

Preregistered (protocol.json is written before any training and checked on resume):
  * Arms (calibrations from ../ball_continuity_20261111/jump_calibration.py, PFF development videos only):
      vj = gaps and jumps of the adopted Viterbi tracker (path 1); fj = frozen C00_release gaps plus the jumps of the
      frozen tracker (isolates the jumps).
  * Training ball of an arm: raw PFF ball -> jumps -> the view smoothing -> the arm's holes. Jumps: episodes start on
    frames with a ball until rate * ball frames are covered; duration and distance are drawn from the measured episodes,
    direction uniform, position = ball at the episode start + offset (a static wrong object), clipped to the pitch +-3 m.
    Seeded by SHA-256(SALT + match + view).
  * Same partition, cfg, seeds and early stopping as promoted5 (fit 28, inner 4, outer 8 PFF); reserved matches unused.
  * Selection exactly as promoted5 (seed-averaged inner probabilities, standalone lambda, blend alpha over
    {0, .25, .5, .75, 1} with T1w_external fold-1 computed on the same arm views). Only inner PFF data choose anything.
  * Outer 8, both smoothings, on two ball families: "arm" (the arm's own holes + jumps: what the arm expects at
    deployment) and "standard" (g6h / g8h of promoted5). Frozen promoted5 is scored on both with its frozen alpha /
    lambdas; on the arm family its NN is re-run and T1w recomputed on the same balls.
  * Adoption (fixed now): an arm replaces promoted5 when its paired outer blend delta is > 0 on both arm-family views and
    >= -0.005 on both standard views. If both arms qualify, vj (it matches the adopted tracker). The arm family is a
    simulation from the same model used in training, so its delta is optimistic; the standard family guards regressions.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bancos"))
sys.path.insert(0, str(HERE))

import a30_protocol as ap  # noqa: E402
import absorcion  # noqa: E402
import inyeccion_huecos as ih  # noqa: E402
import modelo as mo  # noqa: E402
import onside_paths  # noqa: E402
import posesion_rnn as pr  # noqa: E402
from torch_reproducibility import ensure_reproducible_process  # noqa: E402

STUDY = ROOT / "src/out/pr2_external_retrain/joint_redblue_20261110"
RUN = STUDY / "blue_architecture/runs/run_20261110"
PROMOTED = RUN / "candidates/promoted5"
SEAL = RUN / "supplemental_seal.json"
REFERENCE = STUDY / "reference/calibration_fd568aadb8d4e3f5/source_24aaedccf67fdbde"
STANDARD = STUDY / "calibrations/C00_release/calibration.json"
CONTINUITY = ROOT / "src/out/ball_continuity_20261111"
ARMS = {"vj": CONTINUITY / "calibration_vj.json", "fj": CONTINUITY / "calibration_fj.json"}
T1W_BOOSTER = REFERENCE / "T1w_external_fold1.txt"
T1W_SHA256 = "fe277da3d507f6a31802d2dc8bfee533da870934abe8cc0c99434b14e6b27b69"
T1W_LAMBDA = [32.0, 96.0]
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
SALT = "promoted5-jumps-20261111:"
MARGIN_M = 3.0
REG_TOLERANCE = -0.005


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def protocol_record() -> dict[str, Any]:
    seal = read(SEAL)
    recipe = read(PROMOTED / "a30.json")
    frozen = seal["models"]["promoted5"]
    return {"fit_pff": seal["fit"], "inner": seal["inner"], "outer": seal["outer"], "cfg": frozen["cfg"],
            "seeds": recipe["seeds"], "alphas": list(ALPHAS), "salt": SALT, "margin_m": MARGIN_M,
            "arms": {a: ap.external.sha256(p) for a, p in ARMS.items()}, "standard_sha256": ap.external.sha256(STANDARD),
            "seal_sha256": ap.external.sha256(SEAL), "t1w_sha256": T1W_SHA256,
            "frozen_promoted5": {"standalone_lambda": frozen["standalone_lambda"], "blend": frozen["blend"]},
            "adoption": "arm blend delta > 0 on both arm views and >= -0.005 on both standard views; vj first",
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def inject_jumps(ball: np.ndarray, mid: str, view: str, jumps: dict[str, Any], pitch: tuple) -> np.ndarray:
    out = ball.copy()
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"{SALT}{mid}:{view}".encode()).digest()[:8], "little"))
    visible = np.flatnonzero(np.isfinite(ball[:, 0]))
    target = int(round(jumps["rate"] * len(visible)))
    durations, distances = np.asarray(jumps["durations_s"]), np.asarray(jumps["distances_m"])
    free = np.ones(len(ball), bool)
    covered, scale = 0, np.array(pitch, float)
    low, high = -MARGIN_M / scale, 1 + MARGIN_M / scale
    while covered < target:
        start = int(visible[rng.integers(len(visible))])
        if not free[start]:
            continue
        k = int(rng.integers(len(durations)))
        length = max(1, int(round(durations[k] * pr.FPS)))
        angle = rng.uniform(0, 2 * np.pi)
        offset = distances[k] * np.array([np.cos(angle), np.sin(angle)]) / scale
        position = np.clip(ball[start] + offset, low, high)
        span = np.arange(start, min(start + length, len(ball)))
        span = span[free[span] & np.isfinite(ball[span, 0])]
        out[span] = position
        free[span] = False
        covered += len(span)
    return out


def arm_ball(d: dict, mid: str, view: str, calibration: dict) -> np.ndarray:
    raw = np.asarray(d["ball_xy"], float)
    if "jumps" in calibration:
        raw = inject_jumps(raw, mid, view, calibration["jumps"], tuple(d["pitch"]))
    ball = pr.dd.smooth_ball(raw, d["period"], pr.VIEWS[view])
    return ih.inject_holes(ball, d["label"], d["period"], mid, target=calibration["share"],
                           draw=ih.length_model(calibration), anchor_p=calibration["anchor_probability"])[0]


def build_match(mid: str, calibration_path: str) -> dict:
    """pr.build_match with arm_ball; view keys stay g6h / g8h so pr.entries works unchanged."""
    pr.restrict_training_reads((Path(onside_paths.CHALLENGE_DIR) / "data", Path(onside_paths.EVALUATION_DIR) / "gt",
                                Path(onside_paths.OUT_DIR) / "balon_yolo"))
    d = pr.rp.load_match(mid)
    per = np.asarray(d["period"])
    calibration = pr.external_gap_model(calibration_path)
    out = {"mid": mid, "per": per,
           "targets": {key: pr.mp_pff.label_codes(d[key]).astype(np.int64) for key in ("label", "label_poss")},
           "unc": np.asarray(d["uncertain"], bool), "gt": pr.rp.ev.make_gt(d, "label")}
    ev_f, ev_t = pr.touch_teams(d["label"], d["event_f"])
    out["touch"] = pr.touch_targets(len(per), ev_f, ev_t).astype(np.int64)
    for view in pr.VIEWS:
        ball = arm_ball(d, mid, view, calibration)
        out[view] = {"X": pr.build_features(ball, d["players_slots"], per, d["pitch"]).astype(np.float16),
                     "Xs": pr.build_features(ball, pr.swapped(d["players_slots"]), per, d["pitch"]).astype(np.float16)}
    return out


def t1w_match(mid: str, calibration_path: str) -> list[dict[str, Any]]:
    """T1w_external fold-1 probabilities on the arm balls of one match (same feature path as ap.t1w_entries)."""
    booster = pr.mp_pff.lightgbm.Booster(model_file=str(T1W_BOOSTER))
    calibration = pr.external_gap_model(calibration_path)
    d = pr.rp.load_match(mid)
    out = []
    for view in pr.VIEWS:
        ball = arm_ball(d, mid, view, calibration)
        context = pr.rp.build_method_ctx(ball, d["players_slots"], d["period"], d["pitch"])
        b3, touches, _ = pr.rp.pipeline(context)
        absorbed = absorcion.absorb(b3, pr.mp_pff.TAUS[0], d["period"])
        features, names = mo.build_features(ball, d["players_slots"], d["period"], d["pitch"], b3, absorbed, touches)
        out.append({"mid": mid, "view": view, "probabilities": mo.predict_proba(booster, features, names),
                    "per": np.asarray(d["period"])})
    return out


def pool_map(function, mids: list[str], argument: str, processes: int) -> list:
    with ProcessPoolExecutor(processes, mp_context=mp.get_context("spawn")) as ex:
        return list(ex.map(function, mids, [argument] * len(mids), timeout=pr.BUILD_TIMEOUT_S * len(mids)))


def select(averaged: list[np.ndarray], entries: list[dict[str, Any]], t1w: list[np.ndarray]) -> dict[str, Any]:
    standalone, standalone_obj = pr.choose_lambda(averaged, entries)
    candidates = []
    for alpha in ALPHAS:
        lam, objective = pr.choose_lambda([alpha * p + (1 - alpha) * t for p, t in zip(averaged, t1w, strict=True)],
                                          entries)
        candidates.append({"alpha": alpha, "lambda": list(lam), "inner_objective": objective})
    best = max(c["inner_objective"] for c in candidates)
    blend = next(c for c in candidates if c["inner_objective"] >= best - pr.rp.ep.TIE)
    return {"standalone_lambda": list(standalone), "standalone_objective": standalone_obj, "blend": blend,
            "blend_candidates": candidates}


def save_probs(directory: Path, entries: list[dict], values: list[np.ndarray]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for e, p in zip(entries, values, strict=True):
        np.savez(directory / f"{e['mid']}__{e['view']}.npz", probabilities=p.astype(np.float32), period=e["per"],
                 gt=e["gt"])


def run(arm: str, device_name: str, processes: int) -> None:
    ensure_reproducible_process()
    ap.guard_training()
    protocol = protocol_record()
    path = HERE / "protocol.json"
    if path.exists() and read(path) != protocol:
        raise ValueError("Preregistered protocol changed; use a new namespace")
    ap.save_json(path, protocol)
    if ap.external.sha256(T1W_BOOSTER) != T1W_SHA256:
        raise ValueError("T1w_external booster hash differs")
    out = HERE / arm
    device = ap.device_of(device_name)
    started = time.monotonic()
    mids = protocol["fit_pff"] + protocol["inner"] + protocol["outer"]
    built = {b["mid"]: b for b in pool_map(build_match, mids, str(ARMS[arm]), processes)}
    standard = pr.build_matches(protocol["outer"], processes, str(STANDARD))
    print(json.dumps({"stage": "built", "seconds": round(time.monotonic() - started, 1)}), flush=True)
    t1w = {(r["mid"], r["view"]): r for rows in pool_map(t1w_match, protocol["inner"] + protocol["outer"],
                                                           str(ARMS[arm]), processes) for r in rows}
    ap.save_json(out / "t1w_done.json", {"seconds": time.monotonic() - started})
    inner = pr.entries(protocol["inner"], built, protocol["cfg"])
    outer = pr.entries(protocol["outer"], built, protocol["cfg"])
    outer_std = pr.entries(protocol["outer"], standard, protocol["cfg"])
    manifest = read(PROMOTED / "a30.json")
    save_probs(out / "frozen_outer_arm", outer, [ap.model_probabilities(PROMOTED, e, manifest, device) for e in outer])
    save_probs(out / "t1w_outer_arm", outer, [t1w[(e["mid"], e["view"])]["probabilities"] for e in outer])
    inner_all, outer_all, std_all = [], [], []
    for seed in protocol["seeds"]:
        cfg = {**protocol["cfg"], "seed": seed}
        directory = out / "models" / str(seed)
        print(json.dumps({"stage": "train", "arm": arm, "seed": seed}), flush=True)
        _, inner_probabilities = ap.fit_one(directory, built, protocol["fit_pff"], protocol["inner"], cfg, device)
        inner_all.append(inner_probabilities)
        net = pr.Net(outer[0]["X"].shape[1], cfg).to(device)
        net.load_state_dict(ap.torch.load(directory / "weights.pt", map_location=device, weights_only=True))
        outer_all.append([pr.predict_tta(net, e["X"], e["Xs"], e["per"], cfg, device)[:, :3] for e in outer])
        std_all.append([pr.predict_tta(net, e["X"], e["Xs"], e["per"], cfg, device)[:, :3] for e in outer_std])
        del net
        print(json.dumps({"stage": "trained", "arm": arm, "seed": seed,
                          "seconds": round(time.monotonic() - started, 1)}), flush=True)
    average = lambda runs, n: [np.mean([r[i] for r in runs], axis=0) for i in range(n)]  # noqa: E731
    inner_avg = average(inner_all, len(inner))
    save_probs(out / "inner", inner, inner_avg)
    save_probs(out / "outer_arm", outer, average(outer_all, len(outer)))
    save_probs(out / "outer_standard", outer_std, average(std_all, len(outer_std)))
    selection = select(inner_avg, inner, [t1w[(e["mid"], e["view"])]["probabilities"] for e in inner])
    ap.save_json(out / "selection.json", {"complete": True, **selection, "seconds": time.monotonic() - started})
    print(json.dumps({"stage": "selected", "arm": arm, "blend": selection["blend"]}), flush=True)


def reference_outer(mid: str, view: str) -> dict[str, np.ndarray]:
    manifest = read(REFERENCE / "outer_manifest.json")
    row = next(r for r in manifest["entries"] if r["match"] == mid and r["view"] == view)
    if ap.external.sha256(REFERENCE / row["file"]) != row["sha256"]:
        raise ValueError(f"T1w outer probability bytes differ: {row['file']}")
    return npz(REFERENCE / row["file"])


def report() -> None:
    protocol = read(HERE / "protocol.json")
    frozen = protocol["frozen_promoted5"]
    result: dict[str, Any] = {"complete": True, "arms": {}}
    for arm in ARMS:
        if not (HERE / arm / "selection.json").exists():
            continue
        selection = read(HERE / arm / "selection.json")
        rows = []
        for mid in protocol["outer"]:
            for view in pr.VIEWS:
                fa = npz(HERE / arm / "frozen_outer_arm" / f"{mid}__{view}.npz")
                ta = npz(HERE / arm / "t1w_outer_arm" / f"{mid}__{view}.npz")
                na = npz(HERE / arm / "outer_arm" / f"{mid}__{view}.npz")
                ns = npz(HERE / arm / "outer_standard" / f"{mid}__{view}.npz")
                fs = npz(RUN / "outer/promoted5" / f"{mid}__{view}.npz")
                ts = reference_outer(mid, view)
                if not (np.array_equal(fa["period"], fs["period"]) and np.array_equal(ns["gt"], fs["gt"])):
                    raise ValueError(f"Outer periods or ground truth differ: {mid} {view}")
                gt, per = fs["gt"], fs["period"]

                def score(p: np.ndarray, lam: list[float]) -> float:
                    return pr.spell_f1(gt, pr.decode(p, per, tuple(lam)))

                fa_, sa_ = frozen["blend"]["alpha"], selection["blend"]["alpha"]
                rows.append({"match": mid, "view": view,
                             "arm_frozen_blend": score(fa_ * fa["probabilities"] + (1 - fa_) * ta["probabilities"],
                                                       frozen["blend"]["lambda"]),
                             "arm_new_blend": score(sa_ * na["probabilities"] + (1 - sa_) * ta["probabilities"],
                                                    selection["blend"]["lambda"]),
                             "arm_frozen_nn": score(fa["probabilities"], frozen["standalone_lambda"]),
                             "arm_new_nn": score(na["probabilities"], selection["standalone_lambda"]),
                             "arm_t1w": score(ta["probabilities"], T1W_LAMBDA),
                             "std_frozen_blend": score(fa_ * fs["probabilities"] + (1 - fa_) * ts["probabilities"],
                                                       frozen["blend"]["lambda"]),
                             "std_new_blend": score(sa_ * ns["probabilities"] + (1 - sa_) * ts["probabilities"],
                                                    selection["blend"]["lambda"]),
                             "std_frozen_nn": score(fs["probabilities"], frozen["standalone_lambda"]),
                             "std_new_nn": score(ns["probabilities"], selection["standalone_lambda"])})
        means, deltas = {}, {}
        for view in pr.VIEWS:
            vr = [r for r in rows if r["view"] == view]
            means[view] = {k: float(np.mean([r[k] for r in vr])) for k in vr[0] if k not in ("match", "view")}
            deltas[view] = {f"{fam}_{kind}": pr.rp.ep.delta_stats([r[f"{fam}_new_{kind}"] - r[f"{fam}_frozen_{kind}"]
                                                                   for r in vr])
                            for fam in ("arm", "std") for kind in ("blend", "nn")}
        if abs(means["g6h"]["std_frozen_blend"] - 0.699172750677265) > 1e-9 or \
                abs(means["g8h"]["std_frozen_blend"] - 0.6968560975788046) > 1e-9:
            raise ValueError("Frozen promoted5 standard outer blend did not reproduce")
        qualifies = all(deltas[v]["arm_blend"]["mean"] > 0 and deltas[v]["std_blend"]["mean"] >= REG_TOLERANCE
                        for v in pr.VIEWS)
        result["arms"][arm] = {"selection": selection, "rows": rows, "means": means, "deltas": deltas,
                               "qualifies": qualifies}
    qualified = [a for a in ARMS if result["arms"].get(a, {}).get("qualifies")]
    result["adopted"] = qualified[0] if qualified else None
    ap.save_json(HERE / "report.json", result)
    print(json.dumps({a: {"means": r["means"], "qualifies": r["qualifies"]} for a, r in result["arms"].items()} |
                     {"adopted": result["adopted"]}, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="action", required=True)
    runner = sub.add_parser("run")
    runner.add_argument("--arm", choices=tuple(ARMS), required=True)
    runner.add_argument("--device", choices=("cpu", "0"), required=True)
    runner.add_argument("--processes", type=int, required=True)
    sub.add_parser("report")
    args = parser.parse_args()
    if args.action == "run":
        run(args.arm, args.device, args.processes)
    else:
        report()


if __name__ == "__main__":
    main()
