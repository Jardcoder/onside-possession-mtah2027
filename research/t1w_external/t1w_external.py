"""Train an external-calibrated T1w reference with untouched outer matches."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any

sys.dont_write_bytecode = True
if any(os.environ.get(key) != "" for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")):
    raise RuntimeError("CPU execution requires CUDA/HIP/ROCR_VISIBLE_DEVICES to be explicitly empty")

ROOT = Path(__file__).resolve().parents[5]
for directory in (ROOT / "src", ROOT / "bancos"):
    sys.path.insert(0, str(directory))

import onside_paths
from training_data_guard import restrict_training_reads

restrict_training_reads((Path(onside_paths.CHALLENGE_DIR) / "data", Path(onside_paths.EVALUATION_DIR) / "gt", Path(onside_paths.OUT_DIR) / "balon_yolo", Path(onside_paths.OUT_DIR) / "modelo_cache"))

import absorcion
import a30_protocol as protocol
import modelo as model
import modelo_transfer as transfer
import pff
import ronda5_t1 as weighted

OUTER = ["3813", "3818", "3823", "3838", "3843", "3844", "10503", "10508"]
VIEWS = tuple(transfer.H_BALLS)
BASE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: Any) -> None:
    protocol.save_json(path, value)


def source_hashes() -> dict[str, str]:
    modules = (protocol, protocol.pr, model, transfer, weighted, absorcion, protocol.pr.rp, protocol.pr.dd, protocol.pr.ih, protocol.pr.mp_pff, protocol.pr.rp.ep, protocol.pr.rp.ev, protocol.pr.rp.pm, protocol.pr.rp.pff_split)
    paths = [Path(module.__file__).resolve() for module in modules]
    paths += [Path(__file__).resolve(), ROOT / "bancos/pff.py", ROOT / "src/training_data_guard.py", ROOT / "src/onside_paths.py"]
    paths += [Path(module.__file__).resolve() for module in tuple(sys.modules.values()) if getattr(module, "__file__", None) and Path(module.__file__).suffix == ".py" and any(Path(module.__file__).resolve().is_relative_to(ROOT / directory) for directory in ("src", "bancos"))]
    return {str(path): sha256(path) for path in paths}


def input_hashes(mid: str) -> dict[str, str]:
    paths = {f"{folder}/{mid}{suffix}": Path(onside_paths.PFF_DIR) / folder / f"{mid}{suffix}" for folder, suffix in (("Metadata", ".json"), ("Rosters", ".json"), ("Event Data", ".json"), ("Tracking Data", ".jsonl.bz2"))}
    native_cache = pff._cache_path(mid)
    if not native_cache.is_file():
        raise FileNotFoundError(f"Declared preexisting native PFF cache is missing: {mid}")
    paths[f"cache/{native_cache.name}"] = native_cache
    return {name: sha256(path) for name, path in paths.items()}


def frozen_plan(calibration_path: Path) -> dict[str, Any]:
    development = list(protocol.pr.rp.pff_split.DESARROLLO)
    if len(OUTER) != 8 or len(set(OUTER)) != 8 or not set(OUTER) <= set(development):
        raise ValueError("Outer fold must contain eight distinct development matches")
    training = [mid for mid in development if mid not in OUTER]
    fit, inner = protocol.split_training(training)
    calibration = protocol.pr.external_gap_model(str(calibration_path))
    calibration_mids = {item["clip"]["match_id"] for item in calibration["sources"]}
    if not calibration_mids <= set(training) or calibration_mids & set(OUTER):
        raise ValueError("Calibration sources must be outside the outer fold and inside training32")
    return {"schema_version": 1, "variant": "T1w_external_fit_only", "fold": 1, "outer": OUTER, "training": training, "fit": fit, "inner": inner, "ground_truth": "PFF label", "class_order": list(model.STATES), "views": VIEWS, "calibration_path": str(calibration_path.resolve()), "calibration_sha256": sha256(calibration_path), "calibration_detector_sha256": calibration["detector_sha256"], "calibration_protocol_version": calibration["protocol_version"], "calibration_matches": sorted(calibration_mids), "calibration_original_excluded_matches": calibration["excluded_matches"], "calibration_reused_aggregate": True, "source_sha256": source_hashes(), "step": transfer.STEP, "median_scope": "fit28 only; historical T1w used training32", "native_tracking_cache": "Preexisting native PFF cache reused and hashed alongside raw inputs; raw-to-cache equality not independently regenerated", "lambda_ha": protocol.pr.LAM_HA, "lambda_d": protocol.pr.LAM_D, "outer_scored": False, "model_parameters": model.LGB_PARAMS, "maximum_rounds": model.MAX_ROUNDS, "early_stopping_rounds": model.EARLY_STOP}


def feature_paths(directory: Path, mid: str, view: str) -> tuple[Path, Path, Path]:
    stem = directory / "features" / f"{mid}__{view}"
    return stem.with_suffix(".npy"), stem.with_suffix(".npz"), stem.with_suffix(".json")


def build_features(directory: Path, mid: str, calibration_path: Path, plan: dict[str, Any]) -> dict[str, Any]:
    if mid not in plan["training"]:
        raise ValueError("Feature preparation is restricted to the preregistered training32")
    fingerprint = {"calibration_sha256": plan["calibration_sha256"], "source_sha256": plan["source_sha256"], "input_sha256": input_hashes(mid)}
    cached = all(feature_paths(directory, mid, view)[2].exists() for view in VIEWS)
    if cached:
        for view in VIEWS:
            x_path, meta_path, provenance_path = feature_paths(directory, mid, view)
            existing = json.loads(provenance_path.read_text())
            if existing["fingerprint"] != fingerprint or not x_path.exists() or not meta_path.exists():
                raise ValueError(f"Feature cache does not match frozen sources: {mid}/{view}")
            if sha256(x_path) != existing["features_sha256"] or sha256(meta_path) != existing["metadata_sha256"]:
                raise ValueError(f"Feature or metadata bytes changed: {mid}/{view}")
            features = protocol.pr.np.load(x_path, mmap_mode="r", allow_pickle=False)
            metadata = load_meta(directory, mid, view)
            if features.shape != (existing["rows"], existing["columns"]) or str(features.dtype) != existing["dtype"] or features.dtype != protocol.pr.np.float32:
                raise ValueError(f"Feature cache shape or dtype differs: {mid}/{view}")
            if len(metadata["names"]) != features.shape[1] or any(len(metadata[key]) != len(features) for key in ("y", "per", "gt")):
                raise ValueError(f"Metadata cache dimensions differ: {mid}/{view}")
        return {"mid": mid, "cached": True}
    started = time.monotonic()
    data = protocol.pr.rp.load_match(mid)
    calibration = protocol.pr.external_gap_model(str(calibration_path))
    rows = []
    for view in VIEWS:
        ball = protocol.pr.training_ball(data, mid, view, calibration)
        context = protocol.pr.rp.build_method_ctx(ball, data["players_slots"], data["period"], data["pitch"])
        b3, touches, _ = protocol.pr.rp.pipeline(context)
        absorbed = absorcion.absorb(b3, protocol.pr.mp_pff.TAUS[0], data["period"])
        features, names = model.build_features(ball, data["players_slots"], data["period"], data["pitch"], b3, absorbed, touches)
        x_path, meta_path, provenance_path = feature_paths(directory, mid, view)
        x_path.parent.mkdir(parents=True, exist_ok=True)
        protocol.pr.mp_pff.atomic_save(str(x_path), features)
        protocol.pr.mp_pff.atomic_savez(str(meta_path), names=names, y=protocol.pr.mp_pff.label_codes(data["label"]), per=data["period"], gt=protocol.pr.rp.ev.make_gt(data, "label"))
        info = {"fingerprint": fingerprint, "rows": int(len(features)), "columns": int(features.shape[1]), "dtype": str(features.dtype), "features_sha256": sha256(x_path), "metadata_sha256": sha256(meta_path)}
        save_json(provenance_path, info)
        rows.append({"view": view, **{key: info[key] for key in ("rows", "columns", "dtype")}})
        del features, ball, context, b3, touches, absorbed
    return {"mid": mid, "cached": False, "views": rows, "elapsed_seconds": time.monotonic() - started, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def load_meta(directory: Path, mid: str, view: str) -> dict[str, Any]:
    with protocol.pr.np.load(feature_paths(directory, mid, view)[1], allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def rows_and_weights(directory: Path, mids: list[str], median: float, names: list[str]) -> tuple[Any, Any, Any]:
    np = protocol.pr.np
    count = sum(len(load_meta(directory, mid, view)["y"][(transfer.STEP // 2) * index::transfer.STEP]) for mid in mids for index, view in enumerate(VIEWS))
    x = np.empty((2 * count, len(names)), dtype=np.float32)
    y = np.empty(2 * count, dtype=np.int8)
    w = np.empty(2 * count, dtype=np.float64)
    position = 0
    for mid in mids:
        for index, view in enumerate(VIEWS):
            offset = (transfer.STEP // 2) * index
            metadata = load_meta(directory, mid, view)
            if list(metadata["names"]) != names:
                raise ValueError("Feature names differ across newly built inputs")
            chunk = np.ascontiguousarray(np.load(feature_paths(directory, mid, view)[0], mmap_mode="r")[offset::transfer.STEP])
            chunk_y = metadata["y"][offset::transfer.STEP]
            augmented_x, augmented_y = model.augment(chunk, chunk_y, names)
            chunk_weight = weighted.raw_weights("T1w", metadata["y"], metadata["per"], np.zeros(len(metadata["y"])), median)[offset::transfer.STEP]
            size = len(chunk_y)
            x[position:position + size], x[count + position:count + position + size] = augmented_x[:size], augmented_x[size:]
            y[position:position + size], y[count + position:count + position + size] = augmented_y[:size], augmented_y[size:]
            w[position:position + size], w[count + position:count + position + size] = chunk_weight, chunk_weight
            position += size
    if position != count:
        raise ValueError("Row preparation did not fill the preregistered allocation")
    return x, y, w


def train(directory: Path, calibration_path: Path, plan: dict[str, Any]) -> dict[str, Any]:
    np = protocol.pr.np
    started = time.monotonic()
    for mid in plan["training"]:
        if not all(feature_paths(directory, mid, view)[2].exists() for view in VIEWS):
            raise FileNotFoundError(f"Prepare source-aware features before fitting: {mid}")
        build_features(directory, mid, calibration_path, plan)
    metas = [load_meta(directory, mid, VIEWS[0]) for mid in plan["fit"]]
    median = weighted.median_length([meta["y"] for meta in metas], [meta["per"] for meta in metas])
    names = list(metas[0]["names"])
    del metas
    x_train, y_train, w_train = rows_and_weights(directory, plan["fit"], median, names)
    x_inner, y_inner, w_inner = rows_and_weights(directory, plan["inner"], median, names)
    w_train, w_inner, scale = weighted.normalise_weights("T1w", w_train, w_inner)
    fit_started = time.monotonic()
    booster = weighted.fit_weighted(x_train, y_train, w_train, x_inner, y_inner, w_inner, threads=2, rounds=model.MAX_ROUNDS, params=dict(model.LGB_PARAMS))
    fit_seconds = time.monotonic() - fit_started
    training_rows, inner_rows = len(y_train), len(y_inner)
    del x_train, x_inner, y_train, y_inner, w_train, w_inner
    checkpoint = directory / "T1w_external_fold1.txt"
    booster.save_model(str(checkpoint), num_iteration=booster.best_iteration)
    probabilities, entries, probability_hashes = [], [], {}
    for mid in plan["inner"]:
        for view in VIEWS:
            meta = load_meta(directory, mid, view)
            probability = model.predict_proba(booster, np.load(feature_paths(directory, mid, view)[0], mmap_mode="r"), names).astype(np.float32)
            if probability.shape != (len(meta["per"]), 3) or not np.isfinite(probability).all() or np.any(probability < 0) or np.any(probability > 1) or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-6):
                raise ValueError(f"Inner probabilities are invalid: {mid}/{view}")
            path = directory / "inner" / f"{mid}__{view}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            protocol.pr.mp_pff.atomic_savez(str(path), probabilities=probability, gt=meta["gt"], per=meta["per"])
            probabilities.append(probability)
            entries.append({"mid": mid, "view": view, "gt": meta["gt"], "per": meta["per"]})
            probability_hashes[path.name] = sha256(path)
    lam, objective = protocol.pr.choose_lambda(probabilities, entries)
    result = {"complete": True, "plan": plan, "feature_names": names, "feature_provenance_sha256": {f"{mid}__{view}": sha256(feature_paths(directory, mid, view)[2]) for mid in plan["training"] for view in VIEWS}, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "median_L": median, "weight_scale": scale, "fit_rows_augmented": training_rows, "inner_rows_augmented": inner_rows, "best_iteration": int(booster.best_iteration), "lambda": lam, "inner_objective": objective, "inner_probabilities_sha256": probability_hashes, "fit_seconds": fit_seconds, "elapsed_seconds": time.monotonic() - started, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "outer_scored": False, "runtime": {"device": "cpu", "threads": 2, "numpy": np.__version__, "lightgbm": protocol.pr.mp_pff.lightgbm.__version__}}
    save_json(directory / "model.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("plan", "probe", "build", "train"))
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--match", required=True)
    args = parser.parse_args()
    plan = frozen_plan(args.calibration)
    directory = BASE / ("calibration_" + plan["calibration_sha256"][:16]) / ("source_" + sha256(Path(__file__))[:16])
    plan_path = directory / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != json.loads(json.dumps(plan)):
        raise ValueError("Preregistered inputs changed; use a new reference directory")
    save_json(plan_path, plan)
    if args.stage == "plan":
        print(json.dumps(plan), flush=True)
    elif args.stage == "probe":
        if args.match not in plan["fit"]:
            raise ValueError("Timing probe must use a fit development match")
        result = build_features(directory, args.match, args.calibration, plan)
        save_json(directory / ("probe_cached.json" if result["cached"] else "probe.json"), result)
        print(json.dumps(result), flush=True)
    elif args.stage == "build":
        for mid in plan["training"]:
            print(json.dumps(build_features(directory, mid, args.calibration, plan)), flush=True)
    elif args.stage == "train":
        print(json.dumps(train(directory, args.calibration, plan)), flush=True)


if __name__ == "__main__":
    main()
