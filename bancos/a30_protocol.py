"""PFF-only selection, five-seed training, out-of-fold evaluation and frozen inference."""

import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch

import absorcion
import a30_external as external
import features_reto as fr
import modelo as mo
import onside_paths
import posesion_rnn as pr
import predecir_reto_m1 as submission
from torch_reproducibility import (ensure_reproducible_process, runtime_description,
                                   seed_deterministically, tensor_sha256)
from training_data_guard import restrict_training_reads


@functools.lru_cache(maxsize=1)
def guard_training() -> None:
    restrict_training_reads((Path(onside_paths.CHALLENGE_DIR) / "data",
                             Path(onside_paths.EVALUATION_DIR) / "gt",
                             Path(onside_paths.OUT_DIR) / "balon_yolo"))


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    if protocol["selection_ground_truth"] != "label" or protocol["anchor_probability"] != 0:
        raise ValueError("The PFF selection ground truth is label; external injection must be unanchored")
    if protocol["gpu_share"] != pr.GPU_SHARE:
        raise ValueError("Protocol GPU share and runtime budget differ")
    if protocol["lambda_ha"] != list(pr.LAM_HA) or protocol["lambda_d"] != list(pr.LAM_D):
        raise ValueError("Protocol lambda grid and runtime grid differ")
    return protocol


def device_of(requested: str) -> torch.device:
    seed_deterministically(0)
    if requested == "0" and not torch.cuda.is_available():
        raise RuntimeError("Requested GPU 0 is unavailable")
    device = torch.device("cpu" if requested == "cpu" else "cuda:0")
    pr.limit_gpu(device)
    return device


def configuration(protocol: dict[str, Any], candidate: dict[str, Any], seed: int) -> dict[str, Any]:
    return {**protocol["network"], **candidate, "seed": seed,
            "maximum_training_seconds": protocol["maximum_training_seconds_per_model"]}


def split_training(mids: list[str]) -> tuple[list[str], list[str]]:
    if set(mids) - set(pr.rp.pff_split.DESARROLLO):
        raise ValueError("A30 may train only on PFF development matches")
    ordered = sorted(mids, key=lambda mid: hashlib.sha256((pr.mp_pff.ES_SALT + mid).encode()).hexdigest())
    return ordered[pr.mp_pff.ES_MATCHES:], ordered[:pr.mp_pff.ES_MATCHES]


def source_name(path: Path, repository: Path, challenge: Path) -> str:
    if path.is_relative_to(repository):
        return str(path.relative_to(repository))
    if path.is_relative_to(challenge):
        return str(Path("challenge") / path.relative_to(challenge))
    raise ValueError(f"Model source is outside the declared repositories: {path}")


def freeze_run(directory: Path, protocol_path: Path, calibration_path: Path,
               fit: list[str], validation: list[str]) -> dict[str, Any]:
    modules = [Path(module.__file__) for module in (pr, external, mo, pr.dd, pr.ih, pr.rp, pr.mp_pff,
                                                   pr.rp.ep, pr.rp.ev, pr.rp.pm, pr.rp.pff_split)]
    modules += [Path(onside_paths.ROOT) / "bancos/pff.py"]
    modules += [Path(__file__), Path(onside_paths.ROOT) / "src/gpu_budget.py",
                Path(onside_paths.ROOT) / "src/training_data_guard.py",
                Path(onside_paths.ROOT) / "src/torch_reproducibility.py"]
    manifest = {"protocol_sha256": external.sha256(protocol_path),
                "calibration_sha256": external.sha256(calibration_path),
                "fit": fit, "early_stopping": validation,
                "pff_input_sha256": {f"{folder}/{mid}{suffix}": external.sha256(Path(onside_paths.PFF_DIR) / folder / f"{mid}{suffix}")
                                     for mid in fit + validation
                                     for folder, suffix in (("Metadata", ".json"), ("Rosters", ".json"),
                                                            ("Event Data", ".json"), ("Tracking Data", ".jsonl.bz2"))},
                "source_sha256": {source_name(path, Path(onside_paths.ROOT), Path(onside_paths.CHALLENGE_DIR)): external.sha256(path)
                                  for path in modules}}
    path = directory / "run_manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError(f"Run inputs changed; use a new output directory: {directory}")
    save_json(path, manifest)
    return manifest


def fit_one(directory: Path, built: dict, fit: list[str], validation: list[str],
            cfg: dict[str, Any], device: torch.device) -> tuple[dict[str, Any], list[np.ndarray]]:
    manifest_path = directory / "model.json"
    checkpoint = directory / "weights.pt"
    training, early = pr.entries(fit, built, cfg), pr.entries(validation, built, cfg)
    if manifest_path.exists():
        metadata = json.loads(manifest_path.read_text())
        if metadata["cfg"] != cfg or metadata["fit"] != fit or metadata["early_stopping"] != validation:
            raise ValueError(f"Cached model configuration differs: {directory}")
        if external.sha256(checkpoint) != metadata["checkpoint_sha256"]:
            raise ValueError(f"Cached checkpoint hash differs: {directory}")
        net = pr.Net(training[0]["X"].shape[1], cfg).to(device)
        net.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    else:
        started = time.monotonic()
        net, history = pr.train(training, early, cfg, device, print)
        directory.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint.with_suffix(".tmp")
        torch.save({key: value.detach().cpu() for key, value in net.state_dict().items()}, temporary)
        temporary.replace(checkpoint)
        metadata = {"cfg": cfg, "fit": fit, "early_stopping": validation,
                    "history": history, "elapsed_seconds": time.monotonic() - started,
                    "checkpoint_sha256": external.sha256(checkpoint), "tensor_sha256": tensor_sha256(net.state_dict()),
                    "runtime": runtime_description(str(device))}
        save_json(manifest_path, metadata)
    probabilities = [pr.predict_tta(net, e["X"], e["Xs"], e["per"], cfg, device)[:, :3] for e in early]
    del net
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metadata, probabilities


def select_and_train(directory: Path, protocol_path: Path, calibration_path: Path,
                     mids: list[str], processes: int, device: torch.device) -> dict[str, Any]:
    guard_training()
    protocol = read_protocol(protocol_path)
    fit, validation = split_training(mids)
    calibration = pr.external_gap_model(str(calibration_path))
    calibration_matches = {source["clip"]["match_id"] for source in calibration["sources"]}
    if not calibration_matches <= set(mids):
        raise ValueError("External calibration includes a match outside this training fold")
    run_manifest = freeze_run(directory, protocol_path, calibration_path, fit, validation)
    if shutil.disk_usage(directory).free < protocol["minimum_free_disk_bytes"]:
        raise RuntimeError("Insufficient free disk for the declared A30 run")
    print(json.dumps({"stage": "build_pff", "matches": len(mids), "fit": len(fit), "early_stopping": len(validation)}), flush=True)
    built = pr.build_matches(mids, processes, str(calibration_path))
    seed = protocol["seeds"][0]
    candidates, probability_by_candidate = [], []
    for candidate in protocol["model_candidates"]:
        cfg = configuration(protocol, candidate, seed)
        print(json.dumps({"stage": "candidate", "candidate": candidate["name"], "seed": seed}), flush=True)
        metadata, probabilities = fit_one(directory / "selection" / candidate["name"], built, fit, validation, cfg, device)
        lam, objective = pr.choose_lambda(probabilities, pr.entries(validation, built, cfg))
        candidates.append({"name": candidate["name"], "cfg": cfg, "lambda": lam, "objective": objective,
                           "tensor_sha256": metadata["tensor_sha256"]})
        probability_by_candidate.append(probabilities)
        save_json(directory / "selection.json", {"complete": False, "candidates": candidates})
    selected_index = int(np.flatnonzero(np.asarray([c["objective"] for c in candidates]) >=
                                      max(c["objective"] for c in candidates) - pr.rp.ep.TIE)[0])
    selected = candidates[selected_index]
    ensemble = [probability_by_candidate[selected_index]]
    weights = []
    for current_seed in protocol["seeds"]:
        cfg = {**selected["cfg"], "seed": current_seed}
        if current_seed == seed:
            model_dir = directory / "selection" / selected["name"]
            metadata = json.loads((model_dir / "model.json").read_text())
        else:
            model_dir = directory / "seeds" / str(current_seed)
            metadata, probabilities = fit_one(model_dir, built, fit, validation, cfg, device)
            ensemble.append(probabilities)
        output = directory / f"a30_s{current_seed}.pt"
        shutil.copyfile(model_dir / "weights.pt", output)
        weights.append({"seed": current_seed, "file": output.name, "sha256": external.sha256(output),
                        "tensor_sha256": metadata["tensor_sha256"]})
    early = pr.entries(validation, built, selected["cfg"])
    lam, objective = pr.choose_lambda([np.mean([p[i] for p in ensemble], axis=0) for i in range(len(early))], early)
    manifest = {"schema_version": 2, "protocol": protocol, "run_manifest": run_manifest,
                "cfg": selected["cfg"], "seeds": protocol["seeds"], "lambda": lam,
                "selection": {"complete": True, "candidates": candidates, "selected": selected["name"],
                              "ensemble_objective": objective, "ground_truth": "PFF label"},
                "calibration": calibration, "weights": weights,
                "adoption": "candidate; acceptance requires the separate complete paired PFF comparison",
                "runtime": runtime_description(str(device))}
    save_json(directory / "selection.json", manifest["selection"])
    save_json(directory / "a30.json", manifest)
    return manifest


def model_probabilities(directory: Path, entry: dict, manifest: dict[str, Any], device: torch.device) -> np.ndarray:
    probabilities = []
    for weight in manifest["weights"]:
        checkpoint = directory / weight["file"]
        if external.sha256(checkpoint) != weight["sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {checkpoint.name}")
        net = pr.Net(entry["X"].shape[1], manifest["cfg"]).to(device)
        net.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        probabilities.append(pr.predict_tta(net, entry["X"], entry["Xs"], entry["per"], manifest["cfg"], device)[:, :3])
        del net
    return np.mean(probabilities, axis=0)


def validate_folds(directory: Path, external_dir: Path, protocol_path: Path, detector: Path,
                   processes: int, device: torch.device) -> None:
    guard_training()
    protocol = read_protocol(protocol_path)
    reviews = Path(onside_paths.ROOT) / "annotations/ball/reviews"
    for fold, held in enumerate(pr.rp.FOLDS):
        fold_dir = directory / f"fold_{fold}"
        calibration_path = fold_dir / "calibration.json"
        calibration = external.calibrate(external_dir, reviews, protocol, external.sha256(detector), set(held))
        save_json(calibration_path, calibration)
        training = [mid for index, matches in enumerate(pr.rp.FOLDS) if index != fold for mid in matches]
        manifest = select_and_train(fold_dir, protocol_path, calibration_path, training, processes, device)
        built = pr.build_matches(list(held), processes, str(calibration_path))
        entries = pr.entries(list(held), built, manifest["cfg"])
        rows = []
        for entry in entries:
            probabilities = model_probabilities(fold_dir, entry, manifest, device)
            codes = pr.decode(probabilities, entry["per"], manifest["lambda"])
            rows.append({"match": entry["mid"], "ball": entry["view"], "fold": fold,
                         "spell_f1": pr.spell_f1(entry["gt"], codes)})
        save_json(fold_dir / "oof_scores.json", {"complete": True, "ground_truth": "PFF label", "rows": rows})
        print(json.dumps({"stage": "outer_fold_complete", "fold": fold, "scores": len(rows)}), flush=True)


def predict(directory: Path, output: Path, device: torch.device) -> None:
    """Use the frozen PFF decisions; the public GT is not opened here."""
    manifest = json.loads((directory / "a30.json").read_text())
    if manifest["schema_version"] != 2 or not manifest["selection"]["complete"]:
        raise ValueError("Prediction requires a complete external-protocol model manifest")
    with np.load(pr.YOLO_TRACK, allow_pickle=False) as track:
        if str(track["selection_sha256"]) != hashlib.sha256(
                json.dumps(manifest["calibration"], sort_keys=True).encode()).hexdigest():
            raise ValueError("Challenge track was not generated with this model's frozen PFF tracker selection")
        inputs = pr.challenge_inputs(track["ball_xy"])
    probabilities = [model_probabilities(directory, {**view, "per": inputs["per"]}, manifest, device)
                     for view in inputs["views"]]
    codes = pr.decode(np.mean(probabilities, axis=0), inputs["per"], manifest["lambda"])
    labels = fr.to_video(np.asarray(mo.STATES)[codes], {"frames": inputs["frames"], "n_video": inputs["n_video"]})
    submission.check_labels(labels)
    submission.write_csv(str(output), labels)
    result = subprocess.run([sys.executable, str(Path(onside_paths.EVALUATION_DIR) / "validate_submission.py"), str(output)],
                            capture_output=True, text=True, timeout=60)
    validator_log = output.with_suffix(output.suffix + ".validation.log")
    validator_log.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"Official CSV validator failed with exit code {result.returncode}; inspect {validator_log}")
    save_json(output.with_suffix(output.suffix + ".provenance.json"), {
        "model_manifest_sha256": external.sha256(directory / "a30.json"), "csv_sha256": external.sha256(output),
        "track_sha256": external.sha256(Path(pr.YOLO_TRACK)), "public_gt_read": False,
        "runtime": runtime_description(str(device))})
    print(json.dumps({"stage": "prediction_validated", "rows": len(labels), "csv_sha256": external.sha256(output)}))


def t1w_entries(booster: Any, mids: list[str], calibration: dict[str, Any]) -> tuple[list[np.ndarray], list[dict]]:
    """Use the exact external ball transformation used by A30, with T1w's own feature builder."""
    probabilities, entries = [], []
    for mid in mids:
        data = pr.rp.load_match(mid)
        for view in pr.VIEWS:
            ball = pr.training_ball(data, mid, view, calibration)
            context = pr.rp.build_method_ctx(ball, data["players_slots"], data["period"], data["pitch"])
            b3, touches, _ = pr.rp.pipeline(context)
            absorbed = absorcion.absorb(b3, pr.mp_pff.TAUS[0], data["period"])
            features, names = mo.build_features(ball, data["players_slots"], data["period"], data["pitch"], b3, absorbed, touches)
            probabilities.append(mo.predict_proba(booster, features, names))
            entries.append({"mid": mid, "view": view, "per": data["period"], "gt": pr.rp.ev.make_gt(data, "label")})
    return probabilities, entries


def compare_t1w(directory: Path, reference: Path, output: Path) -> None:
    """Paired 40-match comparison. No partial fold result can pass the adoption rule."""
    required = [reference / f"T1w_k{k}.txt" for k in range(len(pr.rp.FOLDS))]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing historical T1w boosters: " + ", ".join(missing))
    rows = []
    protocol_hashes: set[str] = set()
    for fold, held in enumerate(pr.rp.FOLDS):
        fold_dir = directory / f"fold_{fold}"
        candidate = json.loads((fold_dir / "oof_scores.json").read_text())
        manifest = json.loads((fold_dir / "a30.json").read_text())
        protocol_hashes.add(manifest["run_manifest"]["protocol_sha256"])
        if not candidate["complete"] or candidate["ground_truth"] != "PFF label":
            raise ValueError(f"Incomplete outer fold: {fold}")
        expected = {(mid, view) for mid in held for view in pr.VIEWS}
        measured = {(row["match"], row["ball"]) for row in candidate["rows"]}
        if measured != expected or len(candidate["rows"]) != len(expected):
            raise ValueError(f"Missing or duplicate A30 OOF scores in fold {fold}")
        booster = pr.mp_pff.lightgbm.Booster(model_file=str(required[fold]))
        early_probabilities, early = t1w_entries(booster, manifest["run_manifest"]["early_stopping"], manifest["calibration"])
        lam, _ = pr.choose_lambda(early_probabilities, early)
        probabilities, entries = t1w_entries(booster, list(held), manifest["calibration"])
        own = {(row["match"], row["ball"]): row["spell_f1"] for row in candidate["rows"]}
        for probability, entry in zip(probabilities, entries):
            score = pr.spell_f1(entry["gt"], pr.decode(probability, entry["per"], lam))
            a30 = own[(entry["mid"], entry["view"])]
            rows.append({"match": entry["mid"], "ball": entry["view"], "fold": fold,
                         "a30_spell_f1": a30, "t1w_spell_f1": score, "delta": a30 - score,
                         "t1w_lambda": lam, "t1w_checkpoint_sha256": external.sha256(required[fold])})
    if len(protocol_hashes) != 1:
        raise ValueError("OOF model protocols differ across folds")
    stats = {view: pr.rp.ep.delta_stats([row["delta"] for row in rows if row["ball"] == view]) for view in pr.VIEWS}
    conditions, accepted = pr.rp.verdict(stats["g6h"], stats["g8h"])
    save_json(output, {"complete": True, "matches": len(pr.rp.pff_split.DESARROLLO), "rows": rows,
                       "ground_truth": "PFF label", "statistics": stats, "conditions": conditions,
                       "accepted": accepted, "shared_ball": "PFF g6h/g8h with fold-specific external calibration",
                       "protocol_sha256": protocol_hashes.pop(),
                       "reference": "Historical pretrained T1w, original model training unchanged"})
    print(json.dumps({"stage": "paired_t1w_comparison", "accepted": accepted, "statistics": stats}))


def package_model(directory: Path, output: Path) -> None:
    """Copy the five verified checkpoints and aggregate provenance into a reviewable release."""
    manifest = json.loads((directory / "a30.json").read_text())
    if manifest["schema_version"] != 2 or len(manifest["weights"]) != 5 or not manifest["selection"]["complete"]:
        raise ValueError("Packaging requires a complete five-seed external-protocol ensemble")
    if set(manifest["seeds"]) != {row["seed"] for row in manifest["weights"]}:
        raise ValueError("Checkpoint seeds and model manifest differ")
    sources = [directory / "a30.json"]
    for row in manifest["weights"]:
        source = directory / row["file"]
        if external.sha256(source) != row["sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {source.name}")
        sources.append(source)
    output.mkdir(parents=True, exist_ok=True)
    for source in sources:
        target = output / source.name
        if target.exists() and external.sha256(target) != external.sha256(source):
            raise FileExistsError(f"Refusing to replace a different packaged artifact: {target}")
        shutil.copyfile(source, target)
    (output / "SHA256SUMS").write_text("".join(f"{external.sha256(output / source.name)}  {source.name}\n" for source in sources))
    print(json.dumps({"stage": "model_packaged", "checkpoints": 5}))


def main() -> None:
    ensure_reproducible_process()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    calibration = commands.add_parser("calibrar")
    for name in ("external-dir", "protocolo", "detector", "salida"):
        calibration.add_argument("--" + name, type=Path, required=True)
    calibration.add_argument("--exclude-matches", nargs="*", required=True)
    train = commands.add_parser("entrenar")
    train.add_argument("--calibracion", type=Path, required=True)
    validation = commands.add_parser("validar")
    validation.add_argument("--external-dir", type=Path, required=True)
    validation.add_argument("--detector", type=Path, required=True)
    for command in (train, validation):
        command.add_argument("--protocolo", type=Path, required=True)
        command.add_argument("--salida-dir", type=Path, required=True)
        command.add_argument("--procesos", type=int, required=True)
        command.add_argument("--device", choices=("cpu", "0"), required=True)
    prediction = commands.add_parser("predecir")
    prediction.add_argument("--pesos-dir", type=Path, required=True)
    prediction.add_argument("--salida", type=Path, required=True)
    prediction.add_argument("--device", choices=("cpu", "0"), required=True)
    comparison = commands.add_parser("comparar")
    for name in ("salida-dir", "t1w-dir", "salida"):
        comparison.add_argument("--" + name, type=Path, required=True)
    package = commands.add_parser("empaquetar")
    package.add_argument("--pesos-dir", type=Path, required=True)
    package.add_argument("--salida-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "calibrar":
        guard_training()
        protocol = read_protocol(args.protocolo)
        value = external.calibrate(args.external_dir, Path(onside_paths.ROOT) / "annotations/ball/reviews", protocol,
                                   external.sha256(args.detector), set(args.exclude_matches))
        save_json(args.salida, value)
        print(json.dumps({"stage": "external_calibration", "tracker": value["tracker"]["name"], "share": value["share"]}))
    elif args.command == "entrenar":
        select_and_train(args.salida_dir, args.protocolo, args.calibracion,
                         list(pr.rp.pff_split.DESARROLLO), args.procesos, device_of(args.device))
    elif args.command == "validar":
        validate_folds(args.salida_dir, args.external_dir, args.protocolo, args.detector, args.procesos, device_of(args.device))
    elif args.command == "predecir":
        predict(args.pesos_dir, args.salida, device_of(args.device))
    elif args.command == "comparar":
        compare_t1w(args.salida_dir, args.t1w_dir, args.salida)
    else:
        package_model(args.pesos_dir, args.salida_dir)
