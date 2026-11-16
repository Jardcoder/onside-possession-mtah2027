"""Border detector for T1w: nested evaluation with the entry rule, and the final detector.

    ./run.sh bordes [--procesos P] [--salida-modelos DIR] [--sobrescribir]

T1w decodes with one price for changing state everywhere. This script trains a LightGBM detector of possession edges on the T1w feature matrix
(every column but the ball-presence family, `borde.border_columns`) and decodes with a cheaper change near its peaks (`borde.decode`). It reads only
caches that `./run.sh modelo`, `transfer` and `ronda5` leave under ONSIDE_OUT: the features and metadata of g6h / g8h and of the hole-free g6 / g8, the out of
fold probabilities of T1w, its boosters and `lambda_final_t1w.json`. It does not touch the 24 held-out matches or the challenge data.

The PFF labels (`gt_label` of the cache metadata) are read on purpose, as when the T1w boosters are trained: to build the target of the detector (a
frame is positive if a true edge is at most `borde.TARGET_TOL` frames away) and to score. They never enter a feature.

1. Detector per outer fold (`ronda_pff.FOLDS`): trained on the matches of the other four folds, both balls; scored on the 8 matches of its fold.
2. Setting (threshold of the peaks, f, k, g) from a grid of 720, chosen nested like lambda (`barrido_lambda.nested_select`): for each fold with
   the matches of the other folds, then read on its own matches. The entry rule `ronda_pff.verdict` judges these out of fold deltas against T1w,
   on g6h and g8h, with the same lambda per fold that T1w uses.
3. Controls. (a) The same settings with the detector probability shifted 60 s (same number of peaks, wrong place): it must not gain. (b) The same detector and
   settings on the balls without injected holes (g6, g8, T1w out of fold on them): in g6h / g8h the holes are anchored to real dead-ball stops, so a detector
   that had learnt "a hole starts or ends, so there is an edge" would lose its gain there.
4. Final detector: refit on all 40 matches with the setting that is best over all of them; weights and metadata (SHA-256, features, setting, hashes of the
   lambda file and of the T1w boosters) go to models/. Existing weights are not overwritten unless `--sobrescribir` is given.

The grid was widened three times on these same 40 matches: from 32 to 192 settings after a first run put the best one on its edge, from 192 to 480
because the chosen setting sat on the edge in threshold, f and k, and from 480 to 720 (f 0.9 and 1.0) because f was still on its edge. With f = 1 a
change near a peak costs the plain lambda and only the rest of the match is dearer (g). That is why the held-out confirmation matters.
"""
import argparse
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import onside_paths  # noqa: E402
onside_paths.add_challenge_to_path()
import barrido_lambda as bl  # noqa: E402
import borde  # noqa: E402
import evaluar_pff as ep  # noqa: E402
import modelo as mo  # noqa: E402
import modelo_pff as mp_  # noqa: E402
import modelo_transfer as mt  # noqa: E402
import pff_split  # noqa: E402
import possession_metrics as pm  # noqa: E402
import predecir_reto_m1 as m1  # noqa: E402
import ronda_pff as rp  # noqa: E402

BALLS = mt.H_BALLS                                       # ("g6h", "g8h")
AXES = {"threshold": (0.05, 0.075, 0.1, 0.15, 0.2, 0.3), "f": (1.0, 0.9, 0.75, 0.5, 0.25, 0.1), "k": (1, 2, 3, 4, 6), "g": (1.25, 1.5, 2.0, 2.5)}
GRID = [dict(zip(AXES, v)) for v in itertools.product(*AXES.values())]           # 720 settings {"threshold", "f", "k", "g"}
LGB_PARAMS = {"objective": "binary", "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 100, "feature_fraction": 0.8,
              "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "seed": 20261108, "deterministic": True,
              "force_row_wise": True, "verbosity": -1}
ROUNDS = 300
NUM_THREADS = 8                                          # fixed: the model text records it, so the SHA-256 of the weights does not depend on the machine
SHIFT_FRAMES = 60 * 25                                   # the control moves the detector probability 60 s
LAMBDA_FILE = m1.MODELS["T1w"][1]
REPORT = os.path.join(onside_paths.OUT_DIR, "bordes_informe.txt")
_DETECTORS = {}
_T1W = []


# ------------------------------------------------------------------------------------------------ inputs

def file_sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def t1w_boosters_files():
    return [os.path.join(m1.MODELS["T1w"][0], "modelos", f"T1w_k{k}.txt") for k in range(len(rp.FOLDS))]


def lambda_of_matches(mids):
    """({match: (lam_ha, lam_d) of its outer fold}, folds that had no nested choice and use the final lambda)."""
    j = json.load(open(LAMBDA_FILE))
    final = tuple(j["lambda"])
    per_fold = {int(k): tuple(v) for k, v in j.get("chosen_by_fold", {}).items()}
    missing = [k for k in range(len(rp.FOLDS)) if k not in per_fold]
    return {m: per_fold.get(rp.fold_of(m), final) for m in mids}, missing


def missing_inputs(mids):
    paths = [LAMBDA_FILE] + t1w_boosters_files()
    for m in mids:
        for b in BALLS:
            paths += [m1.probs_file("T1w", b, m), mt.path_meta_t(m, b), mt.path_x_of(m, b), mp_.path_meta(m, mt.BASE_OF[b]), mt.path_x_of(m, mt.BASE_OF[b])]
    return sorted(set(p for p in paths if not os.path.exists(p)))


def keys_without_fold(mids, fold):
    """(match, ball) pairs of the matches that are not in `fold`: what the detector of that fold is trained on."""
    return [(m, b) for m in mids if m not in fold for b in BALLS]


def fold_detector(mid):
    """The detector of the outer fold that holds `mid`, which never saw it."""
    return _DETECTORS[rp.fold_of(mid)]


def t1w_boosters():
    if not _T1W:
        _T1W.extend(m1.boosters_of("T1w", cache=True))
    return _T1W


def load_match(mid, balon, lam, base=True):
    """Everything one (match, ball) needs: the detector features, its target, the T1w path and the labels to score. For a ball with injected holes the
    out of fold probabilities of T1w come from the cache; for the hole-free ball (g6 / g8) from the T1w booster of the fold of the match."""
    z = mt.meta_of(mid, balon)
    names = list(z["names"])
    gt = np.asarray(z["gt_label"]).astype(np.int8)
    per = np.asarray(z["per"])
    X = np.load(mt.path_x_of(mid, balon), mmap_mode="r")
    if balon in BALLS:
        P = np.load(m1.probs_file("T1w", balon, mid)).astype(float)
    else:
        P = mo.predict_proba(t1w_boosters()[rp.fold_of(mid)], np.asarray(X), names)
    if not (len(P) == len(gt) == len(X) == len(per)):
        raise ValueError(f"lengths differ in {mid} {balon}")
    logp = np.log(np.clip(P, 1e-9, 1.0))
    runs = pm.valid_runs(gt)
    d = {"names": names, "gt": gt, "per": per, "logp": logp, "runs": runs, "calif": gt != borde.X,
         "y": borde.near(borde.transitions(gt, runs), borde.TARGET_TOL, len(gt)),
         "F": np.asarray(X[:, borde.border_columns(names)], np.float32), "minutes": sum(b - a for a, b in runs) / 25 / 60}
    d["base"] = mo.viterbi(logp, per, [lam[0]], [lam[1]])[0] if base else None
    return d


def training_rows(args):
    """Positive frames plus a NEG_FRACTION sample of the negatives (weight 1 / NEG_FRACTION), graded frames only."""
    mid, balon, lam = args
    d = load_match(mid, balon, lam, base=False)
    rng = np.random.default_rng(int(mid) * 2 + (balon == "g8h"))
    pos = np.flatnonzero(d["calif"] & d["y"])
    neg = np.flatnonzero(d["calif"] & ~d["y"])
    neg = neg[rng.random(len(neg)) < borde.NEG_FRACTION]
    idx = np.sort(np.concatenate([pos, neg]))
    return d["F"][idx], d["y"][idx].astype(np.int8), np.where(d["y"][idx], 1.0, 1.0 / borde.NEG_FRACTION), d["names"]


def train_detector(F, y, w):
    import lightgbm as lgb
    return lgb.train(dict(LGB_PARAMS, num_threads=NUM_THREADS), lgb.Dataset(F, label=y, weight=w), num_boost_round=ROUNDS)


# ------------------------------------------------------------------------------------------------ evaluation

def init_worker(models):
    import lightgbm as lgb
    for k, text in models.items():
        _DETECTORS[k] = lgb.Booster(model_str=text)


def paths_of_grid(p, logp, per, lam):
    """(len(GRID), N) int8 paths of every setting of the grid for the detector probability p: the same multipliers and the same Viterbi as `borde.decode`."""
    return borde.viterbi_variable(logp, per, lam[0], lam[1], borde.multipliers(p, GRID, per))


def spell_f1(d, codes):
    return pm.spell_scores(d["gt"], np.asarray(codes, np.int8), d["runs"])["mean_f1"]


def grid_deltas(d, p, lam):
    """(len(GRID),) spellF1 of every setting minus the one of T1w, and the possessions per minute of every setting."""
    f_base = spell_f1(d, d["base"])
    paths = paths_of_grid(p, d["logp"], d["per"], lam)
    return (np.array([spell_f1(d, x) - f_base for x in paths]),
            np.array([len(pm.possession_spells(x, d["runs"])) for x in paths]) / d["minutes"], f_base)


def evaluate_job(args):
    mid, balon, lam = args
    det = fold_detector(mid)
    d = load_match(mid, balon, lam)
    p = det.predict(d["F"], num_threads=1)
    delta, spells, f_base = grid_deltas(d, p, lam)
    out = {"f1_base": f_base, "delta": delta, "spells_min": spells,
           "spells_min_base": len(pm.possession_spells(d["base"], d["runs"])) / d["minutes"],
           "spells_min_gt": len(pm.possession_spells(d["gt"], d["runs"])) / d["minutes"],
           "peaks_min": np.array([len(borde.peaks(p, u, borde.PEAK_WINDOW)) for u in AXES["threshold"]]) / d["minutes"]}
    out["delta_shifted"] = grid_deltas(d, np.roll(p, SHIFT_FRAMES), lam)[0]
    e = load_match(mid, mt.BASE_OF[balon], lam)                                                       # the same match and ball without injected holes
    if not (np.array_equal(e["gt"], d["gt"]) and np.array_equal(e["per"], d["per"]) and e["names"] == d["names"]):
        raise ValueError(f"labels, periods or columns differ between {balon} and {mt.BASE_OF[balon]} in {mid}")
    out["delta_noholes"], _, out["f1_base_noholes"] = grid_deltas(e, det.predict(e["F"], num_threads=1), lam)
    return out


def nested_block(title, nested, out, rule=True, balls=BALLS):
    """Report block of out of fold deltas against T1w (mean, wins / losses, CI95, median per ball) and, for the real run, the entry rule."""
    s = {b: ep.delta_stats(nested[:, j]) for j, b in enumerate(BALLS)}
    out.append(title)
    out.append(rp.HEADER)
    for b, label in zip(BALLS, balls):
        out.append(rp.fmt_stats(f"T1w + borders {label}", s[b]))
    ok = None
    if rule:
        conds, ok = rp.verdict(s["g6h"], s["g8h"])
        for c, o in conds:
            out.append(f"  {'OK  ' if o else 'FAIL'} {c}")
        out.append(f"  entry rule: {'PASSES' if ok else 'DOES NOT PASS'}")
    return s, ok


# ------------------------------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procesos", type=int, default=min(9, os.cpu_count() or 1), help="worker processes (default: up to 9)")
    ap.add_argument("--salida-modelos", default=borde.MODELS_DIR, help="where the final weights and their metadata are written (default: ONSIDE_MODELS_DIR, ./models)")
    ap.add_argument("--sobrescribir", action="store_true", help="overwrite weights that already exist in the output folder")
    args = ap.parse_args()
    model_path = os.path.join(args.salida_modelos, borde.MODEL_FILE)
    meta_path = os.path.join(args.salida_modelos, borde.META_FILE)
    if (os.path.exists(model_path) or os.path.exists(meta_path)) and not args.sobrescribir:
        sys.exit(f"{args.salida_modelos} already holds detector weights: pass --sobrescribir to replace them, or --salida-modelos DIR to write elsewhere")
    os.environ.setdefault("OMP_NUM_THREADS", "1")        # one BLAS / OpenMP thread per worker (the workers inherit it); LightGBM training gets its threads explicitly
    t0 = time.time()
    mids = list(pff_split.DESARROLLO)
    assert not set(mids) & set(pff_split.RESERVADOS), "held-out matches must not be used"
    missing = missing_inputs(mids)
    if missing:
        sys.exit(f"{len(missing)} input files missing, e.g. {missing[0]} (run ./run.sh modelo, transfer and ronda5 first)")
    lam, no_lambda = lambda_of_matches(mids)
    jobs = [(m, b, lam[m]) for m in mids for b in BALLS]
    out = [f"BORDES: T1w + border detector, {len(mids)} development matches, balls {', '.join(BALLS)}",
           f"lambda per outer fold from {os.path.basename(LAMBDA_FILE)}" + (f" (folds without a nested choice, final lambda used: {no_lambda})" if no_lambda else ""),
           f"settings: {len(GRID)} ({', '.join(f'{k} {v}' for k, v in AXES.items())}); peaks window {borde.PEAK_WINDOW} frames; target tolerance {borde.TARGET_TOL} frames"]

    with mp.get_context("fork").Pool(args.procesos) as pool:                 # fork before LightGBM has started any OpenMP thread
        rows = pool.map(training_rows, jobs, chunksize=1)
    names = rows[0][3]
    assert all(r[3] == names for r in rows), "the feature layout differs between matches"
    border_names = [names[i] for i in borde.border_columns(names)]
    out.append(f"detector features: {len(border_names)} of the {len(names)} T1w columns (ball-presence family left out)")
    data = {(m, b): r[:3] for (m, b, _), r in zip(jobs, rows)}
    del rows

    models = {}
    for k, fold in enumerate(rp.FOLDS):
        keys = keys_without_fold(mids, fold)
        F, y, w = (np.concatenate([data[key][i] for key in keys]) for i in range(3))
        models[k] = train_detector(F, y, w).model_to_string()
        out.append(f"fold {k}: detector trained on {len(keys) // len(BALLS)} matches, {len(y)} rows, {int(y.sum())} positives")
    out.append(f"time to train the 5 detectors: {time.time() - t0:.0f} s")

    with mp.get_context("spawn").Pool(args.procesos, initializer=init_worker, initargs=(models,)) as pool:
        res = pool.map(evaluate_job, jobs, chunksize=1)
    res = dict(zip([(m, b) for m, b, _ in jobs], res))
    n = len(mids)
    stack = lambda key: np.stack([np.stack([res[(m, b)][key] for b in BALLS]) for m in mids])           # (n, 2, K)
    delta, shifted, noholes = stack("delta"), stack("delta_shifted"), stack("delta_noholes")
    fold_of = np.array([rp.fold_of(m) for m in mids])
    chosen, nested = bl.nested_select(delta, fold_of)                                              # (n, 2) out of fold deltas against T1w
    at_chosen = lambda a: np.stack([a[i, :, chosen[int(fold_of[i])]] for i in range(n)])             # each match at the setting of its own fold

    out.append("")
    for b in BALLS:
        out.append(f"T1w spellF1 on {b}: {np.mean([res[(m, b)]['f1_base'] for m in mids]):.4f}, on {mt.BASE_OF[b]} (no injected holes): "
                   f"{np.mean([res[(m, b)]['f1_base_noholes'] for m in mids]):.4f} (same lambda per fold)")
    s, ok = nested_block("\nDELTA spellF1 AGAINST T1w, setting chosen nested (out of fold):", nested, out)
    nested_block("\nCONTROL (a): same settings, detector shifted 60 s (same number of peaks, wrong place; it must not gain):", at_chosen(shifted), out, rule=False)
    nested_block("\nCONTROL (b): same detector and settings on the balls without injected holes (g6 / g8; T1w out of fold on them):", at_chosen(noholes), out,
                 rule=False, balls=[mt.BASE_OF[b] for b in BALLS])
    out.append("\nSetting chosen by each outer fold: " + "; ".join(f"fold {k}: {GRID[chosen[k]]}" for k in sorted(chosen)))
    best, obj = mp_.pick_lambda(delta)
    cfg = dict(GRID[best])
    out.append(f"Setting best over all {n} matches (out of fold detector, so optimistic): {cfg}  min of the two ball means {obj[best]:+.4f}")
    edges = [name for name, vals in AXES.items() if cfg[name] in (min(vals), max(vals))]
    out.append(f"  on the edge of the grid in: {', '.join(edges) if edges else 'none'}")
    every = [(m, b) for m in mids for b in BALLS]
    out.append(f"possessions per minute: T1w {np.mean([res[k]['spells_min_base'] for k in every]):.2f}, "
               f"with borders {np.mean([res[k]['spells_min'][best] for k in every]):.2f}, truth {np.mean([res[k]['spells_min_gt'] for k in every]):.2f}")
    peaks_ref = float(np.mean([res[k]["peaks_min"][AXES["threshold"].index(cfg["threshold"])] for k in every]))
    out.append(f"detector peaks per minute at that threshold: {peaks_ref:.2f}")

    final = train_detector(*(np.concatenate([data[(m, b)][i] for m in mids for b in BALLS]) for i in range(3)))
    os.makedirs(args.salida_modelos, exist_ok=True)
    final.save_model(model_path + ".tmp")
    os.replace(model_path + ".tmp", model_path)
    import lightgbm
    meta = {"model": borde.MODEL_FILE, "model_sha256": borde.sha256_text(model_path), "lightgbm": lightgbm.__version__,
            "feature_names": border_names, "config": cfg, "peak_window": borde.PEAK_WINDOW, "target_tolerance": borde.TARGET_TOL,
            "negative_fraction": borde.NEG_FRACTION, "rounds": ROUNDS, "params": dict(LGB_PARAMS, num_threads=NUM_THREADS),
            "trained_on": f"{n} PFF development matches, balls {', '.join(BALLS)}", "reference_peaks_per_min": peaks_ref,
            "lambda_file_sha256": file_sha256(LAMBDA_FILE), "t1w_boosters_sha256": [file_sha256(p) for p in t1w_boosters_files()],
            "nested": {b: {k: s[b][k] for k in ("mean", "median", "lo", "hi", "wins", "losses", "n")} for b in BALLS},
            "entry_rule_passes": bool(ok)}
    with open(meta_path + ".tmp", "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(meta_path + ".tmp", meta_path)
    out.append(f"\nfinal detector refit on all {n} matches -> {model_path}")
    out.append(f"  SHA-256 {meta['model_sha256']}; metadata {meta_path}")
    out.append(f"\ntotal time {time.time() - t0:.0f} s")
    text = "\n".join(out)
    print(text)
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
