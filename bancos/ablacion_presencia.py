"""Ablation of the ball-presence family for the border detector: does leaving it out hold on PFF alone?

    ./run.sh ablacion-presencia [--procesos P]

The detector of `./run.sh bordes` reads the T1w feature matrix without the ball-presence family (`borde.PRESENCE` and its windows). This script trains, per
outer fold, two detectors on the same training rows (same matches, folds, sampling, seed and rounds as `ronda_bordes`): `without` reads the border columns
(what `models/detector_bordes.*` use, so it must reproduce the nested result of `./run.sh bordes`) and `with` reads every T1w column. Each variant picks its setting
nested on the same grid (`barrido_lambda.nested_select`), and the report gives

1. the out of fold delta of each variant against T1w on g6h and g8h, with the entry rule `ronda_pff.verdict`;
2. the paired per-match difference `with - without` of those nested deltas (mean, wins / losses, CI95, median per ball).

It reads only the PFF development caches that `./run.sh bordes` reads (the 24 held-out matches and the challenge data are not touched) and writes only
ONSIDE_OUT/bordes_ablacion_presencia.txt. It never writes models/. The PFF labels are read on purpose, as in `ronda_bordes`, to build the target and to score.
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ronda_bordes as rb  # noqa: E402  (sets the paths of src/ and of the challenge code)
import barrido_lambda as bl  # noqa: E402
import borde  # noqa: E402
import evaluar_pff as ep  # noqa: E402
import modelo_transfer as mt  # noqa: E402
import onside_paths  # noqa: E402
import pff_split  # noqa: E402
import ronda_pff as rp  # noqa: E402

VARIANTS = ("without", "with")
REPORT = os.path.join(onside_paths.OUT_DIR, "bordes_ablacion_presencia.txt")
_DETECTORS = {}


def variant_columns(names):
    """{variant: column indices of the T1w feature matrix}: `without` = the border columns of the shipped detector, `with` = every column."""
    return {"without": borde.border_columns(names), "with": np.arange(len(names), dtype=np.int64)}


def load_full(mid, balon, lam, base=True):
    """`ronda_bordes.load_match` plus `F_all`: every T1w column, float32 (so `F_all[:, border_columns]` equals its `F`)."""
    d = rb.load_match(mid, balon, lam, base=base)
    d["F_all"] = np.asarray(np.load(mt.path_x_of(mid, balon), mmap_mode="r"), np.float32)
    return d


def training_rows(args):
    """The rows of `ronda_bordes.training_rows` (same sampling and seed), with every T1w column."""
    mid, balon, lam = args
    d = load_full(mid, balon, lam, base=False)
    rng = np.random.default_rng(int(mid) * 2 + (balon == "g8h"))
    pos = np.flatnonzero(d["calif"] & d["y"])
    neg = np.flatnonzero(d["calif"] & ~d["y"])
    neg = neg[rng.random(len(neg)) < borde.NEG_FRACTION]
    idx = np.sort(np.concatenate([pos, neg]))
    return d["F_all"][idx], d["y"][idx].astype(np.int8), np.where(d["y"][idx], 1.0, 1.0 / borde.NEG_FRACTION), d["names"]


def init_worker(models):
    import lightgbm as lgb
    for key, text in models.items():
        _DETECTORS[key] = lgb.Booster(model_str=text)


def evaluate_job(args):
    """(variant -> (len(GRID),) spellF1 deltas against T1w) for one (match, ball), each variant with the detector of the fold of the match."""
    mid, balon, lam = args
    d = load_full(mid, balon, lam)
    cols = variant_columns(d["names"])
    k = rp.fold_of(mid)
    return {v: rb.grid_deltas(d, _DETECTORS[(v, k)].predict(d["F_all"][:, cols[v]], num_threads=1), lam)[0] for v in VARIANTS}


def paired_block(diff, out):
    """Report block of the paired per-match difference `with - without` (n, 2), one row per ball."""
    out.append(rp.HEADER)
    s = {b: ep.delta_stats(diff[:, j]) for j, b in enumerate(rb.BALLS)}
    for b in rb.BALLS:
        out.append(rp.fmt_stats(f"with - without {b}", s[b]))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procesos", type=int, default=min(9, os.cpu_count() or 1), help="worker processes (default: up to 9)")
    args = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    t0 = time.time()
    mids = list(pff_split.DESARROLLO)
    assert not set(mids) & set(pff_split.RESERVADOS), "held-out matches must not be used"
    missing = rb.missing_inputs(mids)
    if missing:
        sys.exit(f"{len(missing)} input files missing, e.g. {missing[0]} (run ./run.sh modelo, transfer and ronda5 first)")
    lam, no_lambda = rb.lambda_of_matches(mids)
    jobs = [(m, b, lam[m]) for m in mids for b in rb.BALLS]

    with mp.get_context("fork").Pool(args.procesos) as pool:
        rows = pool.map(training_rows, jobs, chunksize=1)
    names = rows[0][3]
    assert all(r[3] == names for r in rows), "the feature layout differs between matches"
    cols = variant_columns(names)
    data = {(m, b): r[:3] for (m, b, _), r in zip(jobs, rows)}
    del rows
    out = [f"ABLATION OF THE BALL-PRESENCE FAMILY: border detector without / with it, {len(mids)} development matches, balls {', '.join(rb.BALLS)}",
           f"lambda per outer fold from {os.path.basename(rb.LAMBDA_FILE)}" + (f" (folds without a nested choice, final lambda used: {no_lambda})" if no_lambda else ""),
           f"settings: {len(rb.GRID)} ({', '.join(f'{k} {v}' for k, v in rb.AXES.items())})",
           f"columns: without {len(cols['without'])}, with {len(cols['with'])} (ball-presence family: {len(cols['with']) - len(cols['without'])})"]

    models = {}
    for k, fold in enumerate(rp.FOLDS):
        keys = rb.keys_without_fold(mids, fold)
        F, y, w = (np.concatenate([data[key][i] for key in keys]) for i in range(3))
        for v in VARIANTS:
            models[(v, k)] = rb.train_detector(np.ascontiguousarray(F[:, cols[v]]), y, w).model_to_string()
        del F
    out.append(f"time to train the 10 detectors: {time.time() - t0:.0f} s")

    with mp.get_context("spawn").Pool(args.procesos, initializer=init_worker, initargs=(models,)) as pool:
        res = pool.map(evaluate_job, jobs, chunksize=1)
    res = dict(zip([(m, b) for m, b, _ in jobs], res))
    fold_of = np.array([rp.fold_of(m) for m in mids])
    nested, passes = {}, {}
    for v in VARIANTS:
        delta = np.stack([np.stack([res[(m, b)][v] for b in rb.BALLS]) for m in mids])        # (n, 2, K)
        chosen, nested[v] = bl.nested_select(delta, fold_of)
        _, passes[v] = rb.nested_block(f"\nDELTA spellF1 AGAINST T1w, detector {v} the ball-presence family, setting chosen nested (out of fold):", nested[v], out)
        out.append("Setting chosen by each outer fold: " + "; ".join(f"fold {k}: {rb.GRID[chosen[k]]}" for k in sorted(chosen)))

    out.append("\nPAIRED DIFFERENCE PER MATCH (with - without), nested deltas of each variant:")
    s = paired_block(nested["with"] - nested["without"], out)
    gain = min(s[b]["mean"] for b in rb.BALLS)
    out.append(f"  min of the two ball means: {gain:+.4f}; CI95 lower bound above 0 in both balls: {all(s[b]['lo'] > 0 for b in rb.BALLS)}")
    out.append(f"  entry rule of each variant against T1w: without {'PASSES' if passes['without'] else 'DOES NOT PASS'}, with {'PASSES' if passes['with'] else 'DOES NOT PASS'}")
    out.append(f"\ntotal time {time.time() - t0:.0f} s")
    text = "\n".join(out)
    print(text)
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
