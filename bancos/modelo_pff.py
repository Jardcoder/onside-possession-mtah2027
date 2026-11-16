"""M1 on the PFF development bank: LightGBM per frame + Viterbi, nested validation 5 x 8 against the current bases.

    challenge/.venv/bin/python bancos/modelo_pff.py [--paso N] [--procesos P] [--hilos T] [--rehacer] [--solo-resumen]
    challenge/.venv/bin/python bancos/modelo_pff.py --cronometro 5 10        # time ONE training of 32 matches per step, nothing else

Method side: src/modelo.py (features, team swap, classifier, Viterbi). This file is the evaluation side.

Cache (src/out/modelo_cache/<CODE_VERSION>/): per (match, ball) the feature matrix `<mid>__<ball>.npy` and `<mid>__<ball>.npz` (label codes
for training, the two ground truths for scoring, B3 / B3 + absorption 0.8 s / B3 + absorption 1.6 s for the comparisons and the period),
both with the fingerprint of the code that produced them (the method modules of bancos/ronda_pff.fingerprint + src/modelo.py + the builder
functions of this file + the loader pff.py + the NumPy version); a stale file is recomputed (--rehacer recomputes everything: features, inner,
outer and importances). The inner / outer / importance results carry the fingerprint of the training too (every function that trains, selects,
predicts or scores, step, hyper-parameters, grids, seeds, loader, LightGBM and NumPy versions). Every cache write is atomic (temp file in the
same directory + os.replace).

Nested validation (FOLDS of ronda_pff): for each outer fold k the training set is the other 32 matches. The 4 inner folds are
those other 4 FOLDS (8 matches each): for each inner fold j one model is trained with the 24 remaining matches (g6 and g8 together, both
team views) and predicts the 8 matches of j in g6 and g8 -> out-of-sample probabilities -> Viterbi on the 6 x 5 grid
lam_ha in {0, 2, 4, 8, 16, 32} x lam_d in {0, 2, 4, 8, 16} -> mean spellF1 (label) per ball over the 32 matches ->
the (lam_ha, lam_d) maximising min(mean g6, mean g8) (ties within TIE: the first of the grid, lam_ha-major). Then one model with the
32 matches predicts the 8 of fold k in g6, g8, g4 and raw (g4 and raw only as stress: the model never sees them), decoded with the chosen
(lam_ha, lam_d). 25 trainings in total. Inside every training 4 of its own matches (the first 4 by SHA-256(ES_SALT + id)) are set aside for
early stopping; the rows of the matches of the fold being predicted never touch training, early stopping or the choice of lambda.
Training frames: one of every STEP frames of each match (offset 0 for g6 and STEP // 2 for g8), original and swapped view.
Comparisons per match, out of fold: M1 - base vigente (B3 + absorption tau 1.6 s, the one that decides), M1 - (B3 + tau 0.8 s, as
preregistered) and M1 - B3, with the entry rule through ronda_pff.verdict.
"""
import argparse
import csv
import hashlib
import inspect
import itertools
import json
import multiprocessing as mp
import os
import sys
import time

import lightgbm
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import onside_paths  # noqa: E402
onside_paths.add_challenge_to_path()
import absorcion as ab  # noqa: E402
import evaluar_pff as ep  # noqa: E402
import modelo as mo  # noqa: E402
import pff_split  # noqa: E402
import possession_metrics as pm  # noqa: E402
import ronda_pff as rp  # noqa: E402

CODE_VERSION = "v1"
CACHE = os.path.join(onside_paths.OUT_DIR, "modelo_cache", CODE_VERSION)
REPORT = os.path.join(onside_paths.OUT_DIR, "modelo_pff_informe.txt")
BALLS = ("g6", "g8", "g4", "crudo")
TRAIN_BALLS = ("g6", "g8")
LAM_HA = (0, 2, 4, 8, 16, 32)
LAM_D = (0, 2, 4, 8, 16)
GRID = list(itertools.product(LAM_HA, LAM_D))
GRID_HA = np.array([g[0] for g in GRID], float)
GRID_D = np.array([g[1] for g in GRID], float)
TAUS = (0.8, 1.6)                              # absorption of the two bases (0.8 preregistered in 8.23, 1.6 current after round 3)
STEP = 5
ES_SALT = "onside-modelo-es-v1"
ES_MATCHES = 4
TIE = ep.TIE
METHODS = ("GT", "B3", "B3_t0.8", "B3_t1.6", "M1", "M1_raw")
FIELDS = ["partido", "balon", "gt", "metodo", "spell_f1", "n_gt", "n_pred", "c05", "c10", "c20", "live_gt", "lam_ha", "lam_d"]


# ------------------------------------------------------------------------------------------------ fingerprints

def _src(*objs):
    h = hashlib.sha256()
    for o in objs:
        h.update(rp.source_of(o))
    return h


_FP = {}


# Keys of the match dict (rp.load_match: pff_split.load_dev + clip_periods) that the cached features and meta are built from:
# build_one / build_one_h / dominio.build_h read them, through rp.ball_variant, rp.build_method_ctx, mo.build_features, ih.inject_holes and rp.ev.make_gt.
# `uncertain`, `clipped` and `event_f` are loaded too but never reach a cached feature or meta, so a change in them must not invalidate the cache.
LOADER_KEYS = ("ball_xy", "players_slots", "period", "label", "label_poss", "pitch")
LOADER_MEMO = os.path.join(onside_paths.OUT_DIR, "modelo_cache", "loader_content.json")


def _feed(h, name, v):
    """Feeds a name and a value (array, dict of arrays or scalar) into the hash: dtype and shape included, NaN bit patterns as the loader made them."""
    h.update(name.encode() + b"\0")
    if isinstance(v, dict):
        for k in sorted(v):
            _feed(h, f"{name}.{k}", v[k])
        return
    a = np.ascontiguousarray(np.asarray(v))
    h.update(f"{a.dtype.str}{a.shape}".encode() + b"\0")
    h.update(a.tobytes())


def content_hash(mids=None, load=None):
    """SHA-256 of the CONTENT the pipeline reads from the loader: LOADER_KEYS of rp.load_match for every development match, in the fixed order of
    pff_split.DESARROLLO. Keys the pipeline does not read (and keys the loader adds later) do not enter. Never loads a held-out match."""
    mids = list(pff_split.DESARROLLO if mids is None else mids)
    assert not set(mids) & set(pff_split.RESERVADOS)
    load = load or rp.load_match
    h = hashlib.sha256()
    h.update(repr(LOADER_KEYS).encode())
    for mid in mids:
        d = load(mid)
        h.update(b"match\0" + mid.encode() + b"\0")
        for k in LOADER_KEYS:
            _feed(h, k, d[k])
    return h.hexdigest()


def loader_hash():
    """Fingerprint of the loader as the pipeline sees it: content_hash of the 40 development matches. It is memoised in LOADER_MEMO under the
    SHA-256 of pff.py, so the 40 matches are loaded once per version of the file; a new pff.py that only ADDS keys recomputes
    the content hash once and gets the same value."""
    path = os.path.join(pff_split.LOADER_DIR, "pff.py")
    if not os.path.exists(path):
        return "pff.py missing"
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    code = _src(rp.load_match, ep.clip_periods).hexdigest() + repr(LOADER_KEYS)      # what turns the file into the hashed view
    try:
        memo = json.load(open(LOADER_MEMO))
    except (OSError, ValueError):
        memo = {}
    if memo.get(sha, {}).get("code") != code:
        memo[sha] = {"code": code, "hash": content_hash()}
        os.makedirs(os.path.dirname(LOADER_MEMO), exist_ok=True)
        atomic_json(LOADER_MEMO, memo)
    return memo[sha]["hash"]


def fp_features():
    """Code that decides the cached features and meta (computed once per process: the files may change while a long run goes on)."""
    if "f" not in _FP:
        h = _src(mo, build_match, build_one, label_codes)
        h.update(rp.fingerprint("R2", "tau1.6", (1.6,)).encode())
        h.update(repr((rp.BALLS, TAUS, mo.WINDOWS_S, mo.SPEED_HALF_S, mo.CAP_S)).encode())
        h.update(loader_hash().encode())
        h.update(np.__version__.encode())
        _FP["f"] = h.hexdigest()
    return _FP["f"]


def model_functions():
    """Every function whose code decides the inner / outer / importance results: trains, selects, predicts, scores, or reads the training
    cache. Helpers outside this file that they call: mo.* and pm.* (modules already in fp_features), rp.spell_row (added by fp_models)."""
    return (train_model, load_rows, load_meta, path_x, path_meta, eval_stats, task_grid, task_final, predict_matches, pick_lambda,
            run_inner, run_outer, rp.spell_row)


def fp_models(step):
    h = hashlib.sha256()
    h.update(fp_features().encode())
    h.update(_src(*model_functions()).hexdigest().encode())
    h.update(repr((step, sorted(mo.LGB_PARAMS.items()), mo.MAX_ROUNDS, mo.EARLY_STOP, GRID, rp.FOLDS, ES_SALT, ES_MATCHES,
                   TRAIN_BALLS, BALLS, TIE, METHODS, FIELDS, rp.BINS, TAUS, loader_hash(), lightgbm.__version__, np.__version__)).encode())
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ atomic writes

def atomic_write(path, writer, mode="wb", **kw):
    """writer(file) into a temp file of the same directory, then os.replace: a crash never leaves a half-written `path`."""
    tmp = f"{path}.tmp{os.getpid()}"
    try:
        with open(tmp, mode, **kw) as f:
            writer(f)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def atomic_save(path, arr):
    atomic_write(path, lambda f: np.save(f, arr))


def atomic_savez(path, **arrays):
    atomic_write(path, lambda f: np.savez(f, **arrays))


def atomic_json(path, obj):
    atomic_write(path, lambda f: json.dump(obj, f), "w")


def atomic_csv(path, fp, fields, rows):
    def w(f):
        f.write(f"# {fp}\n")
        wr = csv.DictWriter(f, fields)
        wr.writeheader()
        wr.writerows(rows)
    atomic_write(path, w, "w", newline="")


# ------------------------------------------------------------------------------------------------ feature cache

def path_x(mid, balon):
    return os.path.join(CACHE, f"{mid}__{balon}.npy")


def path_meta(mid, balon):
    return os.path.join(CACHE, f"{mid}__{balon}.npz")


def label_codes(lab):
    """"H"/"A"/"D" -> 0/1/2 (the classifier's classes; same codes as possession_metrics for the three scored labels)."""
    lab = np.asarray(lab)
    out = np.full(len(lab), -1, np.int8)
    for k, c in mo.CODE.items():
        out[lab == k] = c
    assert (out >= 0).all(), "unexpected label"
    return out


def build_one(d, balon, fp):
    """Features + meta of one (match, ball): the tracking arrays go through the method pipeline; the GT only goes to the meta."""
    ball = rp.ball_variant(d, balon)
    m = rp.build_method_ctx(ball, d["players_slots"], d["period"], d["pitch"])
    b3, tf, _ = rp.pipeline(m)
    b3a = [rp.pipeline(m, r2=t)[0] for t in TAUS]
    X, names = mo.build_features(ball, d["players_slots"], d["period"], d["pitch"], b3, b3a[0], tf)
    atomic_save(path_x(d["mid"], balon), X)                                  # the meta (it carries the fingerprint) goes last
    atomic_savez(path_meta(d["mid"], balon), fp=fp, names=np.array(names), y=label_codes(d["label"]),
             gt_label=rp.ev.make_gt(d, "label"), gt_poss=rp.ev.make_gt(d, "label_poss"), per=d["period"].astype(np.int8),
             b3=label_codes(b3), b3_t08=label_codes(b3a[0]), b3_t16=label_codes(b3a[1]))


def feature_ok(mid, balon, fp):
    try:
        if not (os.path.exists(path_x(mid, balon)) and os.path.exists(path_meta(mid, balon))):
            return False
        return str(np.load(path_meta(mid, balon))["fp"]) == fp
    except (OSError, ValueError, KeyError):
        return False


def build_match(mid, fp, rehacer=False):
    """Every ball variant of one development match (loaded through the guard of pff_split)."""
    todo = [b for b in BALLS if rehacer or not feature_ok(mid, b, fp)]
    if not todo:
        return mid, 0
    d = rp.load_match(mid)
    d["mid"] = mid
    os.makedirs(CACHE, exist_ok=True)
    for b in todo:
        build_one(d, b, fp)
    return mid, len(todo)


def load_meta(mid, balon):
    z = np.load(path_meta(mid, balon))
    return {k: z[k] for k in z.files}


# ------------------------------------------------------------------------------------------------ training

def load_rows(mids, step, balls=TRAIN_BALLS):
    """(X, y) of the training rows of `mids`: one frame of every `step`, offset 0 for g6 and step // 2 for g8."""
    xs, ys = [], []
    for mid in mids:
        for i, b in enumerate(balls):
            off = (step // 2) * i
            xm = np.load(path_x(mid, b), mmap_mode="r")
            xs.append(np.ascontiguousarray(xm[off::step]))
            ys.append(np.load(path_meta(mid, b))["y"][off::step])
    return np.concatenate(xs), np.concatenate(ys)


def train_model(train_mids, step, threads=None, names=None):
    """(booster, seconds, n_rows): `train_mids` minus its 4 early-stopping matches, both views; early stopping on those 4 (both views)."""
    t0 = time.time()
    order = sorted(train_mids, key=lambda m: hashlib.sha256((ES_SALT + m).encode()).hexdigest())
    es, fit = order[:ES_MATCHES], order[ES_MATCHES:]
    Xtr, ytr = mo.augment(*load_rows(fit, step), names)
    Xva, yva = mo.augment(*load_rows(es, step), names)
    booster = mo.fit(Xtr, ytr, Xva, yva, threads=threads)
    return booster, time.time() - t0, len(ytr)


# ------------------------------------------------------------------------------------------------ evaluation (workers)

def eval_stats(gt, codes, runs):
    """(spellF1 mean over the IoUs, n GT spells, n predicted spells, [n spells < 0.5 / 1 / 2 s]) of a prediction without X."""
    sc = pm.spell_scores(gt, codes, runs)
    dur, _ = rp.spell_row(codes, runs)
    return sc["mean_f1"], sc["n_gt"], sc["n_pred"], [int((dur < t).sum()) for t in rp.BINS]


def task_grid(mid, balon, P):
    """spellF1 (label) of every (lam_ha, lam_d) of the grid on the out-of-sample probabilities P of one match and ball -> (K,)."""
    z = load_meta(mid, balon)
    gt = z["gt_label"]
    runs = pm.valid_runs(gt)
    paths = mo.viterbi(np.log(np.clip(P, 1e-9, 1.0)), z["per"], GRID_HA, GRID_D)
    return np.array([pm.spell_scores(gt, p, runs)["mean_f1"] for p in paths])


def task_final(mid, balon, P, lam):
    """Rows of the methods of METHODS for one match and ball, both ground truths."""
    z = load_meta(mid, balon)
    paths = mo.viterbi(np.log(np.clip(P, 1e-9, 1.0)), z["per"], [lam[0], 0], [lam[1], 0])
    preds = {"B3": z["b3"], "B3_t0.8": z["b3_t08"], "B3_t1.6": z["b3_t16"], "M1": paths[0], "M1_raw": paths[1]}
    rows = []
    for g, key in (("label", "gt_label"), ("label_poss", "gt_poss")):
        gt = z[key]
        runs = pm.valid_runs(gt)
        live = rp.spell_row(gt, runs)[1]
        for name in METHODS:
            codes = gt if name == "GT" else preds[name]
            f1, n_gt, n_pred, c = eval_stats(gt, codes, runs)
            rows.append({"partido": mid, "balon": balon, "gt": g, "metodo": name, "spell_f1": f1, "n_gt": n_gt, "n_pred": n_pred,
                         "c05": c[0], "c10": c[1], "c20": c[2], "live_gt": live, "lam_ha": lam[0], "lam_d": lam[1]})
    return rows


# ------------------------------------------------------------------------------------------------ nested run

def inner_path(k, j):
    return os.path.join(CACHE, f"inner_k{k}_j{j}.npz")


def outer_path(k):
    return os.path.join(CACHE, f"outer_k{k}.csv")


def importance_path(k):
    return os.path.join(CACHE, f"importance_k{k}.npz")


def times_path():
    return os.path.join(CACHE, "tiempos.json")


def load_if(path, fp):
    """The cache file when it carries the fingerprint, else None (npz: key `fp`; csv: first line `# fp`)."""
    if not os.path.exists(path):
        return None
    try:
        if path.endswith(".npz"):
            z = np.load(path)
            return {k: z[k] for k in z.files} if str(z["fp"]) == fp else None
        with open(path) as f:
            if f.readline().strip() != f"# {fp}":
                return None
            return list(csv.DictReader(f))
    except (OSError, ValueError, KeyError):
        return None


def add_time(key, seconds):
    t = json.load(open(times_path())) if os.path.exists(times_path()) else {}
    t[key] = seconds
    atomic_json(times_path(), t)


def predict_matches(booster, mids, balls, names):
    return {(m, b): mo.predict_proba(booster, np.load(path_x(m, b)), names).astype(np.float32) for m in mids for b in balls}


def pick_lambda(f1):
    """f1 (n_matches, 2, K) spellF1 of the matches of the 4 inner folds (g6, g8) -> (index in GRID, objective)."""
    obj = np.minimum(f1[:, 0, :].mean(0), f1[:, 1, :].mean(0))
    return int(np.flatnonzero(obj >= obj.max() - TIE)[0]), obj


def run_inner(pool, k, j, step, fp, threads, names, rehacer=False):
    cached = None if rehacer else load_if(inner_path(k, j), fp)
    if cached is not None:
        return cached["f1"], cached["mids"]
    train = [m for q, f in enumerate(rp.FOLDS) if q not in (k, j) for m in f]
    held = list(rp.FOLDS[j])
    booster, secs, nrows = train_model(train, step, threads, names)
    probs = predict_matches(booster, held, TRAIN_BALLS, names)
    res = pool.starmap(task_grid, [(m, b, probs[(m, b)]) for m in held for b in TRAIN_BALLS])
    f1 = np.array(res).reshape(len(held), len(TRAIN_BALLS), len(GRID))
    atomic_savez(inner_path(k, j), fp=fp, f1=f1, mids=np.array(held), rounds=booster.best_iteration, rows=nrows, seconds=secs)
    add_time(f"inner_k{k}_j{j}", secs)
    print(f"  inner k{k} j{j}: {nrows} rows, {booster.best_iteration} rounds, {secs:.0f}s", file=sys.stderr, flush=True)
    return f1, np.array(held)


def run_outer(pool, k, step, fp, threads, names, rehacer=False):
    if not rehacer and load_if(outer_path(k), fp) is not None and load_if(importance_path(k), fp) is not None:
        return
    train = [m for q, f in enumerate(rp.FOLDS) if q != k for m in f]
    f1s = []
    for j in range(len(rp.FOLDS)):
        if j != k:
            f1s.append(run_inner(pool, k, j, step, fp, threads, names, rehacer)[0])
    f1 = np.concatenate(f1s)
    idx, obj = pick_lambda(f1)
    lam = GRID[idx]
    booster, secs, nrows = train_model(train, step, threads, names)
    probs = predict_matches(booster, rp.FOLDS[k], BALLS, names)
    rows = []
    for r in pool.starmap(task_final, [(m, b, probs[(m, b)], lam) for m in rp.FOLDS[k] for b in BALLS]):
        rows += r
    gain = booster.feature_importance("gain", iteration=booster.best_iteration)
    atomic_savez(importance_path(k), fp=fp, gain=gain, names=np.array(names), lam=np.array(lam), obj=obj, rounds=booster.best_iteration,
             rows=nrows, inner_f1=f1)
    add_time(f"outer_k{k}", secs)
    atomic_csv(outer_path(k), fp, FIELDS, rows)                              # last: it is what marks the fold as done
    print(f"outer k{k}: lambda {lam}, objective {obj[idx]:.4f}, {nrows} rows, {booster.best_iteration} rounds, {secs:.0f}s",
          file=sys.stderr, flush=True)


def build_all(pool, rehacer):
    fp = fp_features()
    mids = list(pff_split.DESARROLLO)
    assert not set(mids) & set(pff_split.RESERVADOS)
    t0 = time.time()
    res = pool.starmap(build_match, [(m, fp, rehacer) for m in mids], chunksize=1)
    print(f"features: {sum(n for _, n in res)} (match, ball) files computed in {time.time() - t0:.0f}s", file=sys.stderr, flush=True)


def cronometro(steps, threads):
    names = list(np.load(path_meta(rp.FOLDS[0][0], "g6"))["names"])
    train = [m for q, f in enumerate(rp.FOLDS) if q != 0 for m in f]
    for s in steps:
        for label, mids in (("32 matches", train), ("24 matches", train[:24])):
            booster, secs, nrows = train_model(mids, s, threads, names)
            print(f"step {s:>2} | {label} | {nrows} rows | {booster.best_iteration} rounds | {secs:.0f}s | threads {threads}", flush=True)


# ------------------------------------------------------------------------------------------------ report

def read_rows(fp):
    rows = []
    for k in range(len(rp.FOLDS)):
        r = load_if(outer_path(k), fp)
        if r is None:
            raise SystemExit(f"stale or missing outer result {outer_path(k)} (run without --solo-resumen)")
        rows += r
    return rows


def f1_of(rows, metodo, balon, gt="label"):
    return {r["partido"]: float(r["spell_f1"]) for r in rows if r["metodo"] == metodo and r["balon"] == balon and r["gt"] == gt}


def delta(rows, a, b, balon, gt="label"):
    fa, fb = f1_of(rows, a, balon, gt), f1_of(rows, b, balon, gt)
    return np.array([fa[m] - fb[m] for m in pff_split.DESARROLLO])


def block_vs(rows, a, b, out):
    P = out.append
    s = {x: ep.delta_stats(delta(rows, a, b, x)) for x in rp.MAIN_BALLS}
    P(f"\nTabla 8.20 ({a} - {b}, delta spellF1 fuera de pliegue, n=40, GT label):\n{rp.HEADER}")
    for x in rp.MAIN_BALLS:
        P(rp.fmt_stats(f"{a} - {b}, {x}", s[x]))
    P(f"min(g6, g8) = {min(s['g6']['mean'], s['g8']['mean']):+.4f}")
    conds, ok = rp.verdict(s["g6"], s["g8"])
    for c, o in conds:
        P(f"  [{'x' if o else ' '}] {c}")
    P(f"VEREDICTO {a} contra {b}: {'ENTRA' if ok else 'NO ENTRA'} segun 8.20.1 aplicado literalmente")
    P(f"Informativo (sin veto):\n{rp.HEADER}")
    for x in ("g4", "crudo"):
        P(rp.fmt_stats(f"{a} - {b}, {x}", ep.delta_stats(delta(rows, a, b, x))))
    for x in rp.MAIN_BALLS:
        P(rp.fmt_stats(f"{a} - {b}, {x}, label_poss", ep.delta_stats(delta(rows, a, b, x, "label_poss"))))
    return ok


def report(step, fp, out):
    P = out.append
    rows = read_rows(fp)
    imp = [load_if(importance_path(k), fp) for k in range(len(rp.FOLDS))]
    if any(z is None for z in imp):
        raise SystemExit("stale or missing importance file (run without --solo-resumen)")
    times = json.load(open(times_path())) if os.path.exists(times_path()) else {}
    P("M1 (8.23): LightGBM por frame + Viterbi, 40 partidos de desarrollo, validacion anidada 5 x 8, spellF1 GT label, deltas fuera de pliegue.")
    P(f"Submuestreo de entrenamiento: 1 de cada {step} frames (g8 con desfase {step // 2}), vista original + vista con equipos intercambiados.")
    tr = {k: v for k, v in times.items() if k.startswith(("inner_", "outer_"))}
    P(f"Entrenamientos: {len(tr)} (4 internos + 1 externo por pliegue); tiempo de entrenamiento sumado {sum(tr.values()) / 60:.1f} min; "
      f"tiempo de pared de la corrida que los produjo (cache de rasgos incluida si se hizo en ella): {times.get('total_wall_s', float('nan')) / 60:.1f} min.")
    P(f"LightGBM: {dict(sorted(mo.LGB_PARAMS.items()))}; hasta {mo.MAX_ROUNDS} rondas, early stopping {mo.EARLY_STOP} sobre {ES_MATCHES} partidos del propio entrenamiento.")
    P("\n=== (lam_ha, lam_d) elegido por pliegue (maximiza min(media g6, media g8) con probabilidades internas fuera de muestra) ===")
    lams = []
    for k, z in enumerate(imp):
        lam = tuple(int(v) for v in z["lam"])
        lams.append(lam)
        edge = [n for n, v, grid in (("lam_ha", lam[0], LAM_HA), ("lam_d", lam[1], LAM_D)) if v in (grid[0], grid[-1])]
        P(f"  pliegue {k + 1}: {lam}   objetivo interno {np.max(z['obj']):.4f}   rondas {int(z['rounds'])}   filas {int(z['rows'])}"
          + (f"   BORDE de la rejilla: {', '.join(edge)}" if edge else ""))
    P(f"  -> {'ESTABLE (los 5 coinciden)' if len(set(lams)) == 1 else 'INESTABLE (difieren entre pliegues)'}; "
      f"lam_ha {sorted({l[0] for l in lams})}, lam_d {sorted({l[1] for l in lams})}")
    oks = {}
    for base in ("B3_t1.6", "B3_t0.8", "B3"):
        oks[base] = block_vs(rows, "M1", base, out)
    P("\nSin suavizado Viterbi (M1_raw: lam_ha = lam_d = 0, argmax por frame), informativo:")
    P(rp.HEADER)
    for base in ("B3_t1.6", "B3"):
        for x in rp.MAIN_BALLS:
            P(rp.fmt_stats(f"M1_raw - {base}, {x}", ep.delta_stats(delta(rows, "M1_raw", base, x))))
    P("\nspellF1 medio de los 40 partidos (label):")
    P(f"{'':<10}" + "".join(f"{b:>9}" for b in BALLS))
    for name in ("B3", "B3_t0.8", "B3_t1.6", "M1_raw", "M1"):
        P(f"{name:<10}" + "".join(f"{np.mean(list(f1_of(rows, name, b).values())):>9.4f}" for b in BALLS))
    P("\n=== Contadores (suma de los 40 partidos; por minuto = por minuto vivo del GT, como el diagnostico de semantica de ronda_pff) ===")
    for b in rp.MAIN_BALLS:
        gt_live = sum(float(r["live_gt"]) for r in rows if r["metodo"] == "GT" and r["balon"] == b and r["gt"] == "label")
        n_gt = sum(int(r["n_pred"]) for r in rows if r["metodo"] == "GT" and r["balon"] == b and r["gt"] == "label")
        P(f"-- balon {b}: GT {n_gt} posesiones, {gt_live:.1f} min vivos")
        P(f"{'serie':<10}{'posesiones':>11}{'pred/GT':>9}{'<0,5 s /min':>13}{'<1 s /min':>11}{'<2 s /min':>11}")
        for name in METHODS:
            rr = [r for r in rows if r["metodo"] == name and r["balon"] == b and r["gt"] == "label"]
            n = sum(int(r["n_pred"]) for r in rr)
            c = [sum(int(r[k]) for r in rr) / gt_live for k in ("c05", "c10", "c20")]
            P(f"{name:<10}{n:>11}{n / n_gt:>9.2f}{c[0]:>13.2f}{c[1]:>11.2f}{c[2]:>11.2f}")
    P("\n=== Top 20 rasgos por ganancia (media de los 5 modelos externos, normalizada a 100) ===")
    g = np.mean([z["gain"] / z["gain"].sum() for z in imp], axis=0) * 100
    names = list(imp[0]["names"])
    for i in np.argsort(-g)[:20]:
        P(f"  {names[i]:<34}{g[i]:>7.2f}")
    return oks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paso", type=int, default=STEP)
    ap.add_argument("--procesos", type=int, default=min(12, os.cpu_count() or 1))
    ap.add_argument("--hilos", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--rehacer", action="store_true")
    ap.add_argument("--solo-resumen", action="store_true")
    ap.add_argument("--cronometro", type=int, nargs="+", default=None)
    args = ap.parse_args()
    # the pool is forked before LightGBM / OpenMP starts any thread: the workers only run Viterbi and the scorer
    pool = mp.get_context("fork").Pool(args.procesos)
    t0 = time.time()
    if not args.solo_resumen:
        build_all(pool, args.rehacer)
        names = list(np.load(path_meta(rp.FOLDS[0][0], "g6"))["names"])
        if args.cronometro:
            cronometro(args.cronometro, args.hilos)
            return
        fp = fp_models(args.paso)
        for k in range(len(rp.FOLDS)):
            run_outer(pool, k, args.paso, fp, args.hilos, names, args.rehacer)
        add_time("total_wall_s", time.time() - t0)
    out = []
    report(args.paso, fp_models(args.paso), out)
    text = "\n".join(out)
    print(text)
    with open(REPORT, "w") as f:
        f.write(text + "\n")
    print(f"\n-> {REPORT}")


if __name__ == "__main__":
    main()
