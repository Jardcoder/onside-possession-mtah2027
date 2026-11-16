"""Round 5 (preregistered 2026-10-29): three weighted variants of T1 on the 40 PFF development matches.

    challenge/.venv/bin/python bancos/ronda5_t1.py [--hilos T] [--procesos P] [--rehacer] [--solo-resumen] [--lam-ha A [A ...]] [--lam-d D [D ...]]

T1 (bancos/modelo_transfer.py) is the LightGBM + Viterbi model trained on the balls with injected holes (g6h + g8h). The three variants keep EVERYTHING of T1
(feature matrices of the transfer cache, mo.LGB_PARAMS, mo.MAX_ROUNDS, mo.EARLY_STOP, the 5 outer folds rp.FOLDS, 4 early-stopping matches per fold chosen
as in modelo_transfer, both team views, one frame in STEP, 5 boosters per variant) and change only the weight of each training frame (LightGBM `weight`):

  T1w   segment balance: median_L / L, L = frames of the maximal stretch of `label` (H, A or D, inside a period) that holds the frame, median_L = median of L
        over the stretches of the 32 training matches of the fold; clipped to [0.1, 10], normalised to mean 1 over the training rows
  T1m   restart mask: 0 on the frames with restart_window > 0, 1 elsewhere (label unchanged; not renormalised)
  T1wm  the product of the two (clipped balance x mask), renormalised to mean 1 over the training rows

The swap augmentation duplicates the rows, so it duplicates the weights; the early-stopping set carries the same weights (same median_L and scale as the
training rows). Out-of-fold probabilities of every match with the booster of its fold go to src/out/modelo_cache/ronda5/probs/<variant>__<ball>__<mid>.npy (T1's
own, in modelo_cache/transfer, are only READ). Selection and measure: lambda nested exactly as bancos/barrido_lambda.py (same GRID, same pick_lambda, nested_select),
per variant and for T1, chosen on `label`; the nested per-match spellF1 is read against `label` (main) and against `label_rw` (secondary, no veto; bancos/reinicios.py).
Entry rule (against nested T1 on `label`): ronda_pff.verdict on the per-match delta (min of the two means >= +0.006, bootstrap CI95 lower bound > 0 in both,
median > 0 in both). Report: src/out/ronda5_informe.txt (preregistration block first, results below the mark).

--lam-ha / --lam-d (evaluation only, no training): redo the nested selection and the whole report on a wider lambda grid (lam_ha x lam_d, lam_ha-major as bl.GRID). The grid
is patched into barrido_lambda's module attributes (bl.GRID, bl.GRID_HA, bl.GRID_D) before the pool forks; the sweeps are cached in sweep_<variant>_<grid hash>.npz (the grid is also
in their fingerprint), the report goes to src/out/ronda5_lam_informe.txt (ronda5_informe.txt is not touched) and the final lambda of T1w and T1wm to src/out/lambda_final_t1w.json / lambda_final_t1wm.json.

Control: fold 0 of T1 retrained here with unit weights (variant T1c) must give the probabilities stored by modelo_transfer.

Cache (per booster, resumable): ronda5/modelos/<variant>_k<k>.txt, ronda5/probs/..., ronda5/done_<variant>_k<k>.json (written last; carries the fingerprint =
modelo_transfer.fp_models() + source of the training functions of this file + definition of the variant + hash of the restart windows).
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
import diagnostico_t1 as dg  # noqa: E402
import evaluar_pff as ep  # noqa: E402
import modelo as mo  # noqa: E402
import modelo_pff as mp_  # noqa: E402
import modelo_transfer as mt  # noqa: E402
import pff_split  # noqa: E402
import possession_metrics as pm  # noqa: E402
import reinicios as rt  # noqa: E402
import ronda_pff as rp  # noqa: E402

CACHE = os.path.join(onside_paths.OUT_DIR, "modelo_cache", "ronda5")
PROBS = os.path.join(CACHE, "probs")
MODELOS = os.path.join(CACHE, "modelos")
REPORT = os.path.join(onside_paths.OUT_DIR, "ronda5_informe.txt")
REPORT_LAM = os.path.join(onside_paths.OUT_DIR, "ronda5_lam_informe.txt")
FINAL = {v: os.path.join(onside_paths.OUT_DIR, f"lambda_final_{v.lower()}.json") for v in ("T1w", "T1wm")}   # final lambda of the variants src/predecir_reto_m1.py can decode
DEFAULT_GRID = tuple(bl.GRID)
MARK = "=== RESULTADOS ==="
BALLS = mt.H_BALLS                                       # ("g6h", "g8h")
STEP = mt.STEP
VARIANTS = ("T1w", "T1m", "T1wm")
CONTROL = "T1c"                                          # T1 retrained with unit weights (fold 0 only)
SERIES = ("T1",) + VARIANTS
BALANCE = {"T1w": True, "T1m": False, "T1wm": True, CONTROL: False}      # segment balance on
MASK = {"T1w": False, "T1m": True, "T1wm": True, CONTROL: False}         # restart mask on
NORMALISE = {"T1w": True, "T1m": False, "T1wm": True, CONTROL: False}    # renormalise to mean 1 over the training rows
CLIP = (0.1, 10.0)
GT_POSS_PER_MIN = 3.53                                   # GT possessions per live minute (the number the instruction quotes; recomputed in the report)
CONTROL_TOL = 1e-4                                       # max |probability difference| of the control against the stored T1 probabilities
FPS = pm.FPS


# ------------------------------------------------------------------------------------------------ weights (pure, tested)

def segment_bounds(y, per):
    """Frame indices that delimit the maximal stretches of constant `y` inside a period (first 0, last len(y))."""
    y, per = np.asarray(y), np.asarray(per)
    cut = np.flatnonzero((np.diff(y) != 0) | (np.diff(per) != 0)) + 1
    return np.concatenate(([0], cut, [len(y)]))


def segment_lengths(y, per):
    """(N,) the length in frames of the stretch of `y` that holds each frame."""
    d = np.diff(segment_bounds(y, per))
    return np.repeat(d, d)


def median_length(ys, pers):
    """Median of the stretch lengths over the stretches of several matches (every stretch counts once)."""
    return float(np.median(np.concatenate([np.diff(segment_bounds(y, per)) for y, per in zip(ys, pers)])))


def raw_weights(variant, y, per, rw, med):
    """(N,) float weights of one match before normalisation: clipped median_L / L (balance) times 1 outside restart windows (mask), per the flags of `variant`."""
    w = np.ones(len(y))
    if BALANCE[variant]:
        w *= np.clip(med / segment_lengths(y, per), *CLIP)
    if MASK[variant]:
        w *= (np.asarray(rw) == 0)
    return w


def normalise_weights(variant, w_tr, w_va):
    """(w_tr, w_va, scale): the same scale for both sets, mean 1 over the training rows (only for the variants that renormalise)."""
    scale = float(np.mean(w_tr)) if NORMALISE[variant] else 1.0
    return w_tr / scale, w_va / scale, scale


# ------------------------------------------------------------------------------------------------ training

def fit_weighted(X_tr, y_tr, w_tr, X_va, y_va, w_va, threads=None, rounds=mo.MAX_ROUNDS, params=None):
    """mo.fit with a weight per row in the training and in the early-stopping Dataset (same parameters, same callback); returns the Booster."""
    import lightgbm as lgb
    p = dict(mo.LGB_PARAMS if params is None else params)
    if threads:
        p["num_threads"] = threads
    dtr = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, free_raw_data=True)
    dva = lgb.Dataset(X_va, label=y_va, weight=w_va, reference=dtr, free_raw_data=True)
    return lgb.train(p, dtr, num_boost_round=rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(mo.EARLY_STOP, verbose=False)])


def load_weights_t(mids, step, balls, variant, med):
    """(n,) unnormalised weights of the training rows of mt.load_rows_t(mids, step, balls), in the same order (same ball offsets)."""
    out = []
    for mid in mids:
        rw = rt.load_restart(mid)["rw"]
        for i, b in enumerate(balls):
            z = mt.meta_of(mid, b)
            out.append(raw_weights(variant, z["y"], z["per"], rw, med)[(step // 2) * i::step])
    return np.concatenate(out)


def train_variant(train_mids, variant, threads, names, step=STEP):
    """(booster, seconds, n_rows, info): mt.train_model_t (same 4 early-stopping matches, both views, same rows) with the weights of `variant`."""
    t0 = time.time()
    balls = mt.TRAIN["T1"]
    order = sorted(train_mids, key=lambda m: hashlib.sha256((mp_.ES_SALT + m).encode()).hexdigest())
    es, fit = order[:mp_.ES_MATCHES], order[mp_.ES_MATCHES:]
    med = median_length([mt.meta_of(m, balls[0])["y"] for m in train_mids], [mt.meta_of(m, balls[0])["per"] for m in train_mids])
    Xtr, ytr = mo.augment(*mt.load_rows_t(fit, step, balls), names)
    Xva, yva = mo.augment(*mt.load_rows_t(es, step, balls), names)
    wtr, wva, scale = normalise_weights(variant, load_weights_t(fit, step, balls, variant, med), load_weights_t(es, step, balls, variant, med))
    wtr, wva = np.concatenate([wtr, wtr]), np.concatenate([wva, wva])          # the swap augmentation duplicates the rows: duplicate the weights
    assert len(wtr) == len(ytr) and len(wva) == len(yva)
    booster = fit_weighted(Xtr, ytr, wtr, Xva, yva, wva, threads=threads)
    info = {"median_L": med, "scale": scale, "zero_share": float((wtr == 0).mean()), "w_p50": float(np.median(wtr)), "w_p99": float(np.percentile(wtr, 99)),
            "w_max": float(wtr.max()), "val_rows": int(len(yva))}
    return booster, time.time() - t0, len(ytr), info


# ------------------------------------------------------------------------------------------------ fingerprints and cache

def training_functions():
    """Every function of this file whose code decides a trained booster; the evaluation half of the file is not in the fingerprint (editing the report must not
    retrain). mt.fp_models() already covers mt.train_model_t, mt.load_rows_t and, through its fingerprint of the features, modelo.py and the loader."""
    return (fit_weighted, train_variant, load_weights_t, raw_weights, normalise_weights, segment_bounds, segment_lengths, median_length, run_fold)


_FP = {}


def fp_variant(variant):
    if variant not in _FP:
        h = hashlib.sha256()
        h.update(mt.fp_models().encode())
        h.update(mp_._src(*training_functions()).hexdigest().encode())
        h.update(repr((variant, BALANCE[variant], MASK[variant], NORMALISE[variant], CLIP, STEP, mt.TRAIN["T1"], rt.restart_hash())).encode())
        _FP[variant] = h.hexdigest()
    return _FP[variant]


def probs_path(variant, balon, mid):
    return os.path.join(mt.PROBS if variant == "T1" else PROBS, f"{variant}__{balon}__{mid}.npy")


def done_path(variant, k):
    return os.path.join(CACHE, f"done_{variant}_k{k}.json")


def read_done(variant, k, fp):
    """The marker of a trained booster when it carries `fp` and its probabilities are on disk, else None."""
    try:
        d = json.load(open(done_path(variant, k)))
    except (OSError, ValueError):
        return None
    ok = d.get("fp") == fp and all(os.path.exists(probs_path(variant, b, m)) for m in rp.FOLDS[k] for b in BALLS)
    return d if ok else None


def run_fold(variant, k, fp, threads, names, rehacer=False):
    """Trains `variant` for the outer fold k (the other 32 matches), saves the booster and the out-of-fold probabilities of the 8 matches of k; the marker goes last."""
    if not rehacer and read_done(variant, k, fp) is not None:
        return
    train = [m for q, f in enumerate(rp.FOLDS) if q != k for m in f]
    booster, secs, nrows, info = train_variant(train, variant, threads, names)
    os.makedirs(MODELOS, exist_ok=True)
    os.makedirs(PROBS, exist_ok=True)
    model_path = os.path.join(MODELOS, f"{variant}_k{k}.txt")
    tmp = f"{model_path}.tmp{os.getpid()}"
    booster.save_model(tmp, num_iteration=booster.best_iteration)
    os.replace(tmp, model_path)
    for mid in rp.FOLDS[k]:
        for b in BALLS:
            mp_.atomic_save(probs_path(variant, b, mid), mo.predict_proba(booster, np.load(mt.path_x_of(mid, b)), names).astype(np.float32))
    mp_.atomic_json(done_path(variant, k), {"fp": fp, "secs": secs, "rounds": int(booster.best_iteration), "rows": nrows, **info})
    print(f"{variant} k{k}: {nrows} rows, {booster.best_iteration} rounds, {secs:.0f}s", file=sys.stderr, flush=True)


# ------------------------------------------------------------------------------------------------ sweep and selection

def task_sweep(variant, mid, balon):
    """(spellF1 against `label`, against `label_rw`), each (K,) over bl.GRID, of the out-of-fold probabilities of `variant` on one match and ball."""
    z = mt.meta_of(mid, balon)
    gt = z["gt_label"]
    runs = pm.valid_runs(gt)
    gt_rw = rt.dead_where_restart(gt, rt.load_restart(mid)["rw"])
    P = np.load(probs_path(variant, balon, mid)).astype(float)
    paths = mo.viterbi(np.log(np.clip(P, 1e-9, 1.0)), z["per"], bl.GRID_HA, bl.GRID_D)
    return (np.array([pm.spell_scores(gt, p, runs)["mean_f1"] for p in paths]), np.array([pm.spell_scores(gt_rw, p, runs)["mean_f1"] for p in paths]))


def num(text):
    """int when the text is a whole number (so the grid and the JSON read 8, not 8.0), else float."""
    v = float(text)
    return int(v) if v.is_integer() else v


def set_grid(lam_ha, lam_d):
    """Replaces the lambda grid of the sweeps of this process (before the pool forks): lam_ha-major product, as barrido_lambda.GRID. Returns the grid."""
    bl.LAM_HA, bl.LAM_D = tuple(lam_ha), tuple(lam_d)
    bl.GRID = list(itertools.product(bl.LAM_HA, bl.LAM_D))
    bl.GRID_HA = np.array([g[0] for g in bl.GRID], float)
    bl.GRID_D = np.array([g[1] for g in bl.GRID], float)
    return bl.GRID


def sweep_path(variant):
    """Cache of the sweep: the original name for the default grid, a grid hash in the name otherwise (the two never collide)."""
    if tuple(bl.GRID) == DEFAULT_GRID:
        return os.path.join(CACHE, f"sweep_{variant}.npz")
    return os.path.join(CACHE, f"sweep_{variant}_{hashlib.sha256(repr(bl.GRID).encode()).hexdigest()[:10]}.npz")


def sweep(pool, variant, rehacer=False):
    """(f1, f1_rw), each (40, 2, K): the grid sweep of one variant, cached with its fingerprint (T1: the transfer fingerprint)."""
    mids = list(pff_split.DESARROLLO)
    assert not set(mids) & set(pff_split.RESERVADOS)
    fp = (mt.fp_models() if variant == "T1" else fp_variant(variant)) + repr((bl.GRID, BALLS)) + rt.restart_hash() + mp_._src(task_sweep).hexdigest()
    if not rehacer and os.path.exists(sweep_path(variant)):
        z = np.load(sweep_path(variant))
        if str(z["fp"]) == fp:
            return z["f1"], z["f1_rw"]
    res = pool.starmap(task_sweep, [(variant, m, b) for m in mids for b in BALLS])
    f1 = np.array([r[0] for r in res]).reshape(len(mids), len(BALLS), len(bl.GRID))
    f1_rw = np.array([r[1] for r in res]).reshape(len(mids), len(BALLS), len(bl.GRID))
    os.makedirs(CACHE, exist_ok=True)
    mp_.atomic_savez(sweep_path(variant), fp=fp, f1=f1, f1_rw=f1_rw)
    return f1, f1_rw


def apply_chosen(f1, fold_of, chosen):
    """(n_matches, n_balls): each match read at the grid index chosen for its fold (nested_select's reading, for a second score array)."""
    out = np.zeros(f1.shape[:2])
    for k, i in chosen.items():
        out[fold_of == k] = f1[fold_of == k][:, :, i]
    return out


def task_count(variant, mid, balon, gi):
    """Counters of the path of `variant` decoded at grid point `gi` (the lambda chosen for the fold of the match), one match and ball."""
    z = mt.meta_of(mid, balon)
    gt = z["gt_label"]
    runs = pm.valid_runs(gt)
    P = np.load(probs_path(variant, balon, mid)).astype(float)
    path = mo.viterbi(np.log(np.clip(P, 1e-9, 1.0)), z["per"], [bl.GRID_HA[gi]], [bl.GRID_D[gi]])[0].astype(np.int8)
    dur, hit = dg.transition_recall(gt, path, tols=(50,))
    spells = pm.possession_spells(path, runs)
    return {"frames": int(sum(b - a for a, b in runs)), "n_pred": len(spells), "short": int(sum(b - a < 2 * FPS for a, b, _ in spells)),
            "n_tr_gt": len(dur), "tr_hit": int(hit.sum()), "spurious": len(dg.spurious_transitions(gt, path)),
            "n_tr": len(dg.direct_transitions(pm.fill_abstentions(path, runs), runs)), "d_frames": int(sum((path[a:b] == pm.D).sum() for a, b in runs)),
            "n_gt": len(pm.possession_spells(gt, runs)), "gt_short": int(sum(b - a < 2 * FPS for a, b, _ in pm.possession_spells(gt, runs))),
            "gt_d_frames": int(sum((gt[a:b] == pm.D).sum() for a, b in runs))}


# ------------------------------------------------------------------------------------------------ preregistration

def prereg_text():
    P = []
    P.append("RONDA 5 PREINSCRITA (2026-10-29): tres variantes de T1 con peso por frame, 40 partidos de desarrollo de PFF, balones g6h y g8h.")
    P.append("Este bloque se escribe ANTES de entrenar ninguna variante; los resultados se anaden debajo de la marca. Nada de lo de abajo se cambia despues de ver un resultado.")
    P.append("\n=== Que se prueba ===")
    P.append("T1 (bancos/modelo_transfer.py) entrena con peso 1 en todos los frames. La pregunta: pesar los frames por segmento o ignorar los reinicios, mejora el spellF1 de T1?")
    P.append("Mismo constructor de rasgos (las matrices g6h / g8h de modelo_cache/transfer), mismos hiperparametros (modelo.LGB_PARAMS, MAX_ROUNDS, EARLY_STOP), mismos 5 pliegues externos (ronda_pff.FOLDS),")
    P.append(f"mismo esquema de entrenamiento que T1 (paso {STEP}, 4 partidos de early stopping por pliegue elegidos como en modelo_transfer, simetria por intercambio), 5 boosters por variante.")
    P.append("La UNICA diferencia es el peso por frame (LightGBM `weight`):")
    P.append("  T1w   balance por segmento: peso = mediana_L / L; L = frames del tramo maximal de `label` (H, A o D, dentro de un periodo) que contiene al frame; mediana_L = mediana de L sobre los tramos de los")
    P.append(f"        32 partidos de entrenamiento del pliegue (cada tramo cuenta una vez); recortado a [{CLIP[0]}, {CLIP[1]:g}] y normalizado a media 1 sobre los frames de entrenamiento.")
    P.append("  T1m   mascara de reinicios: peso 0 en los frames con restart_window > 0, 1 en el resto (sin renormalizar). La etiqueta no cambia.")
    P.append("  T1wm  el producto de los dos (T1w recortado por T1m), renormalizado a media 1 sobre los frames de entrenamiento.")
    P.append("Detalles: la simetria por intercambio duplica las filas y tambien los pesos. Los pesos se aplican tambien al conjunto de validacion del early stopping (mismos mediana_L y escala que el entrenamiento).")
    P.append("restart_window viene del loader (bancos/pff.py, PFF_VERSION 3.1-restart-window), alineado fila a fila con ronda_pff.load_match (bancos/reinicios.py, con asserts por partido).")
    P.append("Control: el pliegue 0 de T1 se reentrena aqui con pesos 1 (T1c) y debe dar las probabilidades guardadas de T1 (diferencia maxima < 1e-4); si no, la corrida se detiene.")
    P.append("\n=== Seleccion y medida ===")
    P.append(f"Probabilidades fuera de pliegue de cada partido con el booster de su pliegue. lambda anidado como bancos/barrido_lambda.py (rejilla GRID de {len(bl.GRID)} puntos, pick_lambda, nested_select),")
    P.append("por variante y tambien para T1 (sus probabilidades existentes), elegido sobre `label`. Se mide el spellF1 anidado por partido de cada variante contra `label` (principal) y contra `label_rw`")
    P.append("(`label` con D en todos los frames con restart_window > 0; secundaria, sin veto), en g6h y g8h.")
    P.append("\n=== REGLA DE ENTRADA PREINSCRITA (8.20, contra T1 anidado en `label`, por variante) ===")
    P.append(f"  delta por partido = spellF1 anidado de la variante - spellF1 anidado de T1, n = 40, `label`, fuera de pliegue.")
    P.append(f"  ENTRA si: min(media g6h, media g8h) del delta >= +{rp.ENTRY_MIN}; el limite inferior del IC95 bootstrap por partido (evaluar_pff.delta_stats, semilla fija) > 0 en los dos;")
    P.append("  y la mediana por partido > 0 en los dos. Si no, NO ENTRA. Se aplica la funcion ronda_pff.verdict literalmente.")
    P.append("\n=== Contadores por variante (informativos) ===")
    P.append(f"Posesiones por minuto (GT {GT_POSS_PER_MIN}), % de posesiones < 2 s, recall de cambios H <-> A a +-50 frames, % de cambios espurios, D total (% de los frames validos), con el lambda de cada pliegue.")
    P.append("\n=== Huella del entrenamiento ===")
    P.append("Cada booster se guarda con una huella: modelo_transfer.fp_models() + fuente de las funciones de entrenamiento de este archivo + definicion de la variante + hash de las ventanas de reinicio.")
    P.append("La corrida es reanudable por booster; la evaluacion (barrido, informe) no entra en la huella.")
    return "\n".join(P)


def header_in_report():
    if not os.path.exists(REPORT):
        return None
    t = open(REPORT).read()
    return t.split("\n" + MARK)[0] if MARK in t else None


# ------------------------------------------------------------------------------------------------ report

def pct(a, b):
    return 100.0 * a / b if b else float("nan")


def control_text():
    """Control of the weighted training path: T1c (unit weights) fold 0 against the stored T1 probabilities and booster."""
    d = read_done(CONTROL, 0, fp_variant(CONTROL))
    diffs = []
    for mid in rp.FOLDS[0]:
        for b in BALLS:
            diffs.append(float(np.abs(np.load(probs_path(CONTROL, b, mid)).astype(float) - np.load(probs_path("T1", b, mid)).astype(float)).max()))
    import lightgbm as lgb
    ref_rounds = lgb.Booster(model_file=os.path.join(mt.MODELOS, "T1_k0.txt")).num_trees() // 3
    return diffs, d, ref_rounds


def report(pool):
    out = []
    P = out.append
    mids = list(pff_split.DESARROLLO)
    fold_of = np.array([rp.fold_of(m) for m in mids])
    # control
    diffs, d, ref_rounds = control_text()
    P("=== Control: T1 con pesos 1 reentrenado aqui (pliegue 0) contra las probabilidades guardadas de T1 ===")
    P(f"  diferencia maxima de probabilidad en los 8 partidos x 2 balones: {max(diffs):.2e} (media de los maximos {np.mean(diffs):.2e}); rondas {d['rounds']} contra {ref_rounds} del booster guardado; "
      f"{'OK' if max(diffs) < CONTROL_TOL else 'FUERA DE TOLERANCIA'} (< {CONTROL_TOL:g})")
    # sweeps
    F = {}
    for v in SERIES:
        F[v] = sweep(pool, v)
    if os.path.exists(bl.CACHE):
        ref = np.load(bl.CACHE)
        if str(ref["fp"]) == mt.fp_models() + repr((bl.GRID, BALLS)):
            P(f"  barrido de T1 aqui contra src/out/modelo_cache/transfer/barrido_lambda.npz: diferencia maxima {np.abs(F['T1'][0] - ref['f1']).max():.2e}")
    N, N_rw, CH = {}, {}, {}
    for v in SERIES:
        CH[v], N[v] = bl.nested_select(F[v][0], fold_of)
        N_rw[v] = apply_chosen(F[v][1], fold_of, CH[v])
    P("\nspellF1 medio de los 40 partidos con lambda anidado (elegido sobre `label`), fuera de pliegue:")
    P(f"{'':<8}{'label g6h':>11}{'label g8h':>11}{'rw g6h':>9}{'rw g8h':>9}")
    for v in SERIES:
        P(f"{v:<8}{N[v][:, 0].mean():>11.4f}{N[v][:, 1].mean():>11.4f}{N_rw[v][:, 0].mean():>9.4f}{N_rw[v][:, 1].mean():>9.4f}")
    P("(rw = contra `label_rw`; el lambda sale de `label` tambien ahi)")
    P("\nLambda elegido por pliegue externo (sobre `label`, con los 32 partidos de los otros pliegues):")
    for v in SERIES:
        P(f"  {v:<6}" + "   ".join(f"k{k}: {bl.GRID[CH[v][k]]}" for k in sorted(CH[v])))
    P("\n=== Delta por partido contra T1 anidado, GT `label` (principal), n = 40 ===")
    verdicts = {}
    for v in VARIANTS:
        P(f"\n{v} - T1:\n{rp.HEADER}")
        s = {b: ep.delta_stats(N[v][:, i] - N["T1"][:, i]) for i, b in enumerate(BALLS)}
        for b in BALLS:
            P(rp.fmt_stats(f"{v} - T1, {b}", s[b]))
        conds, ok = rp.verdict(s["g6h"], s["g8h"])
        for c, o in conds:
            P(f"  [{'x' if o else ' '}] {c}")
        verdicts[v] = ok
        P(f"VEREDICTO {v}: {'ENTRA' if ok else 'NO ENTRA'}")
    P("\n=== Secundaria: delta por partido contra T1 anidado, GT `label_rw` (sin veto), n = 40 ===")
    for v in VARIANTS:
        P(f"\n{v} - T1:\n{rp.HEADER}")
        for i, b in enumerate(BALLS):
            P(rp.fmt_stats(f"{v} - T1, {b}, rw", ep.delta_stats(N_rw[v][:, i] - N_rw["T1"][:, i])))
    # counters
    P("\n=== Contadores (suma de los 40 partidos, lambda de cada pliegue sobre `label`) ===")
    P("posesiones/min = por minuto de racha valida; < 2 s = % de las posesiones; recall H <-> A a +-50 frames (de las transiciones GT, existencia); espurias = % de las transiciones del modelo sin GT a +-50;")
    P("D = % de los frames validos predichos D. GT: posesiones/min, % < 2 s y % D de la propia GT.")
    for i, b in enumerate(BALLS):
        P(f"\n-- balon {b}")
        P(f"{'serie':<8}{'pos/min':>9}{'% < 2 s':>9}{'recall +-50':>13}{'% espurias':>12}{'D %':>8}{'D min':>8}{'posesiones':>12}")
        gt_done = False
        for v in SERIES:
            jobs = [(v, m, b, CH[v][rp.fold_of(m)]) for m in mids]
            rs = pool.starmap(task_count, jobs)
            fr = sum(r["frames"] for r in rs)
            if not gt_done:
                n_gt = sum(r["n_gt"] for r in rs)
                P(f"{'GT':<8}{n_gt / (fr / FPS / 60):>9.2f}{pct(sum(r['gt_short'] for r in rs), n_gt):>9.1f}{'':>13}{'':>12}{pct(sum(r['gt_d_frames'] for r in rs), fr):>8.1f}"
                  f"{sum(r['gt_d_frames'] for r in rs) / FPS / 60:>8.1f}{n_gt:>12}")
                gt_done = True
            n_pred = sum(r["n_pred"] for r in rs)
            d_fr = sum(r["d_frames"] for r in rs)
            P(f"{v:<8}{n_pred / (fr / FPS / 60):>9.2f}{pct(sum(r['short'] for r in rs), n_pred):>9.1f}{pct(sum(r['tr_hit'] for r in rs), sum(r['n_tr_gt'] for r in rs)):>12.1f}%"
              f"{pct(sum(r['spurious'] for r in rs), sum(r['n_tr'] for r in rs)):>11.1f}%{pct(d_fr, fr):>8.1f}{d_fr / FPS / 60:>8.1f}{n_pred:>12}")
    finals = {}
    if tuple(bl.GRID) != DEFAULT_GRID:
        P(f"\n=== Rejilla ampliada: lam_ha {bl.LAM_HA} x lam_d {bl.LAM_D} ===")
        for v in SERIES:
            fi, count = bl.final_choice(CH[v], F[v][0])
            finals[v] = {"lambda": list(bl.GRID[fi]), "chosen_by_fold": {str(k): list(bl.GRID[i]) for k, i in CH[v].items()}, "times_chosen": {str(bl.GRID[i]): c for i, c in count.items()}}
            used = {bl.GRID[i] for i in CH[v].values()} | {bl.GRID[fi]}
            edges = sorted({f"lam_ha = {a} ({bl.on_edge(a, bl.LAM_HA)})" for a, _ in used if bl.on_edge(a, bl.LAM_HA)} | {f"lam_d = {d} ({bl.on_edge(d, bl.LAM_D)})" for _, d in used if bl.on_edge(d, bl.LAM_D)})
            obj = np.minimum(F[v][0][:, 0, :].mean(0), F[v][0][:, 1, :].mean(0))
            P(f"  {v:<6} LAMBDA FINAL {bl.GRID[fi]} (veces elegido {count}); objetivo con los 40 partidos {obj[fi]:.4f}; en el borde de la rejilla: {'; '.join(edges) or 'NO'}")
            P(f"         mejor punto con los 40 partidos (informativo): {bl.GRID[int(np.argmax(obj))]} {obj.max():.4f}")
    # training
    P("\n=== Entrenamiento ===")
    P(f"{'variante':<8}{'pliegue':>8}{'rondas':>8}{'filas':>10}{'seg':>7}{'mediana_L':>11}{'escala':>9}{'% peso 0':>10}{'p50 w':>8}{'p99 w':>8}{'max w':>8}")
    tot = 0.0
    for v in (CONTROL,) + VARIANTS:
        for k in range(len(rp.FOLDS)):
            dn = read_done(v, k, fp_variant(v))
            if dn is None:
                continue
            tot += dn["secs"]
            P(f"{v:<8}{k:>8}{dn['rounds']:>8}{dn['rows']:>10}{dn['secs']:>7.0f}{dn['median_L']:>11.1f}{dn['scale']:>9.3f}{100 * dn['zero_share']:>10.1f}{dn['w_p50']:>8.2f}{dn['w_p99']:>8.2f}{dn['w_max']:>8.2f}")
    P(f"Entrenamiento sumado: {tot / 60:.1f} min ({tot / 3600:.2f} h). T1 (referencia, modelo_transfer): ~165 s por booster.")
    return "\n".join(out), verdicts, finals


# ------------------------------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procesos", type=int, default=min(9, os.cpu_count() or 1))
    ap.add_argument("--hilos", type=int, default=min(9, os.cpu_count() or 1))
    ap.add_argument("--rehacer", action="store_true")
    ap.add_argument("--solo-resumen", action="store_true")
    ap.add_argument("--lam-ha", type=num, nargs="+", help="lam_ha of a wider lambda grid (evaluation only; report to ronda5_lam_informe.txt)")
    ap.add_argument("--lam-d", type=float, nargs="+", default=list(bl.LAM_D), help="lam_d of the wider grid (default: barrido_lambda.LAM_D)")
    args = ap.parse_args()
    assert args.hilos <= 9 and args.procesos <= 9, "at most 9 cores"
    assert not set(pff_split.DESARROLLO) & set(pff_split.RESERVADOS)
    t0 = time.time()
    wide = args.lam_ha is not None
    if wide:
        set_grid(args.lam_ha, args.lam_d)
        args.solo_resumen = True                                         # no training, ronda5_informe.txt untouched
        print(f"grid lam_ha {bl.LAM_HA} x lam_d {bl.LAM_D}", file=sys.stderr, flush=True)
    rt.restart_hash()                                                    # loads the 40 windows once: forked workers inherit them
    if not args.solo_resumen:
        if header_in_report() is None:
            with open(REPORT, "w") as f:                                 # first: the rule is on paper before any variant is trained
                f.write(prereg_text() + "\n\n" + MARK + "\n")
        names = list(np.load(mt.path_meta_t(rp.FOLDS[0][0], "g6h"))["names"])
        os.makedirs(CACHE, exist_ok=True)
        run_fold(CONTROL, 0, fp_variant(CONTROL), args.hilos, names, args.rehacer)
        diffs, d, ref_rounds = control_text()
        print(f"control T1c fold 0: max |dP| {max(diffs):.2e}, rounds {d['rounds']} vs {ref_rounds}", file=sys.stderr, flush=True)
        if max(diffs) >= CONTROL_TOL:
            raise SystemExit(f"CONTROL FAILED: the weighted training path with unit weights deviates {max(diffs):.2e} from the stored T1 probabilities; stopping (report it)")
        for v in VARIANTS:
            for k in range(len(rp.FOLDS)):
                run_fold(v, k, fp_variant(v), args.hilos, names, args.rehacer)
    head = header_in_report()
    if head is None:
        raise SystemExit("the report has no preregistered header: run without --solo-resumen")
    t1 = time.time()
    with mp.get_context("fork").Pool(args.procesos) as pool:
        text, verdicts, finals = report(pool)
    text += f"\n\nTiempo: entrenamiento de esta corrida {t1 - t0:.0f} s (los boosters ya en cache no cuentan); evaluacion {time.time() - t1:.0f} s."
    if wide:
        full = (f"RONDA 5, REJILLA DE LAMBDA AMPLIADA (solo evaluacion, sin reentrenar): lam_ha {bl.LAM_HA} x lam_d {bl.LAM_D}; mismo criterio, mismo pick_lambda, mismos boosters y probabilidades "
                f"que ronda5_informe.txt (que no se toca).\n\n" + MARK + "\n" + text)
        out_path = REPORT_LAM
        for v, path in FINAL.items():
            mp_.atomic_json(path, finals[v])
    else:
        full, out_path = head + "\n\n" + MARK + "\n" + text, REPORT
    print(text)
    with open(out_path, "w") as f:
        f.write(full + "\n")
    print(f"\n-> {out_path}" + ("".join(f"\n-> {path}" for path in FINAL.values()) if wide else ""))


if __name__ == "__main__":
    main()
