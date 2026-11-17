"""Transfer test of M1: does its gain survive a ball that is missing during live play, as in the challenge?

    challenge/.venv/bin/python bancos/modelo_transfer.py [--hilos T] [--procesos P] [--rehacer] [--solo-resumen] [--cronometro]
    challenge/.venv/bin/python bancos/modelo_transfer.py --calibracion        # stage 1 only: calibration + preregistered rule -> report

Variant `h` of the PFF bank (bancos/inyeccion_huecos.py): the already degraded ball (g6, g8) gets extra holes inside live play, calibrated on the
challenge aggregates (src/out/regimen_reto.csv, bancos/data/cache/huecos_largo_atd.json). The GT only decides where a hole may open.

Models (the 5 outer models of bancos/modelo_pff.py, same hyper-parameters, same step, g6 + g8 together, fixed lambda = LAMBDA, the one the 5
folds of M1 chose): T0 trained without holes (as M1), T1 trained with holes (g6h + g8h). Out of fold, 40 matches. Base: B3 + absorption
tau 1.6 s computed on the ball of each variant (the base also gets the holes). Everything reuses src/modelo.py and bancos/modelo_pff.py by import;
the M1 cache (src/out/modelo_cache/v1) is only READ (its fingerprint is stale because ronda_pff / absorcion grew round 4 after M1 ran, but the
rows are identical: checked by recomputing features of two matches, see tests/test_modelo_transfer.py), everything new goes to
src/out/modelo_cache/transfer/:
  <mid>__<ball>.npy / .npz           features and meta of the h variants (ball in H_BALLS), same layout as the M1 cache plus `hole`
  caminos/<T0|T1>__<ball>__<mid>.npy Viterbi path (N,) int8, 0 = H, 1 = A, 2 = D (documented in the report)
  caminos/huecos__<ball>__<mid>.npy  injected-hole mask (N,) bool
  probs/<T0|T1>__<ball>__<mid>.npy   out-of-fold probabilities (N, 3) float32 (H, A, D) of the model of the fold that holds the match (what the lambda sweep of
                                     bancos/barrido_lambda.py decodes; the Viterbi path above is the same probabilities at LAMBDA)
  modelos/<T0|T1>_k<k>.txt           LightGBM booster of the outer fold k
  rows_<T0|T1>_k<k>.csv              per (match, ball, series) rows, first line `# fingerprint`
"""
import argparse
import hashlib
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
import absorcion as ab  # noqa: E402
import dfl_degradado as dd  # noqa: E402
import evaluar_pff as ep  # noqa: E402
import inyeccion_huecos as ih  # noqa: E402
import modelo as mo  # noqa: E402
import modelo_pff as mp_  # noqa: E402
import pff_split  # noqa: E402
import possession_metrics as pm  # noqa: E402
import regimen_reto as rr  # noqa: E402
import ronda_pff as rp  # noqa: E402

CACHE = os.path.join(onside_paths.OUT_DIR, "modelo_cache", "transfer")
CAMINOS = os.path.join(CACHE, "caminos")
PROBS = os.path.join(CACHE, "probs")
MODELOS = os.path.join(CACHE, "modelos")
REPORT = os.path.join(onside_paths.OUT_DIR, "modelo_transfer_informe.txt")
MARK = "=== RESULTADOS ==="
H_BALLS = ("g6h", "g8h")
NO_H_BALLS = ("g6", "g8")
EVAL_BALLS = NO_H_BALLS + H_BALLS
BASE_OF = {"g6h": "g6", "g8h": "g8"}
LAMBDA = (8, 16)                                # (lam_ha, lam_d) chosen by the 5 outer folds of M1
TRAIN = {"T0": NO_H_BALLS, "T1": H_BALLS}
EVAL = {"T0": EVAL_BALLS, "T1": H_BALLS}
M1_REF = {"g6": 0.6160, "g8": 0.6125}           # spellF1 mean of M1 (src/out/modelo_pff_informe.txt)
CONTROL_TOL = 0.01
RHOS = (0.0, 0.3, 0.5, 0.7, 0.85, 1.0)          # grid of the calibration of ANCHOR_P
ABSENT_TOL = 0.005                              # calibration criteria, fixed before measuring: no ball within +-0.005, holes > 5 s per hour within +-10 %
GAPS_TOL = 0.10
STEP = mp_.STEP
FIELDS = ["partido", "balon", "modelo", "spell_f1", "n_gt", "n_pred", "c05", "c10", "c20", "live_gt", "n_live_nb", "n_dfalse_nb", "n_hole",
          "n_dfalse_hole"]
SERIES = ("GT", "B3", "B3_t1.6")


# ------------------------------------------------------------------------------------------------ fingerprints

_FP = {}


def fp_features():
    """Code and calibration that decide the cached h features and meta (computed once per process)."""
    if "f" not in _FP:
        h = mp_._src(mo, ih, build_one_h, build_match_h)
        h.update(rp.fingerprint("R2", "tau1.6", (1.6,)).encode())
        h.update(open(dd.CAL_LARGO, "rb").read())
        h.update(mp_.loader_hash().encode())
        h.update(np.__version__.encode())
        h.update(repr((rp.BALLS, mp_.TAUS, mo.WINDOWS_S, mo.SPEED_HALF_S, mo.CAP_S, ih.SEED_SALT, ih.TARGET_ABSENT, ih.ANCHOR_P,
                       ih.MAX_TRIES, H_BALLS)).encode())
        _FP["f"] = h.hexdigest()
    return _FP["f"]


def model_functions():
    """Every function whose code decides the rows of the T0 / T1 stages: trains, predicts, decodes, scores, reads the caches. Helpers outside
    this file: mo.* and pm.* (modules already in fp_features), mp_.eval_stats and rp.spell_row (added by fp_models)."""
    return (train_model_t, load_rows_t, task_rows, meta_of, path_x_of, path_x_t, path_meta_t, rows_path, run_stage, mp_.eval_stats,
            rp.spell_row)


def fp_models():
    h = hashlib.sha256()
    h.update(fp_features().encode())
    h.update(mp_._src(*model_functions()).hexdigest().encode())
    h.update(repr((STEP, sorted(mo.LGB_PARAMS.items()), mo.MAX_ROUNDS, mo.EARLY_STOP, LAMBDA, rp.FOLDS, mp_.ES_SALT, mp_.ES_MATCHES,
                   TRAIN, EVAL, FIELDS, SERIES, rp.BINS, mp_.loader_hash(), mp_.lightgbm.__version__, np.__version__)).encode())
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ features of the h variants

def path_x_t(mid, balon):
    return os.path.join(CACHE, f"{mid}__{balon}.npy")


def path_meta_t(mid, balon):
    return os.path.join(CACHE, f"{mid}__{balon}.npz")


def path_x_of(mid, balon):
    """Feature file of a (match, ball): the M1 cache for g6 / g8 (read only), the transfer cache for the h variants."""
    return path_x_t(mid, balon) if balon in H_BALLS else mp_.path_x(mid, balon)


def meta_of(mid, balon):
    z = np.load(path_meta_t(mid, balon) if balon in H_BALLS else mp_.path_meta(mid, balon))
    return {k: z[k] for k in z.files}


def build_one_h(d, balon_h, fp):
    """Features + meta of one (match, h ball): holes injected into the degraded ball, then the same pipeline as modelo_pff.build_one.
    The GT (`label`) only goes to `inject_holes` (where a hole may open) and to the meta; the methods see the tracking arrays only."""
    ball0 = rp.ball_variant(d, BASE_OF[balon_h])
    ball, hole = ih.inject_holes(ball0, d["label"], d["period"], d["mid"])
    m = rp.build_method_ctx(ball, d["players_slots"], d["period"], d["pitch"])
    b3, tf, _ = rp.pipeline(m)
    b3a = [ab.absorb(b3, t, d["period"]) for t in mp_.TAUS]                  # the same post-process rp.pipeline(m, r2=t) applies
    X, names = mo.build_features(ball, d["players_slots"], d["period"], d["pitch"], b3, b3a[0], tf)
    mp_.atomic_save(path_x_t(d["mid"], balon_h), X)                          # the meta (it carries the fingerprint) goes last
    mp_.atomic_savez(path_meta_t(d["mid"], balon_h), fp=fp, names=np.array(names), y=mp_.label_codes(d["label"]),
             gt_label=rp.ev.make_gt(d, "label"), gt_poss=rp.ev.make_gt(d, "label_poss"), per=d["period"].astype(np.int8),
             b3=mp_.label_codes(b3), b3_t08=mp_.label_codes(b3a[0]), b3_t16=mp_.label_codes(b3a[1]), hole=hole)


def feature_ok(mid, balon, fp):
    try:
        if not (os.path.exists(path_x_t(mid, balon)) and os.path.exists(path_meta_t(mid, balon))):
            return False
        return str(np.load(path_meta_t(mid, balon))["fp"]) == fp
    except (OSError, ValueError, KeyError):
        return False


def build_match_h(mid, fp, rehacer=False):
    todo = [b for b in H_BALLS if rehacer or not feature_ok(mid, b, fp)]
    if not todo:
        return mid, 0
    d = rp.load_match(mid)
    d["mid"] = mid
    os.makedirs(CACHE, exist_ok=True)
    for b in todo:
        build_one_h(d, b, fp)
    return mid, len(todo)


def build_all(pool, rehacer):
    fp = fp_features()
    mids = list(pff_split.DESARROLLO)
    assert not set(mids) & set(pff_split.RESERVADOS)
    t0 = time.time()
    res = pool.starmap(build_match_h, [(m, fp, rehacer) for m in mids], chunksize=1)
    print(f"features h: {sum(n for _, n in res)} (match, ball) files computed in {time.time() - t0:.0f}s", file=sys.stderr, flush=True)


# ------------------------------------------------------------------------------------------------ training

def load_rows_t(mids, step, balls):
    """(X, y) of the training rows of `mids`: one frame of every `step`, offset 0 for the first ball and step // 2 for the second (as modelo_pff)."""
    xs, ys = [], []
    for mid in mids:
        for i, b in enumerate(balls):
            off = (step // 2) * i
            xs.append(np.ascontiguousarray(np.load(path_x_of(mid, b), mmap_mode="r")[off::step]))
            ys.append(meta_of(mid, b)["y"][off::step])
    return np.concatenate(xs), np.concatenate(ys)


def train_model_t(train_mids, balls, threads, names, step=STEP):
    """(booster, seconds, n_rows): as modelo_pff.train_model (same 4 early-stopping matches, both views) with the balls and cache of this file."""
    t0 = time.time()
    order = sorted(train_mids, key=lambda m: hashlib.sha256((mp_.ES_SALT + m).encode()).hexdigest())
    es, fit = order[:mp_.ES_MATCHES], order[mp_.ES_MATCHES:]
    Xtr, ytr = mo.augment(*load_rows_t(fit, step, balls), names)
    Xva, yva = mo.augment(*load_rows_t(es, step, balls), names)
    booster = mo.fit(Xtr, ytr, Xva, yva, threads=threads)
    return booster, time.time() - t0, len(ytr)


# ------------------------------------------------------------------------------------------------ evaluation

def task_rows(modelo, mid, balon, codes, present):
    """One row: spellF1 and counters of a series on (match, ball); `codes` (N,) 0/1/2, `present` (N,) bool = the ball the method saw."""
    z = meta_of(mid, balon)
    gt = z["gt_label"]
    runs = pm.valid_runs(gt)
    f1, n_gt, n_pred, c = mp_.eval_stats(gt, codes, runs)
    live_nb = (z["y"] != 2) & ~present                                     # live frames (GT) with no ball
    hole = z["hole"] if "hole" in z else np.zeros(len(codes), bool)
    codes = np.asarray(codes)
    return {"partido": mid, "balon": balon, "modelo": modelo, "spell_f1": f1, "n_gt": n_gt, "n_pred": n_pred, "c05": c[0], "c10": c[1],
            "c20": c[2], "live_gt": rp.spell_row(gt, runs)[1], "n_live_nb": int(live_nb.sum()), "n_dfalse_nb": int((live_nb & (codes == 2)).sum()),
            "n_hole": int(hole.sum()), "n_dfalse_hole": int((hole & (codes == 2)).sum())}


def rows_path(modelo, k):
    return os.path.join(CACHE, f"rows_{modelo}_k{k}.csv")


def times_path():
    return os.path.join(CACHE, "tiempos.json")


def add_time(key, seconds):
    t = json.load(open(times_path())) if os.path.exists(times_path()) else {}
    t[key] = seconds
    mp_.atomic_json(times_path(), t)


def run_stage(pool, modelo, k, fp, threads, names, rehacer=False):
    """Trains `modelo` for the outer fold k (the other 32 matches), decodes the 8 held-out matches, saves paths, returns nothing (rows cached)."""
    if not rehacer and mp_.load_if(rows_path(modelo, k), fp) is not None:
        return
    train = [m for q, f in enumerate(rp.FOLDS) if q != k for m in f]
    booster, secs, nrows = train_model_t(train, TRAIN[modelo], threads, names)
    os.makedirs(MODELOS, exist_ok=True)
    os.makedirs(CAMINOS, exist_ok=True)
    os.makedirs(PROBS, exist_ok=True)
    model_path = os.path.join(MODELOS, f"{modelo}_k{k}.txt")
    tmp_model = f"{model_path}.tmp{os.getpid()}"
    booster.save_model(tmp_model, num_iteration=booster.best_iteration)
    os.replace(tmp_model, model_path)
    ip = names.index("ball_present")
    jobs = []
    for mid in rp.FOLDS[k]:
        for b in EVAL[modelo]:
            X = np.load(path_x_of(mid, b))
            P = mo.predict_proba(booster, X, names)
            mp_.atomic_save(os.path.join(PROBS, f"{modelo}__{b}__{mid}.npy"), P.astype(np.float32))
            path = mo.viterbi(np.log(np.clip(P, 1e-9, 1.0)), meta_of(mid, b)["per"], [LAMBDA[0]], [LAMBDA[1]])[0].astype(np.int8)
            mp_.atomic_save(os.path.join(CAMINOS, f"{modelo}__{b}__{mid}.npy"), path)
            present = X[:, ip] > 0.5
            jobs.append((modelo, mid, b, path, present))
            if modelo == "T0":                                              # the series that need no model go with the first stage
                z = meta_of(mid, b)
                jobs += [("GT", mid, b, z["gt_label"], present), ("B3", mid, b, z["b3"], present), ("B3_t1.6", mid, b, z["b3_t16"], present)]
                if b in H_BALLS:
                    mp_.atomic_save(os.path.join(CAMINOS, f"huecos__{b}__{mid}.npy"), z["hole"])
    rows = pool.starmap(task_rows, jobs)
    mp_.atomic_csv(rows_path(modelo, k), fp, FIELDS, rows)                    # last: it is what marks the stage as done
    add_time(f"{modelo}_k{k}", secs)
    print(f"{modelo} k{k}: {nrows} rows, {booster.best_iteration} rounds, {secs:.0f}s", file=sys.stderr, flush=True)


def read_rows(fp, modelos=("T0", "T1")):
    rows = []
    for m in modelos:
        for k in range(len(rp.FOLDS)):
            r = mp_.load_if(rows_path(m, k), fp)
            if r is None:
                raise SystemExit(f"stale or missing {rows_path(m, k)} (run without --solo-resumen)")
            rows += r
    return rows


def f1_of(rows, modelo, balon):
    return {r["partido"]: float(r["spell_f1"]) for r in rows if r["modelo"] == modelo and r["balon"] == balon}


def delta(rows, a, b, balon):
    fa, fb = f1_of(rows, a, balon), f1_of(rows, b, balon)
    return np.array([fa[m] - fb[m] for m in pff_split.DESARROLLO])


def control(rows):
    """[(ball, T0 mean without holes, M1 reference, deviation, ok)] of the control: T0 on g6 / g8 must reproduce M1."""
    out = []
    for b in NO_H_BALLS:
        v = float(np.mean(list(f1_of(rows, "T0", b).values())))
        out.append((b, v, M1_REF[b], v - M1_REF[b], abs(v - M1_REF[b]) <= CONTROL_TOL))
    return out


def transfer_verdict(s6, s8):
    """Preregistered rule: (label, text). s6, s8 = ep.delta_stats of T1 - base on g6h and g8h."""
    lo = min(s6["mean"], s8["mean"])
    if lo >= 0.05 and s6["lo"] > 0 and s8["lo"] > 0:
        return "TRANSFERIBLE", "min(g6h, g8h) >= +0.05 con IC95 inferior > 0 en los dos"
    if lo > 0:
        return "GANANCIA REAL PERO FRAGIL", ("min(g6h, g8h) entre 0 y +0.05" if lo < 0.05 else
                                              "min(g6h, g8h) >= +0.05 pero el IC95 inferior no es > 0 en los dos")
    return "EL SALTO ERA EL ATAJO", "T1 no gana a la base en PFF+h (min(g6h, g8h) <= 0)"


# ------------------------------------------------------------------------------------------------ calibration (stage 1)

def regime_work(mid):
    """Regime metrics of the h variants of one match (tracking of the ball with holes only), plus the share of absent frames in live play."""
    d = rp.load_match(mid)
    out = {}
    for b in H_BALLS:
        ball, hole = ih.inject_holes(rp.ball_variant(d, BASE_OF[b]), d["label"], d["period"], mid)
        r = rr.regime(ball, d["players_slots"], d["period"], d["pitch"])
        nan = np.isnan(ball[:, 0])
        r["absent_vivo"] = float((nan & (d["label"] != "D")).sum() / max(nan.sum(), 1))
        r["hole_frac"] = float(hole.mean())
        out[b] = r
    raw = np.isnan(rp.ball_variant(d, "g6")[:, 0])
    out["base_absent_vivo"] = float((raw & (d["label"] != "D")).sum() / max(raw.sum(), 1))
    return mid, out


def scan_work(mid):
    d = rp.load_match(mid)
    ball = rp.ball_variant(d, "g6")
    hrs = len(ball) / dd.FPS / 3600
    res = []
    for rho in RHOS:
        bh, _ = ih.inject_holes(ball, d["label"], d["period"], mid, anchor_p=rho)
        present = ~np.isnan(bh[:, 0])
        res.append((1 - present.mean(), sum(1 for g in rr.reglas.gap_runs(present, d["period"]) if g["end"] - g["start"] > rr.GAP_S * dd.FPS) / hrs))
    return res


def read_challenge():
    """{(variant, smoothing, metric): (reto, pff_mean, p10, p50, p90)} from src/out/regimen_reto.csv (aggregates only)."""
    out = {}
    for line in open(rr.OUT).read().splitlines()[1:]:
        v, s, m, *rest = line.split(",")
        out[(v, s, m)] = tuple(float(x) if x else np.nan for x in rest[:5])
    return out


def calibration_text(pool):
    P = []
    ch = read_challenge()
    reto_abs, reto_gaps = ch[("completo", "g6", "sin_balon")][0], ch[("completo", "g6", "huecos_h")][0]
    assert abs(reto_abs - ih.TARGET_ABSENT) < 1e-4 and abs(reto_gaps - ih.TARGET_HUECOS_H) < 1e-3, "constants of inyeccion_huecos.py disagree with regimen_reto.csv"
    P.append("TRANSFERENCIA DE M1 A UN BALON QUE FALTA CON EL JUEGO VIVO (variante h de PFF)")
    P.append("Sospecha: en PFF el 89 % de los frames sin balon son balon muerto; si M1 aprendio 'sin balon = D', en el reto parte posesiones.")
    P.append("Informe en dos tiempos: este bloque (calibracion y regla) se escribe ANTES de entrenar o medir ningun modelo; los resultados se anaden debajo de la marca.")
    P.append("\n=== Calibracion de la inyeccion de huecos (bancos/inyeccion_huecos.py) ===")
    P.append("Sobre el balon ya degradado (g6, g8) se pone NaN en tramos dentro de juego vivo segun la GT (label != D); la GT solo decide DONDE puede abrirse un hueco, nunca entra en un rasgo.")
    P.append("Reglas del hueco: todos sus frames son vivos y con balon (los NaN existentes no se tocan), dentro de un periodo, con un frame visible a cada lado, salvo que se ancle a un hueco existente (ver abajo).")
    P.append("Semilla: SHA-256('" + ih.SEED_SALT + "' + id del partido); los mismos huecos para g6 y g8 de un partido.")
    P.append(f"Objetivo (agregados del reto, variante 'completo' de src/out/regimen_reto.csv): sin balon {reto_abs:.4f} del tiempo, huecos > 5 s {reto_gaps:.1f} por hora.")
    P.append("Longitudes: histogramas de racha de balon ausente del reto por clase (< 5, 5-10, 10-20, >= 20 s; huecos_largo_atd.json, solo agregados): clase con probabilidad ~ numero de rachas, "
             "longitud del histograma de la clase (uniforme dentro del bin). Es dd._len_gap_model sin el peso (1 - P(muerto | ausente)), que daria solo huecos de 5-10 s.")
    P.append(f"Tasa: por partido se abren huecos hasta que la fraccion sin balon llega a {ih.TARGET_ABSENT} (un partido que ya la supera no recibe ninguno).")
    P.append("Problema encontrado al calibrar: con huecos libres la fraccion sin balon sale bien pero los huecos > 5 s por hora salen muy por encima del reto, porque las rachas vivas de PFF "
             "son cortas (mediana 25 s, 98 por partido) y los huecos largos solo caben cuando son cortos. Segundo parametro, rho = ANCHOR_P: un hueco se ancla con probabilidad rho al borde vivo de un hueco "
             "existente (el hueco crece hacia el juego vivo y se fusionan: balon perdido alrededor de una parada y recuperado despues); con 1 - rho cae libre en el juego vivo.")
    P.append(f"Criterio fijado antes de medir, para elegir rho en la rejilla {RHOS}: el menor rho con fraccion sin balon dentro de +-{ABSENT_TOL} y huecos > 5 s por hora dentro de +-{GAPS_TOL:.0%} del reto "
             "(el menor rho = mas huecos libres en medio del juego).")
    scan = np.array(pool.map(scan_work, list(pff_split.DESARROLLO)))                       # (40, len(RHOS), 2)
    P.append(f"\n{'rho':>6}{'sin balon':>11}{'huecos>5s/h':>13}{'vs reto':>9}")
    chosen = None
    for i, rho in enumerate(RHOS):
        a, g = scan[:, i, 0].mean(), scan[:, i, 1].mean()
        ok = abs(a - reto_abs) <= ABSENT_TOL and abs(g / reto_gaps - 1) <= GAPS_TOL
        chosen = rho if ok and chosen is None else chosen
        P.append(f"{rho:>6.2f}{a:>11.4f}{g:>13.1f}{g / reto_gaps - 1:>+9.1%}{'   <- cumple' if ok else ''}")
    P.append(f"(reto: {reto_abs:.4f}, {reto_gaps:.1f}; PFF sin huecos: {ch[('completo', 'g6', 'sin_balon')][1]:.4f}, {ch[('completo', 'g6', 'huecos_h')][1]:.1f})")
    P.append(f"ELEGIDO: rho = {chosen}   (ANCHOR_P en el codigo = {ih.ANCHOR_P})")
    assert chosen == ih.ANCHOR_P, f"ANCHOR_P of the code ({ih.ANCHOR_P}) is not the one the preregistered criterion picks ({chosen})"
    res = dict(pool.map(regime_work, list(pff_split.DESARROLLO)))
    mids = list(pff_split.DESARROLLO)
    P.append("\n=== Comprobacion: PFF+h se parece al reto (metricas de bancos/regimen_reto.py, solo tracking, 40 partidos de desarrollo) ===")
    P.append(f"{'metrica':<11}{'reto':>9}{'PFF g6':>9}{'g6h':>9}{'g6h p10':>9}{'g6h p50':>9}{'g6h p90':>9}{'pct reto':>9}{'g8h':>9}")
    mean = {b: {} for b in H_BALLS}
    for m in rr.METRICS:
        v = {b: np.array([res[i][b][m] for i in mids]) for b in H_BALLS}
        for b in H_BALLS:
            mean[b][m] = float(v[b].mean())
        reto, pff = ch[("completo", "g6", m)][0], ch[("completo", "g6", m)][1]
        p10, p50, p90 = np.percentile(v["g6h"], [10, 50, 90])
        P.append(f"{m:<11}{reto:>9.3f}{pff:>9.3f}{mean['g6h'][m]:>9.3f}{p10:>9.3f}{p50:>9.3f}{p90:>9.3f}{rr.pct_rank(reto, v['g6h']):>8.0f}%{mean['g8h'][m]:>9.3f}")
    d0 = rr.distance({m: ch[("completo", "g6", m)][0] for m in rr.METRICS}, {m: ch[("completo", "g6", m)][1] for m in rr.METRICS})
    P.append("distancia media |log(reto / PFF)| sobre las metricas con las que ambos son positivos: "
             f"PFF g6 {d0[0]:.3f} ({d0[1]}/{len(rr.METRICS)}); PFF+h g6h {rr.distance({m: ch[('completo', 'g6', m)][0] for m in rr.METRICS}, mean['g6h'])[0]:.3f}; "
             f"g8h {rr.distance({m: ch[('completo', 'g6', m)][0] for m in rr.METRICS}, mean['g8h'])[0]:.3f}")
    hf = np.mean([res[i]["g6h"]["hole_frac"] for i in mids])
    P.append(f"frames inyectados: {100 * hf:.1f} % de los frames; fraccion de los frames sin balon que son de juego vivo (GT): "
             f"PFF {100 * np.mean([res[i]['base_absent_vivo'] for i in mids]):.1f} %, PFF+h {100 * np.mean([res[i]['g6h']['absent_vivo'] for i in mids]):.1f} %")
    ok_a = abs(mean["g6h"]["sin_balon"] - reto_abs) <= ABSENT_TOL and abs(mean["g8h"]["sin_balon"] - reto_abs) <= ABSENT_TOL
    ok_g = all(abs(mean[b]["huecos_h"] / reto_gaps - 1) <= GAPS_TOL for b in H_BALLS)
    P.append(f"PFF+h imita al reto en fraccion sin balon (+-{ABSENT_TOL}): {'SI' if ok_a else 'NO'}; en huecos > 5 s por hora (+-{GAPS_TOL:.0%}): {'SI' if ok_g else 'NO'}")
    P.append("\n=== Modelos y medidas ===")
    P.append(f"T0: M1 entrenado sin huecos (g6 + g8), evaluado en g6 / g8 (control) y en g6h / g8h. T1: mismo entrenamiento con g6h + g8h, evaluado en g6h / g8h. Base: B3 + absorcion tau 1,6 sobre el balon de cada variante.")
    P.append(f"Solo los 5 modelos externos de modelo_pff (FOLDS de ronda_pff, 32 partidos de entrenamiento, 4 de ellos para early stopping), mismos hiperparametros, paso {STEP}, lambda fijo {LAMBDA} (lo que eligieron los 5 pliegues de M1).")
    P.append(f"Control: T0 sin huecos debe reproducir M1 (g6 {M1_REF['g6']}, g8 {M1_REF['g8']}) con desviacion <= {CONTROL_TOL}; si no, se detiene la corrida y se reporta.")
    P.append("\n=== REGLA PREINSCRITA (escrita antes de medir) ===")
    P.append("Medida: delta de spellF1 por partido de T1 - base en g6h y g8h (n = 40, GT label, fuera de pliegue), IC95 bootstrap por partido (evaluar_pff.delta_stats).")
    P.append("  M1 es TRANSFERIBLE si min(g6h, g8h) de T1 - base >= +0.05 con IC95 inferior > 0 en los dos.")
    P.append("  Entre 0 y +0.05 (o >= +0.05 sin IC95 inferior > 0 en los dos): ganancia real pero FRAGIL.")
    P.append("  Si T1 no gana a la base en PFF+h (min(g6h, g8h) <= 0): el salto era el atajo.")
    P.append("  Ademas se informa el veredicto 8.20 de T1 contra la base (rp.verdict, +0.006 con IC y mediana), T0 - base y T1 - T0.")
    return "\n".join(P), chosen


def write_header(text):
    with open(REPORT, "w") as f:
        f.write(text + "\n\n" + MARK + "\n")


def header_in_report():
    if not os.path.exists(REPORT):
        return None
    t = open(REPORT).read()
    return t.split("\n" + MARK)[0] if MARK in t else None


# ------------------------------------------------------------------------------------------------ report

def stat_lines(rows, a, b, out):
    s = {x: ep.delta_stats(delta(rows, a, b, x)) for x in H_BALLS}
    out.append(f"\n{a} - {b}, delta spellF1 por partido, n=40, GT label:\n{rp.HEADER}")
    for x in H_BALLS:
        out.append(rp.fmt_stats(f"{a} - {b}, {x}", s[x]))
    out.append(f"min(g6h, g8h) = {min(s['g6h']['mean'], s['g8h']['mean']):+.4f}")
    return s


def report(fp, out):
    P = out.append
    rows = read_rows(fp)
    times = json.load(open(times_path())) if os.path.exists(times_path()) else {}
    ctrl = control(rows)
    P("\n=== Control: T0 sin huecos contra M1 ===")
    for b, v, ref, dev, ok in ctrl:
        P(f"  T0 {b}: {v:.4f}   M1 {ref:.4f}   desviacion {dev:+.4f}   {'OK' if ok else 'FUERA DE TOLERANCIA'} (<= {CONTROL_TOL})")
    P("\nspellF1 medio de los 40 partidos (label, fuera de pliegue):")
    P(f"{'':<10}" + "".join(f"{b:>9}" for b in EVAL_BALLS))
    for name in ("B3", "B3_t1.6", "T0", "T1"):
        cells = []
        for b in EVAL_BALLS:
            f1 = f1_of(rows, name, b)
            cells.append(f"{np.mean(list(f1.values())):>9.4f}" if f1 else f"{'-':>9}")
        P(f"{name:<10}" + "".join(cells))
    P("(T1 solo se entrena y evalua con huecos; g6 / g8 sin huecos son el control de T0 y de la base)")
    stats = {}
    for a, b in (("T0", "B3_t1.6"), ("T1", "B3_t1.6"), ("T1", "T0")):
        stats[(a, b)] = stat_lines(rows, a, b, out)
    s6, s8 = stats[("T1", "B3_t1.6")]["g6h"], stats[("T1", "B3_t1.6")]["g8h"]
    P("\n=== Veredicto 8.20 de T1 contra la base (B3 + tau 1,6) en PFF+h ===")
    conds, ok = rp.verdict(s6, s8)
    for c, o in conds:
        P(f"  [{'x' if o else ' '}] {c}")
    P(f"VEREDICTO 8.20 T1 contra la base: {'ENTRA' if ok else 'NO ENTRA'} (aplicado literalmente, informativo)")
    label, why = transfer_verdict(s6, s8)
    P("\n=== VEREDICTO SEGUN LA REGLA PREINSCRITA ===")
    P(f"{label}: {why}. min(g6h, g8h) de T1 - base = {min(s6['mean'], s8['mean']):+.4f}; IC95 inferior g6h {s6['lo']:+.4f}, g8h {s8['lo']:+.4f}.")
    P("\n=== Contadores (suma de los 40 partidos; por minuto = por minuto vivo del GT) y D falsa ===")
    P("D falsa = fraccion de los frames vivos de la GT sin balon (todos / solo los de huecos inyectados) que la serie marca como D. La GT es 0 por definicion.")
    for b in EVAL_BALLS:
        rr_ = {n: [r for r in rows if r["modelo"] == n and r["balon"] == b] for n in ("GT", "B3", "B3_t1.6", "T0", "T1")}
        gt_live = sum(float(r["live_gt"]) for r in rr_["GT"])
        n_gt = sum(int(r["n_pred"]) for r in rr_["GT"])
        P(f"-- balon {b}: GT {n_gt} posesiones, {gt_live:.1f} min vivos; frames vivos sin balon {sum(int(r['n_live_nb']) for r in rr_['GT'])}, de huecos inyectados {sum(int(r['n_hole']) for r in rr_['GT'])}")
        P(f"{'serie':<10}{'posesiones':>11}{'pred/GT':>9}{'<0,5 s /min':>13}{'<1 s /min':>11}{'<2 s /min':>11}{'D falsa':>10}{'D falsa huecos':>16}")
        for n, rs in rr_.items():
            if not rs:
                continue
            npred = sum(int(r["n_pred"]) for r in rs)
            c = [sum(int(r[k]) for r in rs) / gt_live for k in ("c05", "c10", "c20")]
            nb, df = sum(int(r["n_live_nb"]) for r in rs), sum(int(r["n_dfalse_nb"]) for r in rs)
            nh, dh = sum(int(r["n_hole"]) for r in rs), sum(int(r["n_dfalse_hole"]) for r in rs)
            P(f"{n:<10}{npred:>11}{npred / n_gt:>9.2f}{c[0]:>13.2f}{c[1]:>11.2f}{c[2]:>11.2f}{df / nb if nb else float('nan'):>10.3f}"
              f"{(dh / nh if nh else float('nan')):>16.3f}")
    tr = {k: v for k, v in times.items() if k.startswith(("T0_", "T1_"))}
    P(f"\nTiempos: {len(tr)} entrenamientos, {sum(tr.values()) / 60:.1f} min de entrenamiento sumado; "
      f"{', '.join(f'{k} {v:.0f}s' for k, v in tr.items())}; pared de la corrida que los produjo: {times.get('total_wall_s', float('nan')) / 60:.1f} min "
      f"(features h incluidas: {times.get('features_h_s', float('nan')) / 60:.1f} min).")
    P("\n=== Caminos Viterbi y mascaras guardados (para la descomposicion D contra H/A) ===")
    P(f"Carpeta: {CAMINOS}")
    P("  <T0|T1>__<balon>__<id>.npy   camino Viterbi (lambda_ha = 8, lambda_d = 16) de un partido y balon: np.load -> (N,) int8, 0 = H, 1 = A, 2 = D (mo.CODE; mo.STATES = ('H', 'A', 'D')).")
    P("      T0: balon en g6, g8 (sin huecos) y g6h, g8h (con huecos); T1: g6h, g8h. Fuera de pliegue: el modelo de cada partido es el del pliegue que lo contiene (rp.FOLDS).")
    P("  huecos__<balon>__<id>.npy    mascara de frames con hueco inyectado, balon en g6h, g8h: np.load -> (N,) bool (True = el balon se puso a NaN en esa variante; g6 / g8 no tienen: todo False).")
    P("  Alineacion: N = frames de los periodos 1 y 2 de PFF en el orden de bancos/pff.py (ronda_pff.load_match, que recorta la prorroga), a 25 fps, identicos entre g6, g8, g6h, g8h y T0 / T1.")
    P("  Periodo, GT y balon: <id>__<balon>.npz de modelo_cache/v1 (g6, g8; solo lectura) o de modelo_cache/transfer (g6h, g8h); claves `per` (N,) int8 periodo 1 / 2, `y` (N,) int8 0/1/2 = H/A/D de `label` (la GT de la etiqueta),")
    P("  `gt_label` y `gt_poss` (la GT como la lee el evaluador, con 'X' en el primer frame del periodo 2) y, en g6h / g8h, `hole` (la misma mascara). Los rasgos estan en <id>__<balon>.npy, columna 'ball_present' de `names`:")
    P("  balon visible = columna > 0.5; B3 y B3 + absorcion: claves `b3`, `b3_t08`, `b3_t16` (0/1/2) del mismo npz, calculadas sobre el balon de esa variante.")
    P(f"  Modelos (boosters de LightGBM): {MODELOS}/<T0|T1>_k<pliegue 0..4>.txt")


# ------------------------------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procesos", type=int, default=min(9, os.cpu_count() or 1))
    ap.add_argument("--hilos", type=int, default=min(9, os.cpu_count() or 1))
    ap.add_argument("--rehacer", action="store_true")
    ap.add_argument("--solo-resumen", action="store_true")
    ap.add_argument("--calibracion", action="store_true", help="stage 1 only: calibration, check against the challenge, preregistered rule")
    ap.add_argument("--cronometro", action="store_true", help="time ONE training of 32 matches (T0, fold 0), nothing else")
    args = ap.parse_args()
    assert args.hilos <= 9 and args.procesos <= 9, "at most 9 cores"
    pool = mp.get_context("fork").Pool(args.procesos)
    t0 = time.time()
    if not args.solo_resumen:
        if args.calibracion or (header_in_report() is None and not args.cronometro):
            text, _ = calibration_text(pool)                  # first: the calibration and the rule are on paper before any model is trained
            write_header(text)
            print(text)
            if args.calibracion:
                return
        build_all(pool, args.rehacer)
        add_time("features_h_s", time.time() - t0)
        names = list(np.load(path_meta_t(rp.FOLDS[0][0], "g6h"))["names"])
        if args.cronometro:
            train = [m for q, f in enumerate(rp.FOLDS) if q != 0 for m in f]
            _, secs, nrows = train_model_t(train, TRAIN["T0"], args.hilos, names)
            print(f"one training (32 matches, {nrows} rows, {args.hilos} threads): {secs:.0f}s")
            return
        fp = fp_models()
        for k in range(len(rp.FOLDS)):
            run_stage(pool, "T0", k, fp, args.hilos, names, args.rehacer)
        ctrl = control(read_rows(fp, ("T0",)))
        print("control T0 sin huecos vs M1: " + "; ".join(f"{b} {v:.4f} (M1 {r:.4f}, {d:+.4f})" for b, v, r, d, _ in ctrl), file=sys.stderr, flush=True)
        if not all(ok for *_, ok in ctrl):
            raise SystemExit(f"CONTROL FAILED: T0 without holes deviates more than {CONTROL_TOL} from M1; stopping before T1 (report it)")
        for k in range(len(rp.FOLDS)):
            run_stage(pool, "T1", k, fp, args.hilos, names, args.rehacer)
        add_time("total_wall_s", time.time() - t0)
    head = header_in_report()
    if head is None:
        raise SystemExit("the report has no preregistered header: run without --solo-resumen")
    out = []
    report(fp_models(), out)
    text = head + "\n\n" + MARK + "\n" + "\n".join(out)
    print("\n".join(out))
    with open(REPORT, "w") as f:
        f.write(text + "\n")
    print(f"\n-> {REPORT}")


if __name__ == "__main__":
    main()
