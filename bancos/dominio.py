"""Domain shift of the M1 / T1 features between the PFF development bank and the challenge match (before taking M1 to the CSV).

    challenge/.venv/bin/python bancos/dominio.py [--procesos P] [--hilos T] [--paso N] [--solo-resumen]

Question: are the features of M1 distinguishable between the challenge (Barça - Athletic, automatic tracking, one match) and PFF (40 development
matches)? Telling two matches apart is always easy, so the figure that counts is the AUC of challenge-vs-PFF against the distribution of
"PFF match i vs the other 39", with the same classifier, the same frame subsampling and the same validation. Measurement only: no fix is tried.

Inputs (no ground truth enters any number of the challenge side):
  * Challenge: limpieza.clean_atd (no smoothing, as src/predecir_reto.py and bancos/regimen_reto.py), period split at the second-half start, then
    the constructor of bancos/modelo_transfer.build_one_h without holes: rp.build_method_ctx -> rp.pipeline (B3 + touches) -> absorcion tau 0.8 ->
    modelo.build_features. The ball of the challenge is the cleaned raw ball (the regime of PFF g6, see regimen_reto), no h holes (it has its own).
  * PFF: the features of the h variant g6h (the ones T1 trains on) from src/out/modelo_cache/transfer/ when the fingerprint of
    modelo_transfer.fp_features() holds; a match whose file is stale is rebuilt in memory with the same constructor (nothing is written to the
    caches of other scripts). Only development matches (pff_split.load_dev guards the held-out ones).
  * T1 boosters (src/out/modelo_cache/transfer/modelos/T1_k*.txt) for the importances and for the share of D on the frames without ball; the
    Viterbi paths of T1 out of fold (caminos/T1__g6h__<mid>.npy) for the same figure in PFF.

Classifier (reference and challenge identical): LightGBM binary, 100 rounds, positives = the target match, negatives = the rest of PFF, class
weights balanced, one frame of every STEP. Validation: 5 folds. The negatives are grouped by match (rp.FOLDS, minus the target when it is a PFF
match); a single match cannot be held out as a group, so the target is cut in runs of 100 s dealt round-robin over the 5 folds (the same for the
challenge and for each reference match; contiguous fifths make the clock features anti-learn). Out-of-fold scores of the 5 folds are pooled into one AUC.
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
import features_reto as fr  # noqa: E402
import inyeccion_huecos as ih  # noqa: E402
import modelo as mo  # noqa: E402
import modelo_pff as mp_  # noqa: E402
import modelo_transfer as mt  # noqa: E402
import pff_split  # noqa: E402
import regimen_reto as rr  # noqa: E402
import ronda_pff as rp  # noqa: E402

CACHE = os.path.join(onside_paths.OUT_DIR, "modelo_cache", "dominio")
REPORT = os.path.join(onside_paths.OUT_DIR, "dominio_informe.txt")
MARK = "=== RESULTADOS ==="
BALL = "g6h"                                    # the PFF variant T1 trains on and the one the rule is applied to
STEP = 10                                       # frame subsampling, the same for the challenge, the reference and the univariate AUCs
K = len(rp.FOLDS)                               # 5 folds
DOM_PARAMS = {"objective": "binary", "num_leaves": 31, "learning_rate": 0.1, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
              "min_data_in_leaf": 100, "max_bin": 63, "verbosity": -1, "seed": 20261029, "deterministic": True, "force_row_wise": True}
DOM_ROUNDS = 100
BLOCK = 250                                     # rows (of the subsample) of one time block of the target match: 250 x STEP / 25 fps = 100 s
EXCESS = 0.10                                   # rule: univariate AUC of the challenge over the p90 of the reference by >= EXCESS
TOP_N = 15
RISK_BASES = ("ball_present", "ball_since_visible_s", "ball_until_visible_s", "b3_dead", "b3abs_dead")
QUANTS = (0.05, 0.25, 0.5, 0.75, 0.95)
LAMBDA = mt.LAMBDA

RULE = [
    "REGLA (fijada antes de medir, tal cual):",
    f"  Un rasgo queda MARCADO si su AUC univariada en el reto supera el p90 de la referencia por >= {EXCESS:.2f} Y esta entre los {TOP_N} mas importantes de M1 o en la lista de riesgo.",
    "  Veredicto: 'transferible' si ningun rasgo queda marcado y la AUC global esta por debajo del maximo de la referencia; 'con desplazamiento' en cualquier otro caso, con la lista de rasgos marcados.",
    "  No se propone ni se prueba ningun arreglo: solo se mide.",
]


# ------------------------------------------------------------------------------------------------ pure helpers (tested)

def risk_features(names):
    """Names of the risk list: the ball presence, the time since / until the ball is visible, B3 dead and B3 + absorption dead, with their windows."""
    out = []
    for nm in names:
        base = nm.split("@")[0]
        if base in RISK_BASES:
            out.append(nm)
    return out


def rank_columns(X):
    """Average ranks (1-based) of every column; NaN counts as a value below every finite one (all NaN tie at the bottom)."""
    from scipy.stats import rankdata
    X = np.asarray(X, float)
    out = np.empty(X.shape, float)
    for j in range(X.shape[1]):
        col = np.where(np.isfinite(X[:, j]), X[:, j], -np.inf)
        out[:, j] = rankdata(col, method="average")
    return out


def auc_from_ranks(R, mask):
    """Mann-Whitney AUC per column of the rows in `mask` (positives) against the others, from the ranks R of the pooled rows. (F,) in [0, 1]."""
    mask = np.asarray(mask, bool)
    n1, n0 = int(mask.sum()), int((~mask).sum())
    return (R[mask].sum(0) - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def direction_free(a):
    """max(AUC, 1 - AUC): how well a feature separates the two sets whichever side is larger."""
    a = np.asarray(a, float)
    return np.maximum(a, 1.0 - a)


def univariate_vs_rest(X, gid, targets):
    """{target: (F,) direction-free AUC of the rows with gid == target against the rows with any other gid}, from one ranking of the pool."""
    R = rank_columns(X)
    return {t: direction_free(auc_from_ranks(R, gid == t)) for t in targets}


def univariate_vs_pool(Xt, Xp):
    """(F,) direction-free AUC of the target rows Xt against the pool Xp."""
    R = rank_columns(np.concatenate([Xp, Xt]))
    mask = np.zeros(len(R), bool)
    mask[len(Xp):] = True
    return direction_free(auc_from_ranks(R, mask))


def time_blocks(n, k=K, size=BLOCK):
    """Fold (0..k-1) of n subsampled frames in time order: consecutive runs of `size` rows dealt round-robin over the k folds. Interleaved, not
    contiguous: with k contiguous thirds of the match the clock features of the held-out block are outside what the training positives cover and
    the AUC falls below 0.5 (measured on a first run). A run is long enough (default 100 s) for the 2 s windows not to cross folds."""
    return (np.arange(n) // size) % k


def group_folds(gid, target=None, folds=rp.FOLDS):
    """Fold (0..K-1) of every row of the negatives from its match id (`folds` = rp.FOLDS); the rows of `target` (a PFF match) get -1."""
    fold_of = {m: k for k, f in enumerate(folds) for m in f}
    out = np.array([fold_of[g] for g in gid])
    if target is not None:
        out[gid == target] = -1
    return out


def exceed_marks(auc_reto, ref_p90, important, risk, excess=EXCESS):
    """Per feature: (excess over the p90 of the reference, marked). Marked = excess >= `excess` and (important or risk)."""
    exc = np.asarray(auc_reto, float) - np.asarray(ref_p90, float)
    return exc, (exc >= excess - 1e-12) & (np.asarray(important, bool) | np.asarray(risk, bool))


def verdict(n_marked, auc_global, ref_max):
    """('transferible' | 'con desplazamiento', reason)."""
    if n_marked == 0 and auc_global < ref_max:
        return "transferible", "ningun rasgo marcado y la AUC global del reto esta por debajo del maximo de la referencia"
    why = []
    if n_marked:
        why.append(f"{n_marked} rasgo(s) marcado(s)")
    if not auc_global < ref_max:
        why.append("la AUC global del reto no esta por debajo del maximo de la referencia")
    return "con desplazamiento", "; ".join(why)


# ------------------------------------------------------------------------------------------------ data

def _build_h_in_memory(mid):
    """Features of (mid, g6h) rebuilt like modelo_transfer.build_one_h, without writing anything (used only when the cache fingerprint is stale)."""
    d = rp.load_match(mid)
    ball0 = rp.ball_variant(d, mt.BASE_OF[BALL])
    ball, _ = ih.inject_holes(ball0, d["label"], d["period"], mid)
    m = rp.build_method_ctx(ball, d["players_slots"], d["period"], d["pitch"])
    b3, tf, _ = rp.pipeline(m)
    b3a = ab.absorb(b3, mp_.TAUS[0], d["period"])
    X, names = mo.build_features(ball, d["players_slots"], d["period"], d["pitch"], b3, b3a, tf)
    return X, list(names)


def load_pff_rows(mid, fp):
    """(X rows 0::STEP float32, names, source) of one development match: cache when its fingerprint holds, in-memory rebuild otherwise."""
    assert mid in pff_split.DESARROLLO and mid not in pff_split.RESERVADOS
    if mt.feature_ok(mid, BALL, fp):
        names = list(np.load(mt.path_meta_t(mid, BALL))["names"])
        return np.ascontiguousarray(np.load(mt.path_x_t(mid, BALL), mmap_mode="r")[::STEP]), names, "cache"
    X, names = _build_h_in_memory(mid)
    return np.ascontiguousarray(X[::STEP]), names, "recalculado"


# ------------------------------------------------------------------------------------------------ domain classifier

_G = {}                                          # pooled data shared with the workers by fork


def domain_auc(target):
    """Pooled out-of-fold AUC of 'target vs the rest of PFF'. target = 'reto' or a PFF match id. Returns (auc, [AUC per fold])."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    Xp, gid, Xr, threads = _G["Xp"], _G["gid"], _G["Xr"], _G["threads"]
    if target == "reto":
        Xt, Xo, go = Xr, Xp, gid
    else:
        sel = gid == target
        Xt, Xo, go = Xp[sel], Xp[~sel], gid[~sel]
    ft = time_blocks(len(Xt))
    fo = group_folds(go)
    scores, labels, per_fold = [], [], []
    p = dict(DOM_PARAMS, num_threads=threads)
    for k in range(K):
        pos_tr, neg_tr = Xt[ft != k], Xo[fo != k]
        pos_te, neg_te = Xt[ft == k], Xo[fo == k]
        X_tr = np.concatenate([pos_tr, neg_tr])
        y_tr = np.concatenate([np.ones(len(pos_tr)), np.zeros(len(neg_tr))])
        w_tr = np.concatenate([np.full(len(pos_tr), len(neg_tr) / len(pos_tr)), np.ones(len(neg_tr))])
        bst = lgb.train(p, lgb.Dataset(X_tr, label=y_tr, weight=w_tr, free_raw_data=True), num_boost_round=DOM_ROUNDS)
        X_te = np.concatenate([pos_te, neg_te])
        y_te = np.concatenate([np.ones(len(pos_te)), np.zeros(len(neg_te))])
        s = bst.predict(X_te)
        scores.append(s)
        labels.append(y_te)
        per_fold.append(float(roc_auc_score(y_te, s)))
    return float(roc_auc_score(np.concatenate(labels), np.concatenate(scores))), per_fold


def domain_importance(target, threads):
    """Gain share (F,) of one domain classifier fitted on ALL the rows of 'target vs the rest' (no validation: only to see which features carry the separation)."""
    import lightgbm as lgb
    Xp, gid, Xr = _G["Xp"], _G["gid"], _G["Xr"]
    Xt, Xo = (Xr, Xp) if target == "reto" else (Xp[gid == target], Xp[gid != target])
    X = np.concatenate([Xt, Xo])
    y = np.concatenate([np.ones(len(Xt)), np.zeros(len(Xo))])
    w = np.concatenate([np.full(len(Xt), len(Xo) / len(Xt)), np.ones(len(Xo))])
    bst = lgb.train(dict(DOM_PARAMS, num_threads=threads), lgb.Dataset(X, label=y, weight=w), num_boost_round=DOM_ROUNDS)
    g = bst.feature_importance("gain")
    return g / g.sum()


def fp_reference(fp_feat, names):
    h = hashlib.sha256()
    h.update(fp_feat.encode())
    h.update(repr((STEP, K, BLOCK, sorted(DOM_PARAMS.items()), DOM_ROUNDS, rp.FOLDS, BALL, list(names))).encode())
    h.update(mp_._src(domain_auc, time_blocks, group_folds).hexdigest().encode())
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ importances and T1 on the challenge

def importance_share(boosters, names):
    """Mean over the boosters of the gain share of every feature -> (F,)."""
    g = np.array([b.feature_importance("gain") for b in boosters], float)
    assert g.shape[1] == len(names)
    return (g / g.sum(1, keepdims=True)).mean(0)


def m1_importance(names):
    """Same figure from the M1 cache (src/out/modelo_cache/v1/importance_k*.npz), None when it is not there."""
    try:
        zs = [np.load(mp_.importance_path(k)) for k in range(K)]
    except OSError:
        return None
    assert all(list(z["names"]) == list(names) for z in zs)
    g = np.array([z["gain"] for z in zs], float)
    return (g / g.sum(1, keepdims=True)).mean(0)


def dead_share(codes, no_ball):
    """Share of the frames without ball that are D (code 2) and share of D among the frames with ball."""
    return float((codes[no_ball] == 2).mean()), float((codes[~no_ball] == 2).mean())


# ------------------------------------------------------------------------------------------------ report

def _fmt_q(x):
    return "".join(f"{v:>9.3g}" for v in x)


def dist_row(nm, a, b):
    """One feature: quantiles / zero share / NaN share of the challenge a and of PFF b (NaN outside the quantiles)."""
    def one(v):
        f = np.isfinite(v)
        q = np.quantile(v[f], QUANTS) if f.any() else np.full(len(QUANTS), np.nan)
        return q, float((v[f] == 0).sum() / len(v)), float(1 - f.mean())
    qa, za, na = one(a)
    qb, zb, nb = one(b)
    return (f"{nm:<24} reto {_fmt_q(qa)} {za:>7.3f} {na:>7.3f}\n{'':<24} PFF  {_fmt_q(qb)} {zb:>7.3f} {nb:>7.3f}")


def header_text():
    P = []
    P.append("DOMINIO: los rasgos de M1 / T1, PFF (40 partidos de desarrollo) contra el reto (Barca - Athletic)")
    P.append("Informe en dos tiempos: este bloque (diseno y regla) se escribe ANTES de medir; los resultados se anaden debajo de la marca.")
    P.append("")
    P.append("Diseno:")
    P.append("  Rasgos del reto: limpieza.clean_atd sin suavizar (como predecir_reto), luego recortado al formato de PFF con src/reto_periodos.py (periodo 1 = saque 11.185 .. 90.651, "
             "periodo 2 = saque 113.742 .. 190.197; reloj desde el saque; sin previa, descanso ni post), y el constructor de T1 (modelo_transfer.build_one_h sin huecos "
             "anadidos: rp.pipeline -> absorcion tau 0,8 -> modelo.build_features). Sin la GT del reto.")
    P.append(f"  Rasgos de PFF: variante {BALL} (la que entrena T1) de src/out/modelo_cache/transfer/ si la huella de modelo_transfer.fp_features() vale; si no, se recalcula en memoria ese partido.")
    P.append(f"  Submuestreo: 1 de cada {STEP} frames, igual para el reto, la referencia y las AUC univariadas.")
    P.append(f"  Clasificador: LightGBM binario ({DOM_ROUNDS} rondas, {DOM_PARAMS['num_leaves']} hojas, pesos balanceados), positivos = el partido objetivo, negativos = el resto de PFF.")
    P.append(f"  Validacion en {K} pliegues. Negativos agrupados por partido (rp.FOLDS, sin el objetivo si es de PFF). Un partido solo no se puede dejar fuera como grupo: el objetivo se corta en tramos de "
             f"{BLOCK * STEP / 25:.0f} s repartidos por turno entre los {K} pliegues, igual para el reto y para cada partido de la referencia. AUC = puntuaciones fuera de pliegue de los 5 pliegues juntas.")
    P.append("  Nota: una primera corrida con el objetivo cortado en 5 tramos contiguos dio AUC por debajo de 0,5 (los rasgos de reloj quedan fuera de lo que cubren los positivos de entrenamiento); se cambio a tramos intercalados de 100 s antes de aplicar la regla. La regla no cambio.")
    P.append("  Referencia: 'partido PFF i contra los otros 39' para los 40 partidos de desarrollo, mismo clasificador, mismo submuestreo, misma validacion.")
    P.append("  AUC univariada por rasgo: Mann-Whitney sobre los rangos de la muestra, sin direccion (max(AUC, 1 - AUC)); NaN cuenta como un valor por debajo de todos los finitos. "
             "Reto contra PFF (40 partidos); referencia: partido i contra los otros 39 de la misma muestra.")
    P.append("  Lista de riesgo: ball_present, ball_since_visible_s, ball_until_visible_s, b3_dead y b3abs_dead, cada uno con todas sus ventanas (@p / @f).")
    P.append(f"  Los {TOP_N} mas importantes: ganancia media normalizada de los 5 boosters T1 (src/out/modelo_cache/transfer/modelos), que es el modelo que iria al reto; "
             "la misma cuenta con la cache de M1 se informa aparte como sensibilidad.")
    P.append("  Ambito: todas las comparaciones (AUC global, univariadas, secciones c, d y e) solo con frames graduados del reto dentro de los periodos; los 2.934 frames graduados 110.808-113.741 "
             "(todos D en la GT, antes del saque) se reportan aparte en (d).")
    P.append("")
    P.extend(RULE)
    P.append("")
    P.append("CORRECCION DE LA MEDICION (v1, 2026-10-29; la regla no cambia): la primera version (informe guardado como dominio_informe_v0.txt) construyo los rasgos del reto sobre el video entero, con el periodo 1 "
             "desde el frame 0 y el periodo 2 hasta el final. Eso metia 40.462 frames no graduados (previa, descanso y post, sin balon) y dejaba el reloj (t_in_period_s, t_to_period_end_s) con la "
             "duracion del video en vez de la del periodo: la mediana de t_in_period_s era 1.990 s contra 1.520 s en PFF y el reloj llevaba el 79 % de la separacion; ademas la cifra de D sobre "
             "frames sin balon mezclaba esos frames. Ahora el reto tiene el mismo formato de periodos que PFF y se compara solo sobre frames graduados.")
    return "\n".join(P)


def write_header(text):
    with open(REPORT, "w") as f:
        f.write(text + "\n\n" + MARK + "\n")


def header_in_report():
    if not os.path.exists(REPORT):
        return None
    t = open(REPORT).read()
    return t.split("\n" + MARK)[0] if MARK in t else None


# ------------------------------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procesos", type=int, default=3)
    ap.add_argument("--hilos", type=int, default=3, help="LightGBM threads per process (procesos * hilos <= 9)")
    ap.add_argument("--solo-resumen", action="store_true", help="reuse the cached reference AUCs, never train")
    args = ap.parse_args()
    assert args.procesos * args.hilos <= 9, "at most 9 cores"
    t_all = time.time()
    head = header_in_report()
    if head is None:
        head = header_text()
        write_header(head)
    print(head)

    mids = list(pff_split.DESARROLLO)
    assert not set(mids) & set(pff_split.RESERVADOS)
    fp_feat = mt.fp_features()
    t0 = time.time()
    rows, srcs, names = [], [], None
    for mid in mids:
        X, nm, src = load_pff_rows(mid, fp_feat)
        names = names or nm
        assert nm == names
        rows.append(X)
        srcs.append(src)
    Xp = np.concatenate(rows)
    gid = np.concatenate([np.full(len(r), m) for r, m in zip(rows, mids)])
    del rows
    Xr_full, names_r, per_r, info_r = fr.build_reto()
    assert names_r == names, "the challenge features do not have the columns of the PFF features"
    Xr = np.ascontiguousarray(Xr_full[::STEP])
    print(f"datos: PFF {Xp.shape} ({srcs.count('cache')} de cache, {srcs.count('recalculado')} recalculados), reto {Xr.shape} en {time.time() - t0:.0f}s", file=sys.stderr, flush=True)

    # --- reference and global AUC (cached, resumable)
    os.makedirs(CACHE, exist_ok=True)
    fp = fp_reference(fp_feat, names)
    path = os.path.join(CACHE, "auc_global.json")
    saved = json.load(open(path)) if os.path.exists(path) else {}
    saved = saved if saved.get("fp") == fp else {"fp": fp, "auc": {}}
    reto_hash = hashlib.sha256(np.ascontiguousarray(Xr).tobytes()).hexdigest()           # the reference does not depend on the challenge rows, its AUC does
    if saved.get("reto_hash") != reto_hash:
        saved["auc"].pop("reto", None)
        saved["reto_hash"] = reto_hash
    todo = [t for t in ["reto"] + mids if t not in saved["auc"]]
    if todo and args.solo_resumen:
        raise SystemExit(f"--solo-resumen: {len(todo)} AUC missing in {path}")
    if todo:
        _G.update(Xp=Xp, gid=gid, Xr=Xr, threads=args.hilos)
        with mp.get_context("fork").Pool(args.procesos) as pool:
            for t, res in zip(todo, pool.imap(domain_auc, todo)):
                saved["auc"][t] = res
                json.dump(saved, open(path, "w"))
                print(f"  AUC {t}: {res[0]:.4f}  (pliegues {', '.join(f'{v:.3f}' for v in res[1])})  [{time.time() - t_all:.0f}s]", file=sys.stderr, flush=True)
    auc_reto, folds_reto = saved["auc"]["reto"]
    ref = np.array([saved["auc"][m][0] for m in mids])

    # --- univariate AUCs
    uni_reto = univariate_vs_pool(Xr, Xp)
    uni_ref = univariate_vs_rest(Xp, gid, mids)
    uni_ref = np.array([uni_ref[m] for m in mids])                                       # (40, F)
    p50, p90, vmax = np.percentile(uni_ref, 50, axis=0), np.percentile(uni_ref, 90, axis=0), uni_ref.max(0)

    # --- importances, risk list, marks
    boosters = fr.t1_boosters()
    imp = importance_share(boosters, names)
    imp_m1 = m1_importance(names)
    order = np.argsort(-imp)
    top = set(order[:TOP_N].tolist())
    top_m1 = set(np.argsort(-imp_m1)[:TOP_N].tolist()) if imp_m1 is not None else set()
    risk = set(names.index(n) for n in risk_features(names))
    is_top = np.array([j in top for j in range(len(names))])
    is_risk = np.array([j in risk for j in range(len(names))])
    exc, marked = exceed_marks(uni_reto, p90, is_top, is_risk)
    marked_m1 = exceed_marks(uni_reto, p90, np.array([j in top_m1 for j in range(len(names))]), is_risk)[1] if imp_m1 is not None else None
    label, why = verdict(int(marked.sum()), auc_reto, ref.max())

    out = []
    P = out.append
    P("\n=== (a) AUC global: reto contra PFF frente a la referencia 'partido PFF i contra los otros 39' ===")
    P(f"AUC del reto contra PFF: {auc_reto:.4f}   (pliegues, tramos de tiempo repartidos por turno: {', '.join(f'{v:.3f}' for v in folds_reto)})")
    P(f"Referencia (n = {len(ref)} partidos de desarrollo): mediana {np.median(ref):.4f}   p90 {np.percentile(ref, 90):.4f}   maximo {ref.max():.4f}   minimo {ref.min():.4f}   media {ref.mean():.4f}")
    P(f"El reto queda en el percentil {rr.pct_rank(auc_reto, ref):.0f} de la referencia; {int((ref >= auc_reto).sum())} de {len(ref)} partidos tienen una AUC >= la del reto.")
    top_ref = sorted(zip(mids, ref), key=lambda t: -t[1])[:5]
    P("Los 5 partidos de la referencia mas distinguibles: " + ", ".join(f"{m} {a:.4f}" for m, a in top_ref))

    P("\n=== (b) AUC univariada por rasgo (sin direccion), reto contra PFF, y referencia (partido PFF i contra los otros 39) ===")
    P(f"imp = ganancia media de T1 (rango entre parentesis); R = lista de riesgo; T = entre los {TOP_N} mas importantes de T1; M = marcado; m = marcado si los 15 fueran los de la cache de M1.")
    P(f"{'rasgo':<26}{'reto':>7}{'ref p50':>9}{'ref p90':>9}{'ref max':>9}{'exceso p90':>12}{'imp %':>8}{'rango':>7}  flags")
    rank_imp = np.empty(len(names), int)
    rank_imp[order] = np.arange(1, len(names) + 1)
    for j in np.argsort(-exc):
        flags = ("R" if is_risk[j] else "-") + ("T" if is_top[j] else "-") + ("M" if marked[j] else "-") + ("m" if marked_m1 is not None and marked_m1[j] and not marked[j] else "-")
        P(f"{names[j]:<26}{uni_reto[j]:>7.3f}{p50[j]:>9.3f}{p90[j]:>9.3f}{vmax[j]:>9.3f}{exc[j]:>+12.3f}{100 * imp[j]:>8.2f}{rank_imp[j]:>7}  {flags}")
    P(f"\nRasgos con exceso >= {EXCESS:.2f} sobre el p90 de la referencia (sin mirar importancia ni riesgo): {int((exc >= EXCESS).sum())} de {len(names)}.")

    P(f"\n=== (c) Distribucion en el reto y en PFF ({BALL}) de la lista de riesgo y de los {TOP_N} mas importantes de T1 (muestra 1/{STEP}) ===")
    P(f"{'rasgo':<24}      {'q05':>9}{'q25':>9}{'q50':>9}{'q75':>9}{'q95':>9} {'ceros':>7} {'NaN':>7}   (fraccion de frames; ceros = valor exactamente 0)")
    P(f"Top {TOP_N} de T1: " + ", ".join(f"{names[j]} ({100 * imp[j]:.1f} %)" for j in order[:TOP_N]))
    if imp_m1 is not None:
        om = np.argsort(-imp_m1)[:TOP_N]
        P(f"Top {TOP_N} de la cache de M1: " + ", ".join(f"{names[j]} ({100 * imp_m1[j]:.1f} %)" for j in om) + f"   (en comun con T1: {len(top & top_m1)})")
    shown = list(dict.fromkeys([names[j] for j in order[:TOP_N]] + risk_features(names)))
    for nm in shown:
        j = names.index(nm)
        P(dist_row(nm + (" *" if j in risk else ""), Xr[:, j], Xp[:, j]))
    P("(* = en la lista de riesgo)")
    pm_nan = np.array([np.mean(~np.isfinite(Xp[gid == m, names.index('ball_since_visible_s')])) for m in mids])
    P(f"ball_since_visible_s NaN por partido de PFF: min {pm_nan.min():.4f}, mediana {np.median(pm_nan):.4f}, max {pm_nan.max():.4f}; reto {np.mean(~np.isfinite(Xr[:, names.index('ball_since_visible_s')])):.4f}")
    ip = names.index("ball_present")
    pf_absent = np.array([np.mean(Xp[gid == m, ip] < 0.5) for m in mids])
    P(f"frames sin balon por partido de PFF ({BALL}): min {pf_absent.min():.3f}, mediana {np.median(pf_absent):.3f}, max {pf_absent.max():.3f}, media {pf_absent.mean():.3f}; reto {np.mean(Xr_full[:, ip] < 0.5):.3f}")

    # --- (d) frames without ball and T1
    P("\n=== (d) Frames sin balon y lo que predice T1 ===")
    no_ball_r = Xr_full[:, ip] < 0.5
    Ps = [mo.predict_proba(b, Xr_full, names) for b in boosters]
    P_ens = np.mean(Ps, axis=0)
    dec = lambda Pm: fr.decode(Pm, per_r, LAMBDA)      # noqa: E731
    vit_ens, vit_each = dec(P_ens), [dec(Pm) for Pm in Ps]
    arg_ens = P_ens.argmax(1)
    arg_each = [Pm.argmax(1) for Pm in Ps]
    # PFF out of fold: each match with the T1 booster of its fold (argmax on the subsample, Viterbi on the full-rate saved path)
    ip_rows = Xp[:, ip] < 0.5
    arg_p = np.empty(len(Xp), np.int8)
    for k in range(K):
        sel = np.isin(gid, rp.FOLDS[k])
        arg_p[sel] = mo.predict_proba(boosters[k], Xp[sel], names).argmax(1)
    nb_p, nbD_p, wbD_p, nbD_gt = [], [], [], []
    for m in mids:
        path_c = os.path.join(mt.CAMINOS, f"T1__{BALL}__{m}.npy")
        v = np.load(path_c)
        absent = np.load(mt.path_x_t(m, BALL), mmap_mode="r")[:, ip] < 0.5
        y = mt.meta_of(m, BALL)["y"]
        nb_p.append(absent.mean())
        nbD_p.append((v[absent] == 2).mean())
        wbD_p.append((v[~absent] == 2).mean())
        nbD_gt.append((y[absent] == 2).mean())
    nb_p, nbD_p, wbD_p, nbD_gt = map(np.array, (nb_p, nbD_p, wbD_p, nbD_gt))
    P(f"Ambito del reto: solo los {len(no_ball_r)} frames graduados dentro de los dos periodos (periodo 1 = saque .. 90.651, periodo 2 = saque 113.742 .. 190.197), como PFF; fuera quedan "
      f"{info_r['n_video'] - len(no_ball_r)} frames ({info_r['n_video'] - len(no_ball_r) - info_r['pre2_n']} no graduados: previa, descanso y post, mas {info_r['pre2_n']} graduados antes del saque del 2.o tiempo).")
    P(f"Aparte: los {info_r['pre2_n']} frames graduados 110.808-113.741 (antes del saque del 2.o tiempo) son todos D en la GT segun FAQ.md:38-39 (no se lee la GT); el tracking tiene "
      f"{100 * info_r['pre2_no_ball']:.1f} % de ellos sin balon. No entran en ninguna AUC ni en las cifras de abajo; el modelo no los ve (no son de ningun periodo).")
    P(f"Reto: frames sin balon {no_ball_r.mean():.4f} de los frames del ambito ({int(no_ball_r.sum())} de {len(no_ball_r)}). PFF {BALL}: {nb_p.mean():.4f} de media por partido (min {nb_p.min():.3f}, max {nb_p.max():.3f}).")
    P(f"{'':<46}{'reto (T1, 5 boosters)':>24}{'reto (cada booster)':>22}{'PFF g6h (T1 fuera de pliegue)':>32}")
    d_ens, a_ens = dead_share(vit_ens, no_ball_r), dead_share(arg_ens, no_ball_r)
    d_each = np.array([dead_share(v, no_ball_r) for v in vit_each])
    a_each = np.array([dead_share(v, no_ball_r) for v in arg_each])
    a_pff = (float((arg_p[ip_rows] == 2).mean()), float((arg_p[~ip_rows] == 2).mean()))
    P(f"{'D tras Viterbi, sobre los frames SIN balon':<46}{d_ens[0]:>24.3f}{f'{d_each[:, 0].min():.3f} a {d_each[:, 0].max():.3f}':>22}{nbD_p.mean():>25.3f} (min {nbD_p.min():.2f}, max {nbD_p.max():.2f})")
    P(f"{'D tras Viterbi, sobre los frames CON balon':<46}{d_ens[1]:>24.3f}{f'{d_each[:, 1].min():.3f} a {d_each[:, 1].max():.3f}':>22}{wbD_p.mean():>25.3f} (min {wbD_p.min():.2f}, max {wbD_p.max():.2f})")
    P(f"{'D por argmax (sin Viterbi), SIN balon':<46}{a_ens[0]:>24.3f}{f'{a_each[:, 0].min():.3f} a {a_each[:, 0].max():.3f}':>22}{a_pff[0]:>25.3f} (muestra 1/{STEP})")
    P(f"{'D por argmax (sin Viterbi), CON balon':<46}{a_ens[1]:>24.3f}{f'{a_each[:, 1].min():.3f} a {a_each[:, 1].max():.3f}':>22}{a_pff[1]:>25.3f} (muestra 1/{STEP})")
    P(f"En PFF {BALL}, fraccion de los frames sin balon que son D en la GT (solo PFF, lado de evaluacion): media {nbD_gt.mean():.3f} (min {nbD_gt.min():.2f}, max {nbD_gt.max():.2f}). Del reto no se lee la GT.")
    P(f"D predicha en total (Viterbi): reto {np.mean(vit_ens == 2):.3f}; PFF {BALL} fuera de pliegue {np.mean([np.mean(np.load(os.path.join(mt.CAMINOS, f'T1__{BALL}__{m}.npy')) == 2) for m in mids]):.3f}.")
    P(f"Lambda = {LAMBDA} (la de T1). Ensamble = media de las probabilidades de los 5 boosters T1; PFF usa el booster del pliegue de cada partido (fuera de pliegue).")

    # --- what carries the separation (descriptive only)
    _G.update(Xp=Xp, gid=gid, Xr=Xr, threads=args.hilos)
    P("\n=== (e) Que rasgos llevan la separacion (clasificador de dominio ajustado con todos los datos, ganancia; descriptivo, no entra en la regla) ===")
    best_ref = max(mids, key=lambda m: saved["auc"][m][0])
    for tgt, lab in (("reto", "reto"), (best_ref, f"partido {best_ref} (la AUC mas alta de la referencia, {saved['auc'][best_ref][0]:.3f})")):
        gi = domain_importance(tgt, min(9, args.procesos * args.hilos))
        P(f"{lab}: " + ", ".join(f"{names[j]} {100 * gi[j]:.1f} %" for j in np.argsort(-gi)[:10]))

    # --- verdict
    P("\n=== VEREDICTO ===")
    P("\n".join(RULE))
    P(f"\nAUC global del reto {auc_reto:.4f}; maximo de la referencia {ref.max():.4f}; p90 {np.percentile(ref, 90):.4f}.")
    mk = [j for j in np.argsort(-exc) if marked[j]]
    if mk:
        P(f"Rasgos MARCADOS ({len(mk)}):")
        P(f"{'rasgo':<26}{'AUC reto':>10}{'ref p90':>9}{'exceso':>9}{'imp %':>8}{'rango':>7}  motivo")
        for j in mk:
            P(f"{names[j]:<26}{uni_reto[j]:>10.3f}{p90[j]:>9.3f}{exc[j]:>+9.3f}{100 * imp[j]:>8.2f}{rank_imp[j]:>7}  {'riesgo' if is_risk[j] else ''}{' + ' if is_risk[j] and is_top[j] else ''}{'top15' if is_top[j] else ''}")
    else:
        P("Rasgos MARCADOS: ninguno.")
    if marked_m1 is not None:
        extra = [names[j] for j in np.flatnonzero(marked_m1 & ~marked)]
        P(f"Sensibilidad (15 mas importantes segun la cache de M1 en vez de T1): marcados {int(marked_m1.sum())}; ademas de los de arriba: {', '.join(extra) if extra else 'ninguno'}.")
    P(f"VEREDICTO: {label.upper()}: {why}.")
    P(f"\nTiempo de ejecucion: {(time.time() - t_all) / 60:.1f} min. Rasgos de PFF: {srcs.count('cache')} partidos de la cache, {srcs.count('recalculado')} recalculados en memoria. "
      f"Suposiciones: reto con el balon limpiado sin suavizar (regimen PFF g6, sin huecos anadidos); PFF = g6h; importancias de T1.")
    text = "\n".join(out)
    print(text)
    with open(REPORT, "w") as f:
        f.write(head + "\n\n" + MARK + "\n" + text + "\n")
    print(f"\n-> {REPORT}")


if __name__ == "__main__":
    main()
