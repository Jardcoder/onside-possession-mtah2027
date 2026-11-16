"""Sweep of the Viterbi penalties of T1 on the out-of-fold probabilities: the lambda for the challenge.

    challenge/.venv/bin/python bancos/barrido_lambda.py [--procesos P] [--rehacer]

lambda only changes the decoding, not the training: the out-of-fold probabilities of T1 (bancos/modelo_transfer.py, src/out/modelo_cache/transfer/probs/T1__<ball>__<mid>.npy,
balls g6h and g8h, the model of each match is the booster of the outer fold that holds it) are decoded on the grid lam_ha in {4, 8, 16} x lam_d in {8, 16, 24, 32, 48}
and scored with spellF1 (label, bancos/modelo_pff.pick_lambda criterion: maximise min(mean g6h, mean g8h), ties within evaluar_pff.TIE go to the first of the grid, lam_ha-major).

Nested selection: for every outer fold k, lambda is chosen with the 32 matches of the other 4 folds and evaluated on the 8 matches of k. Reported: the lambda chosen in each fold,
the nested spellF1 against the fixed LAMBDA = (8, 16), whether lam_d lands on an edge of the grid, and the final lambda for the challenge (the most chosen one; a tie goes to the
best mean over the 40 matches). The final lambda is written to src/out/lambda_final.json, which src/predecir_reto_m1.py reads.

Caveat: the probabilities of the matches of the 4 other folds come from boosters that were trained with the matches of fold k (the 5 outer boosters, no inner models as in
bancos/modelo_pff.py). It touches the choice of lambda, not the evaluation: the 8 matches of fold k are always scored with the probabilities of the booster that never saw them.
"""
import argparse
import collections
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
import evaluar_pff as ep  # noqa: E402
import modelo as mo  # noqa: E402
import modelo_pff as mp_  # noqa: E402
import modelo_transfer as mt  # noqa: E402
import pff_split  # noqa: E402
import possession_metrics as pm  # noqa: E402
import ronda_pff as rp  # noqa: E402

LAM_HA = (4, 8, 16)
LAM_D = (8, 16, 24, 32, 48, 64, 96, 128, 192)
GRID = list(itertools.product(LAM_HA, LAM_D))               # lam_ha-major, the order that breaks ties
GRID_HA = np.array([g[0] for g in GRID], float)
GRID_D = np.array([g[1] for g in GRID], float)
FIXED = mt.LAMBDA                                           # (8, 16): the lambda of M1 / T1 so far
BALLS = mt.H_BALLS                                          # ("g6h", "g8h"): what T1 is trained and evaluated on
REPORT = os.path.join(onside_paths.OUT_DIR, "barrido_lambda_informe.txt")
FINAL = os.path.join(onside_paths.OUT_DIR, "lambda_final.json")
CACHE = os.path.join(mt.CACHE, "barrido_lambda.npz")


# ------------------------------------------------------------------------------------------------ pure selection (tested)

def nested_select(f1, fold_of, grid=GRID, pick=mp_.pick_lambda):
    """Nested choice of lambda. f1 (n_matches, n_balls, K) spellF1 of every match and ball at every grid point; fold_of (n_matches,) outer fold of each match.
    For every fold k the lambda is `pick` (maximise min over the balls of the mean over matches, ties to the first of the grid) on the matches of the OTHER folds;
    the matches of k are then read at that lambda. Returns (chosen {k: grid index}, nested (n_matches, n_balls) spellF1 with each match at the lambda of its own fold)."""
    f1, fold_of = np.asarray(f1, float), np.asarray(fold_of)
    chosen, nested = {}, np.zeros(f1.shape[:2])
    for k in sorted(set(fold_of.tolist())):
        held = fold_of == k
        chosen[k] = pick(f1[~held])[0]
        nested[held] = f1[held][:, :, chosen[k]]
    return chosen, nested


def final_choice(chosen, f1, grid=GRID, pick=mp_.pick_lambda):
    """Grid index of the final lambda: the most chosen across the outer folds; a tie goes to the best objective (min of the two ball means) over ALL the matches
    (and, if still tied, to the first of the grid). Returns (index, {index: times chosen})."""
    count = collections.Counter(chosen.values())
    obj = np.minimum(np.asarray(f1, float)[:, 0, :].mean(0), np.asarray(f1, float)[:, 1, :].mean(0))
    top = max(count.values())
    best = sorted(i for i, c in count.items() if c == top)
    return max(best, key=lambda i: (round(obj[i], 12), -i)), dict(count)


def on_edge(value, values):
    """'high' / 'low' / None: whether `value` is the largest / smallest of the grid axis `values`."""
    return "high" if value == max(values) else "low" if value == min(values) else None


# ------------------------------------------------------------------------------------------------ sweep

def probs_path(modelo, balon, mid):
    return os.path.join(mt.PROBS, f"{modelo}__{balon}__{mid}.npy")


def task_sweep(mid, balon):
    """spellF1 (label) at every grid point of the out-of-fold probabilities of T1 on one match and ball -> (K,)."""
    z = mt.meta_of(mid, balon)
    gt = z["gt_label"]
    runs = pm.valid_runs(gt)
    P = np.load(probs_path("T1", balon, mid)).astype(float)
    paths = mo.viterbi(np.log(np.clip(P, 1e-9, 1.0)), z["per"], GRID_HA, GRID_D)
    return np.array([pm.spell_scores(gt, p, runs)["mean_f1"] for p in paths])


def sweep(pool, rehacer=False):
    mids = list(pff_split.DESARROLLO)
    assert not set(mids) & set(pff_split.RESERVADOS)
    fp = mt.fp_models() + repr((GRID, BALLS))
    if not rehacer and os.path.exists(CACHE):
        z = np.load(CACHE)
        if str(z["fp"]) == fp:
            return z["f1"]
    res = pool.starmap(task_sweep, [(m, b) for m in mids for b in BALLS])
    f1 = np.array(res).reshape(len(mids), len(BALLS), len(GRID))
    mp_.atomic_savez(CACHE, fp=fp, f1=f1, mids=np.array(mids), grid=np.array(GRID))
    return f1


# ------------------------------------------------------------------------------------------------ report

def report(f1):
    mids = list(pff_split.DESARROLLO)
    fold_of = np.array([next(k for k, f in enumerate(rp.FOLDS) if m in f) for m in mids])
    chosen, nested = nested_select(f1, fold_of)
    final_i, count = final_choice(chosen, f1)
    fixed_i = GRID.index(tuple(FIXED))
    fixed = f1[:, :, fixed_i]
    obj = np.minimum(f1[:, 0, :].mean(0), f1[:, 1, :].mean(0))
    out = []
    P = out.append
    P("BARRIDO DE LAMBDA DE T1: lam_ha en {4, 8, 16} x lam_d en {8, 16, 24, 32, 48} sobre las probabilidades T1 fuera de pliegue, 40 partidos, g6h y g8h, spellF1 (label).")
    P("Criterio (el de bancos/modelo_pff.pick_lambda): maximizar min(media g6h, media g8h); empate dentro de 1e-6 al primero de la rejilla (lam_ha primero).")
    P("Seleccion anidada: para cada pliegue externo se elige con los 32 partidos de los otros 4 pliegues y se evalua en los 8 del pliegue.")
    P("Reserva: las probabilidades de esos 32 partidos vienen de boosters entrenados con los partidos del pliegue evaluado (hay 5 boosters externos, no modelos internos); toca la eleccion de lambda, no la evaluacion.")
    P("\nspellF1 medio de los 40 partidos por punto de la rejilla (min de g6h y g8h; g6h / g8h entre parentesis):")
    P(f"{'ha / d':>14}" + "".join(f"{d:>26}" for d in LAM_D))
    for a, ha in enumerate(LAM_HA):
        cells = []
        for j, d in enumerate(LAM_D):
            i = a * len(LAM_D) + j
            cells.append(f"{obj[i]:.4f} ({f1[:, 0, i].mean():.4f}/{f1[:, 1, i].mean():.4f})")
        P(f"{ha:>14}" + "".join(f"{c:>26}" for c in cells))
    best = int(np.argmax(obj))
    P(f"Mejor punto con los 40 partidos (informativo, no es la seleccion): lambda {GRID[best]}, objetivo {obj[best]:.4f}.")
    P("\nLambda elegido por pliegue externo (con los 32 partidos de los otros pliegues):")
    for k in sorted(chosen):
        held = fold_of == k
        P(f"  pliegue {k}: lambda {GRID[chosen[k]]}   evaluado en sus {int(held.sum())} partidos: g6h {nested[held][:, 0].mean():.4f}, g8h {nested[held][:, 1].mean():.4f}   "
          f"(con lambda {tuple(FIXED)} fijo: g6h {fixed[held][:, 0].mean():.4f}, g8h {fixed[held][:, 1].mean():.4f})")
    P("\nspellF1 anidado contra lambda fijo (8, 16), 40 partidos fuera de pliegue, delta por partido (anidado - fijo):")
    P(f"{'':<28}{'media g6h':>11}{'media g8h':>11}")
    P(f"{'anidado':<28}{nested[:, 0].mean():>11.4f}{nested[:, 1].mean():>11.4f}")
    P(f"{'fijo (8, 16)':<28}{fixed[:, 0].mean():>11.4f}{fixed[:, 1].mean():>11.4f}")
    P(rp.HEADER)
    stats = {}
    for b, ball in enumerate(BALLS):
        stats[ball] = ep.delta_stats(nested[:, b] - fixed[:, b])
        P(rp.fmt_stats(f"anidado - fijo, {ball}", stats[ball]))
    P(f"min(g6h, g8h) del delta = {min(s['mean'] for s in stats.values()):+.4f}")
    P(f"\nVeces que salio cada lambda en los {len(chosen)} pliegues: " + ", ".join(f"{GRID[i]} x{c}" for i, c in sorted(count.items(), key=lambda t: (-t[1], t[0]))))
    P(f"LAMBDA FINAL para el reto: {GRID[final_i]} ({'el mas elegido' if len(count) == 1 or list(count.values()).count(max(count.values())) == 1 else 'empate entre los mas elegidos; gana el de mejor media con los 40 partidos'}), "
      f"objetivo con los 40 partidos {obj[final_i]:.4f}; (8, 16) fijo da {obj[fixed_i]:.4f}.")
    edges = {GRID[i][1]: on_edge(GRID[i][1], LAM_D) for i in count}
    edge_ha = {GRID[i][0]: on_edge(GRID[i][0], LAM_HA) for i in count}
    P(f"lam_d en el borde de la rejilla ({LAM_D[0]} o {LAM_D[-1]}): "
      + ("; ".join(f"lam_d = {d} ({e})" for d, e in sorted(edges.items()) if e) or "NO") + f". Elegido final: lam_d = {GRID[final_i][1]}"
      + (f" (borde {on_edge(GRID[final_i][1], LAM_D)})" if on_edge(GRID[final_i][1], LAM_D) else "") + ".")
    P("lam_ha en el borde (4 o 16): " + ("; ".join(f"lam_ha = {a} ({e})" for a, e in sorted(edge_ha.items()) if e) or "NO") + ".")
    if LAM_D[-1] in [GRID[i][1] for i in count]:
        P(f"AVISO: lam_d = {LAM_D[-1]} salio elegido en algun pliegue: la rejilla no se amplia mas (instruccion).")
    return "\n".join(out), GRID[final_i], {"lambda": list(GRID[final_i]), "chosen_by_fold": {str(k): list(GRID[i]) for k, i in chosen.items()},
                                           "times_chosen": {str(GRID[i]): c for i, c in count.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procesos", type=int, default=min(9, os.cpu_count() or 1))
    ap.add_argument("--rehacer", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    with mp.get_context("fork").Pool(args.procesos) as pool:
        f1 = sweep(pool, args.rehacer)
    text, lam, info = report(f1)
    text += f"\n\nTiempo: {time.time() - t0:.0f} s."
    print(text)
    with open(REPORT, "w") as f:
        f.write(text + "\n")
    mp_.atomic_json(FINAL, info)
    print(f"\n-> {REPORT}\n-> {FINAL}")


if __name__ == "__main__":
    main()
