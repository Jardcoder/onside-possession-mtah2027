"""CSV of the challenge with T1 (M1 trained with ball holes calibrated to the challenge).

    challenge/.venv/bin/python src/predecir_reto_m1.py [--modelo T1|T1w|T1wm] [--lam HA D] [--salida RUTA] [--solo-csv]

Features of the challenge as in bancos/features_reto.py (cleaned ATD without smoothing, cut to the PFF layout of src/reto_periodos.py), probabilities = mean of the 5 T1
boosters (src/out/modelo_cache/transfer/modelos), Viterbi per period with the lambda of src/out/lambda_final.json (bancos/barrido_lambda.py; `--lam` overrides it),
back to the video index: frames 110,808-113,741 (graded, before the 2nd-half kick-off) are D (the ground truth gives them all as D, FAQ.md:38-39), frames outside the graded
blocks are X. Same format as src/predecir_reto.py (`frame,label`, 0-based, 199,319 rows; challenge/starter/baseline.write_submission).

`--modelo T1w` / `--modelo T1wm` swaps the 5 boosters for the round 5 ones (src/out/modelo_cache/ronda5/modelos/<modelo>_k<k>.txt), the lambda for
src/out/lambda_final_<modelo>.json (bancos/ronda5_t1.py --lam-ha ...), the CSV for src/out/reto_m1_<modelo>.csv and the sanity report for
predecir_reto_m1_<modelo>_informe.txt (<modelo> in lower case); without it nothing changes.

Sanity check (NOT a selection): the CSV, the official baseline and the current base (B3 + absorption tau 1.6 s, same cut and same D before the kick-off) are scored with the
official evaluator on the public ground-truth window (7,500 frames, 4 D spells); plus possessions per minute and share of D in graded frames, challenge and PFF (T1 out of fold, g6h / g8h).
"""
import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "bancos"))
import onside_paths  # noqa: E402
onside_paths.add_challenge_to_path()
import absorcion as ab  # noqa: E402
import baseline as bl  # noqa: E402
import features_reto as fr  # noqa: E402
import modelo as mo  # noqa: E402
import modelo_transfer as mt  # noqa: E402
import pff_split  # noqa: E402
import possession_metrics as pm  # noqa: E402
import reto_periodos as rpd  # noqa: E402
import ronda_pff as rp  # noqa: E402
from games import GAME, GAMES, half_video_blocks  # noqa: E402

FPS = 25
OUT = os.path.join(onside_paths.OUT_DIR, "reto_m1_t1.csv")
OUT_BASE = os.path.join(onside_paths.OUT_DIR, "reto_base_b3_t16.csv")
BASELINE = os.path.join(onside_paths.OUT_DIR, "baseline_smooth50.csv")
LAMBDA_FILE = os.path.join(onside_paths.OUT_DIR, "lambda_final.json")
REPORT = os.path.join(onside_paths.OUT_DIR, "predecir_reto_m1_informe.txt")
RONDA5 = os.path.join(onside_paths.OUT_DIR, "modelo_cache", "ronda5")
# model name -> (directory of its 5 boosters and out-of-fold probabilities, lambda file, CSV, sanity report)
MODELS = {
    "T1": (mt.CACHE, LAMBDA_FILE, OUT, REPORT),
    **{m: (RONDA5, os.path.join(onside_paths.OUT_DIR, f"lambda_final_{m.lower()}.json"), os.path.join(onside_paths.OUT_DIR, f"reto_m1_{m.lower()}.csv"),
           os.path.join(onside_paths.OUT_DIR, f"predecir_reto_m1_{m.lower()}_informe.txt")) for m in ("T1w", "T1wm")},
}
GT_PUBLIC = os.path.join(onside_paths.EVALUATION_DIR, "gt", "BAR-ATH_possession_gt_public.csv")
SYMBOLS = {"H", "A", "D", "X"}


# ------------------------------------------------------------------------------------------------ pure helpers (tested)

def graded_mask(n=None, game=GAME):
    """(n,) bool: the graded frames of the video (the two half blocks of games.half_video_blocks: 11,185-90,651 and 110,808-190,197)."""
    m = np.zeros(GAMES[game]["n_video_frames"] if n is None else n, bool)
    for _, a, b in half_video_blocks(game):
        m[a:b + 1] = True
    return m


def check_labels(labels, n=None, game=GAME):
    """Raises AssertionError unless `labels` is a full-length array of H / A / D / X with X exactly outside the graded blocks and D on the frames before the 2nd kick-off."""
    n = GAMES[game]["n_video_frames"] if n is None else n
    labels = np.asarray(labels)
    assert labels.shape == (n,), f"{labels.shape} rows != {n}"
    bad = set(np.unique(labels).tolist()) - SYMBOLS
    assert not bad, f"labels outside H/A/D/X: {sorted(bad)}"
    g = graded_mask(n, game)
    assert (labels[~g] == "X").all(), "frames outside the graded blocks must be X"
    assert (labels[g] != "X").all(), "no X inside the graded blocks"
    lo, hi = rpd.pre_kickoff_2nd(game)
    assert (labels[lo:hi + 1] == "D").all(), "frames before the 2nd-half kick-off must be D"


def write_csv(path, labels):
    """`frame,label` through the starter's writer (the same file format as the baseline and src/predecir_reto.py)."""
    return bl.write_submission(path, [str(x) for x in labels])


def lambda_final(path=LAMBDA_FILE):
    return tuple(json.load(open(path))["lambda"])


def boosters_of(modelo):
    """The 5 outer boosters (fold 0..4) of `modelo`: T1 reads modelo_cache/transfer/modelos (features_reto.t1_boosters), the others modelo_cache/<dir>/modelos."""
    if modelo == "T1":
        return fr.t1_boosters()
    import lightgbm as lgb
    return [lgb.Booster(model_file=os.path.join(MODELS[modelo][0], "modelos", f"{modelo}_k{k}.txt")) for k in range(fr.N_FOLDS)]


def probs_file(modelo, balon, mid):
    """Out-of-fold probabilities of `modelo` on one PFF match and ball."""
    return os.path.join(MODELS[modelo][0], "probs", f"{modelo}__{balon}__{mid}.npy")


def counters(labels, mask=None):
    """(share of D, possessions, possessions per minute, minutes) of an array of H / A / D / X over the frames of `mask` (default: the non-X frames)."""
    labels = np.asarray(labels)
    mask = labels != "X" if mask is None else mask
    codes = pm.label_codes(labels)
    spells = pm.possession_spells(codes, pm.valid_runs(codes))
    minutes = int(mask.sum()) / FPS / 60
    return float((labels[mask] == "D").mean()), len(spells), len(spells) / minutes, minutes


# ------------------------------------------------------------------------------------------------ PFF side (out of fold)

def _pff_row(args):
    mid, balon, lam, modelo = args
    z = mt.meta_of(mid, balon)
    P = np.load(probs_file(modelo, balon, mid)).astype(float)
    t1 = fr.decode(P, z["per"], lam)
    runs = pm.valid_runs(z["gt_label"])
    out = {}
    for name, codes in ((modelo, t1), ("B3+abs 1.6", z["b3_t16"]), ("GT", z["y"])):
        out[name] = (int((np.asarray(codes) == 2).sum()), len(pm.possession_spells(np.asarray(codes, np.int8), runs)), len(codes))
    return out


def pff_counters(lam, procesos=9, modelo="T1"):
    """{ball: {series: (share of D, possessions per minute)}} pooled over the 40 development matches."""
    mids = list(pff_split.DESARROLLO)
    assert not set(mids) & set(pff_split.RESERVADOS)
    jobs = [(m, b, lam, modelo) for b in mt.H_BALLS for m in mids]
    with mp.get_context("fork").Pool(procesos) as pool:
        rows = pool.map(_pff_row, jobs)
    res = {}
    for b in mt.H_BALLS:
        rb = [r for (m, bb, _, _), r in zip(jobs, rows) if bb == b]
        res[b] = {}
        for s in rb[0]:
            d, sp, n = (sum(r[s][i] for r in rb) for i in range(3))
            res[b][s] = (d / n, sp / (n / FPS / 60))
    return res


# ------------------------------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", choices=sorted(MODELS), default="T1", help="which 5 boosters decode the challenge (default T1, as always)")
    ap.add_argument("--lam", type=float, nargs=2, metavar=("HA", "D"), help="penalties of the Viterbi; default: src/out/lambda_final.json (bancos/barrido_lambda.py)")
    ap.add_argument("--salida", default=None, help="default: the CSV of the model (reto_m1_t1.csv / reto_m1_t1w.csv / reto_m1_t1wm.csv)")
    ap.add_argument("--solo-csv", action="store_true", help="write and validate the CSV only, no sanity check")
    ap.add_argument("--procesos", type=int, default=9)
    args = ap.parse_args()
    t0 = time.time()
    modelo = args.modelo
    _, lam_file, out_csv, report_path = MODELS[modelo]
    args.salida = args.salida or out_csv
    lam = tuple(args.lam) if args.lam else lambda_final(lam_file)
    print("modelo:", modelo, "lambda:", lam, flush=True)
    X, names, per, info = fr.build_reto()
    print(f"rasgos del reto {X.shape} en {time.time() - t0:.0f}s", flush=True)
    P = fr.ensemble_proba(boosters_of(modelo), X, names)
    letters = np.array(mo.STATES)[fr.decode(P, per, lam)]
    labels = fr.to_video(letters, info)
    check_labels(labels)
    write_csv(args.salida, labels)
    r = subprocess.run([sys.executable, os.path.join(onside_paths.EVALUATION_DIR, "validate_submission.py"), args.salida], capture_output=True, text=True)
    out = [f"CSV: {args.salida}", f"modelo {modelo}; lambda {lam}; filas {len(labels)}", "validate_submission: " + (r.stdout.strip() or r.stderr.strip())[-400:]]
    if not args.solo_csv:
        base = fr.to_video(np.asarray(ab.absorb(info["b3"], fr.BASE_TAU, per), dtype="<U1"), info)
        check_labels(base)
        write_csv(OUT_BASE, base)
        if not os.path.exists(BASELINE):
            bl.write_submission(BASELINE, bl.smooth(bl.build(), 50))
        official = np.array(bl_labels(BASELINE))
        gt = pm.label_codes(pm.read_label_csv(GT_PUBLIC))
        out.append("\nCHEQUEO DE CORDURA (no es seleccion): spellF1 / macroF1 con el evaluador oficial sobre la ventana de GT publica "
                   "(7.500 frames, 4 tramos D; una muestra minima, no se elige nada con esta cifra)")
        series = ((f"{modelo} (este CSV)", labels), ("baseline oficial (suavizado 50)", official), ("base B3 + absorcion tau 1,6", base))
        for name, lab in series:
            res = pm.evaluate(gt, pm.label_codes(lab), bootstrap_n=0)
            out.append(f"  {name:<34} spellF1 {res['spell_f1_mean']:.4f}   macroF1 {res['macro_f1']:.4f}")
        g = graded_mask(len(labels))
        within = g.copy()                                             # graded frames inside the two playing periods
        within[rpd.pre_kickoff_2nd()[0]:rpd.pre_kickoff_2nd()[1] + 1] = False
        out.append(f"\nContadores sobre los {int(g.sum())} frames graduados del reto ({g.sum() / FPS / 60:.1f} min; incluyen los {info['pre2_n']} D antes del saque del 2.o tiempo):")
        out.append(f"{'serie':<34}{'D % graduados':>15}{'D % sin los 2.934':>19}{'posesiones':>12}{'pos / min':>11}")
        for name, lab in series:
            lab = np.where(g, np.asarray(lab), "X")                  # the baseline labels every frame of the video: only the graded ones count
            d, n_sp, per_min, _ = counters(lab, g)
            d_in = float((lab[within] == "D").mean())
            out.append(f"{name:<34}{100 * d:>15.1f}{100 * d_in:>19.1f}{n_sp:>12}{per_min:>11.2f}")
        pf = pff_counters(lam, args.procesos, modelo)
        out.append(f"\nPFF (40 partidos de desarrollo, {modelo} fuera de pliegue con el mismo lambda; frames de los dos periodos; posesiones por minuto de esos frames):")
        out.append(f"{'balon':<8}{'serie':<16}{'D %':>8}{'pos / min':>11}")
        for b, d in pf.items():
            for s, (dshare, ppm) in d.items():
                out.append(f"{b:<8}{s:<16}{100 * dshare:>8.1f}{ppm:>11.2f}")
    out.append(f"\nTiempo: {time.time() - t0:.0f} s.")
    text = "\n".join(out)
    print(text)
    if not args.solo_csv:
        with open(report_path, "w") as f:
            f.write(text + "\n")


def bl_labels(path):
    return [str(x).strip().upper() for x in pm.read_label_csv(path)["label"].values]


if __name__ == "__main__":
    main()
