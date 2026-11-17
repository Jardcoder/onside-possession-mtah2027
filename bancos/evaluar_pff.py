"""PFF World Cup 2022 as decision bank: the accepted ladder on the 40 development matches (bancos/pff_split.py).

    challenge/.venv/bin/python bancos/evaluar_pff.py [--partidos MID ...] [--procesos P] [--solo-resumen]
        [--bootstrap N] [--sin-dfl] [--suavizado MODE ...]
--suavizado g3 g4 ...: only the ball variants "crudo"/"<dfl_degradado.BALL_SMOOTH key>" given (ball smoothed inside each
visible stretch), writing src/out/pff_escalera_<mode>.csv per mode and printing the label and label_poss tables
(calibration of the smoothing against the challenge ball, bancos/calibrar_balon.py --pff).
Writes src/out/pff_escalera.csv (one row per match / ball variant / ground truth / step) and prints the summary tables.

Ladder, cumulative, each step against the previous one (order as accepted in the DFL work):
  B0  starter baseline (reglas "B0": nearest player, +-50 frame majority, D where the ball is absent)
  B1  + carry-over of the label across ball holes, capped at 5 s (reglas fill_gaps "F"; longer holes are D)
  B2  + touches v2, combination r3.5_dv2.0_w5_c1.0x12 with cap 5 s (evaluar_toques.PARAMS, the one chosen in every fold)
  B3  + team of each deflection by direction of the impulse (equipo_toque mode "dir"; it needs no attack direction:
      it is the cosine between the ball impulse and the vector player -> ball)
Ablation: "B1 + dir without touches v2" does not exist. `dir` (equipo_toque) only chooses the team of the deflection
candidates that toques.deflection_candidates produces (it needs the impulse vn - vp of each candidate and the player -> ball
vector at its peak); control touches carry no impulse, and without candidates there is nothing for `dir` to label. The
package v2 + `dir` is therefore measured as one step, B3 - B1 (it equals (B2 - B1) + (B3 - B2) in the mean); the sides B2 - B1
(v2 alone, `dir` off) and B3 - B2 (`dir` given v2) are in the same tables.
Side step B1b = F with cap 10 s: it is the "base" every DFL touches-v2 delta was measured against (B2 - B1b is comparable to
that figure; B2 - B1 is not, the cap moves from 10 to 5).

Inputs of every method: ball_xy, players_slots, period, pitch of the RAW tracking (pff.load smoothed=False, through
pff_split.load_dev). Nothing derived from events or labels enters a prediction (ball_status, is_touch, uncertain, label*,
video_missing, ball_visible, ball_z, players_visible are never passed; tests/test_evaluar_pff.py greps the method modules).
Extra time (periods 3-4: 10506, 10508, 10517) is cut before anything runs: the metric covers periods 1-2 only.

Ball variants: "crudo" = the PFF ball as it is (main run); "g6" = the same ball with the Gaussian sigma 6 smoothing of the
len+g6 DFL bank (dfl_degradado.smooth_ball, applied inside each visible stretch). Ground truths: "label" (last touch,
main) and "label_poss" (possession sequence, informative), both with an X on the first frame of period 2 (make_gt).
`uncertain` frames are NOT masked: the scorer splits possessions at every GT X, so masking them would cut the spells
(see the report); only their share is recorded.
"""
import argparse
import csv
import functools
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
import dfl_degradado as dd  # noqa: E402
import equipo_toque as eq  # noqa: E402
import evaluar_equipo as eqv  # noqa: E402
import evaluar_toques as ev  # noqa: E402
import pff_split  # noqa: E402
import possession_metrics as pm  # noqa: E402
import reglas  # noqa: E402
import toques  # noqa: E402

OUT = os.path.join(onside_paths.OUT_DIR, "pff_escalera.csv")
COMBO = "r3.5_dv2.0_w5_c1.0x12_g5"
V2, CAP = ev.PARAMS[COMBO]                     # detector parameters and hole cap of the combination in force (5 s)
MODE = "dir"
STEPS = ("B0", "B1", "B1b", "B2", "B3")
PAIRS = (("B1", "B0"), ("B2", "B1"), ("B3", "B2"), ("B3", "B1"), ("B3", "B0"), ("B2", "B1b"))     # (step, reference)
BALLS = ("crudo", "g6")
GTS = ("label", "label_poss")
MAIN_RUN = ("crudo", "label")
LAST_PERIOD = 2
TIE = 1e-6                                     # |delta| below this is a tie (same tolerance as the reproducibility check)
BOOT_N, BOOT_SEED = 10000, 20261028
CI98 = 98.3                                    # 1 - 0.05 / 3: Bonferroni interval for the three comparisons of the ablation
FIELDS = ["partido", "balon", "gt", "paso", "spell_f1", "macro_f1", "n_gt", "n_pred", "frames", "pct_uncertain", "recortado"]


def clip_periods(d, last=LAST_PERIOD):
    """Copy of the arrays the ladder and the metric need, restricted to periods 1..last (extra time out)."""
    m = np.asarray(d["period"]) <= last
    return {"ball_xy": d["ball_xy"][m], "players_slots": {k: v[m] for k, v in d["players_slots"].items()},
            "period": d["period"][m], "label": d["label"][m], "label_poss": d["label_poss"][m],
            "uncertain": d["uncertain"][m], "pitch": d["pitch"], "clipped": int((~m).sum())}


def ladder(ball, slots, per, pitch):
    """{step: (N,) "H"/"A"/"D"} from the tracking arrays only (this signature is the whole input of the methods)."""
    present = ~np.isnan(ball[:, 0])
    near = toques.prepare(ball, slots, pitch)
    runs = toques.visible_runs(present, per)
    b0 = reglas.predict("B0", ball, slots, per, pitch)
    cand = toques.deflection_candidates(near, runs, r=V2["r"], dv=V2["dv"], win=V2["win"])
    ctrl = toques.control_touches(near, runs, r_ctrl=V2["r_ctrl"], k=V2["k"])
    t0 = toques.merge_touches((cand["frame"], cand["team"]), ctrl)
    td = toques.merge_touches((cand["frame"], eq.deflection_teams(MODE, cand, near, slots, pitch, r=V2["r"], win=V2["win"])), ctrl)
    kw = dict(max_gap_s=CAP, near=near, base=b0, **V2)
    return {"B0": b0,
            "B1": reglas.fill_gaps(b0, present, per, "F", CAP),
            "B1b": reglas.fill_gaps(b0, present, per, "F", ev.BASE_GAP),
            "B2": toques.predict(ball, slots, per, pitch, touches=t0, **kw),
            "B3": toques.predict(ball, slots, per, pitch, touches=td, **kw)}


def work(mid, balls=BALLS):
    d = clip_periods(pff_split.load_dev(mid))
    rows = []
    for balon in balls:
        ball = d["ball_xy"] if balon == "crudo" else dd.smooth_ball(d["ball_xy"], d["period"], balon)
        preds = ladder(ball, d["players_slots"], d["period"], d["pitch"])
        for gt_key in GTS:
            gt = ev.make_gt(d, gt_key)
            for step in STEPS:
                r = pm.evaluate(gt, pm.label_codes(preds[step]), bootstrap_n=0)
                rows.append({"partido": mid, "balon": balon, "gt": gt_key, "paso": step, "spell_f1": r["spell_f1_mean"],
                             "macro_f1": r["macro_f1"], "n_gt": r["n_gt_spells"], "n_pred": r["n_pred_spells"],
                             "frames": len(d["period"]), "pct_uncertain": round(100 * float(d["uncertain"].mean()), 3),
                             "recortado": d["clipped"]})
    return rows


def load_rows(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("spell_f1", "macro_f1", "n_gt", "n_pred", "frames", "pct_uncertain", "recortado"):
            r[k] = float(r[k])
    return rows


# ------------------------------------------------------------------------------------------------ statistics

def delta_stats(delta, n_boot=BOOT_N, seed=BOOT_SEED):
    """Per-match paired differences -> mean, median, wins / losses / ties, 95 % and 98.3 % bootstrap CI of the mean
    (resampling matches, fixed seed, same resamples for both) and the sd between matches (ddof 1)."""
    delta = np.asarray(delta, float)
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(0, len(delta), (n_boot, len(delta)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    tail = (100 - CI98) / 2
    lo98, hi98 = np.percentile(means, [tail, 100 - tail])
    return {"n": len(delta), "mean": float(delta.mean()), "median": float(np.median(delta)),
            "wins": int((delta > TIE).sum()), "losses": int((delta < -TIE).sum()),
            "ties": int((np.abs(delta) <= TIE).sum()), "lo": float(lo), "hi": float(hi),
            "lo98": float(lo98), "hi98": float(hi98),
            "sd": float(delta.std(ddof=1)) if len(delta) > 1 else float("nan")}


def per_match(rows, balon, gt, key="spell_f1"):
    """{step: {match: value}} of one ball variant and ground truth."""
    out = {s: {} for s in STEPS}
    for r in rows:
        if r["balon"] == balon and r["gt"] == gt:
            out[r["paso"]][r["partido"]] = r[key]
    return out


def table(rows, balon, gt, n_boot=BOOT_N):
    """[(step, reference or None, mean spellF1 of the step, stats of the delta)] for one ball variant and ground truth."""
    v = per_match(rows, balon, gt)
    mids = sorted(v["B0"], key=int)
    out = []
    for step, ref in (("B0", None),) + PAIRS:
        mean = float(np.mean([v[step][m] for m in mids]))
        out.append((step, ref, mean, delta_stats([v[step][m] - v[ref][m] for m in mids], n_boot) if ref else None))
    return out


def print_table(rows, balon, gt, n_boot=BOOT_N, title=""):
    t = table(rows, balon, gt, n_boot)
    n = t[1][3]["n"]
    print(f"\n== PFF {title or f'{balon} / {gt}'} (n={n} partidos de desarrollo) ==")
    print(f"{'paso':<9}{'spellF1':>9}{'delta':>9}{'gana':>6}{'pierde':>7}{'empata':>7}{'IC95 inf':>10}{'IC95 sup':>10}"
          f"{'IC98.3 inf':>12}{'IC98.3 sup':>12}{'mediana':>9}{'sd partidos':>12}")
    for step, ref, mean, s in t:
        if s is None:
            print(f"{step:<9}{mean:>9.4f}")
        else:
            print(f"{step + '-' + ref:<9}{mean:>9.4f}{s['mean']:>+9.4f}{s['wins']:>6}{s['losses']:>7}{s['ties']:>7}"
                  f"{s['lo']:>+10.4f}{s['hi']:>+10.4f}{s['lo98']:>+12.4f}{s['hi98']:>+12.4f}{s['median']:>+9.4f}{s['sd']:>12.4f}")


def print_matches(rows, balon, gt):
    v = per_match(rows, balon, gt)
    meta = {r["partido"]: r for r in rows if r["balon"] == balon and r["gt"] == gt and r["paso"] == "B0"}
    print(f"\n-- por partido, {balon} / {gt}: spellF1 de cada paso (n_gt spells, % incierto, frames recortados de prórroga) --")
    print(f"{'partido':<8}" + "".join(f"{s:>8}" for s in STEPS) + f"{'n_gt':>6}{'%unc':>7}{'recort':>8}")
    for m in sorted(v["B0"], key=int):
        print(f"{m:<8}" + "".join(f"{v[s][m]:>8.4f}" for s in STEPS)
              + f"{meta[m]['n_gt']:>6.0f}{meta[m]['pct_uncertain']:>7.1f}{meta[m]['recortado']:>8.0f}")


# ------------------------------------------------------------------------------------------ DFL reference (CSV only)

DFL_KEY = "spell_f1_ut"                        # last-touch GT, the same definition as the PFF `label`; spell_f1 is BallPossession


def _dfl_deltas(rows, banco, num, den, key=DFL_KEY):
    """Per-match delta (mean over seeds first) of two row selectors on the rows of one bank."""
    acc = {}
    for r in rows:
        if r["banco"] != banco:
            continue
        for name, sel in (("num", num), ("den", den)):
            if sel(r):
                acc.setdefault(r["partido"], {"num": [], "den": []})[name].append(r[key])
    return np.array([np.mean(a["num"]) - np.mean(a["den"]) for a in acc.values() if a["num"] and a["den"]])


def print_dfl_reference(n_boot=BOOT_N):
    """The same deltas from the DFL CSVs already in src/out (nothing is re-run), measured with spell_f1_ut (last touch,
    like the PFF `label`) in the toques and team CSVs. len+g6 is not in them. fase1_paso1.csv only has the BallPossession
    metric, so the B1 - B0 line stays in it and says so."""
    rows_t = ev.load_rows(ev.out_path("m1"))
    rows_e = eqv.load_rows(ev.out_path("m1", stem="equipo"))
    banks = sorted({r["banco"] for r in rows_t})
    print("\n== DFL (src/out/toques_v2_m1.csv, equipo_m1.csv, fase1_paso1.csv), 5 partidos de desarrollo, 5 semillas, "
          f"metrica {DFL_KEY} (ultimo toque; B2 y B3) ==")
    print(f"len+g6 (banco principal) en esas CSV: {'SI' if 'len+g6' in banks else 'NO'}; bancos presentes: {', '.join(banks)}")
    print(f"{'banco':<15}{'B2 - F10':>10}{'gana':>6}{'sd':>8}   {'B3 - B2 (dir fijo, cap 5)':>27}{'gana':>6}{'sd':>8}")
    for b in banks:
        d2 = _dfl_deltas(rows_t, b, lambda r: r["combo"] == COMBO, lambda r: r["combo"] == "base")
        row = f"{b:<15}{d2.mean():>+10.4f}{int((d2 > TIE).sum()):>4}/{len(d2)}{d2.std(ddof=1):>8.4f}   "
        d3 = _dfl_deltas(rows_e, b, lambda r: r["metodo"] == "grid" and r["modo"] == MODE and r["cap"] == CAP,
                         lambda r: r["metodo"] == "grid" and r["modo"] == "T0" and r["cap"] == CAP) \
            if any(r["banco"] == b for r in rows_e) else np.array([])
        row += f"{d3.mean():>+27.4f}{int((d3 > TIE).sum()):>4}/{len(d3)}{d3.std(ddof=1):>8.4f}" if len(d3) else f"{'-':>27}"
        print(row)
    f1 = os.path.join(onside_paths.OUT_DIR, "fase1_paso1.csv")
    with open(f1) as f:
        r1 = [dict(r, partido=r["partido"], spell_f1=float(r["spell_f1_mean"])) for r in csv.DictReader(f)]
    print("B1 - B0 (fase1_paso1.csv: BallPossession spell_f1, NO last touch, no spell_f1_ut in that CSV; balón sin suavizar, "
          "sin m1, 1 semilla; F con tope 5 s y 10 s):")
    for b in sorted({r["banco"] for r in r1}):
        out = []
        for gap in ("5", "10"):
            d1 = _dfl_deltas(r1, b, lambda r, g=gap: r["variante"] == "F" and r["max_gap_s"] == g, lambda r: r["variante"] == "B0",
                             key="spell_f1")
            out.append(f"tope {gap}s {d1.mean():+.4f} ({int((d1 > TIE).sum())}/{len(d1)}, sd {d1.std(ddof=1):.4f})")
        print(f"  {b:<10}" + " | ".join(out))


def run_mode(mode, args):
    """Ladder with the ball variant `mode` only, own CSV, label tables."""
    mids = args.partidos or list(pff_split.DESARROLLO)
    assert not set(mids) & set(pff_split.RESERVADOS), "held-out matches are not scored here"
    path = os.path.join(os.path.dirname(OUT), f"pff_escalera_{mode}.csv")
    if args.solo_resumen:
        rows = [r for r in load_rows(path) if r["partido"] in mids]
    else:
        rows = []
        with mp.get_context("fork").Pool(min(args.procesos, len(mids))) as pool:
            for res in pool.imap_unordered(functools.partial(work, balls=(mode,)), mids):
                rows += res
        rows.sort(key=lambda r: (int(r["partido"]), GTS.index(r["gt"]), STEPS.index(r["paso"])))
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, FIELDS)
            w.writeheader()
            w.writerows(rows)
    for gt in GTS:
        print_table(rows, mode, gt, args.bootstrap, title=f"balón {mode}, GT {gt}")
    print(f"-> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partidos", nargs="*", default=None)
    ap.add_argument("--procesos", type=int, default=min(12, os.cpu_count() or 1))
    ap.add_argument("--solo-resumen", action="store_true", help="re-print the summary from the CSV of a previous run")
    ap.add_argument("--bootstrap", type=int, default=BOOT_N)
    ap.add_argument("--sin-dfl", action="store_true", help="skip the DFL reference tables")
    ap.add_argument("--salida", default=OUT)
    ap.add_argument("--suavizado", nargs="+", default=None, choices=["crudo"] + sorted(dd.BALL_SMOOTH))
    args = ap.parse_args()
    if args.suavizado:
        for mode in args.suavizado:
            run_mode(mode, args)
        return
    mids = args.partidos or list(pff_split.DESARROLLO)
    assert not set(mids) & set(pff_split.RESERVADOS), "held-out matches are not scored here"
    t0 = time.time()
    if args.solo_resumen:
        rows = [r for r in load_rows(args.salida) if r["partido"] in mids]
    else:
        rows = []
        with mp.get_context("fork").Pool(min(args.procesos, len(mids))) as pool:
            for i, res in enumerate(pool.imap_unordered(work, mids), 1):
                rows += res
                print(f"  {i}/{len(mids)} done ({time.time() - t0:.0f}s)", file=sys.stderr)
        rows.sort(key=lambda r: (int(r["partido"]), BALLS.index(r["balon"]), GTS.index(r["gt"]), STEPS.index(r["paso"])))
        os.makedirs(os.path.dirname(args.salida), exist_ok=True)
        with open(args.salida, "w", newline="") as f:
            w = csv.DictWriter(f, FIELDS)
            w.writeheader()
            w.writerows(rows)
    print_table(rows, *MAIN_RUN, args.bootstrap, title="corrida principal: balón crudo, GT label")
    print_matches(rows, *MAIN_RUN)
    for balon, gt in ((b, g) for b in BALLS for g in GTS if (b, g) != MAIN_RUN):
        print_table(rows, balon, gt, args.bootstrap, title=f"informativa: balón {balon}, GT {gt}")
    unc = {r["partido"]: r["pct_uncertain"] for r in rows if r["balon"] == "crudo" and r["gt"] == "label" and r["paso"] == "B0"}
    print(f"\n`uncertain`: {np.mean(list(unc.values())):.1f} % de los frames en media (min {min(unc.values()):.1f}, max "
          f"{max(unc.values()):.1f}); no se enmascara (ver docstring).")
    if not args.sin_dfl:
        print_dfl_reference(args.bootstrap)
    print(f"\ntotal {time.time() - t0:.0f}s -> {args.salida}")


if __name__ == "__main__":
    main()
