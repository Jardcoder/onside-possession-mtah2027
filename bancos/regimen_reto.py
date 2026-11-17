"""Detector regime of the challenge against the PFF bank under g4 / g6 / g8 / raw ball smoothing (decides which smoothing to trust).

    challenge/.venv/bin/python bancos/regimen_reto.py [--procesos P]

Inputs, tracking only. Challenge: the full-match tracking cleaned exactly as src/predecir_reto.py does (limpieza.clean_atd, no
smoothing, period split at the second-half start). PFF: pff_split.load_dev on the 40 development matches, extra time cut, raw ball
or dfl_degradado.smooth_ball inside each visible stretch. Nothing derived from events or reference annotations is read or printed:
every number comes from the predictions of B3 (evaluar_pff ladder, combination in force) and of C2 (src/veto_dueno.py, r_chg 2.5,
d_rel inf, as the round chose it). tests/test_regimen_reto.py greps this file for any such field.

Metrics, per minute of visible ball unless noted (25 fps everywhere):
  toques, toques_dfl, toques_ctl   B3 touches: all / deflections / control
  cambios_b3, cambios_c2           owner changes H<->A in the prediction (D ignored, per period) of B3 / of B3 + C2
  veto_frac                        share of the B3 deflections dropped by C2 (touches B3 - touches C2, only deflections drop)
  pos05, pos1, pos2                B3 possessions (maximal constant H/A run) shorter than 0.5 / 1 / 2 s, per minute of predicted live play
  huecos_h                         ball gaps > 5 s per hour of match
  sin_balon                        share of frames with no ball
The challenge is reported over the whole file (as the predictor sees it) and over the live halves (src/limpieza.live_mask: kickoff to
whistle, from the public game metadata). Distance per smoothing: mean |log(challenge / PFF mean)| over the fixed metric list,
keeping those where both are positive.
"""
import argparse
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import onside_paths  # noqa: E402
onside_paths.add_challenge_to_path()
import dfl_degradado as dd  # noqa: E402
import equipo_toque as eq  # noqa: E402
import evaluar_pff as ep  # noqa: E402
import pff_split  # noqa: E402
import reglas  # noqa: E402
import ronda_pff as rp  # noqa: E402
import toques  # noqa: E402
import veto_dueno as vd  # noqa: E402

SMOOTHS = ("g4", "g6", "g8", "crudo")
C2_PARAMS = (2.5, float("inf"))                # r_chg, d_rel as chosen by the round
FPS = toques.FPS
GAP_S = 5.0
METRICS = ("toques", "toques_dfl", "toques_ctl", "cambios_b3", "cambios_c2", "veto_frac", "pos05", "pos1", "pos2",
           "huecos_h", "sin_balon")
OUT = os.path.join(onside_paths.OUT_DIR, "regimen_reto.csv")


def owner_changes(pred, period, mask):
    """H<->A changes of the owner between consecutive non-D frames of the same period (frames outside `mask` ignored)."""
    n = 0
    for p in np.unique(period):
        s = pred[(period == p) & mask]
        s = s[s != "D"]
        n += int((s[1:] != s[:-1]).sum())
    return n


def possession_lengths(pred, period, mask):
    """Durations (s) of the maximal constant H/A runs, split at D, at period changes and at the borders of `mask`."""
    lab = np.where(mask, pred, "X")
    cut = np.flatnonzero((lab[1:] != lab[:-1]) | (period[1:] != period[:-1])) + 1
    edges = np.concatenate(([0], cut, [len(lab)]))
    return np.array([(b - a) / FPS for a, b in zip(edges[:-1], edges[1:]) if lab[a] in ("H", "A")])


def regime(ball, slots, per, pitch, mask=None):
    """{metric: value} of one match / ball variant from the tracking arrays only."""
    per = np.asarray(per)
    mask = np.ones(len(per), bool) if mask is None else mask
    present = ~np.isnan(ball[:, 0])
    ctx = rp.build_method_ctx(ball, slots, per, pitch)
    dv, win, r = rp.B3_C1
    cand = toques.deflection_candidates(ctx["near"], ctx["runs"], r=r, dv=dv, win=win)
    cf, ct = ctx["ctrl"]
    is_defl = np.concatenate([np.ones(len(cand["frame"]), bool), np.zeros(len(cf), bool)])[
        np.argsort(np.concatenate([cand["frame"], cf]), kind="stable")]
    pred3, tf3, _ = rp.pipeline(ctx)
    pred2, tf2, _ = rp.pipeline(ctx, c2=C2_PARAMS)
    assert len(tf3) == len(is_defl)
    inm = mask[tf3]
    n_defl, n_ctl = int((is_defl & inm).sum()), int((~is_defl & inm).sum())
    kept = set(map(int, tf2))                                      # only deflections can be dropped by C2
    n_veto = sum(1 for f, d in zip(tf3, is_defl) if d and mask[f] and int(f) not in kept)
    vis_min = (present & mask).sum() / FPS / 60
    live = (pred3 != "D") & mask
    plen = possession_lengths(pred3, per, mask)
    live_min = live.sum() / FPS / 60
    gaps = [g for g in reglas.gap_runs(present, per) if g["start"] < len(mask) and mask[g["start"]]
            and (g["end"] - g["start"]) > GAP_S * FPS]
    return {"toques": (n_defl + n_ctl) / vis_min, "toques_dfl": n_defl / vis_min, "toques_ctl": n_ctl / vis_min,
            "cambios_b3": owner_changes(pred3, per, mask) / vis_min, "cambios_c2": owner_changes(pred2, per, mask) / vis_min,
            "veto_frac": n_veto / max(n_defl, 1), "pos05": (plen < 0.5).sum() / live_min, "pos1": (plen < 1).sum() / live_min,
            "pos2": (plen < 2).sum() / live_min, "huecos_h": len(gaps) / (mask.sum() / FPS / 3600),
            "sin_balon": float(1 - present[mask].mean())}


def pff_work(mid):
    raw = pff_split.load_dev(mid)
    keep = np.asarray(raw["period"]) <= ep.LAST_PERIOD            # extra time out, as evaluar_pff
    ball, per = raw["ball_xy"][keep], raw["period"][keep]
    slots = {k: v[keep] for k, v in raw["players_slots"].items()}
    out = {}
    for s in SMOOTHS:
        b = ball if s == "crudo" else dd.smooth_ball(ball, per, s)
        out[s] = regime(b, slots, per, raw["pitch"])
    return mid, out


def challenge_regime():
    import atd
    import limpieza
    d = limpieza.clean_atd()                                       # same cleaning as predecir_reto
    g = atd.GAMES[atd.GAME]
    lo2 = g["halves"][2][0] - g["offsets"][2]
    per = np.where(np.arange(len(d["ball_xy"])) < lo2, 1, 2)
    return {"completo": regime(d["ball_xy"], d["players_slots"], per, atd.PITCH),
            "vivo": regime(d["ball_xy"], d["players_slots"], per, atd.PITCH, mask=limpieza.live_mask())}


def pct_rank(x, ref):
    ref = np.asarray(ref)
    return 100 * ((ref < x).sum() + 0.5 * (ref == x).sum()) / len(ref)


def distance(reto, mean):
    v = [abs(np.log(reto[m] / mean[m])) for m in METRICS if reto[m] > 0 and mean[m] > 0]
    return float(np.mean(v)), len(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procesos", type=int, default=min(12, os.cpu_count() or 1))
    args = ap.parse_args()
    mids = list(pff_split.DESARROLLO)
    assert not set(mids) & set(pff_split.RESERVADOS)
    with mp.get_context("fork").Pool(min(args.procesos, len(mids))) as pool:
        res = dict(pool.imap_unordered(pff_work, mids))
    reto = challenge_regime()
    tab = {s: {m: np.array([res[i][s][m] for i in mids]) for m in METRICS} for s in SMOOTHS}
    rows = ["variante,suavizado,metrica,reto,pff_media,p10,p50,p90,percentil_reto"]
    for var in ("completo", "vivo"):
        r = reto[var]
        print(f"\n===== reto sobre {'el archivo completo (como predecir_reto)' if var == 'completo' else 'las mitades vivas'} "
              f"vs PFF ({len(mids)} partidos de desarrollo) =====")
        for s in SMOOTHS:
            print(f"\n-- PFF {s} --\n{'metrica':<12}{'reto':>9}{'media':>9}{'p10':>9}{'p50':>9}{'p90':>9}{'pct reto':>10}")
            mean = {}
            for m in METRICS:
                v = tab[s][m]
                p10, p50, p90 = np.percentile(v, [10, 50, 90])
                mean[m] = float(v.mean())
                pr = pct_rank(r[m], v)
                print(f"{m:<12}{r[m]:>9.3f}{mean[m]:>9.3f}{p10:>9.3f}{p50:>9.3f}{p90:>9.3f}{pr:>9.0f}%")
                rows.append(f"{var},{s},{m},{r[m]:.5f},{mean[m]:.5f},{p10:.5f},{p50:.5f},{p90:.5f},{pr:.1f}")
        print("\ndistancia media |log(reto/PFF media)|:")
        for s in SMOOTHS:
            dist, k = distance(r, {m: float(tab[s][m].mean()) for m in METRICS})
            print(f"  {s:<6}{dist:.3f}  ({k}/{len(METRICS)} metricas)")
            rows.append(f"{var},{s},distancia,{dist:.5f},,,,,{k}")
    with open(OUT, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
