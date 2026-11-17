"""Does the bank reproduce the ATD gaps by length? challenge | pd_cal+g6 | len+g6 on the development matches.

    challenge/.venv/bin/python bancos/verificar_huecos_largo.py [--semillas N] [--procesos P] [--dead-motion KEY|none]
        [--sin-spell] [--recalibrar]

Classes of absent-ball run length (dfl_degradado.LEN_NAMES), calibration in data/cache/huecos_largo_atd.json
(dfl_degradado.calibrar_por_largo; the challenge numbers below come from that file, never from the ATD frames):
  a  gaps per 90 min and share of the frames, per class, and the total without ball
  b  share of the frames of the gap (>= 1 s from both borders) labelled "D" against the calibrated P(dead | absent)
  c  mean player speed inside the gaps (2 s window, same borders as the calibration)
  d  visible ball more than 1 m outside the pitch (challenge 2.7 %)
  e  diagnostic of the hole cap, not a selection: mean spellF1 of the current base (toques v2 + deflection team "dir") at
     caps 5, 8, 10, 15, 20 s and without cap, in pd_cal+g6 and len+g6
Seeds 0..N-1 pooled over the five development matches. The challenge row of (a) is the raw ball as the calibration reads it,
with the cleaned ball (what the pipeline sees; the banks are measured after limpieza) next to it.
"""
import argparse
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
import dfl  # noqa: E402
import dfl_degradado as dd  # noqa: E402
import equipo_toque as eq  # noqa: E402
import evaluar_equipo as ee  # noqa: E402
import evaluar_toques as ev  # noqa: E402
import muerto  # noqa: E402
import possession_metrics as pm  # noqa: E402
import reglas  # noqa: E402
import toques  # noqa: E402

BANCOS = ("pd_cal+g6", "len+g6")
REF_D = "pd0.60+g6"                     # only in (d): the bank vetoed for its off-pitch ball
CAPS = (5, 8, 10, 15, 20, None)         # None = no cap
NO_CAP_S = 10 ** 6
MIN90 = 90 * 60 * dd.FPS


def work(job):
    banco, mid, seed, dead_motion, spell = job
    d = ev.make_bancos(ev.motion_key(dead_motion))[banco](mid, seed)
    ball, slots, per, pitch = d["ball_xy"], d["players_slots"], d["period"], dd.ATD_PITCH
    n = len(ball)
    cut = np.flatnonzero(np.diff(per)) + 1
    segs = list(zip(np.concatenate(([0], cut)), np.concatenate((cut, [n]))))
    slots_m = np.concatenate([slots["H"], slots["A"]], axis=1) * np.array(pitch)
    out_m = muerto.ball_features(ball, per, pitch)[0]
    vis = ~np.isnan(out_m)
    res = {"banco": banco, "partido": mid, "semilla": seed, "n": n, "absent": int((~d["ball_present"]).sum()),
           "out_gt1": float((out_m[vis] > 1).mean())}
    if banco == BANCOS[1]:
        res["placed_check"] = dd.gap_placement_check(mid, seed)
    if banco in BANCOS:
        res["classes"] = dd.gap_class_stats(~d["ball_present"], segs, slots_m, dd.LEN_WINDOWS[dd.LEN_WIN], label=d["label"])
    if spell and banco in BANCOS:
        gt = ev.make_gt(d)
        present = d["ball_present"]
        near = toques.prepare(ball, slots, pitch)
        b0 = reglas.predict("B0", ball, slots, per, pitch)
        touches = eq.detect_touches("dir", near, slots, pitch, toques.visible_runs(present, per), **ee.V2)
        res["spell"] = {}
        for cap in CAPS:
            pred = toques.predict(ball, slots, per, pitch, max_gap_s=cap or NO_CAP_S, near=near, base=b0, touches=touches, **ee.V2)
            res["spell"][cap] = pm.evaluate(gt, pm.label_codes(pred), bootstrap_n=0)["spell_f1_mean"]
    return res


def pooled(rs, c):
    """Pooled per-class figures of a list of job results (sums over matches and seeds)."""
    n = sum(r["n"] for r in rs)
    cl = [r["classes"][c] for r in rs]
    kept = sum(x["kept"] for x in cl)
    return {"per90": sum(x["n"] for x in cl) * MIN90 / n, "frac": sum(x["frames"] for x in cl) / n,
            "speed": sum(x["mean_speed"] * x["kept"] for x in cl if x["kept"]) / kept if kept else np.nan,
            "d_kept": sum(x["d_kept"] * x["kept"] for x in cl if x["kept"]) / kept if kept else np.nan,
            "d_all": sum(x["d_all"] * x["frames"] for x in cl) / max(sum(x["frames"] for x in cl), 1)}


def fmt(v, f="%.2f"):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f % v


def report(rows, cal, seeds, mids):
    rs = {b: [r for r in rows if r["banco"] == b] for b in BANCOS}
    names, cls = dd.LEN_NAMES, cal["classes"]
    reto_l = cal["limpio"]
    tot = lambda b: 100 * sum(r["absent"] for r in rs[b]) / sum(r["n"] for r in rs[b])  # noqa: E731
    print(f"\n===== a. gaps per 90 min | % of the frames (bank: {len(mids)} matches x {len(seeds)} seeds, after limpieza) =====")
    print(f"{'class':<8}{'reto raw':>16}{'reto limpio':>16}{'pd_cal+g6':>16}{'len+g6':>16}")
    for c, nm in enumerate(names):
        cells = [(cls[c]["per_90min"], 100 * cls[c]["frac_live"]), (reto_l[c]["per_90min"], 100 * reto_l[c]["frac_live"])]
        cells += [(pooled(rs[b], c)["per90"], 100 * pooled(rs[b], c)["frac"]) for b in BANCOS]
        print(f"{nm:<8}" + "".join(f"{x[0]:>9.1f} {x[1]:>5.1f} %" for x in cells))
    print(f"{'no ball':<8}{100 * cal['ball_absent_frac_live']:>14.1f} %{100 * cal['ball_absent_frac_live_limpio']:>14.1f} %"
          f"{tot(BANCOS[0]):>14.1f} %{tot(BANCOS[1]):>14.1f} %")

    oc = [r["placed_check"] for r in rows if r["banco"] == BANCOS[1] and "placed_check" in r]
    if oc:
        print(f"in-play gaps of len+g6 whose final length falls outside the drawn class: {sum(x['out_of_class'] for x in oc)} "
              f"of {sum(x['placed'] for x in oc)} placed (seeds x matches pooled)")
    print("\n===== b. share of the gap frames (>= 1 s from the borders) with label D, against the calibrated P(dead | absent) =====")
    print(f"{'class':<8}{'pd calibrated [95 %]':>24}{'pd_cal+g6':>11}{'len+g6':>9}{'all frames':>12}{'seeds min-max (len)':>22}")
    for c, nm in enumerate(names):
        q = cls[c][dd.LEN_WIN]
        pc, lc = pooled(rs[BANCOS[0]], c), pooled(rs[BANCOS[1]], c)
        per_seed = [pooled([r for r in rs[BANCOS[1]] if r["semilla"] == s], c)["d_kept"] for s in seeds]
        ci = f"{fmt(q['pd'], '%.2f')} [{fmt(q['pd_lo'])}, {fmt(q['pd_hi'])}]"
        print(f"{nm:<8}{ci:>24}{fmt(pc['d_kept']):>11}{fmt(lc['d_kept']):>9}{fmt(lc['d_all']):>12}"
              f"{fmt(np.nanmin(per_seed)) + ' - ' + fmt(np.nanmax(per_seed)):>22}")
    print("(\"all frames\" = len+g6 with the 1 s borders included; pd_cal+g6 all frames: " +
          ", ".join(f"{nm} {fmt(pooled(rs[BANCOS[0]], c)['d_all'])}" for c, nm in enumerate(names)) + ")")

    print(f"\n===== c. mean player speed (m/s) inside the gaps, {dd.LEN_WINDOWS[dd.LEN_WIN] // dd.FPS} s window, >= 1 s from the borders "
          f"(DFL dead {cal['windows'][dd.LEN_WIN]['dfl_mean_speed_dead']:.2f}, alive {cal['windows'][dd.LEN_WIN]['dfl_mean_speed_alive']:.2f}) =====")
    print(f"{'class':<8}{'reto raw':>10}{'reto limpio':>13}{'pd_cal+g6':>11}{'len+g6':>9}{'|pd_cal-reto|':>15}{'|len-reto|':>12}"
          f"{'seeds min-max pd_cal':>22}{'seeds min-max len':>19}")
    for c, nm in enumerate(names):
        rv = cls[c][dd.LEN_WIN]["mean_speed"]
        pc, lc = pooled(rs[BANCOS[0]], c)["speed"], pooled(rs[BANCOS[1]], c)["speed"]
        rng_ = [(lambda v: fmt(np.nanmin(v)) + " - " + fmt(np.nanmax(v)))(
            [pooled([r for r in rs[b] if r["semilla"] == s], c)["speed"] for s in seeds]) for b in BANCOS]
        print(f"{nm:<8}{fmt(rv):>10}{fmt(reto_l[c]['mean_speed']):>13}{fmt(pc):>11}{fmt(lc):>9}"
              f"{fmt(abs(pc - rv) if rv else None):>15}{fmt(abs(lc - rv) if rv else None):>12}{rng_[0]:>22}{rng_[1]:>19}")
    print(f"(reto kept frames per class: " + ", ".join(f"{nm} {cls[c][dd.LEN_WIN]['kept_frames']}" for c, nm in enumerate(names)) + ")")


def report_d(rows):
    print("\n===== d. visible ball more than 1 m outside the pitch, % of the visible frames (challenge: see header) =====")
    print(f"{'bank':<12}{'mean':>8}{'min':>8}{'max':>8}   per match (mean over seeds)")
    for b in (REF_D,) + BANCOS:
        rs = [r for r in rows if r["banco"] == b]
        v = np.array([r["out_gt1"] for r in rs]) * 100
        per = {m: np.mean([r["out_gt1"] for r in rs if r["partido"] == m]) * 100 for m in sorted({r["partido"] for r in rs})}
        print(f"{b:<12}{v.mean():>8.1f}{v.min():>8.1f}{v.max():>8.1f}   " + " ".join(f"{m} {x:.1f}" for m, x in per.items()))


def report_e(rows, seeds, mids):
    print(f"\n===== e. spellF1 of toques v2 {ee.V2_TAG} + team dir by hole cap (diagnostic, {len(mids)} matches x {len(seeds)} seeds) =====")
    lab = [("no cap" if c is None else f"{c} s") for c in CAPS]
    print(f"{'bank':<12}" + "".join(f"{x:>9}" for x in lab))
    for b in BANCOS:
        rs = [r for r in rows if r["banco"] == b and "spell" in r]
        print(f"{b:<12}" + "".join(f"{np.mean([r['spell'][c] for r in rs]):>9.4f}" for c in CAPS))
    print("per match (mean over seeds):")
    for b in BANCOS:
        for m in mids:
            rs = [r for r in rows if r["banco"] == b and r["partido"] == m and "spell" in r]
            print(f"{b:<12}{m:<8}" + "".join(f"{np.mean([r['spell'][c] for r in rs]):>9.4f}" for c in CAPS))
    d = {c: np.mean([r["spell"][c] for r in rows if r["banco"] == "len+g6" and "spell" in r]) -
            np.mean([r["spell"][c] for r in rows if r["banco"] == "pd_cal+g6" and "spell" in r]) for c in CAPS}
    print(f"{'len - pd_cal':<12}" + "".join(f"{d[c]:>+9.4f}" for c in CAPS))
    sd = {b: np.std([np.mean([r["spell"][5] for r in rows if r["banco"] == b and r["semilla"] == s and "spell" in r]) for s in seeds])
          for b in BANCOS}
    print("sd between seeds of the 5 s cap (mean over matches): " + ", ".join(f"{b} {v:.4f}" for b, v in sd.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--semillas", type=int, default=5)
    ap.add_argument("--procesos", type=int, default=min(10, os.cpu_count() or 1))
    ap.add_argument("--dead-motion", default=ev.DEAD_MOTION_DEFAULT, choices=["none"] + list(dd.DEAD_MOTION))
    ap.add_argument("--sin-spell", action="store_true", help="skip (e)")
    ap.add_argument("--recalibrar", action="store_true", help="recompute huecos_largo_atd.json first")
    args = ap.parse_args()
    mids = list(dfl.DESARROLLO)
    assert not set(mids) & set(dfl.RESERVADOS)
    seeds = list(range(args.semillas))
    if args.recalibrar:
        dd.calibrar_por_largo()
    cal = dd.load_calibration_largo()
    t0 = time.time()
    jobs = [(b, m, s, args.dead_motion, not args.sin_spell) for b in BANCOS + (REF_D,) for m in mids for s in seeds]
    rows = []
    with mp.get_context("fork").Pool(min(args.procesos, len(jobs))) as pool:
        for i, res in enumerate(pool.imap_unordered(work, jobs), 1):
            rows.append(res)
            print(f"  {i}/{len(jobs)} done ({time.time() - t0:.0f}s)", file=sys.stderr)
    print(f"challenge: visible ball more than 1 m outside the pitch {100 * cal['ball_out_gt_1m_frac']:.1f} % | dead motion {args.dead_motion}")
    report(rows, cal, seeds, mids)
    report_d(rows)
    if not args.sin_spell:
        report_e(rows, seeds, mids)
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
