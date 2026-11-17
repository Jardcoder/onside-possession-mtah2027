"""Phase 1, step 2b: persistent owner between touches (src/toques.py) on the degraded DFL banks.

    challenge/.venv/bin/python bancos/evaluar_toques.py [--banco NAME] [--partidos MID ...] [--semillas N]
        [--procesos P] [--solo-resumen] [--sin-efecto-b1] [--dead-motion KEY|none]

Banks (name = gap model + smoothing of the bank ball, see dfl_degradado.load_degraded(ball_smooth=...)):
  len+g6         main bank (2026-10-28): gaps coupled by length class (gap_mode "coupled_len", P(dead | gap) calibrated
                 per length, bancos/verificar_huecos_largo.py), Gaussian sigma 6 (the challenge ball is smoothed)
  pd_cal+g6      control with veto: one calibrated gap coupling (0.98) for every length (main bank until 2026-10-28)
  pd0.60+g6      control with veto: P(dead | ball absent) = 0.60 (sees 3 to 6 times more ball off the pitch than the challenge)
  indep+g6       control with veto: gaps independent of the game state
  pd0.60+g8      informative, no veto: Gaussian sigma 8
  pd0.60+causal  informative, no veto: backward-only filter (sigma 15 frames), delay reported by calibrar_balon.py
All banks carry the dead-ball motion `--dead-motion` (default m1, dfl_degradado.DEAD_MOTION); `none` reproduces the old
runs without it. The same smoothed ball feeds the base and the detector. No smoothing code lives here.

Per bank / match / seed it scores the fixed GRID plus the base (F, max_gap_s=10) against the DFL labels, and, in
the main bank, the detector against the DFL event touches (+-12 frames, visible-ball frames only, one-to-one
matching of maximum cardinality, then minimum temporal distance). Then:
  * leave-one-match-out selection: for each development match the combination (detector parameters AND the
    hole cap max_gap_s) with the best mean spellF1 on the other matches (all seeds, main bank) is scored on
    the held-out match, in every bank. The base stays fixed at F with cap 10 under the same smoothing;
  * touch diagnostics: precision / recall with a TP only when the team matches, and without requiring it;
  * effect of the owner-across-holes fix (toques.predict keep_owner): the same selected combination with and
    without it, in every bank.
Development matches only (dfl.RESERVADOS never enter). Writes src/out/toques_v2_<KEY>.csv and toques_v2_<KEY>_sinb1.csv
(toques_v2.csv and toques_v2_sinb1.csv with --dead-motion none).

GRID, fixed before looking at any result (the previous optimum sat on the edge: dv 2.5, r 2.5, win 7, control
1.0 m x 12): 4 x 4 x 3 x 3 detector combinations x 3 caps = 432.
  dv    {1.5, 2, 2.5, 3}      |change of ball velocity| (m/s) between the win frames before and after
  r     {2, 2.5, 3, 3.5}      player-to-ball distance (m) for a deflection to count as a touch
  win   {5, 7, 9}             smoothing / difference window (frames)
  ctrl  {off, (1.0 m, 12 frames), (1.5 m, 12 frames)}   control rule; (0, 0) = deflections only
  cap   {5, 10, 20}           max_gap_s: holes up to this many seconds keep the label / owner, longer ones are D
"""
import argparse
import csv
import itertools
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
import possession_metrics as pm  # noqa: E402
import reglas  # noqa: E402
import toques  # noqa: E402

MAIN = "len+g6"
VETO = ("pd_cal+g6", "pd0.60+g6", "indep+g6")     # controls that cannot lose (mean >= CONTROL_MIN); the other banks are informative
DEAD_MOTION_DEFAULT = "m1"
MAIN_MIN, MAIN_WINS, CONTROL_MIN = 0.015, 4, -0.005


def make_bancos(dead_motion=None):
    """The six banks as {name: (mid, seed) -> match}; dead_motion (None, a dfl_degradado.DEAD_MOTION key or a dict)
    adds the dead-ball motion to all of them (in memory, before the smoothing)."""
    kw = {"dead_motion": dead_motion}
    return {
        "pd0.60+g6": lambda m, s: dd.load_degraded(m, s, "coupled", p_dead_given_absent=0.60, ball_smooth="g6", **kw),
        "pd0.60+g8": lambda m, s: dd.load_degraded(m, s, "coupled", p_dead_given_absent=0.60, ball_smooth="g8", **kw),
        "pd0.60+causal": lambda m, s: dd.load_degraded(m, s, "coupled", p_dead_given_absent=0.60, ball_smooth="causal", **kw),
        "pd_cal+g6": lambda m, s: dd.load_degraded(m, s, "coupled", ball_smooth="g6", **kw),
        "indep+g6": lambda m, s: dd.load_degraded(m, s, "indep", ball_smooth="g6", **kw),
        "len+g6": lambda m, s: dd.load_degraded(m, s, "coupled_len", ball_smooth="g6", **kw),
    }


BANCOS = make_bancos()           # names only: the jobs build their banks with make_bancos(dead_motion)


def motion_key(dead_motion):
    """CLI value -> make_bancos argument ('none' = no dead-ball motion)."""
    return None if dead_motion in (None, "none") else dead_motion


def out_path(dead_motion, stem="toques_v2", suffix=""):
    """src/out/<stem>[_<KEY>]<suffix>.csv; without dead-ball motion the name is the old one."""
    key = motion_key(dead_motion)
    return os.path.join(onside_paths.OUT_DIR, stem + ("" if key is None else f"_{key}") + suffix + ".csv")


def criterio(banco, dm):
    """Verdict for a bank from its per-match dSpell array: True/False, None for informative banks."""
    if banco == MAIN:
        return bool(dm.mean() >= MAIN_MIN and (dm > 0).sum() >= MAIN_WINS)
    if banco in VETO:
        return bool(dm.mean() >= CONTROL_MIN)
    return None


def criterio_txt(banco, dm):
    ok = criterio(banco, dm)
    return "informativo (sin veto)" if ok is None else ("CUMPLE" if ok else "NO CUMPLE")
TOL = 12                       # frames, detector vs DFL event touches
BASE_GAP = 10

GRID_AXES = {"r": (2.0, 2.5, 3.0, 3.5), "dv": (1.5, 2.0, 2.5, 3.0), "win": (5, 7, 9),
             "ctrl": ((0.0, 0), (1.0, 12), (1.5, 12))}
CAPS = (5, 10, 20)
DETECTORS = [{"r": r, "dv": dv, "win": w, "r_ctrl": c[0], "k": c[1]}
             for r, dv, w, c in itertools.product(*GRID_AXES.values())]
assert len(DETECTORS) * len(CAPS) <= 432


def tag(p, cap):
    return "r{r}_dv{dv}_w{win}_c{r_ctrl}x{k}_g{cap}".format(cap=cap, **p)


COMBOS = [(tag(p, g), p, g) for p in DETECTORS for g in CAPS]
TAGS = [c[0] for c in COMBOS]
PARAMS = {t: (p, g) for t, p, g in COMBOS}
FIELDS = ["banco", "partido", "semilla", "combo", "spell_f1", "spell_f1_ut", "n_gt", "n_pred",
          "n_det", "n_ev", "tp_time", "team_ok", "tp_team"]


def make_gt(d, key="label"):
    """Same GT as evaluar_fase1: X on the first frame of each later period so no spell crosses the break."""
    lab = np.array(d[key], dtype="<U1")
    lab[np.flatnonzero(np.diff(d["period"]) != 0) + 1] = "X"
    return pm.label_codes(lab)


_TOUCHES = {}


def event_touches(mid):
    """(frames, team codes) of the DFL event touches of a match (one per frame), for diagnosis only."""
    if mid not in _TOUCHES:
        import pandas as pd
        T = pd.read_parquet(os.path.join(dfl.CACHE, "touches.parquet"))
        T = T[T.match == mid].sort_values("f0").drop_duplicates("f0", keep="first")
        _TOUCHES[mid] = (T.f0.values.astype(int), np.where(T.team.values == "H", toques.HOME, toques.AWAY))
    return _TOUCHES[mid]


def match_one_to_one(tf, tt, ef, et, tol=TOL, same_team=False):
    """One-to-one matching of detected touches (tf, tt) with event touches (ef, et), both sorted by frame.
    A pair needs |tf - ef| <= tol (and the same team if `same_team`). Among all matchings it takes the one with
    the most pairs and, among those, the least total temporal distance (no scipy: on a line an optimal matching
    never crosses, so a DP over each cluster of touches closer than `tol` is exact). Returns [(det index, event index)]."""
    tf, ef = np.asarray(tf), np.asarray(ef)
    if len(tf) == 0 or len(ef) == 0:
        return []
    t = np.concatenate((tf, ef))
    kind = np.concatenate((np.zeros(len(tf), int), np.ones(len(ef), int)))
    idx = np.concatenate((np.arange(len(tf)), np.arange(len(ef))))
    order = np.lexsort((kind, t))
    t, kind, idx = t[order], kind[order], idx[order]
    cut = np.concatenate(([0], np.flatnonzero(np.diff(t) > tol) + 1, [len(t)]))      # clusters cannot interact
    big, pairs = 10 ** 6, []
    for lo, hi in zip(cut[:-1], cut[1:]):
        k = kind[lo:hi]
        di, ei = idx[lo:hi][k == 0], idx[lo:hi][k == 1]
        if len(di) == 0 or len(ei) == 0:
            continue
        n, m = len(di), len(ei)
        w = np.full((n, m), -1, np.int64)                                            # pair value: big - distance
        for a in range(n):
            dist = np.abs(ef[ei] - tf[di[a]])
            ok = dist <= tol
            if same_team:
                ok &= et[ei] == tt[di[a]]
            w[a, ok] = big - dist[ok]
        f = np.zeros((n + 1, m + 1), np.int64)
        for a in range(1, n + 1):
            for b in range(1, m + 1):
                v = max(f[a - 1, b], f[a, b - 1])
                if w[a - 1, b - 1] >= 0:
                    v = max(v, f[a - 1, b - 1] + w[a - 1, b - 1])
                f[a, b] = v
        a, b = n, m
        while a > 0 and b > 0:                                                      # backtrack
            if w[a - 1, b - 1] >= 0 and f[a, b] == f[a - 1, b - 1] + w[a - 1, b - 1]:
                pairs.append((int(di[a - 1]), int(ei[b - 1])))
                a, b = a - 1, b - 1
            elif f[a, b] == f[a - 1, b]:
                a -= 1
            else:
                b -= 1
    return sorted(pairs)


def detector_counts(tf, tt, present, mid):
    """One-to-one counts against the event touches inside visible-ball frames: n_det, n_ev, tp_time (matching
    by time only), team_ok (of those pairs, same team) and tp_team (a separate matching in which a pair needs
    the same team to count)."""
    ef, et = event_touches(mid)
    keep = present[ef]
    ef, et = ef[keep], et[keep]
    pt = match_one_to_one(tf, tt, ef, et)
    pe = match_one_to_one(tf, tt, ef, et, same_team=True)
    return {"n_det": len(tf), "n_ev": len(ef), "tp_time": len(pt),
            "team_ok": sum(int(tt[i] == et[j]) for i, j in pt), "tp_team": len(pe)}


def work(job):
    banco, mid, seed, dead_motion = job
    d = make_bancos(motion_key(dead_motion))[banco](mid, seed)
    ball, slots, per, pitch = d["ball_xy"], d["players_slots"], d["period"], dd.ATD_PITCH
    gt, gt_ut = make_gt(d), make_gt(d, "label_ut")
    present = ~np.isnan(ball[:, 0])
    near = toques.prepare(ball, slots, pitch)
    b0 = reglas.predict("B0", ball, slots, per, pitch)
    runs = toques.visible_runs(present, per)
    rows = []

    def row(name, pred, extra=None):
        r = pm.evaluate(gt, pm.label_codes(pred), bootstrap_n=0)
        u = pm.evaluate(gt_ut, pm.label_codes(pred), bootstrap_n=0)
        out = {"banco": banco, "partido": mid, "semilla": seed, "combo": name, "spell_f1": r["spell_f1_mean"],
               "spell_f1_ut": u["spell_f1_mean"], "n_gt": r["n_gt_spells"], "n_pred": r["n_pred_spells"]}
        out.update(extra or {})
        rows.append(out)

    row("base", reglas.fill_gaps(b0, present, per, "F", BASE_GAP))
    for p in DETECTORS:
        tf, tt = toques.detect_touches(near, runs, **p)
        extra = detector_counts(tf, tt, present, mid) if banco == MAIN else None
        for g in CAPS:
            pred = toques.predict(ball, slots, per, pitch, max_gap_s=g, near=near, base=b0, touches=(tf, tt), **p)
            row(tag(p, g), pred, extra)
    return rows


def work_sin_b1(job):
    """Same combination as the grid but with the owner reset at every visible stretch (keep_owner=False)."""
    banco, mid, seed, name, dead_motion = job
    d = make_bancos(motion_key(dead_motion))[banco](mid, seed)
    p, g = PARAMS[name]
    pred = toques.predict(d["ball_xy"], d["players_slots"], d["period"], dd.ATD_PITCH, max_gap_s=g,
                          keep_owner=False, **p)
    r = pm.evaluate(make_gt(d), pm.label_codes(pred), bootstrap_n=0)
    return [{"banco": banco, "partido": mid, "semilla": seed, "combo": name, "spell_f1": r["spell_f1_mean"]}]


def load_rows(path):
    with open(path) as f:
        out = []
        for r in csv.DictReader(f):
            for k in FIELDS[4:]:
                r[k] = float(r[k]) if r.get(k, "") != "" else np.nan
            r["semilla"] = int(r["semilla"])
            out.append(r)
    return out


def write_rows(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)


def table(rows, banco):
    """{combo: {partido: {semilla: row}}} for one bank."""
    t = {}
    for r in rows:
        if r["banco"] == banco:
            t.setdefault(r["combo"], {}).setdefault(r["partido"], {})[r["semilla"]] = r
    return t


def select(t, mids, seeds):
    """Leave-one-match-out: {held match: combo} with the best mean spellF1 on the other matches."""
    sel = {}
    for held in mids:
        rest = [m for m in mids if m != held]
        score = {c: np.mean([t[c][m][s]["spell_f1"] for m in rest for s in seeds]) for c in TAGS}
        sel[held] = max(score, key=score.get)
    return sel


def summarize(rows, rows_b1, mids, seeds):
    bancos = [b for b in BANCOS if any(r["banco"] == b for r in rows)]
    tabs = {b: table(rows, b) for b in bancos}
    if MAIN not in tabs or len(mids) < 2:
        print("main bank or >1 match missing: no selection")
        return
    sel = select(tabs[MAIN], mids, seeds)
    for key, title in (("spell_f1", "label (BallPossession)"), ("spell_f1_ut", "label_ut (last touch)")):
        print(f"\n===== GT = {title}: held-out match, combination chosen on the other {len(mids) - 1} "
              f"({MAIN}, {len(seeds)} seeds); base = F, cap {BASE_GAP} s, same smoothing =====")
        for b in bancos:
            t = tabs[b]
            per = {}
            for m in sel:
                dif = np.array([t[sel[m]][m][s][key] - t["base"][m][s][key] for s in seeds])
                per[m] = (dif, np.mean([t[sel[m]][m][s]["n_pred"] for s in seeds]),
                          np.mean([t["base"][m][s]["n_pred"] for s in seeds]), t["base"][m][seeds[0]]["n_gt"],
                          np.mean([t["base"][m][s][key] for s in seeds]))
            print(f"\n-- banco {b} --")
            print(f"{'partido':<9}{'base':>8}{'toques':>8}{'dSpell':>9}{'n_pred':>8}{'n_pred_F':>9}{'n_gt':>6}  combo")
            for m, (dif, npd, npb, ngt, base_f) in per.items():
                print(f"{m:<9}{base_f:>8.4f}{base_f + dif.mean():>8.4f}{dif.mean():>+9.4f}{npd:>8.0f}{npb:>9.0f}{ngt:>6.0f}  {sel[m]}")
            dm = np.array([v[0].mean() for v in per.values()])
            seed_sd = np.std([np.mean([v[0][i] for v in per.values()]) for i in range(len(seeds))])
            print(f"media dSpell {dm.mean():+.4f} | gana {int((dm > 0).sum())}/{len(dm)} | sd entre semillas {seed_sd:.4f} | "
                  f"n_pred medio {np.mean([v[1] for v in per.values()]):.0f} (base {np.mean([v[2] for v in per.values()]):.0f}) "
                  f"frente a n_gt {np.mean([v[3] for v in per.values()]):.0f}")
            if key == "spell_f1":
                print("criterio (main: media >= +0,015 y >= 4/5; controles con veto pd0.60+g6 e indep+g6: media >= -0,005):",
                      criterio_txt(b, dm))

    # touch diagnostics, main bank, one-to-one
    t = tabs[MAIN]

    def pr(c, ms):
        agg = {k: sum(t[c][m][s][k] for m in ms for s in seeds) for k in ("n_det", "n_ev", "tp_time", "team_ok", "tp_team")}
        n = len(ms) * len(seeds)
        p0, r0 = agg["tp_time"] / max(agg["n_det"], 1), agg["tp_time"] / max(agg["n_ev"], 1)
        p1, r1 = agg["tp_team"] / max(agg["n_det"], 1), agg["tp_team"] / max(agg["n_ev"], 1)
        return p0, r0, p1, r1, agg["team_ok"] / max(agg["tp_time"], 1), agg["n_det"] / n, agg["n_ev"] / n

    print(f"\n===== touches vs DFL event touches, one-to-one (+-{TOL} frames, visible-ball frames, banco {MAIN}) =====")
    print("tiempo = TP sin exigir equipo | equipo = TP sólo si el equipo coincide (otro emparejamiento) | team = equipo correcto entre los TP de tiempo")
    print(f"{'combo':<34}{'P tiempo':>9}{'R tiempo':>9}{'P equipo':>9}{'R equipo':>9}{'team':>7}{'n_det':>7}{'n_ev':>6}")
    for m in sel:
        p0, r0, p1, r1, tm, nd, ne = pr(sel[m], [m])
        print(f"{sel[m] + ' [' + m + ']':<34}{p0:>9.3f}{r0:>9.3f}{p1:>9.3f}{r1:>9.3f}{tm:>7.3f}{nd:>7.0f}{ne:>6.0f}")
    seen = {}
    for c in TAGS:
        seen.setdefault(c.rsplit("_g", 1)[0], c)                  # detector parameters only (the cap does not change them)
    allg = sorted(seen.values(), key=lambda c: -2 * pr(c, mids)[2] * pr(c, mids)[3] / max(pr(c, mids)[2] + pr(c, mids)[3], 1e-9))
    print("grid ordered by F1 with team (all dev matches pooled), best 5 and worst 3:")
    for c in allg[:5] + allg[-3:]:
        p0, r0, p1, r1, tm, nd, ne = pr(c, mids)
        print(f"{c.rsplit('_g', 1)[0]:<34}{p0:>9.3f}{r0:>9.3f}{p1:>9.3f}{r1:>9.3f}{tm:>7.3f}{nd:>7.0f}{ne:>6.0f}")

    # effect of the owner-across-holes fix
    if rows_b1:
        print("\n===== effect of keeping the owner across holes (same selected combination, spellF1 with fix minus without) =====")
        tb = {b: table(rows_b1, b) for b in bancos if any(r["banco"] == b for r in rows_b1)}
        for b, tt_ in tb.items():
            per = {m: np.mean([tabs[b][sel[m]][m][s]["spell_f1"] - tt_[sel[m]][m][s]["spell_f1"] for s in seeds]) for m in sel}
            print(f"banco {b:<14} media {np.mean(list(per.values())):+.4f} | " +
                  " ".join(f"{m} {v:+.4f}" for m, v in per.items()))
        print("sin el arreglo, dSpell contra la base (media de partidos):")
        for b, tt_ in tb.items():
            d0 = np.mean([np.mean([tt_[sel[m]][m][s]["spell_f1"] - tabs[b]["base"][m][s]["spell_f1"] for s in seeds]) for m in sel])
            d1 = np.mean([np.mean([tabs[b][sel[m]][m][s]["spell_f1"] - tabs[b]["base"][m][s]["spell_f1"] for s in seeds]) for m in sel])
            print(f"banco {b:<14} sin {d0:+.4f} | con {d1:+.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--banco", choices=list(BANCOS), default=None)
    ap.add_argument("--partidos", nargs="*", default=None)
    ap.add_argument("--semillas", type=int, default=1, help="degradation seeds 0..N-1 (default 1 = seed 0)")
    ap.add_argument("--procesos", type=int, default=min(10, os.cpu_count() or 1))
    ap.add_argument("--solo-resumen", action="store_true", help="re-print the summary from the CSVs of a previous run")
    ap.add_argument("--sin-efecto-b1", action="store_true", help="skip the keep_owner=False comparison")
    ap.add_argument("--dead-motion", default=DEAD_MOTION_DEFAULT, choices=["none"] + list(dd.DEAD_MOTION),
                    help="dead-ball motion of every bank (dfl_degradado.DEAD_MOTION key; none = old runs)")
    args = ap.parse_args()
    mids = args.partidos or list(dfl.DESARROLLO)
    assert not set(mids) & set(dfl.RESERVADOS), "held-out matches are not scored here"
    seeds = list(range(args.semillas))
    t0 = time.time()
    rows_b1 = []
    out, out_b1 = out_path(args.dead_motion), out_path(args.dead_motion, suffix="_sinb1")
    if args.solo_resumen:
        rows = load_rows(out)
        if os.path.exists(out_b1):
            rows_b1 = load_rows(out_b1)
    else:
        bancos = [args.banco] if args.banco else list(BANCOS)
        jobs = [(b, m, s, args.dead_motion) for b in bancos for m in mids for s in seeds]
        rows = []
        with mp.Pool(min(args.procesos, len(jobs))) as pool:
            for i, res in enumerate(pool.imap_unordered(work, jobs), 1):
                rows += res
                print(f"  {i}/{len(jobs)} done ({time.time() - t0:.0f}s)", file=sys.stderr)
        write_rows(out, rows)
        if not args.sin_efecto_b1 and MAIN in bancos and len(mids) > 1:
            rows_all = [r for r in rows if r["partido"] in mids]
            sel = select(table(rows_all, MAIN), mids, seeds)
            jobs = [(b, m, s, sel[m], args.dead_motion) for b in bancos for m in mids for s in seeds]
            with mp.Pool(min(args.procesos, len(jobs))) as pool:
                for res in pool.imap_unordered(work_sin_b1, jobs):
                    rows_b1 += res
            write_rows(out_b1, rows_b1)
    rows = [r for r in rows if r["partido"] in mids]
    summarize(rows, rows_b1, mids, seeds)
    print(f"\ntotal {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
