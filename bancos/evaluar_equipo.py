"""Team of each v2 deflection (src/equipo_toque.py) against toques v2, leave-one-match-out, on the five degraded DFL banks.

    challenge/.venv/bin/python bancos/evaluar_equipo.py [--partidos MID ...] [--semillas N] [--procesos P]
        [--dead-motion KEY|none] [--solo-resumen]

Only the team given to each deflection changes. The detector is the v2 combination r3.5_dv2.0_w5_c1.0x12 (the one chosen
in every fold of toques_v2_m1.csv): same touch frames, same control touches, same hole filling. 12 modes, fixed before
looking at any spellF1 (equipo_toque.MODES): T0, lag {2,4,6}, minwin {3,6}, dir, kink, dir+tie {1,2} m, kink+tie {1,2} m.

Fold of held-out match e (main bank, development matches only, 5 seeds each): the (mode, hole cap in {5, 10} s) with the
best mean spellF1 on the other matches is scored on e in every bank. Ties go to the earlier mode of MODES (T0 first).
Reference: toques v2 = the combination chosen by evaluar_toques.select from toques_v2_m1.csv; base = F with cap 10.
Criterion: evaluar_toques.criterio on dSpell(equipo - v2) per match.

Team accuracy (diagnostic, not used for the selection): main bank, all development matches x seeds. Touches matched
one-to-one in time with the DFL event touches (+-12 frames, visible-ball frames; the pairs do not depend on the team, so
they are the same for every mode); accuracy = same team among the matched. Split by what the T0 touch does to the T0
prediction (cap 5 s) as in diagnostico_toques.py: (a) changes the owner, (b) same team as the label just before,
(c) no H/A label before. The partition is fixed by T0 so every mode is read on the same touches. "defl" = deflections only
(the part the modes change); "all" adds the control touches (what diagnostico_toques.py reports).
Writes src/out/equipo_<KEY>.csv (grid rows of every mode and cap, base, v2 and the selected mode).
"""
import argparse
import csv
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
import evaluar_toques as ev  # noqa: E402
import possession_metrics as pm  # noqa: E402
import reglas  # noqa: E402
import toques  # noqa: E402

V2 = {"r": 3.5, "dv": 2.0, "win": 5, "r_ctrl": 1.0, "k": 12}
V2_TAG = "r3.5_dv2.0_w5_c1.0x12"
CAPS = (5, 10)
PART_CAP = 5                                     # cap of the T0 prediction that defines the owner-change partition
GROUPS = "abc"
COUNT_FIELDS = [f"{k}_{g}_{c}" for k in ("all", "defl") for g in GROUPS for c in ("tp", "ok")]
FIELDS = ["banco", "partido", "semilla", "metodo", "modo", "cap", "spell_f1", "spell_f1_ut", "n_gt", "n_pred"] + COUNT_FIELDS
EDGES = {"lag": (2, 6), "minwin": (3, 6), "tie": (1.0, 2.0)}      # first / last value of each parametric family
NUM = ["spell_f1", "spell_f1_ut", "n_gt", "n_pred"] + COUNT_FIELDS


def score(pred, gt, gt_ut):
    r = pm.evaluate(gt, pm.label_codes(pred), bootstrap_n=0)
    u = pm.evaluate(gt_ut, pm.label_codes(pred), bootstrap_n=0)
    return r["spell_f1_mean"], u["spell_f1_mean"], r["n_gt_spells"], r["n_pred_spells"]


def touch_counts(d, ctrl, teams_by_mode, pred0, per, present, mid):
    """{mode: {count field: int}}: matched touches and correct teams per owner-change group (see module docstring)."""
    fd, td = d["frame"], d["team"]
    fc, tc_ = ctrl
    frames = np.concatenate((fd, fc))
    order = np.argsort(frames, kind="stable")           # same order as toques.merge_touches
    tf = frames[order]
    is_defl = order < len(fd)
    t0_all = np.concatenate((td, tc_))[order]
    ef, et = ev.event_touches(mid)
    keep = present[ef]
    ef, et = ef[keep], et[keep]
    pairs = ev.match_one_to_one(tf, t0_all, ef, et)     # time only: independent of the team
    pi = np.array([i for i, _ in pairs], int)
    pj = np.array([j for _, j in pairs], int)
    pf = np.maximum(tf - 1, 0)
    prev = np.where((tf > 0) & (per[pf] == per[tf]), pred0[pf], "D")
    letter = np.array(["H", "A"])[t0_all.astype(int)]
    grp = np.where(prev == "D", "c", np.where(prev == letter, "b", "a"))
    out = {}
    for mode, td_m in teams_by_mode.items():
        t_all = np.concatenate((td_m, tc_))[order]
        c = {}
        for g in GROUPS:
            in_g = (grp[pi] == g) if len(pi) else np.zeros(0, bool)
            ok = (t_all[pi] == et[pj]) if len(pi) else np.zeros(0, bool)
            dflt = is_defl[pi] if len(pi) else np.zeros(0, bool)
            c[f"all_{g}_tp"], c[f"all_{g}_ok"] = int(in_g.sum()), int((in_g & ok).sum())
            c[f"defl_{g}_tp"], c[f"defl_{g}_ok"] = int((in_g & dflt).sum()), int((in_g & dflt & ok).sum())
        out[mode] = c
    return out


def work(job):
    banco, mid, seed, dead_motion = job
    d0 = ev.make_bancos(ev.motion_key(dead_motion))[banco](mid, seed)
    ball, slots, per, pitch = d0["ball_xy"], d0["players_slots"], d0["period"], dd.ATD_PITCH
    gt, gt_ut = ev.make_gt(d0), ev.make_gt(d0, "label_ut")
    present = ~np.isnan(ball[:, 0])
    near = toques.prepare(ball, slots, pitch)
    b0 = reglas.predict("B0", ball, slots, per, pitch)
    runs = toques.visible_runs(present, per)
    cand = toques.deflection_candidates(near, runs, r=V2["r"], dv=V2["dv"], win=V2["win"])
    ctrl = toques.control_touches(near, runs, r_ctrl=V2["r_ctrl"], k=V2["k"])
    rows, cache, teams_by_mode, preds = [], {}, {}, {}
    base = {"banco": banco, "partido": mid, "semilla": seed}
    s, u, ngt, npred = score(reglas.fill_gaps(b0, present, per, "F", ev.BASE_GAP), gt, gt_ut)
    rows.append(dict(base, metodo="base", modo="F", cap=ev.BASE_GAP, spell_f1=s, spell_f1_ut=u, n_gt=ngt, n_pred=npred))
    for mode in eq.MODES:
        teams = eq.deflection_teams(mode, cand, near, slots, pitch, r=V2["r"], win=V2["win"])
        teams_by_mode[mode] = teams
        touches = toques.merge_touches((cand["frame"], teams), ctrl)
        for cap in CAPS:
            key = (teams.tobytes(), cap)                 # modes that give the same teams give the same prediction
            if key not in cache:
                pred = toques.predict(ball, slots, per, pitch, max_gap_s=cap, near=near, base=b0, touches=touches, **V2)
                cache[key] = (pred, score(pred, gt, gt_ut))
            pred, (s, u, ngt, npred) = cache[key]
            preds[(mode, cap)] = pred
            rows.append(dict(base, metodo="grid", modo=mode, cap=cap, spell_f1=s, spell_f1_ut=u, n_gt=ngt, n_pred=npred))
    if banco == ev.MAIN:
        counts = touch_counts(cand, ctrl, teams_by_mode, preds[("T0", PART_CAP)], per, present, mid)
        for r in rows:
            if r["metodo"] == "grid":
                r.update(counts[r["modo"]])
    return rows


# ------------------------------------------------------------------------------------------ CSV and selection

def write_rows(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)


def load_rows(path):
    with open(path) as f:
        out = []
        for r in csv.DictReader(f):
            for k in NUM:
                r[k] = float(r[k]) if r.get(k, "") != "" else np.nan
            r["semilla"], r["cap"] = int(r["semilla"]), int(float(r["cap"]))
            out.append(r)
    return out


def grid_table(rows, banco):
    """{(mode, cap): {match: {seed: row}}} of the grid rows of one bank."""
    t = {}
    for r in rows:
        if r["metodo"] == "grid" and r["banco"] == banco:
            t.setdefault((r["modo"], r["cap"]), {}).setdefault(r["partido"], {})[r["semilla"]] = r
    return t


SETTINGS = [(m, c) for m in eq.MODES for c in CAPS]


def select(t, mids, seeds):
    """Leave-one-match-out: {held match: (mode, cap)} with the best mean spellF1 on the other matches (ties: first in MODES)."""
    sel = {}
    for held in mids:
        rest = [m for m in mids if m != held]
        sc = {k: np.mean([t[k][m][s]["spell_f1"] for m in rest for s in seeds]) for k in SETTINGS}
        best = max(sc.values())
        sel[held] = next(k for k in SETTINGS if sc[k] == best)
    return sel


def v2_rows(mids, seeds, dead_motion):
    """The v2 reference rows (combination chosen per fold from toques_v2_m1.csv) and the fold choices."""
    rows = [r for r in ev.load_rows(ev.out_path(dead_motion)) if r["partido"] in mids]
    sel = ev.select(ev.table(rows, ev.MAIN), mids, seeds)
    out = []
    for r in rows:
        if r["combo"] == sel[r["partido"]] and r["semilla"] in seeds:
            out.append({"banco": r["banco"], "partido": r["partido"], "semilla": r["semilla"], "metodo": "v2",
                        "modo": r["combo"], "cap": int(r["combo"].rsplit("_g", 1)[1]), "spell_f1": r["spell_f1"],
                        "spell_f1_ut": r["spell_f1_ut"], "n_gt": r["n_gt"], "n_pred": r["n_pred"]})
    return out, sel


def selected_rows(rows, sel, seeds):
    out = []
    for r in rows:
        if r["metodo"] == "grid" and r["semilla"] in seeds and (r["modo"], r["cap"]) == sel[r["partido"]]:
            out.append(dict({k: v for k, v in r.items() if k not in COUNT_FIELDS}, metodo="equipo"))
    return out


# ---------------------------------------------------------------------------------------------------- summary

def edge_note(mode, cap):
    kind, L, D = eq.parse_mode(mode)
    notes = []
    fam = {"lag": L, "minwin": L}.get(kind)
    if kind in ("lag", "minwin") and fam in EDGES[kind]:
        notes.append(f"{kind} L={fam} en el borde")
    if D is not None and D in EDGES["tie"]:
        notes.append(f"tie D={D} en el borde")
    if kind == "T0":
        notes.append("T0: borde (sin cambio)")
    if cap in (min(CAPS), max(CAPS)):
        notes.append(f"cap {cap} en el borde de {{5,10}}")
    return "; ".join(notes)


def accuracy_table(rows, mids, seeds):
    sel_rows = [r for r in rows if r["metodo"] == "grid" and r["banco"] == ev.MAIN and r["cap"] == PART_CAP
                and r["partido"] in mids and r["semilla"] in seeds]
    print(f"\n===== exactitud de equipo entre los toques bien ubicados (banco {ev.MAIN}, {len(mids)} partidos x {len(seeds)} semillas, "
          f"particion fijada por T0 cap {PART_CAP}) =====")
    print("cambia = toque que cambia el dueño (a) | no cambia = mismo equipo que la etiqueta previa (b) | sin prev = sin etiqueta H/A previa (c)")
    print(f"{'modo':<13}{'all cambia':>11}{'all no camb':>12}{'all sin prev':>13}{'all total':>10}"
          f"{'defl cambia':>12}{'defl no camb':>13}{'defl total':>11}")
    acc = {}
    for mode in eq.MODES:
        rs = [r for r in sel_rows if r["modo"] == mode]
        S = lambda f: sum(r[f] for r in rs)  # noqa: E731
        a = {}
        for k in ("all", "defl"):
            for g in GROUPS:
                a[f"{k}_{g}"] = (S(f"{k}_{g}_ok"), S(f"{k}_{g}_tp"))
            a[f"{k}_t"] = (sum(a[f"{k}_{g}"][0] for g in GROUPS), sum(a[f"{k}_{g}"][1] for g in GROUPS))
        acc[mode] = a
        f = lambda key: f"{a[key][0] / max(a[key][1], 1):.3f}"  # noqa: E731
        print(f"{mode:<13}{f('all_a'):>11}{f('all_b'):>12}{f('all_c'):>13}{f('all_t'):>10}"
              f"{f('defl_a'):>12}{f('defl_b'):>13}{f('defl_t'):>11}")
    n = acc["T0"]
    print(f"toques bien ubicados (TP de tiempo): all cambia {n['all_a'][1]}, no cambia {n['all_b'][1]}, sin prev {n['all_c'][1]}; "
          f"defl cambia {n['defl_a'][1]}, no cambia {n['defl_b'][1]}")
    return acc


def mode_means(rows, mids, seeds):
    print(f"\n===== spellF1 medio por modo y tope, banco {ev.MAIN}, todos los partidos y semillas (diagnóstico, no es selección) =====")
    t = grid_table(rows, ev.MAIN)
    print(f"{'modo':<13}" + "".join(f"{'cap ' + str(c):>10}" for c in CAPS))
    for mode in eq.MODES:
        print(f"{mode:<13}" + "".join(f"{np.mean([t[(mode, c)][m][s]['spell_f1'] for m in mids for s in seeds]):>10.4f}" for c in CAPS))


def summarize(rows, mids, seeds):
    rows = [r for r in rows if r["partido"] in mids and r["semilla"] in seeds]
    acc = accuracy_table(rows, mids, seeds)
    mode_means(rows, mids, seeds)
    tabs = {b: grid_table(rows, b) for b in ev.BANCOS if any(r["banco"] == b for r in rows)}
    ref = {}
    for r in rows:
        if r["metodo"] in ("v2", "base"):
            ref.setdefault((r["banco"], r["metodo"]), {}).setdefault(r["partido"], {})[r["semilla"]] = r
    sel = select(tabs[ev.MAIN], mids, seeds)
    print("\n===== modo y tope elegidos por pliegue (mejor spellF1 medio en los otros partidos) =====")
    for m in mids:
        print(f"{m}  {sel[m][0]:<12} cap {sel[m][1]:<3} {edge_note(*sel[m])}")
    for key, title in (("spell_f1", "label (BallPossession)"), ("spell_f1_ut", "label_ut (last touch)")):
        print(f"\n===== GT = {title}: partido retenido, modo elegido contra toques v2 (y base F cap 10), {len(seeds)} semillas =====")
        for b, t in tabs.items():
            print(f"\n-- banco {b} --")
            print(f"{'partido':<9}{'base':>8}{'v2':>8}{'equipo':>8}{'eq-v2':>9}{'eq-base':>9}{'n_pred v2':>10}{'n_pred eq':>10}{'n_gt':>6}  modo")
            dm, db = [], []
            for m in mids:
                g = lambda x, f=key: np.mean([x[m][s][f] for s in seeds])  # noqa: E731
                e_ = t[sel[m]]
                v2_, b_ = ref[(b, "v2")], ref[(b, "base")]
                dm.append(g(e_) - g(v2_))
                db.append(g(e_) - g(b_))
                npd = lambda x: np.mean([x[m][s]["n_pred"] for s in seeds])  # noqa: E731
                print(f"{m:<9}{g(b_):>8.4f}{g(v2_):>8.4f}{g(e_):>8.4f}{dm[-1]:>+9.4f}{db[-1]:>+9.4f}{npd(v2_):>10.0f}{npd(e_):>10.0f}"
                      f"{b_[m][seeds[0]]['n_gt']:>6.0f}  {sel[m][0]}/{sel[m][1]}")
            dm = np.array(dm)
            sd = np.std([np.mean([t[sel[m]][m][s][key] - ref[(b, 'v2')][m][s][key] for m in mids]) for s in seeds])
            print(f"media dSpell (equipo - v2) {dm.mean():+.4f} | gana {int((dm > 0).sum())}/{len(dm)} | sd entre semillas {sd:.4f} | "
                  f"media equipo - base {np.mean(db):+.4f}")
            if key == "spell_f1":
                print("criterio (main: media >= +0,015 y >= 4/5; vetos pd0.60+g6 e indep+g6: media >= -0,005):", ev.criterio_txt(b, dm))
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partidos", nargs="*", default=None)
    ap.add_argument("--semillas", type=int, default=5)
    ap.add_argument("--procesos", type=int, default=min(10, os.cpu_count() or 1))
    ap.add_argument("--dead-motion", default=ev.DEAD_MOTION_DEFAULT, choices=["none"] + list(dd.DEAD_MOTION))
    ap.add_argument("--solo-resumen", action="store_true", help="re-print the summary from the CSV of a previous run")
    args = ap.parse_args()
    mids = args.partidos or list(dfl.DESARROLLO)
    assert not set(mids) & set(dfl.RESERVADOS) and len(mids) >= 3
    seeds = list(range(args.semillas))
    path = ev.out_path(args.dead_motion, stem="equipo")
    t0 = time.time()
    if args.solo_resumen:
        rows = load_rows(path)
    else:
        jobs = [(b, m, s, args.dead_motion) for b in ev.BANCOS for m in mids for s in seeds]
        rows = []
        with mp.get_context("fork").Pool(min(args.procesos, len(jobs))) as pool:
            for i, res in enumerate(pool.imap_unordered(work, jobs), 1):
                rows += res
                print(f"  {i}/{len(jobs)} done ({time.time() - t0:.0f}s)", file=sys.stderr)
        v2, _ = v2_rows(mids, set(seeds), args.dead_motion)
        sel = select(grid_table(rows, ev.MAIN), mids, seeds)
        rows = rows + v2 + selected_rows(rows, sel, set(seeds))
        write_rows(path, rows)
    summarize(rows, mids, seeds)
    print(f"\ntotal {time.time() - t0:.0f}s -> {path}")


if __name__ == "__main__":
    main()
