"""Oracle diagnosis on PFF (development matches): where is the spellF1 ceiling of B3 and which stage loses what.

    challenge/.venv/bin/python bancos/oraculo_pff.py [--procesos P] [--balones g6 g8] [--partidos MID ...]

DIAGNOSTIC ONLY. Unlike every method of the ladder, this script reads `label`, `is_touch` and the GT edges on purpose:
they are oracles that replace one stage of B3 at a time (nothing here is a method and nothing here is submitted).

  B3  reference (evaluar_pff.ladder, must equal src/out/pff_escalera_<ball>.csv)
  O1  real touches (`is_touch` frames) with the real team (`label` at that frame), through the same owner_labels / hole / D rules
  O2  real touches, team by the detector: nearest player, or `dir` (equipo_toque) when two players are within r, with the
      impulse of the ball around that frame measured exactly as deflection_candidates does
  O3  touches detected by B3, team replaced by `label` at their frame (a touch whose frame is GT D keeps its team)
  O4  B3 with the GT D frames forced to D and the predicted D outside GT D filled with the previous owner
  O5  O1 + O4
  O6  the GT itself with every possession edge moved U{-4..4} frames (bancos/ruido_gt_pff.shift_edges), scored against the
      original GT: practical ceiling given the label noise

Same split, same extra-time cut, same GT (make_gt) and same statistics (delta_stats, bootstrap over matches) as evaluar_pff.
Also: possession counts and an error taxonomy of the B3 spells (heuristic, at the headline IoU 0.5, see classify_spells).
Aggregates only; the informe goes to src/out/oraculo_pff_informe.txt.
"""
import argparse
import functools
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
import evaluar_toques as ev  # noqa: E402
import pff_split  # noqa: E402
import possession_metrics as pm  # noqa: E402
import reglas  # noqa: E402
import ruido_gt_pff as rg  # noqa: E402
import toques  # noqa: E402
from baseline import HOME, AWAY, NOBODY  # noqa: E402

V2, CAP = ep.V2, ep.CAP
VARIANTS = ("B3", "O1", "O2", "O3", "O4", "O5", "O6")
OUT_TXT = os.path.join(onside_paths.OUT_DIR, "oraculo_pff_informe.txt")
CATS_FN = ("fusion", "partida", "equipo_equivocado", "borde_D", "borde", "otro")
CATS_FP = ("inventada", "equipo_equivocado", "fusion", "partida", "borde_D", "borde")
MIN_OVERLAP = 5                                  # frames: an overlap shorter than this does not count as a link between spells
D_SHARE = 0.3


def team_of_label(lab):
    return np.where(lab == "H", HOME, AWAY).astype(np.int8)


def gt_touches(lab, present, runs, is_touch):
    """(frames, teams, n_all, n_in_run, n_dead): is_touch frames inside visible runs whose label is H or A."""
    f = np.flatnonzero(is_touch)
    in_run = np.zeros(len(lab), bool)
    for a, b in runs:
        in_run[a:b] = True
    keep = f[in_run[f]]
    alive = keep[np.isin(lab[keep], ("H", "A"))]
    return alive, team_of_label(lab[alive]), len(f), len(keep), len(keep) - len(alive)


def impulse_candidates(frames, near, runs, win, r):
    """Pseudo deflection candidates at `frames` (inside visible runs): same ball impulse as toques.deflection_candidates
    (moving average `win`, velocities over `win` frames); zeros where the frame is too close to the stretch border, so
    `dir` falls back to the nearest player. team = nearest player at the frame; frames with nobody are dropped."""
    ball = near["ball_m"]
    starts = np.array([a for a, _ in runs])
    cols = {k: [] for k in ("frame", "team", "peak", "start", "stop", "vp", "vn", "dv")}
    cache = {}
    for f in frames:
        k = int(np.searchsorted(starts, f, side="right") - 1)
        a, b = runs[k]
        if not (a <= f < b) or near["team"][f] == NOBODY:
            continue
        if k not in cache:
            L = b - a
            ps = toques._moving_average(ball[a:b], win)
            if L > 2 * win:
                vp = np.zeros((L, 2))
                vn = np.zeros((L, 2))
                vp[win:L - win] = (ps[win:L - win] - ps[:L - 2 * win]) * toques.FPS / win
                vn[win:L - win] = (ps[2 * win:] - ps[win:L - win]) * toques.FPS / win
            else:
                vp = vn = np.zeros((L, 2))
            cache[k] = (vp, vn)
        vp, vn = cache[k]
        i = f - a
        for key, v in zip(cols, (f, near["team"][f], f, a, b, vp[i], vn[i], float(np.linalg.norm(vn[i] - vp[i])))):
            cols[key].append(v)
    out = {k: np.asarray(v) for k, v in cols.items()}
    out["frame"], out["team"] = out["frame"].astype(int), out["team"].astype(np.int8)
    for k in ("peak", "start", "stop"):
        out[k] = out[k].astype(int)
    out["vp"], out["vn"], out["dv"] = out["vp"].reshape(-1, 2).astype(float), out["vn"].reshape(-1, 2).astype(float), out["dv"].astype(float)
    return out


def force_dead(pred, gt_lab):
    """O4: GT D frames forced to D; predicted D outside GT D filled with the previous owner (the last non-D predicted
    label before it, or the next one if none exists before it)."""
    p = np.asarray(pred, "<U1").copy()
    idx = np.flatnonzero(p != "D")
    if len(idx):
        pos = np.arange(len(p))
        prev = np.searchsorted(idx, pos, side="right") - 1
        nxt = np.minimum(np.searchsorted(idx, pos, side="left"), len(idx) - 1)
        src = np.where(prev >= 0, idx[np.maximum(prev, 0)], idx[nxt])
        fill = (p == "D") & (gt_lab != "D")
        p[fill] = p[src[fill]]
    p[gt_lab == "D"] = "D"
    return p


def predict_with(touches, ball, d, near, base, present, runs):
    return toques.predict(ball, d["players_slots"], d["period"], d["pitch"], touches=touches, max_gap_s=CAP,
                          near=near, base=base, **V2)


def spell_f1(gt, pred):
    r = pm.evaluate(gt, pm.label_codes(pred), bootstrap_n=0)
    return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]


def classify_spells(gt_codes, pred_codes, thr=pm.SPELL_IOU_HEADLINE):
    """Taxonomy of the spells that do not match one-to-one at IoU `thr` (headline 0.5), heuristic and exclusive, first rule wins.
    Unmatched GT spells (misses): fusion (its best same-team prediction also covers another GT spell of that team, overlap >= 5
    frames), partida (>= 2 same-team predictions overlap it), equipo_equivocado (the other team is predicted on the largest share
    of its frames), borde_D (>= 30 % of its frames predicted D), borde (a single same-team prediction with IoU < thr), otro.
    Unmatched predicted spells (false positives): inventada (< 30 % of its frames in a GT possession of its team and the largest
    share is GT D), equipo_equivocado (largest share is the other team), fusion / partida (same links seen from the prediction),
    borde_D (>= 30 % of its frames in GT D), borde. Returns ({cat: n} of misses, {cat: n} of false positives, n_gt, n_pred)."""
    runs = pm.valid_runs(gt_codes)
    filled = pm.fill_abstentions(pred_codes, runs)
    gs = pm.possession_spells(gt_codes, runs)
    ps = pm.possession_spells(filled, runs)
    ov = pm.spell_overlaps(gs, ps)
    flags = pm.matched_gt_flags(ov, len(gs), thr)
    matched_pred = set()
    used_g = set()
    for iou, i, j in ov:
        if iou < thr:
            break
        if i in used_g or j in matched_pred:
            continue
        used_g.add(i)
        matched_pred.add(j)
    g_to_p = {i: [] for i in range(len(gs))}
    p_to_g = {j: [] for j in range(len(ps))}
    for iou, i, j in ov:
        inter = min(gs[i][1], ps[j][1]) - max(gs[i][0], ps[j][0])
        if inter >= MIN_OVERLAP:
            g_to_p[i].append((inter, j))
            p_to_g[j].append((inter, i))
    fn = dict.fromkeys(CATS_FN, 0)
    for i, (a, b, k) in enumerate(gs):
        if flags[i]:
            continue
        seg = filled[a:b]
        n = b - a
        same = int((seg == k).sum())
        other = int((seg == (1 - k)).sum())
        dead = int((seg == pm.D).sum())
        best = max(g_to_p[i])[1] if g_to_p[i] else None
        if best is not None and len(p_to_g[best]) >= 2:
            fn["fusion"] += 1
        elif len(g_to_p[i]) >= 2:
            fn["partida"] += 1
        elif other > max(same, dead):
            fn["equipo_equivocado"] += 1
        elif dead >= D_SHARE * n:
            fn["borde_D"] += 1
        elif best is not None:
            fn["borde"] += 1
        else:
            fn["otro"] += 1
    fp = dict.fromkeys(CATS_FP, 0)
    for j, (a, b, k) in enumerate(ps):
        if j in matched_pred:
            continue
        seg = gt_codes[a:b]
        n = b - a
        same = int((seg == k).sum())
        other = int((seg == (1 - k)).sum())
        dead = int((seg == pm.D).sum())
        best = max(p_to_g[j])[1] if p_to_g[j] else None
        if same < D_SHARE * n and dead >= other:
            fp["inventada"] += 1
        elif same < D_SHARE * n:
            fp["equipo_equivocado"] += 1
        elif len(p_to_g[j]) >= 2:
            fp["fusion"] += 1
        elif best is not None and len(g_to_p[best]) >= 2:
            fp["partida"] += 1
        elif dead >= D_SHARE * n:
            fp["borde_D"] += 1
        else:
            fp["borde"] += 1
    return fn, fp, len(gs), len(ps)


def work(mid, balls=("g6",)):
    d = ep.clip_periods(pff_split.load_dev(mid))
    full = pff_split.load_dev(mid)                  # clip_periods does not carry is_touch: same mask here
    is_touch = np.asarray(full["is_touch"])[np.asarray(full["period"]) <= ep.LAST_PERIOD]
    lab = d["label"]
    rows = []
    for balon in balls:
        ball = dd.smooth_ball(d["ball_xy"], d["period"], balon)
        slots, per, pitch = d["players_slots"], d["period"], d["pitch"]
        present = ~np.isnan(ball[:, 0])
        near = toques.prepare(ball, slots, pitch)
        runs = toques.visible_runs(present, per)
        base = reglas.predict("B0", ball, slots, per, pitch)
        preds = ep.ladder(ball, slots, per, pitch)
        gt = ev.make_gt(d, "label")
        tf_gt, tt_gt, n_all, n_run, n_dead = gt_touches(lab, present, runs, is_touch)
        cand = toques.deflection_candidates(near, runs, r=V2["r"], dv=V2["dv"], win=V2["win"])
        ctrl = toques.control_touches(near, runs, r_ctrl=V2["r_ctrl"], k=V2["k"])
        td = toques.merge_touches(
            (cand["frame"], eq.deflection_teams(ep.MODE, cand, near, slots, pitch, r=V2["r"], win=V2["win"])), ctrl)
        out = {"B3": preds["B3"]}
        out["O1"] = predict_with((tf_gt, tt_gt), ball, d, near, base, present, runs)
        pc = impulse_candidates(tf_gt, near, runs, V2["win"], V2["r"])
        t2 = eq.deflection_teams(ep.MODE, pc, near, slots, pitch, r=V2["r"], win=V2["win"])
        out["O2"] = predict_with((pc["frame"], t2), ball, d, near, base, present, runs)
        labt = lab[td[0]]
        known = np.isin(labt, ("H", "A"))
        t3 = np.where(known, team_of_label(np.where(known, labt, "H")), td[1]).astype(np.int8)
        out["O3"] = predict_with((td[0], t3), ball, d, near, base, present, runs)
        out["O4"] = force_dead(out["B3"], lab)
        out["O5"] = force_dead(out["O1"], lab)
        shifted = [rg.shift_edges(lab, per, np.random.default_rng([rg.SEED, r, int(mid)]))[0] for r in range(1, rg.REPS + 1)]
        for v in VARIANTS:
            if v == "O6":
                res = np.array([spell_f1(gt, s) for s in shifted])
                sc, mf, npred, ngt = res[:, 0].mean(), res[:, 1].mean(), res[:, 2].mean(), res[0, 3]
            else:
                sc, mf, npred, ngt = spell_f1(gt, out[v])
            row = {"partido": mid, "balon": balon, "var": v, "spell_f1": sc, "macro_f1": mf, "n_pred": npred, "n_gt": ngt}
            if v == "B3":
                fn, fpc, _, _ = classify_spells(gt, pm.label_codes(out["B3"]))
                row.update({"fn": fn, "fp": fpc})
            if v == "O1":
                row.update({"n_touch_all": n_all, "n_touch_in_run": n_run, "n_touch_dead": n_dead, "n_touch_used": len(tf_gt),
                            "t2_agree": float(np.mean(t2 == tt_gt[np.isin(tf_gt, pc["frame"])])) if len(t2) else float("nan"),
                            "n_det": len(td[0]), "n_det_dead": int((~known).sum()),
                            "t3_changed": int((t3 != td[1]).sum())})
            rows.append(row)
    return rows


def fmt_table(rows, balon):
    by = {v: {r["partido"]: r for r in rows if r["balon"] == balon and r["var"] == v} for v in VARIANTS}
    mids = sorted(by["B3"], key=int)
    lines = [f"{'variante':<8}{'spellF1':>9}{'delta B3':>10}{'gana':>6}{'pierde':>7}{'IC95 inf':>10}{'IC95 sup':>10}"
             f"{'macroF1':>9}{'n_pred':>8}{'n_gt':>7}"]
    ref = np.array([by["B3"][m]["spell_f1"] for m in mids])
    for v in VARIANTS:
        a = np.array([by[v][m]["spell_f1"] for m in mids])
        mf = np.mean([by[v][m]["macro_f1"] for m in mids])
        npd = np.sum([by[v][m]["n_pred"] for m in mids])
        ngt = np.sum([by[v][m]["n_gt"] for m in mids])
        if v == "B3":
            lines.append(f"{v:<8}{a.mean():>9.4f}{'':>10}{'':>6}{'':>7}{'':>10}{'':>10}{mf:>9.4f}{npd:>8.0f}{ngt:>7.0f}")
        else:
            s = ep.delta_stats(a - ref)
            lines.append(f"{v:<8}{a.mean():>9.4f}{s['mean']:>+10.4f}{s['wins']:>6}{s['losses']:>7}{s['lo']:>+10.4f}{s['hi']:>+10.4f}"
                         f"{mf:>9.4f}{npd:>8.0f}{ngt:>7.0f}")
    return "\n".join(lines), by, mids


def report(rows, balls):
    out = []
    for balon in balls:
        txt, by, mids = fmt_table(rows, balon)
        main = " (PRINCIPAL)" if balon == "g6" else " (informativo)"
        out.append(f"== Oraculos PFF, balon {balon}{main}, GT label, n={len(mids)} partidos de desarrollo ==")
        out.append(txt)
        out.append("(O6: n_pred es la media de 5 repeticiones; delta contra B3 por partido, IC95 bootstrap sobre partidos, semilla fija)")
        o1 = [by["O1"][m] for m in mids]
        s = lambda k: int(np.sum([r[k] for r in o1]))      # noqa: E731
        out.append(f"\nToques GT ({balon}): is_touch {s('n_touch_all')}, dentro de tramos con balon {s('n_touch_in_run')}, de ellos con label D "
                   f"{s('n_touch_dead')} (descartados), usados {s('n_touch_used')}. Toques detectados por B3: {s('n_det')}; con label D en su frame "
                   f"{s('n_det_dead')} (O3 conserva su equipo); equipo cambiado por O3 en {s('t3_changed')}. "
                   f"Acuerdo de equipo del detector (O2) con el real: {np.nanmean([r['t2_agree'] for r in o1]):.3f} (media por partido).")
        npd = {v: np.sum([by[v][m]["n_pred"] for m in mids]) for v in VARIANTS}
        ngt = np.sum([by["B3"][m]["n_gt"] for m in mids])
        out.append(f"\nPosesiones B3 ({balon}): predichas {npd['B3']:.0f} contra GT {ngt:.0f} (razon {npd['B3'] / ngt:.2f}); "
                   f"por partido media {npd['B3'] / len(mids):.0f} contra {ngt / len(mids):.0f}.")
        fn = {c: sum(by["B3"][m]["fn"][c] for m in mids) for c in CATS_FN}
        fp = {c: sum(by["B3"][m]["fp"][c] for m in mids) for c in CATS_FP}
        tp_gt = ngt - sum(fn.values())
        out.append(f"\nErrores de B3 ({balon}) por tipo, IoU 0,5, emparejamiento uno a uno (heuristica exclusiva, ver classify_spells). "
                   f"Emparejadas {tp_gt:.0f} de {ngt:.0f} posesiones GT ({100 * tp_gt / ngt:.1f} %), {npd['B3'] - tp_gt:.0f} predichas sin pareja.")
        out.append(f"{'tipo':<20}{'GT perdidas':>13}{'% de perdidas':>15}   {'tipo':<20}{'predichas sobrantes':>20}{'% de sobrantes':>16}")
        tfn, tfp = sum(fn.values()), sum(fp.values())
        for a, b in zip(CATS_FN, CATS_FP + ("",) * (len(CATS_FN) - len(CATS_FP))):
            l = f"{a:<20}{fn[a]:>13}{100 * fn[a] / tfn:>14.1f}%   "
            l += f"{b:<20}{fp[b]:>20}{100 * fp[b] / tfp:>15.1f}%" if b else ""
            out.append(l)
        out.append(f"{'total':<20}{tfn:>13}{'':>15}   {'total':<20}{tfp:>20}")
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procesos", type=int, default=min(12, os.cpu_count() or 1))
    ap.add_argument("--balones", nargs="+", default=["g6", "g8"])
    ap.add_argument("--partidos", nargs="*", default=None)
    ap.add_argument("--salida", default=OUT_TXT)
    args = ap.parse_args()
    mids = args.partidos or list(pff_split.DESARROLLO)
    assert not set(mids) & set(pff_split.RESERVADOS), "held-out matches are not scored here"
    rows = []
    with mp.get_context("fork").Pool(min(args.procesos, len(mids))) as pool:
        for res in pool.imap_unordered(functools.partial(work, balls=tuple(args.balones)), mids):
            rows += res
    txt = report(rows, args.balones)
    print(txt)
    if args.salida:
        os.makedirs(os.path.dirname(args.salida), exist_ok=True)
        with open(args.salida, "w") as f:
            f.write(txt + "\n")


if __name__ == "__main__":
    main()
