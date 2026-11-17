"""Where does T1 lose spellF1 on the 40 PFF development matches (to decide whether a per-segment model, M2, is worth it).

    challenge/.venv/bin/python bancos/diagnostico_t1.py [--procesos P]

DIAGNOSTIC ONLY, reads the GT on purpose. T1 = the out-of-fold probabilities (src/out/modelo_cache/transfer/probs/T1__<ball>__<mid>.npy, balls g6h and g8h)
decoded with the final lambda (src/out/lambda_final.json); the reference is the base of the ladder (B3 + absorption tau 1.6, meta `b3_t16`). Nothing is retrained.

  1. sanity: mean spellF1 of T1 per ball against the lambda row of src/out/barrido_lambda_informe.txt (the report is not written if it differs)
  2. error taxonomy (oraculo_pff.classify_spells, IoU 0.5) of T1 and of the base, summed over the 40 matches
  3. edges of the matched possessions (one-to-one greedy IoU >= 0.5): start / end shift (pred - GT, frames) by GT context (preceded / followed by D or by the other team)
  4. GT direct H <-> A transitions that the model also has (same direction within +-25 / +-50 frames) by duration of the possession that starts there,
     and the model transitions with no GT transition of the same direction within +-50 frames (spurious) by duration of the possession they open
  5. possession duration distribution and possessions per minute, GT against T1 and base
  6. cheap ceilings on T1: perfect D (GT D forced, T1 D on a GT live frame filled with the likelier of H / A) and perfect team (T1 D kept, T1 live frame
     takes the GT team when the GT is live)
  7. spellF1 per match, the 5 worst
  8. unmatched GT possessions by classify_spells category (replicated per possession in classify_gt_spells) by GT duration: < 1, 1-2, 2-3, 3-5, 5-10, >= 10 s
  9. restarts: GT possessions opened by a restart (first frame inside a restart_window, or first frame after a D stretch that contains the restart
     event; bancos/reinicios.restart_started) by restart kind, and the start shift of the matched ones against `label` and against `label_rw`
 10. recall of the H <-> A changes outside / inside a restart window (+-25 frames of margin), next to the total recall
 11. secondary GT `label_rw` (`label` with D on every frame of a restart window): spellF1 of T1 and of the base against `label` and against `label_rw`
(8 to 11 are the 2026-10-29 extension; they come after the "Lectura" block of the report.)

Aggregates only; the informe goes to src/out/diagnostico_t1_informe.txt.
"""
import argparse
import functools
import multiprocessing as mp
import os
import re
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
import features_reto as fr  # noqa: E402
import modelo_transfer as mt  # noqa: E402
import oraculo_pff as op  # noqa: E402
import pff_split  # noqa: E402
import possession_metrics as pm  # noqa: E402
import predecir_reto_m1 as pr  # noqa: E402
import reinicios as rt  # noqa: E402
import ronda_pff as rp  # noqa: E402

BALLS = mt.H_BALLS
SERIES = ("T1", "base")
FPS = pm.FPS
IOU = pm.SPELL_IOU_HEADLINE
EDGE_BIG = 12                                              # frames: |shift| above this counts as a displaced edge
TOLS = (25, 50)                                            # frames: tolerance of the transition recall (the spurious ones use the last)
# duration bins in frames: the recall bins (< 2 s, 2-5 s, 5-10 s, >= 10 s) and the distribution bins (the last one split at 30 s)
BINS_R = ((0, 2 * FPS, "< 2 s"), (2 * FPS, 5 * FPS, "2-5 s"), (5 * FPS, 10 * FPS, "5-10 s"), (10 * FPS, np.inf, ">= 10 s"))
BINS_D = BINS_R[:3] + ((10 * FPS, 30 * FPS, "10-30 s"), (30 * FPS, np.inf, ">= 30 s"))
BINS_S = ((0, 1 * FPS, "< 1 s"), (1 * FPS, 2 * FPS, "1-2 s"), (2 * FPS, 3 * FPS, "2-3 s"), (3 * FPS, 5 * FPS, "3-5 s"), (5 * FPS, 10 * FPS, "5-10 s"),
          (10 * FPS, np.inf, ">= 10 s"))                  # section 8: finer than BINS_R below 5 s
CTX = ("D", "otro")                                        # context of an edge in the GT: D, or the other team (None: the run border, not counted)
REPORT = os.path.join(onside_paths.OUT_DIR, "diagnostico_t1_informe.txt")
BARRIDO = os.path.join(onside_paths.OUT_DIR, "barrido_lambda_informe.txt")
CORDURA_TOL = 5e-5                                         # the barrido informe prints 4 decimals


# ------------------------------------------------------------------------------------------------ pure functions (tested)

def spell_pairs(gt, pred, thr=IOU):
    """(gt_spells, pred_spells, pairs): the one-to-one greedy match of oraculo_pff.classify_spells (IoU descending, pred abstentions filled), pairs = [(i, j)]."""
    runs = pm.valid_runs(gt)
    gs = pm.possession_spells(gt, runs)
    ps = pm.possession_spells(pm.fill_abstentions(pred, runs), runs)
    pairs, used_g, used_p = [], set(), set()
    for iou, i, j in pm.spell_overlaps(gs, ps):
        if iou < thr:
            break
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        pairs.append((i, j))
    return gs, ps, pairs


def edge_shifts(gt, pred, thr=IOU):
    """[(d_start, d_end, ctx_start, ctx_end)] for every matched possession: d = predicted edge - GT edge (frames; > 0 = the prediction opens / closes later).
    ctx_start: what precedes the GT possession ("D" restart, "otro" the other team = steal, None = it opens a valid run); ctx_end: what follows it."""
    runs = pm.valid_runs(gt)
    starts, stops = {a for a, _ in runs}, {b for _, b in runs}
    gs, ps, pairs = spell_pairs(gt, pred, thr)
    out = []
    for i, j in pairs:
        a, b, _ = gs[i]
        ctx_s = None if a in starts else ("D" if gt[a - 1] == pm.D else "otro")
        ctx_e = None if b in stops else ("D" if gt[b] == pm.D else "otro")
        out.append((ps[j][0] - a, ps[j][1] - b, ctx_s, ctx_e))
    return out


def direct_transitions(codes, runs):
    """[(frame, from, to, duration)] of the direct H <-> A changes inside valid runs; duration = frames of the possession that starts at `frame`."""
    segs = pm.segments(codes, runs)
    return [(b[0], a[2], b[2], b[1] - b[0]) for a, b in zip(segs, segs[1:])
            if a[1] == b[0] and a[2] in pm.POSSESSION_LABELS and b[2] in pm.POSSESSION_LABELS and a[2] != b[2]]


def _near(frames, f, tol):
    """True where `f` (sorted array `frames`) has an element within +-tol."""
    if not len(frames):
        return False
    k = np.searchsorted(frames, f)
    return any(0 <= m < len(frames) and abs(int(frames[m]) - f) <= tol for m in (k - 1, k))


def transition_recall(gt, pred, tols=TOLS):
    """(durations (n,), hit (n, len(tols))): for every GT direct H <-> A transition, the duration of the possession that starts there and whether the prediction
    has a transition of the same direction within +-tol frames (existence, not one-to-one)."""
    runs = pm.valid_runs(gt)
    pt = direct_transitions(pm.fill_abstentions(pred, runs), runs)
    by_dir = {d: np.array(sorted(t[0] for t in pt if (t[1], t[2]) == d)) for d in ((pm.H, pm.A), (pm.A, pm.H))}
    gt_t = direct_transitions(gt, runs)
    dur = np.array([t[3] for t in gt_t], int)
    hit = np.array([[_near(by_dir[(t[1], t[2])], t[0], tol) for tol in tols] for t in gt_t], bool).reshape(len(gt_t), len(tols))
    return dur, hit


def spurious_transitions(gt, pred, tol=TOLS[-1]):
    """Durations (frames) of the possessions opened by a predicted direct H <-> A transition with no GT direct transition of the same direction within +-tol."""
    runs = pm.valid_runs(gt)
    gt_t = direct_transitions(gt, runs)
    by_dir = {d: np.array(sorted(t[0] for t in gt_t if (t[1], t[2]) == d)) for d in ((pm.H, pm.A), (pm.A, pm.H))}
    pt = direct_transitions(pm.fill_abstentions(pred, runs), runs)
    return np.array([t[3] for t in pt if not _near(by_dir[(t[1], t[2])], t[0], tol)], int)


def ceiling_dead(gt, pred, P, per_stretch=False):
    """T1 with perfect D: GT D -> D; where the GT is live and the prediction is D, the likelier of H / A of P (N, 3: H, A, D) at that frame.
    per_stretch=True gives every maximal stretch of filled frames one team (the one with the larger summed probability) instead of the per-frame argmax,
    which flickers inside long D stretches and fragments the possession."""
    gt, out = np.asarray(gt), np.asarray(pred).copy()
    out[(gt == pm.D)] = pm.D
    fill = (gt != pm.D) & (gt != pm.X) & (out == pm.D)
    Q = np.asarray(P)[:, :2]
    if not per_stretch:
        out[fill] = np.argmax(Q, axis=1)[fill]
        return out
    edges = np.diff(np.concatenate(([0], fill.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        out[a:b] = int(Q[a:b].sum(0).argmax())
    return out


def ceiling_team(gt, pred):
    """T1 with perfect team: the D of T1 stays; a frame T1 calls live takes the GT team if the GT is live there (GT D or X: T1 stays)."""
    gt, out = np.asarray(gt), np.asarray(pred).copy()
    live = (out != pm.D) & np.isin(gt, pm.POSSESSION_LABELS)
    out[live] = gt[live]
    return out


def classify_gt_spells(gt, pred, thr=IOU):
    """[(start, stop, team, cat)] one per GT possession; cat = None if matched, else the miss category of oraculo_pff.classify_spells (same rules, replicated
    here to keep the category of each possession: classify_spells only returns the counts). The counts per category equal its `fn` (tested, and asserted in `work`)."""
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    gs = pm.possession_spells(gt, runs)
    ps = pm.possession_spells(filled, runs)
    ov = pm.spell_overlaps(gs, ps)
    flags = pm.matched_gt_flags(ov, len(gs), thr)
    g_to_p = {i: [] for i in range(len(gs))}
    p_to_g = {j: [] for j in range(len(ps))}
    for iou, i, j in ov:
        inter = min(gs[i][1], ps[j][1]) - max(gs[i][0], ps[j][0])
        if inter >= op.MIN_OVERLAP:
            g_to_p[i].append((inter, j))
            p_to_g[j].append((inter, i))
    out = []
    for i, (a, b, k) in enumerate(gs):
        if flags[i]:
            out.append((a, b, k, None))
            continue
        seg = filled[a:b]
        n = b - a
        same, other, dead = int((seg == k).sum()), int((seg == (1 - k)).sum()), int((seg == pm.D).sum())
        best = max(g_to_p[i])[1] if g_to_p[i] else None
        if best is not None and len(p_to_g[best]) >= 2:
            cat = "fusion"
        elif len(g_to_p[i]) >= 2:
            cat = "partida"
        elif other > max(same, dead):
            cat = "equipo_equivocado"
        elif dead >= op.D_SHARE * n:
            cat = "borde_D"
        elif best is not None:
            cat = "borde"
        else:
            cat = "otro"
        out.append((a, b, k, cat))
    return out


def restart_start_records(gt, pred, rw, ev_frame, ev_kind, thr=IOU):
    """[(kind, how, shift, lag)] one per GT possession of `gt` (kind 0 = not opened by a restart, see reinicios.restart_started); shift = predicted start - GT start
    (frames) of the matched possession (one-to-one greedy IoU >= thr, `spell_pairs`), NaN if the possession is not matched; lag = GT start - frame of the
    latest restart event at or before it (frames; NaN if there is none)."""
    started = rt.restart_started(gt, rw, ev_frame, ev_kind)
    gs, ps, pairs = spell_pairs(gt, pred, thr)
    assert [(a, b, k) for a, b, k, _, _ in started] == list(gs)
    shift = {i: ps[j][0] - gs[i][0] for i, j in pairs}
    ev = np.sort(np.asarray(ev_frame, int))
    lag = lambda a: float(a - ev[np.searchsorted(ev, a, side="right") - 1]) if len(ev) and ev[0] <= a else np.nan      # noqa: E731
    return [(kind, how, float(shift.get(i, np.nan)), lag(a)) for i, (a, _, _, kind, how) in enumerate(started)]


def recall_by_window(gt, pred, rw, tols=TOLS, margin=rt.MARGIN):
    """(hit (n, len(tols)) bool, near (n,) bool): the transition recall of `transition_recall` per GT direct H <-> A change, and whether the change lies
    within +-margin frames of a restart window (near = True)."""
    runs = pm.valid_runs(gt)
    frames = [t[0] for t in direct_transitions(gt, runs)]
    _, hit = transition_recall(gt, pred, tols)
    return hit, rt.near_window(rw, frames, margin)


def duration_bins(durations, bins):
    """Counts of `durations` (frames) in each (lo, hi, label) bin, lo <= d < hi."""
    d = np.asarray(durations, float)
    return np.array([int(((d >= lo) & (d < hi)).sum()) for lo, hi, _ in bins])


def shift_stats(values):
    """{n, median, mean, p10, p90, big} of the shifts (frames); big = % with |shift| > EDGE_BIG. NaN if empty."""
    v = np.asarray(values, float)
    if not len(v):
        return {"n": 0, **dict.fromkeys(("median", "mean", "p10", "p90", "big"), float("nan"))}
    return {"n": len(v), "median": float(np.median(v)), "mean": float(v.mean()), "p10": float(np.percentile(v, 10)),
            "p90": float(np.percentile(v, 90)), "big": 100.0 * float((np.abs(v) > EDGE_BIG).mean())}


# ------------------------------------------------------------------------------------------------ per match

def spell_f1(gt, pred):
    return pm.spell_scores(gt, pred, pm.valid_runs(gt))["mean_f1"]


def work(mid, balon, lam):
    """Every statistic of one match and ball (small arrays and counters, nothing per frame)."""
    z = mt.meta_of(mid, balon)
    gt = np.asarray(z["gt_label"])
    runs = pm.valid_runs(gt)
    P = np.load(os.path.join(mt.PROBS, f"T1__{balon}__{mid}.npy")).astype(float)
    t1 = fr.decode(P, z["per"], lam)
    codes = {"T1": t1, "base": np.asarray(z["b3_t16"])}
    rs = rt.load_restart(mid)
    gt_rw = rt.dead_where_restart(gt, rs["rw"])
    live_rw = (rs["rw"] > 0) & np.isin(gt, pm.POSSESSION_LABELS)
    res = {"mid": mid, "balon": balon, "frames": int(sum(b - a for a, b in runs)), "gt_dur": np.array([b - a for a, b, _ in pm.possession_spells(gt, runs)], int),
           "rw_frames": int(((rs["rw"] > 0) & (gt != pm.X)).sum()), "rw_live": int(live_rw.sum()), "n_rw": int(len(rs["ev_frame"]))}
    for s, c in codes.items():
        fn, fp, n_gt, n_pred = op.classify_spells(gt, c)
        dur, hit = transition_recall(gt, c)
        res[s] = {"f1": spell_f1(gt, c), "fn": fn, "fp": fp, "n_gt": n_gt, "n_pred": n_pred, "edges": edge_shifts(gt, c), "tr_dur": dur, "tr_hit": hit,
                  "spurious": spurious_transitions(gt, c), "n_trans": len(direct_transitions(pm.fill_abstentions(c, runs), runs)), "dur": np.array([b - a for a, b, _ in pm.possession_spells(c, runs)], int)}
        # extension 8 to 11
        cats = classify_gt_spells(gt, c)
        got = {k: sum(x[3] == k for x in cats) for k in op.CATS_FN}
        assert got == fn, f"{mid} {balon} {s}: classify_gt_spells {got} != classify_spells {fn}"
        res[s]["gt_cat"] = [(b - a, x) for a, b, _, x in cats]
        res[s]["rs"] = restart_start_records(gt, c, rs["rw"], rs["ev_frame"], rs["ev_kind"])
        res[s]["rs_rw"] = restart_start_records(gt_rw, c, rs["rw"], rs["ev_frame"], rs["ev_kind"])
        res[s]["tr_win"] = recall_by_window(gt, c, rs["rw"])
        res[s]["f1_rw"] = spell_f1(gt_rw, c)
        res[s]["n_gt_rw"] = len(pm.possession_spells(gt_rw, runs))
    ca, cs = ceiling_dead(gt, t1, P), ceiling_dead(gt, t1, P, per_stretch=True)
    res["ceil_d"], res["ceil_ds"] = spell_f1(gt, ca), spell_f1(gt, cs)
    res["np_ceil_d"], res["np_ceil_ds"] = len(pm.possession_spells(ca, runs)), len(pm.possession_spells(cs, runs))
    ct = ceiling_team(gt, t1)
    res["ceil_team"], res["np_ceil_team"] = spell_f1(gt, ct), len(pm.possession_spells(ct, runs))
    return res


# ------------------------------------------------------------------------------------------------ sanity against the barrido informe

def barrido_row(lam, path=BARRIDO):
    """(g6h, g8h) means printed in the barrido informe at lambda `lam`, or None if the file or the point is not there."""
    if not os.path.exists(path):
        return None
    lines = open(path).read().splitlines()
    head = next((i for i, l in enumerate(lines) if "ha / d" in l), None)
    if head is None:
        return None
    lam_d = [int(x) for x in lines[head].split("ha / d")[1].split()]
    if lam[1] not in lam_d:
        return None
    for l in lines[head + 1:head + 4]:
        if l.split()[0] == str(lam[0]):
            cells = re.findall(r"\(([\d.]+)/([\d.]+)\)", l)
            return tuple(float(x) for x in cells[lam_d.index(lam[1])])
    return None


# ------------------------------------------------------------------------------------------------ report

def pct(a, b):
    return 100.0 * a / b if b else float("nan")


def fmt_shift(label, st):
    return (f"{label:<38}{st['n']:>7}{st['median']:>9.1f}{st['mean']:>9.1f}{st['p10']:>8.1f}{st['p90']:>8.1f}{st['big']:>10.1f}%")


def report(rows, lam, ref):
    out = []
    P = out.append
    mids = sorted({r["mid"] for r in rows}, key=int)
    P(f"DIAGNOSTICO DE T1: donde pierde spellF1, 40 partidos de desarrollo, GT label, T1 con lambda final {tuple(lam)}, base = B3 + absorcion tau 1,6 (b3_t16).")
    P("Emparejamiento: uno a uno, IoU >= 0,5 (el greedy de oraculo_pff.classify_spells). Desplazamientos en frames (25 fps), pred - GT: positivo = la prediccion abre / cierra mas tarde.")
    for b in BALLS:
        R = {r["mid"]: r for r in rows if r["balon"] == b}
        rs = [R[m] for m in mids]
        frames = sum(r["frames"] for r in rs)
        minutes = frames / FPS / 60
        P(f"\n{'=' * 24} balon {b} {'=' * 24}")
        # 1
        P(f"\n1. CORDURA: spellF1 medio de T1 {np.mean([r['T1']['f1'] for r in rs]):.4f}" + (f" (barrido_lambda_informe, fila {tuple(lam)}: {ref[BALLS.index(b)]:.4f})" if ref else "")
          + f"; base {np.mean([r['base']['f1'] for r in rs]):.4f}.")
        # 2
        P("\n2. TAXONOMIA de errores (classify_spells, IoU 0,5), suma de los 40 partidos:")
        n_gt = sum(r["T1"]["n_gt"] for r in rs)
        for s in SERIES:
            n_pred = sum(r[s]["n_pred"] for r in rs)
            fn = {c: sum(r[s]["fn"][c] for r in rs) for c in op.CATS_FN}
            fp = {c: sum(r[s]["fp"][c] for r in rs) for c in op.CATS_FP}
            tfn, tfp = sum(fn.values()), sum(fp.values())
            P(f"  {s}: posesiones GT {n_gt}, predichas {n_pred}; emparejadas {n_gt - tfn} ({pct(n_gt - tfn, n_gt):.1f} % de las GT); GT perdidas {tfn}, predichas sobrantes {tfp}.")
            P(f"  {'tipo':<20}{'GT perdidas':>13}{'% de perdidas':>15}   {'tipo':<20}{'predichas sobrantes':>20}{'% de sobrantes':>16}")
            for a, c in zip(op.CATS_FN, op.CATS_FP + ("",) * (len(op.CATS_FN) - len(op.CATS_FP))):
                l = f"  {a:<20}{fn[a]:>13}{pct(fn[a], tfn):>14.1f}%   "
                l += f"{c:<20}{fp[c]:>20}{pct(fp[c], tfp):>15.1f}%" if c else ""
                P(l)
        # 3
        P("\n3. BORDES de las posesiones emparejadas (desplazamiento en frames, pred - GT; % = |desplazamiento| > 12):")
        P(f"{'':<38}{'n':>7}{'mediana':>9}{'media':>9}{'p10':>8}{'p90':>8}{'% >12':>11}")
        for s in SERIES:
            e = [x for r in rs for x in r[s]["edges"]]
            ds, de = [x[0] for x in e], [x[1] for x in e]
            P(fmt_shift(f"{s} inicio, todas", shift_stats(ds)))
            P(fmt_shift(f"{s} inicio, tras D (reinicio)", shift_stats([x[0] for x in e if x[2] == "D"])))
            P(fmt_shift(f"{s} inicio, tras otro equipo (robo)", shift_stats([x[0] for x in e if x[2] == "otro"])))
            P(fmt_shift(f"{s} fin, todas", shift_stats(de)))
            P(fmt_shift(f"{s} fin, antes de D", shift_stats([x[1] for x in e if x[3] == "D"])))
            P(fmt_shift(f"{s} fin, antes de otro equipo", shift_stats([x[1] for x in e if x[3] == "otro"])))
        # 4
        P("\n4. CAMBIOS de posesion H <-> A (transicion directa, sin D en medio):")
        P(f"  Recall: transiciones GT con una del modelo de la misma direccion a +-{TOLS[0]} / +-{TOLS[1]} frames, por duracion de la posesion GT que empieza ahi (existencia, no uno a uno).")
        P(f"  {'':<8}" + "".join(f"{lab:>26}" for _, _, lab in BINS_R) + f"{'todas':>26}")
        for s in SERIES:
            dur = np.concatenate([r[s]["tr_dur"] for r in rs])
            hit = np.concatenate([r[s]["tr_hit"] for r in rs]) if len(dur) else np.zeros((0, len(TOLS)), bool)
            cells = []
            for lo, hi, _ in BINS_R + ((0, np.inf, ""),):
                m = (dur >= lo) & (dur < hi)
                cells.append(f"{m.sum():>5} {' '.join(f'{pct(hit[m, k].sum(), m.sum()):5.1f}%' for k in range(len(TOLS)))}")
            P(f"  {s:<8}" + "".join(f"{c:>26}" for c in cells))
        P(f"  (cada celda: n transiciones GT, recall a +-{TOLS[0]}, recall a +-{TOLS[1]})")
        P(f"  Espurias: transiciones del modelo sin transicion GT de la misma direccion a +-{TOLS[-1]} frames, por duracion de la posesion que abren.")
        P(f"  {'':<8}" + "".join(f"{lab:>14}" for _, _, lab in BINS_R) + f"{'espurias':>14}{'transiciones':>14}{'% espurias':>12}")
        for s in SERIES:
            sp = np.concatenate([r[s]["spurious"] for r in rs])
            n_tr = sum(r[s]["n_trans"] for r in rs)
            cnt = duration_bins(sp, BINS_R)
            P(f"  {s:<8}" + "".join(f"{c:>14}" for c in cnt) + f"{len(sp):>14}{n_tr:>14}{pct(len(sp), n_tr):>11.1f}%")
        # 5
        P("\n5. DURACION de las posesiones (conteo y % de las posesiones; % de los frames con posesion) y posesiones por minuto:")
        gt_dur = np.concatenate([r["gt_dur"] for r in rs])
        series_dur = {"GT": gt_dur, **{s: np.concatenate([r[s]["dur"] for r in rs]) for s in SERIES}}
        P(f"  {'':<6}" + "".join(f"{lab:>16}" for _, _, lab in BINS_D) + f"{'total':>9}{'pos/min':>9}{'mediana s':>11}")
        for s, d in series_dur.items():
            cnt = duration_bins(d, BINS_D)
            fr_share = [100 * float(d[(d >= lo) & (d < hi)].sum()) / d.sum() for lo, hi, _ in BINS_D]
            P(f"  {s:<6}" + "".join(f"{f'{c} {100 * c / len(d):.0f}%/{f:.0f}%':>16}" for c, f in zip(cnt, fr_share))
              + f"{len(d):>9}{len(d) / minutes:>9.2f}{np.median(d) / FPS:>11.1f}")
        P("  (cada celda: posesiones, % de las posesiones / % de los frames con posesion)")
        # 6
        P("\n6. TECHOS baratos sobre T1 (spellF1 medio por balon; delta contra T1 por partido):")
        keys = ("T1", "ceil_d", "ceil_ds", "ceil_team", "base")
        a = {k: np.array([r[k]["f1"] if k in SERIES else r[k] for r in rs]) for k in keys}
        npos = {"T1": sum(r["T1"]["n_pred"] for r in rs), "base": sum(r["base"]["n_pred"] for r in rs), **{k: sum(r["np_" + k] for r in rs) for k in keys[1:4]}}
        P(f"  {'variante':<46}{'spellF1':>9}{'delta T1':>10}{'gana':>6}{'pierde':>7}{'pos pred':>10}")
        for k, lab in (("T1", "T1"), ("ceil_d", "(a) T1 con D perfecto (argmax por frame)"), ("ceil_ds", "(a2) igual, un equipo por tramo rellenado"),
                       ("ceil_team", "(b) T1 con equipo perfecto"), ("base", "base b3_t16")):
            if k == "T1":
                P(f"  {lab:<46}{a[k].mean():>9.4f}{'':>10}{'':>6}{'':>7}{npos[k]:>10}")
            else:
                s = ep.delta_stats(a[k] - a["T1"])
                P(f"  {lab:<46}{a[k].mean():>9.4f}{s['mean']:>+10.4f}{s['wins']:>6}{s['losses']:>7}{npos[k]:>10}")
        # 7
        P("\n7. POR PARTIDO: spellF1 de T1 (min / mediana / max) y los 5 peores:")
        f1s = np.array([R[m]["T1"]["f1"] for m in mids])
        P(f"  min {f1s.min():.4f}, mediana {np.median(f1s):.4f}, max {f1s.max():.4f}")
        P(f"  {'partido':<10}{'T1':>8}{'base':>8}{'pos pred T1':>13}{'pos GT':>8}{'pos pred base':>15}")
        for m in sorted(mids, key=lambda m: R[m]["T1"]["f1"])[:5]:
            r = R[m]
            P(f"  {m:<10}{r['T1']['f1']:>8.4f}{r['base']['f1']:>8.4f}{r['T1']['n_pred']:>13}{r['T1']['n_gt']:>8}{r['base']['n_pred']:>15}")
    return "\n".join(out)


def lectura(rows, lam):
    """Plain statements read off the figures of the first ball (g6h); no recommendations."""
    b = BALLS[0]
    rs = [r for r in rows if r["balon"] == b]
    minutes = sum(r["frames"] for r in rs) / FPS / 60
    n_gt = sum(r["T1"]["n_gt"] for r in rs)
    n_t1 = sum(r["T1"]["n_pred"] for r in rs)
    gt_ppm, t1_ppm = n_gt / minutes, n_t1 / minutes
    fn = {c: sum(r["T1"]["fn"][c] for r in rs) for c in op.CATS_FN}
    fp = {c: sum(r["T1"]["fp"][c] for r in rs) for c in op.CATS_FP}
    tfn, tfp = sum(fn.values()), sum(fp.values())
    top_fn = sorted(fn, key=fn.get, reverse=True)[:2]
    top_fp = sorted(fp, key=fp.get, reverse=True)[:2]
    e = [x for r in rs for x in r["T1"]["edges"]]
    st_d, st_r = shift_stats([x[0] for x in e if x[2] == "D"]), shift_stats([x[0] for x in e if x[2] == "otro"])
    en_d = shift_stats([x[1] for x in e if x[3] == "D"])
    dur = np.concatenate([r["T1"]["tr_dur"] for r in rs])
    hit = np.concatenate([r["T1"]["tr_hit"] for r in rs])
    rec = [pct(hit[:, k].sum(), len(dur)) for k in range(len(TOLS))]
    sp = sum(len(r["T1"]["spurious"]) for r in rs)
    n_tr = sum(r["T1"]["n_trans"] for r in rs)
    f1 = {k: np.mean([r[k]["f1"] if k in SERIES else r[k] for r in rs]) for k in ("T1", "ceil_d", "ceil_ds", "ceil_team")}
    gt_dur, t1_dur = np.concatenate([r["gt_dur"] for r in rs]), np.concatenate([r["T1"]["dur"] for r in rs])
    gt_short, t1_short = (gt_dur < BINS_D[0][1]).sum(), (t1_dur < BINS_D[0][1]).sum()
    gt_long, t1_long = (gt_dur >= BINS_D[-1][0]).sum(), (t1_dur >= BINS_D[-1][0]).sum()
    word = lambda v: "mas tarde" if v > 0 else "mas temprano"      # noqa: E731
    out = [f"- Posesiones por minuto ({b}): GT {gt_ppm:.2f}, T1 {t1_ppm:.2f}; T1 empareja {100 * (n_gt - tfn) / n_gt:.1f} % de las posesiones GT a IoU 0,5.",
           f"- GT perdidas de T1: {top_fn[0]} {pct(fn[top_fn[0]], tfn):.1f} %, {top_fn[1]} {pct(fn[top_fn[1]], tfn):.1f} %; predichas sobrantes: {top_fp[0]} {pct(fp[top_fp[0]], tfp):.1f} %, {top_fp[1]} {pct(fp[top_fp[1]], tfp):.1f} %.",
           f"- Inicio de las emparejadas: mediana {st_d['median']:+.1f} frames tras D ({word(st_d['median'])}, {st_d['big']:.0f} % con |d| > 12) y {st_r['median']:+.1f} tras otro equipo ({st_r['big']:.0f} % con |d| > 12); "
           f"fin antes de D: mediana {en_d['median']:+.1f}.",
           f"- Cambios H <-> A de la GT con transicion de T1 de la misma direccion: {rec[0]:.1f} % a +-{TOLS[0]} frames, {rec[1]:.1f} % a +-{TOLS[1]} (de {len(dur)}).",
           f"- Transiciones H <-> A de T1 sin transicion GT de la misma direccion a +-{TOLS[-1]} frames: {sp} de {n_tr} ({pct(sp, n_tr):.1f} %).",
           f"- Techos sobre T1 ({b}): {f1['T1']:.4f} -> D perfecto {f1['ceil_d']:.4f} ({f1['ceil_d'] - f1['T1']:+.4f}; con un equipo por tramo rellenado {f1['ceil_ds']:.4f}, {f1['ceil_ds'] - f1['T1']:+.4f}), equipo perfecto {f1['ceil_team']:.4f} ({f1['ceil_team'] - f1['T1']:+.4f}).",
           f"- Posesiones < 2 s: GT {pct(gt_short, n_gt):.0f} % de las posesiones, T1 {pct(t1_short, n_t1):.0f} %; >= 30 s: GT {pct(gt_long, n_gt):.0f} %, T1 {pct(t1_long, n_t1):.0f} %. "
           "Las lecturas son de g6h; g8h en las tablas."]
    return "\n".join(out)


def cell(n, tot):
    return f"{n:>5} {pct(n, tot):5.1f}%"


def report_extra(rows, lam):
    """Sections 8 to 11 (2026-10-29), per ball."""
    out = []
    P = out.append
    mids = sorted({r["mid"] for r in rows}, key=int)
    P(f"AMPLIACION (secciones 8 a 11): T1 con lambda final {tuple(lam)} y la base b3_t16, 40 partidos de desarrollo; reinicios = restart_window del loader (PFF_VERSION 3.1-restart-window).")
    P("Reinicio que abre una posesion GT: (a) 'dentro': su primer frame cae dentro de una restart_window (ventana = [frame del evento, frame del siguiente toque)), o (b) 'despues de D': su primer frame sigue")
    P("a un tramo D (maximal, dentro de la racha valida) que contiene el frame del evento de un reinicio (el ultimo del tramo si hay varios). El tipo es el de esa ventana o evento.")
    for b in BALLS:
        R = {r["mid"]: r for r in rows if r["balon"] == b}
        rs = [R[m] for m in mids]
        n_valid = sum(r["frames"] for r in rs)
        P(f"\n{'=' * 24} balon {b} {'=' * 24}")
        # 8
        P("\n8. POSESIONES GT NO EMPAREJADAS por categoria de classify_spells y por duracion de la posesion GT (suma de los 40 partidos).")
        P("   Cada celda: n (% de las posesiones GT de ese tramo). 'emparejadas' = IoU >= 0,5 uno a uno. Las categorias replican classify_spells (mismos conteos, comprobado partido a partido).")
        for s_ in SERIES:
            gc = [x for r in rs for x in r[s_]["gt_cat"]]
            dur = np.array([d for d, _ in gc], int)
            cat = np.array([c if c else "emparejada" for _, c in gc])
            cols = ("emparejada",) + op.CATS_FN
            P(f"  {s_}: {len(gc)} posesiones GT, {int((cat != 'emparejada').sum())} no emparejadas.")
            P(f"  {'duracion GT':<12}{'total':>7}" + "".join(f"{c:>19}" for c in cols))
            for lo, hi, lab in BINS_S + ((0, np.inf, "todas"),):
                m = (dur >= lo) & (dur < hi)
                P(f"  {lab:<12}{int(m.sum()):>7}" + "".join(f"{cell(int((cat[m] == c).sum()), int(m.sum())):>19}" for c in cols))
            P(f"  reparto de cada categoria por tramo (% de la categoria; n total entre parentesis):")
            P(f"  {'duracion GT':<12}" + "".join(f"{f'{c} ({int((cat == c).sum())})':>29}" for c in op.CATS_FN))
            for lo, hi, lab in BINS_S:
                m = (dur >= lo) & (dur < hi)
                P(f"  {lab:<12}" + "".join(f"{pct(int(((cat == c) & m).sum()), int((cat == c).sum())):>28.1f}%" for c in op.CATS_FN))
        # 9
        P("\n9. REINICIOS: posesiones GT abiertas por un reinicio y desplazamiento del inicio (pred - GT, frames) de las emparejadas, contra `label` y contra `label_rw`.")
        n_ev = sum(r["n_rw"] for r in rs)
        P(f"   Reinicios del loader en los periodos 1-2: {n_ev} ({n_ev / (n_valid / FPS / 60):.2f} por minuto); frames en ventana {100 * sum(r['rw_frames'] for r in rs) / n_valid:.1f} % de los frames validos, "
          f"de ellos vivos en `label` (H o A) {100 * sum(r['rw_live'] for r in rs) / max(sum(r['rw_frames'] for r in rs), 1):.1f} %.")
        lag = np.array([x[3] for r in rs for x in r["T1"]["rs"] if x[1] == "in"], float)
        lag_rw = np.array([x[3] for r in rs for x in r["T1"]["rs_rw"] if x[1] == "after"], float)
        P(f"   Inicio de la posesion GT menos el frame del evento de reinicio: `label` ('dentro', n {len(lag)}) mediana {np.nanmedian(lag):.0f}, {pct(int((lag == 0).sum()), len(lag)):.1f} % exactamente en el frame del evento; "
          f"`label_rw` ('despues D', n {len(lag_rw)}) mediana {np.nanmedian(lag_rw):.0f}, p10 {np.nanpercentile(lag_rw, 10):.0f}, p90 {np.nanpercentile(lag_rw, 90):.0f}.")
        for gt_key, lab in (("rs", "label"), ("rs_rw", "label_rw")):
            for s_ in SERIES:
                recs = [x for r in rs for x in r[s_][gt_key]]
                n_all = len(recs)
                P(f"  GT {lab}, {s_}: {n_all} posesiones GT.")
                P(f"  {'tipo de reinicio':<20}{'dentro':>8}{'despues D':>11}{'total':>8}{'% de GT':>9}{'empar.':>8}" + f"{'':<1}" + f"{'mediana':>9}{'media':>9}{'p10':>8}{'p90':>8}{'% >12':>9}")
                for kind, name in list(rt.KINDS.items()) + [(-1, "TODOS los reinicios"), (0, "sin reinicio")]:
                    sel = [x for x in recs if (x[0] > 0 if kind == -1 else x[0] == kind)]
                    if not sel and kind > 0:
                        continue
                    n_in, n_af = sum(x[1] == "in" for x in sel), sum(x[1] == "after" for x in sel)
                    sh = np.array([x[2] for x in sel], float)
                    st = shift_stats(sh[~np.isnan(sh)])
                    P(f"  {name:<20}{n_in:>8}{n_af:>11}{len(sel):>8}{pct(len(sel), n_all):>8.1f}%{st['n']:>8}{'':<1}{st['median']:>9.1f}{st['mean']:>9.1f}{st['p10']:>8.1f}{st['p90']:>8.1f}{st['big']:>8.1f}%")
        # 10
        P(f"\n10. RECALL de los cambios H <-> A (existencia, a +-{TOLS[0]} / +-{TOLS[1]} frames) fuera y dentro de una restart_window (+-{rt.MARGIN} frames de margen), junto al recall total.")
        P(f"   {'':<8}" + "".join(f"{lab:>30}" for lab in ("todos", "fuera de ventana", "dentro de ventana (+-25)")))
        for s_ in SERIES:
            hit = np.concatenate([r[s_]["tr_win"][0] for r in rs])
            near = np.concatenate([r[s_]["tr_win"][1] for r in rs])
            cells = []
            for m in (np.ones(len(near), bool), ~near, near):
                cells.append(f"{int(m.sum()):>5} " + " ".join(f"{pct(hit[m, k].sum(), m.sum()):5.1f}%" for k in range(len(TOLS))))
            P(f"   {s_:<8}" + "".join(f"{c:>30}" for c in cells))
        P(f"   (cada celda: n transiciones GT, recall a +-{TOLS[0]}, recall a +-{TOLS[1]})")
        # 11
        P("\n11. GT SECUNDARIA `label_rw` (`label` con D en todos los frames con restart_window > 0): spellF1 medio por partido (40) contra cada GT.")
        P(f"   Posesiones GT: label {sum(r['T1']['n_gt'] for r in rs)}, label_rw {sum(r['T1']['n_gt_rw'] for r in rs)}.")
        f = {(s_, g): np.array([R[m][s_][k] for m in mids]) for s_ in SERIES for g, k in (("label", "f1"), ("label_rw", "f1_rw"))}
        P(f"   {'':<8}{'label':>9}{'label_rw':>10}{'rw - label':>12}")
        for s_ in SERIES:
            P(f"   {s_:<8}{f[(s_, 'label')].mean():>9.4f}{f[(s_, 'label_rw')].mean():>10.4f}{(f[(s_, 'label_rw')] - f[(s_, 'label')]).mean():>+12.4f}")
        P(f"   Delta por partido T1 - base:\n{rp.HEADER}")
        for g in ("label", "label_rw"):
            P(rp.fmt_stats(f"T1 - base, contra {g}", ep.delta_stats(f[("T1", g)] - f[("base", g)])))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procesos", type=int, default=min(9, os.cpu_count() or 1))
    args = ap.parse_args()
    t0 = time.time()
    mids = list(pff_split.DESARROLLO)
    assert not set(mids) & set(pff_split.RESERVADOS)
    lam = pr.lambda_final()
    with mp.get_context("fork").Pool(args.procesos) as pool:
        rows = pool.starmap(functools.partial(work, lam=lam), [(m, b) for b in BALLS for m in mids])
    ref = barrido_row(lam)
    for b in BALLS:
        got = np.mean([r["T1"]["f1"] for r in rows if r["balon"] == b])
        print(f"cordura {b}: spellF1 de T1 {got:.4f}" + (f", barrido_lambda_informe {ref[BALLS.index(b)]:.4f}" if ref else ", sin fila de referencia"))
        if ref and abs(got - ref[BALLS.index(b)]) > CORDURA_TOL:
            sys.exit(f"CORDURA FALLA en {b}: {got:.4f} contra {ref[BALLS.index(b)]:.4f} del barrido; no se escribe el informe.")
    text = (report(rows, lam, ref) + "\n\n== Lectura ==\n" + lectura(rows, lam) + "\n\n== Ampliacion (secciones 8 a 11) ==\n" + report_extra(rows, lam)
            + f"\n\nTiempo: {time.time() - t0:.0f} s.")
    print(text)
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as f:
        f.write(text + "\n")
    print(f"\n-> {REPORT}")


if __name__ == "__main__":
    main()
