"""Preregistered rounds on the PFF development bank: nested validation of the candidates against B3.

    challenge/.venv/bin/python bancos/ronda_pff.py [--ronda 1|2|3|4] [--procesos P] [--rehacer] [--solo-resumen] [--sin-semantica]

Round 1 = C1, C2, C3 (--ronda 1); round 2 = R2 (default). Candidate R2: absorption of micro-possessions of B3 (src/absorcion.py),
tau in {0; 0.2; 0.4; 0.6; 0.8} s, tau = 0 the null configuration.

Round 4 = R4, conditional absorption by control (src/absorcion.py `absorb_conditional`), tau in {1.2; 1.6; 2.0; 3.0} s x mode in
{any, own}; measured against the CURRENT BASE B3 + unconditional absorption tau 1.6 (cached as ("R2", "tau1.6")), on `label`, `label_aerial` (stress GT,
bancos/gt_aerial.py) and `label_poss`, with its own cache directory (CACHE4) and its own decision rule (`verdict4`, not `verdict`). Rounds 1 to 3 are
untouched: their functions are not edited (the fingerprint of their cache files hashes them).

Candidates (each on top of B3, measured against B3; B3 = toques v2 `r3.5_dv2.0_w5_c1.0x12`, team `dir`, cap 5 s):
  C1  recalibration of v2: dv x win x r (3 x 3 x 2 = 18), `dir` and the control touches fixed.
  C2  geometric veto of the owner change at the deflections (src/veto_dueno.py): r_chg x d_rel (3 x 4 = 12).
  C3  live / dead decision of the holes longer than 5 s (src/huecos_vivos.py): theta x L_max x mode (3 x 2 x 2 = 12), edge 1 m.
Every configuration of every candidate is run on the 40 development matches with the ball variants crudo / g4 / g6 / g8
(g4 and crudo only as stress tests) and both ground truths (`label`, `label_poss` informative). Each (match, candidate,
configuration) is cached in src/out/ronda_cache/<CODE_VERSION>/<match>__<cand>__<tag>.csv. A cache file is accepted only if it carries
the fingerprint (`fingerprint()`: source of the modules and functions that take part in the prediction + the parameters + the ball
smoothing), its identity (match, candidate, tag), the full set of 8 (ball, GT) rows with finite spellF1 and, on the g6 / label row,
the counters; otherwise it is recomputed (--rehacer recomputes everything).

Nested validation: 5 folds of 8 matches (FOLDS, fixed here, stratified by phase). In each fold the configuration of each
candidate is chosen on the 32 training matches maximising min(mean delta g6, mean delta g8) against B3; ties (within TIE): fewer
free parameters, then more matches won (min over g6 / g8), then the order of the grid. The reported deltas are the
out-of-fold ones: every match is scored with the configuration its fold chose without seeing it. Entry rule (`verdict`):
min(mean g6, mean g8) >= +0.006, lower bound of the 95 % bootstrap CI > 0 in g6 and in g8, median per match > 0 in both.

Choices fixed in the code (decided by simplicity before any figure; each is repeated in the report):
  * free parameters for the tie: C1 3; C2 2 + (d_rel finite); C3 2 + (L_max finite).
  * "main parameter" of the stability check: C1 dv, C2 r_chg, C3 theta.
  * matches won in the tie = min(wins g6, wins g8) on the 32 training matches.
  * a candidate is a single intervention (C2 has three conditions but one mechanism): no ablation.
  * counters (informative, ball g6, `label`): see counters().
Round 2, same style: free parameters of R2 for the tie: 0 for tau = 0 (null configuration), 1 otherwise; "main parameter": tau; R2 is
applied to the prediction of B3 as a post-process (labels + period array only); in the semantics report R2 uses, per match, the tau
its fold chose (out of fold); a stretch is "micro" when its duration is strictly below tau.
Inputs of the methods: ball, player slots, period and pitch of the raw tracking through pff_split.load_dev (the loader's own smoothing is never switched on);
`label`, `label_poss`, `is_touch` are read only here, to score and to count, never passed to a method.
"""
import argparse
import csv
import functools
import hashlib
import inspect
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
import absorcion as ab  # noqa: E402
import dfl_degradado as dd  # noqa: E402
import diagnostico_pff_sobrantes as dps  # noqa: E402
import equipo_toque as eq  # noqa: E402
import evaluar_pff as ep  # noqa: E402
import evaluar_toques as ev  # noqa: E402
import gt_aerial as ga  # noqa: E402
import huecos_vivos as hv  # noqa: E402
import pff_split  # noqa: E402
import possession_metrics as pm  # noqa: E402
import reglas  # noqa: E402
import toques  # noqa: E402
import veto_dueno as vd  # noqa: E402

CODE_VERSION = "v1"
CACHE = os.path.join(onside_paths.OUT_DIR, "ronda_cache", CODE_VERSION)
REPORT = {1: os.path.join(onside_paths.OUT_DIR, "ronda_pff_informe.txt"),
          2: os.path.join(onside_paths.OUT_DIR, "ronda_pff_informe_r2.txt"),
          3: os.path.join(onside_paths.OUT_DIR, "ronda_pff_informe_r3.txt"),
          4: os.path.join(onside_paths.OUT_DIR, "ronda_pff_informe_r4.txt")}
CACHE4 = os.path.join(onside_paths.OUT_DIR, "ronda_cache", CODE_VERSION + "_r4")      # round 4 rows carry a third GT: own directory
CHALLENGE_GT = os.path.join(onside_paths.EVALUATION_DIR, "gt", "BAR-ATH_possession_gt_public.csv")

BALLS = ("crudo", "g4", "g6", "g8")
MAIN_BALLS = ("g6", "g8")
GTS = ("label", "label_poss")
GTS4 = GTS + ("label_aerial",)                  # round 4 only; rounds 1-3 keep their cache rows unchanged
V2 = ep.V2
TIE = ep.TIE
EDGE_M = 1.0                                   # C3: edge veto of the holes
ENTRY_MIN = 0.006
ROUNDS = {1: ("C1", "C2", "C3"), 2: ("R2",), 3: ("R3",), 4: ("R4",)}
R2_LIKE = ("R2", "R3")                         # R3 = the same mechanism (src/absorcion.py) with the round 3 grid
ENTRY_ORDER = ROUNDS[1]
COUNTERS = ["n_det", "n_ev", "sobrantes", "cambia", "dentro", "parte", "parte_toques", "n_huecos", "reales_vivos", "vivos", "vivos_ok"]
FIELDS = ["partido", "balon", "gt", "cand", "tag", "spell_f1", "n_gt", "n_pred"] + COUNTERS + ["huella"]

# 5 folds of 8: inside each phase the development ids are ordered by SHA-256(FOLD_SALT + id) and dealt round-robin (rank % 5),
# which gives 6 group + 2 knockout matches per fold. Blind like the split: only id and phase are read. tests/test_ronda_pff.py
# regenerates the literal.
FOLD_SALT = "onside-ronda-v1"
FOLDS = (
    ("3826", "3827", "3836", "3842", "3846", "3852", "10506", "10509"),
    ("3813", "3818", "3823", "3838", "3843", "3844", "10503", "10508"),
    ("3816", "3819", "3820", "3832", "3837", "3859", "10504", "10512"),
    ("3829", "3833", "3840", "3848", "3851", "3855", "10510", "10515"),
    ("3812", "3817", "3847", "3850", "3853", "3856", "10507", "10516"),
)


def make_folds(phases, dev=pff_split.DESARROLLO, salt=FOLD_SALT, k=5):
    """Fold assignment regenerated from {id: phase}: tuple of k tuples sorted by integer id."""
    out = [[] for _ in range(k)]
    for ph in sorted(set(phases[i] for i in dev)):
        ids = sorted((i for i in dev if phases[i] == ph), key=lambda i: hashlib.sha256((salt + i).encode()).hexdigest())
        for r, i in enumerate(ids):
            out[r % k].append(i)
    return tuple(tuple(sorted(f, key=int)) for f in out)


def fold_of(mid, folds=FOLDS):
    return next(k for k, f in enumerate(folds) if mid in f)


# ------------------------------------------------------------------------------------------------ candidates

C1_GRID = [(f"dv{dv:g}_w{w}_r{r:g}", (dv, w, r)) for dv, w, r in itertools.product((1.5, 2.0, 2.5), (5, 7, 9), (3.0, 3.5))]
C2_GRID = [(f"rchg{rc:g}_dr{'inf' if np.isinf(dr) else format(dr, 'g')}", (rc, dr))
           for rc, dr in itertools.product((1.5, 2.0, 2.5), (0.0, 0.5, 1.0, np.inf))]
C3_GRID = [(f"th{th:g}_L{'inf' if np.isinf(lm) else format(lm, 'g')}_{m}", (th, lm, m))
           for th, lm, m in itertools.product((0.65, 0.85, 1.05), (10.0, np.inf), ("pre", "split"))]
R2_GRID = [(f"tau{t:g}", (t,)) for t in (0.0, 0.2, 0.4, 0.6, 0.8)]
R3_GRID = [(f"tau{t:g}", (t,)) for t in (0.0, 0.8, 1.0, 1.2, 1.6, 2.0)]
R4_GRID = [(f"tau{t:g}_{m}", (t, m)) for t, m in itertools.product((1.2, 1.6, 2.0, 3.0), ("any", "own"))]
GRIDS = {"C1": C1_GRID, "C2": C2_GRID, "C3": C3_GRID, "R2": R2_GRID, "R3": R3_GRID, "R4": R4_GRID}
BASE_OF = {"R4": ("R2", "tau1.6")}               # (cand, tag) each candidate is measured against; the rest: B3. Round 3's R2 tau 1.6 is the base of the models (M1, T1)
B3_C1 = (V2["dv"], V2["win"], V2["r"])         # the base is the C1 grid point dv2.0_w5_r3.5
MAIN_PARAM = {"C1": 0, "C2": 0, "C3": 0, "R2": 0, "R3": 0, "R4": 0}       # index of dv / r_chg / theta / tau in the parameter tuple
MAIN_NAME = {"C1": "dv", "C2": "r_chg", "C3": "theta", "R2": "tau", "R3": "tau", "R4": "tau"}
NAMES = {"C1": "recalibracion de v2 (dv, win, r)", "C2": "veto geometrico del cambio de dueño (r_chg, d_rel)",
         "C3": "hueco vivo o muerto (theta, L_max, modo)", "R2": "absorcion de microposesiones (tau)",
         "R3": "absorcion de microposesiones, rejilla ampliada (tau)",
         "R4": "absorcion condicional por control (tau, modo)"}


def n_free(cand, p):
    """Free parameters of a configuration (tie-break 1): the ones that actually act."""
    if cand in R2_LIKE:
        return 0 if p[0] == 0 else 1
    if cand == "R4":
        return 2                                  # tau and mode both act in every configuration of the grid
    return 3 if cand == "C1" else 2 + (0 if np.isinf(p[1]) else 1)


# ------------------------------------------------------------------------------------------------ method side

def pipeline(ctx, c1=None, c2=None, c3=None, r2=None):
    """B3 with the candidates applied in the order C1, C2, C3, then R2 (None = off; r2 = tau in seconds). Returns (pred, touch frames, alive mask or None).
    Method side only: it reads ctx's tracking arrays (ball, slots, per, pitch, near, runs, b0, ctrl, dists, tab)."""
    dv, win, r = c1 or B3_C1
    cand = toques.deflection_candidates(ctx["near"], ctx["runs"], r=r, dv=dv, win=win)
    steams = eq.deflection_teams(ep.MODE, cand, ctx["near"], ctx["slots"], ctx["pitch"], r=r, win=win)
    cf, ct = ctx["ctrl"]
    tf, tt = toques.merge_touches((cand["frame"], steams), (cf, ct))
    if c2 is not None:
        is_defl = np.concatenate([np.ones(len(cand["frame"]), bool), np.zeros(len(cf), bool)])[
            np.argsort(np.concatenate([cand["frame"], cf]), kind="stable")]
        keep = vd.veto_owner_change(tf, tt, is_defl, ctx["dists"], ctx["runs"], ctx["per"], ctx["b0"], c2[0], c2[1],
                                    int(ep.CAP * toques.FPS))
        tf, tt = tf[keep], tt[keep]
    pred = toques.predict(ctx["ball"], ctx["slots"], ctx["per"], ctx["pitch"], touches=(tf, tt), max_gap_s=ep.CAP,
                          near=ctx["near"], base=ctx["b0"], **V2)
    alive = None
    if c3 is not None:
        theta, l_max, mode = c3
        alive = hv.decide(ctx["tab"], theta, l_max, EDGE_M)
        pred = hv.fill(pred, ctx["tab"], alive, mode)
    if r2 is not None:
        pred = ab.absorb(pred, r2, ctx["per"])
    return pred, tf, alive


def build_method_ctx(ball, slots, per, pitch):
    present = ~np.isnan(ball[:, 0])
    near = toques.prepare(ball, slots, pitch)
    runs = toques.visible_runs(present, per)
    b0 = reglas.predict("B0", ball, slots, per, pitch)
    return {"ball": ball, "slots": slots, "per": per, "pitch": pitch, "near": near, "runs": runs, "b0": b0,
            "b1": reglas.fill_gaps(b0, present, per, "F", ep.CAP),
            "ctrl": toques.control_touches(near, runs, r_ctrl=V2["r_ctrl"], k=V2["k"]),
            "dists": vd.team_distances(ball, slots, pitch),
            "tab": hv.hole_table(ball, slots, per, pitch)}


# ------------------------------------------------------------------------------------------------ evaluation side

def counters(m, ev_ctx, tf, pred, alive):
    """Informative counters of one prediction (ball g6, `label`): touches with no event touch (`sobrantes`), of those that change
    the owner of B1 (`cambia`), lie inside one GT possession (`dentro`) and leave its owner (`parte`: UNIQUE GT possessions split; `parte_toques`: touches that split);
    holes longer than 5 s: all, really alive (GT share of D <= 0.5), decided alive and decided alive that are really alive."""
    ex = dps.extras(tf, ev_ctx["event_f"], m["b1"], pred, ev_ctx["label"])
    tab = m["tab"]
    real = np.array([np.mean(ev_ctx["label"][a:b] == "D") <= 0.5 for a, b in zip(tab["start"], tab["end"])], bool)
    al = np.zeros(len(real), bool) if alive is None else alive
    return {"n_det": ex["n_det"], "n_ev": ex["n_ev"], "sobrantes": ex["sobrantes"], "cambia": ex["cambia"],
            "dentro": ex["dentro"], "parte": ex["parte"], "parte_toques": ex["parte_toques"], "n_huecos": len(real), "reales_vivos": int(real.sum()),
            "vivos": int(al.sum()), "vivos_ok": int((al & real).sum())}


def load_match(mid):
    raw = pff_split.load_dev(mid)
    keep = np.asarray(raw["period"]) <= ep.LAST_PERIOD
    d = ep.clip_periods(raw)
    d["event_f"] = np.flatnonzero(np.asarray(raw["is_touch"])[keep])
    return d


def ball_variant(d, balon):
    return d["ball_xy"] if balon == "crudo" else dd.smooth_ball(d["ball_xy"], d["period"], balon)


def cache_file(mid, cand, tag, ronda=1):
    return os.path.join(CACHE4 if ronda == 4 else CACHE, f"{mid}__{cand}__{tag}.csv")


def all_configs(ronda=1):
    """B3 plus the grid of each candidate. R3 is computed and cached as R2 (same mechanism, same fingerprint, same files): the
    taus shared with round 2 are reused; load_cache renames the candidate back to R3. Round 4 compares against R2 tau 1.6, so it is in the set."""
    if ronda == 4:
        return [("B3", "B3", None), ("R2", "tau1.6", dict(R3_GRID)["tau1.6"])] + [("R4", tag, p) for tag, p in GRIDS["R4"]]
    return [("B3", "B3", None)] + [("R2" if c == "R3" else c, tag, p) for c in ROUNDS[ronda] for tag, p in GRIDS[c]]


def args_of(cand, p):
    return {"C1": {"c1": p}, "C2": {"c2": p}, "C3": {"c3": p}, "R2": {"r2": p[0] if p else None}, None: {}}[None if cand == "B3" else cand]


# ------------------------------------------------------------------------------------------------ cache validity

FP_MODULES = (toques, eq, reglas, vd, hv, dd, ep, ev, pff_split, pm, dps)      # everything a prediction or its score goes through
FP_FUNCTIONS = ("pipeline", "build_method_ctx", "counters", "load_match", "ball_variant", "args_of", "work")


def source_of(obj):
    """Bytes of the source of a module (its file) or of a function."""
    if inspect.ismodule(obj):
        with open(obj.__file__, "rb") as f:
            return f.read()
    return inspect.getsource(obj).encode()


def fingerprint(cand, tag, p):
    """SHA-256 of the code and the inputs that decide the rows of (candidate, tag): source of the modules in FP_MODULES (and of
    absorcion for R2) and of the functions of this file that build the prediction, the parameters, the ball smoothing, the
    detector parameters and the caps."""
    h = hashlib.sha256()
    for m in FP_MODULES + ((ab,) if cand in ("R2", "R4") else ()):
        h.update(source_of(m))
    for name in FP_FUNCTIONS:
        h.update(source_of(globals()[name]))
    h.update(repr((cand, tag, p, sorted(dd.BALL_SMOOTH.items()), BALLS, GTS, sorted(V2.items()), ep.CAP, ep.MODE, EDGE_M,
                   ep.LAST_PERIOD)).encode())
    return h.hexdigest()


FP4_FUNCTIONS = ("pred_r4", "work4", "load_match4")


def fingerprint4(cand, tag, p):
    """Fingerprint of a round 4 cache file: the one of rounds 1-3 (modules, prediction functions, parameters) plus the GT builder, the
    functions of this file that are specific to round 4 and the GT list."""
    h = hashlib.sha256(fingerprint(cand, tag, p).encode())
    h.update(source_of(ga))
    for name in FP4_FUNCTIONS:
        h.update(source_of(globals()[name]))
    h.update(repr((GTS4, ga.CAP_FRAMES)).encode())
    return h.hexdigest()


def cache_ok(mid, cand, tag, p, ronda=1):
    """True when the cache file of (match, candidate, tag) has the fingerprint, the identity, the distinct (ball, GT) rows (8; 12 in
    round 4) with spellF1 in [0, 1] and the counters (all finite) on the g6 / label row."""
    gts = GTS4 if ronda == 4 else GTS
    path = cache_file(mid, cand, tag, ronda)
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            rd = csv.DictReader(f)
            if rd.fieldnames != FIELDS:
                return False
            rows = list(rd)
        fp = (fingerprint4 if ronda == 4 else fingerprint)(cand, tag, p)
        if len(rows) != len(BALLS) * len(gts) or {(r["balon"], r["gt"]) for r in rows} != set(itertools.product(BALLS, gts)):
            return False
        for r in rows:
            if (r["partido"], r["cand"], r["tag"], r["huella"]) != (mid, cand, tag, fp) or not 0.0 <= float(r["spell_f1"]) <= 1.0:
                return False
            if (r["balon"], r["gt"]) == ("g6", "label") and not all(np.isfinite(float(r[k])) for k in COUNTERS):
                return False
        return True
    except (ValueError, KeyError, OSError):
        return False


def work(mid, todo):
    """Rows of every (candidate, tag) in `todo` for match `mid`, all ball variants and both ground truths; written to the cache."""
    d = load_match(mid)
    rows = {(c, t): [] for c, t, _ in todo}
    fprints = {(c, t): fingerprint(c, t, p) for c, t, p in todo}
    for balon in BALLS:
        m = build_method_ctx(ball_variant(d, balon), d["players_slots"], d["period"], d["pitch"])
        gts = {g: ev.make_gt(d, g) for g in GTS}
        for cand, tag, p in todo:
            pred, tf, alive = pipeline(m, **args_of(cand, p))
            for g in GTS:
                r = pm.evaluate(gts[g], pm.label_codes(pred), bootstrap_n=0)
                row = {"partido": mid, "balon": balon, "gt": g, "cand": cand, "tag": tag, "spell_f1": r["spell_f1_mean"],
                       "n_gt": r["n_gt_spells"], "n_pred": r["n_pred_spells"], "huella": fprints[(cand, tag)]}
                if balon == "g6" and g == "label":
                    row.update(counters(m, d, tf, pred, alive))
                rows[(cand, tag)].append(row)
    os.makedirs(CACHE, exist_ok=True)
    for (cand, tag), rs in rows.items():
        with open(cache_file(mid, cand, tag), "w", newline="") as f:
            w = csv.DictWriter(f, FIELDS)
            w.writeheader()
            w.writerows(rs)
    return mid, len(todo)


def load_match4(mid):
    """`load_match` plus `label_aerial` (evaluation side: derived from the PFF events, never passed to a method)."""
    raw = pff_split.load_dev(mid)
    keep = np.asarray(raw["period"]) <= ep.LAST_PERIOD
    d = ep.clip_periods(raw)
    d["event_f"] = np.flatnonzero(np.asarray(raw["is_touch"])[keep])
    d["label_aerial"] = ga.build(mid, raw)[0][keep]
    return d


def pred_r4(ctx, b3, cand, p):
    """Prediction of one round 4 configuration from the prediction `b3` of B3 (the post-processes act on the labels only, so B3 is
    computed once per ball): B3 itself, R2 (unconditional absorption, tau = p[0]) or R4 (conditional, p = (tau, mode))."""
    if cand == "B3":
        return b3
    if cand == "R2":
        return ab.absorb(b3, p[0], ctx["per"])
    cf, ct = ctx["ctrl"]
    return ab.absorb_conditional(b3, p[0], cf, np.where(ct == toques.HOME, "H", "A"), p[1], ctx["per"])


def work4(mid, todo):
    """`work` of round 4: B3 once per ball, every configuration of `todo` as a post-process, three ground truths; rows go to CACHE4."""
    d = load_match4(mid)
    rows = {(c, t): [] for c, t, _ in todo}
    fprints = {(c, t): fingerprint4(c, t, p) for c, t, p in todo}
    for balon in BALLS:
        m = build_method_ctx(ball_variant(d, balon), d["players_slots"], d["period"], d["pitch"])
        gts = {g: ev.make_gt(d, g) for g in GTS4}
        b3, tf, _ = pipeline(m)
        for cand, tag, p in todo:
            pred = pred_r4(m, b3, cand, p)
            for g in GTS4:
                r = pm.evaluate(gts[g], pm.label_codes(pred), bootstrap_n=0)
                row = {"partido": mid, "balon": balon, "gt": g, "cand": cand, "tag": tag, "spell_f1": r["spell_f1_mean"],
                       "n_gt": r["n_gt_spells"], "n_pred": r["n_pred_spells"], "huella": fprints[(cand, tag)]}
                if balon == "g6" and g == "label":
                    row.update(counters(m, d, tf, pred, None))
                rows[(cand, tag)].append(row)
    os.makedirs(CACHE4, exist_ok=True)
    for (cand, tag), rs in rows.items():
        with open(cache_file(mid, cand, tag, 4), "w", newline="") as f:
            w = csv.DictWriter(f, FIELDS)
            w.writeheader()
            w.writerows(rs)
    return mid, len(todo)


def run_grid(procesos, rehacer=False, ronda=1):
    mids = list(pff_split.DESARROLLO)
    assert not set(mids) & set(pff_split.RESERVADOS)
    jobs = []
    for mid in mids:
        todo = [c for c in all_configs(ronda) if rehacer or not cache_ok(mid, *c, ronda=ronda)]
        if todo:
            jobs.append((mid, todo))
    if not jobs:
        print("cache complete: nothing to compute", file=sys.stderr)
        return
    t0 = time.time()
    with mp.get_context("fork").Pool(min(procesos, len(jobs))) as pool:
        pool.starmap(work4 if ronda == 4 else work, jobs, chunksize=1)
    print(f"computed {sum(len(t) for _, t in jobs)} (match, configuration) pairs in {time.time() - t0:.0f}s", file=sys.stderr)


def load_cache(ronda=1):
    """{(cand, tag, balon, gt): {mid: spell_f1}} and {(cand, tag, mid): counters dict} from the cache files (valid ones only)."""
    f1, cnt = {}, {}
    for mid in pff_split.DESARROLLO:
        for cand, tag, p in all_configs(ronda):
            if not cache_ok(mid, cand, tag, p, ronda):
                raise SystemExit(f"stale or incomplete cache file: {cache_file(mid, cand, tag, ronda)} (run without --solo-resumen)")
            with open(cache_file(mid, cand, tag, ronda)) as f:
                for r in csv.DictReader(f):
                    f1.setdefault((cand, tag, r["balon"], r["gt"]), {})[mid] = float(r["spell_f1"])
                    if r["sobrantes"] != "":
                        cnt[(cand, tag, mid)] = {k: float(r[k]) for k in COUNTERS}
    if ronda == 3:                                  # configurations cached as R2 (see all_configs) belong to R3 here
        f1 = {("R3" if k[0] == "R2" else k[0],) + k[1:]: v for k, v in f1.items()}
        cnt = {("R3" if k[0] == "R2" else k[0],) + k[1:]: v for k, v in cnt.items()}
    return f1, cnt


# ------------------------------------------------------------------------------------------------ selection

def deltas(f1, cand, tag, balon, mids, gt="label"):
    a, b = f1[(cand, tag, balon, gt)], f1[BASE_OF.get(cand, ("B3", "B3")) + (balon, gt)]
    return np.array([a[m] - b[m] for m in mids])


def select(f1, cand, train):
    """(tag, params) of the configuration of `cand` that maximises min(mean g6, mean g8) of the delta against B3 over `train`;
    ties within TIE: fewer free parameters, more matches won (min g6 / g8), earlier in the grid."""
    scored = []
    for i, (tag, p) in enumerate(GRIDS[cand]):
        d6, d8 = (deltas(f1, cand, tag, b, train) for b in MAIN_BALLS)
        wins = min(int((d6 > TIE).sum()), int((d8 > TIE).sum()))
        scored.append((min(d6.mean(), d8.mean()), n_free(cand, p), wins, i, tag, p))
    best = max(s[0] for s in scored)
    tied = [s for s in scored if s[0] >= best - TIE]
    s = min(tied, key=lambda s: (s[1], -s[2], s[3]))
    return s[4], s[5]


def nested(f1, cand, folds=FOLDS):
    """({mid: (tag, params)} chosen without seeing the match, [tag per fold])."""
    chosen, per_fold = {}, []
    for k, test in enumerate(folds):
        train = [m for j, f in enumerate(folds) if j != k for m in f]
        tag, p = select(f1, cand, train)
        per_fold.append(tag)
        for m in test:
            chosen[m] = (tag, p)
    return chosen, per_fold


def oof_deltas(f1, cand, chosen, balon, gt="label"):
    mids = list(pff_split.DESARROLLO)
    base = BASE_OF.get(cand, ("B3", "B3"))
    return np.array([f1[(cand, chosen[m][0], balon, gt)][m] - f1[base + (balon, gt)][m] for m in mids])


def verdict(s6, s8):
    """Entry rule literally: [(condition, ok)] and the overall decision."""
    conds = [(f"min(media g6, media g8) = {min(s6['mean'], s8['mean']):+.4f} >= +{ENTRY_MIN}", min(s6["mean"], s8["mean"]) >= ENTRY_MIN),
             (f"IC95 inf g6 = {s6['lo']:+.4f} > 0", s6["lo"] > 0), (f"IC95 inf g8 = {s8['lo']:+.4f} > 0", s8["lo"] > 0),
             (f"mediana g6 = {s6['median']:+.4f} > 0", s6["median"] > 0), (f"mediana g8 = {s8['median']:+.4f} > 0", s8["median"] > 0)]
    return conds, all(ok for _, ok in conds)


def check_base(f1, out):
    """B3 recomputed here against the accepted ladder (src/out/pff_escalera_<ball>.csv, `label`): max |difference| per ball."""
    for b in BALLS:
        ref = {r["partido"]: r["spell_f1"] for r in ep.load_rows(os.path.join(onside_paths.OUT_DIR, f"pff_escalera_{b}.csv"))
               if r["gt"] == "label" and r["paso"] == "B3"}
        mine = f1[("B3", "B3", b, "label")]
        out.append(f"B3 {b}: media {np.mean(list(mine.values())):.4f}, diferencia maxima con la escalera aceptada "
                   f"{max(abs(mine[m] - ref[m]) for m in mine):.2e}")


def fmt_stats(label, s):
    return (f"{label:<28}{s['mean']:>+9.4f}{s['wins']:>5}/{s['losses']:<3}{s['ties']:>4}{s['lo']:>+10.4f}{s['hi']:>+10.4f}"
            f"{s['median']:>+10.4f}{s['sd']:>9.4f}")


HEADER = f"{'':<28}{'media':>9}{'gana/pierde':>13}{'emp':>4}{'IC95 inf':>10}{'IC95 sup':>10}{'mediana':>10}{'sd':>9}"


# ------------------------------------------------------------------------------------------------ report

def report_candidate(f1, cnt, cand, out):
    chosen, per_fold = nested(f1, cand)
    P = lambda s="": out.append(s)                                           # noqa: E731
    P(f"\n=== {cand}: {NAMES[cand]} ({len(GRIDS[cand])} configuraciones) ===")
    P("Configuracion elegida por pliegue (entrenamiento = los otros 32):")
    mp_idx = MAIN_PARAM[cand]
    params = dict(GRIDS[cand])
    for k, tag in enumerate(per_fold):
        P(f"  pliegue {k + 1}: {tag}   (partidos de prueba: {len(FOLDS[k])})")
    main_vals = {params[t][mp_idx] for t in per_fold}
    estable = len(main_vals) == 1
    P(f"  parametro principal ({MAIN_NAME[cand]}): valores {sorted(main_vals)} -> "
      f"{'ESTABLE' if estable else 'INESTABLE'}; configuracion identica en los 5: {'si' if len(set(per_fold)) == 1 else 'no'}")
    s = {b: ep.delta_stats(oof_deltas(f1, cand, chosen, b)) for b in MAIN_BALLS}
    P(f"\nTabla 8.20 (delta contra B3, spellF1, fuera de pliegue, n=40, GT label):\n{HEADER}")
    for b in MAIN_BALLS:
        P(fmt_stats(f"{cand} - B3, {b}", s[b]))
    P(f"min(g6, g8) = {min(s['g6']['mean'], s['g8']['mean']):+.4f}")
    conds, ok = verdict(s["g6"], s["g8"])
    for c, o in conds:
        P(f"  [{'x' if o else ' '}] {c}")
    P(f"VEREDICTO {cand}: {'ENTRA' if ok else 'NO ENTRA'} segun 8.20.1 aplicado literalmente"
      f"{' (marcado INESTABLE por 8.21)' if ok and not estable else ''}")
    P(f"\nInformativo (mismas configuraciones fuera de pliegue, sin veto):\n{HEADER}")
    for b in ("g4", "crudo"):
        P(fmt_stats(f"{cand} - B3, {b}", ep.delta_stats(oof_deltas(f1, cand, chosen, b))))
    for b in MAIN_BALLS:
        P(fmt_stats(f"{cand} - B3, {b}, label_poss", ep.delta_stats(oof_deltas(f1, cand, chosen, b, "label_poss"))))
    mids = list(pff_split.DESARROLLO)
    P("\nPanorama de la rejilla con los 40 partidos (EN MUESTRA, sólo para ver el paisaje, no elige nada): min(g6, g8) de la media")
    land = [(min(deltas(f1, cand, t, "g6", mids).mean(), deltas(f1, cand, t, "g8", mids).mean()), t) for t, _ in GRIDS[cand]]
    P("  " + "  ".join(f"{t}:{v:+.4f}" for v, t in land))
    counter_block(cnt, cand, chosen, out)
    return {"chosen": chosen, "per_fold": per_fold, "ok": ok, "estable": estable, "stats": s}


def counter_block(cnt, cand, chosen, out):
    """Objective counter of each candidate (ball g6, label), per match mean over the 40, configuration of the fold vs B3."""
    mids = list(pff_split.DESARROLLO)
    c = np.array([[cnt[(cand, chosen[m][0], m)][k] for k in COUNTERS] for m in mids])
    b = np.array([[cnt[("B3", "B3", m)][k] for k in COUNTERS] for m in mids])
    ix = {k: i for i, k in enumerate(COUNTERS)}

    def pm_(a, k):
        return a[:, ix[k]].mean()
    out.append("Contador objetivo (g6, por partido, media de los 40; candidato fuera de pliegue | B3):")
    if cand == "C1":
        out.append(f"  toques sobrantes (sin toque de evento, todos los eventos): {pm_(c, 'sobrantes'):.1f} | {pm_(b, 'sobrantes'):.1f}"
                   f"; detectados {pm_(c, 'n_det'):.1f} | {pm_(b, 'n_det'):.1f}; eventos {pm_(b, 'n_ev'):.1f}")
    elif cand in ("C2",) + R2_LIKE:
        out.append(f"  posesiones GT partidas (identificadores unicos, `parte`): {pm_(c, 'parte'):.1f} | {pm_(b, 'parte'):.1f}"
                   f"; por toque (`parte_toques`): {pm_(c, 'parte_toques'):.1f} | {pm_(b, 'parte_toques'):.1f}"
                   f"; sobrantes que cambian dueño (cambia) {pm_(c, 'cambia'):.1f} | {pm_(b, 'cambia'):.1f}"
                   f"; dentro de una posesion GT {pm_(c, 'dentro'):.1f} | {pm_(b, 'dentro'):.1f}; toques {pm_(c, 'n_det'):.1f} | {pm_(b, 'n_det'):.1f}")
    else:
        v, ok, real = c[:, ix["vivos"]].sum(), c[:, ix["vivos_ok"]].sum(), c[:, ix["reales_vivos"]].sum()
        out.append(f"  huecos > 5 s por partido {pm_(c, 'n_huecos'):.1f}; realmente vivos (GT, D <= 50 %) {pm_(c, 'reales_vivos'):.1f}; "
                   f"decididos vivos {pm_(c, 'vivos'):.1f} (B3: 0); acierto de los vivos {ok / v:.3f} ({int(ok)}/{int(v)})" if v else
                   f"  huecos > 5 s por partido {pm_(c, 'n_huecos'):.1f}; decididos vivos 0")
        if v:
            out.append(f"  recuperados de los realmente vivos {ok / real:.3f}")


# ------------------------------------------------------------------------------------------ overlap and combination

def combo_work(mid, cfgs, names):
    """Per match: modified frames (vs B3, g6) of each entering candidate and of the combination, and spellF1 of B3 / combination
    for g6 / g8 (label, label_poss) and g4 / crudo (label)."""
    d = load_match(mid)
    res = {"f1": {}, "mod": {}}
    for balon in BALLS:
        m = build_method_ctx(ball_variant(d, balon), d["players_slots"], d["period"], d["pitch"])
        base, _, _ = pipeline(m)
        kw = {n: args_of(n, cfgs[n]) for n in names}
        comb = {}
        for n in names:
            comb.update(kw[n])
        pc, _, _ = pipeline(m, **comb)
        for g in GTS:
            gt = ev.make_gt(d, g)
            res["f1"][(balon, g)] = (pm.evaluate(gt, pm.label_codes(base), bootstrap_n=0)["spell_f1_mean"],
                                     pm.evaluate(gt, pm.label_codes(pc), bootstrap_n=0)["spell_f1_mean"])
        if balon == "g6":
            masks = {n: pipeline(m, **kw[n])[0] != base for n in names}
            masks["combo"] = pc != base
            res["mod"] = {"n": {k: int(v.sum()) for k, v in masks.items()},
                          "inter": {f"{a}&{b}": int((masks[a] & masks[b]).sum()) for a, b in itertools.combinations(names, 2)},
                          "union": int(np.logical_or.reduce([masks[n] for n in names]).sum()), "frames": len(base)}
    return mid, res


def report_combination(entering, sel, procesos, out):
    P = out.append
    P(f"\n=== Solapamiento y combinacion fuera de pliegue ({' -> '.join(entering)}) contra B3, informativo ===")
    jobs = []
    for mid in pff_split.DESARROLLO:
        k = fold_of(mid)
        jobs.append((mid, {n: dict(GRIDS[n])[sel[n]["per_fold"][k]] for n in entering}, entering))
    with mp.get_context("fork").Pool(min(procesos, len(jobs))) as pool:
        res = dict(pool.starmap(combo_work, jobs, chunksize=1))
    mids = list(pff_split.DESARROLLO)
    n = {k: sum(res[m]["mod"]["n"][k] for m in mids) for k in res[mids[0]]["mod"]["n"]}
    fr = sum(res[m]["mod"]["frames"] for m in mids)
    P("Frames modificados respecto de B3 (g6, suma de los 40; % de los frames):")
    for k, v in n.items():
        P(f"  {k:<7}{v:>9}  {100 * v / fr:.2f} %")
    for key in res[mids[0]]["mod"]["inter"]:
        i = sum(res[m]["mod"]["inter"][key] for m in mids)
        a, b = key.split("&")
        P(f"  interseccion {key}: {i} frames ({100 * i / max(1, min(n[a], n[b])):.1f} % del menor)")
    P(f"  union de los candidatos: {sum(res[m]['mod']['union'] for m in mids)}; combinacion: {n['combo']}")
    hdr = HEADER
    P(f"\nDelta de la combinacion contra B3 (fuera de pliegue):\n{hdr}")
    for balon, g in (("g6", "label"), ("g8", "label"), ("g4", "label"), ("crudo", "label"), ("g6", "label_poss"), ("g8", "label_poss")):
        d = np.array([res[m]["f1"][(balon, g)][1] - res[m]["f1"][(balon, g)][0] for m in mids])
        P(fmt_stats(f"combo - B3, {balon}, {g}", ep.delta_stats(d)))
    s6 = ep.delta_stats(np.array([res[m]["f1"][("g6", "label")][1] - res[m]["f1"][("g6", "label")][0] for m in mids]))
    s8 = ep.delta_stats(np.array([res[m]["f1"][("g8", "label")][1] - res[m]["f1"][("g8", "label")][0] for m in mids]))
    P(f"min(g6, g8) = {min(s6['mean'], s8['mean']):+.4f}; 8.20.1 sobre la combinacion: {'CUMPLE' if verdict(s6, s8)[1] else 'NO CUMPLE'} (informativo)")


# ------------------------------------------------------------------------------------------ semantics diagnostic

BINS = (0.5, 1.0, 2.0)


def spell_row(codes, runs):
    sp = pm.possession_spells(codes, runs)
    dur = np.array([(b - a) / toques.FPS for a, b, _ in sp])
    return dur, sum(b - a for a, b, _ in sp) / toques.FPS / 60


def sem_work(mid, taus=()):
    """(match, GT row, B3 row, {tau: R2 row}) of possession durations (g6, periods 1-2)."""
    d = ep.clip_periods(pff_split.load_dev(mid))
    gt = ev.make_gt(d, "label")
    runs = pm.valid_runs(gt)
    m = build_method_ctx(dd.smooth_ball(d["ball_xy"], d["period"], "g6"), d["players_slots"], d["period"], d["pitch"])
    pred, _, _ = pipeline(m)
    return mid, spell_row(gt, runs), spell_row(pm.label_codes(pred), runs), \
        {t: spell_row(pm.label_codes(ab.absorb(pred, t, m["per"])), runs) for t in taus}


SEM_HEADER = f"{'serie':<44}{'min vivos':>10}{'posesiones':>11}{'/min':>7}{'<0,5 s':>8}{'<1 s':>7}{'<2 s':>7}{'media s':>9}"


def sem_line(name, dur, live, live_ref=None):
    """One row of the possessions-by-duration table: `dur` durations (s), `live` minutes of live play of the series, `live_ref` the
    minutes the rates are divided by (default: its own)."""
    den = live_ref or live
    cnt = [int((dur < t).sum()) for t in BINS]
    return (f"{name:<44}{live:>10.1f}{len(dur):>11}{len(dur) / den:>7.2f}{cnt[0] / den:>8.2f}{cnt[1] / den:>7.2f}{cnt[2] / den:>7.2f}"
            f"{dur.mean():>9.2f}")


def report_semantics(procesos, out, chosen_tau=None, grid=R2_GRID, cand="R2"):
    """Possessions per minute by duration. chosen_tau = {mid: tau} adds the R2 lines (out of fold, then every tau in sample)."""
    P = out.append
    gt = pm.label_codes(pm.read_label_csv(CHALLENGE_GT))
    r_gt = spell_row(gt, pm.valid_runs(gt))
    taus = tuple(t for (t,) in dict(grid).values()) if chosen_tau else ()
    with mp.get_context("fork").Pool(min(procesos, 40)) as pool:
        res = pool.map(functools.partial(sem_work, taus=taus), list(pff_split.DESARROLLO))
    cat = lambda rows: (np.concatenate([r[0] for r in rows]), sum(r[1] for r in rows))      # noqa: E731
    g_dur, p_dur = cat([r[1] for r in res]), cat([r[2] for r in res])
    P("\n=== Diagnostico de semantica (sin decision): posesiones por minuto de juego vivo ===")
    P("Posesion = tramo H o A maximo (pm.possession_spells); minuto vivo = frames H/A de la propia serie / 25 / 60; "
      "(c) tambien por minuto vivo del GT de PFF. PFF: 40 de desarrollo, periodos 1-2, suma.")
    P(SEM_HEADER)

    line = lambda *a: P(sem_line(*a))                                       # noqa: E731
    line("(a) GT publico del reto", *r_gt)
    line("(b) GT PFF `label` (objetivo)", *g_dur)
    line("(c) B3 g6 sobre PFF (min vivos propios)", *p_dur)
    line("(c) B3 g6 sobre PFF (min vivos del GT)", p_dur[0], p_dur[1], g_dur[1])
    if chosen_tau:
        oof = cat([r[3][chosen_tau[r[0]]] for r in res])
        line(f"(d) {cand} tau fuera de pliegue (min propios)", *oof)
        line(f"(d) {cand} tau fuera de pliegue (min del GT)", oof[0], oof[1], g_dur[1])
        for t in taus:
            x = cat([r[3][t] for r in res])
            line(f"    {cand} tau={t:g} s en muestra (min del GT)", x[0], x[1], g_dur[1])


# ------------------------------------------------------------------------------------------------ round 4 report

R4_LOSS_OK = 0.005                              # R4 is adopted if it loses at most this much min(g6, g8) in `label` against the base ...
R4_LOSS_ASK = 0.010                             # ... and wins in `label_aerial`; if it loses more than this, the bet goes to manual review


def verdict4(label6, label8, aer6, aer8):
    """Decision rule of round 4, literally. Inputs: mean deltas against the base. Returns ([(condition, ok)], "ADOPTA" | "NO ADOPTA" | "REVISION MANUAL").
    Loses at most 0.005: min(g6, g8) in `label` >= -0.005. Wins in `label_aerial`: both means > 0 (g6 and g8). More than 0.01 lost in `label`:
    the base bets that short possessions are noise and the decision is a manual one, whatever label_aerial says."""
    lo = min(label6, label8)
    conds = [(f"pierde <= {R4_LOSS_OK} en min(g6, g8) de `label`: {lo:+.4f}", lo >= -R4_LOSS_OK),
             (f"gana en `label_aerial` (g6 {aer6:+.4f}, g8 {aer8:+.4f} > 0)", aer6 > 0 and aer8 > 0)]
    if lo < -R4_LOSS_ASK:
        return conds, "REVISION MANUAL"
    return conds, "ADOPTA" if all(ok for _, ok in conds) else "NO ADOPTA"


def sem_work4(mid, cfgs=()):
    """(match, GT row, B3 row, {tag: row}) of possession durations (g6, periods 1-2) for the configurations `cfgs` [(cand, tag, p)]."""
    d = ep.clip_periods(pff_split.load_dev(mid))
    gt = ev.make_gt(d, "label")
    runs = pm.valid_runs(gt)
    m = build_method_ctx(dd.smooth_ball(d["ball_xy"], d["period"], "g6"), d["players_slots"], d["period"], d["pitch"])
    b3, _, _ = pipeline(m)
    return mid, spell_row(gt, runs), spell_row(pm.label_codes(b3), runs), \
        {tag: spell_row(pm.label_codes(pred_r4(m, b3, c, p)), runs) for c, tag, p in cfgs}


def report_semantics4(procesos, out, chosen):
    """Possessions per minute by duration of the base, R4 out of fold and every R4 configuration in sample, against the GT of PFF and the public one."""
    P = out.append
    cfgs = [("R2", "tau1.6", dict(R3_GRID)["tau1.6"])] + [("R4", t, p) for t, p in GRIDS["R4"]]
    gt = pm.label_codes(pm.read_label_csv(CHALLENGE_GT))
    with mp.get_context("fork").Pool(min(procesos, 40)) as pool:
        res = pool.map(functools.partial(sem_work4, cfgs=cfgs), list(pff_split.DESARROLLO))
    cat = lambda rows: (np.concatenate([r[0] for r in rows]), sum(r[1] for r in rows))      # noqa: E731
    g_dur = cat([r[1] for r in res])
    P("\n=== Posesiones por minuto segun duracion (g6, minutos vivos del GT de PFF; sin decision) ===")
    P(SEM_HEADER)
    P(sem_line("(a) GT publico del reto", *spell_row(gt, pm.valid_runs(gt))))
    P(sem_line("(b) GT PFF `label` (objetivo)", *g_dur))
    b3 = cat([r[2] for r in res])
    P(sem_line("(c) B3", b3[0], b3[1], g_dur[1]))
    base = cat([r[3]["tau1.6"] for r in res])
    P(sem_line("(d) base: B3 + tau 1,6 incondicional", base[0], base[1], g_dur[1]))
    oof = cat([r[3][chosen[r[0]][0]] for r in res])
    P(sem_line("(e) R4 fuera de pliegue", oof[0], oof[1], g_dur[1]))
    for t, _ in GRIDS["R4"]:
        x = cat([r[3][t] for r in res])
        P(sem_line(f"    R4 {t} en muestra", x[0], x[1], g_dur[1]))


def report_r4(f1, out):
    """Round 4 report: nested selection against the base, deltas in the three GTs, round 4 verdict, cost of the pure absorption."""
    P = out.append
    cand, base = "R4", BASE_OF["R4"]
    mids = list(pff_split.DESARROLLO)
    chosen, per_fold = nested(f1, cand)
    params = dict(GRIDS[cand])
    P(f"\n=== R4: {NAMES[cand]} ({len(GRIDS[cand])} configuraciones), contra la base B3 + absorcion incondicional tau 1,6 ===")
    P("Configuracion elegida por pliegue (entrenamiento = los otros 32, min(g6, g8) del delta en `label` contra la base):")
    for k, tag in enumerate(per_fold):
        P(f"  pliegue {k + 1}: {tag}   (partidos de prueba: {len(FOLDS[k])})")
    taus, modes = {params[t][0] for t in per_fold}, {params[t][1] for t in per_fold}
    P(f"  parametro principal (tau): valores {sorted(taus)} -> {'ESTABLE' if len(taus) == 1 else 'INESTABLE'}; modo: {sorted(modes)} -> "
      f"{'ESTABLE' if len(modes) == 1 else 'INESTABLE'}; configuracion identica en los 5: {'si' if len(set(per_fold)) == 1 else 'no'}")
    st = {(b, g): ep.delta_stats(oof_deltas(f1, cand, chosen, b, g)) for g in GTS4 for b in BALLS}
    for g, titulo in (("label", "label (decide, g6 y g8; g4 y crudo como estres sin veto)"), ("label_aerial", "label_aerial (estres: decide si gana)"),
                      ("label_poss", "label_poss (informativo)")):
        P(f"\nDelta de R4 contra la base, fuera de pliegue, n=40, GT {titulo}:\n{HEADER}")
        for b in ("g6", "g8", "g4", "crudo"):
            P(fmt_stats(f"R4 - base, {b}, {g}", st[(b, g)]))
    conds, v = verdict4(st[("g6", "label")]["mean"], st[("g8", "label")]["mean"], st[("g6", "label_aerial")]["mean"],
                        st[("g8", "label_aerial")]["mean"])
    P("\nRegla de decision 8.24 (cobertura, no mejora):")
    for c, ok in conds:
        P(f"  [{'x' if ok else ' '}] {c}")
    P(f"  pierde mas de {R4_LOSS_ASK} en `label`: {'SI, revision manual' if v == 'REVISION MANUAL' else 'no'}")
    P(f"VEREDICTO R4: {v}")
    P(f"\nCuanto pierde la absorcion pura (base = B3 + tau 1,6 incondicional) contra B3, 40 partidos, un solo punto fijo (sin seleccion):\n{HEADER}")
    for g in GTS4:
        for b in ("g6", "g8", "g4", "crudo"):
            d = np.array([f1[base + (b, g)][m] - f1[("B3", "B3", b, g)][m] for m in mids])
            P(fmt_stats(f"base - B3, {b}, {g}", ep.delta_stats(d)))
    P("\nPanorama de la rejilla con los 40 partidos (EN MUESTRA, no elige nada): media del delta contra la base, min(g6, g8), por GT")
    for g in GTS4:
        land = [(min(deltas(f1, cand, t, "g6", mids, g).mean(), deltas(f1, cand, t, "g8", mids, g).mean()), t) for t, _ in GRIDS[cand]]
        P(f"  {g:<13}" + "  ".join(f"{t}:{v:+.4f}" for v, t in land))
    return {"chosen": chosen, "per_fold": per_fold, "verdict": v, "stats": st}


# ------------------------------------------------------------------------------------------------ main

def main4(f1, args):
    out = ["Ronda preinscrita 4 en PFF (8.24): 40 partidos de desarrollo, 5 pliegues de 8, delta contra la base B3 + absorcion tau 1,6, "
           "spellF1 en label / label_aerial / label_poss."]
    check_base(f1, out)
    sel = report_r4(f1, out)
    if not args.sin_semantica:
        report_semantics4(args.procesos, out, sel["chosen"])
    text = "\n".join(out)
    print(text)
    with open(REPORT[4], "w") as f:
        f.write(text + "\n")
    print(f"\n-> {REPORT[4]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ronda", type=int, choices=sorted(ROUNDS), default=2)
    ap.add_argument("--procesos", type=int, default=min(12, os.cpu_count() or 1))
    ap.add_argument("--rehacer", action="store_true", help="ignore the cache and recompute everything")
    ap.add_argument("--solo-resumen", action="store_true", help="no computation: report from the cache")
    ap.add_argument("--sin-semantica", action="store_true")
    args = ap.parse_args()
    cands = ROUNDS[args.ronda]
    if not args.solo_resumen:
        run_grid(args.procesos, args.rehacer, args.ronda)
    f1, cnt = load_cache(args.ronda)
    if args.ronda == 4:
        main4(f1, args)
        return
    out = [f"Ronda preinscrita {args.ronda} en PFF (8.20, 8.21, 8.22): 40 partidos de desarrollo, 5 pliegues de 8, delta contra B3, "
           f"spellF1 GT label."]
    check_base(f1, out)
    sel = {c: report_candidate(f1, cnt, c, out) for c in cands}
    entering = [c for c in cands if sel[c]["ok"]]
    out.append(f"\nEntran: {', '.join(entering) or 'ninguno'}")
    if args.ronda == 1 and len(entering) > 1:
        report_combination(entering, sel, args.procesos, out)
    elif args.ronda == 1 and entering:
        out.append("Entra uno solo: no hay combinacion ni solapamiento que medir.")
    if args.ronda in (2, 3):
        mids = list(pff_split.DESARROLLO)
        z = max(float(np.abs(deltas(f1, cands[0], "tau0", b, mids, g)).max()) for b in BALLS for g in GTS)
        out.append(f"Prueba de la configuracion nula: max |{cands[0]}(tau=0) - B3| en spellF1 sobre los 40 partidos, 4 balones y 2 GT = {z:.1e}")
    if not args.sin_semantica:
        c = cands[0]
        tau = {m: dict(GRIDS[c])[sel[c]["chosen"][m][0]][0] for m in sel[c]["chosen"]} if args.ronda in (2, 3) else None
        report_semantics(args.procesos, out, tau, GRIDS[c], c)
    text = "\n".join(out)
    print(text)
    with open(REPORT[args.ronda], "w") as f:
        f.write(text + "\n")
    print(f"\n-> {REPORT[args.ronda]}")


if __name__ == "__main__":
    main()
