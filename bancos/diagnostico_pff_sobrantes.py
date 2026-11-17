"""Extra touches of v2 on the PFF bank: detected touches (B2, ball g6) that match no event touch, and what they do to B2.

    challenge/.venv/bin/python bancos/diagnostico_pff_sobrantes.py [modo] [--procesos P]       # default: g6

Per match (development, periods 1-2): the v2 touches of B2 (deflections with their nearest-player team + control touches,
the same arrays evaluar_pff.ladder builds) are matched one to one, by time only, with the PFF event touches (`is_touch`)
within evaluar_toques.TOL frames (evaluar_toques.match_one_to_one). Two event sets: ALL the event touches, and only
those on a frame with a visible ball (the set the DFL counts use, evaluar_toques.detector_counts); the detected touches
are always inside visible stretches. Then, for each UNMATCHED detected touch at frame f:
  cambia     the B2 prediction at f differs from the B1 prediction at f (the touch changes the owner the B1 rule had there)
  dentro     of those, the GT `label` is constant H or A over [f - TOL, f + TOL] (no GT edge close by: f is inside one
             GT possession)
  parte      of those, B2[f] differs from the GT label: the prediction leaves the GT owner in the middle of that possession;
             counted as UNIQUE GT possessions (maximal constant H / A run of the GT), not touches
  parte_toques  the same condition counted per touch (what `parte` counted up to round 1; several touches can split one possession)
`is_touch` and `label` are read here only to evaluate; no method sees them. Aggregates only.
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
import evaluar_pff as ep  # noqa: E402
import evaluar_toques as ev  # noqa: E402
import pff_split  # noqa: E402
import toques  # noqa: E402

FIELDS = ("n_ev", "n_det", "emparejados", "sobrantes", "cambia", "dentro", "parte", "parte_toques")


def v2_touches(ball, slots, per, pitch):
    """(frames, teams) of the B2 touches, built exactly as evaluar_pff.ladder builds them."""
    present = ~np.isnan(ball[:, 0])
    near = toques.prepare(ball, slots, pitch)
    runs = toques.visible_runs(present, per)
    cand = toques.deflection_candidates(near, runs, r=ep.V2["r"], dv=ep.V2["dv"], win=ep.V2["win"])
    ctrl = toques.control_touches(near, runs, r_ctrl=ep.V2["r_ctrl"], k=ep.V2["k"])
    return toques.merge_touches((cand["frame"], cand["team"]), ctrl)


def inside_gt_possession(gt, f, tol=ev.TOL):
    """True when the GT labels over [f - tol, f + tol] (clipped to the array) are one single H or A: no GT edge near f."""
    seg = gt[max(0, f - tol):f + tol + 1]
    return bool((seg == seg[0]).all() and seg[0] in ("H", "A"))


def extras(tf, event_f, b1, b2, gt, tol=ev.TOL):
    """Counts of one event set: matched / unmatched detected touches and, among the unmatched, cambia / dentro / parte."""
    pairs = ev.match_one_to_one(tf, np.zeros(len(tf), np.int8), event_f, np.zeros(len(event_f), np.int8), tol=tol)
    matched = {i for i, _ in pairs}
    out = dict.fromkeys(FIELDS, 0)
    gt = np.asarray(gt)
    poss_id = np.concatenate(([0], np.cumsum(gt[1:] != gt[:-1])))       # id of the constant run each GT frame belongs to
    split = set()
    out.update(n_ev=len(event_f), n_det=len(tf), emparejados=len(pairs), sobrantes=len(tf) - len(pairs))
    for i, f in enumerate(tf):
        if i in matched or b2[f] == b1[f]:
            continue
        out["cambia"] += 1
        if inside_gt_possession(gt, int(f), tol):
            out["dentro"] += 1
            if b2[f] != gt[f]:
                out["parte_toques"] += 1
                split.add(int(poss_id[f]))
    out["parte"] = len(split)
    return out


def work(mid, mode="g6"):
    raw = pff_split.load_dev(mid)
    keep = np.asarray(raw["period"]) <= ep.LAST_PERIOD
    d = ep.clip_periods(raw)
    ball = d["ball_xy"] if mode == "crudo" else dd.smooth_ball(d["ball_xy"], d["period"], mode)
    preds = ep.ladder(ball, d["players_slots"], d["period"], d["pitch"])
    tf, _ = v2_touches(ball, d["players_slots"], d["period"], d["pitch"])
    event_f = np.flatnonzero(np.asarray(raw["is_touch"])[keep])
    present = ~np.isnan(ball[:, 0])
    return mid, {"todos": extras(tf, event_f, preds["B1"], preds["B2"], d["label"]),
                 "visible": extras(tf, event_f[present[event_f]], preds["B1"], preds["B2"], d["label"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modo", nargs="?", default="g6", choices=["crudo"] + sorted(dd.BALL_SMOOTH))
    ap.add_argument("--procesos", type=int, default=min(12, os.cpu_count() or 1))
    args = ap.parse_args()
    mids = list(pff_split.DESARROLLO)
    assert not set(mids) & set(pff_split.RESERVADOS)
    with mp.get_context("fork").Pool(min(args.procesos, len(mids))) as pool:
        res = dict(pool.imap_unordered(functools.partial(work, mode=args.modo), mids))
    print(f"PFF dev, toques v2 de B2 (balon {args.modo}), n={len(mids)} partidos, TOL={ev.TOL} frames; por partido: media (min..max) | total")
    for conj, nombre in (("todos", "eventos: todos"), ("visible", "eventos: solo con balon visible")):
        a = np.array([[res[m][conj][k] for k in FIELDS] for m in mids], float)
        print(f"\n{nombre}")
        for j, k in enumerate(FIELDS):
            print(f"  {k:<12}{a[:, j].mean():>8.1f}  ({a[:, j].min():.0f}..{a[:, j].max():.0f})  | {a[:, j].sum():.0f}")
        t = a.sum(0)
        print(f"  sobrantes / detectados {t[3] / t[1]:.3f}; emparejados / eventos {t[2] / t[0]:.3f}; detectados / eventos {t[1] / t[0]:.2f}; "
              f"cambia / sobrantes {t[4] / t[3]:.3f}; dentro / cambia {t[5] / t[4]:.3f}; parte / cambia {t[6] / t[4]:.3f}")


if __name__ == "__main__":
    main()
