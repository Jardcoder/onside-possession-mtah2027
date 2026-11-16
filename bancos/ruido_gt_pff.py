"""Label noise floor: spellF1 of B0 and B3 (ball g6) against the PFF `label` with every possession edge moved at random.

    challenge/.venv/bin/python bancos/ruido_gt_pff.py [--reps 5] [--max-shift 4] [--procesos P]

Every change of label between consecutive frames of the same period is an edge. Each edge is moved by an integer drawn
uniformly in [-max_shift, max_shift] (0 included), fixed seed, REPS repetitions. An edge never leaves the open interval
between the new position of the previous edge and the ORIGINAL position of the next one (so no edge crosses a
neighbour, a run keeps at least one frame) and the period changes are never moved. The X on the first frame of period 2 is
set afterwards, exactly as in evaluar_toques.make_gt. B0 and B3 are scored against the same shifted GT.
Development matches only (pff_split.load_dev), periods 1-2, raw tracking. `label` is read only to score (it never enters
a method: evaluar_pff.ladder sees ball, slots, period and pitch). Aggregates per match only.
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
import possession_metrics as pm  # noqa: E402

BALL = "g6"
STEPS = ("B0", "B3")
SEED = 20261029
REPS, MAX_SHIFT = 5, 4


def shift_edges(lab, period, rng, max_shift=MAX_SHIFT):
    """Copy of the (N,) label array with every edge (label change inside a period) shifted by U{-max_shift..max_shift}.
    Edge i stays in [new edge i-1 + 1, original edge i+1 - 1] (the last one up to the end of the period), so edges keep
    their order and no run disappears; period changes are fixed. Returns (labels, number of edges)."""
    lab = np.asarray(lab)
    out = lab.copy()
    cut = np.flatnonzero(np.diff(period) != 0) + 1
    edges_p = np.concatenate(([0], cut, [len(lab)]))
    n_edges = 0
    for lo, hi in zip(edges_p[:-1], edges_p[1:]):
        seg = lab[lo:hi]
        e = np.flatnonzero(seg[1:] != seg[:-1]) + 1                  # first frame of each new run, segment-relative
        n_edges += len(e)
        if not len(e):
            continue
        s = rng.integers(-max_shift, max_shift + 1, len(e))
        new = np.empty(len(e), int)
        for i in range(len(e)):
            low = new[i - 1] + 1 if i else 1
            high = e[i + 1] - 1 if i + 1 < len(e) else len(seg) - 1
            new[i] = min(max(e[i] + s[i], low), high)
        vals = np.concatenate(([seg[0]], seg[e]))
        starts = np.concatenate(([0], new, [len(seg)]))
        out[lo:hi] = np.repeat(vals, np.diff(starts))
    return out, n_edges


def work(mid, reps=REPS, max_shift=MAX_SHIFT):
    """(spellF1 of B0/B3 against the original GT and each repetition (reps+1, 2), edges, frames) for one match."""
    d = ep.clip_periods(pff_split.load_dev(mid))
    ball = dd.smooth_ball(d["ball_xy"], d["period"], BALL)
    preds = ep.ladder(ball, d["players_slots"], d["period"], d["pitch"])
    pcodes = {s: pm.label_codes(preds[s]) for s in STEPS}
    res = np.zeros((reps + 1, len(STEPS)))
    n_edges = 0
    for r in range(reps + 1):
        if r == 0:
            lab = d["label"]
        else:
            lab, n_edges = shift_edges(d["label"], d["period"], np.random.default_rng([SEED, r, int(mid)]), max_shift)
        gt = ev.make_gt({"label": lab, "period": d["period"]})
        for j, s in enumerate(STEPS):
            res[r, j] = pm.evaluate(gt, pcodes[s], bootstrap_n=0)["spell_f1_mean"]
    return mid, res, n_edges, len(d["period"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--max-shift", type=int, default=MAX_SHIFT)
    ap.add_argument("--procesos", type=int, default=min(12, os.cpu_count() or 1))
    args = ap.parse_args()
    mids = list(pff_split.DESARROLLO)
    assert not set(mids) & set(pff_split.RESERVADOS)
    with mp.get_context("fork").Pool(min(args.procesos, len(mids))) as pool:
        out = dict((m, (r, e, n)) for m, r, e, n in pool.imap_unordered(
            functools.partial(work, reps=args.reps, max_shift=args.max_shift), mids))
    res = np.stack([out[m][0] for m in mids])                              # (40, reps+1, 2)
    edges = np.array([out[m][1] for m in mids], float)
    print(f"GT label with every edge moved U{{-{args.max_shift}..{args.max_shift}}} frames, seed {SEED}, {args.reps} reps, "
          f"ball {BALL}, n={len(mids)}; edges per match {edges.mean():.0f} (min {edges.min():.0f}, max {edges.max():.0f})")
    print(f"{'GT':<14}{'B0':>9}{'B3':>9}{'B3-B0':>9}{'gana':>6}{'IC95 inf':>10}{'IC95 sup':>10}{'sd':>8}")
    rows = [("original", res[:, 0])] + [(f"ruido rep {r}", res[:, r]) for r in range(1, args.reps + 1)] \
        + [("ruido media", res[:, 1:].mean(axis=1))]
    for name, a in rows:
        s = ep.delta_stats(a[:, 1] - a[:, 0])
        print(f"{name:<14}{a[:, 0].mean():>9.4f}{a[:, 1].mean():>9.4f}{s['mean']:>+9.4f}{s['wins']:>4}/{s['n']}"
              f"{s['lo']:>+10.4f}{s['hi']:>+10.4f}{s['sd']:>8.4f}")
    base, noisy = res[:, 0], res[:, 1:].mean(axis=1)
    for j, s in enumerate(STEPS):
        drop = noisy[:, j] - base[:, j]
        sd = ep.delta_stats(drop)
        print(f"{s}: cae {-drop.mean():.4f} ({100 * -drop.mean() / base[:, j].mean():.1f} % de {base[:, j].mean():.4f}); "
              f"IC95 de la caida [{-sd['hi']:.4f}, {-sd['lo']:.4f}], cae en {sd['losses']}/{sd['n']} partidos")
    rep_means = res[:, 1:].mean(axis=0)                                    # (reps, 2)
    print("spread across repetitions (mean of 40 matches), min..max: "
          + ", ".join(f"{s} {rep_means[:, j].min():.4f}..{rep_means[:, j].max():.4f}" for j, s in enumerate(STEPS)))


if __name__ == "__main__":
    main()
