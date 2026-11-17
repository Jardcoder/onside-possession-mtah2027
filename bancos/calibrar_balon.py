"""Ball position noise: challenge ATD (cleaned) against the degraded DFL bank, aggregates only.

    challenge/.venv/bin/python bancos/calibrar_balon.py

Metrics are taken inside continuous visible stretches, trimming a margin at each end of every stretch: the
measurement window (MEAS frames: SG / moving-average / 3-frame velocity windows must not touch a border) plus,
for a smoothed bank ball, the support of the filter (it is edge-padded at the stretch borders, so the first
and last frames inside its support are contaminated: Gaussian ceil(3 sigma) both sides, causal filter only on
the leading side, see margins()). Stretches shorter than the two margins plus MIN_BODY frames are dropped.
The reto is also measured with the margins of each smoothed mode so the stretch populations match.
Metrics:
  speed p50/p90/p99 (m/s), |second difference| p50/p90/p99 (cm), std of the residual against a
  Savitzky-Golay filter (9 frames, order 2) and against a 5-frame moving average (cm), lag-1
  autocorrelation of the SG residual, sharp direction changes per visible minute (angle > 30 deg
  between the velocities of the previous and next 3 frames, both > 2 m/s; consecutive frames count once).
Only these aggregates are printed or returned; no per-frame values leave this module.

The bank stretches are cut by period, like the challenge ones (which come from the two live segments), so no
stretch spans the half-time break.

    challenge/.venv/bin/python bancos/calibrar_balon.py                  # reto, bank, bank smoothed (g6, g8, causal)
    challenge/.venv/bin/python bancos/calibrar_balon.py --barrido-causal # sweep of causal filters against the reto
    challenge/.venv/bin/python bancos/calibrar_balon.py --pff            # reto vs PFF (40 development matches) raw / smoothed

--pff: PFF ball (pff_split.load_dev, periods 1-2 only, stretches cut by period) raw and smoothed with the centred
Gaussians g2..g10 and the causal filter. Every mode and the reto are measured over the SAME stretches with ONE common
margin (common_margin(): the elementwise maximum of the margins of all the modes, i.e. the most demanding one), so the
populations match. Summary distance = mean of a FIXED set of 10 terms (errores()): |log(bank / reto)| for the nine
positive metrics (floor FLOOR for zeros, on both sides) and |bank - reto| for lag1; every mode is compared with the same
number of terms. Aggregates only. --margen-por-modo reproduces the older population (each mode with its own margin and
its own reto row) with the same fixed metric set, only to show how much the margin matters.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import dfl  # noqa: E402
import dfl_degradado as dd  # noqa: E402

FPS = 25
MEAS = 6                     # measurement window (needs 4: SG 9 -> 4, MA 5 -> 2, velocity windows 3); 6 kept from the first version
MIN_BODY = 10                # frames that must remain inside a stretch after trimming both margins
SG_WIN, SG_ORDER, MA_WIN = 9, 2, 5
ANGLE_DEG, ANGLE_VMIN, ANGLE_WIN = 30.0, 2.0, 3


def margins(mode=None):
    """(left, right) frames trimmed at each end of a visible stretch for a bank ball smoothed with `mode`
    (None, a dfl_degradado.BALL_SMOOTH key or a (kind, param) pair): measurement window + filter support."""
    if mode is None:
        return MEAS, MEAS
    kind, param = dd.BALL_SMOOTH[mode] if isinstance(mode, str) else mode
    k, _ = dd._kernel(kind, param)
    if kind == "gauss":
        h = len(k) // 2
        return MEAS + h, MEAS + h
    return MEAS + len(k) - 1, MEAS                  # causal: output at t uses frames <= t


def sg_kernel(win=SG_WIN, order=SG_ORDER):
    """Savitzky-Golay smoothing weights (centre value of the least-squares polynomial fit)."""
    h = win // 2
    x = np.arange(-h, h + 1)
    V = np.vander(x, order + 1, increasing=True)
    return np.linalg.pinv(V)[0]


def visible_runs(present, segments=None, margin=(MEAS, MEAS)):
    """[(start, stop)] maximal runs of visible ball long enough to keep MIN_BODY frames after trimming `margin`
    (left, right), inside `segments`."""
    min_run = sum(margin) + MIN_BODY
    segments = segments or [(0, len(present))]
    out = []
    for lo, hi in segments:
        m = present[lo:hi].astype(np.int8)
        d = np.diff(np.concatenate(([0], m, [0])))
        for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            if e - s >= min_run:
                out.append((lo + int(s), lo + int(e)))
    return out


def measure(ball_m, runs, margin=(MEAS, MEAS)):
    """Aggregates of ball_m (N,2) in metres over the interior of the given visible runs (trimmed by `margin`)."""
    k = sg_kernel()
    sp, a2, res_sg, res_ma, ang_events = [], [], [], [], 0
    res_pairs = [[], []]      # lag-1 pairs of the SG residual (x, y pooled)
    n_frames = 0
    for s, e in runs:
        p = ball_m[s:e]
        L = len(p)
        i0, i1 = margin[0], L - margin[1]
        n_frames += i1 - i0
        v = np.linalg.norm(p[1:] - p[:-1], axis=1) * FPS            # v[i] = |p[i+1]-p[i]|
        sp.append(v[i0:i1])
        d2 = p[2:] - 2 * p[1:-1] + p[:-2]                           # centred at i = 1..L-2
        a2.append(np.linalg.norm(d2, axis=1)[i0 - 1:i1 - 1] * 100)
        sg = np.stack([np.convolve(p[:, c], k[::-1], "valid") for c in (0, 1)], 1)   # centred at 4..L-5
        r = p[4:L - 4] - sg
        res_sg.append(r[i0 - 4:i1 - 4] * 100)
        ma = np.stack([np.convolve(p[:, c], np.ones(MA_WIN) / MA_WIN, "valid") for c in (0, 1)], 1)  # centred at 2..L-3
        res_ma.append((p[2:L - 2] - ma)[i0 - 2:i1 - 2] * 100)
        rr = r[i0 - 4:i1 - 4]
        for c in (0, 1):
            res_pairs[0].append(rr[:-1, c])
            res_pairs[1].append(rr[1:, c])
        w = ANGLE_WIN
        vp = (p[i0:i1] - p[i0 - w:i1 - w]) * FPS / w
        vn = (p[i0 + w:i1 + w] - p[i0:i1]) * FPS / w
        np_, nn = np.linalg.norm(vp, axis=1), np.linalg.norm(vn, axis=1)
        ok = (np_ > ANGLE_VMIN) & (nn > ANGLE_VMIN)
        cosang = np.where(ok, (vp * vn).sum(1) / np.where(ok, np_ * nn, 1), 1.0)
        flag = ok & (np.degrees(np.arccos(np.clip(cosang, -1, 1))) > ANGLE_DEG)
        ang_events += int((np.diff(np.concatenate(([0], flag.astype(np.int8)))) == 1).sum())
    sp, a2 = np.concatenate(sp), np.concatenate(a2)
    res_sg, res_ma = np.concatenate(res_sg), np.concatenate(res_ma)
    x, y = np.concatenate(res_pairs[0]), np.concatenate(res_pairs[1])
    minutes = n_frames / FPS / 60
    return {
        "n_runs": len(runs), "visible_min": minutes,
        "speed_p50_p90_p99": np.percentile(sp, [50, 90, 99]),
        "abs_a2_cm_p50_p90_p99": np.percentile(a2, [50, 90, 99]),
        "res_sg_std_cm": float(res_sg.std(axis=0).mean()),
        "res_ma_std_cm": float(res_ma.std(axis=0).mean()),
        "res_sg_lag1": float((x * y).mean() / np.sqrt((x ** 2).mean() * (y ** 2).mean())),
        "sharp_turns_per_min": ang_events / minutes,
    }


def reto(margin=(MEAS, MEAS)):
    import atd
    import limpieza
    d = atd.load_tactical()
    live = limpieza.live_mask()
    clean = limpieza.clean_atd(d, mask=live)
    segs = [(11185, 90652), (110808, 190198)]
    b = clean["ball_xy"] * np.array(atd.PITCH)
    return measure(b, visible_runs(clean["ball_present"], segs, margin), margin)


def period_segments(period):
    """[(lo, hi)] one per period (the half-time break is a change of period)."""
    cut = np.flatnonzero(np.diff(period) != 0) + 1
    edges = np.concatenate(([0], cut, [len(period)]))
    return list(zip(edges[:-1].tolist(), edges[1:].tolist()))


def banco(seed=0, solo_vivo=False, ball_smooth=None):
    """{match: (ball_m, visible runs)} of the pd0.60 bank for the development matches, runs cut by period and
    long enough for margins(ball_smooth).
    ball_smooth: None or a dfl_degradado.BALL_SMOOTH key / (kind, param) pair."""
    out, margin = {}, margins(ball_smooth)
    for mid in dfl.DESARROLLO:
        d = dd.load_degraded(mid, seed, "coupled", p_dead_given_absent=0.60)
        ball = d["ball_xy"] if ball_smooth is None else dd.smooth_ball(d["ball_xy"], d["period"], ball_smooth)
        pres = d["ball_present"] & ((d["ball_status"] == 1) if solo_vivo else True)
        out[mid] = (ball * np.array(dd.ATD_PITCH), visible_runs(pres, period_segments(d["period"]), margin))
    return out


PFF_MODES = ("g2", "g3", "g4", "g6", "g8", "g10", "causal")


def pff_data():
    """{match: (ball_xy normalised, period, pitch (L, W))} of the 40 PFF development matches, periods 1-2 only
    (extra time out, as in evaluar_pff.clip_periods). Loaded through the development guard, raw tracking."""
    import pff_split
    out = {}
    for mid in pff_split.DESARROLLO:
        d = pff_split.load_dev(mid)
        m = np.asarray(d["period"]) <= 2
        out[mid] = (d["ball_xy"][m], d["period"][m], d["pitch"])
    return out


def banco_pff(data, ball_smooth=None, margin=None):
    """Same shape as banco() for the PFF matches; visible = ball not NaN, runs cut by period. `margin` overrides the
    margins of the mode (a common one for every mode, see common_margin)."""
    out = {}
    margin = margin or margins(ball_smooth)
    for mid, (ball, period, pitch) in data.items():
        b = ball if ball_smooth is None else dd.smooth_ball(ball, period, ball_smooth)
        out[mid] = (b * np.array(pitch), visible_runs(~np.isnan(ball[:, 0]), period_segments(period), margin))
    return out


LOG_KEYS = ("speed_p50_p90_p99", "abs_a2_cm_p50_p90_p99", "res_sg_std_cm", "res_ma_std_cm", "sharp_turns_per_min")
DIFF_KEYS = ("res_sg_lag1",)
KEYS = LOG_KEYS + DIFF_KEYS
TERMS = ("v50", "v90", "v99", "a50", "a90", "a99", "sgStd", "maStd", "turns", "lag1")      # order of errores()
FLOOR = 0.01                 # explicit floor for a zero (or negative) positive-type metric, same unit as the metric


def values(m):
    """The 10 metrics of a measure() result in TERMS order (not transformed)."""
    return np.concatenate([np.atleast_1d(m[k]) for k in LOG_KEYS + DIFF_KEYS]).astype(float)


def errores(m, ref):
    """(10,) signed error of the fixed metric set in TERMS order: log(max(bank, FLOOR) / max(reto, FLOOR)) for the nine
    positive metrics and bank - reto for lag1 (which can be negative, so it has no ratio). Never drops a term."""
    a, b = values(m), values(ref)
    out = np.empty(len(a))
    out[:-1] = np.log(np.maximum(a[:-1], FLOOR) / np.maximum(b[:-1], FLOOR))
    out[-1] = a[-1] - b[-1]
    return out


def distancia(m, ref):
    """Mean of |errores(m, ref)| over the fixed 10-term set."""
    return float(np.mean(np.abs(errores(m, ref))))


def common_margin(modes=PFF_MODES):
    """(left, right): elementwise maximum of the margins of the raw ball and of every mode, so that no mode is contaminated
    by its filter support and all of them (and the reto) see exactly the same frames of the same stretches."""
    ms = [margins(None)] + [margins(m) for m in modes]
    return max(m[0] for m in ms), max(m[1] for m in ms)


def main_pff(por_modo=False):
    """Table of the PFF ball modes against the reto. Default: one common margin for everything (same stretches).
    por_modo: each mode with its own margin and its own reto row (older population), same fixed metric set."""
    data = pff_data()
    cm = common_margin()
    rows, refs = {}, {}
    if por_modo:
        refs["crudo"] = reto()
        rows["crudo"] = pooled(banco_pff(data))
        for mode in PFF_MODES:
            mg = margins(mode)
            refs[mode], rows[mode] = reto(mg), pooled(banco_pff(data, mode), mg)
        marg = {"crudo": margins(), **{m: margins(m) for m in PFF_MODES}}
    else:
        ref = reto(cm)
        for mode in (None,) + PFF_MODES:
            rows[mode or "crudo"] = pooled(banco_pff(data, mode, cm), cm)
        refs = {k: ref for k in rows}
        marg = {k: cm for k in rows}
    print(("population: each mode with its own margin" if por_modo else "population: common margin {} for all modes and the reto".format(cm))
          + "; distance = mean of {} fixed terms ({}), floor {}".format(len(TERMS), " ".join(TERMS), FLOOR))
    print(HEADER + "{:>9}".format("dist"))
    r0 = refs["crudo"]
    print(fila("reto (ATD limpio)", r0), " [{} tramos, {:.0f} min, margen {}]".format(r0["n_runs"], r0["visible_min"], marg["crudo"]))
    for k, m in rows.items():
        print(fila("PFF " + k, m), "{:>9.3f}".format(distancia(m, refs[k])),
              " [{} tramos, {:.0f} min, margen {}]".format(m["n_runs"], m["visible_min"], marg[k]))
        if por_modo and k != "crudo":
            print(fila("  reto, mismo margen", refs[k]))
    print("\nerror by term (log ratio bank/reto, lag1 difference), reto of each row's own population")
    print("{:<12}".format("") + "".join("{:>8}".format(t) for t in TERMS))
    for k, m in rows.items():
        print("{:<12}".format("PFF " + k) + "".join("{:>+8.2f}".format(e) for e in errores(m, refs[k])))
    rank = sorted((distancia(m, refs[k]), k) for k, m in rows.items())
    print("ranking (closest to the reto):", ", ".join("{} {:.3f}".format(k, d) for d, k in rank))
    if not por_modo:
        g6, g8, rv = values(rows["g6"]), values(rows["g8"]), values(refs["g6"])
        inside = [bool(min(a, b) <= r <= max(a, b)) for a, b, r in zip(g6, g8, rv)]
        print("\nreto between g6 and g8, term by term (value g6 / reto / value g8):")
        for t, a, r, b, ok in zip(TERMS, g6, rv, g8, inside):
            print("  {:<6}{:>9.3f}{:>9.3f}{:>9.3f}   {}".format(t, a, r, b, "between" if ok else "OUTSIDE"))
        print("between in {}/{} terms".format(sum(inside), len(inside)))


def pooled(parts, margin=(MEAS, MEAS)):
    """Measure over all matches at once: concatenate with a gap so no run crosses matches."""
    balls, runs, off = [], [], 0
    for b, r in parts.values():
        balls.append(b)
        runs += [(s + off, e + off) for s, e in r]
        off += len(b)
    return measure(np.concatenate(balls), runs, margin)


def fila(name, m):
    s, a = m["speed_p50_p90_p99"], m["abs_a2_cm_p50_p90_p99"]
    return ("{:<26}{:>6.2f}{:>6.2f}{:>7.2f}{:>7.2f}{:>7.2f}{:>8.2f}{:>8.2f}{:>8.2f}{:>7.3f}{:>8.2f}".format(
        name, s[0], s[1], s[2], a[0], a[1], a[2], m["res_sg_std_cm"], m["res_ma_std_cm"],
        m["res_sg_lag1"], m["sharp_turns_per_min"]))


HEADER = "{:<26}{:>6}{:>6}{:>7}{:>7}{:>7}{:>8}{:>8}{:>8}{:>7}{:>8}".format(
    "", "v50", "v90", "v99", "a50cm", "a90cm", "a99cm", "sgStd", "maStd", "lag1", "turns/m")


# Causal filters tried against the challenge ball: (kind, parameter in frames), see dfl_degradado._kernel.
CAUSAL_SWEEP = ([("causal_halfgauss", s) for s in (6, 8, 10, 12, 14, 15, 16, 20)]
                + [("causal_ema", s) for s in (4, 6, 8, 10, 12, 16)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--barrido-causal", action="store_true")
    ap.add_argument("--pff", action="store_true")
    ap.add_argument("--margen-por-modo", action="store_true", help="with --pff: each mode with its own margin (older population)")
    args = ap.parse_args()
    if args.pff:
        return main_pff(args.margen_por_modo)
    print(HEADER)
    r = reto()
    print(fila("reto (ATD limpio)", r), " [{} tramos, {:.0f} min visibles, margen {}]".format(
        r["n_runs"], r["visible_min"], margins()))
    if args.barrido_causal:
        print("delay = mean of the kernel in time (frames, ms)")
        for spec in CAUSAL_SWEEP:
            dl = dd.kernel_delay_frames(spec)
            print(fila("{} {:g} d={:.1f}f {:.0f}ms".format(spec[0][7:], spec[1], dl, dl * 40),
                       pooled(banco(ball_smooth=spec, solo_vivo=True), margins(spec))))
        return
    bk = banco()
    print(fila("banco pd0.60 s0 (todo)", pooled(bk)))
    print(fila("banco pd0.60 s0 (vivo)", pooled(banco(solo_vivo=True))))
    for mid, (b, rn) in bk.items():
        print(fila("  " + mid, measure(b, rn)))
    for mode in ("g6", "g8", "causal"):
        mg = margins(mode)
        print(fila("banco " + mode + " (vivo)", pooled(banco(ball_smooth=mode, solo_vivo=True), mg)),
              " delay {:.1f} frames = {:.0f} ms, margen {}".format(dd.kernel_delay_frames(mode), dd.kernel_delay_frames(mode) * 40, mg))
        r = reto(mg)
        print(fila("  reto, mismo margen", r), " [{} tramos, {:.0f} min visibles]".format(r["n_runs"], r["visible_min"]))


if __name__ == "__main__":
    main()
