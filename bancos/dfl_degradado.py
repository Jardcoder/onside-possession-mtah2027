"""DFL degraded to look like the ATD, then cleaned with src/limpieza.py.

    challenge/.venv/bin/python bancos/dfl_degradado.py --calibrar   # ATD aggregates -> data/cache/*.json
    challenge/.venv/bin/python bancos/dfl_degradado.py --calibrar-largo   # P(dead | absent) by gap length -> huecos_largo_atd.json
    challenge/.venv/bin/python bancos/dfl_degradado.py [mid ...]    # build + cache degraded matches
    from dfl_degradado import load_degraded
    d = load_degraded("J03WPY", seed=0)     # same keys as the cleaned ATD (src/limpieza.clean_atd)

Calibration (run once) reads the ATD and writes ONLY aggregates (histograms, rates, transition
counts) to bancos/data/cache/huecos_atd.json and calibracion_atd.json. Degradation steps:

  a. Ball height z is dropped (never returned).
  b. Ball holes. In the ATD the ball is absent in 40.7 % of the live frames, in a few long runs (median
     ~15 s, almost none < 5 s), and the absence is strongly tied to dead-ball stretches: mean player
     speed while the ball is absent is 1.14 m/s against 2.08 m/s while present, the same split as DFL
     BallStatus dead (1.10) against alive (2.3); gaps start more often near the pitch edge (32 % within
     3 m against 12 %) and after a slow ball. No dependence on player density near the last position
     (3.26 players within 10 m at the start of a gap against 3.26 at random). gap_mode="coupled"
     (default) reproduces the dependence: each DFL dead-ball run is hidden with probability
     P(absent | dead) (set so the total stays 40.7 %), in-play gaps are added with the ATD run-length
     histogram and short visible windows are opened inside long gaps to match the ATD run count.
     gap_mode="indep"
     draws alternating absent/present runs from the ATD histograms with no link to the game state.
     gap_mode="coupled_len" is "coupled" with one P(dead | absent) per gap-length class (< 5, 5-10, 10-20, >= 20 s;
     `calibrar_por_largo`, huecos_largo_atd.json, player speed over 2 s inside each gap): the global P is the
     frame-weighted mean of the classes and the in-play gaps are drawn from the class run-length histograms
     weighted by their share of live gaps, so 5-20 s gaps in live play appear at the calibrated rate.
  c. Players: the ATD |second difference| is ~1 mm (smoothed tracker), below DFL's 1 cm quantisation
     noise, so no jitter is added (calibrated amount = 0). What the ATD does have is a per-frame count
     of present slots different from 11 (missing and spurious slots), tracklet fragmentation and ID
     swaps. Counts follow a Markov chain per team (transitions measured in the ATD, one matrix per ball
     state, present/absent); missing players are dropped at random (the ATD ones are only 8 % closer to
     the ball) and come back in a NEW slot; spurious slots copy an opposing player + N(0, 1 m)
     (the ATD slots beyond 11 are not duplicates of a teammate); ID swaps between two teammates are
     injected at the ATD rate; blocks of displaced ball frames at the rate and lengths limpieza removes in the ATD.
  d. The result goes through limpieza.clean_tracking.
"""
import argparse
import hashlib
import json
import sys
from bisect import bisect_right
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
import onside_paths  # noqa: E402
import dfl  # noqa: E402
import limpieza  # noqa: E402

CACHE = Path(onside_paths.BANKS_DIR) / "cache"
CAL_GAPS = CACHE / "huecos_atd.json"
CAL_ATD = CACHE / "calibracion_atd.json"
CAL_LARGO = CACHE / "huecos_largo_atd.json"
ATD_SEGS = [(11185, 90652), (110808, 190198)]       # live halves of the ATD (frames)
FPS = 25
ATD_PITCH = (105.3, 68.0)
N_COUNT = 15                    # counts 0..14 (14 = 14 or more)
E_MAX = 4                       # spurious slots per team at once
SLOTS = {"H": 24, "A": 22}      # same as the ATD
EDGES = np.unique(np.round(np.geomspace(1, 30000, 31)).astype(int))   # run-length bins (frames)
LEN_CUTS = (5 * FPS, 10 * FPS, 20 * FPS)            # gap-length classes (frames): < 5 s, 5-10, 10-20, >= 20 (np.digitize)
LEN_NAMES = ("<5 s", "5-10 s", "10-20 s", ">=20 s")
LEN_WINDOWS = {"win2s": 2 * FPS, "win5s": 5 * FPS}  # speed smoothing windows of calibrar_por_largo; the first drives coupled_len
LEN_WIN = "win2s"
GAP_EDGE = FPS                                      # frames dropped at each border of a gap (the ball is visible next to it).
# 25 frames purify the 2 s window (LEN_WIN, 50 frames: +-25 around a frame stay inside the gap) but NOT the 5 s one (125
# frames), which is only printed as a comparison.
SPEED_BINS = np.linspace(0, 5, 51)


# ---------------------------------------------------------------- shared measurements
def run_lengths(mask, segments):
    """Lengths (frames) of the True runs of `mask` inside each (lo, hi) segment."""
    out = []
    for lo, hi in segments:
        m = mask[lo:hi].astype(int)
        d = np.diff(np.concatenate([[0], m, [0]]))
        out.append(np.flatnonzero(d == -1) - np.flatnonzero(d == 1))
    return np.concatenate(out) if out else np.array([], int)


def hist_edges(lengths):
    e = np.append(EDGES, np.iinfo(np.int64).max)
    return np.histogram(lengths, bins=e)[0].tolist()


def smoothed_player_speed(slots_m, win=5 * FPS):
    """(N,) mean step speed (m/s) over present players, steps > 12 m/s ignored, 5 s moving average."""
    d = np.linalg.norm(slots_m[1:] - slots_m[:-1], axis=2) * FPS
    d = np.where(d > 12, np.nan, d)
    ms = np.full(len(slots_m), np.nan)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ms[1:] = np.nanmean(d, axis=1)
    ms = np.where(np.isnan(ms), np.nanmedian(ms), ms)
    return np.convolve(ms, np.ones(win) / win, "same")


def accel_stats(slots_m, mask=None, k=1, vmax=12):
    """|second difference| (cm) percentiles and robust white-noise sigma (cm) of player tracks."""
    p = ~np.isnan(slots_m[:, :, 0])
    d1 = np.linalg.norm(slots_m[k:] - slots_m[:-k], axis=2) * FPS / k
    a = slots_m[2 * k:] - 2 * slots_m[k:-k] + slots_m[:-2 * k]
    ok = p[2 * k:] & p[k:-k] & p[:-2 * k] & (d1[k:] < vmax) & (d1[:-k] < vmax)
    if mask is not None:
        ok &= mask[k:-k, None]
    mag = np.linalg.norm(a, axis=2)[ok] * 100
    comp = a[ok].ravel() * 100
    sigma_a2 = np.median(np.abs(comp - np.median(comp))) / 0.6745
    return {"abs_a2_cm_p50_p90_p99": np.percentile(mag, [50, 90, 99]).round(2).tolist(),
            "sigma_white_cm": float(sigma_a2 / np.sqrt(6)), "n": int(len(mag))}


def speed_pct(slots_m, mask=None, vmax=14):
    p = ~np.isnan(slots_m[:, :, 0])
    d = np.linalg.norm(slots_m[1:] - slots_m[:-1], axis=2) * FPS
    ok = p[1:] & p[:-1] & (d < vmax)
    if mask is not None:
        ok &= mask[1:, None]
    return np.percentile(d[ok], [50, 90, 99]).round(2).tolist()


# ---------------------------------------------------------------- DFL arrays
def dfl_arrays(mid):
    """DFL match in the ATD frame: metres, origin top-left, ATD pitch size. Returns dict with
    df, ball (N,2), teams {"H": (N,C,2), "A": (N,C,2)} (NaN when off the pitch), dead (N,) bool."""
    from lxml import etree
    df = dfl.load_match(mid)
    env = etree.parse(str(dfl._path("meta", mid))).getroot().find(".//Environment")
    px, py = float(env.get("PitchX")), float(env.get("PitchY"))
    sx, sy = ATD_PITCH[0] / px, ATD_PITCH[1] / py

    def conv(x, y):
        return np.stack([(x.astype(float) + px / 2) * sx, (y.astype(float) + py / 2) * sy], -1)

    teams = {}
    for side, key in (("h", "H"), ("a", "A")):
        cols = sorted({c[:-2] for c in df.columns if c[0] == side and c[1:2].isdigit()})
        teams[key] = np.stack([conv(df[c + "_x"].values, df[c + "_y"].values) for c in cols], 1)
    return {"df": df, "ball": conv(df["ball_x"].values, df["ball_y"].values), "teams": teams,
            "dead": (df["ball_status"].values == 0), "pitch": (px, py)}


# ---------------------------------------------------------------- calibration from the ATD
def mix_dead(g, f_dead, f_alive):
    """Share x in [0, 1] (step 0.005) of the DFL dead component in the speed histogram g (SPEED_BINS counts):
    least squares of g / sum(g) against x f_dead + (1 - x) f_alive."""
    g = g / g.sum()
    grid = np.linspace(0, 1, 201)
    err = [((g - (x * f_dead + (1 - x) * f_alive)) ** 2).sum() for x in grid]
    return float(grid[int(np.argmin(err))])


def calibrar():
    import atd
    d = atd.load_tactical()
    live = limpieza.live_mask()
    segs = ATD_SEGS
    bp = d["ball_present"]
    P = np.array(atd.PITCH)
    N_live = int(live.sum())

    # ---- ball holes
    absent_runs = run_lengths(~bp, segs)
    present_runs = run_lengths(bp, segs)
    clean = limpieza.clean_atd(d, mask=live)
    slots_m = np.concatenate([clean["players_slots"]["H"], clean["players_slots"]["A"]], axis=1) * P
    spd = smoothed_player_speed(slots_m)

    # DFL components of the dead/alive speed signature
    bins = np.linspace(0, 5, 51)
    dead_h, alive_h = np.zeros(50), np.zeros(50)
    dead_share = []
    for mid in dfl.MATCHES:
        a = dfl_arrays(mid)
        s = smoothed_player_speed(np.concatenate([a["teams"]["H"], a["teams"]["A"]], axis=1))
        dead_h += np.histogram(np.clip(s[a["dead"]], 0, 4.999), bins)[0]
        alive_h += np.histogram(np.clip(s[~a["dead"]], 0, 4.999), bins)[0]
        dead_share.append(float(a["dead"].mean()))
    f_dead, f_alive = dead_h / dead_h.sum(), alive_h / alive_h.sum()

    def mix(g):
        return mix_dead(g, f_dead, f_alive)

    g_abs = np.histogram(np.clip(spd[live & ~bp], 0, 4.999), bins)[0].astype(float)
    g_pre = np.histogram(np.clip(spd[live & bp], 0, 4.999), bins)[0].astype(float)
    a_dead_given_absent, b_dead_given_present = mix(g_abs), mix(g_pre)
    p_abs = float((~bp[live]).mean())
    dead_atd = a_dead_given_absent * p_abs + b_dead_given_present * (1 - p_abs)
    p_abs_dead = a_dead_given_absent * p_abs / dead_atd
    p_abs_alive = (1 - a_dead_given_absent) * p_abs / (1 - dead_atd)

    # edge / speed / density at the start of the gaps
    b = d["ball_xy"] * P
    starts = np.array([s for lo, hi in segs for s in (np.flatnonzero(np.diff(np.concatenate([[1], bp[lo:hi]]).astype(int)) == -1) + lo) if s > lo])
    prev = b[starts - 1]
    rnd = np.flatnonzero(live & bp)

    def edge(q):
        return np.minimum.reduce([q[:, 0], P[0] - q[:, 0], q[:, 1], P[1] - q[:, 1]])

    allp = np.concatenate([d["players_slots"]["H"], d["players_slots"]["A"]], axis=1) * P

    def dens(fr, pos):
        return float((np.linalg.norm(allp[fr] - pos[:, None, :], axis=2) < 10).sum(1).mean())

    sp = np.linalg.norm(b[1:] - b[:-1], axis=1) * FPS
    gaps = {
        "note": "aggregates only (ATD live frames = kickoff..final whistle of both halves)",
        "live_frames": N_live,
        "ball_absent_frac_live": p_abs,
        "run_edges_frames": EDGES.tolist(),
        "absent_run_hist": hist_edges(absent_runs),
        "present_run_hist": hist_edges(present_runs),
        "n_absent_runs": int(len(absent_runs)),
        "absent_run_pctl_frames_10_50_90": np.percentile(absent_runs, [10, 50, 90]).round(0).tolist(),
        "dependence": {
            "mean_player_speed_ms_absent": float(spd[live & ~bp].mean()),
            "mean_player_speed_ms_present": float(spd[live & bp].mean()),
            "dfl_mean_speed_dead": float((f_dead * (bins[:-1] + 0.05)).sum()),
            "dfl_mean_speed_alive": float((f_alive * (bins[:-1] + 0.05)).sum()),
            "p_dead_given_absent": a_dead_given_absent,
            "p_dead_given_present": b_dead_given_present,
            "atd_dead_share_estimated": float(dead_atd),
            "dfl_dead_share_range": [min(dead_share), max(dead_share)],
            "p_absent_given_dead": float(p_abs_dead),
            "p_absent_given_alive": float(p_abs_alive),
            "gap_start_within_3m_of_edge": float((edge(prev) < 3).mean()),
            "random_present_within_3m_of_edge": float((edge(b[rnd]) < 3).mean()),
            "players_within_10m_gap_start": dens(starts - 1, prev),
            "players_within_10m_random": dens(rnd, b[rnd]),
            "ball_speed_median_before_gap": float(np.nanmedian(sp[starts - 2])),
            "ball_speed_median_overall_live": float(np.nanmedian(sp[live[1:] & ~np.isnan(sp)])),
        },
    }
    CAL_GAPS.write_text(json.dumps(gaps, indent=1))

    # ---- player counts (raw ATD, before cleaning): chain per team and ball state
    chains = {}
    for team in ("H", "A"):
        c = np.minimum(d["n_present"][team], N_COUNT - 1)
        T = np.zeros((2, N_COUNT, N_COUNT))
        for lo, hi in segs:
            cc, st = c[lo:hi], bp[lo:hi].astype(int)
            np.add.at(T, (st[1:], cc[:-1], cc[1:]), 1)
        # smoothing: one pseudo-count spread over the marginal count distribution of that ball state, so no
        # state is absorbing and rare states (few observed transitions) can still be left
        for b_ in range(2):
            marg = T[b_].sum(0) + 1e-9
            T[b_] += marg / marg.sum()
        chains[team] = T.tolist()
    count_hist = {t: (np.bincount(np.minimum(d["n_present"][t][live], N_COUNT - 1), minlength=N_COUNT) / N_live).round(5).tolist()
                  for t in ("H", "A")}

    # ---- jitter: ATD vs DFL (white-noise sigma from the second difference at 1 frame)
    atd_j = accel_stats(slots_m[:, :, :], live)
    dfl_sig = []
    for mid in dfl.MATCHES:
        a = dfl_arrays(mid)
        dfl_sig.append(accel_stats(np.concatenate([a["teams"]["H"], a["teams"]["A"]], axis=1))["sigma_white_cm"])
    noise_cm = float(np.sqrt(max(0.0, atd_j["sigma_white_cm"] ** 2 - float(np.mean(dfl_sig)) ** 2)))

    # ---- cleaning stats of the ATD: used to inject glitches
    li = np.flatnonzero(live)       # glitch rates are measured on live frames only (outside them the ball is far noisier)
    clean_live = limpieza.clean_tracking(d["ball_xy"][li], {k: v[li] for k, v in d["players_slots"].items()},
                                         to_metres=atd.PITCH, t=li)
    st = clean_live["stats"]
    rep = clean_live["report"]
    n_ball = rep["ball"]["detections"]
    rm_runs = run_lengths(bp[li] & ~clean_live["ball_present"], [(0, len(li))])
    q = np.linspace(0, 1, 21)
    cal = {
        "note": "aggregates only",
        "count_hist_live": count_hist,
        "count_chain": chains,
        "jitter": {"atd_sigma_white_cm": atd_j["sigma_white_cm"], "dfl_sigma_white_cm_mean": float(np.mean(dfl_sig)),
                   "noise_to_add_cm": noise_cm, "atd_abs_a2_cm_p50_p90_p99": atd_j["abs_a2_cm_p50_p90_p99"]},
        "ball_glitch": {
            # what limpieza removes from the ATD ball outside the holes, as blocks of consecutive frames
            "blocks_per_detection": len(rm_runs) / max(n_ball, 1),
            "block_len_hist": hist_edges(rm_runs),
            "jump_dist_m_quantiles": np.quantile(st["ball_jump_dist"], q).round(2).tolist() if st["ball_jump_dist"] else [],
        },
        "player_glitch": {
            "id_swap_per_frame": {t: rep[t]["teleport"]["n"] / 2 / N_live for t in ("H", "A")},
            "jump_dist_m_quantiles": np.quantile(st["player_jump_dist"], q).round(2).tolist() if st["player_jump_dist"] else [],
        },
    }
    CAL_ATD.write_text(json.dumps(cal))
    print("calibration written:", CAL_GAPS.name, CAL_ATD.name)
    print("ball absent live %.3f | P(dead|absent) %.2f P(dead|present) %.2f | P(absent|dead) %.2f P(absent|alive) %.3f | noise to add %.2f cm"
          % (p_abs, a_dead_given_absent, b_dead_given_present, p_abs_dead, p_abs_alive, noise_cm))


def load_calibration():
    if not CAL_GAPS.exists() or not CAL_ATD.exists():
        calibrar()
    return json.loads(CAL_GAPS.read_text()), json.loads(CAL_ATD.read_text())


# ---------------------------------------------------------------- calibration by gap length
def gap_records(absent, segments, spd, edge=GAP_EDGE, label=None):
    """Absent runs inside `segments` [(lo, hi)] as arrays: start, length, cls (index of LEN_NAMES) and, over the
    frames at least `edge` away from both borders of the run, the number (kept), the speed histogram over SPEED_BINS
    (hist), the speed sum (ssum) of `spd` and, with `label`, how many of those frames (d_kept) and of all the
    frames of the run (d_all) are "D"."""
    parts = [(lo + a, n) for lo, hi in segments for a, n in [_runs(absent[lo:hi])]]
    st = np.concatenate([p[0] for p in parts]) if parts else np.zeros(0, int)
    ln = np.concatenate([p[1] for p in parts]) if parts else np.zeros(0, int)
    out = {"start": st, "length": ln, "cls": np.digitize(ln, LEN_CUTS),
           "kept": np.maximum(ln - 2 * edge, 0), "hist": np.zeros((len(st), len(SPEED_BINS) - 1)),
           "ssum": np.zeros(len(st)), "d_kept": np.zeros(len(st), int), "d_all": np.zeros(len(st), int)}
    dead = None if label is None else np.asarray(label) == "D"
    for i, (a, n) in enumerate(zip(st, ln)):
        if dead is not None:
            out["d_all"][i] = dead[a:a + n].sum()
        if n > 2 * edge:
            v = spd[a + edge:a + n - edge]
            out["hist"][i] = np.histogram(np.clip(v, 0, 4.999), SPEED_BINS)[0]
            out["ssum"][i] = v.sum()
            if dead is not None:
                out["d_kept"][i] = dead[a + edge:a + n - edge].sum()
    return out


def gap_class_stats(absent, segments, slots_m, win, label=None):
    """Per length class (LEN_NAMES): n gaps, frames in gaps, kept frames, mean player speed (m/s, `win`-frame moving
    average, kept frames only) and, with `label`, the "D" share of the kept frames and of all the frames of the gaps."""
    r = gap_records(absent, segments, smoothed_player_speed(slots_m, win), label=label)
    out = []
    for c in range(len(LEN_NAMES)):
        m = r["cls"] == c
        kept = int(r["kept"][m].sum())
        out.append({"n": int(m.sum()), "frames": int(r["length"][m].sum()), "kept": kept,
                    "mean_speed": float(r["ssum"][m].sum() / kept) if kept else None,
                    "d_kept": float(r["d_kept"][m].sum() / kept) if kept and label is not None else None,
                    "d_all": float(r["d_all"][m].sum() / max(r["length"][m].sum(), 1)) if label is not None else None})
    return out


def dfl_components(windows):
    """{name: (f_dead, f_alive)}: normalised histograms over SPEED_BINS of the DFL player speed (moving average of
    `windows[name]` frames) in the frames whose whole +-GAP_EDGE neighbourhood has the same ball status, so the
    components are as pure as the kept frames of a gap. Only the development matches (dfl.DESARROLLO): the two reserved ones
    are for the final confirmation. (calibrar(), the old mode, keeps its own mixture and caches untouched.)"""
    h = {k: [np.zeros(len(SPEED_BINS) - 1), np.zeros(len(SPEED_BINS) - 1)] for k in windows}
    box = np.ones(2 * GAP_EDGE + 1)
    for mid in dfl.DESARROLLO:
        a = dfl_arrays(mid)
        players = np.concatenate([a["teams"]["H"], a["teams"]["A"]], axis=1)
        dead = a["dead"]
        pure_dead = np.convolve(dead, box, "same") == len(box)
        pure_alive = np.convolve(~dead, box, "same") == len(box)
        for k, w in windows.items():
            s = np.clip(smoothed_player_speed(players, w), 0, 4.999)
            h[k][0] += np.histogram(s[pure_dead], SPEED_BINS)[0]
            h[k][1] += np.histogram(s[pure_alive], SPEED_BINS)[0]
    return {k: (v[0] / v[0].sum(), v[1] / v[1].sum()) for k, v in h.items()}


def calibrar_por_largo(n_boot=500, seed=0):
    """Gap-length classes of the ATD (absent runs of the raw ball inside the live halves, as calibrar()): share of live
    frames, run-length histogram over EDGES and P(dead | absent, class) by the same speed mixture of DFL components,
    for each LEN_WINDOWS window, over the frames away from the borders of the gaps. Bootstrap interval resampling the
    gaps of each class. Writes ONLY aggregates to huecos_largo_atd.json (huecos_atd.json is not touched)."""
    import atd
    import muerto
    d = atd.load_tactical()
    live = limpieza.live_mask()
    bp = d["ball_present"]
    clean = limpieza.clean_atd(d, mask=live)
    slots_m = np.concatenate([clean["players_slots"]["H"], clean["players_slots"]["A"]], axis=1) * np.array(atd.PITCH)
    n_live = int(live.sum())
    comps = dfl_components(LEN_WINDOWS)
    rng = np.random.default_rng(seed)
    recs = {k: gap_records(~bp, ATD_SEGS, smoothed_player_speed(slots_m, w)) for k, w in LEN_WINDOWS.items()}
    cls = recs[LEN_WIN]["cls"]
    ln = recs[LEN_WIN]["length"]
    classes = []
    for c, name in enumerate(LEN_NAMES):
        m = np.flatnonzero(cls == c)
        e = {"name": name, "n_runs": int(len(m)), "frames": int(ln[m].sum()), "frac_live": float(ln[m].sum() / n_live),
             "per_90min": float(len(m) * 90 * 60 * FPS / n_live), "lo_frames": ([0] + list(LEN_CUTS))[c],
             "hi_frames": (list(LEN_CUTS) + [None])[c], "run_hist": hist_edges(ln[m])}
        for k in LEN_WINDOWS:
            H = recs[k]["hist"][m]
            f_dead, f_alive = comps[k]
            kept = int(recs[k]["kept"][m].sum())
            pd = mix_dead(H.sum(0), f_dead, f_alive) if kept else None
            boot = []
            for _ in range(n_boot if kept else 0):
                g = H[rng.integers(0, len(m), len(m))].sum(0)
                if g.sum() > 0:
                    boot.append(mix_dead(g, f_dead, f_alive))
            e[k] = {"pd": pd, "pd_lo": float(np.percentile(boot, 2.5)) if boot else None,
                    "pd_hi": float(np.percentile(boot, 97.5)) if boot else None, "n_boot": len(boot),
                    "kept_frames": kept, "mean_speed": float(recs[k]["ssum"][m].sum() / kept) if kept else None}
        classes.append(e)
    # the challenge as the pipeline sees it (after limpieza removes ball glitches), same classes
    li = np.flatnonzero(live)
    clean_pres = np.zeros(len(bp), bool)
    clean_pres[li] = limpieza.clean_tracking(d["ball_xy"][li], {k: v[li] for k, v in d["players_slots"].items()},
                                             to_metres=atd.PITCH, t=li)["ball_present"]
    rl = gap_class_stats(~clean_pres, ATD_SEGS, slots_m, LEN_WINDOWS[LEN_WIN])
    ball_m = clean["ball_xy"]
    out_m = muerto.ball_features(ball_m, np.ones(len(bp)), atd.PITCH)[0]
    vis = live & ~np.isnan(out_m)
    res = {"note": "aggregates only. classes: absent runs of the raw ATD ball in the live halves; pd by DFL speed mixture over "
                   "frames >= edge_frames from both borders of each gap",
           "live_frames": n_live, "ball_absent_frac_live": float((~bp[live]).mean()), "edge_frames": GAP_EDGE,
           "class_cuts_frames": list(LEN_CUTS), "run_edges_frames": EDGES.tolist(), "n_boot": n_boot, "seed": seed,
           "windows": {k: {"frames": w, "dfl_mean_speed_dead": float((comps[k][0] * (SPEED_BINS[:-1] + 0.05)).sum()),
                           "dfl_mean_speed_alive": float((comps[k][1] * (SPEED_BINS[:-1] + 0.05)).sum())}
                       for k, w in LEN_WINDOWS.items()},
           "classes": classes,
           "limpio": [{"n": r["n"], "frames": r["frames"], "frac_live": r["frames"] / n_live,
                       "per_90min": r["n"] * 90 * 60 * FPS / n_live, "mean_speed": r["mean_speed"]} for r in rl],
           "ball_absent_frac_live_limpio": float((~clean_pres[live]).mean()),
           "ball_out_gt_1m_frac": float((out_m[vis] > 1).mean())}
    CAL_LARGO.write_text(json.dumps(res, indent=1))
    print("written:", CAL_LARGO.name)
    for k in LEN_WINDOWS:
        print("P(dead | absent, class), window %s (bootstrap %d, 95%% interval):" % (k, n_boot))
        for e in classes:
            q = e[k]
            print("  %-7s n %3d  %5.2f %% of live frames  kept %6d fr  v %s  pd %s [%s, %s]" % (
                e["name"], e["n_runs"], 100 * e["frac_live"], q["kept_frames"],
                "-" if q["mean_speed"] is None else "%.2f" % q["mean_speed"],
                "-" if q["pd"] is None else "%.3f" % q["pd"],
                "-" if q["pd_lo"] is None else "%.3f" % q["pd_lo"], "-" if q["pd_hi"] is None else "%.3f" % q["pd_hi"]))
    return res


def load_calibration_largo():
    if not CAL_LARGO.exists():
        calibrar_por_largo()
    return json.loads(CAL_LARGO.read_text())


# ---------------------------------------------------------------- degradation steps
def _sample_run(rng, hist):
    """One run length from a histogram over EDGES (uniform inside the bin)."""
    h = np.asarray(hist, float)
    i = rng.choice(len(h), p=h / h.sum())
    lo = EDGES[i]
    hi = EDGES[i + 1] if i + 1 < len(EDGES) else lo * 2
    return int(rng.integers(lo, max(hi, lo + 1)))


def _len_gap_model(largo):
    """(P(dead | absent), draw(rng) -> run length) of gap_mode="coupled_len". P is the frame-weighted mean of the
    class values; draw picks a class with probability proportional to its number of runs x (1 - pd_c) (so the frames of
    in-play gaps per class go as f_c (1 - pd_c)) and a length from that class histogram, kept inside the class bounds."""
    cl = largo["classes"]
    pd = np.array([c[LEN_WIN]["pd"] if c[LEN_WIN]["pd"] is not None else 1.0 for c in cl])
    f = np.array([c["frac_live"] for c in cl])
    w = np.array([c["n_runs"] for c in cl]) * (1 - pd)

    def draw(rng):
        if w.sum() <= 0:
            raise ValueError("coupled_len: every class has P(dead | absent) = 1, so there is no in-play gap to draw; "
                             "p_dead_given_absent < 1 cannot be forced on this calibration")
        c = cl[rng.choice(len(cl), p=w / w.sum())]
        h = np.asarray(c["run_hist"], float)
        i = rng.choice(len(h), p=h / h.sum())
        lo, hi = EDGES[i], (EDGES[i + 1] if i + 1 < len(EDGES) else EDGES[i] * 2)
        lo, hi = max(lo, c["lo_frames"]), min(hi, c["hi_frames"] or hi)
        return int(rng.integers(lo, max(hi, lo + 1)))

    return float((f * pd).sum() / f.sum()), draw


def ball_gaps(dead, rng, gaps, mode, p_dead_given_absent=None, largo=None, info=None):
    """Boolean (N,) ball-absent mask. mode "coupled_len" needs `largo` (load_calibration_largo). In that mode the in-play
    gaps are placed on live frames only (never across a dead stretch), keeping >= 1 visible frame from every other gap,
    and the later steps leave them alone; if `info` is a dict it gets "placed" (gaps placed) and "out_of_class" (gaps
    whose final run length is in another class than the drawn one)."""
    N = len(dead)
    absent = np.zeros(N, bool)
    if mode == "indep":
        state = bool(rng.random() < gaps["ball_absent_frac_live"])
        i = 0
        while i < N:
            n = _sample_run(rng, gaps["absent_run_hist"] if state else gaps["present_run_hist"])
            absent[i:i + n] = state
            i += n
            state = not state
        return absent
    dep = gaps["dependence"]
    # keep the ATD total (40.7 %) and the share of absent frames that are dead (98 %) with THIS match's
    # own dead share (the ATD estimate is 52 %, DFL matches go from 41 to 52 %)
    pda, draw = dep["p_dead_given_absent"], lambda r: _sample_run(r, gaps["absent_run_hist"])
    if mode == "coupled_len":
        pda, draw = _len_gap_model(largo)
    pda = pda if p_dead_given_absent is None else p_dead_given_absent
    share, p_abs = float(dead.mean()), gaps["ball_absent_frac_live"]
    p_dead = min(1.0, pda * p_abs / share)
    p_alive = (1 - pda) * p_abs / (1 - share)
    d = np.diff(np.concatenate([[0], dead.astype(int), [0]]))
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    for a, b in zip(s, e):
        if rng.random() < p_dead:
            absent[a:b] = True
    target = p_alive * (~dead).sum()
    added, alive_idx = 0, np.flatnonzero(~dead)
    placed = []                                  # (start, length) of the in-play gaps of coupled_len
    if mode == "coupled_len":
        for _ in range(400000):
            if added >= target:
                break
            a = int(rng.choice(alive_idx))
            n = draw(rng)
            lo, hi = max(a - 1, 0), min(a + n + 1, N)
            if a + n > N or dead[a:a + n].any() or absent[lo:hi].any():
                continue                         # crosses a dead stretch or touches another gap: draw again
            absent[a:a + n] = True
            added += n
            placed.append((a, n))
    else:
        for _ in range(100000):
            if added >= target:
                break
            a = int(rng.choice(alive_idx))
            n = draw(rng)
            seg = slice(a, a + n)
            added += int((~absent[seg] & ~dead[seg]).sum())
            absent[seg] = True
    # The ATD loses and recovers the ball more often than DFL has dead-ball stretches (150 absent runs per
    # 158,857 live frames against ~100 dead runs per DFL match): the extra runs are the ball seen again for
    # a few seconds inside a stoppage. Open short visible windows (ATD present-run histogram, < 10 s)
    # inside long absent runs until the run count matches, then hide more dead runs to keep the 40.7 %.
    ph = np.asarray(gaps["present_run_hist"], float)
    ph[[i for i in range(len(ph)) if i + 1 >= len(EDGES) or EDGES[i + 1] > 251]] = 0
    st, ln = _runs(absent)
    deficit = int(round(gaps["n_absent_runs"] / gaps["live_frames"] * N)) - len(ln)
    big = np.flatnonzero(ln >= 400)
    keep_gap = np.zeros(N, bool)                 # coupled_len: in-play gaps and their visible neighbours stay as placed
    for a, n in placed:
        keep_gap[max(a - 1, 0):a + n + 1] = True
    if placed:
        big = np.array([k for k in big if not keep_gap[st[k]:st[k] + ln[k]].any()], int)
    if deficit > 0 and len(big) and ph.sum():
        w_run = ln[big] / ln[big].sum()
        for _ in range(deficit):
            k = big[rng.choice(len(big), p=w_run)]
            w = _sample_run(rng, ph)
            if ln[k] > w + 250:      # both remaining pieces stay >= 5 s, as in the ATD
                a = st[k] + int(rng.integers(125, ln[k] - w - 125))
                absent[a:a + w] = False
    st, ln = _runs(dead)
    spare = [(a, n) for a, n in zip(st, ln) if not absent[a:a + n].all() and not keep_gap[a:a + n].any()]
    rng.shuffle(spare)
    for a, n in spare:
        if absent.mean() >= gaps["ball_absent_frac_live"]:
            break
        absent[a:a + n] = True
    if info is not None:
        s2, l2 = _runs(absent)
        out = 0
        for a, n in placed:
            k = int(np.searchsorted(s2, a, side="right")) - 1
            out += int(not (s2[k] == a and np.digitize(l2[k], LEN_CUTS) == np.digitize(n, LEN_CUTS)))
        info.update(placed=len(placed), out_of_class=out)
    return absent


def _runs(mask):
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    a, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return a, e - a


def inject_ball_glitches(ball, rng, cal):
    """Blocks of consecutive present ball frames displaced by a random offset (the tracker locks onto
    something else for a while), at the rate and with the block lengths that limpieza removes from the
    ATD ball. ball (N,2) metres with NaN holes, modified in place."""
    g = cal["ball_glitch"]
    pres = np.flatnonzero(~np.isnan(ball[:, 0]))
    if not len(pres) or not g["jump_dist_m_quantiles"]:
        return
    dq = np.array(g["jump_dist_m_quantiles"])
    for _ in range(rng.poisson(g["blocks_per_detection"] * len(pres))):
        k = int(rng.integers(0, len(pres)))
        ln = _sample_run(rng, g["block_len_hist"])
        ang = rng.uniform(0, 2 * np.pi)
        ball[pres[k:k + ln]] += (rng.choice(dq) + 0.5) * np.array([np.cos(ang), np.sin(ang)])


def degrade_team(P, opp, counts_cfg, ball_present, slots, swap_rate, jump_q, rng):
    """P (N,C,2) real players (NaN off the pitch), opp opposing players. Returns (N,S,2)."""
    N, C, _ = P.shape
    pres = ~np.isnan(P[:, :, 0])
    opp_pres = ~np.isnan(opp[:, :, 0])
    out = np.full((N, slots, 2), np.nan)
    cdf = []
    for s in range(2):
        rows = []
        for r in np.asarray(counts_cfg[s], float):
            tot = r.sum()
            rows.append(list(np.cumsum(r / tot)) if tot > 0 else None)
        cdf.append(rows)
    u = rng.random(N)
    swap_u = rng.random(N) < swap_rate
    slot_of = -np.ones(C + E_MAX, int)
    used = np.zeros(slots, bool)
    dropped = np.zeros(C, bool)
    e_on = np.zeros(E_MAX, bool)
    e_anchor = np.zeros(E_MAX, int)
    e_off = np.zeros((E_MAX, 2))
    c = int(pres[0].sum())

    def take_slot(i):
        free = np.flatnonzero(~used)
        s = int(rng.choice(free))
        used[s] = True
        slot_of[i] = s

    def release(i):
        used[slot_of[i]] = False
        slot_of[i] = -1

    for t in range(N):
        row = cdf[int(ball_present[t])][min(c, N_COUNT - 1)]
        if row is not None:
            c = min(bisect_right(row, u[t]), N_COUNT - 1)
        base = pres[t]
        n0 = int(base.sum())
        gone = np.flatnonzero((slot_of[:C] >= 0) & ~base)
        for i in gone:
            release(i)
        want_real, want_extra = min(c, n0), min(max(c - n0, 0), E_MAX)
        vis = base & ~dropped
        n_vis = int(vis.sum())
        if n_vis > want_real:
            for i in rng.choice(np.flatnonzero(vis), n_vis - want_real, replace=False):
                dropped[i] = True
                release(i)
        elif n_vis < want_real:
            cand = np.flatnonzero(base & dropped)
            for i in rng.choice(cand, min(len(cand), want_real - n_vis), replace=False):
                dropped[i] = False
        vis = base & ~dropped
        for i in np.flatnonzero(vis & (slot_of[:C] < 0)):
            take_slot(i)
        n_e = int(e_on.sum())
        if n_e < want_extra and opp_pres[t].any():
            for e in np.flatnonzero(~e_on)[: want_extra - n_e]:
                e_on[e] = True
                e_anchor[e] = int(rng.choice(np.flatnonzero(opp_pres[t])))
                e_off[e] = rng.normal(0, 1.0, 2)
                take_slot(C + e)
        elif n_e > want_extra:
            for e in rng.choice(np.flatnonzero(e_on), n_e - want_extra, replace=False):
                e_on[e] = False
                release(C + e)
        if swap_u[t] and vis.sum() >= 2:
            ids = np.flatnonzero(vis)
            p1 = int(rng.choice(ids))
            others = ids[ids != p1]
            dist = np.linalg.norm(P[t, others] - P[t, p1], axis=1)
            p2 = int(others[np.argmin(np.abs(dist - rng.choice(jump_q)))])
            slot_of[p1], slot_of[p2] = slot_of[p2], slot_of[p1]
        ids = np.flatnonzero(vis)
        out[t, slot_of[ids]] = P[t, ids]
        for e in np.flatnonzero(e_on):
            out[t, slot_of[C + e]] = opp[t, e_anchor[e]] + e_off[e]
    return out


def degrade(mid, seed=0, gap_mode="coupled", p_dead_given_absent=None):
    gaps, cal = load_calibration()
    a = dfl_arrays(mid)
    rng = np.random.default_rng([seed, list(dfl.MATCHES).index(mid)])
    N = len(a["df"])
    absent = ball_gaps(a["dead"], rng, gaps, gap_mode, p_dead_given_absent,
                       load_calibration_largo() if gap_mode == "coupled_len" else None)
    ball = a["ball"].copy()                      # z never enters: only x, y exist here
    ball[absent] = np.nan
    inject_ball_glitches(ball, rng, cal)
    swap_q = cal["player_glitch"]["jump_dist_m_quantiles"] or [3.0]
    raw = {}
    for team, other in (("H", "A"), ("A", "H")):
        raw[team] = degrade_team(a["teams"][team], a["teams"][other],
                                 cal["count_chain"][team], ~absent, SLOTS[team],
                                 cal["player_glitch"]["id_swap_per_frame"][team], swap_q, rng)
        if cal["jitter"]["noise_to_add_cm"] > 0:
            raw[team] = raw[team] + rng.normal(0, cal["jitter"]["noise_to_add_cm"] / 100, raw[team].shape)
    return a, ball, raw


def _f32(x):
    """Metres -> normalised 0-1 (ATD convention), float32 on disk (1e-5 m resolution)."""
    return (x / np.array(ATD_PITCH)).astype(np.float32)


def build(mid, seed=0, gap_mode="coupled", p_dead_given_absent=None):
    a, ball, raw = degrade(mid, seed, gap_mode, p_dead_given_absent)
    df = a["df"]
    clean = limpieza.clean_tracking(ball, raw, to_metres=(1.0, 1.0), t=df["frame"].values)
    import dfl_ultimo_toque
    lab = dfl_ultimo_toque.last_touch_labels(mid)
    path = _cache_path(mid, seed, gap_mode, p_dead_given_absent)
    np.savez_compressed(
        path,
        ball_xy=_f32(clean["ball_xy"]), H=_f32(clean["players_slots"]["H"]), A=_f32(clean["players_slots"]["A"]),
        raw_ball_xy=_f32(ball), raw_H=_f32(raw["H"]), raw_A=_f32(raw["A"]),
        frame=df["frame"].values, period=df["period"].values, ball_status=df["ball_status"].values,
        label=df["label"].values.astype("U1"), label_ut=lab["label_ut"].values.astype("U1"),
        report=json.dumps(clean["report"]))
    return path


def calibration_largo_hash():
    """Short fingerprint of huecos_largo_atd.json (computed first if missing): part of the coupled_len cache name, so a
    recalibration never reuses banks built with the old one."""
    load_calibration_largo()
    return hashlib.sha1(CAL_LARGO.read_bytes()).hexdigest()[:8]


def _cache_path(mid, seed, gap_mode, p_dead_given_absent=None):
    tag = "" if p_dead_given_absent is None else "_pd{:.2f}".format(p_dead_given_absent)
    if gap_mode == "coupled_len":                # the other modes keep their names
        gap_mode += "_" + calibration_largo_hash()
    return CACHE / "degradado_{}_s{}_{}{}.npz".format(mid, seed, gap_mode, tag)


def gap_placement_check(mid, seed=0):
    """{"placed", "out_of_class"} of the in-play gaps of coupled_len for (mid, seed): the same draw as degrade() (same
    rng, same first call), without building the bank."""
    gaps, _ = load_calibration()
    a = dfl_arrays(mid)
    rng = np.random.default_rng([seed, list(dfl.MATCHES).index(mid)])
    info = {}
    ball_gaps(a["dead"], rng, gaps, "coupled_len", None, load_calibration_largo(), info)
    return info


# ---------------------------------------------------------------- ball smoothing (in memory, no cache)
# The challenge ball is smoothed; a bank ball can be smoothed to match it.
#   "g6", "g8"  centred Gaussian, sigma 6 / 8 frames (0.24 / 0.32 s). No delay.
#   "causal"    backward-only filter (output at t uses frames <= t), the control for a challenge ball that
#               was smoothed in real time. Kernel: one-sided Gaussian, sigma 15 frames, chosen with `calibrar_balon.py --barrido-causal`.
BALL_SMOOTH = {"g6": ("gauss", 6.0), "g8": ("gauss", 8.0), "causal": ("causal_halfgauss", 15.0),
               "g2": ("gauss", 2.0), "g3": ("gauss", 3.0), "g4": ("gauss", 4.0), "g10": ("gauss", 10.0)}   # g2..g10: PFF calibration


def _kernel(kind, param):
    """(weights, delay_frames). Centred kernels are applied as a convolution (symmetric); causal ones put
    weight on the frames before t only, kernel[k] = weight of frame t-k."""
    if kind == "gauss":
        h = int(np.ceil(3 * param))
        k = np.exp(-np.arange(-h, h + 1) ** 2 / (2.0 * param ** 2))
        return k / k.sum(), 0.0
    if kind == "causal_halfgauss":                  # one-sided Gaussian, sigma = param frames
        h = int(np.ceil(3 * param))
        k = np.exp(-np.arange(0, h + 1) ** 2 / (2.0 * param ** 2))
        k /= k.sum()
        return k, float((k * np.arange(len(k))).sum())
    if kind == "causal_ema":                        # y[t] = a x[t] + (1-a) y[t-1], a = 1 / (1 + param)
        a = 1.0 / (1.0 + param)
        h = int(np.ceil(np.log(1e-3) / np.log(1 - a)))
        k = a * (1 - a) ** np.arange(h + 1)
        k /= k.sum()
        return k, float((k * np.arange(len(k))).sum())
    raise ValueError(kind)


def kernel_delay_frames(spec):
    """Effective delay (frames, mean of the kernel in time) of a BALL_SMOOTH mode or a (kind, param) pair."""
    kind, param = BALL_SMOOTH[spec] if isinstance(spec, str) else spec
    return _kernel(kind, param)[1]


def visible_stretches(present, period):
    """[(start, stop)] maximal runs of visible ball, split at period changes."""
    present = np.asarray(present, bool)
    period = np.asarray(period)
    n = len(present)
    cut = np.flatnonzero(period[1:] != period[:-1]) + 1
    edges = np.concatenate(([0], cut, [n]))
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        d = np.diff(np.concatenate(([0], present[lo:hi].astype(np.int8), [0])))
        out += [(lo + int(s), lo + int(e)) for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]
    return out


def smooth_ball(ball_xy, period, mode):
    """Copy of ball_xy (N,2) smoothed inside each continuous visible stretch and period (never across a hole
    or the half-time break; holes stay NaN). `mode` is a BALL_SMOOTH key or a (kind, param) pair. Stretch
    borders are edge-padded."""
    kind, param = BALL_SMOOTH[mode] if isinstance(mode, str) else mode
    k, _ = _kernel(kind, param)
    out = ball_xy.copy()
    for a, b in visible_stretches(~np.isnan(ball_xy[:, 0]), period):
        for c in (0, 1):
            x = ball_xy[a:b, c]
            if kind == "gauss":
                h = len(k) // 2
                out[a:b, c] = np.convolve(np.pad(x, h, mode="edge"), k, "valid")
            else:                                   # causal: y[t] = sum_k w[k] x[t-k]
                out[a:b, c] = np.convolve(np.pad(x, (len(k) - 1, 0), mode="edge"), k, "valid")
    return out


# ---------------------------------------------------------------- dead-ball motion (in memory, no cache)
# DFL freezes the ball (speed exactly 0) for the whole of a ball_status==0 stretch; the challenge ball does not stop
# (median 2 m/s while the players are calm). dead_motion adds a bridge displacement to every continuous visible
# stretch of ball_status==0 (cut by period and by holes): with probability q_quieto the ball stays almost still
# (OU speed scale v_quiet), otherwise an OU velocity (time constant tau, RMS speed s, s ~ lognormal(median v_med,
# sigma)) is integrated and its straight line between the two ends is subtracted, so the displacement is 0 on
# both borders (no jump against the live play or the hole). Order in load_degraded: motion first, then smoothing.
# The keys of DEAD_MOTION are filled by bancos/calibrar_muerto.py (parameters also in data/cache/dead_motion_<key>.json).
DEAD_MOTION = {"m1": {"q_quieto": 0.0, "v_med": 3.0, "sigma": 1.0}}      # calibrar_muerto.py, pd_cal+g6, 5 dev matches, seed 0
DEAD_MOTION_DEFAULTS = {"tau": 2.0, "v_quiet": 0.1}      # s, m/s: fixed, not calibrated


def _ou_velocity(n, scale, tau, rng):
    """(n,2) stationary Ornstein-Uhlenbeck velocity (m/s), RMS speed `scale`, time constant tau (s): white noise
    through the exponential kernel (truncated at 8 tau), so no loop over frames."""
    phi = np.exp(-1.0 / (FPS * tau))
    k = phi ** np.arange(int(np.ceil(8 * FPS * tau)))
    k *= np.sqrt(1 - phi ** 2) * scale / np.sqrt(2)             # per-axis std = scale / sqrt(2)
    eps = rng.standard_normal((n + len(k) - 1, 2))
    return np.stack([np.convolve(eps[:, c], k, "valid") for c in (0, 1)], 1)


def bridge_displacement(n, scale, tau, rng):
    """(n,2) metres: integral of an OU velocity (RMS speed `scale`, m/s) minus its straight line, 0 at both ends."""
    x = np.concatenate([np.zeros((1, 2)), np.cumsum(_ou_velocity(n - 1, scale, tau, rng), axis=0) / FPS])
    return x - np.linspace(0, 1, n)[:, None] * x[-1]


def resolve_dead_motion(spec):
    """None, a DEAD_MOTION key or a dict -> complete parameter dict (or None)."""
    if spec is None:
        return None
    p = dict(DEAD_MOTION[spec] if isinstance(spec, str) else spec)
    for k, v in DEAD_MOTION_DEFAULTS.items():
        p.setdefault(k, v)
    return p


def apply_dead_motion(ball_xy, ball_status, period, spec, mid, seed):
    """Copy of ball_xy (N,2), normalised 0-1, with the dead-ball motion added (see above); NaN holes stay NaN and
    stretches shorter than 3 frames or live frames are untouched. Deterministic in (mid, seed, stretch start)."""
    p = resolve_dead_motion(spec)
    out = np.array(ball_xy, float)
    if p is None:
        return out
    scale = np.asarray(ATD_PITCH)
    mi = list(dfl.MATCHES).index(mid)
    stretches = visible_stretches(~np.isnan(out[:, 0]) & (np.asarray(ball_status) == 0), period)
    for a, b in stretches:
        if b - a < 3:
            continue
        rng = np.random.default_rng([seed, mi, a, 31])
        quiet = rng.random() < p["q_quieto"]
        s = p["v_quiet"] if quiet else p["v_med"] * np.exp(p["sigma"] * rng.standard_normal())
        out[a:b] += bridge_displacement(b - a, s, p["tau"], rng) / scale
    return out


def load_degraded(mid, seed=0, gap_mode="coupled", cleaned=True, p_dead_given_absent=None, ball_smooth=None,
                  dead_motion=None):
    """Degraded + cleaned DFL match with the shape of the cleaned ATD.

    keys: ball_xy (N,2), ball_present (N,), players_slots {"H": (N,24,2), "A": (N,22,2)}, n_present
    {"H","A"}, all in the same normalised 0-1 coordinates as the ATD (no ball height anywhere), plus
    frame, period, ball_status, label (BallPossession), label_ut (last touch), report (limpieza
    aggregates). cleaned=False returns the degraded tracking before limpieza.

    ball_smooth=None leaves everything as cached. A BALL_SMOOTH key ("g6", "g8", "causal") smooths ball_xy in
    memory (see smooth_ball); the cache files are never written with it.
    dead_motion=None leaves the frozen DFL dead ball. A DEAD_MOTION key or a dict (q_quieto, v_med, sigma, optional
    tau, v_quiet) adds a bridge displacement to every visible ball_status==0 stretch (see apply_dead_motion), in
    memory and before the smoothing."""
    path = _cache_path(mid, seed, gap_mode, p_dead_given_absent)
    if not path.exists():
        build(mid, seed, gap_mode, p_dead_given_absent)
    z = np.load(path)
    pre = "" if cleaned else "raw_"
    ball, H, A = (z[pre + k].astype(float) for k in ("ball_xy", "H", "A"))
    if dead_motion is not None:
        ball = apply_dead_motion(ball, z["ball_status"], z["period"], dead_motion, mid, seed)
    if ball_smooth is not None:
        ball = smooth_ball(ball, z["period"], ball_smooth)
    return {
        "ball_xy": ball, "ball_present": ~np.isnan(ball[:, 0]),
        "players_slots": {"H": H, "A": A},
        "n_present": {"H": (~np.isnan(H[:, :, 0])).sum(1), "A": (~np.isnan(A[:, :, 0])).sum(1)},
        "frame": z["frame"], "period": z["period"], "ball_status": z["ball_status"],
        "label": z["label"], "label_ut": z["label_ut"], "report": json.loads(str(z["report"])),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mids", nargs="*")
    ap.add_argument("--calibrar", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calibrar-largo", action="store_true", help="P(dead | absent) by gap length -> huecos_largo_atd.json")
    ap.add_argument("--gap-mode", default="coupled", choices=["coupled", "coupled_len", "indep"])
    ap.add_argument("--p-dead", type=float, default=None)
    args = ap.parse_args()
    if args.calibrar:
        calibrar()
    elif args.calibrar_largo:
        calibrar_por_largo()
    else:
        for m in args.mids or dfl.MATCHES:
            print(m, build(m, args.seed, args.gap_mode, args.p_dead).name)
