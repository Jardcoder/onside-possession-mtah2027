"""Team assigned to each deflection of toques v2 (pure array functions, no I/O).

toques.deflection_candidates gives every deflection a touch frame and, as team, the nearest player at that frame
(mode "T0"). Here the same deflections (same frames) can get another team. The control touches and the hole
filling are untouched. Modes (all read ball, player slots and period only; nothing here sees a label, a ball
status or an event, and a slot id is never used as an identity: a slot is only followed through time):

  T0           the candidate's own team (nearest player at the touch frame): the v2 behaviour.
  lag L        nearest player at the frame `peak - L` (clamped to the visible stretch). L in {2, 4, 6}.
  minwin L     the player with the smallest distance to the ball inside [peak - L, peak + L] (clamped to the
               stretch): team of the frame where the nearest-player distance is minimal. L in {3, 6}.
  dir          candidates = the nearest home and the nearest away player at the peak, each only if it is closer
               than `r`. Both: the one with the larger cosine between the ball impulse (vn - vp, the detector's)
               and the vector player -> ball. Only one: that one. None, or equal cosines: T0.
  kink         same two candidates; the one whose own slot changes velocity the most wins: |v_after - v_before|
               of the slot, each the mean of the frame-to-frame velocities over the `win` frames before / after
               the peak (steps faster than MAX_PLAYER_SPEED are slot jumps and are dropped). Invalid data on
               either candidate, or equal values: T0. Only one candidate: that one.
  dir+tie D    T0, except when |d_home - d_away| < D (nearest distances at the peak): then `dir` decides.
  kink+tie D   same with `kink`.
"""
import numpy as np

import toques
from baseline import HOME, AWAY, NOBODY

FPS = toques.FPS
MAX_PLAYER_SPEED = 12.0       # m/s, a faster frame-to-frame step of a slot is a tracking jump
MODES = ("T0", "lag2", "lag4", "lag6", "minwin3", "minwin6", "dir", "kink",
         "dir+tie1.0", "dir+tie2.0", "kink+tie1.0", "kink+tie2.0")


def parse_mode(mode):
    """mode name -> (kind, L, D): kind in T0/lag/minwin/dir/kink, L the frame offset, D the tie width in metres."""
    if mode == "T0":
        return "T0", None, None
    if mode.startswith("lag"):
        return "lag", int(mode[3:]), None
    if mode.startswith("minwin"):
        return "minwin", int(mode[6:]), None
    kind, _, d = mode.partition("+tie")
    if kind in ("dir", "kink"):
        return kind, None, float(d) if d else None
    raise ValueError(f"unknown mode {mode!r}")


def _nearest(players, key, frames, ball_at, scale):
    """Per candidate: distance (m, inf if nobody), slot index and (2,) metre position of the nearest `key` player."""
    p = np.asarray(players[key][frames], float) * scale                    # (C, S, 2)
    d = np.linalg.norm(p - ball_at[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    idx = d.argmin(axis=1)
    rows = np.arange(len(frames))
    return d[rows, idx], idx, p[rows, idx]


def _slot_kink(players, key, idx, peak, start, stop, win, scale):
    """|v_after - v_before| (m/s) of slot `idx` around each peak; NaN when either side has no valid step."""
    fr = np.clip(peak[:, None] + np.arange(-win, win + 1)[None, :], start[:, None], stop[:, None] - 1)
    track = np.asarray(players[key])[fr, idx[:, None]] * scale             # (C, 2 win + 1, 2)
    step = np.diff(track, axis=1) * FPS                                    # (C, 2 win, 2)
    ok = np.isfinite(step).all(axis=2) & (np.linalg.norm(step, axis=2) <= MAX_PLAYER_SPEED)
    step = np.where(ok[:, :, None], step, 0.0)

    def mean_vel(sl):
        n = ok[:, sl].sum(axis=1)
        v = step[:, sl].sum(axis=1) / np.maximum(n, 1)[:, None]
        return v, n

    vb, nb = mean_vel(slice(0, win))
    va, na = mean_vel(slice(win, 2 * win))
    out = np.linalg.norm(va - vb, axis=1)
    out[(nb == 0) | (na == 0)] = np.nan
    return out


def _decide(kind, cand, near, players, scale, r, win):
    """For `dir` / `kink`: (team, decided) per candidate. decided False means "no opinion" (use T0)."""
    peak = cand["peak"]
    ball_at = near["ball_m"][peak]
    dh, ih, ph = _nearest(players, "H", peak, ball_at, scale)
    da, ia, pa = _nearest(players, "A", peak, ball_at, scale)
    inh, ina = dh < r, da < r
    if kind == "dir":
        dvv = cand["vn"] - cand["vp"]
        nd = np.linalg.norm(dvv, axis=1)

        def score(pos):
            vec = ball_at - pos
            den = nd * np.linalg.norm(vec, axis=1)
            return np.where(den > 0, (dvv * vec).sum(axis=1) / np.where(den > 0, den, 1.0), 0.0)

        sh, sa = score(ph), score(pa)
    else:
        sh = _slot_kink(players, "H", ih, peak, cand["start"], cand["stop"], win, scale)
        sa = _slot_kink(players, "A", ia, peak, cand["start"], cand["stop"], win, scale)
    both = inh & ina
    valid = np.where(both, np.isfinite(sh) & np.isfinite(sa) & (sh != sa), False)
    team = np.where(sh > sa, HOME, AWAY).astype(np.int8)
    team = np.where(inh & ~ina, HOME, np.where(ina & ~inh, AWAY, team)).astype(np.int8)
    decided = (inh ^ ina) | valid
    return team, decided, dh, da


def deflection_teams(mode, cand, near, players_slots, pitch, r=3.5, win=5):
    """(C,) int8 HOME/AWAY: the team of each deflection of `cand` (toques.deflection_candidates output) under `mode`.
    `near` is toques.prepare(); r and win are the detector's. The touch frames are never changed."""
    kind, lag, tie = parse_mode(mode)
    t0 = np.asarray(cand["team"], np.int8)
    out = t0.copy()
    if kind == "T0" or len(t0) == 0:
        return out
    scale = np.asarray(pitch, float)
    peak, start, stop = cand["peak"], cand["start"], cand["stop"]
    if kind == "lag":
        f = np.maximum(peak - lag, start)
        t = near["team"][f]
        return np.where(t != NOBODY, t, t0).astype(np.int8)
    if kind == "minwin":
        dmin, team = near["dmin"], near["team"]
        for c in range(len(t0)):
            lo, hi = max(start[c], peak[c] - lag), min(stop[c], peak[c] + lag + 1)
            w = dmin[lo:hi]
            if np.isfinite(w).any():
                out[c] = team[lo + int(np.argmin(w))]
        return out
    team, decided, dh, da = _decide(kind, cand, near, players_slots, scale, r, win)
    if tie is not None:
        with np.errstate(invalid="ignore"):
            decided &= np.abs(dh - da) < tie                               # inf - inf = nan -> False
    return np.where(decided, team, t0).astype(np.int8)


def detect_touches(mode, near, players_slots, pitch, runs, r=2.0, dv=4.0, win=3, r_ctrl=1.0, k=12):
    """toques.detect_touches with the team of every deflection assigned by `mode` (frames and control touches
    as in v2). Returns (frames, teams) sorted."""
    d = toques.deflection_candidates(near, runs, r=r, dv=dv, win=win)
    teams = deflection_teams(mode, d, near, players_slots, pitch, r=r, win=win)
    return toques.merge_touches((d["frame"], teams), toques.control_touches(near, runs, r_ctrl=r_ctrl, k=k))
