"""Automatic cleaning of per-frame tracking (ball + player slots). Rules only, no manual edits.

Same array interface as src/atd.py: ball_xy (N,2), players_slots {"H": (N,S,2), "A": (N,S,2)}
with NaN where nothing is observed. Works on the ATD (normalised 0-1 coordinates, pass
`to_metres=atd.PITCH`) and on the degraded DFL (metres, `to_metres=(1, 1)`). Output keeps the input
units and shapes; cleaned detections become NaN (the ball is NOT interpolated: holes stay NaN).

Rules, applied in this order (counts in the report are attributed to the first rule that
touches a detection, so they are disjoint):

  1. Impossible jumps (per slot and for the ball). A link between two consecutive valid
     detections of the same series is impossible when
         dist > V_MAX * dt + TOL        (dt in seconds from the frame index, so it also covers holes)
     V_MAX player = 12 m/s: the physical sprint ceiling of a footballer is ~11-12 m/s (Bolt's peak
     is 12.4); in clean DFL tracking the 99.99th percentile of player step speed is 9.2 m/s.
     V_MAX ball = 40 m/s (144 km/h): the hardest strikes recorded in elite play are ~35-38 m/s (the
     horizontal component is lower); in the clean DFL alive ball only 0.01-0.1% of steps exceed it
     (the high end is restarts and quantisation).
     TOL (0.15 m player, 0.3 m ball) absorbs position error when dt is 1-2 frames. For players it is
     about 4x the bound 2 x p99.9 of DFL position noise (residual to the centred 5-frame mean,
     0.019 m); the rate of flagged DFL links is flat in TOL (genuine glitches, not noise). Both the
     player and the ball constants reach the submission (its ball is the challenge tracking ball). Calibration without any
     challenge data: research/limpieza_dfl (PFF player tracks proved too noisy; DFL used).
     a) Spikes: a block of up to SPIKE_MAX frames that jumps out and jumps back (the link from the
        last good point to the first point after the block is feasible) is a tracker glitch: the
        whole block becomes NaN.
     b) Teleports: a jump with no return (a track switches to another object: ID swap, ball locked
        on a different thing) cannot be assigned to either side; the first detection after the jump
        becomes NaN and the series continues from the next one.
  2. Duplicates: two slots of the same team closer than DUP_M (0.25 m, in clean DFL they are
     within that distance in 0.7% of frames, i.e. contact) are the same player. The slot with
     the longer current run (continuity) is kept, the other becomes NaN.
  3. Maximum 11 per team per frame. Sequential over frames, only where more than 11 are present:
     the slots that were already selected in the previous frame stay (continuity of the
     trajectory); free places go first to a candidate that appears within HANDOVER_M (6 m) of the
     last position of a slot that has just vanished (the track handed over to another slot, closest
     first), then to the candidates with the longest uninterrupted run. Everything else is NaN.

CLI (prints aggregates only):  challenge/.venv/bin/python src/limpieza.py
"""
import os
import sys

import numpy as np

FPS = 25
V_PLAYER, TOL_PLAYER, SPIKE_MAX_PLAYER = 12.0, 0.15, 12   # m/s, m, frames (0.5 s)
V_BALL, TOL_BALL, SPIKE_MAX_BALL = 40.0, 0.3, 25         # m/s, m, frames (1 s)
DUP_M = 0.25
HANDOVER_M = 6.0
K_MAX = 11


# ---------------------------------------------------------------- rule 1: jumps
def _jump_mask(q, tq, vmax, tol):
    """Boolean (m-1,): link between q[i] and q[i+1] is impossible."""
    dt = np.diff(tq) / FPS
    return np.linalg.norm(np.diff(q, axis=0), axis=1) > vmax * dt + tol


def _remove_jumps_series(p, t, vmax, tol, spike_max):
    """p (n,2) metres with NaN holes, t (n,) frame numbers. Returns (spike_idx, teleport_idx),
    indices into p, plus the list of (block_len) for spikes and the jump distances (m)."""
    valid = np.flatnonzero(~np.isnan(p[:, 0]))
    spikes, teleports, spike_len, jump_dist = [], [], [], []
    if len(valid) < 3:
        return spikes, teleports, spike_len, jump_dist
    removed = np.zeros(len(valid), bool)
    for length in range(1, spike_max + 1):
        ids = np.flatnonzero(~removed)
        if len(ids) < length + 2:
            break
        q, tq = p[valid[ids]], t[valid[ids]]
        jump = _jump_mask(q, tq, vmax, tol)
        out = np.flatnonzero(jump[: len(jump) - length])     # out-jump between q[j], q[j+1]
        if not len(out):
            continue
        back = jump[out + length]                            # return jump between q[j+l], q[j+l+1]
        out = out[back]
        if not len(out):
            continue
        dt = (tq[out + length + 1] - tq[out]) / FPS
        gap = np.linalg.norm(q[out + length + 1] - q[out], axis=1)
        out = out[gap <= vmax * dt + tol]                    # the link around the block is feasible
        for j in out:
            blk = ids[j + 1: j + length + 1]
            if removed[blk].any():
                continue
            jump_dist.append(float(np.linalg.norm(q[j + 1] - q[j])))
            removed[blk] = True
            spike_len.append(length)
    spikes = valid[np.flatnonzero(removed)]
    ids = np.flatnonzero(~removed)
    if len(ids) >= 2:
        q, tq = p[valid[ids]], t[valid[ids]]
        jump = np.flatnonzero(_jump_mask(q, tq, vmax, tol))
        teleports = valid[ids[jump + 1]]
        jump_dist += [float(np.linalg.norm(q[j + 1] - q[j])) for j in jump]
    return spikes, teleports, spike_len, jump_dist


def _remove_jumps(xy, t, vmax, tol, spike_max, stats, key):
    """xy (N,K,2) metres, modified in place. Returns (n_spike, n_teleport) mask arrays (N,K)."""
    N, K, _ = xy.shape
    m_spike = np.zeros((N, K), bool)
    m_tele = np.zeros((N, K), bool)
    for k in range(K):
        s, tp, ln, jd = _remove_jumps_series(xy[:, k], t, vmax, tol, spike_max)
        m_spike[s, k] = True
        m_tele[tp, k] = True
        stats.setdefault(key + "_spike_len", []).extend(ln)
        stats.setdefault(key + "_jump_dist", []).extend(jd)
    xy[m_spike | m_tele] = np.nan
    return m_spike, m_tele


# ---------------------------------------------------------------- rule 2: duplicates
def _run_length(present):
    """(N,S) consecutive-presence run length ending at each frame (0 where absent)."""
    N = present.shape[0]
    idx = np.arange(N)[:, None]
    last_false = np.maximum.accumulate(np.where(~present, idx, -1), axis=0)
    return np.where(present, idx - last_false, 0)


def _dedupe(slots, run):
    """In place. Returns bool mask (N,S) of collapsed (removed) detections."""
    N, S, _ = slots.shape
    gone = np.zeros((N, S), bool)
    for i in range(S):
        for j in range(i + 1, S):
            both = ~np.isnan(slots[:, i, 0]) & ~np.isnan(slots[:, j, 0])
            if not both.any():
                continue
            near = both.copy()
            near[both] = np.linalg.norm(slots[both, i] - slots[both, j], axis=1) < DUP_M
            if not near.any():
                continue
            drop_i = run[:, i] < run[:, j]            # ties keep the lower slot index
            lose = np.where(drop_i, i, j)
            fr = np.flatnonzero(near)
            slots[fr, lose[fr]] = np.nan
            gone[fr, lose[fr]] = True
    return gone


# ---------------------------------------------------------------- rule 3: max 11
def _cap(slots, run, k=K_MAX):
    """In place. Returns mask (N,S) of detections dropped to respect `k` per frame."""
    N, S, _ = slots.shape
    pres = ~np.isnan(slots[:, :, 0])
    dropped = np.zeros((N, S), bool)
    over = np.flatnonzero(pres.sum(1) > k)
    for i in over:
        cand = pres[i]
        prev = pres[i - 1] if i > 0 else np.zeros(S, bool)
        keep = cand & prev
        need = k - int(keep.sum())
        pick = list(np.flatnonzero(keep))
        if need > 0:
            pool = np.flatnonzero(cand & ~keep)
            vacated = np.flatnonzero(prev & ~cand)
            chosen = []
            if len(vacated) and i > 0:
                d = np.linalg.norm(slots[i - 1, vacated][None, :, :] - slots[i, pool][:, None, :], axis=2)
                dmin = d.min(1)
                near = np.flatnonzero(dmin < HANDOVER_M)
                chosen = list(pool[near[np.argsort(dmin[near], kind="stable")]])
            rest = [s for s in pool[np.argsort(-run[i, pool], kind="stable")] if s not in chosen]
            pick += (chosen + rest)[:need]
        else:
            pick = pick[:k]
        drop = np.setdiff1d(np.flatnonzero(cand), pick)
        slots[i, drop] = np.nan
        pres[i, drop] = False
        dropped[i, drop] = True
    return dropped


# ---------------------------------------------------------------- public API
def clean_tracking(ball_xy, players_slots, to_metres=(1.0, 1.0), t=None, mask=None):
    """Clean one match. Returns dict with the same keys as atd.load_tactical (ball_xy, ball_present,
    players_slots, n_present) plus "report" (aggregates) and "stats" (spike length / jump distance
    samples, used to calibrate the degraded bank).

    t     (N,) frame numbers (default arange(N)); gaps in `t` (half-time) are handled by dt.
    mask  (N,) bool, frames over which the report percentages are computed (default all)."""
    scale = np.asarray(to_metres, float)
    N = len(ball_xy)
    t = np.arange(N) if t is None else np.asarray(t)
    mask = np.ones(N, bool) if mask is None else np.asarray(mask, bool)
    ball = ball_xy.astype(float).copy() * scale
    teams = {k: v.astype(float).copy() * scale for k, v in players_slots.items()}
    stats = {}
    rep = {"frames": int(mask.sum())}

    # rule 1
    b3 = ball[:, None, :]
    n_ball = int((~np.isnan(ball[mask, 0])).sum())
    sp, tp = _remove_jumps(b3, t, V_BALL, TOL_BALL, SPIKE_MAX_BALL, stats, "ball")
    ball = b3[:, 0]
    rep["ball"] = {"detections": n_ball,
                   "spike": _agg(sp, mask, n_ball), "teleport": _agg(tp, mask, n_ball)}
    for team, s in teams.items():
        n_det = int((~np.isnan(s[mask, :, 0])).sum())
        sp, tp = _remove_jumps(s, t, V_PLAYER, TOL_PLAYER, SPIKE_MAX_PLAYER, stats, "player")
        run = _run_length(~np.isnan(s[:, :, 0]))
        dup = _dedupe(s, run)
        cap = _cap(s, run)
        rep[team] = {"detections": n_det, "spike": _agg(sp, mask, n_det), "teleport": _agg(tp, mask, n_det),
                     "duplicate": _agg(dup, mask, n_det), "over_11": _agg(cap, mask, n_det)}
    out = {
        "ball_xy": ball / scale,
        "ball_present": ~np.isnan(ball[:, 0]),
        "players_slots": {k: v / scale for k, v in teams.items()},
        "n_present": {k: (~np.isnan(v[:, :, 0])).sum(1) for k, v in teams.items()},
        "report": rep,
        "stats": stats,
    }
    return out


def _agg(m, mask, n_det):
    """frames_pct: % of frames with at least one detection hit; det_pct: % of detections hit."""
    mm = m[mask]
    hit = mm.reshape(len(mm), -1).any(1)
    return {"frames_pct": 100 * float(hit.mean()), "det_pct": 100 * float(mm.sum()) / max(n_det, 1),
            "n": int(mm.sum())}


def clean_atd(d=None, mask=None):
    """Convenience: ATD dict from src/atd.py -> cleaned dict (normalised coordinates)."""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import atd
    d = d or atd.load_tactical()
    return clean_tracking(d["ball_xy"], d["players_slots"], to_metres=atd.PITCH, mask=mask)


def live_mask():
    """Frames of the two halves of BAR-ATH (from challenge/games.py): kickoff to final whistle of
    each half. 158,857 frames; the other 40,462 are pre-kickoff, half-time and post-match."""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import atd
    g = atd.GAMES[atd.GAME]
    m = np.zeros(atd.N_FRAMES, bool)
    for h in (1, 2):
        lo, hi = g["halves"][h]
        m[lo - g["offsets"][h]: hi - g["offsets"][h] + 1] = True
    return m


def format_report(rep):
    rows = ["rule                 frames_pct  det_pct      n"]
    for who in ("ball", "H", "A"):
        for rule in ("spike", "teleport", "duplicate", "over_11"):
            if rule in rep[who]:
                r = rep[who][rule]
                rows.append("{:<4} {:<15} {:>9.2f} {:>8.3f} {:>6d}".format(who, rule, r["frames_pct"], r["det_pct"], r["n"]))
    return "\n".join(rows)


if __name__ == "__main__":
    live = live_mask()
    for name, m in (("live frames", live), ("all frames", None)):
        r = clean_atd(mask=m)
        print("== ATD limpio, {} ({} frames) ==".format(name, r["report"]["frames"]))
        print(format_report(r["report"]))
        print("detections: ball {} H {} A {}".format(r["report"]["ball"]["detections"],
                                                     r["report"]["H"]["detections"], r["report"]["A"]["detections"]))
