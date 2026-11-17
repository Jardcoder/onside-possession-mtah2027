"""M1: per-frame LightGBM classifier {H, A, D} plus a Viterbi decoder. Method side only.

Everything here reads tracking arrays and the labels / touches of the rule-based methods (B3, B3 + absorption); no ground truth
enters a feature. The labels the classifier learns from are passed by the caller to `augment` / `fit`, never read here.

    build_features(ball, slots, per, pitch, b3, b3a, tf) -> (X float32 (N, F), names)
    swap_features(X, names)                              team swap H <-> A on the features
    augment(X, y)                                        both views (original + swapped, labels swapped) for training
    fit(...) / predict_proba(booster, X)                 LightGBM, inference = mean of p(x) and swap(p(swap(x)))
    viterbi(logp, per, lam_ha, lam_d)                    3 states {H, A, D}, per period

Inputs (as in the rest of the method side): ball (N, 2) normalised 0-1 with NaN, slots {"H": (N, S, 2), "A": (N, S', 2)} normalised
with NaN, per (N,), pitch (length_m, width_m), b3 / b3a (N,) "H"/"A"/"D" (B3 and B3 + absorption), tf = frames of the touches of B3.
Time is in seconds, converted with toques.FPS. Frame order is kept: the labelling is offline, so windows look into the future too.

Team symmetry. Feature names carry their role: `name__H` / `name__A` are a pair (they swap), `sgn__name` is +1 for H, -1 for A,
0 / NaN neutral (it changes sign), anything else is invariant. `swap_spec` derives the permutation and signs from the names.
Pitch geometry is not mirrored: a team swap keeps the pitch, so the swapped copy is a valid match with the teams renamed.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import toques  # noqa: E402

FPS = toques.FPS
WINDOWS_S = (0.2, 0.5, 1.0, 2.0)               # past and future window lengths, seconds
SPEED_HALF_S = 0.2                             # half-width of the central difference for velocities (0.4 s span)
CAP_S = 30.0                                   # cap of "time since / until" features
CODE = {"H": 0, "A": 1, "D": 2}
STATES = ("H", "A", "D")

LGB_PARAMS = {"objective": "multiclass", "num_class": 3, "num_leaves": 63, "learning_rate": 0.05, "feature_fraction": 0.8,
              "bagging_fraction": 0.5, "bagging_freq": 1, "min_data_in_leaf": 200, "max_bin": 127, "verbosity": -1,
              "seed": 20261028, "deterministic": True, "force_row_wise": True}
MAX_ROUNDS = 600
EARLY_STOP = 50


# ------------------------------------------------------------------------------------------------ helpers

def _period_bounds(per):
    """(lo, hi) per frame: first frame and one past the last frame of the frame's period."""
    per = np.asarray(per)
    n = len(per)
    cut = np.flatnonzero(per[1:] != per[:-1]) + 1
    starts = np.concatenate(([0], cut))
    stops = np.concatenate((cut, [n]))
    return np.repeat(starts, stops - starts), np.repeat(stops, stops - starts)


def _nanmean(a, axis):
    ok = np.isfinite(a)
    cnt = ok.sum(axis)
    s = np.where(ok, a, 0.0).sum(axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)


def _window_mean(x, lo, hi, w, future):
    """Mean of the finite values of x over the w frames before (excluding) or after (excluding) each frame, inside its period."""
    n = len(x)
    ok = np.isfinite(x)
    c = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    k = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(n)
    if future:
        a, b = np.minimum(i + 1, hi), np.minimum(i + 1 + w, hi)
    else:
        a, b = np.maximum(i - w, lo), i
    cnt = k[b] - k[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (c[b] - c[a]) / np.maximum(cnt, 1), np.nan)


def _since_until(mask, lo, hi):
    """Seconds since the last True (0 on it) and until the next True (0 on it), inside the period; NaN when there is none; cap CAP_S."""
    n = len(mask)
    i = np.arange(n)
    last = np.maximum.accumulate(np.where(mask, i, -1))
    nxt = np.minimum.accumulate(np.where(mask, i, n + 10 * n)[::-1])[::-1]
    since = np.where(last >= lo, (i - last) / FPS, np.nan)
    until = np.where(nxt < hi, (nxt - i) / FPS, np.nan)
    return np.minimum(since, CAP_S), np.minimum(until, CAP_S)


def _speed(p, lo, hi, h):
    """Speed (m/s) of each slot, central difference of +-h frames clipped to the period; p (N, S, 2) in metres."""
    n = len(p)
    i = np.arange(n)
    a, b = np.clip(i - h, lo, hi - 1), np.clip(i + h, lo, hi - 1)
    dt = np.maximum((b - a) / FPS, 1.0 / FPS)
    return np.linalg.norm(p[b] - p[a], axis=2) / dt[:, None]


class _Cols:
    """Ordered feature columns with their names; `window` adds the past / future means of a series."""

    def __init__(self, lo, hi):
        self.names, self.cols, self.lo, self.hi = [], [], lo, hi

    def add(self, name, arr):
        self.names.append(name)
        self.cols.append(np.asarray(arr, np.float32))

    def pair(self, base, h, a):
        self.add(f"{base}__H", h)
        self.add(f"{base}__A", a)

    def window(self, base, series, pair=None):
        """Past `@p<w>` and future `@f<w>` means of `series` over WINDOWS_S; `pair` = (series_H, series_A) for a team pair."""
        for w_s in WINDOWS_S:
            w = max(1, int(round(w_s * FPS)))
            for tag, fut in (("p", False), ("f", True)):
                if pair is None:
                    self.add(f"{base}@{tag}{w_s:g}", _window_mean(series, self.lo, self.hi, w, fut))
                else:
                    self.pair(f"{base}@{tag}{w_s:g}", *(_window_mean(s, self.lo, self.hi, w, fut) for s in pair))


# ------------------------------------------------------------------------------------------------ features

def build_features(ball, slots, per, pitch, b3, b3a, tf):
    """Per-frame features (X float32 (N, F), names) from the tracking arrays and the rule-based labels / touches only."""
    scale = np.asarray(pitch, float)
    length, width = scale
    per = np.asarray(per)
    n = len(per)
    lo, hi = _period_bounds(per)
    idx = np.arange(n)
    ball = np.asarray(ball, float)
    bm = ball * scale
    present = np.isfinite(bm[:, 0])
    C = _Cols(lo, hi)

    # period and clock
    C.add("period", per)
    C.add("t_in_period_s", (idx - lo) / FPS)
    C.add("t_to_period_end_s", (hi - 1 - idx) / FPS)

    # ball: presence, geometry, kinematics
    C.add("ball_present", present)
    C.add("ball_x_m", bm[:, 0])
    C.add("ball_y_m", bm[:, 1])
    C.add("ball_dist_end_m", np.minimum(bm[:, 0], length - bm[:, 0]))
    C.add("ball_dist_side_m", np.minimum(bm[:, 1], width - bm[:, 1]))
    C.add("ball_dist_goal_m", np.minimum(np.hypot(bm[:, 0], bm[:, 1] - width / 2), np.hypot(length - bm[:, 0], bm[:, 1] - width / 2)))
    C.add("ball_dist_halfway_m", np.abs(bm[:, 0] - length / 2))
    h = max(1, int(round(SPEED_HALF_S * FPS)))
    a, b = np.clip(idx - h, lo, hi - 1), np.clip(idx + h, lo, hi - 1)
    dt = np.maximum((b - a) / FPS, 1.0 / FPS)
    v = (bm[b] - bm[a]) / dt[:, None]                       # NaN where either end is missing
    speed = np.linalg.norm(v, axis=1)
    acc = np.linalg.norm(v[b] - v[a], axis=1) / dt          # change of velocity over the next / previous 0.4 s
    C.add("ball_speed_ms", speed)
    C.add("ball_vx_ms", v[:, 0])
    C.add("ball_vy_ms", v[:, 1])
    C.add("ball_acc_ms2", acc)
    since_v, until_v = _since_until(present, lo, hi)
    C.add("ball_since_visible_s", since_v)
    C.add("ball_until_visible_s", until_v)

    # players: distances to the ball, counts, speeds, centroids (per team)
    team = {}
    for key in ("H", "A"):
        p = np.asarray(slots[key], float) * scale
        d = np.linalg.norm(p - bm[:, None, :], axis=2)                   # NaN where the ball or the player is missing
        dinf = np.where(np.isfinite(d), d, np.inf)
        if dinf.shape[1] < 2:
            dinf = np.concatenate([dinf, np.full((n, 2 - dinf.shape[1]), np.inf)], axis=1)
        s2 = np.sort(dinf, axis=1)[:, :2]
        d1, d2 = np.where(np.isfinite(s2[:, 0]), s2[:, 0], np.nan), np.where(np.isfinite(s2[:, 1]), s2[:, 1], np.nan)
        near = np.argmin(dinf[:, :d.shape[1]], axis=1)
        spd = _speed(p, lo, hi, h)
        team[key] = {"d1": d1, "d2": d2,
                     "n2": np.where(present, (d < 2.0).sum(1), np.nan), "n5": np.where(present, (d < 5.0).sum(1), np.nan),
                     "msp": _nanmean(spd, 1), "nsp": np.where(np.isfinite(d1), spd[idx, near], np.nan),
                     "cx": _nanmean(p[:, :, 0], 1), "cy": _nanmean(p[:, :, 1], 1), "spd_all": spd}
    H, A = team["H"], team["A"]
    for nm in ("d1", "d2", "n2", "n5", "msp", "nsp"):
        C.pair(nm, H[nm], A[nm])
    C.pair("cen_dx_m", H["cx"] - bm[:, 0], A["cx"] - bm[:, 0])
    C.pair("cen_dy_m", H["cy"] - bm[:, 1], A["cy"] - bm[:, 1])
    sp_all = _nanmean(np.concatenate([H["spd_all"], A["spd_all"]], axis=1), 1)
    C.add("players_speed_ms", sp_all)
    dmin = np.fmin(H["d1"], A["d1"])
    C.add("dmin_m", dmin)
    d1diff = H["d1"] - A["d1"]
    C.add("sgn__d1_diff_m", d1diff)
    sgn_near = np.where(np.isfinite(d1diff), np.sign(d1diff) * -1.0, np.nan)      # +1 when H is the nearest team
    C.add("sgn__nearest", sgn_near)
    C.add("sgn__attack_side_m", H["cx"] - A["cx"])                              # which end H defends, from the centroids

    # rule-based labels and touches
    b3, b3a = np.asarray(b3), np.asarray(b3a)
    sg = lambda lab: np.where(lab == "H", 1.0, np.where(lab == "A", -1.0, 0.0))     # noqa: E731
    C.add("sgn__b3", sg(b3))
    C.add("b3_dead", b3 == "D")
    C.add("sgn__b3abs", sg(b3a))
    C.add("b3abs_dead", b3a == "D")
    tf = np.asarray(tf, int)
    t_team = b3[tf] if len(tf) else np.array([], dtype=b3.dtype)         # a touch hands the owner to its team at its frame
    th, ta = np.zeros(n), np.zeros(n)
    th[tf[t_team == "H"]] = 1.0
    ta[tf[t_team == "A"]] = 1.0
    since_t, until_t = _since_until((th + ta) > 0, lo, hi)
    C.add("touch_since_s", since_t)
    C.add("touch_until_s", until_t)
    last_sign, nxt_sign = np.full(n, np.nan), np.full(n, np.nan)
    if len(tf):
        ts = np.where(t_team == "H", 1.0, np.where(t_team == "A", -1.0, 0.0))
        j = np.searchsorted(tf, idx, side="right") - 1
        ok = (j >= 0) & (per[tf[np.maximum(j, 0)]] == per)
        last_sign = np.where(ok, ts[np.maximum(j, 0)], np.nan)
        k = np.searchsorted(tf, idx, side="left")
        ok = (k < len(tf)) & (per[tf[np.minimum(k, len(tf) - 1)]] == per)
        nxt_sign = np.where(ok, ts[np.minimum(k, len(tf) - 1)], np.nan)
    C.add("sgn__touch_last_team", last_sign)
    C.add("sgn__touch_next_team", nxt_sign)

    # windows over the past and the future (offline labelling)
    C.window("ball_present", present.astype(float))
    C.window("ball_speed", speed)
    C.window("dmin", dmin)
    C.window("players_speed", sp_all)
    C.window("b3_dead", (b3 == "D").astype(float))
    C.window("b3abs_dead", (b3a == "D").astype(float))
    C.window("d1", None, pair=(H["d1"], A["d1"]))
    C.window("n2", None, pair=(H["n2"], A["n2"]))
    C.window("touch", None, pair=(th, ta))
    C.window("sgn__nearest", sgn_near)
    C.window("sgn__b3", sg(b3))
    C.window("sgn__b3abs", sg(b3a))
    C.window("sgn__d1_diff", d1diff)
    return np.stack(C.cols, axis=1).astype(np.float32), list(C.names)


# ------------------------------------------------------------------------------------------------ team symmetry

def swap_spec(names):
    """(perm, sign): X_swapped = X[:, perm] * sign. Pairs `x__H` / `x__A` exchange, `sgn__x` change sign, the rest are invariant."""
    pos = {nm: i for i, nm in enumerate(names)}
    perm = np.arange(len(names))
    sign = np.ones(len(names), np.float32)
    for nm, i in pos.items():
        if nm.endswith("__H"):
            perm[i] = pos[nm[:-3] + "__A"]
        elif nm.endswith("__A"):
            perm[i] = pos[nm[:-3] + "__H"]
        elif nm.startswith("sgn__"):
            sign[i] = -1.0
    return perm, sign


def swap_features(X, names):
    perm, sign = swap_spec(names)
    return X[:, perm] * sign


def swap_labels(y):
    """Codes 0 = H, 1 = A, 2 = D with H and A exchanged."""
    y = np.asarray(y)
    return np.where(y == 2, 2, 1 - y)


def swap_probs(P):
    """(N, 3) probabilities of (H, A, D) with H and A exchanged."""
    return np.asarray(P)[:, [1, 0, 2]]


def augment(X, y, names):
    """Training set with both views: the rows as they are plus the team-swapped rows with swapped labels."""
    return np.concatenate([X, swap_features(X, names)]), np.concatenate([y, swap_labels(y)])


# ------------------------------------------------------------------------------------------------ classifier

def fit(X_tr, y_tr, X_va, y_va, threads=None, rounds=MAX_ROUNDS, params=None):
    """LightGBM multiclass with early stopping on (X_va, y_va); returns the Booster (best iteration kept)."""
    import lightgbm as lgb
    p = dict(LGB_PARAMS if params is None else params)
    if threads:
        p["num_threads"] = threads
    dtr = lgb.Dataset(X_tr, label=y_tr, free_raw_data=True)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr, free_raw_data=True)
    return lgb.train(p, dtr, num_boost_round=rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])


def predict_proba(booster, X, names):
    """Symmetric inference: mean of p(x) and swap(p(swap(x))). (N, 3) float64 (H, A, D)."""
    it = booster.best_iteration or None
    p1 = booster.predict(X, num_iteration=it)
    p2 = swap_probs(booster.predict(swap_features(X, names), num_iteration=it))
    return 0.5 * (p1 + p2)


# ------------------------------------------------------------------------------------------------ Viterbi

def _penalties(lam_ha, lam_d):
    lam_ha, lam_d = np.atleast_1d(np.asarray(lam_ha, float)), np.atleast_1d(np.asarray(lam_d, float))
    T = np.zeros((len(lam_ha), 3, 3))
    T[:, 0, 1] = T[:, 1, 0] = lam_ha
    for s in (0, 1):
        T[:, s, 2] = T[:, 2, s] = lam_d
    return T


def _viterbi_segment(logp, T):
    n, k = len(logp), len(T)
    score = np.repeat(logp[:1], k, axis=0)                  # (K, 3)
    back = np.zeros((n, k, 3), np.int8)
    for t in range(1, n):
        cand = score[:, :, None] - T                        # (K, from, to)
        arg = cand.argmax(1)
        back[t] = arg
        score = np.take_along_axis(cand, arg[:, None, :], 1)[:, 0, :] + logp[t]
    path = np.zeros((k, n), np.int8)
    ar = np.arange(k)
    path[:, -1] = score.argmax(1)
    for t in range(n - 1, 0, -1):
        path[:, t - 1] = back[t, ar, path[:, t]]
    return path


def viterbi(logp, per, lam_ha, lam_d):
    """Optimal paths over states 0 = H, 1 = A, 2 = D maximising sum(logp) - lam_ha * (H<->A changes) - lam_d * (entries and exits of D),
    decoded independently inside each period. lam_ha, lam_d: scalars or equal-length arrays (a grid is decoded in one pass).
    Returns (K, N) int8 (K = 1 for scalars)."""
    logp = np.asarray(logp, float)
    T = _penalties(lam_ha, lam_d)
    out = np.zeros((len(T), len(logp)), np.int8)
    if not len(logp):
        return out
    lo, hi = _period_bounds(per)
    for a in np.unique(lo):
        b = hi[a]
        out[:, a:b] = _viterbi_segment(logp[a:b], T)
    return out


def decode(P, per, lam_ha, lam_d, eps=1e-9):
    """Viterbi on log(P) -> (K, N) array of "H" / "A" / "D"."""
    return np.array(STATES)[viterbi(np.log(np.clip(P, eps, 1.0)), per, lam_ha, lam_d)]
