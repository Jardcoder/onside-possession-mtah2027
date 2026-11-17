"""Does DFL BallPossession follow 'last touch' or 'control'? Evidence from events.

Touch events are aligned to the tracking by the frame (within +-3 s of EventTime) where
the acting player is closest to the ball (needs < 3 m). Then, for events whose team was NOT
the BallPossession holder 0.5 s before the touch ("rival touches"), we measure whether
BallPossession flips to the toucher in the next 0.5 s and whether it flips back.
Run: .venv/bin/python dfl_analisis.py   (prints tables, writes data/cache/touches.parquet)
"""
import numpy as np, pandas as pd
import dfl

HZ = 25
CATS = {  # name -> (mask on events, actor column, team column)
    "pass": lambda e: (e.kind == "Play"),
    "set_piece_pass": lambda e: e.kind.isin(["ThrowIn", "FreeKick", "GoalKick", "CornerKick", "KickOff"]),
    "clearance": lambda e: (e.kind == "OtherBallAction") & e.clearance,
    "isolated_touch": lambda e: (e.kind == "OtherBallAction") & ~e.clearance,
    "header_duel_winner": lambda e: (e.kind == "TacklingGame") & (e.tackle_type == "air"),
    "ground_duel_winner": lambda e: (e.kind == "TacklingGame") & (e.tackle_type == "ground"),
    "intercepted_pass": lambda e: (e.kind == "BallClaiming") & (e.tackle_type == "InterceptedBall"),
    "blocked_shot_blocker": lambda e: e.blocker.notna(),
}


def actor_of(e, cat):
    if cat == "blocked_shot_blocker":
        return e.blocker_shirt, e.blocker_team
    if cat in ("pass", "set_piece_pass"):
        return e.play_player_shirt, e.play_team
    return e.player_shirt, e.team


def collect(mid):
    d = dfl.load_match(mid)
    e = dfl.load_events(mid)
    e = e[~e.deleted]
    bp = d.ball_possession.values
    lab = d.label.values
    per = d.period.values
    bx, by = d.ball_x.values, d.ball_y.values
    t = d.t_utc.values
    rows = []
    for cat, f in CATS.items():
        sub = e[f(e)]
        shirt, team = actor_of(sub, cat)
        for (i, r), sh, tm in zip(sub.iterrows(), shirt, team):
            if pd.isna(sh) or tm is None or (isinstance(tm, float) and np.isnan(tm)):
                continue
            col = f"{tm.lower()}{int(sh)}"
            if col + "_x" not in d.columns:
                continue
            k = np.searchsorted(t, np.datetime64(r.t_utc))
            lo, hi = max(k - 3 * HZ, 0), min(k + 3 * HZ, len(d) - 1)
            px, py = d[col + "_x"].values[lo:hi + 1], d[col + "_y"].values[lo:hi + 1]
            dist = np.hypot(px - bx[lo:hi + 1], py - by[lo:hi + 1])
            if np.all(np.isnan(dist)):
                continue
            j = lo + int(np.nanargmin(dist))
            if dist[j - lo] > 3.0 or j < 20 or j + 100 > len(d) or per[j - 20] != per[j + 100]:
                continue
            want = 1 if tm == "H" else 2
            rows.append(dict(match=mid, cat=cat, event_id=r.event_id, pchange=r.possession_change,
                             wres=r.winner_result, evaluation=r.evaluation, tackle_type=r.tackle_type, ev_row=k, f0=j, dist=dist[j - lo], team=tm,
                             lag_s=(j - k) / HZ, bp_before=bp[j - 13], bp_at=bp[j], want=want,
                             alive=d.ball_status.values[j], d_after=[lab[j + n] for n in (0,)][0],
                             # first frame in [j, j+1.5 s] where BP == toucher, and flips back
                             first_flip=_first(bp[j:j + 38] == want),
                             flip_back=_first((bp[j:j + 76] != want) & (np.arange(76) > _first(bp[j:j + 38] == want)))
                             if _first(bp[j:j + 38] == want) >= 0 else -1))
    return pd.DataFrame(rows)


def _first(mask):
    w = np.flatnonzero(mask)
    return int(w[0]) if len(w) else -1


def runs(arr):
    """(value, length) of consecutive equal values."""
    ch = np.flatnonzero(arr[1:] != arr[:-1]) + 1
    st = np.r_[0, ch]
    en = np.r_[ch, len(arr)]
    return arr[st], en - st


def possession_runs(mid):
    d = dfl.load_match(mid)
    out = []
    for p in (1, 2):
        dp = d[d.period == p]
        v, n = runs(dp.ball_possession.values)
        out.append(pd.DataFrame({"match": mid, "period": p, "who": v, "frames": n, "alive_frames": 0}))
        # alive only: drop dead frames before computing runs
        a = dp[dp.ball_status == 1].ball_possession.values
        v2, n2 = runs(a)
        out.append(pd.DataFrame({"match": mid, "period": p, "who": v2, "frames": n2, "alive_frames": 1}))
    return pd.concat(out)


if __name__ == "__main__":
    T = pd.concat([collect(m) for m in dfl.MATCHES])
    T.to_parquet(dfl.CACHE / "touches.parquet")
    print("aligned touches per cat:\n", T.cat.value_counts().to_string())
    print("\nalignment lag (s) EventTime->closest-frame: median %.2f, IQR %.2f..%.2f" % (
        T.lag_s.median(), T.lag_s.quantile(.25), T.lag_s.quantile(.75)))
    # rival touches: toucher not the BP holder 0.5 s before and ball alive
    R = T[(T.bp_before != T.want) & (T.alive == 1)]
    def summ(g):
        n = len(g)
        ff = g.first_flip
        return pd.Series({
            "n_rival": n,
            "BP==toucher at touch frame %": 100 * (g.bp_at == g.want).mean(),
            "flip within 0.5s %": 100 * ((ff >= 0) & (ff <= 12)).mean(),
            "flip within 1.5s %": 100 * (ff >= 0).mean(),
            "median flip delay s": (ff[ff >= 0] / HZ).median(),
            "flipped back within 1s of flip %": 100 * ((g.flip_back >= 0) & (g.flip_back - ff <= HZ) & (ff >= 0)).sum() / max((ff >= 0).sum(), 1)})
    S = R.groupby("cat").apply(summ)
    S.loc["ALL"] = summ(R)
    print("\nRIVAL TOUCHES (toucher's team != BallPossession 0.5s before)\n", S.round(1).to_string())
    # possession durations
    P = pd.concat([possession_runs(m) for m in dfl.MATCHES])
    bins = [0, .5, 1, 2, 5, 10, 1e9]
    names = ["<0.5s", "0.5-1s", "1-2s", "2-5s", "5-10s", ">10s"]
    for alive in (0, 1):
        q = P[P.alive_frames == alive]
        sec = q.frames / HZ
        print(f"\nBallPossession run durations (alive_only={bool(alive)}), n={len(q)}")
        print(pd.cut(sec, bins, labels=names, right=False).value_counts().reindex(names).to_string())
    # sandwiches: X, Y(<1s), X
    cnt = tot = 0
    for (m, p, a), g in P[P.alive_frames == 0].groupby(["match", "period", "alive_frames"], sort=False):
        w, n = g.who.values, g.frames.values
        for i in range(1, len(w) - 1):
            tot += 1
            if n[i] < HZ and w[i - 1] == w[i + 1] and w[i] != w[i - 1]:
                cnt += 1
    print(f"\nSandwiched runs <1s (X,Y<1s,X) on raw BallPossession: {cnt} of {tot} interior runs")
