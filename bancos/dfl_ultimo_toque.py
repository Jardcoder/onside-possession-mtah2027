"""Last-touch H/A/D labels for DFL derived from the event stream (independent of BallPossession).

Each touch event (pass, set-piece pass, clearance, isolated touch, duel winner, interception,
blocker of a shot) is aligned to the tracking frame where the acting player is closest to the
ball (+-3 s around EventTime, < 3 m). Between two touches the label is the team of the last
touch; D when BallStatus == 0; before the first touch of a period it falls back to
BallPossession. Misaligned events are dropped, so this is a second opinion, not ground truth.

    from dfl_ultimo_toque import last_touch_labels
    df = last_touch_labels("J03WPY")   # dfl.load_match columns + label_ut + n_touch_frame
CLI: .venv/bin/python dfl_ultimo_toque.py  -> agreement of label_ut with label per match
"""
import numpy as np, pandas as pd
import dfl
from dfl_analisis import collect


def last_touch_labels(mid, touches=None):
    d = dfl.load_match(mid).copy()
    T = touches if touches is not None else collect(mid)
    T = T[T.match == mid].sort_values("f0").drop_duplicates("f0", keep="first")
    cur = np.full(len(d), "?", dtype=object)
    cur[T.f0.values] = T.team.values
    s = pd.Series(cur).replace("?", np.nan)
    # ffill inside each period only
    s = s.groupby(d.period.values).ffill()
    fallback = np.where(d.ball_possession == 1, "H", "A")
    lab = np.where(s.isna(), fallback, s)
    lab = np.where(d.ball_status == 0, "D", lab)
    d["label_ut"] = lab
    d["is_touch_frame"] = False
    d.loc[T.f0.values, "is_touch_frame"] = True
    return d


if __name__ == "__main__":
    rows = []
    for m in dfl.MATCHES:
        d = last_touch_labels(m)
        alive = d.ball_status == 1
        rows.append(dict(match=m, touches=int(d.is_touch_frame.sum()),
                         agree_alive_pct=100 * (d.label_ut == d.label)[alive].mean(),
                         agree_all_pct=100 * (d.label_ut == d.label).mean()))
    print(pd.DataFrame(rows).round(1).to_string(index=False))
