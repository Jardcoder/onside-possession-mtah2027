"""`label_aerial`: stress ground truth, derived only from the PFF events of the development matches.

    from gt_aerial import build, stats
    label_aerial, info = build(mid, raw)      # raw = pff_split.load_dev(mid), full length (all periods)

`label` (PFF last touch) gives the ball to the team of the player of each game event until the next touch. Two kinds of
contact are won by a team without that team getting a game event of its own in time, so `label` keeps the rival's team
while the winner has just touched the ball. `label_aerial` hands the ball to the team that wins the contact, while the
contact lasts. It is a stress test (no veto): it models a reading of the challenge in which a contact won
without control moves the owner. It is NOT the challenge's own definition (CHALLENGE.md excludes touches without control).

What the loader gives (checked 2026-10-29): `is_touch` (frame of the start of each touch game event), `label`, `label_poss`,
`period`, `frame`, `uncertain`; no event type, no event team, no duel result. Those live in the raw Event Data file of the
match (possession event CH, `challengeType`, `challengeWinnerPlayerId`, `challengeOutcomeType`, `eventTime`) and in the
roster (player -> team); this module reads them, for development matches only (`pff_split.DESARROLLO`).

Contact (one per challenge possession event with a known winner whose team is known), fixed before any score was looked at:
  * every aerial duel (`challengeType` A) except a foul (outcome F: the ball is dead, the winner is the fouled player);
  * every other challenge whose outcome is P ("ball goes to a player not involved in the challenge": the winner touched it
    and did not keep it: a contact won without control).
  Not included, and why: outcomes R and D keep the ball with the winner (control, `label` already follows with the next
  touch); K keeps it with the carrier; O and F end in a stop; deflections and keeper touches have no winner and no time of
  their own (pff.py), and a challenge of unknown winner has no team.

Time of the contact: `eventTime` (video seconds) -> output frame `round((t - t0_period) * 25)`, the same clock the loader
resamples with (t0_period = video time of the first native frame of the period). Checked against the loader: for aerial
duels the start of the game event falls 0.04 s after it (median), i.e. the clock agrees.

Interval "while the contact lasts": from the contact frame j up to, excluding, the first frame k > j with `is_touch` (the next
touch event of PFF, where `label` changes anyway) or j + CAP_FRAMES, whichever comes first; never across a period. Inside it
every frame labelled H or A of the other team becomes the winner's team; D frames stay D. Contacts are applied in the order
of their frame (a later one overwrites an earlier one where they overlap). CAP_FRAMES = 25 (1.0 s) is a single choice made
before measuring and is not tuned.
"""
import json
import os
import sys

import numpy as np

import pff_split
import onside_paths

CAP_FRAMES = 25
AERIAL = "A"
NO_CONTROL_OUTCOME = "P"
FOUL = "F"


def _pff():
    """The PFF loader module (its private clock and roster helpers); path handled by pff_split."""
    if pff_split.LOADER_DIR not in sys.path:
        sys.path.insert(0, pff_split.LOADER_DIR)
    import pff
    return pff


def is_contact(ctype, outcome):
    """True for a challenge that the definition above turns into a contact (the winner must be known by the caller)."""
    return (ctype == AERIAL and outcome != FOUL) or outcome == NO_CONTROL_OUTCOME


def contact_events(mid):
    """[(period, video time s, winner "H"/"A", challengeType, outcome)] of the development match `mid`, event order."""
    if mid not in pff_split.DESARROLLO:
        raise PermissionError(f"{mid} is not a development match")
    pff = _pff()
    side = pff._sides(mid)
    out = []
    for e in json.load(open(os.path.join(pff.RAW, "Event Data", f"{mid}.json"))):
        pe = e["possessionEvents"]
        if pe["possessionEventType"] != "CH" or pe["challengeWinnerPlayerId"] is None:
            continue
        t = e["eventTime"] if e["eventTime"] is not None else e["startTime"]
        team = side.get(int(pe["challengeWinnerPlayerId"]))
        if t is None or team not in ("H", "A") or not is_contact(pe["challengeType"], pe["challengeOutcomeType"]):
            continue
        out.append((int(e["gameEvents"]["period"]), float(t), team, pe["challengeType"], pe["challengeOutcomeType"]))
    return out


def apply_contacts(label, period, is_touch, contacts, cap=CAP_FRAMES):
    """Pure core: copy of `label` with the contacts [(frame j, winner team)] applied (see the module docstring)."""
    out = np.array(label, copy=True)
    period = np.asarray(period)
    touch_f = np.flatnonzero(np.asarray(is_touch))
    n = len(out)
    for j, team in sorted(contacts, key=lambda c: c[0]):
        if not 0 <= j < n:
            continue
        nxt = touch_f[np.searchsorted(touch_f, j, side="right"):]
        end = min(int(nxt[0]) if len(nxt) else n, j + cap, n)
        end = j + int(np.argmax(period[j:end] != period[j])) if (period[j:end] != period[j]).any() else end
        seg = out[j:end]
        seg[(seg != "D") & (seg != team)] = team
    return out


def build(mid, raw):
    """(label_aerial (N,) '<U1', info) for the development match `mid`; `raw` = pff_split.load_dev(mid), full length."""
    pff = _pff()
    z = pff._tracking(mid)
    per = np.asarray(raw["period"])
    t0 = {int(p): float(z["vt"][z["period"] == p][0]) for p in np.unique(z["period"])}
    first = {p: int(np.flatnonzero(per == p)[0]) for p in t0}
    contacts, kinds = [], {}
    for p, t, team, ctype, outcome in contact_events(mid):
        if p not in t0:
            continue
        j = first[p] + int(round((t - t0[p]) * pff.FPS))
        if 0 <= j < len(per) and per[j] == p:
            contacts.append((j, team))
            kinds[(ctype, outcome)] = kinds.get((ctype, outcome), 0) + 1
    lab = apply_contacts(raw["label"], per, raw["is_touch"], contacts)
    return lab, {"contacts": len(contacts), "kinds": kinds}


def spells(lab, period):
    """[(start, stop, team)] of the maximal runs of constant H or A (D and a period change cut them)."""
    lab, period = np.asarray(lab), np.asarray(period)
    if not len(lab):
        return []
    cut = np.concatenate(([0], np.flatnonzero((lab[1:] != lab[:-1]) | (period[1:] != period[:-1])) + 1, [len(lab)]))
    return [(int(a), int(b), str(lab[a])) for a, b in zip(cut[:-1], cut[1:]) if lab[a] in ("H", "A")]


def stats(label, aerial, period):
    """Frames and possessions that `aerial` changes with respect to `label` (arrays clipped to the periods that are scored):
    frames changed, possessions of `label` that contain a changed frame, possessions of each GT, and contact windows
    (maximal runs of changed frames)."""
    label, aerial = np.asarray(label), np.asarray(aerial)
    ch = label != aerial
    sp = spells(label, period)
    touched = sum(1 for a, b, _ in sp if ch[a:b].any())
    runs = int((np.diff(np.concatenate(([0], ch.astype(np.int8)))) == 1).sum())
    return {"frames": int(ch.sum()), "total": len(label), "poss_label": len(sp), "poss_aerial": len(spells(aerial, period)),
            "poss_changed": touched, "windows": runs}


def main():
    """Per-match and total effect of `label_aerial` over `label` (periods 1-2, the ones that are scored); writes the report."""
    rows, tot = [], {}
    for mid in pff_split.DESARROLLO:
        raw = pff_split.load_dev(mid)
        lab, info = build(mid, raw)
        keep = np.asarray(raw["period"]) <= 2
        s = stats(np.asarray(raw["label"])[keep], lab[keep], np.asarray(raw["period"])[keep])
        s["contacts"] = info["contacts"]
        rows.append((mid, s))
        for k, v in s.items():
            tot[k] = tot.get(k, 0) + v
    out = ["label_aerial vs label, 40 development matches, periods 1-2 (definition in bancos/gt_aerial.py, cap "
           f"{CAP_FRAMES} frames).",
           f"{'partido':<8}{'contactos':>10}{'ventanas':>9}{'frames':>8}{'% frames':>9}{'posesiones label':>17}{'label_aerial':>13}"
           f"{'label con cambio':>17}"]
    for mid, s in rows + [("total", tot)]:
        out.append(f"{mid:<8}{s['contacts']:>10}{s['windows']:>9}{s['frames']:>8}{100 * s['frames'] / s['total']:>9.2f}"
                   f"{s['poss_label']:>17}{s['poss_aerial']:>13}{s['poss_changed']:>17}")
    out.append(f"posesiones de `label` que cambian: {100 * tot['poss_changed'] / tot['poss_label']:.1f} %; posesiones totales "
               f"{tot['poss_label']} -> {tot['poss_aerial']} ({tot['poss_aerial'] - tot['poss_label']:+d})")
    text = "\n".join(out)
    print(text)
    path = os.path.join(onside_paths.OUT_DIR, "label_aerial_informe.txt")
    with open(path, "w") as f:
        f.write(text + "\n")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
