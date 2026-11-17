"""label_aerial (bancos/gt_aerial.py): the pure core, the definition of a contact, the guard on held-out matches."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "bancos"))
import gt_aerial as ga  # noqa: E402
import onside_paths  # noqa: E402
import pff_split  # noqa: E402

needs_lab = pytest.mark.skipif(not os.path.isdir(pff_split.META_DIR) or not os.path.isdir(os.path.join(onside_paths.PFF_DIR, "Event Data")),
                               reason="PFF data not available (set ONSIDE_PFF_DIR)")


def lab(*parts):
    return np.array(list("".join(k * n for k, n in parts)))


def touches(n, *at):
    t = np.zeros(n, bool)
    t[list(at)] = True
    return t


def test_contact_definition_aerial_without_foul_or_any_challenge_with_outcome_p():
    assert ga.is_contact("A", "D") and ga.is_contact("A", "R") and ga.is_contact("A", "P")
    assert not ga.is_contact("A", "F")                                   # foul: dead ball
    assert ga.is_contact("T", "P") and ga.is_contact("FIFTY", "P") and ga.is_contact("B", "P")
    assert not ga.is_contact("T", "R") and not ga.is_contact("D", "D") and not ga.is_contact("FIFTY", "O")
    assert not ga.is_contact("T", None) and not ga.is_contact(None, None)


def test_contact_lasts_until_the_next_touch_event_when_it_comes_first():
    x = lab(("H", 100), ("A", 100))                                      # PFF: H until the touch of A at 100
    out = ga.apply_contacts(x, np.ones(200, int), touches(200, 100), [(90, "A")])
    assert (out[:90] == "H").all() and (out[90:] == "A").all()           # contact to next touch (10 frames < cap), then A anyway


def test_contact_is_capped_when_the_next_touch_is_far_and_it_reverts_to_the_label():
    x = lab(("H", 300))
    out = ga.apply_contacts(x, np.ones(300, int), touches(300, 250), [(100, "A")], cap=25)
    assert (out[100:125] == "A").all() and (out[:100] == "H").all() and (out[125:] == "H").all()
    assert (ga.apply_contacts(x, np.ones(300, int), touches(300), [(100, "A")], cap=25) == out).all()   # no touch after it: same cap


def test_a_touch_at_the_contact_frame_does_not_end_it_but_the_next_one_does():
    x = lab(("H", 100))
    out = ga.apply_contacts(x, np.ones(100, int), touches(100, 40, 45), [(40, "A")])
    assert (out[40:45] == "A").all() and (out[45:] == "H").all()


def test_dead_frames_period_changes_and_own_team_frames_are_untouched():
    x = lab(("H", 30), ("D", 10), ("A", 10), ("H", 50))
    per = np.array([1] * 60 + [2] * 40)
    out = ga.apply_contacts(x, per, touches(100), [(20, "A")], cap=60)
    assert (out[20:30] == "A").all() and (out[30:40] == "D").all() and (out[40:50] == "A").all()    # D stays D, A stays A
    assert (out[50:60] == "A").all() and (out[60:] == "H").all()                                       # the next period is not reached
    assert (x == lab(("H", 30), ("D", 10), ("A", 10), ("H", 50))).all()                                # input untouched


def test_contacts_are_applied_in_frame_order_and_ones_out_of_range_are_ignored():
    x = lab(("H", 100))
    out = ga.apply_contacts(x, np.ones(100, int), touches(100), [(50, "H"), (10, "A"), (500, "A"), (-3, "A")], cap=40)
    assert (out[10:50] == "A").all() and (out[:10] == "H").all() and (out[50:] == "H").all()


def test_stats_counts_frames_possessions_and_windows():
    a = lab(("H", 40), ("A", 40), ("D", 5), ("H", 40))
    b = a.copy()
    b[10:20] = "A"
    b[85:95] = "A"
    per = np.ones(len(a), int)
    s = ga.stats(a, b, per)
    assert s["frames"] == 20 and s["windows"] == 2 and s["poss_changed"] == 2
    assert s["poss_label"] == 3 and s["poss_aerial"] == 6                 # H A H | A | D | A H : the first window splits the first H, the second one cuts the last H after the D
    assert ga.stats(a, a, per)["frames"] == 0


@needs_lab
def test_events_are_read_only_for_development_matches():
    with pytest.raises(PermissionError):
        ga.contact_events(pff_split.RESERVADOS[0])
    ev = ga.contact_events(pff_split.DESARROLLO[0])
    assert ev and all(t in ("H", "A") and p in (1, 2, 3, 4) and ga.is_contact(c, o) for p, _, t, c, o in ev)


@needs_lab
def test_build_on_a_development_match_only_changes_live_frames_toward_the_winner():
    mid = pff_split.DESARROLLO[0]
    raw = pff_split.load_dev(mid)
    out, info = ga.build(mid, raw)
    base = np.asarray(raw["label"])
    ch = out != base
    assert out.shape == base.shape and 0 < ch.sum() < 0.02 * len(base) and info["contacts"] > 0
    assert (base[ch] != "D").all() and (out[ch] != "D").all()             # a changed frame was live and stays live
    assert np.array_equal(out[~ch], base[~ch])
