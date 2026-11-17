"""Synthetic checks of bancos/balon_yolo.py (projection, tracklets, static-object veto, bridging)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("bancos", "src"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import balon_yolo as by  # noqa: E402

PITCH = (105.0, 68.0)


def test_to_pitch_inverts_the_homography():
    H = np.array([[1000.0, 50.0, 400.0], [10.0, 600.0, 200.0], [0.0, 0.1, 1.0]])
    pts = np.array([[0.2, 0.3], [0.8, 0.9]])
    q = (H @ np.c_[pts, np.ones(2)].T).T
    px = q[:, :2] / q[:, 2:]
    cand = np.c_[px, [0.9, 0.5]][None]
    np.testing.assert_allclose(by.to_pitch(cand, H[None])[0], pts, atol=1e-9)


def test_inside_margin():
    xy = np.array([[0.5, 0.5], [-0.02, 0.5], [-0.05, 0.5]])
    np.testing.assert_array_equal(by.inside(xy, PITCH, 3.0), [True, True, False])


def test_tracklets_follow_motion_and_split_on_jumps():
    frames = np.arange(6)
    xy = np.zeros((6, 1, 2))
    xy[:, 0, 0] = [0, 1, 2, 3, 30, 31]                    # 1 m per frame, then a 27 m jump
    tid = by.build_tracklets(frames, xy, np.ones((6, 1), bool), by.PARAMS)
    assert len(set(tid[:4, 0])) == 1 and tid[4, 0] != tid[3, 0] and tid[5, 0] == tid[4, 0]


def test_static_image_object_is_dropped():
    tab = {"n": np.array([50, 50, 3]), "mean_conf": np.array([0.6, 0.6, 0.9]), "px_spread": np.array([1.0, 30.0, 5.0]),
           "pitch_spread": np.array([10.0, 10.0, 0.5])}
    np.testing.assert_array_equal(by.keep_mask(tab, by.PARAMS), [False, True, False])


def test_bridge_fills_short_holes_only():
    xy = np.array([[0, 0], [np.nan, np.nan], [2, 2], [np.nan, np.nan]] + [[np.nan, np.nan]] * 6 + [[9, 9]], float)
    out = by.bridge(xy, 2)
    np.testing.assert_allclose(out[1], [1, 1])
    assert np.isnan(out[3:10, 0]).all()
