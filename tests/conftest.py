"""Test setup: the challenge modules come from ONSIDE_CHALLENGE_DIR (src/onside_paths.py).

Modules that need the challenge clone or the PFF files are skipped, not failed, when those are not there."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import onside_paths  # noqa: E402

onside_paths.add_challenge_to_path()
CHALLENGE_THERE = os.path.isfile(os.path.join(onside_paths.EVALUATION_DIR, "possession_metrics.py"))


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    """A test module that cannot be imported because the challenge clone is missing is reported as skipped."""
    outcome = yield
    rep = outcome.get_result()
    if rep.failed and not CHALLENGE_THERE and isinstance(collector, pytest.Module):
        rep.outcome = "skipped"
        rep.longrepr = (str(collector.path), 0, "Skipped: challenge repository not found (set ONSIDE_CHALLENGE_DIR, see README.md)")


# ---------------------------------------------------------------- tests that need the real inputs
def _pff_there():
    """All the PFF tracking files (the loader hash reads the 40 development matches)."""
    d = os.path.join(onside_paths.PFF_DIR, "Tracking Data")
    return os.path.isdir(d) and sum(n.endswith(".jsonl.bz2") for n in os.listdir(d)) >= 64


def _calibration_there():
    """Aggregates that the T1 fingerprints hash (./run.sh regimen, ./run.sh calibrar-largo)."""
    return (os.path.isfile(os.path.join(onside_paths.OUT_DIR, "regimen_reto.csv"))
            and os.path.isfile(os.path.join(onside_paths.BANKS_DIR, "cache", "huecos_largo_atd.json")))


def pytest_configure(config):
    config.addinivalue_line("markers", "needs_pff: needs the PFF files under ONSIDE_PFF_DIR")
    config.addinivalue_line("markers", "needs_pff_and_calibration: needs the PFF files and the calibration aggregates")


def pytest_runtest_setup(item):
    if item.get_closest_marker("needs_pff") and not _pff_there():
        pytest.skip("PFF files not found (set ONSIDE_PFF_DIR, see README.md)")
    if item.get_closest_marker("needs_pff_and_calibration") and not (_pff_there() and _calibration_there()):
        pytest.skip("PFF files or calibration aggregates not found (README.md, 'Reproducing the training')")
