"""Hash check of the submission (scripts/verify_submission.py): the comparison itself, and the end to end regeneration."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import verify_submission as vs  # noqa: E402

CHALLENGE = Path(os.environ.get("ONSIDE_CHALLENGE_DIR") or ROOT / "challenge")


def test_expected_hash_is_a_sha256_of_the_tracking_csv():
    line = vs.EXPECTED.read_text().split()
    assert len(line[0]) == 64 and int(line[0], 16) >= 0
    assert line[1] == "submission_tracking.csv"


def test_report_matches_only_the_expected_bytes(tmp_path, monkeypatch):
    good, bad = tmp_path / "good.csv", tmp_path / "bad.csv"
    good.write_bytes(b"x")
    bad.write_bytes(b"y")
    monkeypatch.setattr(vs, "expected_sha256", lambda path=vs.EXPECTED: vs.sha256(good))
    assert vs.report(good) == 0
    assert vs.report(bad) == 1


def test_frozen_assets_match_their_checksums():
    sums = (ROOT / "models/submission_a2p/SHA256SUMS").read_text().splitlines()
    assert sums
    for line in sums:
        digest, name = line.split(maxsplit=1)
        assert vs.sha256(ROOT / "models/submission_a2p" / name) == digest, name


@pytest.mark.skipif(os.environ.get("ONSIDE_VERIFY_SUBMISSION") != "1" or not (CHALLENGE / "data/tactical/BAR-ATH_tactical_atd.txt").exists(),
                    reason="set ONSIDE_VERIFY_SUBMISSION=1 and provide the challenge data to regenerate the CSV (same as ./run.sh verify)")
def test_regenerated_csv_matches_the_expected_hash():
    assert subprocess.run([sys.executable, str(ROOT / "scripts/verify_submission.py")], env=os.environ).returncode == 0
