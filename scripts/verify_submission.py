"""Compare the SHA-256 of the submission CSV with the expected one (scripts/expected_sha256.txt).

    ./run.sh verify [CSV]

Without CSV it regenerates the file with scripts/submission_a2p.py (needs ONSIDE_CHALLENGE_DIR) into a temporary folder and
compares that. With CSV it only compares the given file. Prints the two hashes only (no per frame values); exit code 0 on a match.
"""
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ROOT / "scripts" / "expected_sha256.txt"


def expected_sha256(path: Path = EXPECTED) -> str:
    return path.read_text().split()[0]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(argv: list[str]) -> int:
    if argv:
        csv = Path(argv[0])
        return report(csv)
    with tempfile.TemporaryDirectory() as folder:
        csv = Path(folder) / "submission_tracking.csv"
        subprocess.run([sys.executable, str(ROOT / "scripts" / "submission_a2p.py"), "--output", str(csv)], check=True)
        return report(csv)


def report(csv: Path) -> int:
    got, want = sha256(csv), expected_sha256()
    print(f"expected {want}\nobtained {got}\n{'MATCH' if got == want else 'MISMATCH'}")
    return 0 if got == want else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
