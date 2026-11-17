"""Exercise the challenge-data barrier in a separate real Python process."""

from pathlib import Path
import subprocess
import sys


def test_training_guard_allows_external_files_and_rejects_challenge_reads(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "src"
    protected = tmp_path / "challenge"
    protected.mkdir()
    (protected / "tracking.txt").write_text("synthetic protected input")
    external = tmp_path / "external.txt"
    external.write_text("synthetic external input")
    alias = tmp_path / "alias.txt"
    alias.symlink_to(protected / "tracking.txt")
    code = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from training_data_guard import restrict_training_reads
restrict_training_reads((Path(sys.argv[2]),))
assert Path(sys.argv[3]).read_text() == 'synthetic external input'
for value in sys.argv[4:]:
    try:
        Path(value).read_bytes()
    except PermissionError:
        pass
    else:
        raise AssertionError('Challenge read was allowed')
print('external allowed; protected path and symlink denied')
"""
    result = subprocess.run([sys.executable, "-c", code, str(source), str(protected), str(external),
                             str(protected / "tracking.txt"), str(alias)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
