"""Compare numerical training results and prediction bytes across fresh processes."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from onside_paths import CHALLENGE_DIR


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_training_and_prediction_repeat_in_independent_processes(device: str, tmp_path: Path) -> None:
    if not (Path(CHALLENGE_DIR) / "games.py").is_file():
        pytest.skip("The public challenge evaluator code is required for the CSV writer")
    if device != "cpu" and not torch.cuda.is_available():
        pytest.skip("No GPU is available for the independent-process check")
    results = []
    for name, hash_seed, training_seed in (("first", "12", 7), ("second", "91", 7), ("control", "12", 8)):
        output = tmp_path / name
        process = subprocess.run([sys.executable, str(ROOT / "tests/reproducibility_worker.py"),
                                  str(output), device, str(training_seed)],
                                 env={**os.environ, "PYTHONHASHSEED": hash_seed},
                                 capture_output=True, text=True, timeout=60)
        assert process.returncode == 0, process.stdout + process.stderr
        results.append(json.loads((output / "result.json").read_text()))
    assert results[0] == results[1]
    assert results[0]["tensor_sha256"] != results[2]["tensor_sha256"]
    assert results[0]["runtime"]["deterministic"] is True
    assert results[0]["runtime"]["warn_only"] is False
    assert (tmp_path / "first/prediction.csv").read_bytes() == (tmp_path / "second/prediction.csv").read_bytes()
