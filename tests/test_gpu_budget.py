"""Verify the allocator subtracts desktop VRAM and rejects a breached device budget."""

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
torch = pytest.importorskip("torch")
from gpu_budget import GpuBudget


def test_memory_budget_accounts_for_other_applications(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    device = tmp_path / "device"
    sensor = device / "hwmon/hwmon0"
    sensor.mkdir(parents=True)
    total, desktop = 16 * 1024**3, 3 * 1024**3
    (device / "mem_info_vram_total").write_text(str(total))
    (device / "mem_info_vram_used").write_text(str(desktop))
    (sensor / "power1_average").write_text("120000000")
    monkeypatch.setenv("ONSIDE_GPU_DEVICE_PATH", str(device))
    allocated = []
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda share, device: allocated.append(share))
    report = tmp_path / "resource.json"
    budget = GpuBudget(torch.device("cuda:0"), 0.7, report)
    assert allocated == [pytest.approx((int(total * 0.7) - desktop - 256 * 1024**2) / total)]
    assert json.loads(report.read_text())["sampled_peak_watts"] == 120.0
    (device / "mem_info_vram_used").write_text(str(int(total * 0.71)))
    with pytest.raises(RuntimeError, match="Total device VRAM exceeded"):
        budget.sample()
