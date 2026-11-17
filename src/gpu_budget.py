"""Bound this process's GPU allocation and duty cycle; measure device consumption."""

import json
import os
from pathlib import Path
import time
from typing import Any

import torch


class GpuBudget:
    """Interface to PyTorch's allocator and the selected Linux DRM device."""

    def __init__(self, device: torch.device, share: float, report: Path) -> None:
        if not 0 < share <= 0.7:
            raise ValueError("GPU share must be in (0, 0.7]")
        self.device = device
        self.share = share
        self.report = report
        self.started = time.monotonic()
        self.samples = 0
        self.peak_vram = 0
        self.peak_power = 0.0
        self.sum_power = 0.0
        self.active_seconds = 0.0
        self.paused_seconds = 0.0
        self.device_path: Path | None = None
        self.power_path: Path | None = None
        if device.type == "cuda":
            requested = os.environ.get("ONSIDE_GPU_DEVICE_PATH")
            paths = [Path(requested)] if requested else [
                p for p in Path("/sys/class/drm").glob("card[0-9]*/device")
                if (p / "mem_info_vram_total").is_file()
            ]
            if len(paths) != 1:
                raise RuntimeError("Set ONSIDE_GPU_DEVICE_PATH to the single DRM device to monitor")
            self.device_path = paths[0]
            powers = list(self.device_path.glob("hwmon/*/power1_average"))
            if len(powers) != 1:
                raise RuntimeError("Exactly one readable GPU power1_average sensor is required")
            self.power_path = powers[0]
            total = int((self.device_path / "mem_info_vram_total").read_text())
            used = int((self.device_path / "mem_info_vram_used").read_text())
            # Leave room for driver allocations outside PyTorch and the current desktop.
            allowance = int(total * share) - used - 256 * 1024**2
            if allowance <= 0:
                raise RuntimeError("Existing GPU allocations leave no memory within the requested budget")
            torch.cuda.set_per_process_memory_fraction(allowance / total, device)
        self.sample()

    def sample(self) -> None:
        if self.device_path is None or self.power_path is None:
            return
        total = int((self.device_path / "mem_info_vram_total").read_text())
        used = int((self.device_path / "mem_info_vram_used").read_text())
        power = int(self.power_path.read_text()) / 1e6
        self.samples += 1
        self.peak_vram = max(self.peak_vram, used)
        self.peak_power = max(self.peak_power, power)
        self.sum_power += power
        self.write()
        if used > int(total * self.share):
            raise RuntimeError(f"Total device VRAM exceeded {self.share:.0%}: {used}/{total} bytes")

    def finish_batch(self, started: float) -> None:
        if self.device.type != "cuda":
            return
        torch.cuda.synchronize(self.device)
        active = time.monotonic() - started
        self.sample()
        pause = active * (1 / self.share - 1)
        self.active_seconds += active
        time.sleep(pause)
        self.paused_seconds += pause
        self.sample()

    def write(self) -> None:
        result: dict[str, Any] = {
            "pid": os.getpid(), "device": str(self.device), "share": self.share,
            "elapsed_seconds": time.monotonic() - self.started,
            "samples": self.samples, "peak_device_vram_bytes": self.peak_vram,
            "sampled_peak_watts": self.peak_power,
            "sampled_mean_watts": self.sum_power / self.samples if self.samples else None,
            "active_seconds": self.active_seconds, "paused_seconds": self.paused_seconds,
            "power_limit_enforced": False,
        }
        self.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.report.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(self.report)
