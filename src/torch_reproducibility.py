"""Strict repeatability for a fixed input, software stack and device."""

import hashlib
import importlib
import json
import os
import platform
import random
import sys
from collections.abc import Mapping
from typing import Any

import numpy as np

PROCESS_ENVIRONMENT = {
    "PYTHONHASHSEED": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "HIPBLAS_WORKSPACE_CONFIG": ":4096:8",
    "OMP_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
}


def ensure_reproducible_process() -> None:
    """Restart a CLI before work if interpreter/library startup settings differ."""
    if any(os.environ.get(key) != value for key, value in PROCESS_ENVIRONMENT.items()):
        environment = {**os.environ, **PROCESS_ENVIRONMENT}
        os.execve(sys.executable, [sys.executable, *sys.orig_argv[1:]], environment)


def enforce_deterministic_operations() -> None:
    """Reject unsupported nondeterministic operations instead of warning."""
    torch = importlib.import_module("torch")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def seed_deterministically(seed: int) -> None:
    for key, value in PROCESS_ENVIRONMENT.items():
        os.environ[key] = value
    random.seed(seed)
    np.random.seed(seed)
    torch = importlib.import_module("torch")
    torch.manual_seed(seed)
    enforce_deterministic_operations()


def deterministic_callback(_context: Any) -> None:
    """Restore strict settings after framework setup, validation or teardown."""
    os.environ.update(PROCESS_ENVIRONMENT)
    enforce_deterministic_operations()


def tensor_sha256(state: Mapping[str, Any]) -> str:
    """Hash tensor names, shapes, dtypes and values, excluding file metadata."""
    torch = importlib.import_module("torch")
    result = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode()
        result.update(len(header).to_bytes(8, "big"))
        result.update(header)
        result.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def runtime_description(device: str) -> dict[str, Any]:
    torch = importlib.import_module("torch")
    selected = torch.device("cuda:" + device if device.isdigit() else device)
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "torch": torch.__version__,
        "cuda": torch.version.cuda, "hip": torch.version.hip, "device": str(selected),
        "device_name": torch.cuda.get_device_name(selected) if selected.type == "cuda" else platform.machine(),
        "threads": torch.get_num_threads(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "environment": {key: os.environ.get(key) for key in PROCESS_ENVIRONMENT},
    }
