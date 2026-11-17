"""Reject challenge assets and challenge-derived ball tracks during external training."""

import functools
import os
from pathlib import Path
import sys
from typing import Any


@functools.lru_cache(maxsize=8)
def restrict_training_reads(roots: tuple[Path, ...]) -> None:
    protected = tuple(root.resolve() for root in roots)

    def inspect_open(event: str, arguments: tuple[Any, ...]) -> None:
        if event != "open" or not arguments or isinstance(arguments[0], int):
            return
        path = Path(os.fsdecode(arguments[0])).resolve()
        if path.name.startswith("BAR-ATH") or any(path.is_relative_to(root) for root in protected):
            raise PermissionError(f"External PFF training cannot open challenge material: {path.name}")

    sys.addaudithook(inspect_open)
