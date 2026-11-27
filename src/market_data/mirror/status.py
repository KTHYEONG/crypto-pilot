from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.common.errors import DataIntegrityError

STATUS_FILE = "mirror_status.json"


def read_status(status_dir: Path) -> dict[str, Any]:
    path = status_dir / STATUS_FILE
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DataIntegrityError(f"malformed mirror status {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DataIntegrityError(f"malformed mirror status {path}: not an object")
    return payload


def update_status(status_dir: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    payload = read_status(status_dir)
    mutate(payload)
    status_dir.mkdir(parents=True, exist_ok=True)
    path = status_dir / STATUS_FILE
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)
