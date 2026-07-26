from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from src.domain.futures.compound.config import CompoundEngineConfig
from src.domain.futures.compound.contracts import CausalFold


def _dataclass_to_dict(obj: Any) -> dict[str, object]:
    result: dict[str, object] = {}
    for f in fields(obj):
        v = getattr(obj, f.name)
        if isinstance(v, np.ndarray):
            result[f.name] = v.tolist()
        elif isinstance(v, np.floating):
            result[f.name] = _canonicalize_float(float(v))
        elif isinstance(v, np.integer):
            result[f.name] = int(v)
        elif isinstance(v, float):
            result[f.name] = _canonicalize_float(v)
        elif isinstance(v, Enum):
            result[f.name] = v.value
        elif isinstance(v, Path):
            result[f.name] = str(v)
        elif is_dataclass(v) and not isinstance(v, type):
            result[f.name] = _dataclass_to_dict(v)
        elif isinstance(v, (list, tuple)):
            result[f.name] = [
                _dataclass_to_dict(x) if is_dataclass(x) and not isinstance(x, type) else str(x) if isinstance(x, Path) else x
                for x in v
            ]
        elif v is None:
            result[f.name] = None
        else:
            result[f.name] = v
    return result


def _canonicalize_float(value: float) -> str:
    return format(value, ".12g")


def canonical_config_payload(config: CompoundEngineConfig) -> dict[str, object]:
    raw = _dataclass_to_dict(config)
    return _sort_keys(raw)


def _sort_keys(d: dict[str, object]) -> dict[str, object]:
    return {k: _sort_keys(v) if isinstance(v, dict) else v for k, v in sorted(d.items())}


def compute_strategy_spec_hash(*, config: CompoundEngineConfig) -> str:
    payload = canonical_config_payload(config)
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def compute_risk_policy_hash(*, config: CompoundEngineConfig) -> str:
    risk = canonical_config_payload(config).get("risk", {})
    raw = json.dumps(risk, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def compute_fold_manifest_hash(
    folds: Sequence[CausalFold], *, max_target_horizon_bars: int,
) -> str:
    if not folds:
        raise ValueError("folds must be non-empty")
    payload = {
        "fold_boundaries": [
            [f.fit_start, f.fit_end_exclusive, f.calibration_start,
             f.calibration_end_exclusive, f.oos_start, f.oos_end_exclusive,
             f.purge_bars, f.embargo_bars]
            for f in folds
        ],
        "max_target_horizon_bars": max_target_horizon_bars,
    }
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    inner = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"folds_{len(folds)}_{inner}"


def compute_candidate_hash(
    *, strategy_spec_hash: str, fold_manifest_hash: str,
    descriptor_ids: Sequence[str], risk_policy_hash: str,
) -> str:
    payload = {
        "strategy_spec_hash": strategy_spec_hash,
        "fold_manifest_hash": fold_manifest_hash,
        "descriptor_ids": sorted(descriptor_ids),
        "risk_policy_hash": risk_policy_hash,
    }
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
