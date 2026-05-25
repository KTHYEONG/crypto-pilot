from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np


def build_manifest_hash(payload: dict[str, Any]) -> str:
    """Build deterministic manifest hash from JSON-serializable payload."""
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def ensure_cache_dir(path: Path) -> None:
    """Ensure cache directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def write_manifest(path: Path, payload: dict[str, Any]) -> Path:
    """Write deterministic manifest JSON with hash included."""
    ensure_cache_dir(path.parent)
    manifest = dict(payload)
    manifest["manifest_hash"] = build_manifest_hash(payload)
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    return path


def read_manifest(path: Path) -> dict[str, Any]:
    """Read manifest JSON payload."""
    return dict(json.loads(path.read_text(encoding="utf-8")))


def strategy_ml_cache_paths(base_dir: Path, run_id: str) -> dict[str, Path]:
    """Return canonical cache/artifact paths for strategy ML."""
    root = base_dir / "strategy_ml"
    return {
        "features": root / "features",
        "labels": root / "labels",
        "manifest": root / "manifest",
        "models": Path("logs") / "futures" / "models" / "strategy_ml" / run_id,
    }


def make_cache_key_payload(
    *,
    symbols: tuple[str, ...],
    timeframe: str,
    start: str,
    end: str,
    feature_config: dict[str, Any],
    source_column_availability: dict[str, bool],
    label_horizon: int,
    cost_config: dict[str, float],
    execution_alignment_config: dict[str, Any],
) -> dict[str, Any]:
    """Build deterministic cache-key payload."""
    return {
        "symbols": list(symbols),
        "timeframe": timeframe,
        "date_window": {"start": start, "end": end},
        "feature_config": feature_config,
        "source_column_availability": source_column_availability,
        "label_horizon": int(label_horizon),
        "cost_config": cost_config,
        "execution_alignment_config": execution_alignment_config,
    }


def write_feature_or_label_npz(path: Path, **arrays: np.ndarray) -> Path:
    """Write feature/label artifact as compressed NPZ."""
    ensure_cache_dir(path.parent)
    payload = {k: np.asarray(v) for k, v in arrays.items()}
    np.savez_compressed(path, **payload)  # type: ignore[arg-type]
    return path


def read_feature_or_label_npz(path: Path) -> dict[str, np.ndarray]:
    """Read feature/label artifact from compressed NPZ."""
    with np.load(path, allow_pickle=False) as data:
        return {k: np.asarray(data[k]) for k in data.files}


def write_artifact_manifest(
    manifest_dir: Path,
    *,
    key_payload: dict[str, Any],
    features_file: str | None = None,
    labels_file: str | None = None,
    models_dir: str | None = None,
) -> Path:
    """Write deterministic cache manifest for features/labels/models."""
    payload: dict[str, Any] = {
        "key_payload": key_payload,
        "features_file": features_file,
        "labels_file": labels_file,
        "models_dir": models_dir,
    }
    digest = build_manifest_hash(payload)
    return write_manifest(manifest_dir / f"{digest}.json", payload)


def save_lightgbm_booster(path: Path, booster: lgb.Booster) -> Path:
    """Save LightGBM booster artifact to disk."""
    ensure_cache_dir(path.parent)
    booster.save_model(str(path))
    return path


def save_lightgbm_model(path: Path, model: lgb.LGBMModel) -> Path:
    """Save fitted LightGBM sklearn-wrapper model by persisting its booster."""
    booster = model.booster_
    if booster is None:
        raise RuntimeError("lightgbm model is not fitted")
    return save_lightgbm_booster(path, booster)


def load_lightgbm_booster(path: Path) -> lgb.Booster:
    """Load LightGBM booster artifact from disk."""
    return lgb.Booster(model_file=str(path))
