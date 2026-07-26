from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.compound.config import CompoundEngineConfig
from src.domain.futures.compound.contracts import (
    CompoundEngineResult,
    DeploymentBundle,
    DeploymentCandidate,
    DeploymentVerdict,
    MarketFeatureCube,
)

_logger = logging.getLogger(__name__)


def _compute_bundle_sha256(bundle: DeploymentBundle) -> str:
    payload = {
        "schema_version": bundle.schema_version,
        "promotion_id": bundle.promotion_id,
        "candidate": {
            "active_signal_ids": list(bundle.candidate.active_signal_ids),
            "descriptors": [
                {
                    "signal_id": d.signal_id,
                    "family": d.family,
                    "speed": d.speed,
                    "lookback_hours": d.lookback_hours,
                    "native_timeframe": d.native_timeframe,
                    "target_horizon_hours": d.target_horizon_hours,
                }
                for d in bundle.candidate.descriptors
            ],
            "orientation_signs": list(bundle.candidate.orientation_signs),
            "vote_weights": list(bundle.candidate.vote_weights),
            "model_version": bundle.candidate.model_version,
            "strategy_spec_hash": bundle.candidate.strategy_spec_hash,
            "fold_manifest_hash": bundle.candidate.fold_manifest_hash,
            "trial_count": bundle.candidate.trial_count,
        },
        "data_manifest_hash": bundle.data_manifest_hash,
        "universe_state_hash": bundle.universe_state_hash,
        "config_payload": bundle.config_payload,
        "l2_payload": bundle.l2_payload,
        "l3_payload": bundle.l3_payload,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _build_config_payload(config: CompoundEngineConfig) -> dict[str, object]:
    return {
        "target_ann_vol": config.dynamic_compounding.target_ann_vol,
        "soft_drawdown_limit": config.dynamic_compounding.soft_drawdown_limit,
        "hard_drawdown_limit": config.dynamic_compounding.hard_drawdown_limit,
        "max_gross_leverage": config.dynamic_compounding.max_gross_leverage,
        "kelly_fraction": config.dynamic_compounding.kelly_fraction,
    }


def _build_l2_payload(result: CompoundEngineResult) -> dict[str, object]:
    return result.l2.to_dict()


def _build_l3_payload(result: CompoundEngineResult) -> dict[str, object]:
    return {
        "verdict": result.l3.verdict.value,
        "posterior_growth_probability": result.l3.posterior_growth_probability,
        "holdout_days": result.l3.holdout_days,
        "max_drawdown": result.l3.max_drawdown,
        "daily_cvar95": result.l3.daily_cvar95,
        "reasons": list(result.l3.reasons),
    }


def publish_promoted_strategy(
    *, result: CompoundEngineResult, candidate: DeploymentCandidate | None,
    config: CompoundEngineConfig, destination: Path,
) -> Path | None:
    if result.l3.verdict != DeploymentVerdict.PROMOTE:
        _logger.info(
            "skip publish: l3 verdict=%s (require PROMOTE)", result.l3.verdict.value,
        )
        return None

    if candidate is None:  # pragma: no cover - defensive guard
        _logger.warning("skip publish: candidate is None despite PROMOTE verdict")  # pragma: no cover
        return None  # pragma: no cover

    if not candidate.active_signal_ids:  # pragma: no cover - candidate contract rejects this
        _logger.warning("skip publish: candidate has no active signals")  # pragma: no cover
        return None  # pragma: no cover

    promotion_id = (
        f"promote-{candidate.strategy_spec_hash[:12]}-"
        f"{int(time.time_ns())}"
    )

    pre_bundle = DeploymentBundle(
        schema_version=1,
        promotion_id=promotion_id,
        candidate=candidate,
        data_manifest_hash=result.handoff.data_manifest_hash,
        universe_state_hash="",
        config_payload=_build_config_payload(config),
        l2_payload=_build_l2_payload(result),
        l3_payload=_build_l3_payload(result),
        sha256="",
    )

    sha256 = _compute_bundle_sha256(pre_bundle)
    bundle = DeploymentBundle(
        schema_version=pre_bundle.schema_version,
        promotion_id=pre_bundle.promotion_id,
        candidate=pre_bundle.candidate,
        data_manifest_hash=pre_bundle.data_manifest_hash,
        universe_state_hash=pre_bundle.universe_state_hash,
        config_payload=pre_bundle.config_payload,
        l2_payload=pre_bundle.l2_payload,
        l3_payload=pre_bundle.l3_payload,
        sha256=sha256,
    )

    destination.mkdir(parents=True, exist_ok=True)
    bundle_path = destination / f"{promotion_id}.bundle.json"
    tmp_path = destination / f".{promotion_id}.bundle.json.tmp"

    bundle_dict = {
        "schema_version": bundle.schema_version,
        "promotion_id": bundle.promotion_id,
        "candidate": {
            "active_signal_ids": list(bundle.candidate.active_signal_ids),
            "descriptors": [str(d) for d in bundle.candidate.descriptors],
            "orientation_signs": list(bundle.candidate.orientation_signs),
            "vote_weights": list(bundle.candidate.vote_weights),
            "model_version": bundle.candidate.model_version,
            "strategy_spec_hash": bundle.candidate.strategy_spec_hash,
            "fold_manifest_hash": bundle.candidate.fold_manifest_hash,
            "trial_count": bundle.candidate.trial_count,
        },
        "data_manifest_hash": bundle.data_manifest_hash,
        "universe_state_hash": bundle.universe_state_hash,
        "config_payload": bundle.config_payload,
        "l2_payload": bundle.l2_payload,
        "l3_payload": bundle.l3_payload,
        "sha256": sha256,
    }

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(bundle_dict, f, indent=2, sort_keys=True)

    os.replace(tmp_path, bundle_path)
    _logger.info("deployment bundle published: %s sha256=%s", bundle_path, sha256)

    active_path = destination / "active.json"
    active_tmp = destination / ".active.json.tmp"
    active_dict = {
        "promotion_id": bundle.promotion_id,
        "published_at_ns": int(time.time_ns()),
        "bundle_path": str(bundle_path),
        "sha256": sha256,
        "schema_version": bundle.schema_version,
    }
    with open(active_tmp, "w", encoding="utf-8") as f:
        json.dump(active_dict, f, indent=2, sort_keys=True)
    os.replace(active_tmp, active_path)
    _logger.info("active pointer updated: %s", active_path)

    return bundle_path


def compute_live_target_weights(
    *, bundle: DeploymentBundle, market: MarketFeatureCube,
    previous_weights: NDArray[np.float64], equity: float,
) -> NDArray[np.float64]:  # pragma: no cover - live adapter exercised in deployment integration
    if bundle.data_manifest_hash != market.data_manifest_hash:
        _logger.error(
            "[DATA] bundle data_hash=%s != market_hash=%s, returning zero weights",
            bundle.data_manifest_hash, market.data_manifest_hash,
        )
        return np.zeros(len(market.symbols), dtype=np.float64)

    config_payload = bundle.config_payload
    target_vol_val = config_payload.get("target_ann_vol", 0.15)
    target_vol = float(target_vol_val) if isinstance(target_vol_val, (int, float)) else 0.15

    n_syms = len(market.symbols)
    weights = np.zeros(n_syms, dtype=np.float64)

    if equity <= 0:
        _logger.warning("[SYS] non-positive equity=%.2f, returning zero weights", equity)
        return weights

    candidate = bundle.candidate
    for i, signal_id in enumerate(candidate.active_signal_ids):
        if signal_id not in market.symbols:
            continue
        sym_idx = market.symbols.index(signal_id)
        vote = candidate.vote_weights[i] * candidate.orientation_signs[i]
        weights[sym_idx] = vote

    abs_sum = float(np.sum(np.abs(weights)))
    if abs_sum > 0:
        weights = weights / abs_sum

    weights = weights * target_vol

    prev_nav = np.sum(np.abs(previous_weights)) * equity if equity > 0 else 0
    _logger.debug(
        "live weights computed: n_active=%d target_vol=%.4f prev_nav=%.2f",
        len(candidate.active_signal_ids), target_vol, prev_nav,
    )

    return weights


__all__ = [
    "compute_live_target_weights",
    "publish_promoted_strategy",
]
