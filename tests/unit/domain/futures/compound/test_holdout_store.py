from __future__ import annotations

import pytest

from src.domain.futures.compound.contracts import (
    DeploymentVerdict,
    L3ValidationResult,
    SealedHoldoutManifest,
)
from src.domain.futures.compound.holdout_store import (
    HoldoutNotFoundError,
    HoldoutReuseError,
    SealedHoldoutStore,
)


def _manifest() -> SealedHoldoutManifest:
    return SealedHoldoutManifest(
        holdout_id="unit-holdout",
        start_time_ns=1,
        end_time_ns=2,
        holdout_days=90,
        model_version="model-v1",
        data_manifest_hash="data-v1",
        strategy_spec_hash="spec-v1",
    )


def _result() -> L3ValidationResult:
    return L3ValidationResult(
        verdict=DeploymentVerdict.SHADOW,
        posterior_growth_probability=0.5,
        holdout_days=90,
        max_drawdown=0.1,
        daily_cvar95=-0.02,
        reasons=(),
    )


def test_create_consume_is_idempotent_and_hash_bound(tmp_path) -> None:
    store = SealedHoldoutStore(tmp_path / "holdouts.sqlite3")
    manifest = _manifest()
    store.create(manifest)
    calls = 0

    def evaluate(_manifest: SealedHoldoutManifest) -> L3ValidationResult:
        nonlocal calls
        calls += 1
        return _result()

    assert store.consume(
        holdout_id=manifest.holdout_id,
        model_version=manifest.model_version,
        data_manifest_hash=manifest.data_manifest_hash,
        strategy_spec_hash=manifest.strategy_spec_hash,
        evaluate=evaluate,
    ) == store.consume(
        holdout_id=manifest.holdout_id,
        model_version=manifest.model_version,
        data_manifest_hash=manifest.data_manifest_hash,
        strategy_spec_hash=manifest.strategy_spec_hash,
        evaluate=evaluate,
    )
    assert calls == 1

    with pytest.raises(HoldoutReuseError):
        store.consume(
            holdout_id=manifest.holdout_id,
            model_version=manifest.model_version,
            data_manifest_hash="changed",
            strategy_spec_hash=manifest.strategy_spec_hash,
            evaluate=evaluate,
        )


def test_consume_unknown_holdout_fails_closed(tmp_path) -> None:
    store = SealedHoldoutStore(tmp_path / "holdouts.sqlite3")
    with pytest.raises(HoldoutNotFoundError):
        store.consume(
            holdout_id="missing",
            model_version="model-v1",
            data_manifest_hash="data-v1",
            strategy_spec_hash="spec-v1",
            evaluate=lambda _manifest: _result(),
        )
