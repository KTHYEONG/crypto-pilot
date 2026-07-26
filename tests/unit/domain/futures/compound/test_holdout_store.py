from __future__ import annotations

import pytest

from src.domain.futures.compound.contracts import (
    DeploymentVerdict,
    L3ValidationResult,
    SealedHoldoutManifest,
)
from src.domain.futures.compound.holdout_store import (
    HoldoutConflictError,
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


class TestEnsureSealed:
    def test_ensure_sealed_backfills_empty_spec_hash_on_unconsumed_seal(self, tmp_path) -> None:
        store = SealedHoldoutStore(tmp_path / "backfill.sqlite3")
        store.create(SealedHoldoutManifest(
            holdout_id="backfill-test",
            start_time_ns=1, end_time_ns=2, holdout_days=90,
            model_version="v1", data_manifest_hash="d1",
        ))
        result = store.ensure_sealed(SealedHoldoutManifest(
            holdout_id="backfill-test",
            start_time_ns=1, end_time_ns=2, holdout_days=90,
            model_version="v1", data_manifest_hash="d1",
            strategy_spec_hash="runtime-spec",
        ))
        assert result.strategy_spec_hash == "runtime-spec"

    def test_backfill_reuse_is_idempotent(self, tmp_path) -> None:
        store = SealedHoldoutStore(tmp_path / "backfill2.sqlite3")
        store.create(SealedHoldoutManifest(
            holdout_id="idempotent-test",
            start_time_ns=1, end_time_ns=2, holdout_days=90,
            model_version="v1", data_manifest_hash="d1",
        ))
        r1 = store.ensure_sealed(SealedHoldoutManifest(
            holdout_id="idempotent-test",
            start_time_ns=1, end_time_ns=2, holdout_days=90,
            model_version="v1", data_manifest_hash="d1",
            strategy_spec_hash="spec1",
        ))
        assert r1.strategy_spec_hash == "spec1"
        r2 = store.ensure_sealed(SealedHoldoutManifest(
            holdout_id="idempotent-test",
            start_time_ns=1, end_time_ns=2, holdout_days=90,
            model_version="v1", data_manifest_hash="d1",
            strategy_spec_hash="spec1",
        ))
        assert r2.strategy_spec_hash == "spec1"

    def test_ensure_sealed_consumed_empty_spec_hash_raises_holdout_reuse_error(self, tmp_path) -> None:
        store = SealedHoldoutStore(tmp_path / "empty_consumed.sqlite3")
        store.create(SealedHoldoutManifest(
            holdout_id="fail-test",
            start_time_ns=1, end_time_ns=2, holdout_days=90,
            model_version="v1", data_manifest_hash="d1",
        ))
        calls = 0
        def _eval(_m):
            nonlocal calls
            calls += 1
            return _result()
        store.consume(
            holdout_id="fail-test",
            model_version="v1", data_manifest_hash="d1",
            strategy_spec_hash="",
            evaluate=_eval,
        )
        with pytest.raises(HoldoutReuseError, match="consumed"):
            store.ensure_sealed(SealedHoldoutManifest(
                holdout_id="fail-test",
                start_time_ns=1, end_time_ns=2, holdout_days=90,
                model_version="v1", data_manifest_hash="d1",
                strategy_spec_hash="new-spec",
            ))

    def test_new_insert_when_not_exists(self, tmp_path) -> None:
        store = SealedHoldoutStore(tmp_path / "new_insert.sqlite3")
        manifest = SealedHoldoutManifest(
            holdout_id="new-test",
            start_time_ns=1, end_time_ns=2, holdout_days=90,
            model_version="v1", data_manifest_hash="d1",
            strategy_spec_hash="spec1",
        )
        result = store.ensure_sealed(manifest)
        assert result.holdout_id == "new-test"
        assert result.strategy_spec_hash == "spec1"

    def test_data_hash_mismatch_raises(self, tmp_path) -> None:
        store = SealedHoldoutStore(tmp_path / "mismatch.sqlite3")
        store.create(SealedHoldoutManifest(
            holdout_id="mismatch-test",
            start_time_ns=1, end_time_ns=2, holdout_days=90,
            model_version="v1", data_manifest_hash="d1",
            strategy_spec_hash="spec1",
        ))
        with pytest.raises(HoldoutReuseError, match="data_manifest_hash mismatch"):
            store.ensure_sealed(SealedHoldoutManifest(
                holdout_id="mismatch-test",
                start_time_ns=1, end_time_ns=2, holdout_days=90,
                model_version="v1", data_manifest_hash="d2",
                strategy_spec_hash="spec1",
            ))


class TestConsumeReentryGuard:
    def test_consume_rejects_different_strategy_spec_hash_after_consumption(self, tmp_path) -> None:
        store = SealedHoldoutStore(tmp_path / "reentry.sqlite3")
        store.create(SealedHoldoutManifest(
            holdout_id="reentry-test",
            start_time_ns=1, end_time_ns=2, holdout_days=90,
            model_version="v1", data_manifest_hash="d1",
            strategy_spec_hash="spec1",
        ))
        def _eval(_m):
            return _result()
        store.consume(
            holdout_id="reentry-test",
            model_version="v1", data_manifest_hash="d1",
            strategy_spec_hash="spec1",
            evaluate=_eval,
        )
        with pytest.raises(HoldoutReuseError, match="already consumed"):
            store.consume(
                holdout_id="reentry-test",
                model_version="v1", data_manifest_hash="d1",
                strategy_spec_hash="spec2",
                evaluate=_eval,
            )

    def test_same_spec_reentry_returns_cached(self, tmp_path) -> None:
        store = SealedHoldoutStore(tmp_path / "cached.sqlite3")
        store.create(SealedHoldoutManifest(
            holdout_id="cache-test",
            start_time_ns=1, end_time_ns=2, holdout_days=90,
            model_version="v1", data_manifest_hash="d1",
            strategy_spec_hash="spec1",
        ))
        call_count: list[int] = [0]
        def _eval(_m):
            call_count[0] += 1
            return _result()
        store.consume(
            holdout_id="cache-test",
            model_version="v1", data_manifest_hash="d1",
            strategy_spec_hash="spec1",
            evaluate=_eval,
        )
        assert call_count[0] == 1
        store.consume(
            holdout_id="cache-test",
            model_version="v1", data_manifest_hash="d1",
            strategy_spec_hash="spec1",
            evaluate=_eval,
        )
        assert call_count[0] == 1
