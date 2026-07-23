from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.domain.futures.compound.config import CompoundEngineConfig
from src.domain.futures.compound.contracts import (
    AlphaEventTape,
    DeploymentVerdict,
    L3ValidationResult,
    MarketFeatureCube,
    SealedHoldoutManifest,
)
from src.domain.futures.compound.l1_multiscale import (
    CausalityError,
    run_l1_multiscale,
)
from src.domain.futures.compound.simulator import simulate_multiscale_portfolio
from src.domain.futures.compound.holdout_store import (
    HoldoutReuseError,
    SealedHoldoutStore,
)


@pytest.fixture
def multiscale_catalog():
    from src.domain.futures.compound.alpha_catalog import build_multiscale_alpha_catalog
    return build_multiscale_alpha_catalog()


@pytest.fixture
def complete_market() -> MarketFeatureCube:
    n_bars, n_syms = 750, 5
    rng = np.random.default_rng(42)
    base = np.linspace(100, 120, n_bars).reshape(-1, 1)
    noise = rng.normal(0, 0.5, (n_bars, n_syms))
    close = np.maximum(base + noise * 0.01 * base, 10.0).astype(np.float64)
    return MarketFeatureCube(
        timestamps_ns=np.arange(n_bars, dtype=np.int64) * 3_600_000_000_000,
        symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"),
        fields_2d={
            "open": close.copy(),
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "quote_volume": np.full((n_bars, n_syms), 50_000_000.0, dtype=np.float32),
            "funding": np.zeros((n_bars, n_syms), dtype=np.float32),
            "premium": np.zeros((n_bars, n_syms), dtype=np.float32),
            "mark": close.copy(),
            "index": close.copy(),
            "taker_buy_quote": np.full((n_bars, n_syms), 25_000_000.0, dtype=np.float32),
            "open_interest": np.full((n_bars, n_syms), 1_000_000_000.0, dtype=np.float64),
        },
        available_2d={"core": np.ones((n_bars, n_syms), dtype=np.bool_)},
        eligible_2d=np.ones((n_bars, n_syms), dtype=np.bool_),
        entry_block_2d=np.zeros((n_bars, n_syms), dtype=np.bool_),
        exit_required_2d=np.zeros((n_bars, n_syms), dtype=np.bool_),
        capacity_usdt_2d=np.full((n_bars, n_syms), 1_000_000.0, dtype=np.float64),
        execution_cost_bps_2d=np.full((n_bars, n_syms), 12.0, dtype=np.float32),
        data_manifest_hash="test-h1",
    )


@pytest.fixture
def pit_universe():
    return type("CompoundUniverseResult", (), {
        "symbols": ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"),
        "snapshots": (),
    })()


@pytest.fixture
def sealed_manifest() -> SealedHoldoutManifest:
    return SealedHoldoutManifest(
        holdout_id="test-holdout",
        start_time_ns=1000000,
        end_time_ns=2000000,
        holdout_days=90,
        model_version="v1",
        data_manifest_hash="data-h1",
        strategy_spec_hash="spec-h1",
    )


@pytest.fixture
def l3_result() -> L3ValidationResult:
    return L3ValidationResult(
        verdict=DeploymentVerdict.PROMOTE,
        posterior_growth_probability=0.85,
        holdout_days=90,
        max_drawdown=0.05,
        daily_cvar95=-0.01,
        reasons=(),
    )


# ── S1: 정상 E2E ──────────────────────────────────────────────────────

def test_s1_complete_data_produces_causal_events_and_integrity(
    complete_market, pit_universe, multiscale_catalog,
) -> None:
    handoff = run_l1_multiscale(
        market=complete_market,
        universe=pit_universe,
        catalog=multiscale_catalog,
        config=CompoundEngineConfig().l1_multiscale,
    )
    ledger = simulate_multiscale_portfolio(
        market=complete_market,
        universe=pit_universe,
        handoff=handoff,
        config=CompoundEngineConfig(),
    )
    assert handoff.events.num_rows > 0
    assert np.all(
        handoff.events["first_executable_time_ns"].to_numpy()
        > handoff.events["decision_time_ns"].to_numpy()
    )
    assert ledger.integrity_ok


# ── S2: 데이터/인과 fail-closed ────────────────────────────────────────

def test_s2_future_available_data_fails_closed(
    complete_market, pit_universe, multiscale_catalog,
) -> None:
    future_market = replace(
        complete_market,
        timestamps_ns=complete_market.timestamps_ns[::-1].copy(),
    )
    with pytest.raises((CausalityError, RuntimeError)):
        run_l1_multiscale(
            market=future_market,
            universe=pit_universe,
            catalog=multiscale_catalog,
            config=CompoundEngineConfig().l1_multiscale,
        )


# ── S3: L2 할당자 강건성 ──────────────────────────────────────────────

@pytest.fixture
def correlated_market() -> MarketFeatureCube:
    n_bars, n_syms = 500, 5
    rng = np.random.default_rng(42)
    common = rng.normal(0, 1, (n_bars, 1))
    noise = rng.normal(0, 0.1, (n_bars, n_syms))
    close = 100.0 * np.exp(0.001 * np.cumsum(common + noise * 0.3, axis=0))
    close = np.maximum(close, 10.0).astype(np.float64)

    exit_req = np.zeros((n_bars, n_syms), dtype=np.bool_)
    exit_req[-50:, 0] = True

    return MarketFeatureCube(
        timestamps_ns=np.arange(n_bars, dtype=np.int64) * 3_600_000_000_000,
        symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"),
        fields_2d={
            "open": close.copy(),
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "quote_volume": np.full((n_bars, n_syms), 50_000_000.0, dtype=np.float32),
            "funding": np.zeros((n_bars, n_syms), dtype=np.float32),
            "premium": np.zeros((n_bars, n_syms), dtype=np.float32),
        },
        available_2d={"core": np.ones((n_bars, n_syms), dtype=np.bool_)},
        eligible_2d=np.ones((n_bars, n_syms), dtype=np.bool_),
        entry_block_2d=np.zeros((n_bars, n_syms), dtype=np.bool_),
        exit_required_2d=exit_req,
        capacity_usdt_2d=np.full((n_bars, n_syms), 500_000.0, dtype=np.float64),
        execution_cost_bps_2d=np.full((n_bars, n_syms), 12.0, dtype=np.float32),
        data_manifest_hash="test-h3",
    )


@pytest.fixture
def active_handoff(correlated_market) -> AlphaEventTape:
    import pyarrow as pa
    n = correlated_market.timestamps_ns.size
    events = pa.table({
        "recipe_id": pa.array(["ts_trend_4h_h24"] * n * 2, type=pa.string()),
        "family": pa.array(["trend"] * n * 2, type=pa.string()),
        "native_timeframe": pa.array(["4h"] * n * 2, type=pa.string()),
        "symbol": pa.array(
            list(correlated_market.symbols) * (n * 2 // len(correlated_market.symbols)),
            type=pa.string(),
        )[:n * 2],
        "decision_time_ns": pa.array(
            np.tile(correlated_market.timestamps_ns, 2)[:n * 2], type=pa.int64(),
        ),
        "first_executable_time_ns": pa.array(
            np.tile(correlated_market.timestamps_ns + 3_600_000_000_000, 2)[:n * 2],
            type=pa.int64(),
        ),
        "expiry_time_ns": pa.array(
            np.tile(correlated_market.timestamps_ns + 24 * 3_600_000_000_000, 2)[:n * 2],
            type=pa.int64(),
        ),
        "cumulative_net_mu": pa.array(np.full(n * 2, 0.001, dtype=np.float64), type=pa.float64()),
        "half_life_hours": pa.array(np.full(n * 2, 12.0, dtype=np.float64), type=pa.float64()),
        "alpha_rate_per_hour": pa.array(np.full(n * 2, 4.16e-5, dtype=np.float64), type=pa.float64()),
        "mean_edge_variance": pa.array(np.full(n * 2, 1e-4, dtype=np.float64), type=pa.float64()),
        "residual_variance": pa.array(np.full(n * 2, 1e-4, dtype=np.float64), type=pa.float64()),
        "reliability": pa.array(np.full(n * 2, 0.8, dtype=np.float64), type=pa.float64()),
        "combination_weight": pa.array(np.full(n * 2, 1.0, dtype=np.float64), type=pa.float64()),
        "model_version": pa.array(["v1"] * n * 2, type=pa.string()),
        "data_manifest_hash": pa.array(["h1"] * n * 2, type=pa.string()),
        "fold_manifest_hash": pa.array(["f1"] * n * 2, type=pa.string()),
    })
    return AlphaEventTape(
        events=events,
        recipe_definitions=(),
        evidence=(),
        active_recipe_ids=("ts_trend_4h_h24",),
        model_version="v1",
        data_manifest_hash="h1",
        fold_manifest_hash="f1",
    )


def test_s3_allocator_penalizes_cluster_cost_and_forces_exit(
    correlated_market, pit_universe, active_handoff,
) -> None:
    ledger = simulate_multiscale_portfolio(
        market=correlated_market,
        universe=pit_universe,
        handoff=active_handoff,
        config=CompoundEngineConfig(),
    )
    assert ledger.integrity_ok
    assert np.allclose(
        ledger.target_weights_2d[correlated_market.exit_required_2d],
        0.0,
    )
    assert np.max(np.sum(np.abs(ledger.target_weights_2d), axis=1)) <= 1.0 + 1e-8


# ── S4: L3 영구 봉인 ─────────────────────────────────────────────────

def test_s4_holdout_is_atomic_idempotent_and_hash_bound(
    tmp_path, sealed_manifest, l3_result,
) -> None:
    store = SealedHoldoutStore(tmp_path / "holdouts.sqlite3")
    store.create(sealed_manifest)
    evaluate_calls = 0

    def evaluate(_manifest):
        nonlocal evaluate_calls
        evaluate_calls += 1
        return l3_result

    first = store.consume(
        holdout_id=sealed_manifest.holdout_id,
        model_version=sealed_manifest.model_version,
        data_manifest_hash=sealed_manifest.data_manifest_hash,
        strategy_spec_hash=sealed_manifest.strategy_spec_hash,
        evaluate=evaluate,
    )
    second = store.consume(
        holdout_id=sealed_manifest.holdout_id,
        model_version=sealed_manifest.model_version,
        data_manifest_hash=sealed_manifest.data_manifest_hash,
        strategy_spec_hash=sealed_manifest.strategy_spec_hash,
        evaluate=evaluate,
    )
    assert first == second
    assert evaluate_calls == 1
    with pytest.raises(HoldoutReuseError):
        store.consume(
            holdout_id=sealed_manifest.holdout_id,
            model_version=sealed_manifest.model_version,
            data_manifest_hash="changed",
            strategy_spec_hash=sealed_manifest.strategy_spec_hash,
            evaluate=evaluate,
        )
