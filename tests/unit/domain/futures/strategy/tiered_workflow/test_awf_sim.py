from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps
from src.domain.futures.strategy.candidate_contracts import (
    SignalSleeveKey,
    ValidatedSignalBatch,
    ValidatedSignalEvent,
)
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.tiered_workflow.awf_sim import (
    build_causal_net_sleeve_returns,
    build_l2_simulation_cache,
)
from src.domain.futures.strategy.walk_forward import WFFold


def _make_event(
    symbol: str = "BTCUSDT",
    strategy_id: str = "donchian_72",
    native_tf: str = "4h",
    decision_idx: int = 100,
) -> ValidatedSignalEvent:
    return ValidatedSignalEvent(
        decision_idx=decision_idx,
        decision_time=np.datetime64("2026-01-15"),
        symbol=symbol,
        strategy_id=strategy_id,
        native_tf=native_tf,
        activation_context="all",
        side=1,
        expected_gross_bps=50.0,
        q10_gross_bps=10.0,
        q90_gross_bps=90.0,
        expected_holding_bars=12,
        registry_version="r1",
        model_version="m1",
    )


def test_build_cache_uses_signal_sleeve_key() -> None:
    aligned = MagicMock(spec=AlignedMarketData)
    aligned.symbols = ("BTCUSDT",)
    aligned.datetimes = np.array([np.datetime64("2026-01-15")] * 200)
    aligned.close_2d = np.ones((200, 1), dtype=np.float64)
    aligned.execution_cost_bps_2d = np.full((200, 1), 3.8, dtype=np.float64)
    aligned.funding_2d = np.zeros((200, 1), dtype=np.float64)
    aligned.beta_vs_market_1d = np.zeros(1, dtype=np.float64)
    aligned.volume_usdt_2d = np.ones((200, 1), dtype=np.float64)
    aligned.turnover_2d = np.ones((200, 1), dtype=np.float64)
    aligned.open_2d = np.ones((200, 1), dtype=np.float64)
    aligned.high_2d = np.ones((200, 1), dtype=np.float64)
    aligned.low_2d = np.ones((200, 1), dtype=np.float64)

    ev_4h = _make_event(symbol="BTCUSDT", strategy_id="trend_ma:ema_12_72", native_tf="4h", decision_idx=100)
    ev_6h = _make_event(symbol="BTCUSDT", strategy_id="trend_ma:ema_12_72", native_tf="6h", decision_idx=100)
    ev_1d = _make_event(
        symbol="BTCUSDT",
        strategy_id="btc_regime_pullback:btc_pullback_100_slow",
        native_tf="1d",
        decision_idx=100,
    )

    batch = ValidatedSignalBatch(
        events=(ev_4h, ev_6h, ev_1d),
        start_idx=100,
        end_idx=103,
        symbols=("BTCUSDT",),
        registry_version="rv",
        model_version="mv",
    )

    cache = build_l2_simulation_cache(aligned, batch, "4h")

    assert len(cache.sleeve_keys) == 3
    native_tfs = {sk.native_tf for sk in cache.sleeve_keys}
    assert native_tfs == {"4h", "6h", "1d"}

    for sk in cache.sleeve_keys:
        assert isinstance(sk, SignalSleeveKey)
        assert sk.symbol == "BTCUSDT"
        assert sk.native_tf

    # backward-compat properties
    assert len(cache.sleeve_ids) == 3
    assert len(cache.sleeve_to_tf) == 3
    for stf, sk in zip(cache.sleeve_to_tf, cache.sleeve_keys, strict=True):
        assert stf == sk.native_tf


def test_build_causal_net_sleeve_returns_uses_only_active_prior_signal_and_cost() -> None:
    # Arrange
    aligned = MagicMock(spec=AlignedMarketData)
    aligned.symbols = ("BTCUSDT",)
    aligned.datetimes = np.array([np.datetime64("2026-01-15")] * 8)
    aligned.close_2d = np.array([[100.0], [110.0], [121.0], [121.0], [121.0], [121.0], [121.0], [121.0]])
    aligned.execution_cost_bps_2d = np.full((8, 1), 10.0, dtype=np.float64)
    aligned.funding_2d = np.zeros((8, 1), dtype=np.float64)
    aligned.beta_vs_market_1d = np.zeros(1, dtype=np.float64)
    aligned.volume_usdt_2d = np.ones((8, 1), dtype=np.float64)
    aligned.turnover_2d = np.ones((8, 1), dtype=np.float64)
    aligned.open_2d = np.ones((8, 1), dtype=np.float64)
    aligned.high_2d = np.ones((8, 1), dtype=np.float64)
    aligned.low_2d = np.ones((8, 1), dtype=np.float64)
    batch = ValidatedSignalBatch(
        events=(_make_event(decision_idx=0),), start_idx=0, end_idx=3,
        symbols=("BTCUSDT",), registry_version="rv", model_version="mv",
    )
    cache = build_l2_simulation_cache(aligned, batch, "4h")

    # Act
    returns = build_causal_net_sleeve_returns(cache=cache, aligned=aligned, start=0, end=4)

    # Assert
    assert returns.dtype == np.float32
    assert returns.shape == (4, 1)
    assert returns[0, 0] == 0.0
    assert returns[1, 0] == np.float32(0.1 - 0.001 / 12.0)


def _make_aligned(n_bars: int = 20, n_sym: int = 2) -> MagicMock:
    close = np.ones((n_bars, n_sym), dtype=np.float64) * 100.0
    aligned = MagicMock()
    aligned.symbols = ("BTC", "ETH") if n_sym >= 2 else ("BTC",)
    aligned.close_2d = close
    aligned.datetimes = np.array(
        [np.datetime64("2024-01-01", "ns") + np.timedelta64(i * 4, "h") for i in range(n_bars)],
        dtype="datetime64[ns]",
    )
    aligned.funding_2d = np.zeros((n_bars, n_sym), dtype=np.float64)
    aligned.active_mask = np.ones((n_bars, n_sym), dtype=bool)
    aligned.warm_mask = np.ones((n_bars, n_sym), dtype=bool)
    aligned.entry_block_mask = np.zeros((n_bars, n_sym), dtype=bool)
    aligned.kill_mask = np.zeros((n_bars, n_sym), dtype=bool)
    aligned.execution_cost_bps_2d = np.full((n_bars, n_sym), 4.0, dtype=np.float64)
    aligned.beta_vs_market_1d = np.zeros(n_sym, dtype=np.float64)
    return aligned


def _make_signal_batch(n_bars: int = 20) -> ValidatedSignalBatch:
    datetimes = np.array(
        [np.datetime64("2024-01-01", "ns") + np.timedelta64(i * 4, "h") for i in range(n_bars)],
        dtype="datetime64[ns]",
    )
    return ValidatedSignalBatch(
        events=(
            ValidatedSignalEvent(
                decision_idx=0,
                decision_time=datetimes[0],
                symbol="BTC",
                strategy_id="trend:fast",
                activation_context="all",
                side=1,
                expected_net_bps=0.0,
                expected_gross_bps=20.0,
                q10_net_bps=0.0,
                q10_gross_bps=10.0,
                q90_net_bps=0.0,
                q90_gross_bps=30.0,
                expected_holding_bars=1,
                registry_version="test",
                model_version="test",
            ),
            ValidatedSignalEvent(
                decision_idx=0,
                decision_time=datetimes[0],
                symbol="ETH",
                strategy_id="trend:fast",
                activation_context="all",
                side=-1,
                expected_net_bps=0.0,
                expected_gross_bps=5.0,
                q10_net_bps=0.0,
                q10_gross_bps=2.0,
                q90_net_bps=0.0,
                q90_gross_bps=8.0,
                expected_holding_bars=1,
                registry_version="test",
                model_version="test",
            ),
        ),
        start_idx=1,
        end_idx=3,
        symbols=("BTC", "ETH"),
        registry_version="test",
        model_version="test",
    )


def _make_folds() -> tuple[WFFold, ...]:
    return (WFFold(fit_start=0, fit_end=1, cal_start=1, cal_end=1, oos_start=1, oos_end=4),)


def test_awf_sim_wires_kelly_shrink_to_equal_from_config(mocker: MagicMock) -> None:
    from src.domain.futures.strategy.tiered_workflow.awf_sim import (
        _run_awf_simulation,
        build_l2_simulation_cache,
    )
    from src.domain.futures.strategy.tiered_workflow.dataclasses import (
        Layer2AllocationConfig,
    )

    spy = mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.awf_sim.diagonal_kelly_weights",
        return_value=np.zeros(2, dtype=np.float64),
    )
    aligned = _make_aligned()
    signal_batch = _make_signal_batch()
    config = Layer2AllocationConfig(
        k_rank=2,
        rank_buffer=0,
        kelly_fraction=0.5,
        no_trade_band=0.0,
        rebalance_bars=1,
        kelly_shrink_to_equal=0.3,
    )
    awf_folds = _make_folds()
    caps = PortfolioCaps(gross=2.0, per_symbol=1.0, net=1.0, beta=2.0, target_ann_vol=10.0)
    cache = build_l2_simulation_cache(aligned, signal_batch, "4h")

    _run_awf_simulation(
        cache=cache,
        signal_batch=signal_batch,
        aligned=aligned,
        awf_folds=awf_folds,
        config=config,
        caps=caps,
        tf="4h",
    )

    call_kwargs = spy.call_args.kwargs
    assert "kelly_shrink_to_equal" in call_kwargs
    assert call_kwargs["kelly_shrink_to_equal"] == pytest.approx(0.3)
