from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from src.domain.futures.strategy.candidate_contracts import (
    SignalSleeveKey,
    ValidatedSignalBatch,
    ValidatedSignalEvent,
)
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.tiered_workflow.awf_sim import (
    build_l2_simulation_cache,
)


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
