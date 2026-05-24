from __future__ import annotations

import numpy as np

from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import StrategyMLConfig
from src.domain.futures.strategy.labels import build_label_panel


def _aligned_for_labels() -> AlignedMarketData:
    dt = np.array(
        [np.datetime64("2026-01-01") + np.timedelta64(4 * i, "h") for i in range(4)],
        dtype="datetime64[ns]",
    )
    open_2d = np.array([[100.0, 200.0], [110.0, 210.0], [120.0, 220.0], [130.0, 230.0]])
    close_2d = np.array([[101.0, 201.0], [121.0, 221.0], [132.0, 232.0], [143.0, 243.0]])
    return AlignedMarketData(
        datetimes=dt,
        symbols=("BTCUSDT", "ETHUSDT"),
        open_2d=open_2d,
        high_2d=close_2d + 1.0,
        low_2d=close_2d - 1.0,
        close_2d=close_2d,
        volume_2d=np.full((4, 2), 1000.0, dtype=np.float64),
        funding_2d=np.zeros((4, 2), dtype=np.float64),
        active_mask=np.ones((4, 2), dtype=bool),
        warm_mask=np.ones((4, 2), dtype=bool),
        entry_block_mask=np.zeros((4, 2), dtype=bool),
        kill_mask=np.zeros((4, 2), dtype=bool),
    )


def test_build_label_panel_uses_t_plus_1_open_close_alignment() -> None:
    """Verify B2 beta-residualized labels use t+1 execution alignment.

    With only 4 bars, trailing beta defaults to 1.0 (insufficient history).
    Expected values: gross_long - beta * market_fwd_ret (equal-weighted).
    t=0: gross=log(121/110), mfr=mean(log(121/110), log(221/210)), beta=1.0
    t=2: gross=log(143/130), mfr=mean(log(143/130), log(243/230)), beta=1.0
    """
    open_2d = np.array([[100.0, 200.0], [110.0, 210.0], [120.0, 220.0], [130.0, 230.0]])
    close_2d = np.array([[101.0, 201.0], [121.0, 221.0], [132.0, 232.0], [143.0, 243.0]])
    # market forward return at t=0: equal-weighted log return from open[1] to close[1]
    mfr0 = float(np.nanmean(np.log(close_2d[1] / open_2d[1])))
    mfr2 = float(np.nanmean(np.log(close_2d[3] / open_2d[3])))
    expected_t0 = np.log(121.0 / 110.0) - 1.0 * mfr0  # beta=1.0 (insufficient history)
    expected_t2 = np.log(143.0 / 130.0) - 1.0 * mfr2

    panel = build_label_panel(
        _aligned_for_labels(),
        StrategyMLConfig(label_horizon_bars=1, fee_bps=0.0, slippage_bps=0.0, min_group_size=2),
    )
    np.testing.assert_allclose(panel.long_net_ret[0, 0], expected_t0, rtol=1e-5)
    np.testing.assert_allclose(panel.long_net_ret[2, 0], expected_t2, rtol=1e-5)
    assert np.isnan(panel.long_net_ret[3, 0])


def test_build_label_panel_enforces_eligibility_mask_on_outputs() -> None:
    aligned = _aligned_for_labels()
    aligned = AlignedMarketData(
        datetimes=aligned.datetimes,
        symbols=aligned.symbols,
        open_2d=aligned.open_2d,
        high_2d=aligned.high_2d,
        low_2d=aligned.low_2d,
        close_2d=aligned.close_2d,
        volume_2d=aligned.volume_2d,
        funding_2d=aligned.funding_2d,
        active_mask=np.ones((4, 1), dtype=bool),
        warm_mask=np.array([[True, True], [False, False], [True, True], [True, True]]),
        entry_block_mask=np.zeros((4, 2), dtype=bool),
        kill_mask=np.zeros((4, 2), dtype=bool),
    )
    panel = build_label_panel(aligned, StrategyMLConfig(min_group_size=2))
    assert not panel.eligible_mask[1, 0]
    assert panel.sample_weight[1, 0] == 0.0


def test_build_label_panel_applies_ev_scaled_sample_weight() -> None:
    aligned = _aligned_for_labels()
    panel = build_label_panel(
        aligned,
        StrategyMLConfig(label_horizon_bars=1, fee_bps=0.0, slippage_bps=0.0, min_group_size=2),
    )
    liq_weight = np.clip(np.log1p(np.maximum(aligned.volume_2d, 0.0)), 0.25, 2.0).astype(np.float32)
    expected = np.where(
        panel.eligible_mask,
        liq_weight * (1.0 + 2.0 * np.abs(panel.signed_net_ret)),
        0.0,
    ).astype(np.float32)
    np.testing.assert_allclose(panel.sample_weight, expected, rtol=1e-6, atol=1e-8)
