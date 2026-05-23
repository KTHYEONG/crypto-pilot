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
    panel = build_label_panel(
        _aligned_for_labels(),
        StrategyMLConfig(label_horizon_bars=1, fee_bps=0.0, slippage_bps=0.0, min_group_size=2),
    )
    expected_t0 = np.log(121.0 / 110.0)
    expected_t2 = np.log(143.0 / 130.0)
    np.testing.assert_allclose(panel.long_net_ret[0, 0], expected_t0)
    np.testing.assert_allclose(panel.long_net_ret[2, 0], expected_t2)
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
