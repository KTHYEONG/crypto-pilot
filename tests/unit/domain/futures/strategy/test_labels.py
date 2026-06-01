from __future__ import annotations

import numpy as np

from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import StrategyMLConfig
from src.domain.futures.strategy.labels import build_forward_return_for_horizon, build_label_panel


def _aligned() -> AlignedMarketData:
    dt = np.array(
        [np.datetime64("2026-01-01") + np.timedelta64(4 * i, "h") for i in range(6)],
        dtype="datetime64[ns]",
    )
    open_2d = np.array(
        [[100.0, 200.0], [110.0, 210.0], [120.0, 220.0], [130.0, 230.0], [140.0, 240.0], [150.0, 250.0]],
        dtype=np.float64,
    )
    close_2d = open_2d + 5.0
    return AlignedMarketData(
        datetimes=dt,
        symbols=("BTCUSDT", "ETHUSDT"),
        open_2d=open_2d,
        high_2d=close_2d + 1.0,
        low_2d=close_2d - 1.0,
        close_2d=close_2d,
        volume_2d=np.full((6, 2), 1000.0, dtype=np.float64),
        funding_2d=np.zeros((6, 2), dtype=np.float64),
        active_mask=np.ones((6, 2), dtype=bool),
        warm_mask=np.ones((6, 2), dtype=bool),
        entry_block_mask=np.zeros((6, 2), dtype=bool),
        kill_mask=np.zeros((6, 2), dtype=bool),
    )


def test_build_forward_return_for_horizon_uses_t_plus_1_entry_and_t_plus_h_exit() -> None:
    aligned = _aligned()
    eligible = np.ones((6, 2), dtype=bool)
    beta = np.ones((6, 2), dtype=np.float64)
    out = build_forward_return_for_horizon(
        aligned=aligned,
        eligible_2d=eligible,
        beta_2d=beta,
        horizon_bars=2,
        target_mode="gross",
    )
    expected = np.log(aligned.close_2d[2, 0] / aligned.open_2d[1, 0])
    assert out[0, 0] == np.float32(expected)
    assert np.isnan(out[-1, 0])
    assert np.isnan(out[-2, 0])


def test_build_label_panel_populates_forward_return_by_horizon() -> None:
    cfg = StrategyMLConfig(label_horizon_bars=1, rank_policy_holding_candidates=(2, 3), min_group_size=2)
    panel = build_label_panel(_aligned(), cfg)
    assert panel.forward_return_by_horizon is not None
    assert set(panel.forward_return_by_horizon.keys()) == {1, 2, 3}
    h1 = panel.forward_return_by_horizon[1]
    h2 = panel.forward_return_by_horizon[2]
    valid = np.isfinite(h1) & np.isfinite(h2)
    assert np.any(valid)
    assert not np.allclose(h1[valid], h2[valid])
