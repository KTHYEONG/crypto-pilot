from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import StrategyMLConfig
from src.domain.futures.strategy.features import build_feature_panel


def _aligned(close_2d: np.ndarray) -> AlignedMarketData:
    t_len, n_len = close_2d.shape
    dt = np.array(
        [np.datetime64("2026-01-01") + np.timedelta64(4 * i, "h") for i in range(t_len)],
        dtype="datetime64[ns]",
    )
    high = close_2d + 1.0
    low = np.maximum(close_2d - 1.0, 1e-6)
    return AlignedMarketData(
        datetimes=dt,
        symbols=tuple(f"S{i}" for i in range(n_len)),
        open_2d=close_2d.copy(),
        high_2d=high,
        low_2d=low,
        close_2d=close_2d.copy(),
        volume_2d=np.full((t_len, n_len), 1000.0, dtype=np.float64),
        funding_2d=np.zeros((t_len, n_len), dtype=np.float64),
        active_mask=np.ones((t_len, n_len), dtype=bool),
        warm_mask=np.ones((t_len, n_len), dtype=bool),
        entry_block_mask=np.zeros((t_len, n_len), dtype=bool),
        kill_mask=np.zeros((t_len, n_len), dtype=bool),
    )


def test_build_feature_panel_is_pit_safe_for_past_rows() -> None:
    close_base = np.array([[100.0, 100.0], [102.0, 102.0], [104.0, 104.0], [106.0, 106.0]])
    close_future_spike = close_base.copy()
    close_future_spike[-1, 0] = 10_000.0
    cfg = StrategyMLConfig(min_group_size=2)

    panel_base = build_feature_panel(_aligned(close_base), cfg)
    panel_spike = build_feature_panel(_aligned(close_future_spike), cfg)
    feat_idx = panel_base.feature_names.index("ret_1")
    np.testing.assert_allclose(
        panel_base.values[2, :, feat_idx],
        panel_spike.values[2, :, feat_idx],
    )


def test_build_feature_panel_respects_eligibility_masks() -> None:
    close = np.array([[100.0, 100.0], [101.0, 101.0], [102.0, 102.0], [103.0, 103.0]])
    aligned = _aligned(close)
    aligned = replace(
        aligned,
        kill_mask=np.array([[False, False], [False, True], [False, False], [False, False]]),
    )
    panel = build_feature_panel(aligned, StrategyMLConfig(min_group_size=2))
    assert panel.valid_mask.shape == (4, 2)
    assert not panel.valid_mask[1, 1]


def test_build_feature_panel_market_stats_use_finite_fallback_on_warmup_rows() -> None:
    close = np.array([[100.0, 100.0], [101.0, 101.0], [102.0, 102.0], [103.0, 103.0]])
    panel = build_feature_panel(_aligned(close), StrategyMLConfig(min_group_size=2))
    median_idx = panel.feature_names.index("market_median_ret_6")
    disp_idx = panel.feature_names.index("market_dispersion_6")
    np.testing.assert_allclose(panel.values[:, :, median_idx], 0.0)
    np.testing.assert_allclose(panel.values[:, :, disp_idx], 0.0)


def test_build_feature_panel_supports_group_ablation_toggle() -> None:
    close = np.array([[100.0, 100.0], [101.0, 101.0], [102.0, 102.0], [103.0, 103.0]])
    cfg = StrategyMLConfig(
        min_group_size=2,
        feature_groups_enabled=("trend", "missingness"),
        add_missingness_indicators=True,
    )
    panel = build_feature_panel(_aligned(close), cfg)
    assert "ret_1" in panel.feature_names
    assert "rv_6" not in panel.feature_names
    assert "funding_missing_ind" in panel.feature_names


def test_build_feature_panel_excludes_alias_features() -> None:
    # Arrange: minimal aligned data with enough symbols for CS rank
    close = np.tile(np.linspace(100.0, 110.0, 50)[:, None], (1, 5))
    cfg = StrategyMLConfig(min_group_size=2)

    # Act
    panel = build_feature_panel(_aligned(close), cfg)

    # Assert: alias duplicates must not appear in feature_names
    assert "carry_prior_6" not in panel.feature_names
    assert "xs_reversal_prior_6" not in panel.feature_names
    assert "cs_rank_dollar_volume" not in panel.feature_names
    # Originals must still exist
    assert "cs_rank_funding_6" in panel.feature_names
    assert "cs_rank_ret_6" in panel.feature_names
    assert "dollar_volume_rank" in panel.feature_names


def test_build_feature_panel_relaxes_all_finite_valid_mask() -> None:
    close = np.array([[100.0, 100.0], [101.0, 101.0], [102.0, 102.0], [103.0, 103.0]])
    aligned = _aligned(close)
    aligned = replace(
        aligned,
        funding_2d=np.array(
            [[0.0, 0.0], [np.nan, 0.0], [0.0, 0.0], [0.0, 0.0]],
            dtype=np.float64,
        ),
    )
    panel = build_feature_panel(aligned, StrategyMLConfig(min_group_size=2))
    assert panel.valid_mask[1, 0]
