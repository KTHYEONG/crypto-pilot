from __future__ import annotations

import numpy as np

from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.tiered_workflow.awf_sim import (
    _policy_effect_is_visible,
    _resolve_tradeable_mask,
    _summarize_regime_policy_effects,
)
from src.domain.futures.strategy.tiered_workflow.dataclasses import (
    Layer2AllocationConfig,
    RegimePolicyApplication,
)


def _make_aligned_for_tradeable_masks() -> AlignedMarketData:
    t = 3
    n = 2
    close = np.full((t, n), 100.0, dtype=np.float64)
    return AlignedMarketData(
        datetimes=np.array(
            [
                np.datetime64("2025-01-01T00"),
                np.datetime64("2025-01-01T04"),
                np.datetime64("2025-01-01T08"),
            ]
        ),
        symbols=("BTCUSDT", "ETHUSDT"),
        open_2d=close.copy(),
        high_2d=close.copy(),
        low_2d=close.copy(),
        close_2d=close.copy(),
        volume_2d=np.ones((t, n), dtype=np.float64),
        funding_2d=np.zeros((t, n), dtype=np.float64),
        active_mask=np.ones((t, n), dtype=bool),
        warm_mask=np.ones((t, n), dtype=bool),
        entry_block_mask=np.zeros((t, n), dtype=bool),
        kill_mask=np.zeros((t, n), dtype=bool),
    )


def test_resolve_tradeable_mask_combines_optional_masks() -> None:
    aligned = _make_aligned_for_tradeable_masks()
    execution_eligibility_mask = np.array(
        [
            [True, True],
            [False, True],
            [True, True],
        ],
        dtype=bool,
    )
    strategy_readiness_mask = np.array(
        [
            [True, True],
            [True, False],
            [True, True],
        ],
        dtype=bool,
    )
    promotion_active_mask = np.array(
        [
            [True, True],
            [True, True],
            [False, True],
        ],
        dtype=bool,
    )
    aligned = AlignedMarketData(
        datetimes=aligned.datetimes,
        symbols=aligned.symbols,
        open_2d=aligned.open_2d,
        high_2d=aligned.high_2d,
        low_2d=aligned.low_2d,
        close_2d=aligned.close_2d,
        volume_2d=aligned.volume_2d,
        funding_2d=aligned.funding_2d,
        active_mask=aligned.active_mask,
        warm_mask=aligned.warm_mask,
        entry_block_mask=aligned.entry_block_mask,
        kill_mask=aligned.kill_mask,
        execution_eligibility_mask=execution_eligibility_mask,
        strategy_readiness_mask=strategy_readiness_mask,
        promotion_active_mask=promotion_active_mask,
    )

    row0 = _resolve_tradeable_mask(aligned=aligned, t=0, n_sym=2)
    row1 = _resolve_tradeable_mask(aligned=aligned, t=1, n_sym=2)
    row2 = _resolve_tradeable_mask(aligned=aligned, t=2, n_sym=2)

    np.testing.assert_array_equal(row0, np.array([True, True]))
    np.testing.assert_array_equal(row1, np.array([False, False]))
    np.testing.assert_array_equal(row2, np.array([False, True]))


def test_entry_cooldown_masks_new_symbols_causally() -> None:
    t = 15
    active_mask = np.zeros((t, 1), dtype=bool)
    active_mask[2:, 0] = True
    close = np.full((t, 1), 100.0, dtype=np.float64)
    aligned = AlignedMarketData(
        datetimes=np.array([np.datetime64("2025-01-01T00") + np.timedelta64(i * 4, "h") for i in range(t)]),
        symbols=("BTCUSDT",),
        open_2d=close.copy(),
        high_2d=close.copy(),
        low_2d=close.copy(),
        close_2d=close.copy(),
        volume_2d=np.ones((t, 1), dtype=np.float64),
        funding_2d=np.zeros((t, 1), dtype=np.float64),
        active_mask=active_mask,
        warm_mask=np.ones((t, 1), dtype=bool),
        entry_block_mask=np.zeros((t, 1), dtype=bool),
        kill_mask=np.zeros((t, 1), dtype=bool),
    )
    config = Layer2AllocationConfig(l2_entry_cooldown_bars=3)

    rows = [
        bool(_resolve_tradeable_mask(aligned=aligned, t=i, n_sym=1, config=config)[0])
        for i in range(t)
    ]

    assert rows[2] is False
    assert rows[3] is False
    assert rows[4] is False
    assert rows[5] is True


def test_summarize_regime_policy_effects_ratios() -> None:
    summary = _summarize_regime_policy_effects(
        (
            RegimePolicyApplication(
                sleeve_sigs={},
                sleeve_edges={},
                n_input=4,
                n_allow=1,
                n_downweight=2,
                n_block=1,
                n_pooled=0,
                gross_edge_before_bps=100.0,
                gross_edge_after_bps=60.0,
                abs_mu_before_bps=80.0,
                abs_mu_after_bps=40.0,
                quality_weight_before=10.0,
                quality_weight_after=6.0,
            ),
        )
    )

    assert summary.n_bars == 1
    assert summary.n_sleeves == 4
    assert summary.action_ratio == np.float64(1.0)
    assert summary.pooled_ratio == np.float64(0.0)
    assert summary.block_ratio == np.float64(0.25)
    assert summary.mu_abs_ratio == np.float64(0.5)
    assert summary.quality_weight_ratio == np.float64(0.6)
    assert summary.edge_abs_ratio == np.float64(0.6)


def test_policy_effect_is_visible_respects_thresholds() -> None:
    visible_summary = _summarize_regime_policy_effects(
        (
            RegimePolicyApplication(
                sleeve_sigs={},
                sleeve_edges={},
                n_input=5,
                n_allow=1,
                n_downweight=3,
                n_block=0,
                n_pooled=1,
                gross_edge_before_bps=100.0,
                gross_edge_after_bps=70.0,
                abs_mu_before_bps=100.0,
                abs_mu_after_bps=70.0,
                quality_weight_before=10.0,
                quality_weight_after=7.0,
            ),
        )
    )
    pooled_summary = _summarize_regime_policy_effects(
        (
            RegimePolicyApplication(
                sleeve_sigs={},
                sleeve_edges={},
                n_input=5,
                n_allow=0,
                n_downweight=0,
                n_block=0,
                n_pooled=5,
                gross_edge_before_bps=100.0,
                gross_edge_after_bps=98.0,
                abs_mu_before_bps=100.0,
                abs_mu_after_bps=98.0,
                quality_weight_before=10.0,
                quality_weight_after=9.8,
            ),
        )
    )
    config = Layer2AllocationConfig(
        l2_regime_max_pooled_ratio_for_effective=0.80,
        l2_regime_min_action_ratio_for_effective=0.10,
        l2_regime_min_mu_abs_change=0.03,
    )

    assert _policy_effect_is_visible(visible_summary, mode="soft", config=config) is True
    assert _policy_effect_is_visible(pooled_summary, mode="soft", config=config) is False
    assert _policy_effect_is_visible(pooled_summary, mode="observe", config=config) is True
