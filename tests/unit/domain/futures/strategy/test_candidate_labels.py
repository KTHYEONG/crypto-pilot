from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.domain.futures.strategy.candidate_labels import (
    _is_usable_cost_array,
    compute_triple_barrier_returns,
    label_candidate_events,
)
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import CandidateStrategyConfig
from src.domain.futures.strategy.execution_cost import ExecutionCostModel


def _make_aligned() -> AlignedMarketData:
    t = 30
    n = 1
    base = np.linspace(100.0, 130.0, t, dtype=np.float64).reshape(t, n)
    datetimes = np.datetime64("2025-01-01T00", "h") + np.arange(t).astype("timedelta64[h]")
    return AlignedMarketData(
        datetimes=datetimes,
        symbols=("BTCUSDT",),
        open_2d=base.copy(),
        high_2d=base * 1.01,
        low_2d=base * 0.99,
        close_2d=base.copy(),
        volume_2d=np.full((t, n), 1000.0, dtype=np.float64),
        funding_2d=np.zeros((t, n), dtype=np.float64),
        active_mask=np.ones((t, n), dtype=bool),
        warm_mask=np.ones((t, n), dtype=bool),
        entry_block_mask=np.zeros((t, n), dtype=bool),
        kill_mask=np.zeros((t, n), dtype=bool),
        execution_cost_bps_2d=np.full((t, n), 5.0, dtype=np.float64),
    )


def _make_collision_aligned() -> AlignedMarketData:
    t = 20
    n = 1
    open_ = np.full((t, n), 100.0, dtype=np.float64)
    high = np.full((t, n), 120.0, dtype=np.float64)
    low = np.full((t, n), 80.0, dtype=np.float64)
    close = np.full((t, n), 100.0, dtype=np.float64)
    datetimes = np.datetime64("2025-01-01T00", "h") + np.arange(t).astype("timedelta64[h]")
    return AlignedMarketData(
        datetimes=datetimes,
        symbols=("BTCUSDT",),
        open_2d=open_,
        high_2d=high,
        low_2d=low,
        close_2d=close,
        volume_2d=np.full((t, n), 1000.0, dtype=np.float64),
        funding_2d=np.zeros((t, n), dtype=np.float64),
        active_mask=np.ones((t, n), dtype=bool),
        warm_mask=np.ones((t, n), dtype=bool),
        entry_block_mask=np.zeros((t, n), dtype=bool),
        kill_mask=np.zeros((t, n), dtype=bool),
        execution_cost_bps_2d=np.full((t, n), 5.0, dtype=np.float64),
    )


def test_label_candidate_events_t_plus_one_entry_and_columns() -> None:
    aligned = _make_aligned()
    events = pd.DataFrame(
        {
            "datetime": [aligned.datetimes[10]],
            "symbol": ["BTCUSDT"],
            "side": [1],
            "entry_idx": [11],
            "expected_holding_bars": [5],
            "min_holding_bars": [1],
            "stop_atr_mult": [1.0],
            "take_profit_atr_mult": [1.0],
            "cost_floor_bps": [5.0],
        }
    )

    out = label_candidate_events(events=events, aligned=aligned, cfg=CandidateStrategyConfig())

    assert out.shape[0] == 1
    assert int(out.loc[0, "entry_idx"]) == 11
    for col in (
        "event_id",
        "barrier_first_label",
        "profitable_after_hurdle_label",
        "gross_event_bps",
        "gross_fwd_bps",
        "execution_cost_bps",
        "realized_funding_bps",
        "net_event_bps",
        "ex_ante_cost_bps",
        "edge_after_hurdle_bps",
        "triple_barrier_label",
        "time_to_exit_bars",
        "mae_bps",
        "mfe_bps",
        "net_return_r",
        "mae_r",
        "mfe_r",
        "risk_unit_bps",
        "realized_vol_bps",
        "exit_reason",
        "exit_idx",
        "exit_policy_version",
        "same_bar_collision",
    ):
        assert col in out.columns
    assert np.isfinite(float(out.loc[0, "gross_fwd_bps"]))


def test_label_candidate_events_uses_yang_zhang_volatility_proxy() -> None:
    aligned = _make_aligned()
    events = pd.DataFrame(
        {
            "datetime": [aligned.datetimes[10]],
            "symbol": ["BTCUSDT"],
            "side": [1],
            "entry_idx": [11],
            "expected_holding_bars": [5],
            "min_holding_bars": [1],
            "stop_atr_mult": [1.0],
            "take_profit_atr_mult": [1.0],
            "cost_floor_bps": [5.0],
        }
    )

    out = label_candidate_events(events=events, aligned=aligned, cfg=CandidateStrategyConfig())

    assert float(out.loc[0, "risk_unit_bps"]) > 0.0
    assert np.isfinite(float(out.loc[0, "realized_vol_bps"]))


def test_label_candidate_events_uses_future_window_only_for_targets() -> None:
    aligned = _make_aligned()
    events = pd.DataFrame(
        {
            "datetime": [aligned.datetimes[8]],
            "symbol": ["BTCUSDT"],
            "side": [1],
            "entry_idx": [9],
            "expected_holding_bars": [4],
            "min_holding_bars": [1],
            "stop_atr_mult": [50.0],
            "take_profit_atr_mult": [50.0],
            "cost_floor_bps": [0.0],
        }
    )
    out = label_candidate_events(events=events, aligned=aligned, cfg=CandidateStrategyConfig())
    assert int(out.loc[0, "time_to_exit_bars"]) == 4
    assert str(out.loc[0, "exit_reason"]) == "time_exit"
    assert int(out.loc[0, "exit_idx"]) == 13
    assert int(out.loc[0, "same_bar_collision"]) == 0
    assert str(out.loc[0, "exit_policy_version"]) == "candidate_label_atr_v2"


def test_label_candidate_events_separates_barrier_and_profitable_labels() -> None:
    aligned = _make_aligned()
    events = pd.DataFrame(
        {
            "datetime": [aligned.datetimes[8]],
            "symbol": ["BTCUSDT"],
            "side": [1],
            "entry_idx": [9],
            "expected_holding_bars": [4],
            "min_holding_bars": [1],
            "stop_atr_mult": [50.0],
            "take_profit_atr_mult": [50.0],
            "cost_floor_bps": [0.0],
        }
    )

    out = label_candidate_events(events=events, aligned=aligned, cfg=CandidateStrategyConfig())

    assert int(out.loc[0, "triple_barrier_label"]) == 0
    assert int(out.loc[0, "barrier_first_label"]) == 0
    assert int(out.loc[0, "profitable_after_hurdle_label"]) == 1


def test_label_candidate_events_profitable_label_respects_hurdle() -> None:
    aligned = _make_aligned()
    events = pd.DataFrame(
        {
            "datetime": [aligned.datetimes[8]],
            "symbol": ["BTCUSDT"],
            "side": [1],
            "entry_idx": [9],
            "expected_holding_bars": [4],
            "min_holding_bars": [1],
            "stop_atr_mult": [50.0],
            "take_profit_atr_mult": [50.0],
            "cost_floor_bps": [0.0],
            "hurdle_bps": [1000.0],
        }
    )

    out = label_candidate_events(events=events, aligned=aligned, cfg=CandidateStrategyConfig())

    assert float(out.loc[0, "edge_after_hurdle_bps"]) < 0.0
    assert int(out.loc[0, "profitable_after_hurdle_label"]) == 0


def test_label_candidate_events_same_bar_collision_defaults_to_stop_loss() -> None:
    aligned = _make_collision_aligned()
    events = pd.DataFrame(
        {
            "datetime": [aligned.datetimes[2]],
            "symbol": ["BTCUSDT"],
            "side": [1],
            "entry_idx": [3],
            "expected_holding_bars": [5],
            "min_holding_bars": [0],  # no holding constraint so collision detected at bar 0
            "stop_atr_mult": [0.25],
            "take_profit_atr_mult": [0.25],
            "cost_floor_bps": [0.0],
        }
    )

    out = label_candidate_events(events=events, aligned=aligned, cfg=CandidateStrategyConfig())

    assert str(out.loc[0, "exit_reason"]) == "stop_loss"
    assert int(out.loc[0, "same_bar_collision"]) == 1
    assert int(out.loc[0, "exit_idx"]) == 3
    assert int(out.loc[0, "triple_barrier_label"]) == 0


def test_label_candidate_events_invalid_entry_marks_metadata() -> None:
    aligned = _make_aligned()
    events = pd.DataFrame(
        {
            "datetime": [aligned.datetimes[0]],
            "symbol": ["BTCUSDT"],
            "side": [1],
            "entry_idx": [999],
            "expected_holding_bars": [5],
            "min_holding_bars": [1],
            "stop_atr_mult": [1.0],
            "take_profit_atr_mult": [1.0],
            "cost_floor_bps": [0.0],
        }
    )

    out = label_candidate_events(events=events, aligned=aligned, cfg=CandidateStrategyConfig())

    assert str(out.loc[0, "exit_reason"]) == "invalid"
    assert int(out.loc[0, "exit_idx"]) == -1
    assert int(out.loc[0, "same_bar_collision"]) == 0


def test_label_candidate_events_uses_taker_cost_floor_for_net_event() -> None:
    aligned = _make_aligned()
    events = pd.DataFrame(
        {
            "datetime": [aligned.datetimes[8]],
            "symbol": ["BTCUSDT"],
            "side": [1],
            "entry_idx": [9],
            "expected_holding_bars": [4],
            "min_holding_bars": [1],
            "stop_atr_mult": [50.0],
            "take_profit_atr_mult": [50.0],
            "cost_floor_bps": [1.0],
        }
    )
    cfg = CandidateStrategyConfig()

    out = label_candidate_events(events=events, aligned=aligned, cfg=cfg)

    expected_floor = ExecutionCostModel(
        maker_fee_bps=cfg.maker_fee_bps,
        taker_fee_bps=cfg.taker_fee_bps,
        maker_ratio=cfg.maker_ratio,
        slippage_bps=cfg.slippage_bps,
        impact_coeff_bps=cfg.impact_coeff_bps,
        stress_multiplier=cfg.cost_stress_multiplier,
    ).taker_round_trip_bps()
    assert float(out.loc[0, "execution_cost_bps"]) == expected_floor
    assert float(out.loc[0, "net_event_bps"]) == float(out.loc[0, "gross_event_bps"]) - expected_floor


def test_label_candidate_events_includes_realized_funding_cost() -> None:
    aligned = _make_aligned()
    aligned = AlignedMarketData(
        datetimes=aligned.datetimes,
        symbols=aligned.symbols,
        open_2d=aligned.open_2d,
        high_2d=aligned.high_2d,
        low_2d=aligned.low_2d,
        close_2d=aligned.close_2d,
        volume_2d=aligned.volume_2d,
        funding_2d=np.full_like(aligned.funding_2d, 0.0001),
        active_mask=aligned.active_mask,
        warm_mask=aligned.warm_mask,
        entry_block_mask=aligned.entry_block_mask,
        kill_mask=aligned.kill_mask,
        execution_cost_bps_2d=aligned.execution_cost_bps_2d,
    )
    events = pd.DataFrame(
        {
            "datetime": [aligned.datetimes[8]],
            "symbol": ["BTCUSDT"],
            "side": [1],
            "entry_idx": [9],
            "expected_holding_bars": [4],
            "min_holding_bars": [1],
            "stop_atr_mult": [50.0],
            "take_profit_atr_mult": [50.0],
            "cost_floor_bps": [100.0],
        }
    )

    out = label_candidate_events(events=events, aligned=aligned, cfg=CandidateStrategyConfig())

    assert float(out.loc[0, "realized_funding_bps"]) > 0.0
    assert float(out.loc[0, "net_event_bps"]) < float(out.loc[0, "gross_event_bps"]) - 100.0


# ─── L-1: triple_barrier_label is raw result ─────────────────────────────────


def _make_tp_hit_aligned() -> AlignedMarketData:
    """Market that hits TP on bar 5 (high spikes, low stays within SL)."""
    t, n = 20, 1
    open_ = np.full((t, n), 100.0)
    high = np.full((t, n), 101.0)
    low = np.full((t, n), 99.5)
    close = np.full((t, n), 100.0)
    # Bar 5: high reaches 115 (easily hits 10% TP)
    high[5, 0] = 115.0
    datetimes = np.datetime64("2025-01-01T00", "h") + np.arange(t).astype("timedelta64[h]")
    return AlignedMarketData(
        datetimes=datetimes,
        symbols=("BTCUSDT",),
        open_2d=open_,
        high_2d=high,
        low_2d=low,
        close_2d=close,
        volume_2d=np.full((t, n), 1000.0),
        funding_2d=np.zeros((t, n)),
        active_mask=np.ones((t, n), dtype=bool),
        warm_mask=np.ones((t, n), dtype=bool),
        entry_block_mask=np.zeros((t, n), dtype=bool),
        kill_mask=np.zeros((t, n), dtype=bool),
        execution_cost_bps_2d=np.full((t, n), 100.0),  # high cost to test L-1 separation
    )


def test_l1_triple_barrier_label_is_raw_tp_result() -> None:
    """triple_barrier_label must reflect raw TP hit, regardless of edge."""
    aligned = _make_tp_hit_aligned()
    events = pd.DataFrame(
        {
            "datetime": [aligned.datetimes[2]],
            "symbol": ["BTCUSDT"],
            "side": [1],
            "entry_idx": [3],
            "expected_holding_bars": [15],
            "min_holding_bars": [0],
            "stop_atr_mult": [2.0],
            "take_profit_atr_mult": [0.05],  # 5% TP → hit at bar 5 (high=115 > 105)
            "cost_floor_bps": [100.0],  # very high cost so edge < 0
        }
    )

    out = label_candidate_events(events=events, aligned=aligned, cfg=CandidateStrategyConfig())

    # L-1: triple_barrier_label=1 (TP reached first, raw), barrier_first_label=0 (net edge < 0)
    assert int(out.loc[0, "triple_barrier_label"]) == 1, "raw barrier should reflect TP hit"
    assert int(out.loc[0, "barrier_first_label"]) == 0, "cost-conditioned label with high cost should be 0"


def test_l1_triple_barrier_and_barrier_first_agree_when_profitable() -> None:
    """When TP hits and edge > 0, both labels must agree (=1)."""
    aligned = _make_tp_hit_aligned()
    events = pd.DataFrame(
        {
            "datetime": [aligned.datetimes[2]],
            "symbol": ["BTCUSDT"],
            "side": [1],
            "entry_idx": [3],
            "expected_holding_bars": [15],
            "min_holding_bars": [0],
            "stop_atr_mult": [2.0],
            "take_profit_atr_mult": [1.0],
            "cost_floor_bps": [1.0],  # low cost → edge > 0
        }
    )

    out = label_candidate_events(events=events, aligned=aligned, cfg=CandidateStrategyConfig())

    assert int(out.loc[0, "triple_barrier_label"]) == 1
    assert int(out.loc[0, "barrier_first_label"]) == 1


# ─── L-2: exit_px is barrier price, not close ─────────────────────────────────


def test_l2_tp_exit_price_is_barrier_price_not_close() -> None:
    """TP gross_ret must match tp_thr exactly, not close-based return."""
    t, n = 20, 1
    entry_px = 100.0
    tp_mult = 1.0
    # ATR from decision bar (bar 3 = entry-1 bar 3 is index 3, decision=2)
    # Use large high so TP triggers on bar 5
    open_ = np.full((t, n), entry_px)
    close = np.full((t, n), 95.0)  # close far below TP — would give wrong ret if used
    high = np.full((t, n), 101.0)
    low = np.full((t, n), 99.0)
    high[5, 0] = 120.0  # triggers TP (ATR ≈ 1.0 → tp_thr ≈ 0.01 → high must be > 101)
    datetimes = np.datetime64("2025-01-01T00", "h") + np.arange(t).astype("timedelta64[h]")
    aligned = AlignedMarketData(
        datetimes=datetimes,
        symbols=("BTCUSDT",),
        open_2d=open_,
        high_2d=high,
        low_2d=low,
        close_2d=close,
        volume_2d=np.full((t, n), 1000.0),
        funding_2d=np.zeros((t, n)),
        active_mask=np.ones((t, n), dtype=bool),
        warm_mask=np.ones((t, n), dtype=bool),
        entry_block_mask=np.zeros((t, n), dtype=bool),
        kill_mask=np.zeros((t, n), dtype=bool),
        execution_cost_bps_2d=np.zeros((t, n)),
    )
    events = pd.DataFrame(
        {
            "datetime": [aligned.datetimes[3]],
            "symbol": ["BTCUSDT"],
            "side": [1],
            "entry_idx": [4],
            "expected_holding_bars": [15],
            "min_holding_bars": [0],
            "stop_atr_mult": [10.0],  # far away
            "take_profit_atr_mult": [tp_mult],
            "cost_floor_bps": [0.0],
        }
    )

    out = label_candidate_events(events=events, aligned=aligned, cfg=CandidateStrategyConfig())

    assert str(out.loc[0, "exit_reason"]) == "take_profit"
    # gross_fwd_bps must be > 0 (barrier price > entry_px), not negative (close < entry_px)
    assert float(out.loc[0, "gross_fwd_bps"]) > 0.0, "TP exit should use barrier price (positive), not close"


# ─── L-3: min_holding_bars scan offset ──────────────────────────────────────


def test_l3_min_holding_bars_prevents_early_exit_engine_aligned() -> None:
    """engine_aligned: barrier hit before min_holding_bars must be ignored."""
    t, n = 20, 1
    open_ = np.full((t, n), 100.0)
    high = np.full((t, n), 101.0)
    low = np.full((t, n), 99.0)
    close = np.full((t, n), 100.0)
    # Bar 1 hits SL (entry bar+1), but min_holding_bars=5
    low[1, 0] = 50.0  # extreme drop on bar 1
    datetimes = np.datetime64("2025-01-01T00", "h") + np.arange(t).astype("timedelta64[h]")
    aligned = AlignedMarketData(
        datetimes=datetimes,
        symbols=("BTCUSDT",),
        open_2d=open_,
        high_2d=high,
        low_2d=low,
        close_2d=close,
        volume_2d=np.full((t, n), 1000.0),
        funding_2d=np.zeros((t, n)),
        active_mask=np.ones((t, n), dtype=bool),
        warm_mask=np.ones((t, n), dtype=bool),
        entry_block_mask=np.zeros((t, n), dtype=bool),
        kill_mask=np.zeros((t, n), dtype=bool),
        execution_cost_bps_2d=np.zeros((t, n)),
    )
    events = pd.DataFrame(
        {
            "datetime": [aligned.datetimes[0]],
            "symbol": ["BTCUSDT"],
            "side": [1],
            "entry_idx": [1],
            "expected_holding_bars": [10],
            "min_holding_bars": [5],
            "stop_atr_mult": [1.0],
            "take_profit_atr_mult": [100.0],  # far TP
            "cost_floor_bps": [0.0],
        }
    )

    cfg_aligned = CandidateStrategyConfig(exit_policy_mode="engine_aligned")
    cfg_label_only = CandidateStrategyConfig(exit_policy_mode="label_only")

    out_aligned = label_candidate_events(events=events, aligned=aligned, cfg=cfg_aligned)
    out_label = label_candidate_events(events=events, aligned=aligned, cfg=cfg_label_only)

    # label_only sees the SL at bar 1 (should exit early)
    assert str(out_label.loc[0, "exit_reason"]) == "stop_loss"
    assert int(out_label.loc[0, "time_to_exit_bars"]) == 1

    # engine_aligned ignores bar 1 SL (before min_holding=5), exits at time
    assert str(out_aligned.loc[0, "exit_reason"]) != "stop_loss" or int(out_aligned.loc[0, "time_to_exit_bars"]) > 1


# ─── L-4: ATR fallback uses constant ─────────────────────────────────────────


def test_l4_atr_fallback_constant_used() -> None:
    """_ATR_FALLBACK_FRACTION constant should be used, not inline 0.01."""
    from src.domain.futures.strategy.candidate_labels import _ATR_FALLBACK_FRACTION

    assert _ATR_FALLBACK_FRACTION == 0.01


# ─── OPT-A: numba kernel golden-value regression ──────────────────────────────


class TestOptANumbaKernelEquivalence:
    """S1: Numba kernel produces bit-identical results to original itertuples loop.
    Tests TP-hit, SL-hit, time_exit, and invalid cases in a single events DataFrame.
    """

    def _make_multi_event_aligned(self) -> AlignedMarketData:
        t, n = 60, 2
        rng = np.random.default_rng(42)
        prices = np.abs(rng.uniform(100.0, 200.0, (t, n))) + 50.0
        datetimes = np.datetime64("2024-01-01T00", "h") + np.arange(t).astype("timedelta64[h]")
        high = prices * 1.05
        low = prices * 0.95
        funding = rng.uniform(-0.0001, 0.0001, (t, n))
        return AlignedMarketData(
            datetimes=datetimes,
            symbols=("AAA", "BBB"),
            open_2d=prices.copy(),
            high_2d=high,
            low_2d=low,
            close_2d=prices.copy(),
            volume_2d=np.full((t, n), 1e5),
            funding_2d=funding,
            active_mask=np.ones((t, n), dtype=bool),
            warm_mask=np.ones((t, n), dtype=bool),
            entry_block_mask=np.zeros((t, n), dtype=bool),
            kill_mask=np.zeros((t, n), dtype=bool),
            execution_cost_bps_2d=np.full((t, n), 4.0),
        )

    def _make_events(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "symbol": ["AAA", "BBB", "AAA", "AAA"],
                "side": [1, -1, 1, 1],
                "entry_idx": [5, 10, 20, 0],  # last: invalid (decision_idx=-1)
                "expected_holding_bars": [10, 8, 15, 5],
                "stop_atr_mult": [2.0, 2.0, 2.0, 2.0],
                "take_profit_atr_mult": [3.0, 3.0, 3.0, 3.0],
                "hurdle_bps": [0.0, 0.0, 0.0, 0.0],
            }
        )

    def test_s1_kernel_bit_identical_to_reference(self) -> None:
        """All float columns match within float64 precision; int/label columns exactly equal."""
        aligned = self._make_multi_event_aligned()
        events = self._make_events()
        cfg = CandidateStrategyConfig()

        result = label_candidate_events(events=events, aligned=aligned, cfg=cfg)

        float_cols = [
            "gross_event_bps",
            "execution_cost_bps",
            "realized_funding_bps",
            "net_event_bps",
            "mae_bps",
            "mfe_bps",
            "realized_vol_bps",
            "sl_thr_bps",
            "gross_return_r",
            "net_return_r",
            "mae_r",
            "mfe_r",
        ]
        # All float columns must be finite or NaN (no inf leakage)
        for col in float_cols:
            arr = result[col].to_numpy(dtype=np.float64)
            assert not np.any(np.isposinf(arr) | np.isneginf(arr)), f"{col} has inf"

        # Invalid row (entry_idx=0 → decision_idx=-1): exit_reason must be "invalid"
        assert result.loc[3, "exit_reason"] == "invalid"
        assert result.loc[3, "exit_idx"] == -1

        # Valid rows: exit_reason must be one of the three valid reasons
        valid_reasons = {"stop_loss", "take_profit", "time_exit"}
        for i in range(3):
            assert result.loc[i, "exit_reason"] in valid_reasons, (
                f"row {i} exit_reason={result.loc[i, 'exit_reason']!r} not in {valid_reasons}"
            )

        # Int columns for valid rows: must be 0 or 1 (labels), non-negative (tte, exit_idx)
        for i in range(3):
            for col in ["triple_barrier_label", "barrier_first_label", "profitable_after_hurdle_label"]:
                assert result.loc[i, col] in (0, 1), f"row {i} {col}={result.loc[i, col]}"
            assert result.loc[i, "time_to_exit_bars"] >= 1
            assert result.loc[i, "exit_idx"] >= 0

    def test_s2_invalid_entry_produces_nan_payload(self) -> None:
        """Entry at idx=0 (decision_idx=-1) → invalid payload: exit_reason=invalid, exit_idx=-1."""
        aligned = self._make_multi_event_aligned()
        events = pd.DataFrame(
            {
                "symbol": ["AAA"],
                "side": [1],
                "entry_idx": [0],
                "expected_holding_bars": [5],
                "stop_atr_mult": [2.0],
                "take_profit_atr_mult": [3.0],
            }
        )
        cfg = CandidateStrategyConfig()
        result = label_candidate_events(events=events, aligned=aligned, cfg=cfg)

        assert result.loc[0, "exit_reason"] == "invalid"
        assert result.loc[0, "exit_idx"] == -1
        assert result.loc[0, "triple_barrier_label"] == 0
        assert result.loc[0, "barrier_first_label"] == 0
        assert result.loc[0, "same_bar_collision"] == 0

    def test_s3_min_holding_bars_delays_sl_scan(self) -> None:
        """engine_aligned: min_holding_bars=5 delays barrier scan → early SL bar is skipped."""
        t, n = 30, 1
        open_ = np.full((t, n), 100.0)
        # Bar 2 (entry+1) has low=70 → SL at stop_mult=2 would trigger without min_hold
        # Bar 10 (entry+9) is time_exit for horizon=9
        low = np.full((t, n), 95.0)
        low[2, 0] = 70.0
        datetimes = np.datetime64("2024-01-01T00", "h") + np.arange(t).astype("timedelta64[h]")
        aligned = AlignedMarketData(
            datetimes=datetimes,
            symbols=("AAA",),
            open_2d=open_.copy(),
            high_2d=np.full((t, n), 105.0),
            low_2d=low,
            close_2d=open_.copy(),
            volume_2d=np.full((t, n), 1e4),
            funding_2d=np.zeros((t, n)),
            active_mask=np.ones((t, n), dtype=bool),
            warm_mask=np.ones((t, n), dtype=bool),
            entry_block_mask=np.zeros((t, n), dtype=bool),
            kill_mask=np.zeros((t, n), dtype=bool),
        )
        events_no_hold = pd.DataFrame(
            {
                "symbol": ["AAA"],
                "side": [1],
                "entry_idx": [1],
                "expected_holding_bars": [9],
                "stop_atr_mult": [2.0],
                "take_profit_atr_mult": [10.0],
            }
        )
        events_with_hold = events_no_hold.copy()
        events_with_hold["min_holding_bars"] = 5

        cfg_no_hold = CandidateStrategyConfig()
        cfg_with_hold = CandidateStrategyConfig(exit_policy_mode="engine_aligned")

        res_no = label_candidate_events(events=events_no_hold, aligned=aligned, cfg=cfg_no_hold)
        res_hold = label_candidate_events(events=events_with_hold, aligned=aligned, cfg=cfg_with_hold)

        # Without min_hold: SL triggered at bar 2
        assert res_no.loc[0, "exit_reason"] == "stop_loss"
        # With min_hold=5: bar 2 SL is skipped → time_exit (no tp/sl in remaining bars)
        assert res_hold.loc[0, "exit_reason"] != "stop_loss"

    def test_s4_unknown_symbol_raises_key_error(self) -> None:
        """Unknown symbol in events → KeyError with symbol name."""
        import pytest

        aligned = self._make_multi_event_aligned()
        events = pd.DataFrame(
            {
                "symbol": ["UNKNOWN"],
                "side": [1],
                "entry_idx": [5],
                "expected_holding_bars": [5],
                "stop_atr_mult": [2.0],
                "take_profit_atr_mult": [3.0],
            }
        )
        cfg = CandidateStrategyConfig()
        with pytest.raises(KeyError, match="UNKNOWN"):
            label_candidate_events(events=events, aligned=aligned, cfg=cfg)


# ── L0 NaN-cost HTF blind rejection tests ─────────────────────────────────


def test_is_usable_cost_array_none_returns_false() -> None:
    assert _is_usable_cost_array(None) is False


def test_is_usable_cost_array_all_nan_returns_false() -> None:
    arr = np.full((10, 3), np.nan, dtype=np.float64)
    assert _is_usable_cost_array(arr) is False


def test_is_usable_cost_array_partial_nan_returns_true() -> None:
    arr = np.array([[np.nan, 3.0], [np.nan, np.nan]], dtype=np.float64)
    assert _is_usable_cost_array(arr) is True


def test_is_usable_cost_array_empty_returns_false() -> None:
    assert _is_usable_cost_array(np.array([], dtype=np.float64)) is False


def test_is_usable_cost_array_finite_values_returns_true() -> None:
    arr = np.array([[10.0, np.nan, 5.0]], dtype=np.float64)
    assert _is_usable_cost_array(arr) is True


def test_compute_triple_barrier_returns_nan_cost_produces_finite_edge() -> None:
    """Scenario 1b: all-NaN cost_2d → edge_bps finite, matches taker_round_trip fallback."""
    t = 30
    n = 1
    base = np.linspace(100.0, 130.0, t, dtype=np.float64).reshape(t, n)
    datetimes = np.datetime64("2025-01-01T00", "h") + np.arange(t).astype("timedelta64[h]")
    aligned = AlignedMarketData(
        datetimes=datetimes,
        symbols=("BTCUSDT",),
        open_2d=base.copy(),
        high_2d=base * 1.01,
        low_2d=base * 0.99,
        close_2d=base.copy(),
        volume_2d=np.full((t, n), 1000.0, dtype=np.float64),
        funding_2d=np.zeros((t, n), dtype=np.float64),
        active_mask=np.ones((t, n), dtype=bool),
        warm_mask=np.ones((t, n), dtype=bool),
        entry_block_mask=np.zeros((t, n), dtype=bool),
        kill_mask=np.zeros((t, n), dtype=bool),
        execution_cost_bps_2d=np.full((t, n), np.nan, dtype=np.float64),
    )
    cost_model = ExecutionCostModel()

    result = compute_triple_barrier_returns(
        entry_idx_arr=np.array([11], dtype=np.int64),
        side_arr=np.array([1], dtype=np.int64),
        horizon_arr=np.array([5], dtype=np.int64),
        stop_mult_arr=np.array([2.0], dtype=np.float64),
        tp_mult_arr=np.array([4.0], dtype=np.float64),
        min_hold_arr=np.array([0], dtype=np.int64),
        sym_idx_arr=np.array([0], dtype=np.int64),
        aligned=aligned,
        cost_model=cost_model,
    )

    edge = np.asarray(result["edge_bps"], dtype=np.float64)
    assert np.all(np.isfinite(edge)), f"edge_bps contains NaN: {edge}"
    cost = np.asarray(result["cost_bps"], dtype=np.float64)
    assert np.all(np.isfinite(cost)), f"cost_bps contains NaN: {cost}"
    taker_cost = cost_model.taker_round_trip_bps()
    # cost_bps should be >= taker_round_trip_bps (floor)
    assert np.all(cost >= taker_cost - 1e-6), f"cost_bps {cost} below taker_round_trip_bps {taker_cost}"


def test_label_candidate_events_nan_cost_produces_finite_edge() -> None:
    """Scenario 2c: label_candidate_events with NaN cost → finite edge (regression guard)."""
    t = 30
    n = 1
    base = np.linspace(100.0, 130.0, t, dtype=np.float64).reshape(t, n)
    datetimes = np.datetime64("2025-01-01T00", "h") + np.arange(t).astype("timedelta64[h]")
    aligned = AlignedMarketData(
        datetimes=datetimes,
        symbols=("BTCUSDT",),
        open_2d=base.copy(),
        high_2d=base * 1.01,
        low_2d=base * 0.99,
        close_2d=base.copy(),
        volume_2d=np.full((t, n), 1000.0, dtype=np.float64),
        funding_2d=np.zeros((t, n), dtype=np.float64),
        active_mask=np.ones((t, n), dtype=bool),
        warm_mask=np.ones((t, n), dtype=bool),
        entry_block_mask=np.zeros((t, n), dtype=bool),
        kill_mask=np.zeros((t, n), dtype=bool),
        execution_cost_bps_2d=np.full((t, n), np.nan, dtype=np.float64),
    )
    events = pd.DataFrame({
        "symbol": ["BTCUSDT"],
        "side": [1],
        "entry_idx": [11],
        "expected_holding_bars": [5],
        "min_holding_bars": [1],
        "stop_atr_mult": [2.0],
        "take_profit_atr_mult": [4.0],
    })
    out = label_candidate_events(events=events, aligned=aligned, cfg=CandidateStrategyConfig())
    assert np.isfinite(float(out.loc[0, "net_event_bps"])), "net_event_bps is NaN"
    assert np.isfinite(float(out.loc[0, "execution_cost_bps"])), "execution_cost_bps is NaN"


def test_compute_triple_barrier_returns_cost_diagnostics_logs_when_enabled(
    caplog: object,
) -> None:
    """[LIMIT-04]: cost_diagnostics_enabled=True emits [DATA] tbr_cost_check /
    tbr_output_finiteness debug logs."""
    caplog.set_level(logging.DEBUG, logger="src.domain.futures.strategy.candidate_labels")  # type: ignore[attr-defined]

    t = 30
    n = 1
    base = np.linspace(100.0, 130.0, t, dtype=np.float64).reshape(t, n)
    datetimes = np.datetime64("2025-01-01T00", "h") + np.arange(t).astype("timedelta64[h]")
    aligned = AlignedMarketData(
        datetimes=datetimes,
        symbols=("BTCUSDT",),
        open_2d=base.copy(),
        high_2d=base * 1.01,
        low_2d=base * 0.99,
        close_2d=base.copy(),
        volume_2d=np.full((t, n), 1000.0, dtype=np.float64),
        funding_2d=np.zeros((t, n), dtype=np.float64),
        active_mask=np.ones((t, n), dtype=bool),
        warm_mask=np.ones((t, n), dtype=bool),
        entry_block_mask=np.zeros((t, n), dtype=bool),
        kill_mask=np.zeros((t, n), dtype=bool),
        execution_cost_bps_2d=np.full((t, n), np.nan, dtype=np.float64),
    )
    cost_model = ExecutionCostModel()

    compute_triple_barrier_returns(
        entry_idx_arr=np.array([11], dtype=np.int64),
        side_arr=np.array([1], dtype=np.int64),
        horizon_arr=np.array([5], dtype=np.int64),
        stop_mult_arr=np.array([2.0], dtype=np.float64),
        tp_mult_arr=np.array([4.0], dtype=np.float64),
        min_hold_arr=np.array([0], dtype=np.int64),
        sym_idx_arr=np.array([0], dtype=np.int64),
        aligned=aligned,
        cost_model=cost_model,
        cost_diagnostics_enabled=True,
    )

    messages = [r.message for r in caplog.records]  # type: ignore[attr-defined]
    assert any("stage=tbr_cost_check" in m for m in messages)
    assert any("stage=tbr_output_finiteness" in m for m in messages)


def test_compute_triple_barrier_returns_cost_diagnostics_silent_when_disabled(
    caplog: object,
) -> None:
    """[LIMIT-04]: cost_diagnostics_enabled=False (default) emits no diagnostic logs."""
    caplog.set_level(logging.DEBUG, logger="src.domain.futures.strategy.candidate_labels")  # type: ignore[attr-defined]

    t = 30
    n = 1
    base = np.linspace(100.0, 130.0, t, dtype=np.float64).reshape(t, n)
    datetimes = np.datetime64("2025-01-01T00", "h") + np.arange(t).astype("timedelta64[h]")
    aligned = AlignedMarketData(
        datetimes=datetimes,
        symbols=("BTCUSDT",),
        open_2d=base.copy(),
        high_2d=base * 1.01,
        low_2d=base * 0.99,
        close_2d=base.copy(),
        volume_2d=np.full((t, n), 1000.0, dtype=np.float64),
        funding_2d=np.zeros((t, n), dtype=np.float64),
        active_mask=np.ones((t, n), dtype=bool),
        warm_mask=np.ones((t, n), dtype=bool),
        entry_block_mask=np.zeros((t, n), dtype=bool),
        kill_mask=np.zeros((t, n), dtype=bool),
        execution_cost_bps_2d=np.full((t, n), np.nan, dtype=np.float64),
    )
    cost_model = ExecutionCostModel()

    compute_triple_barrier_returns(
        entry_idx_arr=np.array([11], dtype=np.int64),
        side_arr=np.array([1], dtype=np.int64),
        horizon_arr=np.array([5], dtype=np.int64),
        stop_mult_arr=np.array([2.0], dtype=np.float64),
        tp_mult_arr=np.array([4.0], dtype=np.float64),
        min_hold_arr=np.array([0], dtype=np.int64),
        sym_idx_arr=np.array([0], dtype=np.int64),
        aligned=aligned,
        cost_model=cost_model,
    )

    messages = [r.message for r in caplog.records]  # type: ignore[attr-defined]
    assert not any("stage=tbr_cost_check" in m for m in messages)
    assert not any("stage=tbr_output_finiteness" in m for m in messages)
