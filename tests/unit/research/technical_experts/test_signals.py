from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.research.technical_experts.catalog import TECHNICAL_CANDIDATES, resolve_technical_candidate
from src.research.technical_experts.signals import generate_signal_events


def causal_ohlcv_fixture(n: int = 260) -> pd.DataFrame:
    """Causal 4h OHLCV/volume grid with monotonic prices and volume."""
    index = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    t = np.arange(n, dtype=np.float64)
    close = 100.0 + 0.05 * t + 5.0 * np.sin(t / 12.0)
    open_ = close - 0.2
    return pd.DataFrame({
        "open": open_,
        "high": np.maximum(open_, close) + 0.5,
        "low": np.minimum(open_, close) - 0.5,
        "close": close,
        "volume": 1000.0 + np.abs(np.sin(t / 5.0)) * 500.0,
    }, index=index)


class TestSignalEvents:
    def test_events_index_matches_input_index(self) -> None:
        frame = causal_ohlcv_fixture()
        candidate = resolve_technical_candidate("technical_rsi_trend_pullback_long_v1")
        events = generate_signal_events(frame, candidate)
        assert events.index.equals(frame.index)
        assert set(events.columns) == {"long_entry", "short_entry", "long_exit", "short_exit"}

    def test_signal_events_do_not_read_future_bars(self) -> None:
        # A decision at bar t must be unchanged when every bar after t is
        # mutated: indicator windows never read a future index.
        candidate = resolve_technical_candidate("technical_rsi_trend_pullback_long_v1")
        frame = causal_ohlcv_fixture()
        base = generate_signal_events(frame, candidate)
        mutated = frame.copy()
        mutated.iloc[220:] = mutated.iloc[220:] * 2.0
        after = generate_signal_events(mutated, candidate)
        pd.testing.assert_frame_equal(base.iloc[:220], after.iloc[:220])

    def test_ichimoku_cloud_has_no_forward_shift(self) -> None:
        # The cloud decision uses the current, non-forward span edge; events at
        # t never depend on bars after t.
        candidate = resolve_technical_candidate("technical_ichimoku_cloud_long_v1")
        frame = causal_ohlcv_fixture(n=300)
        base = generate_signal_events(frame, candidate)
        tail = frame.copy()
        tail.iloc[250:] = tail.iloc[250:] * 3.0
        after = generate_signal_events(tail, candidate)
        pd.testing.assert_frame_equal(base.iloc[:250], after.iloc[:250])

    def test_all_candidates_mask_opposite_side_entries(self) -> None:
        frame = causal_ohlcv_fixture(n=400)
        for candidate in TECHNICAL_CANDIDATES:
            events = generate_signal_events(frame, candidate)
            other = "short_entry" if candidate.side == "LONG" else "long_entry"
            assert not events[other].any(), candidate.return_source
            assert set(events.columns) == {"long_entry", "short_entry", "long_exit", "short_exit"}
            assert events.dtypes.eq(bool).all(), candidate.return_source

    def test_trending_series_activates_long_entry(self) -> None:
        n = 300
        index = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
        close = np.linspace(100.0, 300.0, n)
        frame = pd.DataFrame({
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(n, 1000.0),
        }, index=index)
        events = generate_signal_events(
            frame, resolve_technical_candidate("technical_ema_alignment_long_v1"),
        )
        assert events["long_entry"].any()
        assert not events["short_entry"].any()


class TestSignalEventsIntegrity:
    def test_missing_volume_fails_closed(self) -> None:
        frame = causal_ohlcv_fixture().drop(columns=["volume"])
        with pytest.raises(DataIntegrityError, match="volume"):
            generate_signal_events(
                frame, resolve_technical_candidate("technical_mfi_trend_pullback_long_v1"),
            )

    def test_nan_close_fails_closed(self) -> None:
        frame = causal_ohlcv_fixture()
        frame.loc[frame.index[5], "close"] = np.nan
        with pytest.raises(DataIntegrityError, match="close"):
            generate_signal_events(
                frame, resolve_technical_candidate("technical_rsi_trend_pullback_long_v1"),
            )

    def test_non_utc_index_fails_closed(self) -> None:
        frame = causal_ohlcv_fixture()
        frame.index = frame.index.tz_localize(None)
        with pytest.raises(DataIntegrityError, match="tz-aware"):
            generate_signal_events(
                frame, resolve_technical_candidate("technical_rsi_trend_pullback_long_v1"),
            )

    def test_non_monotonic_index_fails_closed(self) -> None:
        frame = causal_ohlcv_fixture(n=10)
        shuffled = frame.iloc[::-1]
        with pytest.raises(DataIntegrityError, match="monotonic"):
            generate_signal_events(
                shuffled, resolve_technical_candidate("technical_rsi_trend_pullback_long_v1"),
            )

    def test_insufficient_history_fails_closed(self) -> None:
        frame = causal_ohlcv_fixture(n=50)
        with pytest.raises(DataIntegrityError, match="at least 201"):
            generate_signal_events(
                frame, resolve_technical_candidate("technical_ema_alignment_long_v1"),
            )

    def test_unknown_family_fails_closed(self) -> None:
        from src.research.technical_experts.contracts import TechnicalCandidate

        bogus = TechnicalCandidate(
            "x", "technical_not_a_family_long_v1", "not_a_family", "LONG",
            {"a": 1}, 10,
        )
        with pytest.raises(ValueError, match="unknown technical family"):
            generate_signal_events(causal_ohlcv_fixture(n=50), bogus)


class TestNewFamilyGatedNotFreebied:
    def test_new_family_gated_not_freebied(self) -> None:
        # new_family_gated_not_freebied: Supertrend/Parabolic SAR/Keltner are
        # implemented as full families -- they generate signal events through
        # generate_signal_events() with the same event surface and opposite-side
        # masking -- but they are not yet admitted to the catalog, so they must
        # pass the same admission gate as existing families before any catalog
        # entry is added.
        from src.research.technical_experts.catalog import (
            TECHNICAL_EXPERT_FAMILIES,
            resolve_technical_candidate,
        )
        from src.research.technical_experts.contracts import TechnicalCandidate

        families = {
            "supertrend": {"period": 10, "mult": 3.0, "regime": 200},
            "parabolic_sar": {"step": 0.02, "max_step": 0.2, "regime": 200},
            "keltner_channel_breakout": {"period": 20, "mult": 2.0, "regime": 200},
        }
        frame = causal_ohlcv_fixture(n=400)
        for family, config in families.items():
            assert family not in TECHNICAL_EXPERT_FAMILIES
            for side in ("long", "short"):
                candidate = TechnicalCandidate(
                    f"c_{family}_{side}_v1",
                    f"technical_{family}_{side}_v1",
                    family,
                    side.upper(),
                    config,
                    201,
                )
                events = generate_signal_events(frame, candidate)
                assert events.index.equals(frame.index)
                assert set(events.columns) == {
                    "long_entry", "short_entry", "long_exit", "short_exit",
                }
                assert events.dtypes.eq(bool).all()
                other = "short_entry" if side == "long" else "long_entry"
                assert not events[other].any(), f"{family} {side}"
            with pytest.raises(ValueError, match="unknown or retired"):
                resolve_technical_candidate(f"technical_{family}_long_v1")


TREND_SCREEN_FAMILY_CONFIGS: dict[str, dict] = {
    "donchian_breakout": {"entry": 55, "exit": 20, "regime": 200},
    "chandelier_trend": {"period": 22, "mult": 3.0, "regime": 200},
    "aroon_trend": {"period": 25, "regime": 200},
    "vortex_trend": {"period": 14, "regime": 200},
    "hull_moving_average": {"period": 55, "regime": 200},
    "regression_slope": {"period": 63, "regime": 200},
    "atr_volatility_breakout": {"period": 20, "mult": 1.5, "regime": 200},
}


class TestTrendScreenFamilyCausality:
    @pytest.mark.parametrize("family", list(TREND_SCREEN_FAMILY_CONFIGS))
    def test_bgp_01_future_tail_invariance(self, family: str) -> None:
        """BGP-01: mutating bars after a cutoff cannot change events at/before it."""
        from src.research.technical_experts.contracts import TechnicalCandidate

        config = TREND_SCREEN_FAMILY_CONFIGS[family]
        candidate = TechnicalCandidate(
            f"technical_{family}_long_v1", f"technical_{family}_long_v1",
            family, "LONG", config, 201,
        )
        frame = causal_ohlcv_fixture(n=400)
        base = generate_signal_events(frame, candidate)
        mutated = frame.copy()
        mutated.iloc[320:] = mutated.iloc[320:] * 2.5
        after = generate_signal_events(mutated, candidate)
        pd.testing.assert_frame_equal(base.iloc[:320], after.iloc[:320])

    def test_trend_screen_families_mask_opposite_side(self) -> None:
        from src.research.technical_experts.contracts import TechnicalCandidate

        frame = causal_ohlcv_fixture(n=400)
        for family, config in TREND_SCREEN_FAMILY_CONFIGS.items():
            for side in ("long", "short"):
                candidate = TechnicalCandidate(
                    f"c_{family}_{side}_v1", f"technical_{family}_{side}_v1",
                    family, side.upper(), config, 201,
                )
                events = generate_signal_events(frame, candidate)
                other = "short_entry" if side == "long" else "long_entry"
                assert not events[other].any(), f"{family} {side}"
                assert set(events.columns) == {
                    "long_entry", "short_entry", "long_exit", "short_exit",
                }
                assert events.dtypes.eq(bool).all()
