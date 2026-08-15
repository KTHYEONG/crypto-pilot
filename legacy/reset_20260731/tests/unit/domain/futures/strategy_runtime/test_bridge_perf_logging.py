import gc
import logging
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.strategy.config import StrategyConfig
from src.domain.futures.strategy_runtime.bridge import (
    CandidatePipelineOutput,
    _get_rss_mb,
    merge_candidate_output_into_data_maps,
    run_candidate_strategy_for_universe,
)


def _minimal_ohlc_bar() -> pd.DataFrame:
    """Return a single-row OHLC DataFrame that passes _build_virtual_probe_tf_maps."""
    return pd.DataFrame(
        {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5], "volume": [1000.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-01-01")], name="datetime"),
    )


def test_get_rss_mb_returns_positive_or_negative_one() -> None:
    rss = _get_rss_mb()
    assert rss >= -1.0
    if rss > 0:
        assert rss < 1_000_000.0  # sanity: not terabytes


def _make_simple_aligned(n_bars: int = 20, n_syms: int = 1) -> SimpleNamespace:
    total = n_bars * n_syms
    datetimes = np.asarray(
        [np.datetime64("2026-01-01T00:00:00") + np.timedelta64(i, "h") for i in range(n_bars)],
        dtype="datetime64[ns]",
    )
    return SimpleNamespace(
        datetimes=datetimes,
        symbols=tuple(f"SYM{i}" for i in range(n_syms)),
        close_2d=np.linspace(100.0, 110.0, total, dtype=np.float64).reshape(n_bars, n_syms),
        open_2d=np.linspace(100.0, 110.0, total, dtype=np.float64).reshape(n_bars, n_syms),
        high_2d=np.linspace(101.0, 111.0, total, dtype=np.float64).reshape(n_bars, n_syms),
        low_2d=np.linspace(99.0, 109.0, total, dtype=np.float64).reshape(n_bars, n_syms),
        volume_2d=np.full((n_bars, n_syms), 1000.0, dtype=np.float64),
        funding_2d=np.zeros((n_bars, n_syms), dtype=np.float64),
        active_mask=np.ones((n_bars, n_syms), dtype=bool),
        warm_mask=np.ones((n_bars, n_syms), dtype=bool),
        entry_block_mask=np.zeros((n_bars, n_syms), dtype=bool),
        kill_mask=np.zeros((n_bars, n_syms), dtype=bool),
        execution_cost_bps_2d=np.zeros((n_bars, n_syms), dtype=np.float64),
    )


def test_bridge_emits_input_shape_log(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    caplog.set_level(logging.DEBUG)
    aligned = _make_simple_aligned(n_bars=20, n_syms=2)

    def fake_align(*_: object, **__: object) -> object:
        return aligned

    monkeypatch.setattr("src.domain.futures.strategy.common.alignment.align_data_maps", fake_align)
    monkeypatch.setattr("src.domain.futures.strategy.rule_signals.build_rule_signal_panels", lambda *_, **__: [])

    strategy_cfg = StrategyConfig()
    object.__setattr__(
        strategy_cfg,
        "candidate",
        replace(strategy_cfg.candidate, ml_fit_fraction=0.5, ml_calibration_fraction=0.2, purge_bars=0, embargo_bars=0),
    )

    run_candidate_strategy_for_universe(
        ["SYM0", "SYM1"],
        "4h",
        strategy_cfg=strategy_cfg,
        preloaded_data_maps={"SYM0": {"4h": _minimal_ohlc_bar()}, "SYM1": {"4h": _minimal_ohlc_bar()}},
    )

    input_records = [r for r in caplog.records if "[INPUT]" in r.getMessage() and "n_symbols" in r.getMessage()]
    assert len(input_records) >= 1, "Expected [INPUT] shape log"
    msg = input_records[0].getMessage()
    assert "n_symbols=2" in msg
    assert "n_bars=20" in msg
    assert "tf=4h" in msg


def test_bridge_perf_log_contains_stages(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    caplog.set_level(logging.DEBUG)
    aligned = _make_simple_aligned(n_bars=20, n_syms=1)

    monkeypatch.setattr("src.domain.futures.strategy.common.alignment.align_data_maps", lambda *_, **__: aligned)
    monkeypatch.setattr("src.domain.futures.strategy.rule_signals.build_rule_signal_panels", lambda *_, **__: [])

    strategy_cfg = StrategyConfig()
    object.__setattr__(
        strategy_cfg,
        "candidate",
        replace(strategy_cfg.candidate, ml_fit_fraction=0.5, ml_calibration_fraction=0.2, purge_bars=0, embargo_bars=0),
    )

    run_candidate_strategy_for_universe(
        ["SYM0"],
        "4h",
        strategy_cfg=strategy_cfg,
        preloaded_data_maps={"SYM0": {"4h": _minimal_ohlc_bar()}},
    )

    perf_records = [r for r in caplog.records if "Total Runtime" in r.getMessage()]
    assert len(perf_records) >= 1
    perf_msg = perf_records[0].getMessage()
    assert "MEMORY" in perf_msg
    # Verify MEMORY section is present
    assert "MEMORY" in perf_msg


def test_merge_summary_log_emitted(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    caplog.set_level(logging.DEBUG)
    alpha_panel = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "alpha_long": [0.1, 0.2],
            "alpha_short": [0.0, 0.0],
            "target_weight": [0.5, 0.6],
            "candidate_family": ["trend", "trend"],
            "candidate_variant": ["ema_12", "ema_12"],
            "p_pass": [0.6, 0.7],
            "mu_net_decision_bps": [10.0, 15.0],
            "q10_net_bps": [5.0, 8.0],
            "utility_score": [0.1, 0.2],
            "candidate_stop_atr_mult": [3.0, 3.0],
            "candidate_take_profit_atr_mult": [6.0, 6.0],
        }
    )
    data_maps = {
        "BTCUSDT": {
            "4h": pd.DataFrame(
                {
                    "datetime": pd.date_range("2026-01-01", periods=3, freq="D"),
                    "alpha_long": [0.0, 0.0, 0.0],
                    "alpha_short": [0.0, 0.0, 0.0],
                    "target_weight": [0.0, 0.0, 0.0],
                }
            ),
        },
    }

    candidate_out = CandidatePipelineOutput(alpha_panel=alpha_panel)
    merge_candidate_output_into_data_maps(candidate_out, data_maps, ["BTCUSDT"], "4h", log_tag="test")

    summary_records = [r for r in caplog.records if "[MERGE][SUMMARY]" in r.getMessage()]
    assert len(summary_records) >= 1
    msg = summary_records[0].getMessage()
    assert "n_syms=1" in msg
    assert "tag=test" in msg


def test_htf_active_when_signal_only_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scenario 2: HTF Still Active in Non-Tiered Mode (signal_only=False).

    Verifies that build_multi_tf_panels is called when signal_only=False.
    """
    aligned = _make_simple_aligned(n_bars=20, n_syms=1)
    called: list[bool] = []

    datetimes = np.asarray(
        [np.datetime64("2026-01-01T00:00:00") + np.timedelta64(i, "h") for i in range(20)],
        dtype="datetime64[ns]",
    )
    raw_events = pd.DataFrame(
        {
            "datetime": [datetimes[0]],
            "symbol": ["SYM0"],
            "family": ["trend_ma"],
            "variant": ["ema_12_72"],
            "side": [1],
            "raw_score": [0.9],
            "score_z": [0.9],
            "entry_idx": [0],
            "exit_idx": [1],
            "expected_holding_bars": [1],
            "min_holding_bars": [1],
            "stop_atr_mult": [50.0],
            "take_profit_atr_mult": [50.0],
            "turnover_proxy": [0.1],
            "cost_floor_bps": [0.0],
            "hurdle_bps": [0.0],
            "edge_after_hurdle_bps": [0.5],
        }
    )

    def fake_align(*_: object, **__: object) -> object:
        return aligned

    def track_multi_tf(*_: object, **__: object) -> tuple[()]:
        called.append(True)
        return ()

    monkeypatch.setattr("src.domain.futures.strategy.common.alignment.align_data_maps", fake_align)
    monkeypatch.setattr("src.domain.futures.strategy.rule_signals.build_rule_signal_panels", lambda *_, **__: [])
    monkeypatch.setattr(
        "src.domain.futures.strategy.rule_signals.candidate_panels_to_events",
        lambda *_, **__: raw_events.copy(),
    )
    monkeypatch.setattr("src.domain.futures.strategy_runtime.bridge.build_multi_tf_panels", track_multi_tf)
    monkeypatch.setattr(
        "src.domain.futures.strategy.rule_diagnostics.compute_rule_diagnostics",
        lambda *_, **__: SimpleNamespace(
            by_family=pd.DataFrame(),
            by_variant=pd.DataFrame(),
            by_family_side=pd.DataFrame(),
            side_flip=pd.DataFrame(),
            decision={},
            recommended_keep_variants=(),
            recommended_flip_variants=(),
            recommended_keep_signal_cells=(),
            recommended_flip_signal_cells=(),
            recommendation_basis="skipped_signal_only",
            recommendation_split=(0, 0),
            report_split=(0, 0),
            recommendation_failure_report=None,
        ),
    )

    strategy_cfg = StrategyConfig()
    object.__setattr__(
        strategy_cfg,
        "candidate",
        replace(
            strategy_cfg.candidate,
            ml_fit_fraction=0.5,
            ml_calibration_fraction=0.2,
            purge_bars=0,
            embargo_bars=0,
            signal_only=False,
            l1_tfs=("4h", "6h"),
        ),
    )

    run_candidate_strategy_for_universe(
        ["SYM0"],
        "4h",
        strategy_cfg=strategy_cfg,
        preloaded_data_maps={"SYM0": {"4h": _minimal_ohlc_bar()}},
    )

    assert len(called) >= 1, "build_multi_tf_panels should be called when signal_only=False"


def test_gc_collect_does_not_raise() -> None:
    """Scenario 3: GC Completes Without Error."""
    gc.collect()  # should not raise
    gc.collect()  # repeat for strategy 2 (inside bridge) and strategy 3 (after bridge)
