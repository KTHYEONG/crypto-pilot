from __future__ import annotations

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps
from src.domain.futures.strategy.candidate_contracts import (
    ValidatedSignalBatch,
    ValidatedSignalEvent,
)
from src.domain.futures.strategy.tiered_workflow.awf_sim import (
    _run_awf_simulation,
    build_l2_simulation_cache,
)
from src.domain.futures.strategy.tiered_workflow.dataclasses import (
    Layer2AllocationConfig,
)
from src.domain.futures.strategy.walk_forward import WFFold


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


def _make_config() -> Layer2AllocationConfig:
    return Layer2AllocationConfig(
        k_rank=2,
        rank_buffer=0,
        kelly_fraction=0.5,
        no_trade_band=0.0,
        rebalance_bars=1,
    )


def _make_folds() -> tuple[WFFold, ...]:
    return (WFFold(fit_start=0, fit_end=1, cal_start=1, cal_end=1, oos_start=1, oos_end=4),)


class TestAwfSimInstrumentation:
    """Spec: docs/specs/l2-parity-instrumentation.md

    S1-S5 test scenarios for _run_awf_simulation fingerprint debug logging.
    """

    def test_run_awf_simulation_fingerprint_is_deterministic(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """S1: 동일 입력 2회 호출 → [AWF-SIM-FP] 2줄, rets_fp 해시 동일."""
        caplog.set_level(logging.DEBUG)
        aligned = _make_aligned()
        signal_batch = _make_signal_batch()
        config = _make_config()
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

        # 두 번째 호출을 위해 동일 입력 재구성
        cache2 = build_l2_simulation_cache(aligned, signal_batch, "4h")
        _run_awf_simulation(
            cache=cache2,
            signal_batch=signal_batch,
            aligned=aligned,
            awf_folds=awf_folds,
            config=config,
            caps=caps,
            tf="4h",
        )

        fp_records = [r for r in caplog.records if "[AWF-SIM-FP]" in str(r.message)]
        assert len(fp_records) == 2, f"Expected 2 [AWF-SIM-FP] records, got {len(fp_records)}"

        rets_fp_values = [
            part.split("=")[1] for rec in fp_records for part in str(rec.message).split() if part.startswith("rets_fp=")
        ]
        assert len(rets_fp_values) == 2
        assert rets_fp_values[0] == rets_fp_values[1], (
            f"Fingerprints differ: {rets_fp_values[0]} vs {rets_fp_values[1]}"
        )

    def test_run_awf_simulation_logs_sim_origin_tag(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """S2: sim_origin='champion_eval' 전달 → 로그에 origin=champion_eval 포함."""
        caplog.set_level(logging.DEBUG)
        aligned = _make_aligned()
        signal_batch = _make_signal_batch()
        config = _make_config()
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
            sim_origin="champion_eval",
        )

        fp_records = [r for r in caplog.records if "[AWF-SIM-FP]" in str(r.message)]
        assert len(fp_records) >= 1
        assert "origin=champion_eval" in str(fp_records[0].message)

    def test_run_awf_simulation_logs_per_fold_oos_bars(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """S3: 2개 fold(oos 길이 상이) 입력 → oos_bars에 실제 fold 길이 일치."""
        caplog.set_level(logging.DEBUG)
        aligned = _make_aligned(n_bars=30)
        signal_batch = _make_signal_batch(n_bars=30)
        config = _make_config()
        caps = PortfolioCaps(gross=2.0, per_symbol=1.0, net=1.0, beta=2.0, target_ann_vol=10.0)

        folds = (
            WFFold(fit_start=0, fit_end=2, cal_start=2, cal_end=2, oos_start=2, oos_end=6),
            WFFold(fit_start=0, fit_end=3, cal_start=3, cal_end=3, oos_start=3, oos_end=9),
        )
        cache = build_l2_simulation_cache(aligned, signal_batch, "4h")

        _run_awf_simulation(
            cache=cache,
            signal_batch=signal_batch,
            aligned=aligned,
            awf_folds=folds,
            config=config,
            caps=caps,
            tf="4h",
        )

        fp_records = [r for r in caplog.records if "[AWF-SIM-FP]" in str(r.message)]
        assert len(fp_records) >= 1
        msg = str(fp_records[0].message)
        assert "n_folds=2" in msg, f"Expected n_folds=2, got: {msg}"
        assert "oos_bars=[4, 6]" in msg, f"Expected oos_bars=[4, 6], got: {msg}"

    def test_run_awf_simulation_empty_folds_logs_gracefully(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """S4: awf_folds=() → crash 없이 n_folds=0 oos_bars=[] 로깅."""
        caplog.set_level(logging.DEBUG)
        aligned = _make_aligned(n_bars=10)
        signal_batch = _make_signal_batch(n_bars=10)
        config = _make_config()
        caps = PortfolioCaps(gross=2.0, per_symbol=1.0, net=1.0, beta=2.0, target_ann_vol=10.0)
        cache = build_l2_simulation_cache(aligned, signal_batch, "4h")

        _run_awf_simulation(
            cache=cache,
            signal_batch=signal_batch,
            aligned=aligned,
            awf_folds=(),
            config=config,
            caps=caps,
            tf="4h",
        )

        fp_records = [r for r in caplog.records if "[AWF-SIM-FP]" in str(r.message)]
        assert len(fp_records) >= 1
        msg = str(fp_records[0].message)
        assert "n_folds=0" in msg, f"Expected n_folds=0, got: {msg}"
        assert "oos_bars=[]" in msg, f"Expected oos_bars=[], got: {msg}"

    def test_run_awf_simulation_skips_fingerprint_when_debug_off(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """S5: DEBUG 비활성 시 [AWF-SIM-FP] 미출력."""
        caplog.set_level(logging.INFO)
        aligned = _make_aligned()
        signal_batch = _make_signal_batch()
        config = _make_config()
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

        fp_records = [r for r in caplog.records if "[AWF-SIM-FP]" in str(r.message)]
        assert len(fp_records) == 0, f"Expected 0 [AWF-SIM-FP] records when DEBUG is off, got {len(fp_records)}"
