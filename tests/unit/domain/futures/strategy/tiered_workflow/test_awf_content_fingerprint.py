from __future__ import annotations

import logging
from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.domain.futures.strategy.tiered_workflow.awf_sim import (
    _content_hash_array,
    _content_hash_cache,
    _content_hash_dataclass,
)
from src.domain.futures.strategy.tiered_workflow.dataclasses import (
    L2SimulationCache,
)


@dataclass
class _ArrDc:
    values: np.ndarray
    label: str = "x"


@dataclass
class _ScalarDc:
    a: int = 1
    b: float = 2.0


class TestContentHashArray:
    """S1 + S2: _content_hash_array 민감도 및 결정성."""

    def test_content_hash_array_detects_tiny_diff(self) -> None:
        a = np.arange(12, dtype=np.float64).reshape(3, 4)
        b = a.copy()
        b[0, 0] += 1e-9
        assert _content_hash_array(a) != _content_hash_array(b)

    def test_content_hash_array_deterministic_and_contiguous_safe(self) -> None:
        a = np.arange(12, dtype=np.float64).reshape(3, 4)
        h1 = _content_hash_array(a)
        h2 = _content_hash_array(a)
        assert h1 == h2
        h3 = _content_hash_array(a.T)
        assert isinstance(h3, str)
        assert len(h3) == 10

    def test_content_hash_array_non_array_fallback(self) -> None:
        assert len(_content_hash_array([1, 2, 3])) == 10
        assert _content_hash_array("hello") == _content_hash_array("hello")
        assert _content_hash_array(42) == _content_hash_array(42)


class TestContentHashDataclass:
    """S3 + S4: _content_hash_dataclass 배열필드 분리 및 None 처리."""

    def test_content_hash_dataclass_separates_array_fields(self) -> None:
        a = np.arange(12, dtype=np.float64).reshape(3, 4)
        b = a.copy()
        b[0, 0] += 1e-9
        dc_a = _ArrDc(values=a)
        dc_b = _ArrDc(values=b)
        h_a = _content_hash_dataclass(dc_a, array_fields=frozenset({"values"}))
        h_b = _content_hash_dataclass(dc_b, array_fields=frozenset({"values"}))
        assert h_a != h_b, "1e-9 diff in array field must produce different hash"

    def test_content_hash_dataclass_scalar_only(self) -> None:
        dc1 = _ScalarDc(a=1, b=2.0)
        dc2 = _ScalarDc(a=1, b=2.0)
        assert _content_hash_dataclass(dc1) == _content_hash_dataclass(dc2)
        dc3 = _ScalarDc(a=2, b=2.0)
        assert _content_hash_dataclass(dc1) != _content_hash_dataclass(dc3)

    def test_content_hash_dataclass_handles_none(self) -> None:
        assert _content_hash_dataclass(None) == "na"
        assert _content_hash_dataclass(42) == "na"
        assert _content_hash_dataclass("not a dataclass") == "na"


class TestContentHashCache:
    """S5: cache 해시 None 필드 내성."""

    @staticmethod
    def _make_cache(*, regime_code_1d: np.ndarray | None = None) -> L2SimulationCache:
        T, N, S = 3, 2, 2
        return L2SimulationCache(
            vol_matrix_2d=np.ones((T, N), dtype=np.float64),
            tradeable_mask_2d=np.ones((T, N), dtype=bool),
            hurdle_2d=np.ones((T, N), dtype=np.float64),
            funding_2d=np.zeros((T, N), dtype=np.float64),
            beta_1d=np.zeros(N, dtype=np.float64),
            expected_gross_bps_2d=np.ones((T, S), dtype=np.float64),
            expected_net_bps_2d=np.zeros((T, S), dtype=np.float64),
            holding_bars_2d=np.ones((T, S), dtype=np.float64),
            side_2d=np.ones((T, S), dtype=np.float64),
            quality_weight_2d=np.ones((T, S), dtype=np.float64),
            signal_mask_2d=np.ones((T, S), dtype=bool),
            sleeve_to_sym=np.array([0, 1], dtype=np.int64),
            sleeve_ids=(("BTC", "trend:fast"), ("ETH", "trend:fast")),
            sleeve_to_tf=("4h", "4h"),
            regime_code_1d=regime_code_1d,
        )

    def test_content_hash_cache_handles_none_fields(self) -> None:
        cache = self._make_cache(regime_code_1d=None)
        h = _content_hash_cache(cache)
        assert isinstance(h, str)
        assert len(h) == 12

    def test_content_hash_cache_deterministic(self) -> None:
        cache = self._make_cache(
            regime_code_1d=np.array([0, 1, 2], dtype=np.int8),
        )
        h1 = _content_hash_cache(cache)
        h2 = _content_hash_cache(cache)
        assert h1 == h2

    def test_content_hash_cache_non_dataclass_returns_na(self) -> None:
        assert _content_hash_cache(None) == "na"
        assert _content_hash_cache("string") == "na"


class TestFingerprint2Integration:
    """S6: [AWF-SIM-FP2] 로그가 DEBUG 레벨에서만 출력."""

    def test_fingerprint2_logged_when_debug_on(self, caplog: pytest.LogCaptureFixture) -> None:
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

        caplog.set_level(logging.DEBUG)
        n_bars = 20
        datetimes = np.array(
            [np.datetime64("2024-01-01", "ns") + np.timedelta64(i * 4, "h") for i in range(n_bars)],
            dtype="datetime64[ns]",
        )
        close = np.ones((n_bars, 2), dtype=np.float64) * 100.0
        aligned = MagicMock()
        aligned.symbols = ("BTC", "ETH")
        aligned.close_2d = close
        aligned.datetimes = datetimes
        aligned.funding_2d = np.zeros((n_bars, 2), dtype=np.float64)
        aligned.active_mask = np.ones((n_bars, 2), dtype=bool)
        aligned.warm_mask = np.ones((n_bars, 2), dtype=bool)
        aligned.entry_block_mask = np.zeros((n_bars, 2), dtype=bool)
        aligned.kill_mask = np.zeros((n_bars, 2), dtype=bool)
        aligned.execution_cost_bps_2d = np.full((n_bars, 2), 4.0, dtype=np.float64)
        aligned.beta_vs_market_1d = np.zeros(2, dtype=np.float64)

        signal_batch = ValidatedSignalBatch(
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
        awf_folds = (WFFold(fit_start=0, fit_end=1, cal_start=1, cal_end=1, oos_start=1, oos_end=4),)
        config = Layer2AllocationConfig(k_rank=2, rebalance_bars=1, no_trade_band=0.0)
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

        fp2_records = [r for r in caplog.records if "[AWF-SIM-FP2]" in str(r.message)]
        assert len(fp2_records) >= 1
        msg = str(fp2_records[0].message)
        assert "origin=unknown" in msg
        assert "cache_ch=" in msg
        assert "cfg_ch=" in msg
        assert "caps_ch=" in msg
        assert "per_fold_fp=" in msg

    def test_fingerprint2_skipped_when_debug_off(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO)
        from unittest.mock import MagicMock

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

        caplog.set_level(logging.INFO)
        n_bars = 20
        datetimes = np.array(
            [np.datetime64("2024-01-01", "ns") + np.timedelta64(i * 4, "h") for i in range(n_bars)],
            dtype="datetime64[ns]",
        )
        close = np.ones((n_bars, 2), dtype=np.float64) * 100.0
        aligned = MagicMock()
        aligned.symbols = ("BTC", "ETH")
        aligned.close_2d = close
        aligned.datetimes = datetimes
        aligned.funding_2d = np.zeros((n_bars, 2), dtype=np.float64)
        aligned.active_mask = np.ones((n_bars, 2), dtype=bool)
        aligned.warm_mask = np.ones((n_bars, 2), dtype=bool)
        aligned.entry_block_mask = np.zeros((n_bars, 2), dtype=bool)
        aligned.kill_mask = np.zeros((n_bars, 2), dtype=bool)
        aligned.execution_cost_bps_2d = np.full((n_bars, 2), 4.0, dtype=np.float64)
        aligned.beta_vs_market_1d = np.zeros(2, dtype=np.float64)

        signal_batch = ValidatedSignalBatch(
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
        awf_folds = (WFFold(fit_start=0, fit_end=1, cal_start=1, cal_end=1, oos_start=1, oos_end=4),)
        config = Layer2AllocationConfig(k_rank=2, rebalance_bars=1, no_trade_band=0.0)
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

        fp2_records = [r for r in caplog.records if "[AWF-SIM-FP2]" in str(r.message)]
        assert len(fp2_records) == 0
