from __future__ import annotations

import numpy as np

from src.domain.futures.compound.contracts import (
    AlphaDefinition,
    AlphaLifecycle,
    CombinedForecast,
    DeploymentVerdict,
    ExecutionLedger,
    MarketFeatureCube,
)


class TestAlphaLifecycle:
    def test_enum_values(self) -> None:
        assert AlphaLifecycle.SHADOW.value == "shadow"
        assert AlphaLifecycle.ACTIVE.value == "active"
        assert AlphaLifecycle.RETIRED.value == "retired"


class TestDeploymentVerdict:
    def test_enum_values(self) -> None:
        assert DeploymentVerdict.PROMOTE.value == "promote"
        assert DeploymentVerdict.SHADOW.value == "shadow"
        assert DeploymentVerdict.REJECT.value == "reject"


class TestAlphaDefinition:
    def test_default_causal_lag(self) -> None:
        ad = AlphaDefinition(
            recipe_id="test", family="trend", horizon_bars=4, lookback_bars=24,
            required_fields=("close",), data_tier="core",
        )
        assert ad.causal_lag_bars == 1


class TestMarketFeatureCube:
    def test_shape(self) -> None:
        n_bars, n_syms = 64, 2
        cube = MarketFeatureCube(
            timestamps_ns=np.arange(n_bars, dtype=np.int64),
            symbols=("A", "B"),
            fields_2d={"close": np.ones((n_bars, n_syms), dtype=np.float64)},
            available_2d={"core": np.ones((n_bars, n_syms), dtype=np.bool_)},
            eligible_2d=np.ones((n_bars, n_syms), dtype=np.bool_),
            entry_block_2d=np.zeros((n_bars, n_syms), dtype=np.bool_),
            exit_required_2d=np.zeros((n_bars, n_syms), dtype=np.bool_),
            capacity_usdt_2d=np.full((n_bars, n_syms), 1e6, dtype=np.float64),
            execution_cost_bps_2d=np.full((n_bars, n_syms), 12.0, dtype=np.float32),
            data_manifest_hash="hash1",
        )
        assert cube.timestamps_ns.shape == (64,)
        assert len(cube.symbols) == 2


class TestCombinedForecast:
    def test_fields(self) -> None:
        cf = CombinedForecast(
            mu_robust_1d=np.array([0.001, 0.002], dtype=np.float64),
            variance_1d=np.array([1e-8, 2e-8], dtype=np.float64),
            support_1d=np.array([True, True], dtype=np.bool_),
        )
        assert cf.mu_robust_1d[0] == 0.001


class TestExecutionLedger:
    def test_minimal(self) -> None:
        ledger = ExecutionLedger(
            timestamps_ns=np.array([0], dtype=np.int64),
            net_returns_1d=np.array([0.0], dtype=np.float64),
            equity_1d=np.array([1.0], dtype=np.float64),
            target_weights_2d=np.zeros((1, 2), dtype=np.float32),
            fee_returns_1d=np.zeros(1, dtype=np.float64),
            slippage_returns_1d=np.zeros(1, dtype=np.float64),
            impact_returns_1d=np.zeros(1, dtype=np.float64),
            funding_returns_1d=np.zeros(1, dtype=np.float64),
            integrity_ok=True,
            integrity_reasons=(),
        )
        assert ledger.integrity_ok


class TestSignalDescriptor:
    def test_default_target_horizon(self) -> None:
        from src.domain.futures.compound.contracts import SignalDescriptor
        sd = SignalDescriptor("test", "test_fam", "fast", 24, "4h")
        assert sd.target_horizon_hours == 4

    def test_custom_target_horizon(self) -> None:
        from src.domain.futures.compound.contracts import SignalDescriptor
        sd = SignalDescriptor("test", "test_fam", "slow", 216, "4h", target_horizon_hours=216)
        assert sd.target_horizon_hours == 216

    def test_signal_descriptor_rejects_non_multiple_of_4_target_horizon(self) -> None:
        from src.domain.futures.compound.contracts import SignalDescriptor
        import pytest
        with pytest.raises(ValueError, match="multiple of 4"):
            SignalDescriptor("test", "test_fam", "fast", 24, "4h", target_horizon_hours=5)
