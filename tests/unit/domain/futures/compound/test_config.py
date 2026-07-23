from __future__ import annotations

import pytest

from src.domain.futures.compound.config import (
    AllocatorConfig,
    CompoundEngineConfig,
    DataPlaneConfig,
    FactorRiskConfig,
    L1EstimatorConfig,
    L1Config,
    L3ValidationConfig,
    RiskOverlayConfig,
)


class TestDataPlaneConfig:
    def test_defaults_are_valid(self) -> None:
        cfg = DataPlaneConfig()
        assert cfg.max_symbols == 120
        assert cfg.alpha_block_bars == 2048

    def test_zero_max_symbols_raises(self) -> None:
        with pytest.raises(AssertionError):
            DataPlaneConfig(max_symbols=0)

    def test_invalid_core_coverage_raises(self) -> None:
        with pytest.raises(AssertionError):
            DataPlaneConfig(min_core_coverage=1.5)


class TestL1EstimatorConfig:
    def test_retire_n_gt_active_n(self) -> None:
        with pytest.raises(AssertionError):
            L1EstimatorConfig(retire_effective_n=10, active_effective_n=20)

    def test_valid(self) -> None:
        cfg = L1EstimatorConfig()
        assert cfg.retire_effective_n > cfg.active_effective_n


class TestL1Config:
    def test_defaults_satisfy_edge_gate_bounds(self) -> None:
        cfg = L1Config()

        assert cfg.max_residual_correlation == pytest.approx(0.60)
        assert cfg.total_outer_folds >= cfg.min_positive_folds
        assert cfg.cost_stress_multiplier >= 1

    def test_invalid_edge_gate_bounds_raise(self) -> None:
        with pytest.raises(AssertionError):
            L1Config(max_residual_correlation=0.0)
        with pytest.raises(AssertionError):
            L1Config(total_outer_folds=3, min_positive_folds=4)
        with pytest.raises(AssertionError):
            L1Config(cost_stress_multiplier=0.5)


class TestAllocatorConfig:
    def test_net_cap_leq_gross_cap(self) -> None:
        with pytest.raises(AssertionError):
            AllocatorConfig(net_cap=0.5, gross_cap=0.3)

    def test_max_iterations_unsigned(self) -> None:
        with pytest.raises(AssertionError):
            AllocatorConfig(max_iterations=100)


class TestFactorRiskConfig:
    def test_variance_floor_positive(self) -> None:
        with pytest.raises(AssertionError):
            FactorRiskConfig(variance_floor=-1e-8)


class TestRiskOverlayConfig:
    def test_knot_order(self) -> None:
        with pytest.raises(AssertionError):
            RiskOverlayConfig(soft_drawdown_start=0.15, drawdown_second_knot=0.08)


class TestL3ValidationConfig:
    def test_probability_order(self) -> None:
        with pytest.raises(AssertionError):
            L3ValidationConfig(promote_probability=0.3, reject_probability=0.5)

    def test_min_holdout_leq_holdout(self) -> None:
        with pytest.raises(AssertionError):
            L3ValidationConfig(holdout_days=20, min_holdout_days=30)


class TestCompoundEngineConfig:
    def test_default_constructs(self) -> None:
        cfg = CompoundEngineConfig()
        assert isinstance(cfg.data, DataPlaneConfig)
        assert isinstance(cfg.l1, L1EstimatorConfig)

    def test_invalid_subconfig_raises(self) -> None:
        with pytest.raises(AssertionError):
            CompoundEngineConfig(
                data=DataPlaneConfig(max_symbols=0)
            )
