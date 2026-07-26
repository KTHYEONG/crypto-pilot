from __future__ import annotations

import pytest

from src.domain.futures.compound.config import (
    AdmissionConfig,
    AllocatorConfig,
    BaselineAllocConfig,
    CalibrationConfig,
    CompoundEngineConfig,
    DataPlaneConfig,
    DenseSimConfig,
    DynamicCompoundingConfig,
    FactorRiskConfig,
    L1EstimatorConfig,
    L1Config,
    L3ValidationConfig,
    LadderConfig,
    RiskModelConfig,
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


class TestCalibrationConfig:
    def test_defaults_valid(self) -> None:
        cfg = CalibrationConfig()
        assert cfg.n_folds >= 3

    def test_invalid_folds_raises(self) -> None:
        with pytest.raises(AssertionError):
            CalibrationConfig(n_folds=2)

    def test_invalid_shrink_raises(self) -> None:
        with pytest.raises(AssertionError):
            CalibrationConfig(family_shrink=-0.1)
        with pytest.raises(AssertionError):
            CalibrationConfig(family_shrink=1.5)


class TestAdmissionConfig:
    def test_defaults_valid(self) -> None:
        cfg = AdmissionConfig()
        assert cfg.n_bootstrap == 500

    def test_zero_bootstrap_raises(self) -> None:
        with pytest.raises(AssertionError):
            AdmissionConfig(n_bootstrap=0)

    def test_invalid_fdr_raises(self) -> None:
        with pytest.raises(AssertionError):
            AdmissionConfig(fdr_q_threshold=0.0)
        with pytest.raises(AssertionError):
            AdmissionConfig(fdr_q_threshold=1.5)

    def test_invalid_sign_consistency_raises(self) -> None:
        with pytest.raises(AssertionError):
            AdmissionConfig(sign_consistency_min=0.0)


class TestRiskModelConfig:
    def test_defaults_valid(self) -> None:
        cfg = RiskModelConfig()
        assert cfg.ewm_half_life_bars == 60
        assert cfg.shrink_delta == 0.3

    def test_zero_half_life_raises(self) -> None:
        with pytest.raises(AssertionError):
            RiskModelConfig(ewm_half_life_bars=0)

    def test_invalid_delta_raises(self) -> None:
        with pytest.raises(AssertionError):
            RiskModelConfig(shrink_delta=-0.1)
        with pytest.raises(AssertionError):
            RiskModelConfig(shrink_delta=1.5)


class TestBaselineAllocConfig:
    def test_defaults_valid(self) -> None:
        cfg = BaselineAllocConfig()
        assert cfg.target_ann_vol == 0.20

    def test_zero_target_vol_raises(self) -> None:
        with pytest.raises(AssertionError):
            BaselineAllocConfig(target_ann_vol=0.0)

    def test_invalid_cap_raises(self) -> None:
        with pytest.raises(AssertionError):
            BaselineAllocConfig(per_symbol_cap=0.0)
        with pytest.raises(AssertionError):
            BaselineAllocConfig(gross_cap=-1.0)


class TestDenseSimConfig:
    def test_defaults_valid(self) -> None:
        cfg = DenseSimConfig()
        assert cfg.bars_per_year == 2190.0

    def test_zero_bars_raises(self) -> None:
        with pytest.raises(AssertionError):
            DenseSimConfig(bars_per_year=0.0)


class TestLadderConfig:
    def test_defaults_valid(self) -> None:
        cfg = LadderConfig()
        assert cfg.cost_bps == 8.0

    def test_negative_cost_raises(self) -> None:
        with pytest.raises(AssertionError):
            LadderConfig(cost_bps=-1.0)

    def test_zero_bootstrap_raises(self) -> None:
        with pytest.raises(AssertionError):
            LadderConfig(n_bootstrap=0)


class TestCompoundEngineConfig:
    def test_default_constructs(self) -> None:
        cfg = CompoundEngineConfig()
        assert isinstance(cfg.data, DataPlaneConfig)
        assert isinstance(cfg.l1, L1EstimatorConfig)
        assert isinstance(cfg.risk_model, RiskModelConfig)
        assert isinstance(cfg.baseline_alloc, BaselineAllocConfig)
        assert isinstance(cfg.dense_sim, DenseSimConfig)
        assert isinstance(cfg.ladder, LadderConfig)

    def test_invalid_subconfig_raises(self) -> None:
        with pytest.raises(AssertionError):
            CompoundEngineConfig(
                data=DataPlaneConfig(max_symbols=0)
            )


class TestDynamicCompoundingConfig:
    def test_dynamic_compounding_config_default_band_and_smoothing(self) -> None:
        cfg = DynamicCompoundingConfig()
        assert cfg.band_frac == 0.60
        assert cfg.alpha_smooth == 0.08
        assert cfg.target_ann_vol == 0.15
