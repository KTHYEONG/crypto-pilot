from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from src.core.settings import MAKER_FEE_BPS, SLIPPAGE_BPS, TAKER_FEE_BPS


@dataclass(slots=True, frozen=True)
class MomentumConfig:
    """XS momentum sleeve parameters."""

    lookback_bars: int = 6
    top_ratio: float = 0.30
    bottom_ratio: float = 0.30
    min_symbols_for_xs: int = 5
    edge_scale_per_bar: float = 1e-3

    def __post_init__(self) -> None:
        """Validate momentum parameter bounds."""
        if self.lookback_bars < 1:
            raise ValueError("lookback_bars must be >= 1")
        if not (0.0 < self.top_ratio <= 0.5):
            raise ValueError("top_ratio must satisfy 0 < top_ratio <= 0.5")
        if not (0.0 < self.bottom_ratio <= 0.5):
            raise ValueError("bottom_ratio must satisfy 0 < bottom_ratio <= 0.5")


@dataclass(slots=True, frozen=True)
class SleeveConfig:
    """Enhanced strategy multi-sleeve switch and parameter settings."""

    # TS momentum disabled: negative IC at all tested horizons (4h t=-6.8, 1d t=-2.3)
    ts_momentum_enabled: bool = False
    ts_momentum_lookback: int = 36
    ts_momentum_skip: int = 1
    reversal_enabled: bool = True
    reversal_lookback: int = 6
    carry_enabled: bool = True
    carry_smooth: int = 6

    def __post_init__(self) -> None:
        """Validate sleeve parameters."""
        if self.ts_momentum_lookback < 1:
            raise ValueError("ts_momentum_lookback must be >= 1")
        if self.ts_momentum_skip < 0:
            raise ValueError("ts_momentum_skip must be >= 0")
        if self.reversal_lookback < 1:
            raise ValueError("reversal_lookback must be >= 1")
        if self.carry_smooth < 1:
            raise ValueError("carry_smooth must be >= 1")


@dataclass(slots=True, frozen=True)
class BlendConfig:
    """Enhanced strategy blending parameters."""

    clip_z: float = 3.0
    min_symbols: int = 5
    ic_window_bars: int = 180
    ic_shrinkage: float = 0.5
    min_mean_ic: float = 0.02
    min_t_stat: float = 2.0
    min_hit_ratio: float = 0.45
    sigma_lookback: int = 30

    def __post_init__(self) -> None:
        """Validate blending parameters."""
        if self.clip_z <= 0.0:
            raise ValueError("clip_z must be positive")
        if self.min_symbols < 1:
            raise ValueError("min_symbols must be >= 1")
        if self.ic_window_bars < self.sigma_lookback:
            raise ValueError("ic_window_bars must satisfy >= sigma_lookback")
        if not (0.0 < self.ic_shrinkage <= 1.0):
            raise ValueError("ic_shrinkage must satisfy 0 < ic_shrinkage <= 1")
        if not (0.0 <= self.min_hit_ratio <= 1.0):
            raise ValueError("min_hit_ratio must satisfy 0 <= min_hit_ratio <= 1")
        if self.sigma_lookback < 1:
            raise ValueError("sigma_lookback must be >= 1")


@dataclass(slots=True, frozen=True)
class RegimeConfig:
    """Rule-based 5-state soft posterior regime settings."""

    enabled: bool = False  # regime provider module removed; keep False until re-implemented
    vol_window: int = 30
    vol_crisis_pct: float = 0.95
    vol_high_pct: float = 0.70
    trend_ma_fast: int = 12
    trend_ma_slow: int = 48
    trend_thr: float = 0.0
    dd_crisis_thr: float = -0.20
    corr_crisis_thr: float = 0.80
    smooth_ewma_bars: int = 6
    gross_floor: float = 0.15

    def __post_init__(self) -> None:
        """Validate regime parameters."""
        if self.vol_window < 1:
            raise ValueError("vol_window must be >= 1")
        if not (0.0 < self.vol_high_pct < self.vol_crisis_pct < 1.0):
            raise ValueError(
                "volatility percentiles must satisfy "
                "0 < vol_high_pct < vol_crisis_pct < 1.0"
            )
        if self.trend_ma_fast >= self.trend_ma_slow:
            raise ValueError("trend_ma_fast must be less than trend_ma_slow")
        if self.trend_ma_fast < 1:
            raise ValueError("trend_ma_fast must be >= 1")
        if self.dd_crisis_thr >= 0.0:
            raise ValueError("dd_crisis_thr must be negative")
        if not (0.0 <= self.corr_crisis_thr <= 1.0):
            raise ValueError("corr_crisis_thr must satisfy 0 <= corr_crisis_thr <= 1.0")
        if self.smooth_ewma_bars < 1:
            raise ValueError("smooth_ewma_bars must be >= 1")
        if not (0.0 <= self.gross_floor <= 1.0):
            raise ValueError("gross_floor must satisfy 0 <= gross_floor <= 1.0")


@dataclass(slots=True, frozen=True)
class StrategyConfig:
    """Top-level strategy switch."""

    name: str = "momentum"
    momentum: MomentumConfig = field(default_factory=MomentumConfig)
    sleeves: SleeveConfig = field(default_factory=SleeveConfig)
    blend: BlendConfig = field(default_factory=BlendConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    ml: StrategyMLConfig = field(default_factory=lambda: StrategyMLConfig())

    def __post_init__(self) -> None:
        """Validate top-level strategy name."""
        if self.name not in {"momentum", "eh_st", "lambdamart", "xs_reversal"}:
            raise ValueError(f"unsupported strategy name: {self.name}")


@dataclass(slots=True, frozen=True)
class FeatureIntegrityConfig:
    """Feature integrity selection thresholds."""

    tau_nan: float = 0.50
    epsilon: float = 1e-9
    tau_psi: float = 0.25
    ic_floor: float | None = None
    tau_corr: float = 0.95
    min_keep: int = 8

    def __post_init__(self) -> None:
        if not (0.0 <= self.tau_nan <= 1.0):
            raise ValueError("tau_nan must be in [0, 1]")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be > 0")
        if self.tau_psi < 0.0:
            raise ValueError("tau_psi must be >= 0")
        if self.ic_floor is not None and self.ic_floor < 0.0:
            raise ValueError("ic_floor must be >= 0 when set")
        if not (0.0 <= self.tau_corr <= 1.0):
            raise ValueError("tau_corr must be in [0, 1]")
        if self.min_keep < 1:
            raise ValueError("min_keep must be >= 1")


@dataclass(slots=True, frozen=True)
class StrategyMLConfig:
    """ML strategy configuration for simple rank-native architecture."""

    name: Literal["lambdamart"] = "lambdamart"
    timeframe: str = "4h"
    seed: int = 42
    n_jobs: int = -1  # -1 resolves dynamically to optimal CPU count
    parallel_folds: bool = True  # Enable fold-level joblib parallelization
    parallel_fold_workers: int = -1  # -1 uses max(1, os.cpu_count() - 2)
    min_group_size: int = 8
    label_horizon_bars: int = 12
    train_months: int = 24
    valid_months: int = 3
    test_months: int = 3
    purge_bars: int = 12
    embargo_bars: int = 12
    max_features: int = 32
    alpha_clip_bps: float = 75.0
    lambda_tail: float = 0.10
    ranker_n_estimators: int = 300
    calibrator_n_estimators: int = 200
    learning_rate: float = 0.03
    ranker_learning_rate: float = 0.03
    calibrator_learning_rate: float = 0.02
    num_leaves: int = 7
    max_depth: int = 3
    min_data_in_leaf: int = 100
    feature_fraction: float = 0.80
    bagging_fraction: float = 0.80
    ranker_feature_fraction: float = 0.80
    calibrator_feature_fraction: float = 0.70
    ranker_bagging_fraction: float = 0.80
    calibrator_bagging_fraction: float = 0.75
    ranker_bagging_freq: int = 1
    calibrator_bagging_freq: int = 1
    lambda_l2: float = 5.0
    ranker_lambda_l2: float = 30.0
    calibrator_lambda_l2: float = 1.0
    ranker_reg_alpha: float = 5.0
    calibrator_reg_alpha: float = 1.5
    calibrator_max_depth_cap: int = 5
    early_stopping_rounds: int = 75
    # per-side 비용: 레이블 생성 시 round-trip(x2)으로 환산됨 (labels.py 참조)
    fee_bps: float = TAKER_FEE_BPS       # Taker 수수료 per side (canonical: core/settings.py)
    slippage_bps: float = SLIPPAGE_BPS   # 슬리피지 per side (canonical: core/settings.py)

    # 앙상블 및 실질 메이커 비용 반영 설정
    ensemble_seeds: list[int] = field(default_factory=lambda: [42, 1004, 2026])
    maker_ratio: float = 0.20            # 보수적 설정을 위해 Taker 비중을 80%로 높게 설정
    maker_fee_bps: float = MAKER_FEE_BPS
    taker_fee_bps: float = TAKER_FEE_BPS
    # IC gate 파라미터: 완화된 초기 임계값 — B2 uplift 확인 후 강화 예정
    ic_gate_min_mean_ic: float = 0.01
    ic_gate_min_t_stat: float = 1.5
    ic_gate_min_hit_ratio: float = 0.45
    ic_gate_warn_only: bool = True  # True=경고만, False=RuntimeError
    # Calibrator target: which return series to use as y_ev for magnitude learning.
    # "beta_residualized" = current default: exec_net_ret after beta-resid, pre-CS-demean
    # "gross"             = raw log return minus funding only (no beta removal, no fee)
    calibrator_target: Literal["beta_residualized", "gross"] = "beta_residualized"
    model_family: Literal["lgbm_regression", "lgbm_huber", "lgbm_lambdarank"] = "lgbm_lambdarank"
    ranking_mode: Literal["pointwise", "group_ndcg"] = "group_ndcg"
    # False: skip ranker stage; calibrator uses zero rank_score
    # (C3 ablation A/B — empirically better OOS IC)
    ranker_enabled: bool = True
    rank_target_mode: Literal["cs_residual", "forward_gross_rank"] = "cs_residual"
    calibrator_target_mode: Literal["signed_ev", "rank_confidence"] = "signed_ev"
    post_cost_admission_mode: Literal[
        "ev_gate", "rank_then_ev_gate", "rank_cs_neutral"
    ] = "rank_cs_neutral"
    rank_portfolio_top_k: int = 4
    rank_portfolio_min_score_spread_bps: float = 0.0
    oos_ic_target_source: Literal["signed_net_ret", "forward_gross_ret"] = "forward_gross_ret"
    regime_gate_enabled: bool = True      # trailing BTC regime → exposure scalar (no look-ahead)
    regime_exposure_bull: float = 1.0      # full deployment in bull (IC > breakeven)
    regime_exposure_bear: float = 0.5      # partial exposure in bear — L/S is already market-hedged
    regime_exposure_chop: float = 1.0      # suppressed in chop (IC ≈ 0)
    ev_mode: Literal["quantile", "prob_x_magnitude"] = "prob_x_magnitude"
    alpha_gate_min_long_nz: float = 0.0
    alpha_gate_min_short_nz: float = 0.0
    alpha_gate_min_xs_preservation: float = 0.0
    alpha_gate_min_tradable_long_nz: float = 0.003
    alpha_gate_min_tradable_short_nz: float = 0.002
    # Numerical tolerance around cost wall comparison to avoid failing on tiny rounding noise.
    alpha_gate_cost_wall_tolerance_bps: float = 0.0
    ev_tail_blend_weight: float = 0.0
    alpha_emit_mode: str = "rank_sized"   # keep for compatibility
    alpha_emit_select_q: float = 0.35
    alpha_emit_weight_k: float = 3.0      # tanh rank-weight steepness
    feature_groups_enabled: tuple[
        Literal[
            "trend",
            "reversal",
            "volatility",
            "carry",
            "liquidity",
            "market_context",
            "microstructure",
            "missingness",
        ],
        ...,
    ] = (
        "trend",
        "reversal",
        "volatility",
        "carry",
        "liquidity",
        "market_context",
        "microstructure",
    )
    add_missingness_indicators: bool = False
    horizon_experiment_enabled: bool = False
    horizon_candidates: tuple[int, ...] = ()
    training_universe_scope: Literal[
        "stage5_passed",
        "stage6_selected",
        "historical_stage6",          # universe-fix.md 호환 (deprecated 예정)
        "historical_stage5_union",    # 신규 기본값 (C1)
    ] = "historical_stage5_union"
    trading_symbols: tuple[str, ...] = ()  # Stage6 selected 심볼 — 거래 마스킹용
    # C1 active_mask 사용 여부 — True면 inference_active_mask, False면 universe_active_mask
    use_inference_active_mask: bool = True
    # Sample weighting 보정 설정
    sample_weight_quality_clip_min: float = 0.50
    sample_weight_cluster_balance_enabled: bool = True
    # Phase 1C: 6mo halflife = 180d x 6bar/day = 1080 bars. None=disabled.
    sample_weight_time_decay_halflife_bars: int | None = 1080

    # Phase 1: rank-native composition
    rank_select_quantile: float = 0.35
    rank_select_quantiles: tuple[float, ...] = (0.25, 0.35, 0.45)
    target_breadth: int = 8                      # minimum effective breadth target
    ic_lcb_z: float = 1.0
    ic_prior_for_gate: float = 0.03             # leak-free IC prior for portfolio net-edge gate
    ev_secondary_tilt_weight: float = 0.0       # blend weight for EV rank tilt (0=rank-only)
    integrity_gate_enabled: bool = True
    feature_selection_enabled: bool = True
    feature_integrity: FeatureIntegrityConfig = field(default_factory=FeatureIntegrityConfig)

    def __post_init__(self) -> None:
        """Validate ML strategy parameters."""
        if not self.ensemble_seeds:
            raise ValueError("ensemble_seeds list must contain at least one integer seed.")
        if not (0.0 <= self.maker_ratio <= 1.0):
            raise ValueError("maker_ratio must be between 0.0 and 1.0.")
        if self.purge_bars < self.label_horizon_bars:
            raise ValueError("purge_bars must be >= label_horizon_bars")
        if self.embargo_bars < self.label_horizon_bars:
            raise ValueError(
                f"embargo_bars ({self.embargo_bars}) must be >= "
                f"label_horizon_bars ({self.label_horizon_bars}) "
                "to prevent valid->test label horizon leakage"
            )
        if self.embargo_bars < 1:
            raise ValueError("embargo_bars must be >= 1")
        if self.max_features > 32:
            raise ValueError("max_features must be <= 32")
        if self.alpha_clip_bps <= 0.0:
            raise ValueError("alpha_clip_bps must be > 0")
        if self.min_group_size < 2:
            raise ValueError("min_group_size must be >= 2")
        if not (0.0 < self.learning_rate <= 0.2):
            raise ValueError("learning_rate must satisfy 0 < lr <= 0.2")
        if not (0.0 < self.ranker_learning_rate <= 0.2):
            raise ValueError("ranker_learning_rate must satisfy 0 < lr <= 0.2")
        if not (0.0 < self.calibrator_learning_rate <= 0.2):
            raise ValueError("calibrator_learning_rate must satisfy 0 < lr <= 0.2")
        if self.num_leaves > 31:
            raise ValueError("num_leaves must be <= 31")
        if self.max_depth > 6:
            raise ValueError("max_depth must be <= 6")
        if self.min_data_in_leaf < 10:
            raise ValueError("min_data_in_leaf must be >= 10")
        if not (0.0 < self.feature_fraction <= 1.0):
            raise ValueError("feature_fraction must satisfy 0 < feature_fraction <= 1")
        if not (0.0 < self.bagging_fraction <= 1.0):
            raise ValueError("bagging_fraction must satisfy 0 < bagging_fraction <= 1")
        if not (0.0 < self.ranker_feature_fraction <= 1.0):
            raise ValueError("ranker_feature_fraction must satisfy 0 < value <= 1")
        if not (0.0 < self.calibrator_feature_fraction <= 1.0):
            raise ValueError("calibrator_feature_fraction must satisfy 0 < value <= 1")
        if not (0.0 < self.ranker_bagging_fraction <= 1.0):
            raise ValueError("ranker_bagging_fraction must satisfy 0 < value <= 1")
        if not (0.0 < self.calibrator_bagging_fraction <= 1.0):
            raise ValueError("calibrator_bagging_fraction must satisfy 0 < value <= 1")
        if self.ranker_bagging_freq < 0:
            raise ValueError("ranker_bagging_freq must be >= 0")
        if self.calibrator_bagging_freq < 0:
            raise ValueError("calibrator_bagging_freq must be >= 0")
        if self.ranker_lambda_l2 < 0.0:
            raise ValueError("ranker_lambda_l2 must be >= 0")
        if self.calibrator_lambda_l2 < 0.0:
            raise ValueError("calibrator_lambda_l2 must be >= 0")
        if self.ranker_reg_alpha < 0.0:
            raise ValueError("ranker_reg_alpha must be >= 0")
        if self.calibrator_reg_alpha < 0.0:
            raise ValueError("calibrator_reg_alpha must be >= 0")
        if self.calibrator_max_depth_cap < 1:
            raise ValueError("calibrator_max_depth_cap must be >= 1")
        if self.calibrator_target not in {"beta_residualized", "gross"}:
            raise ValueError(
                f"calibrator_target must be 'beta_residualized' or 'gross', "
                f"got '{self.calibrator_target}'"
            )
        if self.model_family not in {"lgbm_regression", "lgbm_huber", "lgbm_lambdarank"}:
            raise ValueError(
                "model_family must be 'lgbm_regression', 'lgbm_huber', or 'lgbm_lambdarank'"
            )
        if self.ev_mode not in {"quantile", "prob_x_magnitude"}:
            raise ValueError("ev_mode must be 'quantile' or 'prob_x_magnitude'")
        if self.ranking_mode not in {"pointwise", "group_ndcg"}:
            raise ValueError("ranking_mode must be 'pointwise' or 'group_ndcg'")
        if self.rank_target_mode not in {"cs_residual", "forward_gross_rank"}:
            raise ValueError("rank_target_mode must be 'cs_residual' or 'forward_gross_rank'")
        if self.calibrator_target_mode not in {"signed_ev", "rank_confidence"}:
            raise ValueError(
                "calibrator_target_mode must be 'signed_ev' or 'rank_confidence'"
            )
        if self.post_cost_admission_mode not in {
            "ev_gate", "rank_then_ev_gate", "rank_cs_neutral"
        }:
            raise ValueError(
                "post_cost_admission_mode must be "
                "'ev_gate', 'rank_then_ev_gate', or 'rank_cs_neutral'"
            )
        if not (0.0 < self.rank_select_quantile < 0.5):
            raise ValueError("rank_select_quantile must satisfy 0 < value < 0.5")
        if len(self.rank_select_quantiles) == 0:
            raise ValueError("rank_select_quantiles must be non-empty")
        if any((q <= 0.0 or q >= 0.5) for q in self.rank_select_quantiles):
            raise ValueError("rank_select_quantiles must satisfy 0 < q < 0.5")
        if self.target_breadth < 2:
            raise ValueError("target_breadth must be >= 2")
        if self.ic_lcb_z < 0.0:
            raise ValueError("ic_lcb_z must be >= 0")
        if not (0.0 <= self.ic_prior_for_gate <= 0.2):
            raise ValueError("ic_prior_for_gate must satisfy 0 <= value <= 0.2")
        if not (0.0 <= self.ev_secondary_tilt_weight <= 1.0):
            raise ValueError("ev_secondary_tilt_weight must satisfy 0 <= value <= 1")
        if self.rank_portfolio_top_k < 1:
            raise ValueError("rank_portfolio_top_k must be >= 1")
        if self.rank_portfolio_min_score_spread_bps < 0.0:
            raise ValueError("rank_portfolio_min_score_spread_bps must be >= 0")
        if self.oos_ic_target_source not in {"signed_net_ret", "forward_gross_ret"}:
            raise ValueError(
                "oos_ic_target_source must be 'signed_net_ret' or 'forward_gross_ret'"
            )
        if not (0.0 <= self.alpha_gate_min_long_nz <= 1.0):
            raise ValueError("alpha_gate_min_long_nz must satisfy 0 <= value <= 1")
        if not (0.0 <= self.alpha_gate_min_short_nz <= 1.0):
            raise ValueError("alpha_gate_min_short_nz must satisfy 0 <= value <= 1")
        if not (0.0 <= self.alpha_gate_min_xs_preservation <= 1.0):
            raise ValueError("alpha_gate_min_xs_preservation must satisfy 0 <= value <= 1")
        if not (0.0 <= self.alpha_gate_min_tradable_long_nz <= 1.0):
            raise ValueError("alpha_gate_min_tradable_long_nz must satisfy 0 <= value <= 1")
        if not (0.0 <= self.alpha_gate_min_tradable_short_nz <= 1.0):
            raise ValueError("alpha_gate_min_tradable_short_nz must satisfy 0 <= value <= 1")
        if self.alpha_gate_cost_wall_tolerance_bps < 0.0:
            raise ValueError("alpha_gate_cost_wall_tolerance_bps must be >= 0")
        if not (0.0 <= self.ev_tail_blend_weight <= 1.0):
            raise ValueError("ev_tail_blend_weight must satisfy 0 <= value <= 1")
        if not (0.0 <= self.regime_exposure_bull <= 1.0):
            raise ValueError("regime_exposure_bull must be in [0.0, 1.0]")
        if not (0.0 <= self.regime_exposure_bear <= 1.0):
            raise ValueError("regime_exposure_bear must be in [0.0, 1.0]")
        if not (0.0 <= self.regime_exposure_chop <= 1.0):
            raise ValueError("regime_exposure_chop must be in [0.0, 1.0]")
        if self.training_universe_scope not in {
            "stage5_passed",
            "stage6_selected",
            "historical_stage6",
            "historical_stage5_union",
        }:
            raise ValueError(
                "training_universe_scope must be one of: "
                "'stage5_passed', 'stage6_selected', 'historical_stage6', 'historical_stage5_union'"
            )
        if (
            self.sample_weight_time_decay_halflife_bars is not None
            and self.sample_weight_time_decay_halflife_bars <= 0
        ):
            raise ValueError("sample_weight_time_decay_halflife_bars must be > 0 when set")
        if not (0.0 < self.sample_weight_quality_clip_min <= 1.0):
            raise ValueError("sample_weight_quality_clip_min must satisfy 0 < x <= 1")
        if self.horizon_experiment_enabled:
            if len(self.horizon_candidates) == 0:
                raise ValueError(
                    "horizon_candidates must be non-empty when horizon_experiment_enabled=True"
                )
            if any(h < 1 for h in self.horizon_candidates):
                raise ValueError("horizon_candidates must contain positive integers")
        allowed_groups = {
            "trend",
            "reversal",
            "volatility",
            "carry",
            "liquidity",
            "market_context",
            "microstructure",
            "missingness",
        }
        invalid_groups = [g for g in self.feature_groups_enabled if g not in allowed_groups]
        if invalid_groups:
            raise ValueError(f"unsupported feature group(s): {invalid_groups}")
