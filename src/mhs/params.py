"""Single source of truth for every MHS domain tunable.

The 73 * (and closely-related uppercase) tunables were previously split
across ``src/mhs/contracts.py`` and ``src/application/research/mhs/evaluation.py``
with no ownership rule. This module is the sole owner of all of them, satisfying
the I2 single-source invariant. ``src/mhs/contracts.py`` is reduced to types
only; ``evaluation.py`` imports its tunables from here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class GrowthRiskEnvelope:
    """Immutable growth risk constraint envelope (I1: single risk definition).

    ``max_drawdown`` is the single registered drawdown budget consumed by both
    ``solve_growth_optimal_risk`` and ``_drawdown_budget_reasons``.
    ``leverage_ceiling`` is the maximum allowed exposure scale (unit-gross
    scaffolding, not a risk parameter).  ``ruin_fraction`` and ``max_ruin_prob``
    are identical across all registered envelopes and are not swept (I2).
    """

    name: str
    max_drawdown: float
    max_drawdown_prob: float
    ruin_fraction: float
    max_ruin_prob: float
    horizon_years: float
    leverage_ceiling: float

    def __post_init__(self) -> None:
        if self.leverage_ceiling < 1.0:
            raise ValueError(
                f"leverage_ceiling must be >= 1.0, got {self.leverage_ceiling}"
            )
        if self.max_drawdown <= 0:
            raise ValueError(f"max_drawdown must be > 0, got {self.max_drawdown}")
        if not (0 < self.max_drawdown_prob <= 1.0):
            raise ValueError(
                f"max_drawdown_prob must be in (0, 1], got {self.max_drawdown_prob}"
            )
        if not (0 < self.max_ruin_prob <= 1.0):
            raise ValueError(
                f"max_ruin_prob must be in (0, 1], got {self.max_ruin_prob}"
            )
        if not (0 < self.ruin_fraction < 1.0):
            raise ValueError(
                f"ruin_fraction must be in (0, 1), got {self.ruin_fraction}"
            )
        if self.horizon_years <= 0:
            raise ValueError(f"horizon_years must be > 0, got {self.horizon_years}")


# --- cost / discovery / book construction ------------------------------------

MEASURED_EXECUTION_COST_TIERS_BPS: dict[str, float] = {
    "optimistic": 2.64,
    "base": 4.18,
    "stress": 6.07,
}

DISCOVERY_START: pd.Timestamp = pd.Timestamp("2021-01-01", tz="UTC")

BOOK_BLEND_WEIGHTS: dict[str, float] = {
    "fast_reversal": 0.0,
    "slow_momentum": 1.0,
}

CRASH_REGIME_REFERENCE_SYMBOLS: tuple[str, ...] = ("BTCUSDT",)

REVERSAL_HORIZON_CANDIDATES_HOURS: tuple[int, ...] = (24, 48, 72, 96, 120, 144, 168)

MOMENTUM_HORIZON_CANDIDATES_HOURS: tuple[int, ...] = (
    72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336,
    360, 384, 408, 432, 456, 480, 504,
)

FUNDING_CARRY_LOOKBACK_CANDIDATES_HOURS: tuple[int, ...] = (24, 72, 168, 336, 504)
FUNDING_CARRY_SLEEVE_LOOKBACK_HOURS: int = 168
FUNDING_CARRY_SLEEVE_WEIGHT: float = 0.30

TREND_SLEEVE_HORIZONS_HOURS: tuple[int, ...] = (336, 480, 600, 720, 1080, 1440)

FEATURE_MIN_COVERAGE: float = 0.90

COMMITTEE_MEMBER_SETS: dict[str, tuple[str, ...]] = {
    "flow_momentum": (
        "flow_imb_720h",
        "flow_imb_168h",
        "xs_mom_336h",
        "xs_idio_mom_336h",
        "mom3_skew_168h",
    ),
    "risk_premia": (
        "flow_imb_720h",
        "flow_imb_168h",
        "mom3_skew_168h",
        "lowvol_168h",
        "rev_24h",
    ),
}

COMMITTEE_DEFAULT_MEMBER_SET: str = "flow_momentum"

COMMITTEE_MEMBERS: tuple[str, ...] = COMMITTEE_MEMBER_SETS[
    COMMITTEE_DEFAULT_MEMBER_SET
]

COMMITTEE_TARGET_VOL: float = 0.15
COMMITTEE_TARGET_GROSS: float = 0.92

PNL_TARGET_ANNUAL_VOL: float = 0.20
PNL_VOL_TARGET_EWMA_HALFLIFE_DAYS: int = 20

COMMITTEE_PURGE_HOURS: int = 720

COMMITTEE_OOS_START: pd.Timestamp = pd.Timestamp("2023-01-01", tz="UTC")

COMMITTEE_TRANCHE_COUNT: int = 3

COMMITTEE_REGIME_ADAPTIVE_WINDOW: int = 15

COMMITTEE_GROWTH_RISK_GRID_MULTIPLIERS: tuple[float, ...] = (
    0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0,
)

# DIAGNOSTIC-ONLY constant for the leverage-frontier-scan CLI flag; must never
# be merged with or substituted for COMMITTEE_GROWTH_RISK_GRID_MULTIPLIERS --
# that production grid drives growth_budget_annual_vol's actual target_vol
# solve (changing it changes deployed exposure), while this one drives nothing
# but a read-only report.
LEVERAGE_FRONTIER_SCAN_MULTIPLES: tuple[float, ...] = (
    0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0,
    2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0,
    4.25, 4.5, 4.75, 5.0,
)
COMMITTEE_GROWTH_MAX_DRAWDOWN: float = 0.25
COMMITTEE_GROWTH_MAX_DRAWDOWN_PROB: float = 0.10
COMMITTEE_GROWTH_RUIN_FRACTION: float = 0.60
COMMITTEE_GROWTH_MAX_RUIN_PROB: float = 0.01
COMMITTEE_GROWTH_HORIZON_YEARS: float = 3.0
COMMITTEE_GROWTH_N_PATHS: int = 2000
COMMITTEE_GROWTH_BARS_PER_YEAR: int = 365

GROWTH_RISK_ENVELOPES: dict[str, GrowthRiskEnvelope] = {
    "conservative": GrowthRiskEnvelope(
        name="conservative",
        max_drawdown=COMMITTEE_GROWTH_MAX_DRAWDOWN,
        max_drawdown_prob=COMMITTEE_GROWTH_MAX_DRAWDOWN_PROB,
        ruin_fraction=COMMITTEE_GROWTH_RUIN_FRACTION,
        max_ruin_prob=COMMITTEE_GROWTH_MAX_RUIN_PROB,
        horizon_years=COMMITTEE_GROWTH_HORIZON_YEARS,
        leverage_ceiling=1.0,
    ),
    "balanced": GrowthRiskEnvelope(
        name="balanced",
        max_drawdown=0.35,
        max_drawdown_prob=COMMITTEE_GROWTH_MAX_DRAWDOWN_PROB,
        ruin_fraction=COMMITTEE_GROWTH_RUIN_FRACTION,
        max_ruin_prob=COMMITTEE_GROWTH_MAX_RUIN_PROB,
        horizon_years=COMMITTEE_GROWTH_HORIZON_YEARS,
        leverage_ceiling=1.0,
    ),
    "growth_moderate": GrowthRiskEnvelope(
        name="growth_moderate",
        max_drawdown=1.0,
        max_drawdown_prob=1.0,
        ruin_fraction=COMMITTEE_GROWTH_RUIN_FRACTION,
        max_ruin_prob=COMMITTEE_GROWTH_MAX_RUIN_PROB,
        horizon_years=COMMITTEE_GROWTH_HORIZON_YEARS,
        leverage_ceiling=1.5,
    ),
    "growth": GrowthRiskEnvelope(
        name="growth",
        max_drawdown=1.0,
        max_drawdown_prob=1.0,
        ruin_fraction=COMMITTEE_GROWTH_RUIN_FRACTION,
        max_ruin_prob=COMMITTEE_GROWTH_MAX_RUIN_PROB,
        horizon_years=COMMITTEE_GROWTH_HORIZON_YEARS,
        leverage_ceiling=2.0,
    ),
    # Permanent rung: selected per ADR_20260823_MHS_LEVERAGE_FRONTIER_SCAN
    # with every fold primary_valid and no CAPITAL_INVARIANT_BREACH.
    "growth_extreme": GrowthRiskEnvelope(
        name="growth_extreme",
        max_drawdown=1.0,
        max_drawdown_prob=1.0,
        ruin_fraction=COMMITTEE_GROWTH_RUIN_FRACTION,
        max_ruin_prob=COMMITTEE_GROWTH_MAX_RUIN_PROB,
        horizon_years=COMMITTEE_GROWTH_HORIZON_YEARS,
        leverage_ceiling=3.0,
    ),
    # Budgeted twin of growth_extreme: identical leverage_ceiling so the
    # resolved exposure cap -- and therefore the deployed exposure -- is
    # unchanged, while the 0.60 drawdown budget sits at the registered ceiling
    # and makes the risk contract enforceable ex post.
    "growth_extreme_budgeted": GrowthRiskEnvelope(
        name="growth_extreme_budgeted",
        max_drawdown=0.60,
        max_drawdown_prob=0.10,
        ruin_fraction=COMMITTEE_GROWTH_RUIN_FRACTION,
        max_ruin_prob=COMMITTEE_GROWTH_MAX_RUIN_PROB,
        horizon_years=COMMITTEE_GROWTH_HORIZON_YEARS,
        leverage_ceiling=3.0,
    ),
}

GROWTH_ENVELOPE_DEFAULT: str = "conservative"

SEARCH_TRIALS_ATTEMPTED: int = 70

# The window the CLI defaults (growth_extreme, committee_kelly_sizing, breadth
# 60) were measured on (see pipeline/config.py provenance notes). Any report
# whose window intersects this span is partially in-sample for those defaults;
# the overlap fraction is disclosed observationally, never silently omitted.
DEFAULT_SELECTION_WINDOW: tuple[pd.Timestamp, pd.Timestamp] = (
    pd.Timestamp("2021-01-01", tz="UTC"),
    pd.Timestamp("2025-12-31", tz="UTC"),
)

# MHS-local one-time final-OOS ceiling (2026-08-25 user-authorized decision):
# strictly narrower than any unseal of the shared HOLDOUT_CUTOFF gate.
MHS_FINAL_OOS_CUTOFF_2026H1: pd.Timestamp = pd.Timestamp("2026-06-30 23:59:59", tz="UTC")

RAM_BUDGET_FRACTION: float = 0.85
RAM_RESERVE_FRACTION: float = 0.05
RAM_RESERVE_FLOOR_BYTES: int = 256 * 2**20

WORKER_PEAK_RSS_BYTES: int = 3 * 2**30

REGISTERED_POLICY_THRESHOLDS: dict[str, float | None] = {
    "cap_60_roster": 60.0,
    "primary_annual_return": 0.05,
    # Conventional pass line for the Deflated Sharpe Ratio under the registered
    # trials denominator; below-threshold DSR blocks the Research-GO decision.
    "deflated_sharpe_ratio": 0.95,
    # Upper bound on any admissible drawdown budget: an envelope whose
    # max_drawdown exceeds this can never bind (-100% is capital extinction),
    # so a GO judged under it must be blocked with DRAWDOWN_BUDGET_NON_BINDING.
    "max_drawdown_budget_ceiling": 0.60,
}

FILL_MARK_PRICE_PROTECTION_BAND: float = 0.05
FILL_MARK_MAX_LOG_DIVERGENCE: float = math.log1p(FILL_MARK_PRICE_PROTECTION_BAND)
PNL_VOL_TARGET_MAX_SCALE: float = 1.0 / COMMITTEE_TARGET_GROSS

# --- evaluation.py tunables ----------------------------------------------------

DISCOVERY_GATE_TRANCHE_COUNT: int = 8
STRESS_COST_MULTIPLIER: float = 3.0

DISCOVERY_REVERSAL_CANDIDATES: tuple[int, ...] = (24, 48, 72, 96, 120, 144, 168)
DISCOVERY_MOMENTUM_CANDIDATES: tuple[int, ...] = MOMENTUM_HORIZON_CANDIDATES_HOURS

FEATURE_NAME = "multi_horizon_market_state"
PERIODS_PER_YEAR_1H: float = 365.0 * 24.0

WALK_FORWARD_MIN_TRAIN_BARS: int = 2000

GO_PRIMARY_SHARPE_FLOOR: float = 0.6
ARTIFACT_SCHEMA_VERSION: int = 1
ARTIFACT_CATEGORIES: tuple[str, ...] = (
    "fills",
    "units",
    "notional_weights",
    "ledger",
    "times",
)

REBALANCE_DEADBAND_POSITION_FRACTION: float = 0.25
BOOK_HOLDINGS_STATIONARITY_TOLERANCE: float = 0.25
FOLD_BLEND_PARITY_TOLERANCE: float = 0.25
# fold 실현변동성 log-ratio 관측 허용폭(≈1.42x): 관측 전용, reason code 없음.
FOLD_REALIZED_RISK_PARITY_TOLERANCE: float = 0.35

# --- 증거 게이트 보정(I-CALIB) ---------------------------------------------------
# 등록 상수는 원시 지표 임계값이 아니라 선언된 오차율 alpha이며, 임계값은 각 런에서
# 전략 자신의 pooled 수익률 null로부터 파생된다(실측: 오탈락 5.7%/탐지력 99.6%).
# FOLD_GROWTH_CONCENTRATION_MAX_SHARE = 0.5 는 유도 근거 부재(동일분포 null의
# 83 백분위)로 삭제되었고, alpha 파생 임계값이 이를 대체한다.
EVIDENCE_GATE_ALPHA: float = 0.05
NULL_BOOTSTRAP_MEAN_BLOCK_DAYS: int = 20
NULL_BOOTSTRAP_TRIALS: int = 2000
# I-DETERMINISTIC: 동일 원장 -> 비트 동일 임계값을 보장하는 등록 시드.
NULL_BOOTSTRAP_SEED: int = 20260823
# null 적합에 필요한 최소 유한 pooled 일간 행수(미만 시 fail-closed).
NULL_BOOTSTRAP_MIN_ROWS: int = 250

SIGNAL_EMA_HORIZON_SPAN: float = 1.0

REGIME_CASH_SCALE_FLOOR: float = 0.5
REGIME_CASH_MEDIAN_WINDOW_HOURS: int = 720
REFERENCE_PASS_EQUITY_FLOOR: float = REGIME_CASH_SCALE_FLOOR

PNL_VOL_TARGET_WINDOW_DAYS: int = 21
PNL_VOL_TARGET_SCALE_FLOOR: float = 0.2
PNL_VOL_TARGET_BURN_IN_DAYS: int = 90
PNL_VOL_TARGET_MEDIAN_WINDOW_DAYS: int = 365

# constant_risk 모드 전용 상수(기존 모드의 PNL_VOL_TARGET_* 는 불변).
# 실측(3m 원장, target=0.40 고정) -- halflife가 유일한 다이얼로는 두 게이트를
# 동시에 통과시키지 못하는 단조 트레이드오프가 실측 확인됨:
#   hl=90  -> fold_blend_parity=0.317(FAIL>0.25) / risk_parity=0.234(PASS) / share=0.526
#   hl=120 -> fold_blend_parity=0.264(FAIL>0.25) / risk_parity=0.309(PASS) / share=0.547
#   hl=150 -> fold_blend_parity=0.230(PASS)      / risk_parity=0.373(FAIL>0.35) / share=0.563
# hl=90을 등록: 이 기능이 겨냥한 1차 목표(FOLD_GROWTH_CONCENTRATION, share 최소화)에
# 가장 근접. FOLD_BLEND_PATH_DIVERGENCE는 미해결 -- ADR_20260823_MHS_CONSTANT_RISK_DEPLOYMENT
# 후속 과제(fold가 자체 EWMA를 재적합하지 않고 blend의 연속 스케일 궤적을 날짜로
# 슬라이스해 재사용하는 구조적 대안이 유력, 미구현).
CONSTANT_RISK_EWMA_HALFLIFE_DAYS: int = 90
# EWMA sigma 해석에 필요한 최소 유한 행수(fold 워밍업으로 데드존 제거).
CONSTANT_RISK_MIN_PERIODS_DAYS: int = 45
# sigma* = leverage_ceiling * q_p(sigma_book|train): cap 포화 확률 <= p 보장.
CONSTANT_RISK_CAP_BINDING_QUANTILE: float = 0.10
# 고정 목표 위험(단일 상수, fold 경계별로 재적합하지 않음): growth_budget_annual_vol을
# 경계별로 재해석하면 표본 특이적 해가 fold마다 갈라져 위험 등화가 깨진다(실측: fold0-2
# 실현변동성 0.14~0.16 vs fold3 0.29, log-ratio 0.63 -- FOLD_GROWTH_CONCENTRATION 재발).
# 단일 고정값만이 모든 경계에서 동일 목표를 강제한다; _feasible_constant_risk_target의
# leverage_ceiling*q10(sigma_book|train) 클램프는 경계별로 유지된다(실현 가능성 검증).
# 실측(3m 원장, target=0.45/halflife=60d): 배치 실현위험이 0.39~0.51로 근접했으나
# fold3의 잔여 초과 Sharpe가 share=0.518로 남음 -- 0.40으로 하향.
CONSTANT_RISK_TARGET_ANNUAL_VOL: float = 0.40

# 인과적 자기자본 드로다운 브레이크: brake = clip(1 + k * underwater, floor, 1).
# k=2.0 selected per ADR_20260823_MHS_CONSTANT_RISK_DEPLOYMENT.
EXPOSURE_DRAWDOWN_BRAKE_K: float = 2.0
# 하한 fail-closed 경계(0.1/0.2 실측 동일 결과 -- 튜닝값이 아닌 구속 하한).
EXPOSURE_DRAWDOWN_BRAKE_FLOOR: float = 0.2

FOLD_PANEL_WARMUP_HOURS: int = 720 + 168 + 24

EXECUTION_ROSTER_EXIT_MULTIPLIER: float = 2.0
REBALANCE_TRACKING_ERROR_THRESHOLD: float = 0.20

CAUSAL_BETA_LOOKBACK_BARS: int = 720
CAUSAL_BETA_MIN_PERIODS: int = 360


SIGNAL_PANEL_WINDOW_DAYS: int = 120
SIGNAL_REPLAY_WARMUP_DAYS: int = 30
SIGNAL_RETURN_TAIL_DAYS: int = 400
SIGNAL_OVERLAP_TOLERANCE: float = 1e-9

# Sentinel distinguishing the registered default exposure from an explicit
# committee_target_gross value: a bare MhsDiagnosticRequest() resolves to the
# registered constant without triggering the committee_capital requirement,
# while an explicit non-None value keeps requiring committee_capital=True.
COMMITTEE_TARGET_GROSS_UNSET: object = object()
