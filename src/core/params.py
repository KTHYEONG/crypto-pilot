"""Shared tunables for deployable layers and research.

Constants used exclusively by the research lab are owned by
``src.lab.mhs.params``; shared constants and their dependencies remain here.
"""

from __future__ import annotations

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

#: Single definition of the audit-log retention window (moved from src.live.audit so
#: core never imports live; live imports this constant instead).
AUDIT_LOG_RETENTION_DAYS: int = 90

#: Minimum evidence span for execution-quality summaries; equals the audit retention window.
EXECUTION_QUALITY_MIN_EVIDENCE_DAYS: int = AUDIT_LOG_RETENTION_DAYS

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

STRATEGY_RISK_ENVELOPE: str = "growth_extreme_budgeted"

ACCOUNT_EXPOSURE_CAP: float = GROWTH_RISK_ENVELOPES[STRATEGY_RISK_ENVELOPE].leverage_ceiling

CLI_GROWTH_ENVELOPE_DEFAULT: str = "growth_extreme_budgeted"


SEARCH_TRIALS_ATTEMPTED: int = 70




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

PNL_VOL_TARGET_MAX_SCALE: float = 1.0 / COMMITTEE_TARGET_GROSS

# --- evaluation.py tunables ----------------------------------------------------

STRESS_COST_MULTIPLIER: float = 3.0
# 4시간 점검은 시간별 점검과 동일한 MDD 감소, 결정 시점 단일 점검은 무효.
NAME_DRIFT_TRIM_INTERVAL_HOURS: int = 4





BOOK_HOLDINGS_STATIONARITY_TOLERANCE: float = 0.25

NULL_BOOTSTRAP_MEAN_BLOCK_DAYS: int = 20




# 실측 등록값(측정 전용, 런타임 로직 비참조): 배포 레퍼런스 북 일간수익률의
# leak-free 학습 슬라이스에서 mean/std = 0.1648. 데이터 재측정 시 함께 갱신한다.
COMMITTEE_KELLY_TRAIN_DAILY_SHARPE: float = 0.1648


# 인과적 자기자본 드로다운 브레이크: brake = clip(1 + k * underwater, floor, 1).
# k=2.0 selected per ADR_20260823_MHS_CONSTANT_RISK_DEPLOYMENT.
EXPOSURE_DRAWDOWN_BRAKE_K: float = 2.0
# 하한 fail-closed 경계(0.1/0.2 실측 동일 결과 -- 튜닝값이 아닌 구속 하한).
EXPOSURE_DRAWDOWN_BRAKE_FLOOR: float = 0.2

# Trailing quote-volume window (1h bars) of the liquid-half universe rule. The research
# folds, top-level selection, process backtest and execution-data plan must share one
# value: a divergence lets research validate a universe that production never trades.
# The execution roster ranks this same trailing mean, so the rank window is bound here too.
UNIVERSE_ELIGIBILITY_LOOKBACK_BARS: int = 720
# Observed 1h bars required before a symbol can be liquidity-eligible; missing history is
# ineligible, never zero-filled. Must satisfy 1 <= value <= UNIVERSE_ELIGIBILITY_LOOKBACK_BARS.
UNIVERSE_ELIGIBILITY_MIN_HISTORY_BARS: int = 720


"""Max absolute difference between a fold's own train-window target weight and the shared
reference's prefix weight for the shared replay to be reused (float-association drift only)."""

TRAIN_REFERENCE_PREFIX_RETURN_ATOL: float = 1e-12
"""Accepted absolute drift of a reused train-reference daily return versus the independent
per-fold replay (measured maximum 2.2e-16); the acceptance bound asserted by the equivalence and
perturbation tests."""

EXECUTION_ROSTER_EXIT_MULTIPLIER: float = 2.0



SIGNAL_PANEL_WINDOW_DAYS: int = 400
# A symbol dropped by the panel history filter can never become eligible, so the panel
# filter is bound to the eligibility history requirement rather than tuned separately.
PANEL_MIN_HISTORY_BARS: int = UNIVERSE_ELIGIBILITY_MIN_HISTORY_BARS

# --- canonical run retention -------------------------------------------------------
# Default detail-retention quotas for canonical 3-minute runs; None keeps every
# managed bundle for the byte budget, while the run-count budget defaults to
# only the single most recent unprotected finalized bundle. The CLI references
# these names instead of hiding literals.
DEFAULT_DETAIL_RETENTION_MAX_BYTES: int | None = None
DEFAULT_DETAIL_RETENTION_MAX_RUNS: int | None = 1

# --- strategy growth policy ------------------------------------------------------
# 종목별 클립: 무레버 Sharpe 2.22→2.34, 3m 마크 드로다운 증폭 1.293→1.148 (단일종목 장중 급락 경로 차단).
STRATEGY_NAME_CLIP: float = 0.05
# 0.25 단위 rung 중 원장수익률 부트스트랩이 P(3y MDD > 35%) ≤ 10%를 유지하는 최대치(허용 2.51).
# 전략 북이나 데이터가 바뀌면 동일 부트스트랩으로 반드시 재도출해야 한다.
GROWTH_EXPOSURE_MULTIPLIER: float = 2.5
# 이웃 설정 Sharpe 정점 대비 선택 편향으로 깎이는 평균 비율.
EXPOSURE_SCAN_MEAN_HAIRCUT: float = 0.25
# 하루 만에 이름값의 절반 이상이 움직이면 헷지로 막을 수 없는 갭으로 본다.
EXPOSURE_SCAN_GAP_THRESHOLD: float = 0.50
# 로그성장 곡선을 재는 노출 rung 격자. 8.0에서 멈추면 실제 정점(측정상 L≈10)보다 낮은 지점에서
# argmax가 격자 상한에 그대로 걸려버려(파산확률은 여전히 0) 진짜 위험기반 상한이 아니라 격자
#길이가 답을 정하는 결과가 나온다. 갭 파산확률이 유의미하게 관측되는 구간(L>=12)까지 반드시 포함한다.
EXPOSURE_SCAN_GRID: tuple[float, ...] = tuple(round(1.0 + 0.25 * i, 2) for i in range(61))  # 1.0 ~ 16.0
# 스트레스 곡선이 평평한 구간에서는 추정 잡음이 argmax를 정하므로 최적 근처를 고원으로 둔다.
EXPOSURE_SCAN_PLATEAU_TOLERANCE: float = 0.05
# 동일 입력에 비트 동일 해를 보장하는 등록 시드.
EXPOSURE_SCAN_SEED: int = 20260921

# --- account-scale research ledger -------------------------------------------------
# 선언 최소 소매 시작금 ₩3,000,000(₩1,430/USD 환산).
ACCOUNT_DEFAULT_CAPITAL_USDT: float = 2100.0
# 사전 "엣지 없음(μ=0)"에 2년 표본만큼의 가중을 둔다. 결과를 보고 고른 값이 아니라
# 선언값이며, 2년 이상 쌓인 실적이 있어야 사후 μ가 표본평균의 절반을 넘는다.
ACCOUNT_PRIOR_DAYS: float = 730.0
# 30일 미만의 관측으로는 표본분산을 신뢰할 수 없으므로, 그동안은 최소 rung만 쓴다.
ACCOUNT_MIN_MOMENT_DAYS: int = 30
# 단위북 원장(x1, 주문필터 끔, 충격 0)의 기준 자본. 이 조건에서 수익률은 자본과 무관하다.
ACCOUNT_UNIT_REFERENCE_CAPITAL: float = 1e5
# 사후 μ의 Kelly 비율이 `1 - haircut`이다(0.5 = 사후 half-Kelly). 추정오차가 있을 때
# 과소베팅보다 과대베팅의 손실이 더 크다는 비대칭 때문이다.
ACCOUNT_MEAN_HAIRCUT: float = 0.50
# 관측 최악 단위 일간손실(-4.04%)의 2배인 adverse 충격.
ACCOUNT_SHOCK_PER_UNIT: float = 0.08
# adverse 후에도 자기자본의 10%는 마진 버퍼로 남긴다.
ACCOUNT_MARGIN_RESERVE: float = 0.10
# 개시증거금은 자기자본의 90% 이내로 제한한다.
ACCOUNT_INITIAL_MARGIN_CAP: float = 0.90
# 프로브 구간 0.4~1.0의 중앙 제곱근 충격 계수.
ACCOUNT_IMPACT_Y: float = 0.6
# 노출 rung 격자 상한(진단용 fixed가 그대로 쓰는 값).
ACCOUNT_EXPOSURE_MAX: float = 10.0
# 노출 rung 격자 간격(0.25 단위에서 margin cap이 자기자본 복리에 반응한다).
ACCOUNT_EXPOSURE_STEP: float = 0.25
# 테이커 수수료 6bp(프로브 실측 기준).
ACCOUNT_TAKER_FEE_BPS: float = 6.0
# Binance USD-M 메이커 수수료로, 공식 원장 ExecutionSpec.maker_fee_bps와 같은 값.
ACCOUNT_MAKER_FEE_BPS: float = 2.0
# 공식 원장 passive_timeout_minutes(30분)를 3분봉 개수로 환산한 값. 대기창이 길수록 체결률은
# 오르지만 역선택도 커진다. 공식 원장과 같은 길이를 써야 두 원장이 비교 가능하다.
ACCOUNT_PASSIVE_WINDOW_BARS: int = 10
# 같은 북·같은 창의 공식 3m 원장 대비 CAGR 허용 오차(프로브 실측 잔차 0.3%p의 여유폭).
ACCOUNT_RECON_CAGR_TOLERANCE: float = 0.005
# 같은 정의(3m 종가 경로 최고점 대비)의 MDD 허용 오차.
ACCOUNT_RECON_MDD_TOLERANCE: float = 0.01

# --- live strategy paper ---------------------------------------------------------
# 120일 1h 창에서 전략 비중이 전체 이력과 비트 동일함을 실측했다(로스터 90일 거래 요건 +
# 30일 중앙값 + 720h 피처). 이보다 짧으면 로스터가 비거나 달라진다.
LIVE_SIGNAL_WARMUP_DAYS: int = 120
# 단위 proxy 수익률의 회전 비용. 메이커 체결률 약 98% 실측에 맞춰 메이커 수수료를 쓴다.
LIVE_UNIT_PROXY_COST_BPS: float = ACCOUNT_MAKER_FEE_BPS

# --- instrument lifecycle (spec 34) ---

# Disclosed proxy for a missing real announcement; mirrors Binance's usual ~1-week notice for scheduled delistings.
DELIST_ANNOUNCEMENT_LEAD: pd.Timedelta = pd.Timedelta(days=7)
# Settlement fee charged on delivered notional; equals ExecutionSpec().taker_fee_bps and the live delisting_settlement_fee_bps default.
DELIST_SETTLEMENT_FEE_BPS: float = 5.0
# Mirrors live delisting_settlement_min_flat_bars.
SETTLEMENT_EVIDENCE_MIN_FLAT_BARS: int = 3
# Mirrors live delisting_settlement_price_rtol.
SETTLEMENT_EVIDENCE_PRICE_RTOL: float = 1e-9
# Binance index-average settlement window.
SETTLEMENT_PROXY_TWAP_WINDOW: pd.Timedelta = pd.Timedelta(minutes=30)
# At least half of the ten 3m bars must have traded for the TWAP to describe the final half hour.
SETTLEMENT_PROXY_MIN_BARS: int = 5
# Every lake settlement price lies inside this pre-last-trade liquid range (145/145 measured); outside it a price is treated as an evidence error.
SETTLEMENT_PRICE_ENVELOPE_LOOKBACK: pd.Timedelta = pd.Timedelta(hours=24)
# Trailing flat run length that marks a delisted (forward-filled) tail; mirrors the live min-flat-bars rule.
SETTLEMENT_AUDIT_MIN_TRAILING_FLAT_BARS: int = 3

# --- spec 34: instrument lifecycle settlement (part 2 engine) ---
SETTLEMENT_PRICE_STRESS_HAIRCUT_BPS: float = 450.0
DELIST_ROSTER_BLOCK_LEAD: pd.Timedelta = pd.Timedelta(hours=48)
DELIST_FORCED_EXIT_LEAD: pd.Timedelta = pd.Timedelta(hours=72)
VENUE_HALT_MIN_ZERO_FRACTION: float = 0.9
VENUE_HALT_MIN_PRESENT_SYMBOLS: int = 10
# Measured gap between the longest temporary freeze (96 min) and the shortest dead market (39 days).
EXIT_DEFERRAL_MAX_AGE: pd.Timedelta = pd.Timedelta(hours=24)

# --- spec 38 part 5: decision-grade strategy report --------------------------------
# Stationary block-bootstrap budget for one same-length future path (report only).
REPORT_BOOTSTRAP_PATHS: int = 2000
REPORT_BOOTSTRAP_SEED: int = 20261008

# 2026-07-01 이후 미사용 전진 구간을 보존하는 평가 상한.
# MHS-local one-time final-OOS ceiling (2026-08-25 user-authorized decision):
# strictly narrower than any unseal of the shared HOLDOUT_CUTOFF gate.
MHS_FINAL_OOS_CUTOFF_2026H1: pd.Timestamp = pd.Timestamp("2026-06-30 23:59:59", tz="UTC")

PROCESS_EVALUATION_CEILING: pd.Timestamp = MHS_FINAL_OOS_CUTOFF_2026H1
