"""Frozen Phase 1 MHS contracts.

All literals here are preregistered measurement outputs or frozen architecture
decisions (``docs/architecture/multi-horizon-market-state.md``). A change to any of
them is a new contract revision, never an inline edit at a call site.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class HorizonBand:
    """A measured return band. ``sign`` is a measured band property.

    The fast band reverses (``sign=-1``) and the slow band follows momentum
    (``sign=+1``); both signs were measured on discovery data (spec §2.1) and
    are never inferred at call sites.
    """

    name: str
    horizons_hours: tuple[int, ...]
    sign: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        if not self.horizons_hours:
            raise ValueError("horizons_hours must not be empty")
        if any(h <= 0 for h in self.horizons_hours):
            raise ValueError(f"horizons_hours must all be > 0, got {self.horizons_hours}")
        if tuple(self.horizons_hours) != tuple(sorted(set(self.horizons_hours))):
            raise ValueError(
                f"horizons_hours must be strictly ascending, got {self.horizons_hours}"
            )
        if self.sign not in (-1, 1):
            raise ValueError(f"sign must be -1 or +1, got {self.sign}")


@dataclass(frozen=True, slots=True)
class ExecutionSpec:
    """Passive-execution cost and fill contract.

    ``one_way_taker_bps`` reproduces ``CostModel()``'s 8.0 bp one-way assumption
    so the two cost models stay comparable.
    """

    maker_fee_bps: float = 2.0
    taker_fee_bps: float = 5.0
    taker_slippage_bps: float = 3.0
    passive_timeout_minutes: int = 30
    require_trade_through: bool = True
    ladder_tranches: int = 4

    def __post_init__(self) -> None:
        if min(self.maker_fee_bps, self.taker_fee_bps, self.taker_slippage_bps) < 0:
            raise ValueError("fees and slippage must be non-negative")
        if self.passive_timeout_minutes < 1:
            raise ValueError(f"passive_timeout_minutes must be >= 1, got {self.passive_timeout_minutes}")
        if self.ladder_tranches < 1:
            raise ValueError(f"ladder_tranches must be >= 1, got {self.ladder_tranches}")

    def one_way_taker_bps(self) -> float:
        """One-way all-in taker cost in bps (fee + slippage)."""
        return self.taker_fee_bps + self.taker_slippage_bps


@dataclass(frozen=True, slots=True)
class BookSpec:
    """One frozen Phase 1 book: band, signal horizon, decision step, min symbols.

    ``tranche_count()`` is the number of overlapping phase tranches held
    simultaneously (``horizon_hours // step_hours``), the phase-ensemble
    construction that makes the result independent of an arbitrary
    decision-clock offset (spec §2.4).
    """

    band: HorizonBand
    horizon_hours: int
    step_hours: int
    min_symbols: int = 8

    def __post_init__(self) -> None:
        if self.horizon_hours not in self.band.horizons_hours:
            raise ValueError(
                f"horizon_hours {self.horizon_hours} not in band horizons {self.band.horizons_hours}"
            )
        if self.step_hours < 1:
            raise ValueError(f"step_hours must be >= 1, got {self.step_hours}")
        if self.min_symbols < 2:
            raise ValueError(f"min_symbols must be >= 2, got {self.min_symbols}")
        if self.horizon_hours % self.step_hours != 0:
            raise ValueError("horizon_hours must be divisible by step_hours")

    def tranche_count(self) -> int:
        return self.horizon_hours // self.step_hours


MEASURED_EXECUTION_COST_TIERS_BPS: dict[str, float] = {
    "optimistic": 2.64,
    "base": 4.18,
    "stress": 6.07,
}

# MHS discovery window start for fold evaluation and application orchestration.
MHS_DISCOVERY_START: pd.Timestamp = pd.Timestamp("2021-01-01", tz="UTC")

# Initial Research-GO blend weights: fast_reversal (0.0 weight) maintained for diagnostic evaluation.
PHASE_1_BOOK_BLEND_WEIGHTS: dict[str, float] = {
    "fast_reversal": 0.0,
    "slow_momentum": 1.0,
}

# Fixed reference basket for crash-regime directional tilt (BTCUSDT for continuous listing history).
MHS_CRASH_REGIME_REFERENCE_SYMBOLS: tuple[str, ...] = ("BTCUSDT",)

# Reversal and momentum candidate grids for fold-scoped discovery selection.
REVERSAL_HORIZON_CANDIDATES_HOURS: tuple[int, ...] = (24, 48, 72, 96, 120, 144, 168)
_FAST_BAND = HorizonBand(name="fast_reversal", horizons_hours=REVERSAL_HORIZON_CANDIDATES_HOURS, sign=-1)

MOMENTUM_HORIZON_CANDIDATES_HOURS: tuple[int, ...] = (
    72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336,
    360, 384, 408, 432, 456, 480, 504,
)
_SLOW_BAND = HorizonBand(name="slow_momentum", horizons_hours=MOMENTUM_HORIZON_CANDIDATES_HOURS, sign=1)

# Candidate grid for funding-rate carry return source discovery.
MHS_FUNDING_CARRY_LOOKBACK_CANDIDATES_HOURS: tuple[int, ...] = (24, 72, 168, 336, 504)
# 캐리 슬리브 기본 lookback: 21회 펀딩 정산(1주), 고원 하한 경계.
MHS_FUNDING_CARRY_SLEEVE_LOOKBACK_HOURS: int = 168
# 캐리 슬리브 gross 비중: 0.25-0.35 고원 중앙값 (피팅값 아님).
MHS_FUNDING_CARRY_SLEEVE_WEIGHT: float = 0.30

# Candidate slow band for additive directional trend sleeve.
MHS_TREND_SLEEVE_HORIZONS_HOURS: tuple[int, ...] = (336, 480, 600, 720, 1080, 1440)

# Annual non-null coverage floor (90%) required for capital eligibility.
MHS_FEATURE_MIN_COVERAGE: float = 0.90

# Predefined wealth committee composition across economic families (flow, momentum, skew).
MHS_COMMITTEE_MEMBERS: tuple[str, ...] = (
    "flow_imb_720h",
    "flow_imb_168h",
    "xs_mom_336h",
    "xs_idio_mom_336h",
    "mom3_skew_168h",
)

# Annualized volatility target for committee position sizing.
MHS_COMMITTEE_TARGET_VOL: float = 0.15

MHS_COMMITTEE_TARGET_GROSS: float = 0.92

# 절대 ex-ante 변동성 타겟: 전략 자체 실현변동성 5년 중앙값(0.194)에 근접한 보수적 기준.
# discovery-window(2021-2022) 중앙값 0.23보다 낮아 OOS 과적합 아님.
MHS_PNL_TARGET_ANNUAL_VOL: float = 0.20
# EWMA halflife: 기존 MHS_PNL_VOL_TARGET_WINDOW_DAYS=21과 동일 반응속도.
MHS_PNL_VOL_TARGET_EWMA_HALFLIFE_DAYS: int = 20

# Purge gap (720h) between train and test windows to prevent label overlap leak.
MHS_COMMITTEE_PURGE_HOURS: int = 720

# Earliest bar for committee walk-forward out-of-sample evaluation.
MHS_COMMITTEE_OOS_START: pd.Timestamp = pd.Timestamp("2023-01-01", tz="UTC")

# 24h 결정 격자 x 3 = 실효 72h 신호 수명으로, 최단 멤버 lookback(168h) 대비
# 오버샘플링 배수를 7배->2.3배로 축소하는 구조적 선택 (피팅값 아님).
MHS_COMMITTEE_TRANCHE_COUNT: int = 3

# 위원회 raw 북 자기 proxy return의 causal trailing lag-1 자기상관 게이팅 창(결정
# 행 개수). 실제 3분봉 리플레이로 15~25 구간 전부 3개 fold 알파 게이트 동시 통과를
# 확인(고원, 단일 스파이크 아님); 창 10/90은 CAPITAL_INVARIANT_BREACH로 실패.
MHS_COMMITTEE_REGIME_ADAPTIVE_WINDOW: int = 15

# Discovery-window-only growth-optimal headroom diagnostic: risk-grid
# multipliers of the realized reference_risk, plus the constraint anchors
# (calibrated in the kelly_compounding_improve cycle: selected_risk=1.0x,
# headroom_ratio=1.3%, 25% MDD between the 20% gate and the ~31% k=6 baseline).
MHS_COMMITTEE_GROWTH_RISK_GRID_MULTIPLIERS: tuple[float, ...] = (
    0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0,
)
MHS_COMMITTEE_GROWTH_MAX_DRAWDOWN: float = 0.25
MHS_COMMITTEE_GROWTH_MAX_DRAWDOWN_PROB: float = 0.10
MHS_COMMITTEE_GROWTH_RUIN_FRACTION: float = 0.60
MHS_COMMITTEE_GROWTH_MAX_RUIN_PROB: float = 0.01
MHS_COMMITTEE_GROWTH_HORIZON_YEARS: float = 3.0
MHS_COMMITTEE_GROWTH_N_PATHS: int = 2000
MHS_COMMITTEE_GROWTH_BARS_PER_YEAR: int = 365

PHASE_1_BOOK_SPECS: dict[str, BookSpec] = {
    "fast_reversal": BookSpec(
        band=_FAST_BAND, horizon_hours=48, step_hours=6, min_symbols=8,
    ),
    "slow_momentum": BookSpec(
        band=_SLOW_BAND, horizon_hours=168, step_hours=24, min_symbols=8,
    ),
}

# Cumulative sequential discovery trials count for multiple-testing deflation.
MHS_SEARCH_TRIALS_ATTEMPTED: int = 70

# RAM-guard safety thresholds for parent stage execution.
MHS_RAM_BUDGET_FRACTION: float = 0.85
MHS_RAM_RESERVE_FRACTION: float = 0.05
MHS_RAM_RESERVE_FLOOR_BYTES: int = 256 * 2**20

# Estimated worker RSS budget for fork pools after COW memory optimization.
MHS_WORKER_PEAK_RSS_BYTES: int = 3 * 2**30

# Research-GO policy thresholds (None indicates conservative unregistered state).
# cap_30_roster mirrors the frozen execution_universe_size design cap (attestation
# only, not independently enforced against realized_execution_roster_size).
# primary_annual_return is enforced per anchored fold (see MHS_GO_REASON_PRIMARY_RETURN_BELOW_FLOOR):
# below the worst measured passing fold's net_ann (2024, 0.103) for headroom against
# future fold variance while still requiring a real economic hurdle over funding/costs.
MHS_REGISTERED_POLICY_THRESHOLDS: dict[str, float | None] = {
    "cap_30_roster": 30.0,
    "primary_annual_return": 0.05,
}

# Fill/Mark parity gate: Binance USDⓈ-M last-vs-mark price protection band.
MHS_FILL_MARK_PRICE_PROTECTION_BAND: float = 0.05
# Derived log-divergence threshold; never a literal.
MHS_FILL_MARK_MAX_LOG_DIVERGENCE: float = math.log1p(MHS_FILL_MARK_PRICE_PROTECTION_BAND)
# Two-sided exposure scale upper bound: target_gross * max_scale == 1.0 (I5).
MHS_PNL_VOL_TARGET_MAX_SCALE: float = 1.0 / MHS_COMMITTEE_TARGET_GROSS
