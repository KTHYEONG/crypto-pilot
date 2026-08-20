"""Single source of truth for every MHS domain tunable.

The 73 MHS_* (and closely-related uppercase) tunables were previously split
across ``src/mhs/contracts.py`` and ``src/application/research/mhs/evaluation.py``
with no ownership rule. This module is the sole owner of all of them, satisfying
the I2 single-source invariant. ``src/mhs/contracts.py`` is reduced to types
only; ``evaluation.py`` imports its tunables from here.
"""

from __future__ import annotations

import math

import pandas as pd

# --- cost / discovery / book construction ------------------------------------

MEASURED_EXECUTION_COST_TIERS_BPS: dict[str, float] = {
    "optimistic": 2.64,
    "base": 4.18,
    "stress": 6.07,
}

MHS_DISCOVERY_START: pd.Timestamp = pd.Timestamp("2021-01-01", tz="UTC")

PHASE_1_BOOK_BLEND_WEIGHTS: dict[str, float] = {
    "fast_reversal": 0.0,
    "slow_momentum": 1.0,
}

MHS_CRASH_REGIME_REFERENCE_SYMBOLS: tuple[str, ...] = ("BTCUSDT",)

REVERSAL_HORIZON_CANDIDATES_HOURS: tuple[int, ...] = (24, 48, 72, 96, 120, 144, 168)

MOMENTUM_HORIZON_CANDIDATES_HOURS: tuple[int, ...] = (
    72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336,
    360, 384, 408, 432, 456, 480, 504,
)

MHS_FUNDING_CARRY_LOOKBACK_CANDIDATES_HOURS: tuple[int, ...] = (24, 72, 168, 336, 504)
MHS_FUNDING_CARRY_SLEEVE_LOOKBACK_HOURS: int = 168
MHS_FUNDING_CARRY_SLEEVE_WEIGHT: float = 0.30

MHS_TREND_SLEEVE_HORIZONS_HOURS: tuple[int, ...] = (336, 480, 600, 720, 1080, 1440)

MHS_FEATURE_MIN_COVERAGE: float = 0.90

MHS_COMMITTEE_MEMBER_SETS: dict[str, tuple[str, ...]] = {
    "flow_momentum_v1": (
        "flow_imb_720h",
        "flow_imb_168h",
        "xs_mom_336h",
        "xs_idio_mom_336h",
        "mom3_skew_168h",
    ),
    "risk_premia_v2": (
        "flow_imb_720h",
        "flow_imb_168h",
        "mom3_skew_168h",
        "lowvol_168h",
        "rev_24h",
    ),
}

MHS_COMMITTEE_DEFAULT_MEMBER_SET: str = "flow_momentum_v1"

MHS_COMMITTEE_MEMBERS: tuple[str, ...] = MHS_COMMITTEE_MEMBER_SETS[
    MHS_COMMITTEE_DEFAULT_MEMBER_SET
]

MHS_COMMITTEE_TARGET_VOL: float = 0.15
MHS_COMMITTEE_TARGET_GROSS: float = 0.92

MHS_PNL_TARGET_ANNUAL_VOL: float = 0.20
MHS_PNL_VOL_TARGET_EWMA_HALFLIFE_DAYS: int = 20

MHS_COMMITTEE_PURGE_HOURS: int = 720

MHS_COMMITTEE_OOS_START: pd.Timestamp = pd.Timestamp("2023-01-01", tz="UTC")

MHS_COMMITTEE_TRANCHE_COUNT: int = 3

MHS_COMMITTEE_REGIME_ADAPTIVE_WINDOW: int = 15

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

MHS_SEARCH_TRIALS_ATTEMPTED: int = 70

MHS_RAM_BUDGET_FRACTION: float = 0.85
MHS_RAM_RESERVE_FRACTION: float = 0.05
MHS_RAM_RESERVE_FLOOR_BYTES: int = 256 * 2**20

MHS_WORKER_PEAK_RSS_BYTES: int = 3 * 2**30

MHS_REGISTERED_POLICY_THRESHOLDS: dict[str, float | None] = {
    "cap_30_roster": 30.0,
    "primary_annual_return": 0.05,
}

MHS_FILL_MARK_PRICE_PROTECTION_BAND: float = 0.05
MHS_FILL_MARK_MAX_LOG_DIVERGENCE: float = math.log1p(MHS_FILL_MARK_PRICE_PROTECTION_BAND)
MHS_PNL_VOL_TARGET_MAX_SCALE: float = 1.0 / MHS_COMMITTEE_TARGET_GROSS

# --- evaluation.py tunables ----------------------------------------------------

MHS_DISCOVERY_GATE_TRANCHE_COUNT: int = 8
MHS_STRESS_COST_MULTIPLIER: float = 3.0

MHS_DISCOVERY_REVERSAL_CANDIDATES: tuple[int, ...] = (24, 48, 72, 96, 120, 144, 168)
MHS_DISCOVERY_MOMENTUM_CANDIDATES: tuple[int, ...] = MOMENTUM_HORIZON_CANDIDATES_HOURS

_MHS_FEATURE = "multi_horizon_market_state"
PERIODS_PER_YEAR_1H: float = 365.0 * 24.0

_MHS_WALK_FORWARD_MIN_TRAIN_BARS: int = 2000

MHS_GO_PRIMARY_SHARPE_FLOOR: float = 0.6
MHS_ARTIFACT_SCHEMA_VERSION: int = 1
MHS_ARTIFACT_CATEGORIES: tuple[str, ...] = (
    "fills",
    "units",
    "notional_weights",
    "ledger",
    "times",
)

MHS_REBALANCE_DEADBAND_POSITION_FRACTION: float = 0.25
MHS_BOOK_HOLDINGS_STATIONARITY_TOLERANCE: float = 0.25
MHS_FOLD_BLEND_PARITY_TOLERANCE: float = 0.25
MHS_FOLD_GROWTH_CONCENTRATION_MAX_SHARE: float = 0.5

MHS_SIGNAL_EMA_HORIZON_SPAN: float = 1.0

MHS_REGIME_CASH_SCALE_FLOOR: float = 0.5
MHS_REGIME_CASH_MEDIAN_WINDOW_HOURS: int = 720
MHS_REFERENCE_PASS_EQUITY_FLOOR: float = MHS_REGIME_CASH_SCALE_FLOOR

MHS_PNL_VOL_TARGET_WINDOW_DAYS: int = 21
MHS_PNL_VOL_TARGET_SCALE_FLOOR: float = 0.2
MHS_PNL_VOL_TARGET_BURN_IN_DAYS: int = 90
MHS_PNL_VOL_TARGET_MEDIAN_WINDOW_DAYS: int = 365

MHS_FOLD_PANEL_WARMUP_HOURS: int = 720 + 168 + 24

MHS_EXECUTION_ROSTER_EXIT_MULTIPLIER: float = 2.0
MHS_REBALANCE_TRACKING_ERROR_THRESHOLD: float = 0.20

MHS_CAUSAL_BETA_LOOKBACK_BARS: int = 720
MHS_CAUSAL_BETA_MIN_PERIODS: int = 360


# Sentinel distinguishing the registered default exposure from an explicit
# committee_target_gross value: a bare MhsDiagnosticRequest() resolves to the
# registered constant without triggering the committee_capital requirement,
# while an explicit non-None value keeps requiring committee_capital=True.
_MHS_COMMITTEE_TARGET_GROSS_UNSET: object = object()
