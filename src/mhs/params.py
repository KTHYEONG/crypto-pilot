"""Single source of truth for every MHS domain tunable.

The 73 * (and closely-related uppercase) tunables were previously split
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
COMMITTEE_GROWTH_MAX_DRAWDOWN: float = 0.25
COMMITTEE_GROWTH_MAX_DRAWDOWN_PROB: float = 0.10
COMMITTEE_GROWTH_RUIN_FRACTION: float = 0.60
COMMITTEE_GROWTH_MAX_RUIN_PROB: float = 0.01
COMMITTEE_GROWTH_HORIZON_YEARS: float = 3.0
COMMITTEE_GROWTH_N_PATHS: int = 2000
COMMITTEE_GROWTH_BARS_PER_YEAR: int = 365

SEARCH_TRIALS_ATTEMPTED: int = 70

RAM_BUDGET_FRACTION: float = 0.85
RAM_RESERVE_FRACTION: float = 0.05
RAM_RESERVE_FLOOR_BYTES: int = 256 * 2**20

WORKER_PEAK_RSS_BYTES: int = 3 * 2**30

REGISTERED_POLICY_THRESHOLDS: dict[str, float | None] = {
    "cap_30_roster": 30.0,
    "primary_annual_return": 0.05,
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
FOLD_GROWTH_CONCENTRATION_MAX_SHARE: float = 0.5

SIGNAL_EMA_HORIZON_SPAN: float = 1.0

REGIME_CASH_SCALE_FLOOR: float = 0.5
REGIME_CASH_MEDIAN_WINDOW_HOURS: int = 720
REFERENCE_PASS_EQUITY_FLOOR: float = REGIME_CASH_SCALE_FLOOR

PNL_VOL_TARGET_WINDOW_DAYS: int = 21
PNL_VOL_TARGET_SCALE_FLOOR: float = 0.2
PNL_VOL_TARGET_BURN_IN_DAYS: int = 90
PNL_VOL_TARGET_MEDIAN_WINDOW_DAYS: int = 365

FOLD_PANEL_WARMUP_HOURS: int = 720 + 168 + 24

EXECUTION_ROSTER_EXIT_MULTIPLIER: float = 2.0
REBALANCE_TRACKING_ERROR_THRESHOLD: float = 0.20

CAUSAL_BETA_LOOKBACK_BARS: int = 720
CAUSAL_BETA_MIN_PERIODS: int = 360


# Sentinel distinguishing the registered default exposure from an explicit
# committee_target_gross value: a bare MhsDiagnosticRequest() resolves to the
# registered constant without triggering the committee_capital requirement,
# while an explicit non-None value keeps requiring committee_capital=True.
COMMITTEE_TARGET_GROSS_UNSET: object = object()
