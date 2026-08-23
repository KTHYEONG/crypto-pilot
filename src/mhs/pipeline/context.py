"""Pipeline context: shared state threaded through all MHS stages.

Each stage receives the context plus its specific inputs and returns
only what it produces. The context is the single communication channel
for long-lived state (panel, config, grids, telemetry).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.mhs.telemetry import StageTelemetry


@dataclass
class PipelineContext:
    """Shared pipeline state: panel, config, grids, and long-lived results.

    Every field is mutable (frozen=False, no slots) so stages may explicitly
    ``del ctx.<field>`` at the exact point the original monolith released that
    local variable, preserving the measured peak-RSS memory-release timing.
    """

    # Config
    config: Any  # MhsDiagnosticRequest (renamed to MhsRunConfig in P3)

    # Time bounds
    resolved_end: Any  # pd.Timestamp | None
    start: pd.Timestamp
    end: pd.Timestamp
    rss_budget_bytes: int | None
    rss_reserve_bytes: int | None

    # Panel (S0/S1)
    root: str
    grid_1h: pd.DatetimeIndex
    close: pd.DataFrame
    opens: pd.DataFrame
    quote_vol: pd.DataFrame
    taker_buy_quote: pd.DataFrame | None
    symbols: list[str]

    # Funding (S1)
    funding_by_symbol: dict[str, pd.Series] = field(default_factory=dict)
    funding_dropped: dict[str, str] | None = None
    fold_funding: dict[str, pd.Series] = field(default_factory=dict)
    funded: list[str] = field(default_factory=list)
    bar_period: Any = None  # pd.Timedelta
    funding_window: Any = None  # dict[str, pd.Series]
    bar_funding: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    aligned_symbols: list[str] = field(default_factory=list)

    # Selection (S2)
    eligible: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    log_close: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    specs: Any = None  # dict[str, BookSpec]
    fast: Any = None  # BookSpec
    slow: Any = None  # BookSpec
    fast_grid: pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([]))
    slow_grid: pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([]))
    fast_ema: Any = None
    slow_ema: Any = None
    candidate_books: Any = None  # dict[str, dict[int, pd.DataFrame]] | None
    top_level_horizon: int | None = None
    fold_slow_horizons: dict[int, int | None] = field(default_factory=dict)
    fold_fast_horizons: dict[int, tuple[int, str]] = field(default_factory=dict)
    fold_funding_carry: dict[int, tuple[int | None, int | None, str, float | None]] = field(default_factory=dict)
    _fold_committee_weights: dict[int, Any] | None = None
    _fold_growth_budget_target_vol: dict[int, float] | None = None
    # run-level 일간 참조 수익률: fold worker의 EWMA 워밍업 원천(I-WARM).
    _fold_exposure_warmup_returns: Any = None

    # Book weights (S3)
    w_fast: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    w_slow: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    w_fast_1h: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    w_slow_1h: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    w_fast_execution: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    w_slow_execution: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    execution_mask: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    regime_scale: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    blend_1h: Any = None
    committee_execution_book: Any = None
    phase_fast: Any = None
    phase_slow: Any = None
    phase_blend: Any = None
    signal_48h: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())

    # Committee (S4)
    _committee_weights_by_boundary: dict[str, dict[str, float]] = field(default_factory=dict)
    current_book_for_diagnostic: Any = None
    trend_position: Any = None

    # Replay (S6)
    execution_symbols: list[str] = field(default_factory=list)
    initial_equity: float = 1.0
    minute_grid: Any = None  # pd.DatetimeIndex | None
    has_minute_data: bool = False
    books: dict[str, Any] = field(default_factory=dict)
    blend_report: Any = None  # MhsBookReport | None
    blend_traces: Any = None
    book_reasons: tuple[str, ...] = ()

    # Folds / diagnostics (S7)
    trials_attempted: int = 0
    deflated_sharpe_ratio: Any = None
    bootstrap_ci: Any = None
    placebo_percentile: Any = None
    participation: dict[str, float] = field(default_factory=dict)
    termination_counts: dict[str, int] = field(default_factory=dict)
    unsupported: tuple[str, ...] = ()
    fold_reports: Any = None
    folds: Any = None  # tuple[MhsFoldReport, ...]
    fold_blend_parity: Any = None
    fold_growth_concentration: Any = None
    fold_realized_risk_parity: Any = None
    _pooled_fold_evidence: Any = None
    evidence_calibration: Any = None
    research_go: Any = None
    deployment: Any = None

    # Diagnostic results
    xs_ic: dict[str, float] = field(default_factory=dict)
    regression: Any = None
    horizon_diagnostics: dict[str, Any] = field(default_factory=dict)
    trend_sleeve_diagnostic: Any = None
    multi_feature_diagnostic: Any = None
    committee_diagnostic: Any = None
    committee_member_attribution: dict[str, Any] | None = None
    committee_member_books: dict[str, pd.DataFrame] | None = None
    committee_member_proxy_sharpe: dict[str, float] | None = None
    discovery_qualification: Any = None
    full_history_yearly_net_t: Any = None
    funding_carry_worst_year_corr: Any = None
    _fill_mark_parity_census: Any = None
    _growth_envelope_payload: dict[str, Any] | None = None
    blend_gross: float = 0.0
    blend_cash_fraction: float = 0.0
    realized_execution_roster_size: float | None = None

    # Run lifecycle
    run_start: float = 0.0
    _terminal_report: Any = None
    recorder: Any = None  # _StageRecorder (resource_measurements source)

    # Telemetry
    telemetry: StageTelemetry = field(default_factory=lambda: StageTelemetry(log_run=False))
