"""Execution layer: passive fills, funding panel, and the simulated inventory ledger.

The Research-GO PnL source is ``simulated_inventory_ledger`` fed by the
OHLCV strict-proxy ``strategy_aware_execution_replay``; ``mhs_ledger_pnl`` is
the pinned target-weight *pre-screen* proxy only and must never back Research
GO, OOS, capital, or capacity claims.
"""

from __future__ import annotations

from typing import Literal

from src.common.errors import DataIntegrityError
from src.mhs.types import ExecutionSpec

# Conservative extra settlement/slippage penalty applied to a stress-ledger
# (OHLCV_IMMEDIATE_TAKER) UNKNOWN_TERMINATION forced exit (spec §2.17/§7.5).
TERMINATION_STRESS_PENALTY_BPS = 50.0

# Hard ceiling for the Corwin-Schultz half-spread estimate: a degenerate
# illiquid bar sequence can otherwise produce an unbounded cost.
SPREAD_ESTIMATE_CEILING_BPS: float = 100.0

_ExecutionBound = Literal[
    "OHLCV_STRICT_PROXY",
    "OHLCV_TOUCH_PROXY",
    "OHLCV_IMMEDIATE_TAKER",
    "OHLCV_LADDERED_PROXY",
    "OHLCV_PEG_CHASE_PROXY",
]
_MarkSource = Literal["MARK_PRICE", "OHLCV_CLOSE_FALLBACK"]

_ExecutionGapCode = Literal[
    "MISSING_DECISION_MARK",
    "MISSING_ACTIVE_ORDER_OHLCV",
    "MISSING_HELD_MARK",
    "MISSING_HELD_FUNDING",
    "MISSING_FORCED_EXIT_CLOSE",
]

# Facade keeps every existing src.mhs.execution import site working.
from .accumulator import _BoundExecutionReplayAccumulator  # noqa: E402
from .batch import (  # noqa: E402  # noqa: E402  # noqa: E402
    _rescale_window_weights,
    replay_execution_window_batch,
    replay_execution_window_batch_isolated,
    replay_execution_window_pair,
    replay_execution_windows,
    replay_execution_windows_coupled,
)
from .contracts import (  # noqa: E402  # noqa: E402
    BatchReplayOutcome,
    ExecutionDataGap,
    ExecutionReplayWindow,
    ForwardExecutionObservation,
    IsolatedBoundFailure,
    SimulatedInventoryLedgerResult,
    StrategyExecutionReplayResult,
    bar_funding_panel,
    ruin_guard_equity,
)
from .ledger import simulated_inventory_ledger  # noqa: E402
from .microstructure import (  # noqa: E402  # noqa: E402
    corwin_schultz_half_spread_bps,
    laddered_fill_schedule,
    notional_weighted_shortfall_bps,  # noqa: E402
    passive_fill_shortfall_bps,  # noqa: E402
    peg_chase_fill_schedule,
    peg_chase_partial_schedule,
)
from .pnl import _column_order_row_sum, mhs_ledger_pnl, mhs_ledger_pnl_multi_tier  # noqa: E402
from .strategy_replay import strategy_aware_execution_replay  # noqa: E402

__all__ = [
    "SPREAD_ESTIMATE_CEILING_BPS",
    "TERMINATION_STRESS_PENALTY_BPS",
    "BatchReplayOutcome",
    "DataIntegrityError",
    "ExecutionDataGap",
    "ExecutionReplayWindow",
    "ExecutionSpec",
    "ForwardExecutionObservation",
    "IsolatedBoundFailure",
    "SimulatedInventoryLedgerResult",
    "StrategyExecutionReplayResult",
    "_BoundExecutionReplayAccumulator",
    "_ExecutionBound",
    "_ExecutionGapCode",
    "_MarkSource",
    "_column_order_row_sum",
    "_rescale_window_weights",
    "bar_funding_panel",
    "corwin_schultz_half_spread_bps",
    "laddered_fill_schedule",
    "mhs_ledger_pnl",
    "mhs_ledger_pnl_multi_tier",
    "notional_weighted_shortfall_bps",
    "passive_fill_shortfall_bps",
    "peg_chase_fill_schedule",
    "peg_chase_partial_schedule",
    "replay_execution_window_batch",
    "replay_execution_window_batch_isolated",
    "replay_execution_window_pair",
    "replay_execution_windows",
    "replay_execution_windows_coupled",
    "ruin_guard_equity",
    "simulated_inventory_ledger",
    "strategy_aware_execution_replay",
]
