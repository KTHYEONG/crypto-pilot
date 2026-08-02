from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.research.baseline.backtest import BacktestResult
from src.research.contracts import CostModel
from src.research.expert_portfolio.allocator import _causal_lcb_weight_series, _validate_panel
from src.research.expert_portfolio.contextual_router import (
    compute_causal_contextual_winner_weights,
    compute_causal_per_symbol_contextual_weights,
)
from src.research.expert_portfolio.models import (
    ContextualRouterSpec,
    ExpertDefinition,
    ExpertPortfolioSpec,
)

_EMPTY_TRADE_COLUMNS = (
    "entry_bar",
    "exit_bar",
    "entry_price",
    "exit_price",
    "qty",
    "reason",
    "pnl",
    "return_pct",
)


@dataclass(frozen=True, slots=True)
class ExpertPortfolioBacktestResult:
    """Result of one master expert-portfolio backtest.

    ``backtest_result`` carries the single marked total-equity ledger (its
    ``trades`` frame is empty: the master ledger holds capital, it does not open
    component trades), ``target_weights`` is the per-bar causal target series,
    ``allocation_cost`` is the realised per-bar allocation-turnover cost, and
    ``component_returns`` is the completed component panel that produced the
    ledger.
    """

    backtest_result: BacktestResult
    target_weights: pd.DataFrame
    allocation_cost: pd.Series
    component_returns: pd.DataFrame


def _validate_fixed_weights(
    fixed_weights: pd.DataFrame,
    panel_index: pd.DatetimeIndex,
    weight_columns: list[str],
) -> None:
    if not fixed_weights.index.equals(panel_index):
        raise ValueError(
            "fixed_weights must be aligned to the component_returns index; a "
            "recomputed or reindexed target series is rejected"
        )
    if list(fixed_weights.columns) != weight_columns:
        raise ValueError(
            f"fixed_weights must carry exactly {weight_columns}, got {list(fixed_weights.columns)}"
        )
    values = fixed_weights.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("fixed_weights must contain only finite values")
    # A causal residual weight (e.g. CASH = 1 - sum(other weights)) can land a
    # few float64 epsilons below zero from rounding alone; tolerate that noise
    # rather than reject a genuinely zero-exposure row.
    if (values < -1e-9).any():
        raise ValueError("fixed_weights must be non-negative")


def run_expert_portfolio(
    component_returns: pd.DataFrame,
    spec: ExpertPortfolioSpec,
    costs: CostModel,
    *,
    initial_equity: float = 10_000.0,
    fixed_weights: pd.DataFrame | None = None,
    signal_delay_bars: int = 0,
    decision_context: pd.Series | None = None,
) -> ExpertPortfolioBacktestResult:
    """Execute one master expert-portfolio ledger over a completed component panel.

    The decision made at the close of bar ``t`` applies to the component return
    at bar ``t+1`` only, and the master ledger charges exactly
    ``0.5 * L1(target_t - target_{t-1}) * c_alloc`` of allocation turnover cost
    where ``c_alloc = fee_rate + slippage_rate``.     ``fixed_weights`` is reused
    verbatim for stress and can never be recalculated. Component execution and
    funding costs live inside the component returns; the extra term is solely
    the cost of moving capital between experts. A missing component return on an
    exposed position raises ``DataIntegrityError`` rather than zero-filling.

    The causal target at each decision bar is the block-aware LCB allocation
    from ``compute_causal_lcb_weights``; the batch implementation here uses the
    vectorized ``_causal_lcb_weight_series`` so one time-ordered panel is
    processed in a single pass with no per-bar DataFrame copies.  When the spec
    carries a pre-registered ``router`` the targets instead come from
    ``compute_causal_contextual_winner_weights`` and ``decision_context`` is
    required; ``fixed_weights`` always stays dominant and is reused verbatim for
    stress without any recomputation.
    """
    _validate_panel(component_returns)
    if len(component_returns) < 2:
        raise DataIntegrityError("component_returns must contain at least 2 bars")
    expert_ids = tuple(e.expert_id for e in spec.experts)
    if set(component_returns.columns) != set(expert_ids):
        raise DataIntegrityError(
            f"component_returns columns must match expert ids exactly, "
            f"got {sorted(component_returns.columns)} vs {sorted(expert_ids)}"
        )
    if initial_equity <= 0:
        raise ValueError(f"initial_equity must be > 0, got {initial_equity}")
    if signal_delay_bars < 0:
        raise ValueError(f"signal_delay_bars must be >= 0, got {signal_delay_bars}")

    weight_columns = [*list(expert_ids), "CASH"]
    if fixed_weights is not None:
        _validate_fixed_weights(fixed_weights, component_returns.index, weight_columns)
        target_weights = fixed_weights
    elif spec.router is not None:
        if decision_context is None:
            raise ValueError(
                "decision_context is required when spec.router is configured"
            )
        if spec.router_kind == "per_symbol_winner_v2":
            target_weights = compute_causal_per_symbol_contextual_weights(
                component_returns, decision_context, spec, spec.router,
            )
        else:
            target_weights = compute_causal_contextual_winner_weights(
                component_returns, decision_context, spec, spec.router,
            )
    else:
        target_weights = _causal_lcb_weight_series(component_returns, spec)

    c_alloc = costs.fee_rate + costs.slippage_rate
    n = len(component_returns)
    weights_arr = target_weights.to_numpy(dtype=np.float64)
    cash_vec = np.zeros((n, len(weight_columns)), dtype=np.float64)
    cash_vec[:, -1] = spec.gross_exposure

    source = np.arange(n) - 1 - signal_delay_bars
    applied = np.where((source >= 0)[:, None], weights_arr[np.clip(source, 0, n - 1)], cash_vec)
    prior = np.vstack([cash_vec[[0]], applied[:-1]])
    allocation_cost = 0.5 * np.abs(applied - prior).sum(axis=1) * c_alloc

    returns = component_returns[list(expert_ids)].to_numpy(dtype=np.float64)
    risky = applied[:, :-1]
    if bool((np.isnan(returns) & (risky != 0.0)).any()):
        raise DataIntegrityError(
            "component return is missing on an exposed position; missing returns "
            "are never zero-filled"
        )
    filled = np.where(np.isnan(returns), 0.0, returns)
    ledger_return = (risky * filled).sum(axis=1) - allocation_cost
    if not np.isfinite(ledger_return).all():
        raise DataIntegrityError("master ledger produced non-finite returns")

    equity = (1.0 + ledger_return).cumprod() * initial_equity
    equity = pd.Series(equity, index=component_returns.index, name="equity", dtype=np.float64)
    empty_trades = pd.DataFrame(columns=list(_EMPTY_TRADE_COLUMNS))
    result = BacktestResult(equity=equity, trades=empty_trades, signals=pd.DataFrame())
    allocation_cost_series = pd.Series(
        allocation_cost, index=component_returns.index, name="allocation_cost", dtype=np.float64,
    )
    return ExpertPortfolioBacktestResult(
        backtest_result=result,
        target_weights=target_weights,
        allocation_cost=allocation_cost_series,
        component_returns=component_returns.copy(),
    )


def _check_contract() -> None:
    """Executable assertions locking the frozen master-ledger surface."""
    assert {f.name for f in fields(ExpertPortfolioBacktestResult)} == {
        "backtest_result", "target_weights", "allocation_cost", "component_returns",
    }
    index = pd.date_range("2024-01-01", periods=2, freq="4h", tz="UTC")
    two_bar_panel = pd.DataFrame({"A": [0.01, 0.02], "B": [0.01, -0.01]}, index=index)
    spec = ExpertPortfolioSpec(experts=(
        ExpertDefinition("A", "src", "f", ("S1",), "run", "hash"),
        ExpertDefinition("B", "src", "f", ("S2",), "run", "hash2"),
    ))
    swap_weights = pd.DataFrame(
        {"A": [1.0, 0.0], "B": [0.0, 1.0], "CASH": [0.0, 0.0]}, index=index,
    )
    result = run_expert_portfolio(
        two_bar_panel, spec, CostModel(), initial_equity=10_000.0, fixed_weights=swap_weights,
    )
    assert result.allocation_cost.iloc[1] == CostModel().fee_rate + CostModel().slippage_rate
    assert result.target_weights.equals(swap_weights)
    assert run_expert_portfolio.__name__ == "run_expert_portfolio"

    routed = ExpertPortfolioSpec(
        experts=spec.experts,
        router=ContextualRouterSpec("BTCUSDT", 1, 1, 1),
    )
    context = pd.Series(["up_low_vol"] * 2, index=index)
    missing_error = ""
    try:
        run_expert_portfolio(two_bar_panel, routed, CostModel())
    except ValueError as exc:
        missing_error = str(exc)
    assert "decision_context is required" in missing_error
    routed_result = run_expert_portfolio(
        two_bar_panel, routed, CostModel(), decision_context=context,
    )
    assert list(routed_result.target_weights.columns) == ["A", "B", "CASH"]


_check_contract()
