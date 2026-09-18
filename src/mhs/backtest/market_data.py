"""Causal process market preparation without retained expired buffers."""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.common.errors import DataIntegrityError
from src.common.paths import FUTURES_DATA_DIR
from src.market_data.services.mhs_execution import (
    apply_dynamic_gap_exclusion,
    apply_dynamic_mark_gap_exclusion,
)
from src.mhs.backtest.contracts import ProcessMarketData
from src.mhs.books import rank_weight_book
from src.mhs.data_policy import MHS_DATA_POLICY_DEFAULT, SOURCE_GAP_EXCLUDED_SYMBOLS
from src.mhs.execution.contracts import bar_funding_panel
from src.mhs.features import FEATURE_REGISTRY
from src.mhs.funding import funding_carry_signal
from src.mhs.marks import _load_funding_series, _pit_execution_mask
from src.mhs.panel import liquid_half_eligibility, load_base_panel
from src.mhs.params import (
    CAUSAL_BETA_LOOKBACK_BARS,
    CAUSAL_BETA_MIN_PERIODS,
    CLI_EXECUTION_UNIVERSE_SIZE_DEFAULT,
    PANEL_MIN_HISTORY_BARS,
    PROCESS_FEATURE_CANDIDATES,
    PROCESS_FUNDING_CARRY_CANDIDATES_HOURS,
    PROCESS_MIN_SYMBOLS,
)
from src.mhs.process_features import PROCESS_FEATURE_COLUMN_BLOCK_SIZE, build_process_feature_grid
from src.mhs.regime import beta_neutralize_weights, causal_market_beta
from src.mhs.resources import (
    MhsMemoryBudget,
    _current_tree_swap_bytes,
    assert_mhs_stage_allocation,
    resolve_mhs_memory_budget,
)

_logger = logging.getLogger(__name__)

def build_candidate_member_books(
    panels: Mapping[str, pd.DataFrame],
    bar_funding_1h: pd.DataFrame,
    eligible_1h: pd.DataFrame,
    execution_mask_1h: pd.DataFrame,
    decision_grid: pd.DatetimeIndex,
) -> dict[str, pd.DataFrame]:
    """Build unchanged beta-neutral process books only at decision labels.

    Coverage over the complete evaluation period must not select early members;
    candidates retain the original declared order and fail-closed projection.

    Args:
        panels: Canonically aligned hourly source planes.
        bar_funding_1h: Existing causally aligned hourly funding rates.
        eligible_1h: Existing liquid-universe eligibility for market beta.
        execution_mask_1h: Existing causal execution eligibility.
        decision_grid: UTC process decision labels.
    Returns:
        All declared candidate books in original order, with original decision
        weights, float64 precision and canonical symbol labels.
    Raises:
        ValueError: Existing alignment or feature contracts are invalid.
        DataIntegrityError: Existing input integrity checks fail.
    """
    registry = {spec.name: spec for spec in FEATURE_REGISTRY}
    beta_1h = causal_market_beta(
        np.log(panels["close"]), eligible_1h, CAUSAL_BETA_LOOKBACK_BARS, CAUSAL_BETA_MIN_PERIODS
    )
    beta_grid = beta_1h.reindex(decision_grid)
    del beta_1h
    mask_grid = execution_mask_1h.reindex(decision_grid).fillna(False)
    out: dict[str, pd.DataFrame] = {}
    for name in PROCESS_FEATURE_CANDIDATES:
        spec = registry[name]
        _logger.info("[DATA] stage=candidate_progress candidate=%s", name)
        feature_grid = build_process_feature_grid(spec, panels, decision_grid)
        grid_book = rank_weight_book(feature_grid, mask_grid, 1, PROCESS_MIN_SYMBOLS)
        del feature_grid
        out[name] = beta_neutralize_weights(grid_book, beta_grid, mask_grid, PROCESS_MIN_SYMBOLS)
        del grid_book
    for lookback in PROCESS_FUNDING_CARRY_CANDIDATES_HOURS:
        key = f"funding_carry_{lookback}h"
        _logger.info("[DATA] stage=candidate_progress candidate=%s", key)
        carry_parts = [
            funding_carry_signal(
                bar_funding_1h.iloc[:, left : left + PROCESS_FEATURE_COLUMN_BLOCK_SIZE], lookback
            ).reindex(decision_grid)
            for left in range(0, bar_funding_1h.shape[1], PROCESS_FEATURE_COLUMN_BLOCK_SIZE)
        ]
        signal_grid = pd.concat(carry_parts, axis=1).reindex(columns=list(bar_funding_1h.columns))
        del carry_parts
        grid_book = rank_weight_book(signal_grid, mask_grid, -1, PROCESS_MIN_SYMBOLS)
        del signal_grid
        out[key] = beta_neutralize_weights(grid_book, beta_grid, mask_grid, PROCESS_MIN_SYMBOLS)
        del grid_book
    return out


def _process_memory_budget(budget: MhsMemoryBudget | None) -> MhsMemoryBudget:
    return resolve_mhs_memory_budget(budget)


def _admit_process_stage(
    *, stage: str, estimated_bytes: int, budget: MhsMemoryBudget,
    replay: bool, initial_swap_bytes: int | None,
) -> None:
    assert_mhs_stage_allocation(
        stage=stage, estimated_bytes=int(estimated_bytes),
        budget=budget, replay=replay, initial_swap_bytes=initial_swap_bytes,
    )


def _estimate_panel_bytes(n_bars: int, n_symbols: int, n_planes: int = 6) -> int:
    return max(int(n_bars) * max(int(n_symbols), 0) * max(int(n_planes), 0) * 8 * 2, 0)


def load_process_market_data(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    data_root: str | None = None,
    memory_budget: MhsMemoryBudget | None = None,
) -> ProcessMarketData:
    """Prepare original process inputs without retaining expired wide buffers.

    Resource limits protect the full requested history rather than changing
    its dates, precision, candidates or source coverage.

    Args:
        start: Timezone-aware source start.
        end: Timezone-aware source end within the registered ceiling.
        data_root: Existing OHLCV override, not a funding or mark override.
        memory_budget: Explicit stage limits or the validated defaults.
    Returns:
        Original hourly execution inputs and unchanged decision-grid books.
    Raises:
        RuntimeError: No development symbol has aligned funding.
        DataIntegrityError: Source provenance or resource admission fails.
    """
    budget = _process_memory_budget(memory_budget)
    initial_swap_bytes = _current_tree_swap_bytes()
    _admit_process_stage(
        stage="process_prepare_panel", estimated_bytes=0,
        budget=budget, replay=False, initial_swap_bytes=initial_swap_bytes,
    )
    root = data_root or str(FUTURES_DATA_DIR / "ohlcv")

    def admit_panel(estimated_bytes: int) -> None:
        _admit_process_stage(
            stage="process_prepare_panel", estimated_bytes=estimated_bytes,
            budget=budget, replay=False, initial_swap_bytes=initial_swap_bytes,
        )

    panel = load_base_panel(
        root, "1h",
        ("close", "open", "high", "low", "quote_vol", "taker_buy_quote"),
        start, end, partition="dev", min_bars=PANEL_MIN_HISTORY_BARS,
        data_policy=MHS_DATA_POLICY_DEFAULT,
        allocation_admission=admit_panel,
    )
    close, opens, quote_vol = panel["close"], panel["open"], panel["quote_vol"]
    grid_1h = close.index
    _logger.info("[DATA] stage=base_1h_panel bars=%d symbols=%d", len(grid_1h), len(close.columns))
    funding_by_symbol, _dropped = _load_funding_series(list(close.columns))
    funded = [
        s for s in close.columns
        if s in funding_by_symbol and s not in SOURCE_GAP_EXCLUDED_SYMBOLS
    ]
    if not funded:
        raise RuntimeError("no dev symbol has funding coverage; the MHS ledger requires funding")
    close = close[funded]
    opens = opens[funded]
    quote_vol = quote_vol[funded]
    bar_period = grid_1h[1] - grid_1h[0]
    funding_window = {
        s: funding_by_symbol[s].loc[
            (funding_by_symbol[s].index >= grid_1h[0])
            & (funding_by_symbol[s].index < grid_1h[-1] + bar_period)
        ]
        for s in funded
    }
    bar_funding = bar_funding_panel(funding_window, grid_1h)
    aligned = list(bar_funding.columns)
    if not aligned:
        raise RuntimeError("no dev symbol has causally aligned funding coverage")
    close = close[aligned]
    opens = opens[aligned]
    quote_vol = quote_vol[aligned]
    _logger.info("[DATA] stage=funding_alignment bars=%d symbols=%d", len(grid_1h), len(aligned))
    eligible = liquid_half_eligibility(quote_vol, PANEL_MIN_HISTORY_BARS, PANEL_MIN_HISTORY_BARS)
    mask = _pit_execution_mask(quote_vol, eligible, CLI_EXECUTION_UNIVERSE_SIZE_DEFAULT)
    decision_grid = pd.date_range(start, end, freq="24h", tz="UTC")
    _admit_process_stage(
        stage="process_funding_prefix", estimated_bytes=_estimate_panel_bytes(len(grid_1h), len(aligned), 2),
        budget=budget, replay=False, initial_swap_bytes=initial_swap_bytes,
    )
    log_close_step = np.log(close).reindex(decision_grid)
    # (t, t+24h] 구간 합을 누적합 차분으로 계산한다(결정일마다 전체 스캔 금지).
    day = pd.Timedelta(hours=24)
    prefix = np.vstack([np.zeros((1, len(aligned))), np.cumsum(bar_funding.to_numpy(dtype="float64"), axis=0)])
    left = np.searchsorted(bar_funding.index.to_numpy(), decision_grid.to_numpy(), side="right")
    right = np.searchsorted(bar_funding.index.to_numpy(), (decision_grid + day).to_numpy(), side="right")
    funding_step = pd.DataFrame(prefix[right] - prefix[left], index=decision_grid, columns=aligned)
    _logger.info("[DATA] stage=decision_grid days=%d symbols=%d", len(decision_grid), len(aligned))
    panels: dict[str, pd.DataFrame] = {k: panel[k][aligned] for k in ("close", "open", "high", "low", "quote_vol", "taker_buy_quote")}
    del panel, close, quote_vol, prefix, funding_by_symbol, funding_window
    causal_mask, _ = apply_dynamic_gap_exclusion(mask, "1h", root=root)
    causal_mask, _ = apply_dynamic_mark_gap_exclusion(causal_mask)
    causal_mask, _ = apply_dynamic_gap_exclusion(causal_mask, "3m", root=root)
    n_candidates = len(PROCESS_FEATURE_CANDIDATES) + len(PROCESS_FUNDING_CARRY_CANDIDATES_HOURS)
    _admit_process_stage(
        stage="process_candidate_books", estimated_bytes=_estimate_panel_bytes(len(decision_grid), len(aligned), n_candidates),
        budget=budget, replay=False, initial_swap_bytes=initial_swap_bytes,
    )
    member_books = build_candidate_member_books(panels, bar_funding, eligible, causal_mask, decision_grid)
    del panels
    _logger.info("[DATA] stage=member_books candidates=%d", len(member_books))
    book_columns = list(next(iter(member_books.values())).columns)
    _admit_process_stage(
        stage="process_hourly_ledger", estimated_bytes=_estimate_panel_bytes(len(decision_grid), len(book_columns), 2),
        budget=budget, replay=False, initial_swap_bytes=initial_swap_bytes,
    )
    execution_mask = causal_mask.reindex(decision_grid, fill_value=False)[book_columns]
    return ProcessMarketData(
        grid_1h=grid_1h,
        decision_grid=decision_grid,
        opens_1h=opens,
        bar_funding_1h=bar_funding,
        log_close_step=log_close_step,
        funding_step=funding_step,
        member_books=member_books,
        execution_mask=execution_mask,
    )


def apply_process_execution_availability(target_weights: pd.DataFrame, execution_mask: pd.DataFrame) -> pd.DataFrame:
    """Prevent unavailable targets from being revived by smoothing or adoption.

    Args:
        target_weights: Smoothed and adopted canonical decision targets.
        execution_mask: Exactly aligned causal execution eligibility.

    Returns:
        Targets with unavailable cells exactly zero and available cells unchanged.

    Raises:
        DataIntegrityError: Labels, symbols or mask values are inconsistent.
    """
    if not target_weights.index.equals(execution_mask.index):
        raise DataIntegrityError("execution_mask must share target_weights decision labels exactly")
    if list(target_weights.columns) != list(execution_mask.columns):
        raise DataIntegrityError("execution_mask must share target_weights symbol columns exactly")
    if bool((execution_mask.dtypes.apply(lambda dt: dt.kind != "b")).any()):
        raise DataIntegrityError("execution_mask must be boolean")
    if bool(execution_mask.isna().to_numpy().any()):
        raise DataIntegrityError("execution_mask must not be missing")
    return target_weights.where(execution_mask, other=0.0)
