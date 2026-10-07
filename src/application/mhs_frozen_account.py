"""Frozen-book account replay and growth-exposure derivation services (research only, no live artifact)."""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from src.backtests.catalog import append_backtest_index
from src.common.paths import FUTURES_DATA_DIR
from src.mhs.params import (
    ACCOUNT_DEFAULT_CAPITAL_USDT,
    ACCOUNT_IMPACT_Y,
    ACCOUNT_MAKER_FEE_BPS,
    ACCOUNT_PASSIVE_WINDOW_BARS,
    ACCOUNT_RECON_CAGR_TOLERANCE,
    ACCOUNT_RECON_MDD_TOLERANCE,
    ACCOUNT_TAKER_FEE_BPS,
    ACCOUNT_UNIT_REFERENCE_CAPITAL,
    COMMITTEE_GROWTH_HORIZON_YEARS,
    COMMITTEE_GROWTH_N_PATHS,
    FROZEN_EXPOSURE_GAP_THRESHOLD,
    FROZEN_EXPOSURE_GRID,
    FROZEN_EXPOSURE_MEAN_HAIRCUT,
    FROZEN_EXPOSURE_PLATEAU_TOLERANCE,
    FROZEN_EXPOSURE_SEED,
    NULL_BOOTSTRAP_MEAN_BLOCK_DAYS,
    SETTLEMENT_PRICE_STRESS_HAIRCUT_BPS,
)
from src.mhs.resources import MhsMemoryBudget

if TYPE_CHECKING:
    from src.mhs.account_ledger import AccountLedgerResult
    from src.mhs.frozen_research_candidate import FrozenMhsStrategySpec
    from src.mhs.types import ExecutionSpec

_logger = logging.getLogger(__name__)


class FrozenAccountError(Exception):
    """Operator-facing failure of a frozen account or exposure run.

    `str(exc)` is the complete operator message (the CLI re-raises it verbatim as
    `SystemExit`). The originating exception is chained as `__cause__`. Distinct from
    `ValueError` so request-validation errors and run failures stay separable and no
    upstream `except ValueError` can swallow a failed run.
    """


def frozen_execution_specs() -> tuple[ExecutionSpec, ExecutionSpec]:
    """Registered six/eighteen-basis-point base/stress cost pair with the submit-bar anchor.

    Every order is sized and priced from the last mark published before it is sent, so both
    cases use `decision_anchor="submit_bar"`. Shared by the frozen research command and the
    frozen account replay so both price the same book identically.

    Returns:
        `(base, stress)` where base is 5 bps taker fee + 1 bp slippage and stress widens
        slippage to 13 bps.
    """
    from src.mhs.types import ExecutionSpec

    base = dataclasses.replace(ExecutionSpec(), taker_fee_bps=5.0, taker_slippage_bps=1.0, decision_anchor="submit_bar")
    stress = dataclasses.replace(
        base, taker_slippage_bps=13.0, settlement_price_haircut_bps=SETTLEMENT_PRICE_STRESS_HAIRCUT_BPS
    )
    return base, stress


@dataclass(frozen=True, slots=True)
class FrozenAccountRequest:
    """Typed controls for one frozen-book account replay.

    `runs_root` and `venue_rules_root` are explicit so the caller (CLI) owns the process-wide
    path constants and tests can redirect them; the catalog read for reconciliation and the
    catalog append both resolve to `runs_root.parent.parent / "index.jsonl"`.
    """

    source_start: pd.Timestamp
    evaluation_start: pd.Timestamp
    evaluation_end: pd.Timestamp
    runs_root: Path
    venue_rules_root: Path
    policy: Literal["growth", "fixed"] = "growth"
    execution: Literal["taker", "maker"] = "taker"
    capital: float = ACCOUNT_DEFAULT_CAPITAL_USDT
    impact_y: float = ACCOUNT_IMPACT_Y
    fixed_exposure: float | None = None
    apply_order_filters: bool = True
    venue_rules: Path | None = None
    data_root: Path | None = None
    memory_budget: MhsMemoryBudget | None = None
    export_unit_returns: Path | None = None


@dataclass(frozen=True, slots=True)
class FrozenAccountReport:
    """Persisted outcome of one account replay.

    `payload` is the exact mapping serialized to `account.json`; `result` is the account
    ledger outcome (not the unit reference) for caller-side summary logging.
    """

    run_dir: Path
    payload: dict[str, Any]
    result: AccountLedgerResult


def validate_frozen_account_request(request: FrozenAccountRequest) -> None:
    """Validate account controls before any I/O.

    Args:
        request: Typed account controls.
    Returns:
        None when the request satisfies the run contract.
    Raises:
        ValueError: Non-request input, naive or non-ordered timestamps (require
            source_start < evaluation_start < evaluation_end, all UTC-aware), unknown policy or
            execution, `policy == "fixed"` without `fixed_exposure`, non-finite or non-positive
            `fixed_exposure` or `capital`, or a non-Path `runs_root` / `venue_rules_root`.
    """
    if not isinstance(request, FrozenAccountRequest):
        raise ValueError(f"request must be FrozenAccountRequest, got {request!r}")
    for label in ("source_start", "evaluation_start", "evaluation_end"):
        value = getattr(request, label)
        if not isinstance(value, pd.Timestamp) or value.tzinfo is None:
            raise ValueError(f"{label} must be a timezone-aware Timestamp, got {value!r}")
    source = request.source_start.tz_convert("UTC")
    start = request.evaluation_start.tz_convert("UTC")
    end = request.evaluation_end.tz_convert("UTC")
    if not source < start < end:
        raise ValueError(f"require source_start < evaluation_start < evaluation_end, got {source} {start} {end}")
    if request.policy not in ("growth", "fixed"):
        raise ValueError(f"policy must be 'growth' or 'fixed', got {request.policy!r}")
    if request.execution not in ("taker", "maker"):
        raise ValueError(f"execution must be 'taker' or 'maker', got {request.execution!r}")
    if request.policy == "fixed" and request.fixed_exposure is None:
        raise ValueError("fixed_exposure is required with policy 'fixed'")
    if request.fixed_exposure is not None:
        exposure = request.fixed_exposure
        if isinstance(exposure, bool) or not isinstance(exposure, (int, float)) or not math.isfinite(float(exposure)) or not float(exposure) > 0:
            raise ValueError(f"fixed_exposure must be a positive finite exposure, got {exposure!r}")
    capital = request.capital
    if isinstance(capital, bool) or not isinstance(capital, (int, float)) or not math.isfinite(float(capital)) or not float(capital) > 0:
        raise ValueError(f"capital must be a positive finite capital, got {capital!r}")
    for label in ("runs_root", "venue_rules_root"):
        value = getattr(request, label)
        if not isinstance(value, Path):
            raise ValueError(f"{label} must be a Path, got {value!r}")


def reconcile_unit_reference(
    reference: dict[str, Any] | None,
    *,
    unit_cagr: float,
    unit_mdd: float,
) -> dict[str, Any]:
    """Compare the unit-exposure reference ledger against the latest same-book canonical row.

    The unit ledger (exposure 1, no order filters, no impact, reference capital) replays the
    same book as the canonical 3m frozen run, so its CAGR and 3m-close drawdown must agree
    within the registered tolerances. Disagreement is disclosed, never fatal: the account run
    is research evidence and the reference may legitimately be absent.

    Args:
        reference: Catalog row returned by the same-book lookup (with `name_clip` and
            `exposure_multiplier` merged from its result), or None when no same-book run exists.
        unit_cagr: Unit ledger CAGR over the evaluation window.
        unit_mdd: Unit ledger intraday (3m close path) max drawdown magnitude, >= 0.
    Returns:
        Reconciliation record with status `missing_reference` (reference None), `ok` (both gaps
        present and within `ACCOUNT_RECON_CAGR_TOLERANCE` / `ACCOUNT_RECON_MDD_TOLERANCE`) or
        `mismatch` (either gap missing or outside tolerance).
    Raises:
        ValueError, TypeError, OverflowError: `base_cagr` or `base_max_drawdown` is present but
            not convertible to float.
    """
    base_record: dict[str, Any] = {
        "fixed_exposure": 1.0,
        "capital": ACCOUNT_UNIT_REFERENCE_CAPITAL,
        "order_filters": False,
        "impact_y": 0.0,
        "cagr": unit_cagr,
        "mdd": unit_mdd,
        "mdd_convention": "magnitude",
        "mdd_definition": "3m_close_path",
        "cagr_tolerance": ACCOUNT_RECON_CAGR_TOLERANCE,
        "mdd_tolerance": ACCOUNT_RECON_MDD_TOLERANCE,
    }
    if reference is None:
        return {**base_record, "status": "missing_reference", "reference_canonical": None, "cagr_gap": None, "mdd_gap": None}
    base_cagr = reference.get("base_cagr")
    base_mdd = reference.get("base_max_drawdown")
    cagr_gap = None if base_cagr is None else unit_cagr - float(base_cagr)
    mdd_gap = None if base_mdd is None else unit_mdd - abs(float(base_mdd))
    status = "ok"
    if cagr_gap is None or abs(cagr_gap) > ACCOUNT_RECON_CAGR_TOLERANCE or mdd_gap is None or abs(mdd_gap) > ACCOUNT_RECON_MDD_TOLERANCE:
        status = "mismatch"
    return {
        **base_record,
        "status": status,
        "reference_canonical": {
            "strategy_id": reference.get("strategy_id"),
            "run_dir": reference.get("run_dir"),
            "evaluation_start": reference.get("evaluation_start"),
            "evaluation_end": reference.get("evaluation_end"),
            "base_cagr": reference.get("base_cagr"),
            "base_max_drawdown": reference.get("base_max_drawdown"),
            "name_clip": reference.get("name_clip"),
            "exposure_multiplier": reference.get("exposure_multiplier"),
        },
        "cagr_gap": cagr_gap,
        "mdd_gap": mdd_gap,
    }


def _account_headlines(equity: pd.Series, capital: float) -> tuple[float, float, float]:
    """CAGR, daily max drawdown (negative or zero), and final equity for one account path."""
    values = equity.to_numpy(dtype="float64")
    final = float(values[-1])
    years = len(values) / 365.0
    cagr = float(final / capital) ** (1.0 / years) - 1.0 if final > 0 else -1.0
    running = np.maximum.accumulate(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(running > 0, values / running, 1.0)
    return cagr, float((relative - 1.0).min()), final


def _latest_same_book_reference(
    index_path: Path,
    *,
    execution: str,
    strategy: FrozenMhsStrategySpec,
    evaluation_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
) -> dict[str, Any] | None:
    """Latest canonical ``mhs_frozen`` run of the exact book the account ledger replays.

    A reference must share the strategy id, execution mode, evaluation window, name clip and
    exposure multiplier; the last two live only in the run's ``result.json`` (resolved against
    the catalog directory), so rows whose result is missing or unreadable are skipped. Returns
    the catalog row, or None when no run of the same book exists.
    """
    if not index_path.is_file():
        return None
    reference: dict[str, Any] | None = None
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError("catalog row is not a JSON object")
        if record.get("kind") != "mhs_frozen":
            continue
        if record.get("strategy_id") != strategy.strategy_id:
            continue
        if record.get("execution", "taker") != execution:
            continue
        try:
            row_start = pd.Timestamp(record["evaluation_start"]).tz_convert("UTC")
            row_end = pd.Timestamp(record["evaluation_end"]).tz_convert("UTC")
        except (KeyError, ValueError, TypeError):
            continue
        if row_start != evaluation_start or row_end != evaluation_end:
            continue
        run_dir = record.get("run_dir")
        if not isinstance(run_dir, str):
            continue
        try:
            payload = json.loads((index_path.parent / run_dir / "result.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            raise ValueError("reference result is not a JSON object")
        if payload.get("name_clip") != strategy.name_clip:
            continue
        if payload.get("exposure_multiplier") != strategy.exposure_multiplier:
            continue
        reference = {**record, "name_clip": payload.get("name_clip"), "exposure_multiplier": payload.get("exposure_multiplier")}
    return reference


def _resolve_account_destination(*, runs_root: Path, start: pd.Timestamp, end: pd.Timestamp, policy: str, capital: float, execution: str = "taker") -> Path:
    """Resolve a fresh account run directory named by window, policy, and capital."""
    stem = f"{start:%Y%m%d}_{end:%Y%m%d}_top20_account_{policy}_{capital:.0f}"
    stamp = f"{pd.Timestamp.now(tz='UTC'):%Y%m%dT%H%M%S}Z"
    name = f"{stem}_{stamp}" if execution == "taker" else f"{stem}_maker_{stamp}"
    run_dir = runs_root / name
    suffix = 1
    while os.path.lexists(run_dir):
        suffix += 1
        run_dir = runs_root / f"{name}-{suffix}"
    run_dir.mkdir(parents=True)
    return run_dir


def _export_unit_returns(
    *, equity: pd.Series, strategy_id: str, execution: str,
    start: pd.Timestamp, end: pd.Timestamp, run_dir: Path, dest: Path,
) -> None:
    """Write the unit reference ledger daily returns with run-identity metadata."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    returns = equity.pct_change().iloc[1:]
    returns.index.name = "entry_day"
    returns.name = "unit_return"
    table = pa.Table.from_pandas(returns.to_frame(), preserve_index=True)
    metadata = {
        "strategy_id": strategy_id,
        "execution": execution,
        "evaluation_start": start.isoformat(),
        "evaluation_end": end.isoformat(),
        "run_dir": str(run_dir),
    }
    merged = dict(table.schema.metadata or {})
    merged.update({key: str(value) for key, value in metadata.items()})
    table = table.replace_schema_metadata(merged)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, dest)


def _account_execution_kwargs(execution: str) -> dict[str, Any]:
    """Maker/taker execution kwargs shared by the unit and account replays."""
    if execution == "maker":
        return {
            "execution": "maker",
            "maker_fee_bps": ACCOUNT_MAKER_FEE_BPS,
            "passive_window_bars": ACCOUNT_PASSIVE_WINDOW_BARS,
        }
    return {"execution": "taker"}


def _build_account_payload(
    *, request: FrozenAccountRequest, strategy_id: str, policy: Any, result: AccountLedgerResult,
    unit: AccountLedgerResult, unit_headlines: tuple[float, float, float],
    account_headlines: tuple[float, float, float],
    rules_captured_at: str, venue_path: str, reconciliation: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the exact `account.json` mapping for one account replay."""
    unit_cagr, unit_daily_mdd, _ = unit_headlines
    cagr, daily_mdd, final_equity = account_headlines
    mdd = -result.intraday_max_drawdown
    moment_source = "bayesian_causal_unit_ledger" if request.policy == "growth" else "none"
    exposures = result.daily_exposure.to_numpy(dtype="float64")
    return {
        "strategy_id": strategy_id,
        "capital": request.capital,
        "execution": {
            "mode": request.execution,
            "maker_fee_bps": None if request.execution == "taker" else ACCOUNT_MAKER_FEE_BPS,
            "taker_fee_bps": ACCOUNT_TAKER_FEE_BPS,
            "passive_window_bars": None if request.execution == "taker" else ACCOUNT_PASSIVE_WINDOW_BARS,
            "maker_fill_fraction": result.maker_fill_fraction,
        },
        "policy": {
            "kind": policy.kind,
            "exposure_max": policy.exposure_max,
            "exposure_step": policy.exposure_step,
            "mean_haircut": policy.mean_haircut,
            "prior_days": policy.prior_days,
            "min_moment_days": policy.min_moment_days,
            "shock_per_unit": policy.shock_per_unit,
            "margin_reserve": policy.margin_reserve,
            "initial_margin_cap": policy.initial_margin_cap,
            "impact_y": policy.impact_y,
        },
        "venue_captured_at": rules_captured_at,
        "venue_path": venue_path,
        "evaluation_start": request.evaluation_start.isoformat(),
        "evaluation_end": request.evaluation_end.isoformat(),
        "cagr": cagr,
        "mdd": mdd,
        "daily_mdd": daily_mdd,
        "final_equity": final_equity,
        "liquidated_at": None if result.liquidated_at is None else result.liquidated_at.isoformat(),
        "mean_exposure": float(exposures.mean()),
        "min_exposure": float(exposures.min()),
        "last_exposure": float(exposures[-1]),
        "skipped_orders": result.skipped_orders,
        "untraded_fraction": result.untraded_fraction,
        "initial_margin_breaches": result.initial_margin_breaches,
        "fee_paid": result.fee_paid,
        "impact_paid": result.impact_paid,
        "funding_paid": result.funding_paid,
        "fallback_ladder_symbols": list(result.fallback_ladder_symbols),
        "missing_filter_symbols": list(result.missing_filter_symbols),
        "moment_source": moment_source,
        "entry_anchor": "submit_bar",
        "unit_reference": {
            "capital": ACCOUNT_UNIT_REFERENCE_CAPITAL,
            "cagr": unit_cagr,
            "mdd": -unit.intraday_max_drawdown,
            "daily_mdd": unit_daily_mdd,
            "maker_fill_fraction": unit.maker_fill_fraction,
        },
        "venue_rules_applied_retroactively": True,
        "reconciliation": reconciliation,
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }


def _resolve_account_venue(request: FrozenAccountRequest) -> tuple[Any, Path]:
    """Resolve the venue snapshot for one account replay."""
    from src.common.errors import DataIntegrityError
    from src.market_data.binance.venue_rules import latest_venue_rule_snapshot, load_venue_rule_snapshot

    try:
        venue_path = Path(request.venue_rules) if request.venue_rules is not None else latest_venue_rule_snapshot(request.venue_rules_root)
        rules = load_venue_rule_snapshot(venue_path)
    except (DataIntegrityError, FileNotFoundError, NotADirectoryError, OSError, ValueError) as exc:
        raise FrozenAccountError(f"missing venue snapshot ({exc}); run data collect venue-rules first") from exc
    return rules, venue_path


def _build_account_candidate(request: FrozenAccountRequest) -> tuple[Any, Any, Any]:
    """Build the unlevered account-unit candidate and its assembled replay inputs."""
    from src.common.errors import DataIntegrityError
    from src.mhs.account_sources import assemble_account_inputs
    from src.mhs.frozen_research_candidate import FROZEN_MHS_TOP20_ACCOUNT_UNIT_V2
    from src.mhs.frozen_research_evidence import FrozenMhsReportPeriod
    from src.mhs.frozen_research_run import FrozenMhsBacktestRequest, build_frozen_request_candidate

    base_spec, stress_spec = frozen_execution_specs()
    try:
        frozen_request = FrozenMhsBacktestRequest(
            source_start=request.source_start, evaluation_start=request.evaluation_start,
            evaluation_end=request.evaluation_end, strategy=FROZEN_MHS_TOP20_ACCOUNT_UNIT_V2,
            initial_equity=request.capital, base_spec=base_spec, stress_spec=stress_spec,
            report_periods=(
                FrozenMhsReportPeriod(
                    label="evaluation",
                    start=request.evaluation_start.normalize(),
                    end=(request.evaluation_end - pd.Timedelta(days=1)).normalize(),
                ),
            ),
            data_root=request.data_root, memory_budget=request.memory_budget,
            execution_bound="OHLCV_IMMEDIATE_TAKER",
        )
    except (DataIntegrityError, ValueError) as exc:
        raise FrozenAccountError(f"invalid frozen account request: {exc}") from exc
    try:
        candidate, context = build_frozen_request_candidate(frozen_request)
        parts = assemble_account_inputs(candidate, context)
    except (DataIntegrityError, ValueError, OSError) as exc:
        raise FrozenAccountError(f"frozen account failed: {exc}") from exc
    return frozen_request.strategy, parts, frozen_request


def _replay_account_ledgers(request: FrozenAccountRequest, parts: Any, rules: Any, policy: Any) -> tuple[Any, Any]:
    """Replay the unit reference ledger then the account ledger."""
    from src.common.errors import DataIntegrityError
    from src.mhs.account_ledger import replay_account

    unit_weights, marks, funding_cum, adv, daily_sigma, anchor_times = parts
    execution_kwargs = _account_execution_kwargs(request.execution)
    try:
        unit_policy = dataclasses.replace(policy, kind="fixed", exposure_max=1.0, impact_y=0.0)
        unit = replay_account(
            unit_weights, marks, funding_cum, adv, daily_sigma, rules, unit_policy,
            anchor_times=anchor_times, capital=ACCOUNT_UNIT_REFERENCE_CAPITAL,
            taker_fee_bps=ACCOUNT_TAKER_FEE_BPS, apply_order_filters=False, **execution_kwargs,
        )
    except (DataIntegrityError, ValueError) as exc:
        raise FrozenAccountError(f"frozen account failed: unit reference {exc}") from exc
    if unit.liquidated_at is not None:
        raise FrozenAccountError(f"frozen account failed: unit reference liquidated at {unit.liquidated_at}")
    try:
        result = replay_account(
            unit_weights, marks, funding_cum, adv, daily_sigma, rules, policy,
            anchor_times=anchor_times, capital=request.capital,
            taker_fee_bps=ACCOUNT_TAKER_FEE_BPS, apply_order_filters=request.apply_order_filters,
            unit_equity=unit.daily_equity, **execution_kwargs,
        )
    except (DataIntegrityError, ValueError) as exc:
        raise FrozenAccountError(f"frozen account failed: {exc}") from exc
    return unit, result


def _reconcile_account_unit(request: FrozenAccountRequest, strategy: Any, unit: Any) -> dict[str, Any]:
    """Reconcile the unit ledger against the latest same-book canonical row."""
    unit_cagr, _, _ = _account_headlines(unit.daily_equity, ACCOUNT_UNIT_REFERENCE_CAPITAL)
    unit_mdd = unit.intraday_max_drawdown
    try:
        index_path = request.runs_root.parent.parent / "index.jsonl"
        reference = _latest_same_book_reference(
            index_path, execution=request.execution, strategy=strategy,
            evaluation_start=request.evaluation_start, evaluation_end=request.evaluation_end,
        )
        reconciliation = reconcile_unit_reference(reference, unit_cagr=unit_cagr, unit_mdd=unit_mdd)
        if reconciliation["status"] == "missing_reference":
            _logger.warning(
                "[EVAL] mhs-frozen-account reconciliation status=%s cagr_gap=%s mdd_gap=%s",
                "missing_reference", None, None,
            )
        elif reconciliation["status"] == "mismatch":
            cagr_gap = reconciliation["cagr_gap"]
            mdd_gap = reconciliation["mdd_gap"]
            _logger.warning(
                "[EVAL] mhs-frozen-account reconciliation status=%s cagr_gap=%.4f mdd_gap=%.4f",
                "mismatch",
                float(cagr_gap) if cagr_gap is not None else float("nan"),
                float(mdd_gap) if mdd_gap is not None else float("nan"),
            )
        return reconciliation
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        _logger.warning(
            "[EVAL] mhs-frozen-account reconciliation status=failed error_type=%s error=%s",
            type(exc).__name__, exc,
        )
        return {"status": "failed", "error": str(exc), "error_type": type(exc).__name__}


def _persist_account_run(request: FrozenAccountRequest, strategy: Any, policy: Any, unit: Any, result: Any, reconciliation: dict[str, Any], *, rules_captured_at: str, venue_path: str) -> FrozenAccountReport:
    """Persist account.json, daily parquet, export and catalog row for one replay."""
    # Headlines are derived before the run directory exists so a degenerate ledger leaves no empty run dir.
    unit_headlines = _account_headlines(unit.daily_equity, ACCOUNT_UNIT_REFERENCE_CAPITAL)
    account_headlines = _account_headlines(result.daily_equity, request.capital)
    try:
        run_dir = _resolve_account_destination(
            runs_root=request.runs_root, start=request.evaluation_start, end=request.evaluation_end,
            policy=request.policy, capital=request.capital, execution=request.execution,
        )
        payload = _build_account_payload(
            request=request, strategy_id=strategy.strategy_id, policy=policy, result=result,
            unit=unit, unit_headlines=unit_headlines, account_headlines=account_headlines,
            rules_captured_at=rules_captured_at, venue_path=venue_path, reconciliation=reconciliation,
        )
        (run_dir / "account.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        exposures = result.daily_exposure.to_numpy(dtype="float64")
        pd.DataFrame(
            {"equity": result.daily_equity.to_numpy(dtype="float64"), "exposure": exposures},
            index=result.daily_equity.index,
        ).to_parquet(run_dir / "account_daily.parquet")
        if request.export_unit_returns is not None:
            _export_unit_returns(
                equity=unit.daily_equity, strategy_id=strategy.strategy_id, execution=request.execution,
                start=request.evaluation_start, end=request.evaluation_end, run_dir=run_dir,
                dest=Path(request.export_unit_returns),
            )
        if run_dir.parent == request.runs_root:
            append_backtest_index(
                index_path=request.runs_root.parent.parent / "index.jsonl",
                kind="mhs_frozen_account", run_dir=run_dir, created_at=pd.Timestamp.now(tz="UTC"),
                evaluation_start=request.evaluation_start, evaluation_end=request.evaluation_end,
                strategy_id=strategy.strategy_id, base_cagr=payload["cagr"],
                base_max_drawdown=result.intraday_max_drawdown, execution=request.execution,
            )
    except OSError as exc:
        raise FrozenAccountError(f"frozen account failed: {exc}") from exc
    return FrozenAccountReport(run_dir=run_dir, payload=payload, result=result)


def run_frozen_account(request: FrozenAccountRequest) -> FrozenAccountReport:
    """Replay the frozen account-unit book as one real account and persist account-scale evidence.

    Builds the unlevered clip-0.05 `FROZEN_MHS_TOP20_ACCOUNT_UNIT_V2` candidate and first
    replays it as the unit-exposure reference ledger. That ledger supplies the causal posterior
    moments the growth policy sizes from and anchors reconciliation against the canonical 3m
    ledger, so its failure or liquidation invalidates the account run. The account is then
    replayed with the requested policy, execution model and venue snapshot (venue rules are
    applied retroactively and disclosed as such). Under maker execution both ledgers use the
    canonical strict passive rule and reconciliation only references a same-execution canonical
    run.

    Args:
        request: Validated account controls and explicit run/venue roots.
    Returns:
        Report naming the fresh run directory, the persisted `account.json` mapping and the
        account ledger result.
    Raises:
        ValueError: `validate_frozen_account_request` rejects the request (before any I/O).
        FrozenAccountError: Missing/invalid venue snapshot, invalid frozen request, source
            assembly failure, unit reference replay failure or liquidation, account replay
            failure, or artifact persistence failure (`OSError`). No run directory exists after
            any failure raised before persistence.
    """
    validate_frozen_account_request(request)
    rules, venue_path = _resolve_account_venue(request)
    strategy, parts, _ = _build_account_candidate(request)
    from src.mhs.account_policy import account_growth_policy

    growth_base = account_growth_policy(impact_y=request.impact_y)
    if request.policy == "growth":
        policy = growth_base
    else:
        assert request.fixed_exposure is not None
        policy = dataclasses.replace(growth_base, kind="fixed", exposure_max=request.fixed_exposure)
    unit, result = _replay_account_ledgers(request, parts, rules, policy)
    reconciliation = _reconcile_account_unit(request, strategy, unit)
    return _persist_account_run(
        request, strategy, policy, unit, result, reconciliation,
        rules_captured_at=rules.captured_at.isoformat(), venue_path=str(venue_path),
    )


@dataclass(frozen=True, slots=True)
class FrozenExposureRequest:
    """Inputs for re-deriving the growth exposure rung of one finished frozen run."""

    run_dir: Path
    data_root: Path | None = None
    memory_budget: MhsMemoryBudget | None = None


@dataclass(frozen=True, slots=True)
class FrozenRunArtifacts:
    """Ledger evidence of one finished frozen run, unlevered to exposure 1.0.

    Identity fields are carried verbatim from `result.json` (no coercion) so the derived
    `exposure.json` stays byte-identical to the evidence it was derived from.
    """

    strategy_id: str
    execution_bound: str
    breadth: int
    exposure_multiplier: float
    source_start: pd.Timestamp
    evaluation_start: pd.Timestamp
    evaluation_end: pd.Timestamp
    unit_returns: pd.Series
    unit_max_weight: float


@dataclass(frozen=True, slots=True)
class FrozenExposureReport:
    """Persisted exposure derivation: final `exposure.json` path and its exact mapping."""

    path: Path
    payload: dict[str, Any]


def load_frozen_run_artifacts(run_dir: Path) -> FrozenRunArtifacts:
    """Read and unlever a frozen run's `result.json` and `daily.parquet`.

    Unit returns are `daily["base_return"] / exposure_multiplier` and the unit max name weight
    is `mean(daily["max_name_weight"]) / exposure_multiplier`; both are linear rescalings of
    the levered ledger (residual drift/cost nonlinearity is accepted and disclosed in
    `exposure.json`).

    Args:
        run_dir: Existing frozen run directory.
    Returns:
        Parsed identity and unlevered evidence.
    Raises:
        OSError, ValueError, KeyError, TypeError: Missing, unreadable or malformed artifacts.
    """
    payload = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    daily = pd.read_parquet(run_dir / "daily.parquet")
    exposure_multiplier = float(payload["exposure_multiplier"])
    return FrozenRunArtifacts(
        strategy_id=payload["strategy_id"],
        execution_bound=payload["execution_bound"],
        breadth=payload["breadth"],
        exposure_multiplier=exposure_multiplier,
        source_start=pd.Timestamp(payload["source_start"]).tz_convert("UTC"),
        evaluation_start=pd.Timestamp(payload["evaluation_start"]).tz_convert("UTC"),
        evaluation_end=pd.Timestamp(payload["evaluation_end"]).tz_convert("UTC"),
        unit_returns=daily["base_return"] / exposure_multiplier,
        unit_max_weight=float(daily["max_name_weight"].mean() / exposure_multiplier),
    )


def derive_frozen_exposure(
    artifacts: FrozenRunArtifacts,
    *,
    run_name: str,
    daily_close: pd.DataFrame,
    daily_quote_volume: pd.DataFrame,
    census: tuple[str, ...],
    excluded_symbols: frozenset[str],
) -> dict[str, Any]:
    """Solve the stressed log-growth exposure curve for one frozen run's unit evidence.

    The PIT roster is built from source start (preserving 30-day liquidity and 90-day
    seasoning warm-up) with no trading-exclusion filter, then cut to the evaluation window.
    Only structurally excluded (registry `DELISTED`) symbols feed the gap sample: every other
    symbol's crashes already occurred inside the ledger's realized returns, so re-adding them
    would double-count the same risk. An empty exclusion set yields an empty `GapSample`
    without invoking the sampler on a zero-column frame.

    Args:
        artifacts: Unlevered run evidence.
        run_name: Run directory name recorded as `run_dir`.
        daily_close: Daily last close, float64, from source start to evaluation end.
        daily_quote_volume: Daily summed quote volume (`min_count=1`), float64, same shape.
        census: Panel column order.
        excluded_symbols: Structurally excluded symbols (registry DELISTED).
    Returns:
        The `exposure.json` mapping with every key except `created_at`.
    Raises:
        ValueError: Roster construction or the solver rejects the inputs
            (including `DataIntegrityError`).
    """
    from src.mhs.frozen_research_universe import build_frozen_pit_roster
    from src.mhs.growth_exposure import GapSample, roster_gap_sample, solve_log_growth_exposure

    roster = build_frozen_pit_roster(
        daily_close, daily_quote_volume, census, breadth=artifacts.breadth, blocked_decisions=None,
    )
    in_window = (daily_close.index >= artifacts.evaluation_start) & (daily_close.index < artifacts.evaluation_end)
    gap_symbols = [s for s in census if s in excluded_symbols]
    gaps = (
        roster_gap_sample(
            daily_close.loc[in_window, gap_symbols], roster.loc[in_window, gap_symbols],
            threshold=FROZEN_EXPOSURE_GAP_THRESHOLD,
        )
        if gap_symbols
        else GapSample(magnitudes=np.empty(0, dtype="float64"), events_per_year=0.0)
    )
    solution = solve_log_growth_exposure(
        artifacts.unit_returns,
        max_name_weight=artifacts.unit_max_weight,
        gaps=gaps,
        mean_haircut=FROZEN_EXPOSURE_MEAN_HAIRCUT,
        grid=FROZEN_EXPOSURE_GRID,
        plateau_tolerance=FROZEN_EXPOSURE_PLATEAU_TOLERANCE,
        n_paths=COMMITTEE_GROWTH_N_PATHS,
        horizon_years=COMMITTEE_GROWTH_HORIZON_YEARS,
        mean_block_days=NULL_BOOTSTRAP_MEAN_BLOCK_DAYS,
        seed=FROZEN_EXPOSURE_SEED,
    )
    return {
        "run_dir": run_name,
        "strategy_id": artifacts.strategy_id,
        "execution_bound": artifacts.execution_bound,
        "exposure_multiplier": artifacts.exposure_multiplier,
        "grid": list(solution.grid),
        "growth": list(solution.growth),
        "ruin_probability": list(solution.ruin_probability),
        "argmax": solution.argmax,
        "chosen": solution.chosen,
        "gap_events_per_year": solution.gap_events_per_year,
        "gap_sample_size": solution.gap_sample_size,
        "gap_symbols": sorted(gap_symbols),
        "mean_haircut": FROZEN_EXPOSURE_MEAN_HAIRCUT,
        "plateau_tolerance": FROZEN_EXPOSURE_PLATEAU_TOLERANCE,
        "seed": FROZEN_EXPOSURE_SEED,
        "unlever_assumption": "unit returns and max name weight are linear rescalings of the levered ledger; residual drift/cost nonlinearity is accepted",
    }


def run_frozen_exposure(request: FrozenExposureRequest) -> FrozenExposureReport:
    """Re-derive and atomically persist the growth exposure rung beside one finished frozen run.

    Args:
        request: Run directory, optional OHLCV root override and memory budget.
    Returns:
        Final `exposure.json` path and the persisted mapping.
    Raises:
        FrozenAccountError: Missing run directory, occupied `exposure.json` (checked before any
            read or panel load), invalid run artifacts, resource rejection, data-integrity or
            solver failure, or write failure. `exposure.json` never exists after a failure.
    """
    from src.common.errors import DataIntegrityError
    from src.mhs.growth_exposure import structurally_excluded_symbols
    from src.mhs.panel import load_base_panel
    from src.mhs.resources import _current_tree_swap_bytes, assert_mhs_stage_allocation, resolve_mhs_memory_budget

    run_dir = request.run_dir
    if not run_dir.is_dir():
        raise FrozenAccountError(f"run-dir must be an existing frozen run directory, got {str(run_dir)!r}")
    exposure_path = run_dir / "exposure.json"
    if os.path.lexists(exposure_path):
        raise FrozenAccountError(f"exposure output must be fresh: {exposure_path} already exists")
    try:
        artifacts = load_frozen_run_artifacts(run_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise FrozenAccountError(f"invalid frozen run artifacts: {exc}") from exc
    root = str(request.data_root) if request.data_root is not None else str(FUTURES_DATA_DIR / "ohlcv")
    try:
        resolved = resolve_mhs_memory_budget(request.memory_budget)
        initial_swap_bytes = _current_tree_swap_bytes()

        def _admit_panel(estimated_bytes: int) -> None:
            assert_mhs_stage_allocation(
                stage="frozen_source_panel", estimated_bytes=int(estimated_bytes),
                budget=resolved, replay=False, initial_swap_bytes=initial_swap_bytes,
            )

        assert_mhs_stage_allocation(
            stage="frozen_source_panel", estimated_bytes=0,
            budget=resolved, replay=False, initial_swap_bytes=initial_swap_bytes,
        )
        panel = load_base_panel(
            root, "1h", ("close", "quote_vol"), artifacts.source_start, artifacts.evaluation_end,
            partition="all", selection_mode="causal_history", allocation_admission=_admit_panel,
        )
        daily_close = panel["close"].resample("1D").last().astype("float64")
        daily_quote_volume = panel["quote_vol"].resample("1D").sum(min_count=1).astype("float64")
        census = tuple(panel["close"].columns)
        excluded = structurally_excluded_symbols() if len(census) else frozenset()
        exposure = derive_frozen_exposure(
            artifacts, run_name=run_dir.name, daily_close=daily_close,
            daily_quote_volume=daily_quote_volume, census=census, excluded_symbols=excluded,
        )
        exposure["created_at"] = pd.Timestamp.now(tz="UTC").isoformat()
        tmp_path = run_dir / "exposure.json.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(exposure, handle, sort_keys=True)
        os.replace(tmp_path, exposure_path)
    except (DataIntegrityError, ValueError, OSError) as exc:
        raise FrozenAccountError(f"frozen exposure failed: {exc}") from exc
    return FrozenExposureReport(path=exposure_path, payload=exposure)
