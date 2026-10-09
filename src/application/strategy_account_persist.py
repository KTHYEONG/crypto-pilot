"""Account replay statistics and run persistence for one strategy account."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.backtests.catalog import append_backtest_index
from src.common.errors import DataIntegrityError
from src.core.params import (
    ACCOUNT_MAKER_FEE_BPS,
    ACCOUNT_PASSIVE_WINDOW_BARS,
    ACCOUNT_TAKER_FEE_BPS,
    ACCOUNT_UNIT_REFERENCE_CAPITAL,
    STRESS_COST_MULTIPLIER,
)

if TYPE_CHECKING:
    from src.application.strategy_account import AccountReplayRequest
    from src.engine.account_ledger import AccountLedgerResult


class AccountReplayError(Exception):
    """Operator-facing failure of an account replay or exposure scan.

    `str(exc)` is the complete operator message (the CLI re-raises it verbatim as
    `SystemExit`). The originating exception is chained as `__cause__`. Distinct from
    `ValueError` so request-validation errors and run failures stay separable and no
    upstream `except ValueError` can swallow a failed run.
    """


@dataclass(frozen=True, slots=True)
class AccountReplayReport:
    """Persisted outcome of one account replay.

    `payload` is the exact mapping serialized to `account.json`; `result` is the account
    ledger outcome (not the unit reference) for caller-side summary logging.
    """

    run_dir: Path
    payload: dict[str, Any]
    result: AccountLedgerResult
    stress_result: AccountLedgerResult


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


def _build_account_payload(
    *, request: AccountReplayRequest, strategy_id: str, policy: Any, result: AccountLedgerResult,
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


def _account_path_statistics(
    *, unit: Any, result: Any, stress: Any, request: AccountReplayRequest, strategy: Any,
) -> dict[str, Any]:
    """Statistics of the account and its separately labelled unit reference.

    Legacy ledgers without per-symbol funding attribution expose funding
    statistics as unavailable instead of inventing zero income.
    """
    from src.evaluation.report import statistics_payload, strategy_statistics

    cutoff = strategy.design_data_cutoff
    if not result.daily_equity.index.equals(stress.daily_equity.index):
        raise AccountReplayError("account replay failed: base/stress daily equity indexes differ")
    out: dict[str, Any] = {}
    for case, ledger, capital in (
        ("unit_reference", unit, ACCOUNT_UNIT_REFERENCE_CAPITAL),
        ("base", result, request.capital),
        ("stress", stress, request.capital),
    ):
        equity = ledger.daily_equity
        if equity.empty:
            raise AccountReplayError("account replay failed: no finite daily returns for statistics")
        values = equity.to_numpy(dtype="float64")
        if not np.isfinite(values).all() or (values < 0).any():
            raise AccountReplayError("account replay failed: invalid daily equity")
        previous = np.concatenate(([float(capital)], values[:-1]))
        if bool(((previous == 0) & (values > 0)).any()):
            raise AccountReplayError("account replay failed: equity resurrects after liquidation")
        returns = pd.Series(np.divide(values, previous, out=np.ones_like(values), where=previous > 0) - 1.0, index=equity.index)
        share = pd.Series(0.0, index=returns.index, dtype="float64")
        funding_daily = getattr(ledger, "funding_by_symbol_daily", None)
        per_symbol: dict[str, float] = {}
        if funding_daily is not None:
            if not funding_daily.index.equals(returns.index):
                raise AccountReplayError("account replay failed: funding attribution index mismatch")
            if not np.isfinite(funding_daily.to_numpy(dtype="float64")).all():
                raise AccountReplayError("account replay failed: non-finite funding attribution")
            share = funding_daily.sum(axis=1) / float(capital)
            per_symbol = {str(sym): float(charge) for sym, charge in funding_daily.sum().items()}
        try:
            stats = strategy_statistics(
                returns,
                daily_funding_share=share,
                funding_by_symbol=per_symbol,
                initial_equity=float(capital),
                design_data_cutoff=cutoff,
            )
        except DataIntegrityError as exc:
            raise AccountReplayError(f"account replay failed: statistics {exc}") from exc
        payload = statistics_payload(stats)
        if funding_daily is None:
            payload["funding"] = None
            for year in payload["years"]:
                year["funding_share"] = None
        out[case] = payload
    return out


def _persist_account_run(request: AccountReplayRequest, strategy: Any, policy: Any, unit: Any, result: Any, reconciliation: dict[str, Any], *, stress: Any, rules_captured_at: str, venue_path: str) -> AccountReplayReport:
    """Persist account.json, daily parquet, export and catalog row for one replay."""
    # Headlines are derived before the run directory exists so a degenerate ledger leaves no empty run dir.
    unit_headlines = _account_headlines(unit.daily_equity, ACCOUNT_UNIT_REFERENCE_CAPITAL)
    account_headlines = _account_headlines(result.daily_equity, request.capital)
    try:
        statistics = _account_path_statistics(unit=unit, result=result, stress=stress, request=request, strategy=strategy)
    except AccountReplayError:
        raise
    except (DataIntegrityError, ValueError) as exc:
        raise AccountReplayError(f"account replay failed: statistics {exc}") from exc
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
        payload["statistics"] = statistics
        payload["design_data_cutoff"] = strategy.design_data_cutoff.isoformat()
        payload["stress_execution"] = {
            "mode": request.execution,
            "taker_fee_bps": ACCOUNT_TAKER_FEE_BPS * STRESS_COST_MULTIPLIER,
            "maker_fee_bps": None if request.execution == "taker" else ACCOUNT_MAKER_FEE_BPS,
            "passive_window_bars": None if request.execution == "taker" else ACCOUNT_PASSIVE_WINDOW_BARS,
            "impact_y": request.impact_y,
            "unit_moment_source": "shared_base_unit_reference",
            "liquidated_at": None if stress.liquidated_at is None else stress.liquidated_at.isoformat(),
        }
        payload["statistics_limitations"] = []
        if any(statistics[case]["funding"] is None for case in ("base", "stress", "unit_reference")):
            payload["statistics_limitations"].append("ACCOUNT_FUNDING_ATTRIBUTION_UNAVAILABLE")
        (run_dir / "account.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        exposures = result.daily_exposure.to_numpy(dtype="float64")
        pd.DataFrame(
            {"equity": result.daily_equity.to_numpy(dtype="float64"), "exposure": exposures},
            index=result.daily_equity.index,
        ).to_parquet(run_dir / "account_daily.parquet")
        pd.DataFrame(
            {"equity": stress.daily_equity, "exposure": stress.daily_exposure},
        ).to_parquet(run_dir / "account_stress_daily.parquet")
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
        raise AccountReplayError(f"account replay failed: {exc}") from exc
    return AccountReplayReport(run_dir=run_dir, payload=payload, result=result, stress_result=stress)
