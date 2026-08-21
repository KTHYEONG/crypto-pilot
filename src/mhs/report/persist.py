"""Persist an MHS horizon diagnostic report to disk.

Extracted verbatim from ``src.application.research.mhs.evaluation`` (the legacy
monolith). The COMPACT tier writes a git-committable stripped summary JSON plus a
daily-resampled ledger Parquet; the FULL tier writes the lossless 5-category
unified Parquet audit tables. Both tiers append an observational run-history
record. Artifact-table checksum/reference helpers live in ``artifacts``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import numpy as np
import pandas as pd

from src.application.research.mhs.contracts import (
    MhsBookReport,
    MhsDiagnosticRequest,
    MhsFoldReport,
    MhsOutputTier,
)
from src.application.research.mhs.resources import _peak_rss_bytes
from src.common.errors import DataIntegrityError
from src.mhs.execution import StrategyExecutionReplayResult
from src.mhs.params import ARTIFACT_CATEGORIES
from src.mhs.report.artifacts import (
    _artifact_reference,
    _build_replay_artifact_reference,
    _jsonable,
    _to_timestamped_table,
    _verify_ledger_artifact,
)
from src.mhs.report.schema import MhsHorizonDiagnosticReport
from src.mhs.run_history import append_run_history_record, mhs_run_history_dir

logger = logging.getLogger("MhsHorizonDiagnostic")


def mhs_horizon_diagnostic_report_path() -> str:
    """Single source-controlled report path, sibling to the other ``*_report_path`` helpers."""
    return str(Path("docs/results") / "mhs_horizon_diagnostic.json")


def persist_mhs_report(
    report: MhsHorizonDiagnosticReport,
    target: str | Path,
    *,
    tier: MhsOutputTier = MhsOutputTier.COMPACT,
    request: MhsDiagnosticRequest | None = None,
) -> Path | None:
    """Persist the MHS diagnostic in the requested output tier.

    COMPACT (default) writes a git-committable stripped summary JSON at ``target``
    plus a daily-resampled ``daily_ledger.parquet`` under the sibling
    ``*_artifacts`` directory; per-fill detail is intentionally dropped.
    FULL writes the lossless 5-category unified Parquet audit tables and a
    verbose checksummed JSON under ``*_artifacts/_full/`` (gitignored), keeping
    the pre-tiering behaviour byte-for-byte otherwise.

    After either persistence path completes -- including a COMPACT resample
    failure that returns ``None`` -- one lightweight run-history record is
    appended to ``<target.parent>/mhs_run_history/``. History logging is
    observational: a failure there is swallowed via ``logger.warning`` and
    never changes the returned persisted path.

    Returns the persisted JSON path, or ``None`` when a COMPACT resample
    failure is escalated past the compact artifacts (fail-closed policy).
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    persisted: Path | None
    if tier == MhsOutputTier.FULL:
        persisted = _persist_mhs_report_full(report, target)
    else:
        persisted = _persist_mhs_report_compact(report, target)
    try:
        append_run_history_record(
            build_mhs_run_history_record(report, request, tier, persisted),
            mhs_run_history_dir(target),
        )
    except Exception:  # noqa: BLE001 - observational; never break the research result
        logger.warning(
            "[MHS] run-history record append failed path=%s",
            mhs_run_history_dir(target),
            exc_info=True,
        )
    return persisted


# Backward-compatible alias kept importable from ``evaluation`` for legacy callers.
persist_mhs_horizon_diagnostic_report = persist_mhs_report


def _round_6(value: Any) -> Any:
    """Recursively round every float to 6 decimals (logging.md §4 precision)."""
    if isinstance(value, dict):
        return {k: _round_6(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_6(v) for v in value]
    if isinstance(value, float):
        return round(float(value), 6)
    return value


def _book_summary(book: MhsBookReport) -> dict[str, Any]:
    """Curated scalar slice of one book report; heavy replay objects excluded."""
    return {
        "name": book.name,
        "band": book.band,
        "horizon_hours": book.horizon_hours,
        "step_hours": book.step_hours,
        "tranche_count": book.tranche_count,
        "n_symbols": book.n_symbols,
        "primary_autocorr_sharpe": book.primary_autocorr_sharpe,
        "primary_naive_sharpe": book.primary_naive_sharpe,
        "primary_net_ann": book.primary_net_ann,
        "primary_geometric_cagr": book.primary_geometric_cagr,
        "primary_max_drawdown": book.primary_max_drawdown,
        "primary_annualized_turnover": book.primary_annualized_turnover,
        "stress_naive_sharpe": book.stress_naive_sharpe,
        "failure": book.failure,
        "reference_bound_failures": book.reference_bound_failures,
    }


def _fold_summary(fold: MhsFoldReport) -> dict[str, Any]:
    """Curated scalar slice of one anchored-fold report."""
    return {
        "fold_index": fold.fold_index,
        "validation_start": fold.validation_start,
        "validation_end": fold.validation_end,
        "primary_valid": fold.primary_valid,
        "primary_autocorr_sharpe": fold.primary_autocorr_sharpe,
        "primary_naive_sharpe": fold.primary_naive_sharpe,
        "primary_net_ann": fold.primary_net_ann,
        "primary_geometric_cagr": fold.primary_geometric_cagr,
        "primary_max_drawdown": fold.primary_max_drawdown,
        "stress_naive_sharpe": fold.stress_naive_sharpe,
        "failures": fold.failures,
    }


def build_mhs_run_history_record(
    report: MhsHorizonDiagnosticReport,
    request: MhsDiagnosticRequest | None,
    output_tier: MhsOutputTier,
    persisted_path: Path | None,
) -> dict[str, Any]:
    """Curated, structured summary of one MHS run."""
    record: dict[str, Any] = {
        "run_at": datetime.now(UTC).isoformat(),
        "run_id": uuid4().hex,
        "status": report.status,
        "output_tier": output_tier.value,
        "start": report.start,
        "end": report.end,
        "resolved_end": report.resolved_end,
        "flags": dataclasses.asdict(request) if request is not None else None,
        "perf": {
            "run_elapsed_seconds": report.run_elapsed_seconds,
            "peak_rss_bytes": _peak_rss_bytes(report.resource_measurements),
            "eligible_symbols": report.eligible_symbols,
            "realized_execution_roster_size": report.realized_execution_roster_size,
        },
        "books": {name: _book_summary(book) for name, book in report.books.items()},
        "blend": _book_summary(report.blend) if report.blend is not None else None,
        "blend_target_gross": report.blend_target_gross,
        "blend_cash_fraction": report.blend_cash_fraction,
        "deflated_sharpe_ratio": report.deflated_sharpe_ratio,
        "trials_attempted": report.trials_attempted,
        "folds": [_fold_summary(fold) for fold in report.folds],
        "research_go": {
            "eligible": report.research_go.eligible,
            "reason_codes": report.research_go.reason_codes,
            "evaluated_folds": report.research_go.evaluated_folds,
            "folds_passed": report.research_go.folds_passed,
            "data_integrity_reason_codes": report.research_go.data_integrity_reason_codes,
        },
        "discovery_qualification": report.discovery_qualification,
        "committee_diagnostic": report.committee_diagnostic,
        "full_history_yearly_net_t": report.full_history_yearly_net_t,
        "funding_carry_worst_year_corr": report.funding_carry_worst_year_corr,
        "xs_rank_ic": report.xs_rank_ic,
        "date_clustered_regression": report.date_clustered_regression,
        "horizon_diagnostics": report.horizon_diagnostics,
        "bootstrap_ci": report.bootstrap_ci,
        "placebo_sharpe_percentile": report.placebo_sharpe_percentile,
        "deployment_readiness": {
            "geometric_cagr": report.deployment_readiness.geometric_cagr,
            "max_drawdown": report.deployment_readiness.max_drawdown,
            "calmar": report.deployment_readiness.calmar,
            "probability_final_wealth_below_initial": (
                report.deployment_readiness.probability_final_wealth_below_initial
            ),
            "research_go_eligible": report.deployment_readiness.research_go_eligible,
            "execution_go_eligible": report.deployment_readiness.execution_go_eligible,
            "pilot_go_eligible": report.deployment_readiness.pilot_go_eligible,
            "scale_go_eligible": report.deployment_readiness.scale_go_eligible,
        },
        "termination_counts": report.termination_counts,
        "fold_blend_parity": report.fold_blend_parity,
        "fold_growth_concentration": report.fold_growth_concentration,
        "fill_mark_parity": report.fill_mark_parity,
        "report_path": str(persisted_path) if persisted_path is not None else None,
    }
    return cast(dict[str, Any], _round_6(_jsonable(record)))


def _collect_replay_entries(
    report: MhsHorizonDiagnosticReport,
) -> list[tuple[str, StrategyExecutionReplayResult]]:
    """Stable ordered replay sessions (books, blend, folds) for persistence."""
    replay_entries: list[tuple[str, StrategyExecutionReplayResult]] = []
    for book_name, book_report in report.books.items():
        if book_report.primary is not None:
            replay_entries.append((f"{book_name}_primary", book_report.primary))
        if book_report.stress is not None:
            replay_entries.append((f"{book_name}_stress", book_report.stress))
        if book_report.patient_reference is not None:
            replay_entries.append((f"{book_name}_patient_reference", book_report.patient_reference))
        if book_report.pre_vol_target_reference is not None:
            replay_entries.append(
                (f"{book_name}_pre_vol_target_reference", book_report.pre_vol_target_reference)
            )
    if report.blend is not None:
        if report.blend.primary is not None:
            replay_entries.append(("blend_primary", report.blend.primary))
        if report.blend.stress is not None:
            replay_entries.append(("blend_stress", report.blend.stress))
        if report.blend.patient_reference is not None:
            replay_entries.append(("blend_patient_reference", report.blend.patient_reference))
        if report.blend.pre_vol_target_reference is not None:
            replay_entries.append(("blend_pre_vol_target_reference", report.blend.pre_vol_target_reference))
    for fold_report in report.folds:
        if fold_report.strict is not None:
            replay_entries.append((f"fold{fold_report.fold_index}_strict", fold_report.strict))
        if fold_report.stress is not None:
            replay_entries.append((f"fold{fold_report.fold_index}_stress", fold_report.stress))
    return replay_entries


def _write_json_report(path: Path, payload: Any) -> None:
    """Serialize ``payload`` to ``path`` preferring orjson over stdlib json."""
    with path.open("w", encoding="utf-8") as fh:
        try:
            import orjson

            fh.write(orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode("utf-8"))
        except ImportError:
            json.dump(payload, fh, indent=2, ensure_ascii=False)


def _persist_mhs_report_full(
    report: MhsHorizonDiagnosticReport,
    target: Path,
) -> Path:
    """Lossless tier: the pre-tiering 5-category unified audit tables + JSON.

    Artifacts and the verbose report land under ``*_artifacts/_full/`` so the
    compact daily ledger at the artifact root stays git-trackable. The JSON
    carries the full per-replay SHA-256/checksum references exactly as before.
    """
    artifact_root = target.parent / f"{target.stem}_artifacts" / "_full"
    artifact_root.mkdir(parents=True, exist_ok=True)
    payload = report.to_payload()
    replay_entries = _collect_replay_entries(report)

    tables_by_replay: dict[str, dict[str, pd.DataFrame]] = {}
    for replay_id, replay in replay_entries:
        tables_by_replay[replay_id] = _build_replay_category_tables(replay)

    unified_tables = _write_unified_artifact_tables(tables_by_replay, artifact_root)

    # Keep the FULL artifact directory at exactly the 5 canonical unified tables:
    # superseded per-replay Parquet files from earlier persistence formats are
    # removed so re-persisting to the same directory never leaves orphans.
    canonical_names = {f"{category}.parquet" for category in ARTIFACT_CATEGORIES}
    for stale in artifact_root.glob("*.parquet"):
        if stale.name not in canonical_names:
            stale.unlink()

    # Fail-closed ledger integrity verification per replay_id partition.
    for replay_id, _replay in replay_entries:
        _verify_ledger_artifact(
            unified_tables["ledger"][0], replay_id, len(tables_by_replay[replay_id]["ledger"])
        )

    replay_references = {
        replay_id: _build_replay_artifact_reference(
            replay_id, replay, tables_by_replay[replay_id], artifact_root, unified_tables
        )
        for replay_id, replay in replay_entries
    }

    for book_name, book_report in report.books.items():
        book_payload = payload["books"][book_name]
        if book_report.primary is not None:
            book_payload["primary"] = replay_references[f"{book_name}_primary"]
        if book_report.stress is not None:
            book_payload["stress"] = replay_references[f"{book_name}_stress"]
    if report.blend is not None:
        if report.blend.primary is not None:
            payload["blend"]["primary"] = replay_references["blend_primary"]
        if report.blend.stress is not None:
            payload["blend"]["stress"] = replay_references["blend_stress"]
    for fold_report in report.folds:
        fold_payload = payload["folds"][fold_report.fold_index]
        if fold_report.strict is not None:
            fold_payload["strict"] = replay_references[f"fold{fold_report.fold_index}_strict"]
        if fold_report.stress is not None:
            fold_payload["stress"] = replay_references[f"fold{fold_report.fold_index}_stress"]

    payload["artifacts"] = {
        category: _artifact_reference(frame, path)
        for category, (path, frame) in unified_tables.items()
    }
    payload["replay_ids"] = [replay_id for replay_id, _ in replay_entries]

    report_path = artifact_root / "report.json"
    _write_json_report(report_path, payload)
    logger.info("[MHS] full report persisted path=%s", report_path)
    return report_path


def _replay_category_row_counts(replay: StrategyExecutionReplayResult) -> dict[str, int]:
    """Cheap per-category row counts straight off the replay (no table build)."""
    return {
        "fills": len(replay.simulated_fills),
        "units": len(replay.simulated_units),
        "notional_weights": len(replay.simulated_notional_weights),
        "ledger": len(replay.ledger.equity),
        "times": len(replay.submit_times),
    }


def _ledger_table(replay: StrategyExecutionReplayResult) -> pd.DataFrame:
    """Minimal timestamped ledger table (timestamp, equity, fill_turnover)."""
    equity = replay.ledger.equity
    idx = equity.index
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(idx, utc=True),
            "equity": equity.to_numpy(dtype="float64"),
            "fill_turnover": replay.ledger.fill_turnover.reindex(idx).to_numpy(dtype="float64"),
        }
    )


def _compact_replay_ref(row_counts: dict[str, int]) -> dict[str, dict[str, int]]:
    """Stripped per-replay reference: category -> row_count only."""
    return {category: {"row_count": row_counts[category]} for category in ARTIFACT_CATEGORIES}


def _wire_compact_refs(
    payload: Any,
    report: MhsHorizonDiagnosticReport,
    replay_entries: list[tuple[str, StrategyExecutionReplayResult]],
    row_counts: dict[str, dict[str, int]],
) -> None:
    """Replace verbose per-replay artifact references with row-count stubs."""
    for book_name, book_report in report.books.items():
        book_payload = payload["books"][book_name]
        if book_report.primary is not None:
            book_payload["primary"] = _compact_replay_ref(row_counts[f"{book_name}_primary"])
        if book_report.stress is not None:
            book_payload["stress"] = _compact_replay_ref(row_counts[f"{book_name}_stress"])
        if book_report.patient_reference is not None:
            book_payload["patient_reference"] = _compact_replay_ref(
                row_counts[f"{book_name}_patient_reference"]
            )
        if book_report.pre_vol_target_reference is not None:
            book_payload["pre_vol_target_reference"] = _compact_replay_ref(
                row_counts[f"{book_name}_pre_vol_target_reference"]
            )
    if report.blend is not None:
        if report.blend.primary is not None:
            payload["blend"]["primary"] = _compact_replay_ref(row_counts["blend_primary"])
        if report.blend.stress is not None:
            payload["blend"]["stress"] = _compact_replay_ref(row_counts["blend_stress"])
        if report.blend.patient_reference is not None:
            payload["blend"]["patient_reference"] = _compact_replay_ref(
                row_counts["blend_patient_reference"]
            )
        if report.blend.pre_vol_target_reference is not None:
            payload["blend"]["pre_vol_target_reference"] = _compact_replay_ref(
                row_counts["blend_pre_vol_target_reference"]
            )
    for fold_report in report.folds:
        fold_payload = payload["folds"][fold_report.fold_index]
        if fold_report.strict is not None:
            fold_payload["strict"] = _compact_replay_ref(
                row_counts[f"fold{fold_report.fold_index}_strict"]
            )
        if fold_report.stress is not None:
            fold_payload["stress"] = _compact_replay_ref(
                row_counts[f"fold{fold_report.fold_index}_stress"]
            )


def _persist_mhs_report_compact(
    report: MhsHorizonDiagnosticReport,
    target: Path,
) -> Path | None:
    """Compact tier: daily-resampled ledger Parquet + stripped summary JSON.

    The daily rollup is written first; a fail-closed ``DataIntegrityError`` on
    non-finite equity propagates, while any other resample failure logs and
    escalates past compact persistence (returns ``None``).
    """
    artifact_root = target.parent / f"{target.stem}_artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    replay_entries = _collect_replay_entries(report)

    row_counts: dict[str, dict[str, int]] = {}
    daily_frames: list[pd.DataFrame] = []
    for replay_id, replay in replay_entries:
        row_counts[replay_id] = _replay_category_row_counts(replay)
        try:
            daily = _daily_resample_ledger(_ledger_table(replay))
        except DataIntegrityError:
            raise
        except Exception:  # noqa: BLE001
            logger.error(
                "[MHS] compact daily resample failed replay_id=%s", replay_id, exc_info=True
            )
            return None
        tagged = daily.copy()
        tagged.insert(0, "replay_id", replay_id)
        daily_frames.append(tagged)

    daily_table = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame(
        {"replay_id": pd.Series(dtype="string")}
    )
    daily_path = artifact_root / "daily_ledger.parquet"
    daily_table.to_parquet(daily_path, index=False, compression="snappy")

    payload = report.to_payload()
    _wire_compact_refs(payload, report, replay_entries, row_counts)
    unified_row_counts = {
        category: sum(rc[category] for rc in row_counts.values())
        for category in ARTIFACT_CATEGORIES
    }
    payload["artifacts"] = {
        category: {"file": f"{category}.parquet", "row_count": unified_row_counts[category]}
        for category in ARTIFACT_CATEGORIES
    }
    payload["artifacts"]["daily_ledger"] = {
        "file": daily_path.name,
        "row_count": len(daily_table),
    }
    payload["replay_ids"] = [replay_id for replay_id, _ in replay_entries]

    _write_json_report(target, payload)
    size = target.stat().st_size
    if size > 50_000:
        logger.warning(
            "[MHS] compact report exceeds 50KB size=%d path=%s", size, target
        )
    logger.info("[MHS] compact report persisted path=%s", target)
    return target


def _daily_resample_ledger(ledger_table: pd.DataFrame) -> pd.DataFrame:
    """Resample one replay's minute ledger to a daily OHLCV rollup.

    ``ledger_table`` must carry at least ``timestamp``, ``equity`` and
    ``fill_turnover`` columns (the unified ledger schema). One row is emitted
    per UTC day with ``equity_open/high/low/close``, ``daily_return``
    (close/prev_close - 1), ``daily_turnover`` (sum of fill turnover) and
    ``daily_fill_count`` (count of fill-bearing grid rows). Non-finite or
    non-positive equity fails closed with ``DataIntegrityError``.
    """
    required = {"timestamp", "equity", "fill_turnover"}
    missing = required - set(ledger_table.columns)
    if missing:
        raise DataIntegrityError(f"daily resample requires columns {sorted(missing)}")
    frame = ledger_table.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    equity = frame["equity"].to_numpy(dtype="float64")
    if not np.isfinite(equity).all() or (equity <= 0).any():
        raise DataIntegrityError("ledger equity must be finite and strictly positive")
    grouped = frame.groupby(pd.Grouper(key="timestamp", freq="1D"))
    resampled = pd.DataFrame(
        {
            "equity_open": grouped["equity"].first(),
            "equity_high": grouped["equity"].max(),
            "equity_low": grouped["equity"].min(),
            "equity_close": grouped["equity"].last(),
            "daily_turnover": grouped["fill_turnover"].sum(),
            "daily_fill_count": grouped["fill_turnover"].agg(lambda s: int((s > 0).sum())),
        }
    )
    resampled = resampled.rename_axis("date").reset_index()
    resampled["daily_return"] = (
        resampled["equity_close"] / resampled["equity_close"].shift(1) - 1.0
    ).replace([np.inf, -np.inf], np.nan)
    # The daily rollup is a diagnostic aggregate, never a PnL source, so the
    # numeric columns are safely downcast to float32 (validated finite/positive
    # above) to keep the git-tracked compact artifact lean.
    for col in (
        "equity_open", "equity_high", "equity_low", "equity_close",
        "daily_return", "daily_turnover",
    ):
        resampled[col] = resampled[col].astype("float32")
    return resampled


def _build_replay_category_tables(
    replay: StrategyExecutionReplayResult,
) -> dict[str, pd.DataFrame]:
    """Build the five category tables for one replay, without the ``replay_id`` column."""
    fills = replay.simulated_fills.copy()
    if not fills.empty and "timestamp" in fills.columns:
        fills["timestamp"] = pd.to_datetime(fills["timestamp"], utc=True)

    units_table = _to_timestamped_table(replay.simulated_units)
    notional_table = _to_timestamped_table(replay.simulated_notional_weights)

    ledger = pd.concat(
        {
            "equity": replay.ledger.equity,
            "net_returns": replay.ledger.net_returns,
            "mark_to_market_pnl": replay.ledger.mark_to_market_pnl,
            "funding_charge": replay.ledger.funding_charge,
            "fee_charge": replay.ledger.fee_charge,
            "fill_turnover": replay.ledger.fill_turnover,
        },
        axis=1,
    )
    ledger_table = _to_timestamped_table(ledger)

    times = pd.DataFrame(
        {"submit_time": replay.submit_times, "fill_time": replay.fill_times}
    )
    times["submit_time"] = pd.to_datetime(times["submit_time"], utc=True)
    times["fill_time"] = pd.to_datetime(times["fill_time"], utc=True)

    return {
        "fills": fills,
        "units": units_table,
        "notional_weights": notional_table,
        "ledger": ledger_table,
        "times": times,
    }


def _write_unified_artifact_tables(
    tables_by_replay: dict[str, dict[str, pd.DataFrame]],
    artifact_root: Path,
) -> dict[str, tuple[Path, pd.DataFrame]]:
    """Concatenate per-replay category tables into exactly 5 unified Parquet files.

    Every unified table carries a leading ``replay_id`` column; the 5 files are
    written with snappy compression (much faster than zstd for these wide
    numeric tables) and returned as ``{category: (path, frame)}``.  Cross-replay
    schema promotion (timestamps at different precision, string vs large_string)
    is handled by ``pd.concat`` which promotes dtypes losslessly before the
    single snappy Parquet write (spec O8).
    """
    unified_frames: dict[str, list[pd.DataFrame]] = {
        category: [] for category in ARTIFACT_CATEGORIES
    }
    for replay_id, tables in tables_by_replay.items():
        for category in ARTIFACT_CATEGORIES:
            tagged = tables[category].copy()
            tagged.insert(0, "replay_id", replay_id)
            unified_frames[category].append(tagged)

    unified_tables: dict[str, tuple[Path, pd.DataFrame]] = {}
    for category in ARTIFACT_CATEGORIES:
        frames = unified_frames[category]
        if frames:
            frame = pd.concat(frames, ignore_index=True)
        else:
            frame = pd.DataFrame({"replay_id": pd.Series(dtype="string")})
        path = artifact_root / f"{category}.parquet"
        frame.to_parquet(path, index=False, compression="snappy")
        unified_tables[category] = (path, frame)
    return unified_tables
