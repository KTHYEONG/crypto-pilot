# ruff: noqa
"""Persist an MHS horizon diagnostic report to disk.

Extracted verbatim from ``src.application.research.mhs.evaluation`` (the legacy
monolith). The COMPACT tier writes a git-committable stripped summary JSON plus a
daily-resampled ledger Parquet; the FULL tier writes the lossless 5-category
unified Parquet audit tables. Both tiers append an observational run-history
record. Artifact-table checksum/reference helpers live in ``artifacts``.
"""

from __future__ import annotations

import dataclasses
import io
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import numpy as np
import pandas as pd
from pydantic import SecretStr

from src.application.research.mhs.contracts import (
    MhsBookReport,
    MhsDiagnosticRequest,
    MhsFoldReport,
    MhsOutputTier,
)
from src.application.research.mhs.resources import _peak_rss_bytes
from src.common.errors import DataIntegrityError
from src.live.crypto import derive_key
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


def emit_deployment(report: MhsHorizonDiagnosticReport, request: MhsDiagnosticRequest, artifact_root: Path, *, artifact_key: SecretStr | None = None) -> dict[str, Any]:
    """Emit strategy params + bootstrap (sealed when artifact_key given, else plaintext)."""
    from src.common.errors import DataIntegrityError
    from src.mhs.live_strategy import LiveStrategyParams, capture_params_snapshot, save_strategy_params
    from src.mhs.params import COMMITTEE_MEMBER_SETS, GROWTH_RISK_ENVELOPES, SIGNAL_RETURN_TAIL_DAYS

    if report.status != "COMPLETE":
        raise DataIntegrityError("deployment ineligible: report status not COMPLETE")
    if not getattr(report.research_go, "eligible", False):
        raise DataIntegrityError("deployment ineligible: research_go not eligible")
    if report.blend is None or getattr(report.blend, "target_weights", None) is None:
        raise DataIntegrityError("deployment ineligible: blend target_weights empty")
    tw = report.blend.target_weights
    if tw is None or tw.empty:
        raise DataIntegrityError("deployment ineligible: blend target_weights empty")
    if artifact_key is None:
        logger.warning("[SYS] emit_deployment writing PLAINTEXT artifacts (LIVE_ARTIFACT_KEY unset); fine for local, seal before --deploy-push")
    # admitted members
    member_set_key = getattr(request, "committee_member_set", None)
    if member_set_key not in COMMITTEE_MEMBER_SETS:
        raise DataIntegrityError(f"emit_deployment: unregistered committee_member_set {member_set_key!r}")
    admitted = tuple(COMMITTEE_MEMBER_SETS[member_set_key])
    member_weights = {m: 1.0 / len(admitted) for m in admitted}
    # reference returns for growth budget
    primary = getattr(report.blend, "primary", None)
    if primary is None or getattr(primary, "ledger", None) is None:
        raise DataIntegrityError("emit_deployment requires primary ledger")
    equity = primary.ledger.equity
    ref_returns = equity.resample("1D").last().pct_change().dropna()
    tail = ref_returns.tail(SIGNAL_RETURN_TAIL_DAYS)
    if not tail.empty and tail.index.tz is None:
        tail.index = tail.index.tz_localize("UTC")
    # growth envelope
    from src.mhs.params import GROWTH_RISK_ENVELOPES as _GRE
    envelope = _GRE.get(str(request.growth_envelope))
    if envelope is None:
        raise DataIntegrityError(f"emit_deployment: unregistered growth_envelope {request.growth_envelope!r}")
    from src.application.research.mhs.scaling import _growth_budget_target_vol, resolved_exposure_cap
    from src.application.research.mhs.research_go import _resolved_committee_target_gross
    from src.mhs.live_strategy import BOUND_FLAGS

    gbtv = float(_growth_budget_target_vol(ref_returns, envelope=envelope))
    exp_cap = float(resolved_exposure_cap(request))
    deployed_flags: dict[str, Any] = {}
    for name in BOUND_FLAGS:
        if not hasattr(request, name):
            raise DataIntegrityError(f"emit_deployment: request missing bound flag {name!r}")
        value = _resolved_committee_target_gross(request) if name == "committee_target_gross" else getattr(request, name)
        deployed_flags[name] = value
    params_snapshot = capture_params_snapshot()
    held_row = {str(k): float(v) for k, v in tw.iloc[-1].items()}
    # backtest window
    try:
        from src.research.evaluation.policy import resolve_evaluation_end as _resolve_end
        eval_end = _resolve_end(request.end, unseal_holdout=getattr(request, "final_oos_2026h1", False))
    except Exception:
        eval_end = pd.Timestamp(tw.index[-1])
    try:
        start_ts = pd.Timestamp(request.start) if getattr(request, "start", None) is not None else pd.Timestamp(tw.index[0])
    except Exception:
        start_ts = pd.Timestamp(tw.index[0])
    start_ts = start_ts.tz_localize("UTC") if start_ts.tzinfo is None else start_ts.tz_convert("UTC")
    eval_end = eval_end.tz_localize("UTC") if eval_end.tzinfo is None else eval_end.tz_convert("UTC")
    created_at = pd.Timestamp.now(tz="UTC")
    params = LiveStrategyParams(
        schema_version=1,
        strategy_digest="",
        backtest_window=(start_ts, eval_end),
        created_at=created_at,
        slow_horizon_hours=int(getattr(report.blend, "horizon_hours", 168) or 168),
        committee_member_weights=dict(member_weights),
        admitted_members=tuple(admitted),
        growth_budget_target_vol=gbtv,
        exposure_cap=exp_cap,
        growth_envelope=str(request.growth_envelope),
        execution_universe_size=int(request.execution_universe_size),
        pnl_vol_target_mode=str(request.pnl_vol_target_mode),
        deployed_flags=deployed_flags,
        params_snapshot=params_snapshot,
        bootstrap_held_row=held_row,
    )
    artifact_root = Path(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    params_path = save_strategy_params(artifact_root / "strategy_params.json", params, artifact_key=artifact_key)
    # bootstrap parquet
    tail_df = pd.DataFrame({"reference_daily_return": tail})
    # ensure index tz-aware
    if not tail_df.empty and tail_df.index.tz is None:
        tail_df.index = tail_df.index.tz_localize("UTC")
    import os

    buf = __import__("io").BytesIO()
    tail_df.to_parquet(buf, index=True)
    if artifact_key is not None:
        from src.live.crypto import derive_key, seal_bytes

        payload = seal_bytes(buf.getvalue(), derive_key(artifact_key))
        bootstrap_path = artifact_root / "strategy_bootstrap.parquet.enc"
    else:
        payload = buf.getvalue()
        bootstrap_path = artifact_root / "strategy_bootstrap.parquet"
    tmp = bootstrap_path.with_suffix(bootstrap_path.suffix + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, bootstrap_path)
    # compute digest for return (reload to get digest)
    from src.mhs.live_strategy import load_strategy_params
    loaded = load_strategy_params(params_path, artifact_key=artifact_key)
    return {
        "strategy_digest": loaded.strategy_digest,
        "params_path": str(params_path),
        "bootstrap_path": str(bootstrap_path),
        "n_reference_rows": len(tail),
        "sealed": bool(artifact_key is not None),
    }


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
        "fold_realized_risk_parity": report.fold_realized_risk_parity,
        "evidence_calibration": report.evidence_calibration,
        "fill_mark_parity": report.fill_mark_parity,
        "growth_envelope": report.growth_envelope,
        "committee_member_attribution": report.committee_member_attribution,
        "holdout_tail": report.holdout_tail,
        "parameter_oos_split": report.parameter_oos_split,
        "trial_pool": report.trial_pool,
        "report_path": str(persisted_path) if persisted_path is not None else None,
    }
    return cast(dict[str, Any], _round_6(_jsonable(record)))


#: Legacy replay_id ordering is frozen by artifact partitioning and
#: ``_verify_ledger_artifact``: each book/blend runs
#: primary -> stress -> patient_reference -> pre_vol_target_reference, each
#: fold runs strict -> stress. Any other replay field (touch, ladder, future
#: additions) sorts after these, in declaration order.
_REPLAY_FIELD_RANKS: dict[str, int] = {
    "primary": 0,
    "strict": 1,
    "stress": 2,
    "patient_reference": 3,
    "pre_vol_target_reference": 4,
}


def _replay_field_names(container: Any) -> list[str]:
    """Dataclass field names whose value is a StrategyExecutionReplayResult."""
    return [
        f.name
        for f in dataclasses.fields(container)
        if isinstance(getattr(container, f.name, None), StrategyExecutionReplayResult)
    ]


def _collect_replay_entries(
    report: MhsHorizonDiagnosticReport,
) -> list[tuple[str, StrategyExecutionReplayResult]]:
    """Stable ordered replay sessions (books, blend, folds) for persistence.

    Replay fields are enumerated generically (every dataclass field whose value
    is a ``StrategyExecutionReplayResult``) so touch and ladder are included
    and a future replay field cannot silently leak. The legacy ordering of the
    core fields is preserved exactly; touch/ladder append after
    pre_vol_target_reference.
    """
    replay_entries: list[tuple[str, StrategyExecutionReplayResult]] = []

    def _emit(prefix: str, container: Any) -> None:
        names = enumerate(_replay_field_names(container))
        ordered = sorted(
            names,
            key=lambda pair: (
                _REPLAY_FIELD_RANKS.get(pair[1], len(_REPLAY_FIELD_RANKS)), pair[0],
            ),
        )
        replay_entries.extend(
            (f"{prefix}_{name}", getattr(container, name)) for _idx, name in ordered
        )

    for book_name, book_report in report.books.items():
        _emit(book_name, book_report)
    if report.blend is not None:
        _emit("blend", report.blend)
    for fold_report in report.folds:
        _emit(f"fold{fold_report.fold_index}", fold_report)
    return replay_entries


def _stubbed_report_for_payload(
    report: MhsHorizonDiagnosticReport,
    row_counts: dict[str, dict[str, int]],
) -> MhsHorizonDiagnosticReport:
    """Replace every replay field with ``None`` BEFORE ``to_payload()`` runs.

    ``_jsonable`` would otherwise expand every ledger Series (14.95 s / 0.87 GB
    per 876480-bar replay, all discarded microseconds later by the row-count
    stubs). Covers primary/stress/patient_reference/pre_vol_target_reference,
    touch and ladder on each book and on blend, plus strict/stress on each
    fold. ``row_counts`` must have been captured from the un-stubbed report.

    Also stubs ``target_weights``/``exposure_scale`` on blend: these carry the
    full decision-grid weight matrix (the research-live seam consumed by
    ``--emit-target-weights`` via ``emit_deployed_target_weights`` on the
    *unstubbed* report, called separately) and must never leak into the
    git-committable COMPACT JSON at full resolution.
    """
    del row_counts

    _heavy_scalar_fields = frozenset({"target_weights", "exposure_scale"})

    def _stub(container: Any) -> Any:
        stubs = {
            f.name: None
            for f in dataclasses.fields(container)
            if isinstance(getattr(container, f.name, None), StrategyExecutionReplayResult)
            or (f.name in _heavy_scalar_fields and getattr(container, f.name, None) is not None)
        }
        return dataclasses.replace(container, **stubs)

    return dataclasses.replace(
        report,
        books={name: _stub(book) for name, book in report.books.items()},
        blend=_stub(report.blend) if report.blend is not None else None,
        folds=tuple(_stub(fold) for fold in report.folds),
    )


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
    # Longest-suffix match so underscore-bearing field names
    # (patient_reference, pre_vol_target_reference) split correctly.
    field_names = {
        name
        for container in (*report.books.values(), report.blend, *report.folds)
        if container is not None
        for name in _replay_field_names(container)
    }
    ordered_suffixes = sorted(field_names, key=len, reverse=True)
    for replay_id, _replay in replay_entries:
        ref = _compact_replay_ref(row_counts[replay_id])
        for suffix in ordered_suffixes:
            tag = f"_{suffix}"
            if not replay_id.endswith(tag):
                continue
            container_name = replay_id[: -len(tag)]
            break
        else:  # pragma: no cover - ids are built from these very names
            raise KeyError(f"unresolvable replay id {replay_id!r}")
        if container_name == "blend":
            payload["blend"][suffix] = ref
        elif container_name.startswith("fold"):
            payload["folds"][int(container_name[len("fold"):])][suffix] = ref
        else:
            payload["books"][container_name][suffix] = ref


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

    # The stubbed copy never lets _jsonable expand a ledger Series: every
    # replay field is None before to_payload() runs, then the row-count stubs
    # are wired over the null keys (byte-identical output, ~0 s / ~0 GB).
    payload = _stubbed_report_for_payload(report, row_counts).to_payload()
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
