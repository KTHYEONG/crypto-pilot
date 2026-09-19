"""Invariant guards for immutable, transparent validation evidence persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.mhs.backtest.certification import EvidenceCheck, ProcessValidationResult
from src.mhs.backtest.contracts import ProcessBacktestReport, ProcessInventoryReport, ProcessPath
from src.mhs.deploy_gate import DeployGateResult
from src.mhs.execution.contracts import (
    ExecutionDataGap,
    SimulatedInventoryLedgerResult,
    StrategyExecutionReplayResult,
    TerminalPositionEvidence,
)
from src.mhs.process import ProcessExecutionPolicy
from src.mhs.reporting.inventory import (
    export_inventory_json,
    persist_inventory_evidence,
    process_validation_payload,
)

START = pd.Timestamp("2024-01-01", tz="UTC")
END = pd.Timestamp("2024-01-04", tz="UTC")


def _check(
    requirement: str = "margin_survival",
    status: str = "unverified",
    reasons: tuple[str, ...] = ("PRODUCER_ABSENT",),
) -> EvidenceCheck:
    return EvidenceCheck(
        requirement=requirement,
        status=cast(Any, status),
        procedure_digest="proc",
        input_manifest_digest=None,
        code_digest="code",
        interval_start=START,
        interval_end=END,
        artifact_digest=None,
        reason_codes=reasons,
    )


def _validation(
    checks: tuple[EvidenceCheck, ...] | None = None,
    *,
    go: bool = False,
    reasons: tuple[str, ...] = ("REQUIREMENT_UNVERIFIED:margin_survival",),
    diagnostics: dict[str, float | None] | None = None,
) -> ProcessValidationResult:
    selected = checks if checks is not None else (_check(),)
    return ProcessValidationResult(
        accounting_valid=True,
        historical_acceptance="unverified",
        forward_acceptance="unverified",
        requirements=selected,
        gate=DeployGateResult(go=go, reason_codes=reasons, metrics={"observed_equity_rows": 96.0}),
        diagnostic_metrics=dict(diagnostics) if diagnostics is not None else {"observed_equity_rows": 96.0},
        reason_codes=reasons,
    )


def _grid(periods: int = 288) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=periods, freq="15min", tz="UTC")


def _targets() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=3, freq="24h", tz="UTC")
    return pd.DataFrame({"AUSDT": [0.5, 0.5, 0.5], "BUSDT": [-0.5, -0.5, -0.5]}, index=index, dtype="float64")


def _path(targets: pd.DataFrame) -> ProcessPath:
    hourly = pd.date_range(targets.index[0], targets.index[-1] + pd.Timedelta(hours=23), freq="1h", tz="UTC")
    return ProcessPath(
        one_way_bps=8.0,
        daily_returns=pd.Series(0.001, index=targets.index),
        unit_daily_returns=pd.Series(0.001, index=targets.index),
        exposure=pd.Series(1.0, index=targets.index),
        refits=(),
        leverage_cap=2.0,
        execution_policy=ProcessExecutionPolicy(None),
        unit_target_weights=targets,
        target_weights=targets,
        turnover_1h=pd.Series(0.001, index=hourly),
    )


def _proxy(targets: pd.DataFrame) -> ProcessBacktestReport:
    path = _path(targets)
    return ProcessBacktestReport(
        start=targets.index[0],
        end=targets.index[-1],
        certification_level="process_proxy_1h_ledger",
        n_candidates=2,
        base=path,
        stress=path,
        gate=DeployGateResult(go=False, reason_codes=("PROXY_NEVER_DEPLOYS",), metrics={}),
    )


def _result(
    *,
    gaps: tuple[ExecutionDataGap, ...] = (),
    positions: tuple[TerminalPositionEvidence, ...] = (),
    valid: bool = True,
) -> StrategyExecutionReplayResult:
    grid = _grid()
    levels = pd.Series([1.0 + 0.0001 * i for i in range(len(grid))], index=grid, dtype="float64")
    zeros = pd.Series(0.0, index=grid, dtype="float64")
    ledger = SimulatedInventoryLedgerResult(
        equity=levels,
        net_returns=levels.pct_change().dropna(),
        simulated_units=None,
        mark_to_market_pnl=zeros,
        funding_charge=zeros,
        fee_charge=zeros,
        fill_turnover=zeros,
        fill_source="OHLCV_IMMEDIATE_TAKER",
        mark_source="MARK_PRICE",
        primary_valid=valid,
        invalid_reasons=() if valid else ("MISSING_DATA",),
        data_gaps=gaps,
    )
    fills = pd.DataFrame(
        {
            "timestamp": list(grid[:2]),
            "symbol": ["AUSDT", "BUSDT"],
            "quantity_delta": [1.0, -1.0],
            "fill_price": [100.0, 100.0],
            "fee_bps": [8.0, 8.0],
            "reason": ["immediate_taker", "immediate_taker"],
            "pre_trade_equity": [1.0, 1.0],
        }
    )
    return StrategyExecutionReplayResult(
        simulated_fills=fills,
        ledger=ledger,
        simulated_units=pd.DataFrame(columns=["AUSDT"]),
        simulated_notional_weights=pd.DataFrame(columns=["AUSDT"]),
        fill_source="OHLCV_IMMEDIATE_TAKER",
        mark_source="MARK_PRICE",
        submit_times=pd.Series(dtype="float64"),
        fill_times=pd.Series(dtype="float64"),
        fill_count=2,
        unfilled_count=0,
        fallback_count=0,
        all_intent_shortfall_bps=0.0,
        forced_exit_count=0,
        forced_exit_notional=0.0,
        termination_counts={},
        unsupported_assumptions=(),
        elapsed_seconds=0.0,
        terminal_positions=positions,
    )


def _position(
    symbol: str = "AUSDT",
    status: str = "open_marked",
    *,
    mark: float | None = 100.0,
    funding_complete: bool = True,
) -> TerminalPositionEvidence:
    return TerminalPositionEvidence(
        symbol=symbol,
        quantity=1.0,
        cutoff=pd.Timestamp("2024-01-04", tz="UTC"),
        status=cast(Any, status),
        mark=mark,
        mark_available_at=pd.Timestamp("2024-01-04", tz="UTC") if mark is not None else None,
        funding_complete=funding_complete,
        reason_codes=(),
    )


def _report(
    *,
    validation: ProcessValidationResult | None = "default",  # type: ignore[assignment]
    positions: tuple[TerminalPositionEvidence, ...] = (),
    gaps: tuple[ExecutionDataGap, ...] = (),
) -> ProcessInventoryReport:
    from src.mhs.contracts import MhsResourceMeasurement
    from src.mhs.resources import ProcessTreeMemoryStats

    resolved = _validation() if validation == "default" else validation
    gate = resolved.gate if resolved is not None else DeployGateResult(go=False, reason_codes=("X",), metrics={})
    targets = _targets()
    return ProcessInventoryReport(
        proxy=_proxy(targets),
        base=_result(positions=positions, gaps=gaps),
        stress=_result(),
        gate=gate,
        resource_measurements=(MhsResourceMeasurement(stage="replay", elapsed_ms=3, rss_bytes=8),),
        memory_stats=ProcessTreeMemoryStats(
            tree_pss_peak_bytes=10,
            tree_uss_peak_bytes=9,
            min_system_available_bytes=7,
            max_concurrent_procs=2,
            samples_taken=1,
        ),
        validation=resolved,
    )


def test_priced_open_positions_disclosed_without_invented_exits() -> None:
    import src.mhs.reporting.inventory as rep_inventory

    positions = (_position("AUSDT", "open_marked"), _position("BUSDT", "open_marked"))
    terminal = rep_inventory._inventory_terminal_state(_result(positions=positions))
    assert terminal["priced_open_symbols"] == ["AUSDT", "BUSDT"]
    assert terminal["settled_symbols"] == []
    assert terminal["unresolved_symbols"] == []
    serialized = cast(list[dict[str, object]], terminal["terminal_positions"])
    assert serialized[0]["quantity"] == 1.0
    assert serialized[0]["mark"] == 100.0
    assert terminal["terminal_certified"] is True
    fills = _result(positions=positions).simulated_fills
    assert "delist_settlement" not in fills.get("reason", pd.Series(dtype="object")).tolist()


def test_unknown_economics_stays_visible_and_blocks_certification() -> None:
    import src.mhs.reporting.inventory as rep_inventory

    gap = ExecutionDataGap(
        code=cast(Any, "MISSING_HELD_FUNDING"),
        symbol="AUSDT",
        timestamp=pd.Timestamp("2024-01-02", tz="UTC"),
    )
    positions = (_position("AUSDT", "open_marked", funding_complete=False),)
    result = _result(gaps=(gap,), positions=positions, valid=False)
    terminal = rep_inventory._inventory_terminal_state(result)
    assert terminal["terminal_certified"] is False
    assert terminal["primary_valid"] is False
    assert len(cast(list[object], terminal["data_gaps"])) == 1
    assert len(cast(list[object], terminal["terminal_positions"])) == 1
    assert terminal["priced_open_symbols"] == ["AUSDT"]


def test_single_deployment_verdict_across_summary_and_export(tmp_path: Path) -> None:
    checks = (
        _check("input_seal", "passed", ()),
        _check("margin_survival", "unverified", ("PRODUCER_ABSENT",)),
    )
    validation = _validation(checks, go=False, reasons=("REQUIREMENT_UNVERIFIED:margin_survival",))
    report = _report(validation=validation)
    out, _identity = persist_inventory_evidence(report, tmp_path / "summary.json", evidence_root=tmp_path / "ev")
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["gate"]["go"] is False
    assert summary["financial"]["gate"]["go"] is False
    assert summary["financial"]["gate"] == summary["gate"]
    assert summary["validation"]["go"] is False
    assert summary["validation"]["reason_codes"] == ["REQUIREMENT_UNVERIFIED:margin_survival"]
    exported = export_inventory_json(out, tmp_path / "full.json")
    payload = json.loads(exported.read_text(encoding="utf-8"))
    assert payload["gate"]["go"] is False
    assert payload["validation"]["go"] is False
    assert payload["gate"]["reason_codes"] == summary["gate"]["reason_codes"]


def test_missing_values_serialize_as_null(tmp_path: Path) -> None:
    validation = _validation(diagnostics={"observed_equity_rows": 96.0, "base_ann_log_growth_lcb": None})
    payload = process_validation_payload(validation)
    assert payload["diagnostics"]["base_ann_log_growth_lcb"] is None
    report = _report(validation=validation)
    out, _identity = persist_inventory_evidence(report, tmp_path / "s.json", evidence_root=tmp_path / "ev")
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["validation"]["diagnostics"]["base_ann_log_growth_lcb"] is None
    assert summary["validation"]["historical_acceptance"] == "unverified"
    with pytest.raises(DataIntegrityError):
        process_validation_payload(_validation(diagnostics={"base_ann_log_growth_lcb": float("nan")}))
    with pytest.raises(DataIntegrityError):
        process_validation_payload(_validation(diagnostics={"base_ann_log_growth_lcb": float("inf")}))


def test_legacy_evidence_is_not_upgraded(tmp_path: Path) -> None:
    report = _report()
    out, _identity = persist_inventory_evidence(report, tmp_path / "s.json", evidence_root=tmp_path / "ev")
    summary = json.loads(out.read_text(encoding="utf-8"))
    legacy = dict(summary)
    legacy.pop("validation")
    legacy["gate"] = {"go": True, "reason_codes": ["E3_PASS"], "metrics": {}}
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps(legacy), encoding="utf-8")
    exported = export_inventory_json(legacy_path, tmp_path / "legacy_full.json")
    payload = json.loads(exported.read_text(encoding="utf-8"))
    assert payload["validation"]["status"] == "legacy_unverified"
    assert payload["validation"]["go"] is False
    assert payload["gate"]["go"] is False
    assert "LEGACY_VALIDATION_ABSENT" in payload["gate"]["reason_codes"]
    assert payload["financial"]["gate"]["go"] is False
    assert process_validation_payload(None)["go"] is False
    assert process_validation_payload(None)["historical_acceptance"] == "unverified"


def test_evidence_identity_changes_with_audits(tmp_path: Path) -> None:
    gap = ExecutionDataGap(
        code=cast(Any, "MISSING_HELD_FUNDING"),
        symbol="AUSDT",
        timestamp=pd.Timestamp("2024-01-02", tz="UTC"),
    )
    first = _report()
    second = _report(gaps=(gap,))
    out_a, id_a = persist_inventory_evidence(first, tmp_path / "a.json", evidence_root=tmp_path / "ev")
    out_b, id_b = persist_inventory_evidence(second, tmp_path / "b.json", evidence_root=tmp_path / "ev")
    assert id_a != id_b
    _out_a2, id_a2 = persist_inventory_evidence(first, tmp_path / "a2.json", evidence_root=tmp_path / "ev")
    assert id_a2 == id_a
    assert (tmp_path / "ev" / id_a / "manifest.json").is_file()
    assert json.loads(out_a.read_text(encoding="utf-8"))["evidence_id"] == id_a
    assert json.loads(out_b.read_text(encoding="utf-8"))["evidence_id"] == id_b


def test_export_restores_assessment_fidelity(tmp_path: Path) -> None:
    checks = (
        _check("input_seal", "passed", ()),
        _check("margin_survival", "failed", ("MARGIN_CALL_OBSERVED",)),
    )
    validation = _validation(checks, go=False, reasons=("REQUIREMENT_FAILED:margin_survival",))
    positions = (_position("AUSDT", "settled"), _position("BUSDT", "unresolved", mark=None))
    report = _report(validation=validation, positions=positions)
    out, _identity = persist_inventory_evidence(report, tmp_path / "s.json", evidence_root=tmp_path / "ev")
    payload = json.loads(export_inventory_json(out, tmp_path / "full.json").read_text(encoding="utf-8"))
    assert payload["validation"]["historical_acceptance"] == "unverified"
    assert payload["validation"]["forward_acceptance"] == "unverified"
    by_name = {c["requirement"]: c for c in payload["validation"]["requirements"]}
    assert by_name["margin_survival"]["status"] == "failed"
    assert by_name["margin_survival"]["reason_codes"] == ["MARGIN_CALL_OBSERVED"]
    assert by_name["input_seal"]["status"] == "passed"
    exported_positions = payload["base"]["terminal"]["terminal_positions"]
    assert {p["symbol"]: p["status"] for p in exported_positions} == {"AUSDT": "settled", "BUSDT": "unresolved"}
    assert payload["gate"]["reason_codes"] == ["REQUIREMENT_FAILED:margin_survival"]


def test_pruned_detail_is_missing_evidence(tmp_path: Path) -> None:
    report = _report()
    out, identity = persist_inventory_evidence(report, tmp_path / "s.json", evidence_root=tmp_path / "ev")
    (tmp_path / "ev" / identity / "base_daily.parquet").unlink()
    with pytest.raises(DataIntegrityError):
        export_inventory_json(out, tmp_path / "full.json")


def test_atomic_publication_failure_keeps_last_valid_evidence(tmp_path: Path) -> None:
    first = _report()
    out, _identity = persist_inventory_evidence(first, tmp_path / "s.json", evidence_root=tmp_path / "ev")
    before = out.read_bytes()
    with pytest.raises(ValueError, match="fresh"):
        persist_inventory_evidence(_report(), out, evidence_root=tmp_path / "ev")
    assert out.read_bytes() == before
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["validation"]["go"] is False


def test_requirement_identities_aggregation() -> None:
    same = (_check("input_seal", "passed", ()), _check("availability", "passed", ()))
    payload = process_validation_payload(_validation(same, go=False, reasons=()))
    assert payload["procedure_digest"] == "proc"
    assert payload["code_digest"] == "code"
    assert payload["interval_start"] == START.isoformat()
    assert payload["interval_end"] == END.isoformat()
    assert payload["schema"] == "process_validation/1"
    assert payload["status"] == "assessed"
    assert payload["evidence_role"] is None
    assert payload["actual_capital"] is None
    assert payload["look_ordinal"] is None
