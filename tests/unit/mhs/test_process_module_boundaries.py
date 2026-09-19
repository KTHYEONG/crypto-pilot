"""Process module responsibility boundaries after source-owned decomposition."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
BACKTEST = ROOT / "src" / "mhs" / "backtest"


def _read_module(name: str) -> str:
    return (BACKTEST / name).read_text(encoding="utf-8") if "/" not in name else (ROOT / name).read_text(encoding="utf-8")


def test_contract_fields_match_declared_shapes() -> None:
    import dataclasses

    from src.mhs.backtest.contracts import (
        ProcessBacktestReport,
        ProcessInventoryFailureReport,
        ProcessInventoryReport,
        ProcessMarketData,
        ProcessPath,
        RefitRecord,
    )

    expected = {
        ProcessMarketData: ("grid_1h", "decision_grid", "opens_1h", "bar_funding_1h", "log_close_step", "funding_step", "member_books", "execution_mask", "observation_availability", "funding_known_1h", "input_limitations", "member_evidence"),
        RefitRecord: ("point", "member_weights", "smoothing_halflife_days", "policy_id", "train_start", "n_train_labels"),
        ProcessPath: ("one_way_bps", "daily_returns", "unit_daily_returns", "exposure", "refits", "leverage_cap", "execution_policy", "unit_target_weights", "target_weights", "turnover_1h", "risk_sizing", "signal_available_at", "clock_mode", "policy_choices"),
        ProcessBacktestReport: ("start", "end", "certification_level", "n_candidates", "base", "stress", "gate"),
        ProcessInventoryReport: ("proxy", "base", "stress", "gate", "resource_measurements", "memory_stats", "funding_coverage_gaps", "validation"),
        ProcessInventoryFailureReport: ("status", "start", "end", "data_root", "execution_policy", "stage", "error_code", "error_type", "error_message", "total_decisions", "validated_decisions", "completed_decisions", "completed_windows", "completed_decision_start", "completed_decision_end", "source_gaps", "source_gap_excluded_symbols", "resource_measurements", "memory_stats", "funding_coverage_gaps"),
    }
    expected_defaults = {
        ProcessMarketData: ("observation_availability", "funding_known_1h", "input_limitations", "member_evidence"),
        RefitRecord: ("policy_id", "train_start", "n_train_labels"),
        ProcessPath: ("risk_sizing", "signal_available_at", "clock_mode", "policy_choices"),
        ProcessInventoryReport: ("funding_coverage_gaps", "validation"),
        ProcessInventoryFailureReport: ("funding_coverage_gaps",),
    }
    for cls, names in expected.items():
        fields = dataclasses.fields(cls)
        assert tuple(f.name for f in fields) == names
        defaulted = set(expected_defaults.get(cls, ()))
        for field in fields:
            has_default = (
                field.default is not dataclasses.MISSING
                or field.default_factory is not dataclasses.MISSING
            )
            assert has_default == (field.name in defaulted)
        assert cls.__dataclass_params__.frozen
        assert hasattr(cls, "__slots__")


def test_failure_carries_report_with_chained_cause() -> None:
    import pandas as pd

    from src.common.errors import DataIntegrityError
    from src.mhs.backtest.contracts import ProcessInventoryBacktestError, ProcessInventoryFailureReport
    from src.mhs.process import ProcessExecutionPolicy

    report = ProcessInventoryFailureReport(
        status="failed",
        start=pd.Timestamp("2022-01-01", tz="UTC"),
        end=pd.Timestamp("2022-01-02", tz="UTC"),
        data_root=None,
        execution_policy=ProcessExecutionPolicy(),
        stage="replay",
        error_code="DATA_INTEGRITY",
        error_type="ValueError",
        error_message="boom",
        total_decisions=None,
        validated_decisions=0,
        completed_decisions=0,
        completed_windows=0,
        completed_decision_start=None,
        completed_decision_end=None,
        source_gaps=(),
        source_gap_excluded_symbols=(),
        resource_measurements=(),
        memory_stats=None,
    )
    error = ProcessInventoryBacktestError(report)
    cause = ValueError("boom")
    with pytest.raises(ProcessInventoryBacktestError, match="boom"):
        raise error from cause
    assert error.report is report
    assert error.__cause__ is cause
    assert isinstance(error, DataIntegrityError)
    assert str(error) == "boom"


def test_domain_modules_keep_reporting_direction() -> None:
    for name in ("contracts.py", "market_data.py", "paths.py", "inventory.py"):
        tree = ast.parse(_read_module(name))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert not {m for m in imported if m.startswith(("src.mhs.reporting", "src.application", "tools"))}, name
    init = ast.parse((BACKTEST / "__init__.py").read_text(encoding="utf-8"))
    assert not [n for n in init.body if isinstance(n, (ast.Import, ast.ImportFrom))]


def test_maintained_callers_use_new_owners() -> None:
    import src.application.mhs_backtest as app_backtest
    import src.application.mhs_supervisor as supervisor
    import src.application.mhs_worker as worker
    import src.cli.commands.research.mhs as mhs_cli
    import src.mhs.backtest.inventory as bt_inventory
    from src.mhs.backtest.contracts import ProcessInventoryBacktestError
    from src.mhs.reporting.inventory import (
        persist_process_inventory_failure,
        persist_process_inventory_report,
    )
    from src.mhs.reporting.process import persist_process_targets

    assert app_backtest.evaluate_process_inventory_backtest is bt_inventory.evaluate_process_inventory_backtest
    assert app_backtest.persist_process_inventory_report is persist_process_inventory_report
    assert app_backtest.persist_process_inventory_failure is persist_process_inventory_failure
    assert app_backtest.persist_process_targets is persist_process_targets
    assert app_backtest.ProcessInventoryBacktestError is ProcessInventoryBacktestError
    assert not hasattr(supervisor, "PROCESS_INVENTORY_REPORT_PATH")
    assert not hasattr(supervisor, "PROCESS_REPORT_PATH")
    assert not hasattr(supervisor, "PROCESS_POLICY_REPORT_PATH")
    assert worker.ProcessInventoryBacktestError is ProcessInventoryBacktestError
    for rel in ("src/application/mhs_backtest.py", "src/application/mhs_supervisor.py", "src/application/mhs_worker.py", "src/cli/commands/research/mhs.py", "src/mhs/reporting/inventory.py"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "from src.mhs.process_backtest" not in text, rel
        assert "import src.mhs.process_backtest" not in text, rel
    assert not (ROOT / "src" / "mhs" / "process_backtest.py").exists()
    assert not hasattr(mhs_cli, "_run_mhs_process_backtest")
