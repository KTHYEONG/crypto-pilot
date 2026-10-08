"""Programmatic entry point; delegates to the pipeline orchestrator."""

from __future__ import annotations

from pathlib import Path

from src.lab.mhs.contracts import MhsDiagnosticRequest
from src.lab.mhs.pipeline import orchestrator
from src.lab.mhs.report.schema import MhsHorizonDiagnosticReport


def run_mhs_horizon_diagnostic(
    request: MhsDiagnosticRequest,
    *,
    procedure_registry: Path | None = None,
    history_dir: Path | None = None,
) -> MhsHorizonDiagnosticReport:
    """Programmatic entry point; delegates to ``run_mhs_diagnostic`` (I-ENTRY-EQUIV), forwarding the forward-look store."""
    return orchestrator.run_mhs_diagnostic(request, procedure_registry=procedure_registry, history_dir=history_dir)
