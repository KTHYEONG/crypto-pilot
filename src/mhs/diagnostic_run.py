"""Programmatic entry point; delegates to the pipeline orchestrator."""

from __future__ import annotations

from src.mhs.contracts import MhsDiagnosticRequest
from src.mhs.pipeline import orchestrator
from src.mhs.report.schema import MhsHorizonDiagnosticReport


def run_mhs_horizon_diagnostic(request: MhsDiagnosticRequest) -> MhsHorizonDiagnosticReport:
    """Programmatic entry point; delegates to ``run_mhs_diagnostic`` so research and CLI runs share one composition path (I-ENTRY-EQUIV)."""
    return orchestrator.run_mhs_diagnostic(request)
