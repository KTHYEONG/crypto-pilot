"""Testnet drill harness: safety guard, fault transport, session, report."""

from src.live.testnet_drill.config import DrillConfig
from src.live.testnet_drill.faults import FaultAction, FaultInjectingTransport, FaultRule, WireRecord
from src.live.testnet_drill.guard import DrillRefused, assert_drill_allowed, drill_scratch_root
from src.live.testnet_drill.report import (
    RegistryProposal,
    ReportPaths,
    ScenarioResult,
    VenueObservation,
    Verdict,
    apply_conservation,
    build_registry_proposals,
    summarize_results,
    write_report,
)
from src.live.testnet_drill.session import (
    DrillSession,
    SlateNotClean,
    SlateReport,
    open_drill_session,
    prepare_clean_slate,
    teardown_session,
)

__all__ = [
    "DrillConfig",
    "DrillRefused",
    "DrillSession",
    "FaultAction",
    "FaultInjectingTransport",
    "FaultRule",
    "RegistryProposal",
    "ReportPaths",
    "ScenarioResult",
    "SlateNotClean",
    "SlateReport",
    "VenueObservation",
    "Verdict",
    "WireRecord",
    "apply_conservation",
    "assert_drill_allowed",
    "build_registry_proposals",
    "drill_scratch_root",
    "open_drill_session",
    "prepare_clean_slate",
    "summarize_results",
    "teardown_session",
    "write_report",
]
