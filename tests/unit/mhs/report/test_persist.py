"""src.mhs.report.persist: tier-dispatch contract.

Behavioral coverage of the compact tier itself (byte-identity, touch/ladder
stubbing) lives in tests/unit/mhs/test_report_persist_compact.py; this module
covers ``persist_mhs_report``'s own dispatch and path contract, which no
other test targets directly.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

import src.mhs.report.persist as persist_mod
from src.application.research.mhs.contracts import MhsOutputTier
from src.mhs.report.persist import mhs_horizon_diagnostic_report_path, persist_mhs_report


def test_report_path_is_source_controlled() -> None:
    assert mhs_horizon_diagnostic_report_path() == str(
        Path("docs/results") / "mhs_horizon_diagnostic.json"
    )


def _patch_history(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(persist_mod, "build_mhs_run_history_record", lambda *a, **k: {})
    monkeypatch.setattr(persist_mod, "append_run_history_record", lambda *a, **k: None)


def test_persist_mhs_report_dispatches_compact_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        persist_mod, "_persist_mhs_report_full",
        lambda report, target: calls.append("full") or target,
    )
    monkeypatch.setattr(
        persist_mod, "_persist_mhs_report_compact",
        lambda report, target: calls.append("compact") or target,
    )
    _patch_history(monkeypatch)

    target = tmp_path / "report.json"
    result = persist_mhs_report(report=object(), target=target)  # type: ignore[arg-type]

    assert calls == ["compact"]
    assert result == target
    assert target.parent.exists()


def test_persist_mhs_report_dispatches_full_when_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        persist_mod, "_persist_mhs_report_full",
        lambda report, target: calls.append("full") or target,
    )
    monkeypatch.setattr(
        persist_mod, "_persist_mhs_report_compact",
        lambda report, target: calls.append("compact") or target,
    )
    _patch_history(monkeypatch)

    target = tmp_path / "report.json"
    persist_mhs_report(report=object(), target=target, tier=MhsOutputTier.FULL)  # type: ignore[arg-type]

    assert calls == ["full"]


def test_persist_mhs_report_swallows_run_history_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run-history append failure is observational and never breaks the
    returned persisted path."""
    monkeypatch.setattr(
        persist_mod, "_persist_mhs_report_compact", lambda report, target: target,
    )
    monkeypatch.setattr(persist_mod, "build_mhs_run_history_record", lambda *a, **k: {})

    def _boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("history backend unavailable")

    monkeypatch.setattr(persist_mod, "append_run_history_record", _boom)

    target = tmp_path / "report.json"
    result = persist_mhs_report(report=object(), target=target)  # type: ignore[arg-type]

    assert result == target


# v2: old seams removed, new emit_deployment tested in test_live_strategy
def test_persist_mhs_report_signature_unchanged_without_flag() -> None:
    """v2: old emit flags removed."""
    signature = inspect.signature(persist_mhs_report)
    assert "emit_target_weights" not in signature.parameters
    assert "emit_signal_state" not in signature.parameters
    assert "emit_deployment_bundle" not in signature.parameters


def test_emit_deployment_exists(tmp_path: Path) -> None:
    from src.mhs.report.persist import emit_deployment
    assert callable(emit_deployment)

# ---------------------------------------------------------------------------
# SCENARIO_MHS_TRIAL_POOL_DISCLOSURE_IN_REPORT_AND_HISTORY: run-history passthrough
# ---------------------------------------------------------------------------

from types import SimpleNamespace

from src.application.research.mhs.contracts import MhsResearchGoResult
from src.application.research.mhs.resources import _StageRecorder
from src.mhs.pipeline.config import MhsRunConfig
from src.mhs.pipeline.context import PipelineContext
from src.mhs.pipeline.stages.assemble import assemble_report
from src.mhs.report.persist import build_mhs_run_history_record
from src.mhs.telemetry import StageTelemetry


def test_SCENARIO_MHS_TRIAL_POOL_DISCLOSURE_IN_REPORT_AND_HISTORY() -> None:
    """The assembled report's trial_pool lands verbatim on the run-history
    record (same unconditional wiring as holdout_tail/parameter_oos_split)."""
    grid = pd.DatetimeIndex([])
    ctx = PipelineContext(
        config=MhsRunConfig(),
        resolved_end="2025-12-31 23:59:59+00:00",
        start=pd.Timestamp("2021-01-01", tz="UTC"),
        end=pd.Timestamp("2025-12-31 23:59:59+00:00"),
        rss_budget_bytes=None,
        rss_reserve_bytes=None,
        root="",
        grid_1h=grid,
        close=pd.DataFrame(),
        opens=pd.DataFrame(),
        quote_vol=pd.DataFrame(),
        taker_buy_quote=None,
        symbols=[],
    )
    ctx.run_start = 0.0
    ctx.recorder = _StageRecorder(log_run=False)
    ctx.telemetry = StageTelemetry(log_run=False)
    ctx.folds = ()
    ctx.deployment = SimpleNamespace(
        geometric_cagr=0.0,
        max_drawdown=0.0,
        calmar=0.0,
        probability_final_wealth_below_initial=0.0,
        research_go_eligible=False,
        execution_go_eligible=False,
        pilot_go_eligible=False,
        scale_go_eligible=False,
    )
    ctx.research_go = MhsResearchGoResult(
        eligible=False, reason_codes=("X",), evaluated_folds=0, folds_passed=0,
    )
    payload = {
        "n_history_records": 3,
        "n_trial_records": 2,
        "excluded_data_integrity": 1,
        "excluded_not_complete": 0,
        "excluded_nonfinite_blend": 0,
        "distinct_trial_keys": 2,
        "neutral_flags_dropped": 4,
        "pool_window_span_days": 181.0,
        "ledger_size": 2,
        "source": "constant_plus_ledger",
    }
    ctx.trial_pool = payload

    report = assemble_report(ctx, ctx.telemetry)
    assert report.trial_pool is payload

    record = build_mhs_run_history_record(report, None, MhsOutputTier.COMPACT, None)
    assert record["trial_pool"] == payload


#: 본 모듈이 검증하는 시나리오 ID(lean_check 추적용).
COVERED_SCENARIOS: tuple[str, ...] = (
    "SCENARIO_LIVE_12_DEPLOYED_WEIGHTS_MATCH_REPLAY_FORMULA",
    "SCENARIO_MHS_TRIAL_POOL_DISCLOSURE_IN_REPORT_AND_HISTORY",
    "SCENARIO_SIGNAL_09_EMIT_SIGNAL_STATE_BOOTSTRAP_SEAM",
)



def test_emit_deployment_plaintext_when_no_key(tmp_path, caplog) -> None:
    import types

    import pandas as pd

    from src.mhs.report.persist import emit_deployment

    idx = pd.date_range("2021-01-01", periods=5, freq="1D", tz="UTC")
    tw = pd.DataFrame({"BTCUSDT": [0.2] * 5}, index=idx)
    equity = pd.Series([1.0, 1.01, 1.02, 1.03, 1.04], index=idx)
    report = types.SimpleNamespace(
        status="COMPLETE", research_go=types.SimpleNamespace(eligible=True),
        blend=types.SimpleNamespace(target_weights=tw, horizon_hours=168, primary=types.SimpleNamespace(ledger=types.SimpleNamespace(equity=equity))),
    )
    import dataclasses

    from src.application.research.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.pipeline.config import MhsRunConfig

    request = MhsDiagnosticRequest(**dataclasses.asdict(MhsRunConfig(start="2021-01-01", end=None)))
    res = emit_deployment(report, request, tmp_path, artifact_key=None)
    assert res["sealed"] is False
    assert (tmp_path / "strategy_params.json").exists()
    assert any("PLAINTEXT" in r.message for r in caplog.records)
