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
from src.mhs.contracts import MhsOutputTier
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

from src.mhs.contracts import MhsResearchGoResult
from src.mhs.resources import _StageRecorder
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
    """UPDATED existing test: the plaintext seal path now also carries the report evidence weights (the stub gained committee_member_weights)"""
    import dataclasses
    import types

    import pandas as pd

    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.pipeline.config import MhsRunConfig
    from src.mhs.report.persist import emit_deployment

    idx = pd.date_range("2021-01-01", periods=5, freq="1D", tz="UTC")
    tw = pd.DataFrame({"BTCUSDT": [0.2] * 5}, index=idx)
    equity = pd.Series([1.0, 1.01, 1.02, 1.03, 1.04], index=idx)
    report = types.SimpleNamespace(
        status="COMPLETE",
        research_go=types.SimpleNamespace(eligible=True),
        blend=types.SimpleNamespace(
            target_weights=tw,
            horizon_hours=168,
            primary=types.SimpleNamespace(ledger=types.SimpleNamespace(equity=equity)),
        ),
        committee_member_weights={
            "flow_imb_720h": 0.27,
            "flow_imb_168h": 0.378,
            "xs_mom_336h": 0.0,
            "xs_idio_mom_336h": 0.0,
            "mom3_skew_168h": 0.352,
        },
    )
    request = MhsDiagnosticRequest(**dataclasses.asdict(MhsRunConfig(start="2021-01-01", end=None)))
    assert request.committee_evidence_weighting is True

    res = emit_deployment(report, request, tmp_path, artifact_key=None)

    assert res["sealed"] is False
    assert (tmp_path / "strategy_params.json").exists()
    assert any("PLAINTEXT" in r.message for r in caplog.records)




def test_persist_mhs_report_full_lightweight_json_and_parquet(tmp_path: Path) -> None:
    import json
    from tests.unit.mhs.test_report_persist_compact import _build_report
    from src.mhs.report.artifacts import load_mhs_replay_artifact

    report, _ = _build_report(with_touch_ladder=True)
    target = tmp_path / "mhs_horizon_diagnostic.json"
    persisted = persist_mhs_report(report, target, tier=MhsOutputTier.FULL)
    assert persisted == tmp_path / "mhs_horizon_diagnostic_artifacts" / "_full" / "report.json"
    assert persisted.exists()

    # JSON should be lightweight (under 100KB for test report, not bloated with raw series)
    assert persisted.stat().st_size < 100 * 1024
    payload = json.loads(persisted.read_text(encoding="utf-8"))
    assert "checksum_sha256" in payload["artifacts"]["ledger"]
    assert "checksum_sha256" in payload["artifacts"]["fills"]
    assert "fast_reversal_primary" in payload["replay_ids"]

    # Verify all 5 parquet files exist and load_mhs_replay_artifact works
    artifact_dir = persisted.parent
    for cat in ("fills", "units", "notional_weights", "ledger", "times"):
        p = artifact_dir / f"{cat}.parquet"
        assert p.exists()
        loaded = load_mhs_replay_artifact(artifact_dir, "fast_reversal_primary", cat)
        assert isinstance(loaded, pd.DataFrame)
    loaded_ledger = load_mhs_replay_artifact(artifact_dir, "fast_reversal_primary", "ledger")
    assert not loaded_ledger.empty
    assert "equity" in loaded_ledger.columns


def test_emit_deployment_seals_report_evidence_weights(tmp_path) -> None:
    """GIVEN a report carrying the backtest top_level evidence weights WHEN emit_deployment seals params THEN the sealed committee_member_weights equal those evidence weights, never 1/N"""
    import dataclasses
    import types

    import pandas as pd

    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.pipeline.config import MhsRunConfig
    from src.mhs.report.persist import emit_deployment
    from src.mhs.live_strategy import load_strategy_params

    idx = pd.date_range("2021-01-01", periods=5, freq="1D", tz="UTC")
    tw = pd.DataFrame({"BTCUSDT": [0.2] * 5}, index=idx)
    equity = pd.Series([1.0, 1.01, 1.02, 1.03, 1.04], index=idx)
    report = types.SimpleNamespace(
        status="COMPLETE",
        research_go=types.SimpleNamespace(eligible=True),
        blend=types.SimpleNamespace(
            target_weights=tw,
            horizon_hours=168,
            primary=types.SimpleNamespace(ledger=types.SimpleNamespace(equity=equity)),
        ),
        committee_member_weights={
            "flow_imb_720h": 0.27,
            "flow_imb_168h": 0.378,
            "xs_mom_336h": 0.0,
            "xs_idio_mom_336h": 0.0,
            "mom3_skew_168h": 0.352,
        },
    )
    request = MhsDiagnosticRequest(**dataclasses.asdict(MhsRunConfig(start="2021-01-01", end=None)))
    assert request.committee_evidence_weighting is True

    emit_deployment(report, request, tmp_path, artifact_key=None)

    loaded = load_strategy_params(tmp_path / "strategy_params.json", artifact_key=None)
    assert loaded.policy.committee_member_weights == pytest.approx(report.committee_member_weights)
    assert loaded.policy.committee_member_weights["xs_mom_336h"] == 0.0
    assert loaded.policy.committee_member_weights["xs_idio_mom_336h"] == 0.0
    assert set(loaded.policy.admitted_members) == set(report.committee_member_weights)


def test_emit_deployment_fails_closed_without_report_member_weights(tmp_path) -> None:
    """I-WEIGHT-PARITY fail-closed: evidence weighting on but the report carries no committee_member_weights raises DataIntegrityError instead of sealing equal weights"""
    import dataclasses
    import types

    import pandas as pd

    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.pipeline.config import MhsRunConfig
    from src.mhs.report.persist import emit_deployment
    from src.common.errors import DataIntegrityError

    idx = pd.date_range("2021-01-01", periods=5, freq="1D", tz="UTC")
    tw = pd.DataFrame({"BTCUSDT": [0.2] * 5}, index=idx)
    equity = pd.Series([1.0, 1.01, 1.02, 1.03, 1.04], index=idx)
    report = types.SimpleNamespace(
        status="COMPLETE",
        research_go=types.SimpleNamespace(eligible=True),
        blend=types.SimpleNamespace(
            target_weights=tw,
            horizon_hours=168,
            primary=types.SimpleNamespace(ledger=types.SimpleNamespace(equity=equity)),
        ),
        committee_member_weights=None,
    )
    request = MhsDiagnosticRequest(**dataclasses.asdict(MhsRunConfig(start="2021-01-01", end=None)))
    assert request.committee_evidence_weighting is True

    with pytest.raises(DataIntegrityError, match="committee_member_weights"):
        emit_deployment(report, request, tmp_path, artifact_key=None)

    assert not (tmp_path / "strategy_params.json").exists()


def test_emit_deployment_fails_closed_on_non_admitted_member_weight(tmp_path) -> None:
    """Fail-closed when the weights dict names a member outside the resolved committee_member_set (member-set drift between backtest and seal)"""
    import dataclasses
    import types

    import pandas as pd

    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.pipeline.config import MhsRunConfig
    from src.mhs.report.persist import emit_deployment
    from src.common.errors import DataIntegrityError

    idx = pd.date_range("2021-01-01", periods=5, freq="1D", tz="UTC")
    tw = pd.DataFrame({"BTCUSDT": [0.2] * 5}, index=idx)
    equity = pd.Series([1.0, 1.01, 1.02, 1.03, 1.04], index=idx)
    report = types.SimpleNamespace(
        status="COMPLETE",
        research_go=types.SimpleNamespace(eligible=True),
        blend=types.SimpleNamespace(
            target_weights=tw,
            horizon_hours=168,
            primary=types.SimpleNamespace(ledger=types.SimpleNamespace(equity=equity)),
        ),
        committee_member_weights={"flow_imb_168h": 0.6, "rev_24h": 0.4},
    )
    request = MhsDiagnosticRequest(**dataclasses.asdict(MhsRunConfig(start="2021-01-01", end=None)))
    assert request.committee_evidence_weighting is True

    with pytest.raises(DataIntegrityError, match="non-admitted"):
        emit_deployment(report, request, tmp_path, artifact_key=None)


def test_emit_deployment_fails_closed_on_negative_member_weight(tmp_path) -> None:
    """Fail-closed on a negative or non-finite member weight (a short-the-member book is never a deployable evidence weight)"""
    import dataclasses
    import types

    import pandas as pd

    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.pipeline.config import MhsRunConfig
    from src.mhs.report.persist import emit_deployment
    from src.common.errors import DataIntegrityError

    idx = pd.date_range("2021-01-01", periods=5, freq="1D", tz="UTC")
    tw = pd.DataFrame({"BTCUSDT": [0.2] * 5}, index=idx)
    equity = pd.Series([1.0, 1.01, 1.02, 1.03, 1.04], index=idx)
    report = types.SimpleNamespace(
        status="COMPLETE",
        research_go=types.SimpleNamespace(eligible=True),
        blend=types.SimpleNamespace(
            target_weights=tw,
            horizon_hours=168,
            primary=types.SimpleNamespace(ledger=types.SimpleNamespace(equity=equity)),
        ),
        committee_member_weights={"flow_imb_168h": 0.8, "flow_imb_720h": -0.1},
    )
    request = MhsDiagnosticRequest(**dataclasses.asdict(MhsRunConfig(start="2021-01-01", end=None)))
    assert request.committee_evidence_weighting is True

    with pytest.raises(DataIntegrityError, match="finite"):
        emit_deployment(report, request, tmp_path, artifact_key=None)

    report.committee_member_weights = {"flow_imb_168h": True, "flow_imb_720h": 0.2}
    with pytest.raises(DataIntegrityError, match="finite"):
        emit_deployment(report, request, tmp_path, artifact_key=None)


def test_emit_deployment_fails_closed_on_zero_weight_sum(tmp_path) -> None:
    """Fail-closed when every evidence weight is zero: the deployed book would have no renormalizable mix"""
    import dataclasses
    import types

    import pandas as pd

    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.pipeline.config import MhsRunConfig
    from src.mhs.report.persist import emit_deployment
    from src.common.errors import DataIntegrityError

    idx = pd.date_range("2021-01-01", periods=5, freq="1D", tz="UTC")
    tw = pd.DataFrame({"BTCUSDT": [0.2] * 5}, index=idx)
    equity = pd.Series([1.0, 1.01, 1.02, 1.03, 1.04], index=idx)
    report = types.SimpleNamespace(
        status="COMPLETE",
        research_go=types.SimpleNamespace(eligible=True),
        blend=types.SimpleNamespace(
            target_weights=tw,
            horizon_hours=168,
            primary=types.SimpleNamespace(ledger=types.SimpleNamespace(equity=equity)),
        ),
        committee_member_weights={"flow_imb_168h": 0.0, "flow_imb_720h": 0.0},
    )
    request = MhsDiagnosticRequest(**dataclasses.asdict(MhsRunConfig(start="2021-01-01", end=None)))
    assert request.committee_evidence_weighting is True

    with pytest.raises(DataIntegrityError, match="sum must be"):
        emit_deployment(report, request, tmp_path, artifact_key=None)


def test_emit_deployment_equal_weights_when_evidence_weighting_disabled(tmp_path) -> None:
    """Backward compatibility: with committee_evidence_weighting=False the backtest averages members equally, so emit_deployment must keep sealing 1/N"""
    import dataclasses
    import types

    import pandas as pd

    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.pipeline.config import MhsRunConfig
    from src.mhs.report.persist import emit_deployment
    from src.mhs.live_strategy import load_strategy_params

    idx = pd.date_range("2021-01-01", periods=5, freq="1D", tz="UTC")
    tw = pd.DataFrame({"BTCUSDT": [0.2] * 5}, index=idx)
    equity = pd.Series([1.0, 1.01, 1.02, 1.03, 1.04], index=idx)
    report = types.SimpleNamespace(
        status="COMPLETE",
        research_go=types.SimpleNamespace(eligible=True),
        blend=types.SimpleNamespace(
            target_weights=tw,
            horizon_hours=168,
            primary=types.SimpleNamespace(ledger=types.SimpleNamespace(equity=equity)),
        ),
        committee_member_weights=None,
    )
    config = dataclasses.replace(
        MhsRunConfig(start="2021-01-01", end=None), committee_evidence_weighting=False,
    )
    request = MhsDiagnosticRequest(**dataclasses.asdict(config))
    assert request.committee_evidence_weighting is False

    emit_deployment(report, request, tmp_path, artifact_key=None)

    loaded = load_strategy_params(tmp_path / "strategy_params.json", artifact_key=None)
    expected = 1.0 / len(loaded.policy.admitted_members)
    assert loaded.policy.committee_member_weights == pytest.approx(
        dict.fromkeys(loaded.policy.admitted_members, expected)
    )


def test_resolved_deployment_member_weights_rejects_empty_admitted() -> None:
    """멤버가 하나도 없는 집합은 1/N 계산 전에 fail-closed로 막는다."""
    import dataclasses
    import types

    from src.common.errors import DataIntegrityError
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.pipeline.config import MhsRunConfig
    from src.mhs.report.persist import _resolved_deployment_member_weights

    report = types.SimpleNamespace(committee_member_weights=None)
    request = MhsDiagnosticRequest(**dataclasses.asdict(MhsRunConfig(start="2021-01-01", end=None)))

    with pytest.raises(DataIntegrityError, match="empty member tuple"):
        _resolved_deployment_member_weights(report, request, ())




def test_emit_deployment_v2_binds_policy_and_bootstrap(tmp_path) -> None:
    import dataclasses
    import types
    import pandas as pd
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.live_strategy import load_strategy_bootstrap, load_strategy_params
    from src.mhs.pipeline.config import MhsRunConfig
    from src.mhs.report.persist import emit_deployment

    idx = pd.date_range("2021-01-01", periods=5, freq="1D", tz="UTC")
    tw = pd.DataFrame({"BTCUSDT": [0.2] * 5}, index=idx)
    equity = pd.Series([1.0, 1.01, 1.02, 1.03, 1.04], index=idx)
    weights = {"flow_imb_720h": 0.27, "flow_imb_168h": 0.378, "xs_mom_336h": 0.0, "xs_idio_mom_336h": 0.0, "mom3_skew_168h": 0.352}
    report = types.SimpleNamespace(status="COMPLETE", research_go=types.SimpleNamespace(eligible=True), blend=types.SimpleNamespace(target_weights=tw, horizon_hours=168, primary=types.SimpleNamespace(ledger=types.SimpleNamespace(equity=equity))), committee_member_weights=weights)
    request = MhsDiagnosticRequest(**dataclasses.asdict(MhsRunConfig(start="2021-01-01")))
    result = emit_deployment(report, request, tmp_path)
    params = load_strategy_params(tmp_path / "strategy_params.json")
    bootstrap = load_strategy_bootstrap(tmp_path / "strategy_bootstrap.parquet", expected_sha256=params.bootstrap_sha256)
    assert params.schema_version == 2
    assert params.policy.committee_member_weights == weights
    assert result["strategy_digest"] == params.strategy_digest
    assert result["n_reference_rows"] == len(bootstrap)



def test_emit_deployment_v2_resolves_constant_risk_and_median_volumes(tmp_path) -> None:
    import dataclasses
    import types
    import pandas as pd
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.live_strategy import load_strategy_params
    from src.mhs.pipeline.config import MhsRunConfig
    from src.mhs.report.persist import emit_deployment

    idx = pd.date_range("2021-01-01", periods=5, freq="1D", tz="UTC")
    tw = pd.DataFrame({"BTCUSDT": [0.2] * 5}, index=idx)
    equity = pd.Series([1.0, 1.01, 1.02, 1.03, 1.04], index=idx)

    def _report():
        return types.SimpleNamespace(status="COMPLETE", research_go=types.SimpleNamespace(eligible=True), blend=types.SimpleNamespace(target_weights=tw, horizon_hours=168, primary=types.SimpleNamespace(ledger=types.SimpleNamespace(equity=equity))), committee_member_weights=None)

    base = dataclasses.asdict(MhsRunConfig(start="2021-01-01"))
    constant_request = MhsDiagnosticRequest(**{**base, "pnl_vol_target_mode": "constant_risk", "committee_evidence_weighting": False})
    out = tmp_path / "constant"
    result = emit_deployment(_report(), constant_request, out)
    params = load_strategy_params(out / "strategy_params.json")
    assert params.schema_version == 2
    assert params.policy.sizing.mode == "constant_risk"
    assert params.policy.sizing.kelly_enabled is False
    assert result["n_reference_rows"] == 4
    median_request = MhsDiagnosticRequest(**{**base, "pnl_vol_target_mode": "median_relative", "exposure_scale_two_sided": False, "committee_evidence_weighting": False})
    out_median = tmp_path / "median"
    emit_deployment(_report(), median_request, out_median)
    median_params = load_strategy_params(out_median / "strategy_params.json")
    assert median_params.policy.sizing.mode == "median_relative"
    assert median_params.policy.sizing.target_annual_vol == 0.2
