# ruff: noqa
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.mhs.params import FORWARD_MIN_FOLDS, GrowthRiskEnvelope, MHS_FINAL_OOS_CUTOFF_2026H1

_ENVELOPE = GrowthRiskEnvelope(
    name="unit_test_forward", max_drawdown=0.6, max_drawdown_prob=0.1,
    ruin_fraction=0.6, max_ruin_prob=0.01, horizon_years=1.0, leverage_ceiling=3.0,
)


def _utc(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def _folds(start: str, n: int, mu: float, seed: int) -> list[pd.Series]:
    rng = np.random.default_rng(seed)
    base = _utc(start)
    return [
        pd.Series(mu + 0.01 * rng.standard_normal(84), index=pd.date_range(base + pd.Timedelta(days=91 * i), periods=84, freq="1D"))
        for i in range(n)
    ]


def _request(**overrides):
    import dataclasses
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.pipeline.config import MhsRunConfig

    base = dataclasses.asdict(MhsRunConfig())
    base.update(overrides)
    return MhsDiagnosticRequest(**base)


def _write_history(history_dir: Path, ends: list[str]) -> None:
    history_dir.mkdir(parents=True, exist_ok=True)
    with (history_dir / "active.jsonl").open("w", encoding="utf-8") as fh:
        for end in ends:
            fh.write(json.dumps({"status": "COMPLETE", "resolved_end": end}) + "\n")


# --- fold calendar ---------------------------------------------------------

def test_anchored_folds_through_holdout_equals_legacy_phase_1() -> None:
    from src.mhs.evidence import anchored_purged_folds_through, phase_1_anchored_purged_folds
    from src.quant.evaluation.policy import HOLDOUT_CUTOFF

    legacy = phase_1_anchored_purged_folds()
    assert anchored_purged_folds_through(HOLDOUT_CUTOFF) == legacy
    assert len(legacy) == 16
    assert legacy[-1].validation_end == _utc("2025-12-31")


def test_anchored_folds_through_forward_quarter_appends_quarters() -> None:
    from src.mhs.evidence import anchored_purged_folds_through, phase_1_anchored_purged_folds

    extended = anchored_purged_folds_through(_utc("2026-12-31"))
    assert len(extended) == 20
    assert extended[:16] == phase_1_anchored_purged_folds()
    assert extended[-1].validation_start == _utc("2026-10-08")
    assert extended[-1].validation_end == _utc("2026-12-31")
    with pytest.raises(ValueError):
        anchored_purged_folds_through(pd.Timestamp("2026-12-31"))
    with pytest.raises(ValueError):
        anchored_purged_folds_through(_utc("2022-01-15"))


def test_resolved_anchored_folds_legacy_and_registered() -> None:
    from src.mhs.evidence import phase_1_anchored_purged_folds, resolved_anchored_folds

    legacy = SimpleNamespace(forward_registration_digest=None, end="2026-12-31")
    registered = SimpleNamespace(forward_registration_digest="a" * 32, end="2026-12-31")
    assert resolved_anchored_folds(legacy) == phase_1_anchored_purged_folds()
    assert len(resolved_anchored_folds(registered)) == 20


# --- clocks and horizons ---------------------------------------------------

@pytest.mark.parametrize(
    ("now", "expected"),
    [
        ("2026-09-17", "2026-06-30 23:59:59"),
        ("2026-09-30 12:00", "2026-06-30 23:59:59"),
        ("2026-10-01 00:00", "2026-09-30 23:59:59"),
    ],
)
def test_forward_evaluation_end_ceiling_uses_completed_quarters(now, expected) -> None:
    from src.mhs.preregistration import forward_evaluation_end_ceiling

    assert forward_evaluation_end_ceiling(_utc(now)) == _utc(expected)


def test_forward_evaluation_end_ceiling_rejects_naive_clock() -> None:
    from src.mhs.preregistration import forward_evaluation_end_ceiling

    with pytest.raises(ValueError):
        forward_evaluation_end_ceiling(pd.Timestamp("2026-09-17"))


@pytest.mark.parametrize(
    ("value", "expected"),
    [("2026-12-31", True), ("2026-06-30", True), ("2026-11-30", False), ("2026-12-31 12:00", False)],
)
def test_is_quarter_end_date(value, expected) -> None:
    from src.mhs.preregistration import is_quarter_end_date

    assert is_quarter_end_date(value) is expected


def test_consulted_data_horizon_floors_at_unseal_ceiling_and_tracks_looks(tmp_path) -> None:
    from src.mhs.preregistration import (
        EVENT_EVALUATION, _append_event, consulted_data_horizon,
    )

    history = tmp_path / "history"
    registry = tmp_path / "registry.jsonl"
    # Given no history: the unseal ceiling still counts as consulted
    assert consulted_data_horizon(history, registry) == MHS_FINAL_OOS_CUTOFF_2026H1
    # When a recorded run and a forward evaluation looked further
    _write_history(history, ["2025-12-31 23:59:59+00:00"])
    _append_event(registry, {"event": EVENT_EVALUATION, "procedure_digest": "b" * 32,
                             "resolved_end": "2026-12-31T23:59:59+00:00", "at": "2027-01-05T00:00:00+00:00"})
    # Then the horizon is the latest look
    assert consulted_data_horizon(history, registry) == _utc("2026-12-31 23:59:59")


def test_registry_rejects_unknown_event(tmp_path) -> None:
    from src.mhs.preregistration import load_registrations

    registry = tmp_path / "registry.jsonl"
    registry.write_text(json.dumps({"event": "rewrite"}) + "\n", encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        load_registrations(registry)


# --- registration ----------------------------------------------------------

def test_procedure_identity_ignores_run_control_and_tracks_alpha_fields() -> None:
    from src.mhs.preregistration import procedure_identity_digest

    base = procedure_identity_digest(_request())
    assert base == procedure_identity_digest(_request(end="2026-12-31", log_run=False))
    assert len(base) == 32
    changed = procedure_identity_digest(
        _request(committee_tranche_smoothing=True, committee_regime_adaptive_tranche=False, committee_tranche_count=7)
    )
    assert changed != base


def test_register_procedure_freezes_after_horizon_and_is_append_only(tmp_path) -> None:
    from src.mhs.preregistration import find_registration, load_registrations, register_procedure

    history = tmp_path / "history"
    registry = tmp_path / "registry.jsonl"
    _write_history(history, ["2026-06-30 23:59:59+00:00"])
    request = _request()

    # When registering after the consulted horizon
    registration = register_procedure(request, now=_utc("2026-09-17 09:00"), registry_path=registry, history_dir=history)

    # Then the effective start is the freeze clock and the event is persisted
    assert registration.data_horizon == MHS_FINAL_OOS_CUTOFF_2026H1
    assert registration.effective_start == _utc("2026-09-17 09:00")
    assert find_registration(registration.procedure_digest, registry) == registration
    assert len(load_registrations(registry)) == 1

    # Then a duplicate registration and a clock before the horizon fail closed
    with pytest.raises(DataIntegrityError):
        register_procedure(request, now=_utc("2026-09-18"), registry_path=registry, history_dir=history)
    other = _request(committee_tranche_smoothing=True, committee_regime_adaptive_tranche=False, committee_tranche_count=7)
    with pytest.raises(DataIntegrityError):
        register_procedure(other, now=_utc("2026-05-01"), registry_path=registry, history_dir=history)
    with pytest.raises(ValueError):
        register_procedure(other, now=pd.Timestamp("2026-09-18"), registry_path=registry, history_dir=history)
    with pytest.raises(DataIntegrityError):
        find_registration("c" * 32, registry)
    assert len(load_registrations(registry)) == 1


# --- request validation ----------------------------------------------------

@pytest.mark.parametrize(
    "overrides",
    [
        dict(forward_registration_digest="XYZ", end="2026-12-31"),
        dict(forward_registration_digest="a" * 32, end=None),
        dict(forward_registration_digest="a" * 32, end="2026-11-30"),
        dict(forward_registration_digest="a" * 32, end="2026-12-31", final_oos_2026h1=True),
        dict(forward_registration_digest="a" * 32, end="2026-12-31", fold_safe_horizon_selection=True),
        dict(forward_registration_digest="a" * 32, end="2026-12-31", start="2022-01-01"),
    ],
)
def test_forward_registration_request_validation_rejects(overrides) -> None:
    with pytest.raises(ValueError, match="forward_registration_digest"):
        _request(**overrides)


def test_forward_registration_request_validation_accepts_quarter_end() -> None:
    request = _request(forward_registration_digest="a" * 32, start="2021-01-01", end="2026-12-31")
    assert request.forward_registration_digest == "a" * 32


def test_cli_forward_registration_threads_to_config() -> None:
    import dataclasses
    from src.cli.main import build_root_parser
    from src.mhs.pipeline.config import MhsRunConfig

    base = ["research", "run", "portfolio", "mhs-horizon-diagnostic"]
    assert MhsRunConfig.from_namespace(build_root_parser().parse_args(base)).forward_registration_digest is None
    args = build_root_parser().parse_args([*base, "--end", "2026-12-31", "--forward-registration", "a" * 32])
    assert MhsRunConfig.from_namespace(args).forward_registration_digest == "a" * 32
    assert args.register_procedure is False
    assert build_root_parser().parse_args([*base, "--register-procedure"]).register_procedure is True


# --- forward gate ----------------------------------------------------------

def test_sidak_alpha_and_breadth_required_count() -> None:
    from src.mhs.deploy_gate import breadth_required_count, profitable_fold_critical_count, sidak_alpha

    assert sidak_alpha(0.05, 1) == pytest.approx(0.05)
    assert sidak_alpha(0.05, 2) == pytest.approx(0.0253205655, rel=1e-9)
    assert sidak_alpha(0.05, 20) < sidak_alpha(0.05, 2)
    with pytest.raises(ValueError):
        sidak_alpha(0.05, 0)
    assert breadth_required_count(16, alpha=0.05) == profitable_fold_critical_count(16, alpha=0.05) == 12
    assert breadth_required_count(5, alpha=0.05) == 5
    assert breadth_required_count(4, alpha=0.05) == 4
    assert breadth_required_count(6, alpha=sidak_alpha(0.05, 20)) == 6


def test_forward_gate_blocks_until_minimum_forward_folds() -> None:
    from src.mhs.deploy_gate import GATE_FORWARD_FOLDS_INSUFFICIENT, evaluate_forward_deploy_gate

    history = _folds("2022-01-08", 16, 0.004, seed=1)
    forward = _folds("2026-10-08", FORWARD_MIN_FOLDS - 1, 0.004, seed=2)
    folds = history + forward
    result = evaluate_forward_deploy_gate(
        fold_returns=folds, fold_stress_returns=folds,
        fold_validation_starts=[s.index[0] for s in folds],
        effective_start=_utc("2026-09-17"), n_registrations=1,
        integrity_reasons=(), envelope=_ENVELOPE, n_draws=200,
    )
    assert result.go is False
    assert result.reason_codes == (GATE_FORWARD_FOLDS_INSUFFICIENT,)
    assert result.metrics["n_forward_folds"] == FORWARD_MIN_FOLDS - 1
    assert result.metrics["historical_n_folds"] == 16


def test_forward_gate_integrity_short_circuits() -> None:
    from src.mhs.deploy_gate import GATE_PROCEDURE_NOT_REGISTERED, evaluate_forward_deploy_gate

    result = evaluate_forward_deploy_gate(
        fold_returns=(), fold_stress_returns=(), fold_validation_starts=(),
        effective_start=_utc("2026-09-17"), n_registrations=1,
        integrity_reasons=(GATE_PROCEDURE_NOT_REGISTERED,), envelope=_ENVELOPE,
    )
    assert result.go is False
    assert result.reason_codes == (GATE_PROCEDURE_NOT_REGISTERED,)
    assert result.metrics == {}


def test_forward_gate_verdict_ignores_historical_folds_and_prices_registrations() -> None:
    from src.mhs.deploy_gate import evaluate_forward_deploy_gate, sidak_alpha

    forward = _folds("2026-10-08", 6, 0.004, seed=3)
    stress = [s - 0.0005 for s in forward]
    good_history = _folds("2022-01-08", 16, 0.004, seed=4)
    bad_history = _folds("2022-01-08", 16, -0.004, seed=5)

    def run(history, n_registrations):
        folds = history + forward
        return evaluate_forward_deploy_gate(
            fold_returns=folds, fold_stress_returns=history + stress,
            fold_validation_starts=[s.index[0] for s in folds],
            effective_start=_utc("2026-09-17"), n_registrations=n_registrations,
            integrity_reasons=(), envelope=_ENVELOPE, n_draws=500,
        )

    good = run(good_history, 1)
    bad = run(bad_history, 1)
    # 과거 폴드는 공시일 뿐 판정에 영향이 없다
    assert good.go == bad.go
    assert good.reason_codes == bad.reason_codes
    assert good.metrics == bad.metrics
    assert not any(code.startswith(("E", "F")) for code in good.reason_codes)
    assert good.metrics["n_forward_folds"] == 6
    assert good.metrics["forward_alpha"] == pytest.approx(0.05)

    # 등록 절차가 많을수록 알파가 엄격해지고 하한이 내려간다
    crowded = run(good_history, 20)
    assert crowded.metrics["forward_alpha"] == pytest.approx(sidak_alpha(0.05, 20))
    assert crowded.metrics["oos_ann_log_growth_lcb"] < good.metrics["oos_ann_log_growth_lcb"]


def test_deploy_gate_from_report_requires_matching_registration(tmp_path, monkeypatch) -> None:
    import src.mhs.deploy_gate as dg
    import src.mhs.preregistration as prereg
    import src.mhs.research_go as research_go
    from src.mhs.preregistration import EVENT_REGISTRATION, _append_event

    monkeypatch.setattr(dg, "integrity_reasons_from_report", lambda report, request: ())
    monkeypatch.setattr(research_go, "_resolved_growth_envelope", lambda request: _ENVELOPE)
    registry = tmp_path / "registry.jsonl"
    request = SimpleNamespace(forward_registration_digest="d" * 32)
    report = SimpleNamespace(folds=())

    # Given no registration
    missing = dg.deploy_gate_from_report(report, request, registry_path=registry)
    assert missing.go is False
    assert missing.reason_codes == (dg.GATE_PROCEDURE_NOT_REGISTERED,)

    # Given a registration whose digest the run flags do not reproduce
    _append_event(registry, {"event": EVENT_REGISTRATION, "procedure_digest": "d" * 32,
                             "frozen_at": "2026-09-17T00:00:00+00:00",
                             "data_horizon": "2026-06-30T23:59:59+00:00", "procedure": {}})
    monkeypatch.setattr(prereg, "procedure_identity_digest", lambda request: "e" * 32)
    mismatch = dg.deploy_gate_from_report(report, request, registry_path=registry)
    assert mismatch.go is False
    assert mismatch.reason_codes == (dg.GATE_PROCEDURE_DIGEST_MISMATCH,)


# --- audit coverage: registry edges, evaluation events, entry-point wiring --

def test_procedure_registration_rejects_bad_digest_and_naive_clock() -> None:
    from src.mhs.preregistration import ProcedureRegistration

    with pytest.raises(ValueError):
        ProcedureRegistration("XYZ", _utc("2026-09-17"), _utc("2026-06-30"), {})
    with pytest.raises(ValueError):
        ProcedureRegistration("a" * 32, pd.Timestamp("2026-09-17"), _utc("2026-06-30"), {})
    registration = ProcedureRegistration("a" * 32, _utc("2026-06-01"), _utc("2026-06-30"), {})
    assert registration.effective_start == _utc("2026-06-30")


def test_registry_skips_blank_and_evaluation_lines_and_rejects_malformed(tmp_path) -> None:
    from src.mhs.preregistration import load_registrations

    registry = tmp_path / "registry.jsonl"
    registration = {"event": "registration", "procedure_digest": "a" * 32,
                    "frozen_at": "2026-09-17T00:00:00+00:00", "data_horizon": "2026-06-30T23:59:59+00:00",
                    "procedure": {"k": 1}}
    evaluation = {"event": "evaluation", "procedure_digest": "a" * 32,
                  "resolved_end": "2026-12-31T23:59:59+00:00", "at": "2027-01-05T00:00:00+00:00"}
    registry.write_text("\n" + json.dumps(evaluation) + "\n\n" + json.dumps(registration) + "\n", encoding="utf-8")
    loaded = load_registrations(registry)
    assert [r.procedure_digest for r in loaded] == ["a" * 32]
    assert loaded[0].procedure == {"k": 1}

    malformed = dict(registration)
    del malformed["frozen_at"]
    registry.write_text(json.dumps(malformed) + "\n", encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="malformed registration"):
        load_registrations(registry)


def test_record_forward_evaluation_appends_event_and_advances_horizon(tmp_path) -> None:
    from src.mhs.preregistration import (
        ProcedureRegistration, _read_events, consulted_data_horizon, load_registrations,
        record_forward_evaluation,
    )

    registry = tmp_path / "registry.jsonl"
    registration = ProcedureRegistration("a" * 32, _utc("2026-09-17"), _utc("2026-06-30 23:59:59"), {})

    record_forward_evaluation(registration, _utc("2026-12-31 23:59:59"), now=_utc("2027-01-05"), registry_path=registry)

    assert _read_events(registry) == [{
        "event": "evaluation", "procedure_digest": "a" * 32,
        "resolved_end": "2026-12-31T23:59:59+00:00", "at": "2027-01-05T00:00:00+00:00",
    }]
    assert load_registrations(registry) == ()
    assert consulted_data_horizon(tmp_path / "no_history", registry) == _utc("2026-12-31 23:59:59")
    with pytest.raises(ValueError):
        record_forward_evaluation(registration, pd.Timestamp("2026-12-31"), now=_utc("2027-01-05"), registry_path=registry)
    assert len(_read_events(registry)) == 1


def test_register_procedure_rejects_request_referencing_a_registration(tmp_path) -> None:
    from src.mhs.preregistration import register_procedure

    request = _request(forward_registration_digest="a" * 32, start="2021-01-01", end="2026-12-31")
    with pytest.raises(ValueError, match="must not reference"):
        register_procedure(request, now=_utc("2026-09-17"), registry_path=tmp_path / "r.jsonl", history_dir=tmp_path / "h")
    assert not (tmp_path / "r.jsonl").exists()


def test_forward_evaluation_end_ceiling_before_first_quarter_fails_closed() -> None:
    from src.mhs.preregistration import forward_evaluation_end_ceiling

    with pytest.raises(DataIntegrityError):
        forward_evaluation_end_ceiling(_utc("2021-02-01"))


def test_forward_gate_helpers_fail_closed_on_invalid_inputs() -> None:
    from src.mhs.deploy_gate import breadth_required_count, evaluate_forward_deploy_gate, sidak_alpha

    with pytest.raises(ValueError):
        sidak_alpha(0.0, 1)
    with pytest.raises(ValueError):
        breadth_required_count(0)
    with pytest.raises(ValueError):
        breadth_required_count(4, alpha=1.0)
    folds = _folds("2026-10-08", 4, 0.004, seed=9)
    with pytest.raises(ValueError, match="match in length"):
        evaluate_forward_deploy_gate(
            fold_returns=folds, fold_stress_returns=folds[:3], fold_validation_starts=[s.index[0] for s in folds],
            effective_start=_utc("2026-09-17"), n_registrations=1, integrity_reasons=(), envelope=_ENVELOPE,
        )
    with pytest.raises(ValueError, match="tz-aware"):
        evaluate_forward_deploy_gate(
            fold_returns=folds, fold_stress_returns=folds, fold_validation_starts=[s.index[0] for s in folds],
            effective_start=pd.Timestamp("2026-09-17"), n_registrations=1, integrity_reasons=(), envelope=_ENVELOPE,
        )


def test_deploy_gate_from_report_evaluates_registered_forward_folds(tmp_path, monkeypatch) -> None:
    import src.mhs.deploy_gate as dg
    import src.mhs.preregistration as prereg
    import src.mhs.research_go as research_go
    from src.mhs.preregistration import EVENT_REGISTRATION, _append_event

    monkeypatch.setattr(dg, "integrity_reasons_from_report", lambda report, request: ())
    monkeypatch.setattr(research_go, "_resolved_growth_envelope", lambda request: _ENVELOPE)
    monkeypatch.setattr(prereg, "procedure_identity_digest", lambda request: "d" * 32)
    registry = tmp_path / "registry.jsonl"
    _append_event(registry, {"event": EVENT_REGISTRATION, "procedure_digest": "d" * 32,
                             "frozen_at": "2026-09-17T00:00:00+00:00",
                             "data_horizon": "2026-06-30T23:59:59+00:00", "procedure": {}})

    def replay(returns: pd.Series) -> SimpleNamespace:
        equity = pd.Series(np.cumprod(1.0 + returns.to_numpy()), index=returns.index)
        return SimpleNamespace(ledger=SimpleNamespace(equity=equity))

    history = _folds("2022-01-08", 2, -0.004, seed=11)
    forward = _folds("2026-10-08", 4, 0.004, seed=12)
    folds = []
    for i, series in enumerate(history + forward):
        # 과거 폴드는 naive 문자열, 전진 폴드는 tz-aware로 넣어 두 경로를 모두 검증한다.
        start = str(series.index[0].tz_localize(None)) if i < 2 else series.index[0]
        folds.append(SimpleNamespace(fold_index=i, strict=replay(series), stress=replay(series), validation_start=start))
    report = SimpleNamespace(folds=tuple(reversed(folds)))

    result = dg.deploy_gate_from_report(report, SimpleNamespace(forward_registration_digest="d" * 32),
                                        registry_path=registry, n_draws=200)

    assert result.metrics["n_forward_folds"] == 4
    assert result.metrics["historical_n_folds"] == 2
    assert result.metrics["n_registrations"] == 1
    assert dg.GATE_FORWARD_FOLDS_INSUFFICIENT not in result.reason_codes
    assert result.metrics["oos_ann_log_growth_lcb"] > 0.0


def test_orchestrator_forward_branch_verifies_digest_and_records_look_before_running(monkeypatch) -> None:
    import src.mhs.pipeline.orchestrator as orch
    from src.mhs.pipeline.config import MhsRunConfig
    from src.mhs.preregistration import ProcedureRegistration

    registration = ProcedureRegistration("d" * 32, _utc("2026-01-15"), _utc("2025-12-31 23:59:59"), {})
    recorded: dict[str, object] = {}
    monkeypatch.setattr(orch._prereg, "find_registration", lambda digest: registration)
    monkeypatch.setattr(orch._prereg, "procedure_identity_digest", lambda request: "d" * 32)
    monkeypatch.setattr(
        orch._prereg, "record_forward_evaluation",
        lambda reg, end, *, now: recorded.update(end=end, digest=reg.procedure_digest, now_tz=str(now.tzinfo)),
    )
    # partition=holdout은 기록 직후 파이프라인 진입 전에 실패하므로 무거운 리플레이 없이 분기를 검증한다.
    config = MhsRunConfig(start="2021-01-01", end="2026-06-30", forward_registration_digest="d" * 32, partition="holdout")

    with pytest.raises(RuntimeError, match="dev-only"):
        orch.run_mhs_diagnostic(config)
    assert recorded == {"end": _utc("2026-06-30"), "digest": "d" * 32, "now_tz": "UTC"}

    recorded.clear()
    monkeypatch.setattr(orch._prereg, "procedure_identity_digest", lambda request: "e" * 32)
    with pytest.raises(DataIntegrityError, match="registered procedure digest"):
        orch.run_mhs_diagnostic(config)
    assert recorded == {}


def test_cli_register_procedure_registers_without_running_pipeline(monkeypatch) -> None:
    import src.mhs.pipeline.orchestrator as orch
    import src.mhs.preregistration as prereg
    from src.cli.commands.research import mhs as mhs_cli
    from src.cli.main import build_root_parser

    calls: dict[str, object] = {}
    registration = prereg.ProcedureRegistration("d" * 32, _utc("2026-09-17"), _utc("2026-06-30 23:59:59"), {})

    def fake_register(request, *, now):
        calls.update(forward_field=request.forward_registration_digest, now_tz=str(now.tzinfo))
        return registration

    def forbidden_run(config):
        raise AssertionError("registration must not run the pipeline")

    monkeypatch.setattr(prereg, "register_procedure", fake_register)
    monkeypatch.setattr(orch, "run_mhs_diagnostic", forbidden_run)
    args = build_root_parser().parse_args(
        ["research", "run", "portfolio", "mhs-horizon-diagnostic", "--start", "2021-01-01", "--register-procedure"]
    )

    mhs_cli._run_mhs_horizon_diagnostic(args)

    assert calls == {"forward_field": None, "now_tz": "UTC"}


def test_run_folds_parallel_resolves_folds_from_the_request(monkeypatch) -> None:
    import src.mhs.evaluation.folds as folds_mod

    seen: list[object] = []

    def spy(request):
        seen.append(request)
        return ()

    monkeypatch.setattr(folds_mod, "resolved_anchored_folds", spy)
    request = SimpleNamespace(forward_registration_digest="d" * 32, end="2026-12-31")

    assert folds_mod._run_folds_parallel("root", request, {}, 1.0, None) == ()
    assert seen == [request]


def test_integrity_skips_reliability_eligibility_only_under_forward_protocol(monkeypatch) -> None:
    import src.mhs.deploy_gate as dg
    import src.mhs.execution.integrity as integrity_mod

    monkeypatch.setattr(integrity_mod, "replay_ledger_certified", lambda primary: True)
    fold = SimpleNamespace(strict=object(), failures=())
    ineligible_sealed = SimpleNamespace(eligible=False, input_manifest_digest="f" * 64)
    report = SimpleNamespace(status="COMPLETE", folds=(fold,), blend=SimpleNamespace(primary=object()),
                             backtest_reliability=ineligible_sealed)

    legacy = dg.integrity_reasons_from_report(report, SimpleNamespace(forward_registration_digest=None))
    forward = dg.integrity_reasons_from_report(report, SimpleNamespace(forward_registration_digest="d" * 32))

    assert legacy == (dg.GATE_BACKTEST_RELIABILITY_NOT_ELIGIBLE,)
    assert forward == ()

    # 전진 프로토콜에서도 입력 봉인(I3)은 계속 필수다
    unsealed = SimpleNamespace(status="COMPLETE", folds=(fold,), blend=SimpleNamespace(primary=object()),
                               backtest_reliability=SimpleNamespace(eligible=False, input_manifest_digest=None))
    assert dg.integrity_reasons_from_report(unsealed, SimpleNamespace(forward_registration_digest="d" * 32)) == (
        dg.GATE_INPUT_UNSEALED,
    )
