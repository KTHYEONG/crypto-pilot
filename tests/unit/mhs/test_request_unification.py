"""Single request type invariants (MHS-1 Stage 2 unification)."""

from __future__ import annotations

import dataclasses
import json
import pickle

from src.cli.main import build_root_parser
from src.mhs.contracts import MhsDiagnosticRequest
from src.mhs.pipeline.config import request_from_namespace
from tests.fixtures.mhs_requests import research_baseline

BASE = ["research", "run", "portfolio", "mhs-horizon-diagnostic"]


def test_single_configuration_type() -> None:
    import src.mhs.deployment_policy as deployment_policy
    import src.mhs.params as params
    import src.mhs.pipeline.config as config
    import src.mhs.research_go as research_go

    assert not hasattr(config, "MhsRunConfig")
    assert not hasattr(config, "MemberSet")
    assert not hasattr(params, "COMMITTEE_TARGET_GROSS_UNSET")
    assert not hasattr(research_go, "_resolved_committee_target_gross")
    assert not hasattr(deployment_policy.TargetWeightPolicy, "to_request")


def test_production_defaults_equal_no_arg_cli() -> None:
    args = build_root_parser().parse_args(BASE)
    assert request_from_namespace(args) == MhsDiagnosticRequest()


def test_defaults_are_json_native() -> None:
    payload = dataclasses.asdict(MhsDiagnosticRequest())
    assert json.loads(json.dumps(payload)) == payload


def test_research_baseline_is_identity_empty() -> None:
    from src.mhs.data_policy import MHS_DATA_POLICY_DEFAULT
    from src.mhs.run_history import trial_identity_key

    snapshot = {"K": 1}
    baseline_key = trial_identity_key({
        "flags": dataclasses.asdict(research_baseline()),
        "params_snapshot": snapshot,
    })
    minimal_key = trial_identity_key({
        "flags": {"data_policy": MHS_DATA_POLICY_DEFAULT},
        "params_snapshot": snapshot,
    })
    assert baseline_key == minimal_key
    assert baseline_key == '{"params_snapshot": {"K": 1}}'


def test_research_baseline_reproduces_sentinel_resolution() -> None:
    from src.mhs.types import COMMITTEE_TARGET_GROSS

    assert research_baseline().committee_target_gross is None
    assert research_baseline(committee_capital=True).committee_target_gross == COMMITTEE_TARGET_GROSS
    assert research_baseline(committee_capital=True, committee_target_gross=None).committee_target_gross is None


def test_programmatic_and_cli_identity_coincide() -> None:
    from src.mhs.preregistration import procedure_identity_digest
    from src.mhs.run_history import trial_identity_key

    cli_request = request_from_namespace(build_root_parser().parse_args(BASE))
    programmatic = MhsDiagnosticRequest()
    assert cli_request == programmatic
    assert (
        trial_identity_key({"flags": dataclasses.asdict(cli_request), "params_snapshot": {}})
        == trial_identity_key({"flags": dataclasses.asdict(programmatic), "params_snapshot": {}})
    )
    assert procedure_identity_digest(cli_request) == procedure_identity_digest(programmatic)


def test_horizon_diagnostic_delegates_to_orchestrator(monkeypatch) -> None:
    from src.mhs.diagnostic_run import run_mhs_horizon_diagnostic
    from src.mhs.pipeline import orchestrator

    calls: list[MhsDiagnosticRequest] = []
    sentinel = object()

    def _spy(request: MhsDiagnosticRequest):
        calls.append(request)
        return sentinel

    monkeypatch.setattr(orchestrator, "run_mhs_diagnostic", _spy)
    request = research_baseline()
    assert run_mhs_horizon_diagnostic(request) is sentinel
    assert len(calls) == 1
    assert calls[0] is request


def test_deployment_policy_carries_every_target_weight_field() -> None:
    from src.mhs.deployment_policy import TargetWeightPolicy, build_deployment_policy
    from src.mhs.types import COMMITTEE_TARGET_GROSS

    request = research_baseline(
        committee_capital=True, committee_tranche_smoothing=True, committee_tranche_count=7,
    )
    policy = build_deployment_policy(
        request,
        slow_horizon_hours=168,
        committee_member_weights={"a": 1.0},
        admitted_members=("a",),
        target_annual_vol=0.20,
        exposure_cap=3.0,
    )
    for field in dataclasses.fields(TargetWeightPolicy):
        expected = getattr(request, field.name)
        actual = getattr(policy.target_weights, field.name)
        if expected is None:
            assert actual is None, field.name
        elif isinstance(expected, bool):
            assert actual is expected or actual == expected, field.name
        elif isinstance(expected, int):
            assert actual == int(expected), field.name
        elif isinstance(expected, float):
            assert actual == float(expected), field.name
        else:
            assert actual == str(expected), field.name
    assert policy.target_weights.committee_tranche_count == 7
    assert policy.target_weights.committee_target_gross == COMMITTEE_TARGET_GROSS


def test_pickle_safe_defaults() -> None:
    for request in (MhsDiagnosticRequest(), research_baseline(committee_capital=True)):
        assert pickle.loads(pickle.dumps(request)) == request  # noqa: S301 -- round-trip of a locally constructed request
