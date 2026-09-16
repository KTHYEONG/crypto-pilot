# ruff: noqa
from __future__ import annotations

import dataclasses
import json

import pandas as pd
import pytest

from src.mhs.params import COMMITTEE_TRANCHE_COUNT, COMMITTEE_TRANCHE_COUNT_MAX

_CLI_BASE = ["research", "run", "portfolio", "mhs-horizon-diagnostic"]


def _request(**overrides):
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.pipeline.config import MhsRunConfig

    base = dataclasses.asdict(MhsRunConfig())
    base.update(overrides)
    return MhsDiagnosticRequest(**base)


def _policy(request):
    from src.mhs.deployment_policy import build_deployment_policy

    return build_deployment_policy(
        request, slow_horizon_hours=168, committee_member_weights={"m": 1.0},
        admitted_members=("m",), target_annual_vol=0.35, exposure_cap=3.0,
    )


def _params(policy):
    from src.mhs.live_strategy import LiveStrategyParams

    return LiveStrategyParams(
        schema_version=2, strategy_digest="",
        backtest_window=(pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2025-12-31", tz="UTC")),
        created_at=pd.Timestamp("2026-09-16", tz="UTC"), policy=policy,
        bootstrap_sha256="a" * 64, bootstrap_held_row={"BTCUSDT": 0.2},
    )


def test_committee_tranche_count_max_matches_shortest_member_lookback() -> None:
    # Given the 168h shortest flow_momentum member lookback on a 24h decision grid
    # Then the registered ceiling is 7 and the default sits inside it
    assert COMMITTEE_TRANCHE_COUNT_MAX == 7
    assert 1 <= COMMITTEE_TRANCHE_COUNT <= COMMITTEE_TRANCHE_COUNT_MAX


def test_cli_committee_tranche_count_defaults_and_threads_to_config() -> None:
    from src.cli.main import build_root_parser
    from src.mhs.pipeline.config import MhsRunConfig

    # Given no flag
    default_cfg = MhsRunConfig.from_namespace(build_root_parser().parse_args(_CLI_BASE))
    # Then the default equals the registered constant and the bare config
    assert default_cfg.committee_tranche_count == COMMITTEE_TRANCHE_COUNT
    assert MhsRunConfig().committee_tranche_count == COMMITTEE_TRANCHE_COUNT
    assert dataclasses.asdict(default_cfg) == dataclasses.asdict(MhsRunConfig())

    # When smoothing with an explicit count is requested
    args = build_root_parser().parse_args(
        [*_CLI_BASE, "--committee-tranche-smoothing", "--committee-tranche-count", "7"]
    )
    cfg = MhsRunConfig.from_namespace(args)
    # Then the count threads through and adaptive is disabled
    assert cfg.committee_tranche_count == 7
    assert cfg.committee_tranche_smoothing is True
    assert cfg.committee_regime_adaptive_tranche is False


@pytest.mark.parametrize(
    ("smoothing", "adaptive", "count", "expected"),
    [
        (True, False, 7, 7),
        (False, True, 5, 5),
        (True, False, 1, 1),
        (False, True, COMMITTEE_TRANCHE_COUNT, COMMITTEE_TRANCHE_COUNT),
        (False, False, COMMITTEE_TRANCHE_COUNT, 1),
    ],
)
def test_resolved_committee_tranche_count(smoothing, adaptive, count, expected) -> None:
    from src.mhs.research_go import _resolved_committee_tranche_count

    request = _request(
        committee_tranche_smoothing=smoothing,
        committee_regime_adaptive_tranche=adaptive,
        committee_tranche_count=count,
    )
    assert _resolved_committee_tranche_count(request) == expected


@pytest.mark.parametrize(
    "overrides",
    [
        dict(committee_tranche_smoothing=True, committee_regime_adaptive_tranche=False, committee_tranche_count=0),
        dict(committee_tranche_smoothing=True, committee_regime_adaptive_tranche=False, committee_tranche_count=COMMITTEE_TRANCHE_COUNT_MAX + 1),
        dict(committee_tranche_smoothing=True, committee_regime_adaptive_tranche=False, committee_tranche_count=True),
        dict(committee_tranche_smoothing=True, committee_regime_adaptive_tranche=False, committee_tranche_count=7.0),
        dict(committee_tranche_smoothing=False, committee_regime_adaptive_tranche=False, committee_tranche_count=7),
        dict(committee_capital=False, committee_tranche_smoothing=False, committee_regime_adaptive_tranche=False,
             committee_target_gross=None, funding_carry_sleeve=False, funding_carry_weight=0.0,
             committee_kelly_sizing=False, committee_evidence_weighting=False, committee_tranche_count=5),
    ],
)
def test_committee_tranche_count_rejects_invalid(overrides) -> None:
    with pytest.raises(ValueError, match="committee_tranche_count"):
        _request(**overrides)


@pytest.mark.parametrize("count", [1, COMMITTEE_TRANCHE_COUNT, COMMITTEE_TRANCHE_COUNT_MAX])
def test_committee_tranche_count_accepts_bounds_with_smoothing(count) -> None:
    request = _request(
        committee_tranche_smoothing=True, committee_regime_adaptive_tranche=False, committee_tranche_count=count,
    )
    assert request.committee_tranche_count == count


def test_deployment_policy_roundtrips_committee_tranche_count() -> None:
    # Given a non-default sealed count
    request = _request(
        committee_tranche_smoothing=True, committee_regime_adaptive_tranche=False, committee_tranche_count=7,
    )
    policy = _policy(request)
    # When the live seam restores the request
    restored = policy.target_weights.to_request()
    # Then the live book is rebuilt with the identical count and resolution
    from src.mhs.research_go import _resolved_committee_tranche_count

    assert policy.target_weights.committee_tranche_count == 7
    assert restored.committee_tranche_count == 7
    assert _resolved_committee_tranche_count(restored) == _resolved_committee_tranche_count(request) == 7


def test_strategy_params_default_tranche_count_is_implicit_and_digest_stable(tmp_path) -> None:
    from src.mhs.live_strategy import _compute_strategy_digest, load_strategy_params, save_strategy_params

    # Given a default-count policy
    path = save_strategy_params(tmp_path / "default.json", _params(_policy(_request())))
    raw = json.loads(path.read_text(encoding="utf-8"))
    # Then the key is omitted, the digest equals the legacy key-less digest, and load restores the default
    assert "committee_tranche_count" not in raw["policy"]["target_weights"]
    legacy_raw = {k: v for k, v in raw.items() if k != "strategy_digest"}
    assert _compute_strategy_digest(legacy_raw) == raw["strategy_digest"]
    assert load_strategy_params(path).policy.target_weights.committee_tranche_count == COMMITTEE_TRANCHE_COUNT


def test_strategy_params_non_default_tranche_count_enters_digest(tmp_path) -> None:
    from src.mhs.live_strategy import load_strategy_params, save_strategy_params

    default_path = save_strategy_params(tmp_path / "default.json", _params(_policy(_request())))
    seven_request = _request(
        committee_tranche_smoothing=True, committee_regime_adaptive_tranche=False, committee_tranche_count=7,
    )
    seven_path = save_strategy_params(tmp_path / "seven.json", _params(_policy(seven_request)))
    default_raw = json.loads(default_path.read_text(encoding="utf-8"))
    seven_raw = json.loads(seven_path.read_text(encoding="utf-8"))

    assert seven_raw["policy"]["target_weights"]["committee_tranche_count"] == 7
    assert seven_raw["strategy_digest"] != default_raw["strategy_digest"]
    assert load_strategy_params(seven_path).policy.target_weights.committee_tranche_count == 7


def test_strategy_params_tranche_count_tamper_rejected(tmp_path) -> None:
    from src.common.errors import DataIntegrityError
    from src.mhs.live_strategy import load_strategy_params, save_strategy_params

    seven_request = _request(
        committee_tranche_smoothing=True, committee_regime_adaptive_tranche=False, committee_tranche_count=7,
    )
    path = save_strategy_params(tmp_path / "seven.json", _params(_policy(seven_request)))
    raw = json.loads(path.read_text(encoding="utf-8"))
    # When the sealed count is edited without re-sealing
    raw["policy"]["target_weights"]["committee_tranche_count"] = 5
    path.write_text(json.dumps(raw), encoding="utf-8")
    # Then the digest check fails closed
    with pytest.raises(DataIntegrityError):
        load_strategy_params(path)
