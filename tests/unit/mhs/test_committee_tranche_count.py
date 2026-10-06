# ruff: noqa
from __future__ import annotations

from tests.fixtures.mhs_requests import research_baseline
import dataclasses

import pytest

from src.cli.dataclass_args import explicit_field_values
from src.mhs.contracts import MhsDiagnosticRequest
from src.mhs.params import COMMITTEE_TRANCHE_COUNT, COMMITTEE_TRANCHE_COUNT_MAX
from src.mhs.pipeline.config import resolve_cli_request

_CLI_BASE = ["research", "run", "portfolio", "mhs-horizon-diagnostic"]


def _request(**overrides):
    from src.mhs.contracts import MhsDiagnosticRequest

    base = dataclasses.asdict(MhsDiagnosticRequest())
    base.update(overrides)
    return research_baseline(**base)


def test_committee_tranche_count_max_matches_shortest_member_lookback() -> None:
    # Given the 168h shortest flow_momentum member lookback on a 24h decision grid
    # Then the registered ceiling is 7 and the default sits inside it
    assert COMMITTEE_TRANCHE_COUNT_MAX == 7
    assert 1 <= COMMITTEE_TRANCHE_COUNT <= COMMITTEE_TRANCHE_COUNT_MAX


def test_cli_committee_tranche_count_defaults_and_threads_to_config() -> None:
    from src.cli.main import build_root_parser
    from src.mhs.contracts import MhsDiagnosticRequest

    # Given no flag
    default_cfg = resolve_cli_request(
        explicit_field_values(MhsDiagnosticRequest, build_root_parser().parse_args(_CLI_BASE))
    )
    # Then the default equals the registered constant and the bare config
    assert default_cfg.committee_tranche_count == COMMITTEE_TRANCHE_COUNT
    assert MhsDiagnosticRequest().committee_tranche_count == COMMITTEE_TRANCHE_COUNT
    assert dataclasses.asdict(default_cfg) == dataclasses.asdict(MhsDiagnosticRequest())

    # When smoothing with an explicit count is requested
    args = build_root_parser().parse_args(
        [*_CLI_BASE, "--committee-tranche-smoothing", "--committee-tranche-count", "7"]
    )
    cfg = resolve_cli_request(explicit_field_values(MhsDiagnosticRequest, args))
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


def test_resolved_committee_tranche_count_roundtrips_sealed_count() -> None:
    # Given a non-default sealed count
    request = _request(
        committee_tranche_smoothing=True, committee_regime_adaptive_tranche=False, committee_tranche_count=7,
    )
    # Then the resolution carries the identical count
    from src.mhs.research_go import _resolved_committee_tranche_count

    assert request.committee_tranche_count == 7
    assert _resolved_committee_tranche_count(request) == 7
