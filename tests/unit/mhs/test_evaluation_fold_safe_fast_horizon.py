from __future__ import annotations

import pytest

from src.mhs.evaluation import _fold_safe_fast_horizon
from src.mhs.discovery import DiscoveryQualificationResult


def _selection(
    selected_horizon: int | None,
    admitted: bool,
) -> DiscoveryQualificationResult:
    return DiscoveryQualificationResult(
        selected_horizon=selected_horizon,
        admitted=admitted,
        discovery_scores=() if selected_horizon is None else ((selected_horizon, 2.5),),
        discovery_aggregate_net_t=2.5 if selected_horizon is not None else None,
        qualification_net_t=2.3 if selected_horizon is not None else None,
        qualification_sign_consistent=True if selected_horizon is not None else None,
    )


def test_fold_safe_fast_horizon_python_assertion() -> None:
    # contract.json python_assertion: a fail-closed selection resolves to the
    # frozen 48h default with "frozen_default" source.
    assert _fold_safe_fast_horizon(
        DiscoveryQualificationResult(
            selected_horizon=None, admitted=False, discovery_scores=(),
            discovery_aggregate_net_t=None, qualification_net_t=None,
            qualification_sign_consistent=None,
        ),
        48,
    ) == (48, "frozen_default")


def test_fold_safe_fast_horizon_admitted() -> None:
    # SCENARIO_FOLD_SAFE_FAST_HORIZON_ADMITTED: when the fold-scoped gate
    # admitted a candidate (e.g. 96h), the resolver returns the selected
    # horizon with "fold_train_only_discovery" -- evidence only, the capital
    # allocation stays frozen at 0.0.
    assert _fold_safe_fast_horizon(_selection(96, True), 48) == (
        96, "fold_train_only_discovery",
    )


def test_fold_safe_fast_horizon_not_admitted() -> None:
    # SCENARIO_FOLD_SAFE_FAST_HORIZON_NOT_ADMITTED: a rejected selection
    # (admitted=False) fails closed to the frozen 48h default regardless of any
    # selected_horizon carried by the result.
    assert _fold_safe_fast_horizon(_selection(96, False), 48) == (
        48, "frozen_default",
    )


def test_fold_safe_fast_horizon_no_candidate() -> None:
    # SCENARIO_FOLD_SAFE_FAST_HORIZON_NO_CANDIDATE: selected_horizon is None is
    # the fail-closed branch (gate closed before qualification); the resolver
    # returns the frozen default even if admitted is truthy-adjacent.
    assert _fold_safe_fast_horizon(_selection(None, True), 48) == (
        48, "frozen_default",
    )


def test_fold_safe_fast_horizon_default_horizon_is_customizable() -> None:
    # The default horizon parameter threads the frozen book default through, so
    # the helper stays decoupled from the literal 48.
    assert _fold_safe_fast_horizon(_selection(None, False), 72) == (
        72, "frozen_default",
    )


@pytest.mark.parametrize(
    ("selected", "admitted", "expected"),
    [
        (24, True, (24, "fold_train_only_discovery")),
        (72, True, (72, "fold_train_only_discovery")),
        (168, True, (168, "fold_train_only_discovery")),
    ],
)
def test_fold_safe_fast_horizon_accepts_widened_grid(
    selected: int, admitted: bool, expected: tuple[int, str],
) -> None:
    # The widened REVERSAL_HORIZON_CANDIDATES_HOURS grid is the only source of
    # candidates; every grid member admitted by the gate resolves identically.
    assert _fold_safe_fast_horizon(_selection(selected, admitted), 48) == expected
