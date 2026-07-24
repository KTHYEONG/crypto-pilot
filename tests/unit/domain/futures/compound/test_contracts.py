from __future__ import annotations

import pytest

from src.domain.futures.compound.contracts import ExitPolicyKind, ExitPolicySpec
from src.domain.futures.compound.contracts import L1SleevePosterior


def test_exit_policy_spec_requires_calibration_hash() -> None:
    policy = ExitPolicySpec("time", ExitPolicyKind.TIME, None, None, None, 0, 4, -1, "hash")
    assert policy.kind is ExitPolicyKind.TIME
    with pytest.raises(ValueError, match="calibration_hash"):
        ExitPolicySpec("time", ExitPolicyKind.TIME, None, None, None, 0, 4, -1, "")
    with pytest.raises(ValueError, match="multipliers"):
        ExitPolicySpec("bad", ExitPolicyKind.ASYMMETRIC_ATR, -1.0, 2.0, None, 0, 4, -1, "hash")
    with pytest.raises(ValueError, match="holding"):
        ExitPolicySpec("bad_hold", ExitPolicyKind.TIME, None, None, None, -1, 0, -1, "hash")


def test_l1_sleeve_posterior_rejects_invalid_range() -> None:
    policy = ExitPolicySpec("time", ExitPolicyKind.TIME, None, None, None, 0, 4, -1, "hash")
    with pytest.raises(ValueError, match="posterior range"):
        L1SleevePosterior("s", "sig", "trend", policy, 0.0, -1.0, 0.5, 1.0, (), 0, False, ())
    with pytest.raises(ValueError, match="finite"):
        L1SleevePosterior("s", "sig", "trend", policy, float("nan"), 1.0, 0.5, 1.0, (), 0, False, ())
