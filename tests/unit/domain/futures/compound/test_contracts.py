from __future__ import annotations

import pytest

import numpy as np

from src.domain.futures.compound.contracts import ExitPolicyKind, ExitPolicySpec
from src.domain.futures.compound.contracts import L1SleevePosterior


def _mask(n: int = 2) -> np.ndarray:
    m = np.zeros(n, dtype=np.bool_)
    m[0] = True
    return m


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
        L1SleevePosterior("s", "sig", "trend", 0, 0, _mask(), "h", policy, 0.0, -1.0, 0.5, 1.0, (), 0, False, ())
    with pytest.raises(ValueError, match="finite"):
        L1SleevePosterior("s", "sig", "trend", 0, 0, _mask(), "h", policy, float("nan"), 1.0, 0.5, 1.0, (), 0, False, ())
    with pytest.raises(ValueError, match="member_hash is required"):
        L1SleevePosterior("s", "sig", "trend", 0, 0, _mask(), "", policy, 0.0, 1.0, 0.5, 1.0, (), 0, False, ())
    with pytest.raises(ValueError, match="at least one True"):
        L1SleevePosterior("s", "sig", "trend", 0, 0, np.zeros(2, dtype=bool), "h", policy, 0.0, 1.0, 0.5, 1.0, (), 0, False, ())
    with pytest.raises(ValueError, match="member_mask_1d must be 1-D"):
        L1SleevePosterior("s", "sig", "trend", 0, 0, np.ones((1, 2), dtype=bool), "h", policy, 0.0, 1.0, 0.5, 1.0, (), 0, False, ())
