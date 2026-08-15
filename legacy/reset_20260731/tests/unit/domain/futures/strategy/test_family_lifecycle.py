from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.strategy.config import (
    CandidateStrategyConfig,
    apply_tf_gate_overrides,
    resolve_family_registration_gap,
    resolve_tf_gate_overrides,
)
from src.domain.futures.strategy.family_lifecycle import is_family_tf_retired

# ─── Scenario 1 (Happy Path) ────────────────────────────────────────────────


def test_is_family_tf_retired_returns_true_for_known_retired_pair() -> None:
    family, tf = "residual_reversion", "4h"

    result = is_family_tf_retired(family, tf)

    assert result is True


def test_is_family_tf_retired_returns_false_for_untested_tf() -> None:
    family, tf = "residual_reversion", "1d"

    result = is_family_tf_retired(family, tf)

    assert result is False


# ─── Scenario 2 (Edge Cases) ─────────────────────────────────────────────────


def test_all_signal_families_identical_across_dual_modules() -> None:
    from src.domain.futures.signals.rules import ALL_SIGNAL_FAMILIES
    from src.domain.futures.strategy.rule_signals import ALL_SIGNAL_FAMILIES as _B

    assert ALL_SIGNAL_FAMILIES == _B, "rules.py vs rule_signals.py family lists have drifted"


def test_resolve_family_registration_gap_is_empty_after_config_update() -> None:
    from src.domain.futures.signals.rules import ALL_SIGNAL_FAMILIES
    from src.domain.futures.strategy.family_lifecycle import RETIRED_FAMILIES

    cfg = CandidateStrategyConfig()

    gap = resolve_family_registration_gap(ALL_SIGNAL_FAMILIES, cfg.candidate_families)

    # RETIRED_FAMILIES are intentionally absent from candidate_families
    # ([ADR_20260713_L0_L1_ASSET_GROWTH_RESTRUCTURE], durable-zero gate pass rate)
    # while their signal-generation code is kept for direct unit-test coverage.
    assert set(gap) == set(RETIRED_FAMILIES)


def test_resolve_tf_gate_overrides_1d_returns_nonempty_dict() -> None:
    cfg = CandidateStrategyConfig()

    overrides = resolve_tf_gate_overrides(cfg, "1d")

    assert overrides != {}
    assert overrides["l1_pair_min_effective_obs"] == 7.0


def _make_synthetic_4h_series(n_days: int) -> tuple[np.ndarray, np.ndarray]:
    n_bars = n_days * 6
    datetimes = pd.date_range("2024-01-01", periods=n_bars, freq="4h").to_numpy()
    values = np.cumsum(np.ones((n_bars, 1)), axis=0)
    return datetimes, values


def test_resample_to_htf_and_project_1d_alias_no_lookahead() -> None:
    from src.domain.futures.signals.rules import _resample_to_htf_and_project

    datetimes_4h, values_4h = _make_synthetic_4h_series(n_days=10)

    projected = _resample_to_htf_and_project(
        datetimes_4h=datetimes_4h,
        values_4h=values_4h,
        htf="1D",
        agg_method="last",
        compute_feature_fn=lambda df: df,
    )

    assert np.isnan(projected[:6]).all()


# ─── Scenario 3 (Error Handling) ─────────────────────────────────────────────


def test_is_family_tf_retired_unknown_family_returns_false_not_raise() -> None:
    assert is_family_tf_retired("nonexistent_family", "4h") is False


# ─── l0_signal_yield_improvement: Track B retirement ─────────────────────────


def test_family_tf_retirement_includes_ichimoku_12h() -> None:
    from src.domain.futures.strategy.family_lifecycle import FAMILY_TF_RETIREMENT

    assert ("ichimoku_trend", "12h") in FAMILY_TF_RETIREMENT


def test_family_tf_retirement_does_not_retire_carry_net_of_funding() -> None:
    assert is_family_tf_retired("carry_net_of_funding", "4h") is False


def test_family_tf_retirement_frozenset_immutable() -> None:
    from src.domain.futures.strategy.family_lifecycle import FAMILY_TF_RETIREMENT

    with pytest.raises(AttributeError):
        FAMILY_TF_RETIREMENT.add(("test", "4h"))  # type: ignore[attr-defined]


def test_liquidity_vacuum_breakout_retired_for_all_tfs() -> None:
    from src.domain.futures.strategy.family_lifecycle import FAMILY_TF_RETIREMENT

    for tf in ("1h", "2h", "4h", "6h", "8h", "12h"):
        assert ("liquidity_vacuum_breakout", tf) in FAMILY_TF_RETIREMENT, (
            f"liquidity_vacuum_breakout not retired for {tf}"
        )


def test_vol_contraction_breakout_not_retired() -> None:
    from src.domain.futures.strategy.family_lifecycle import is_family_tf_retired

    assert is_family_tf_retired("vol_contraction_breakout", "4h") is False


def test_sparse_breakout_retest_v2_retired_for_all_tfs() -> None:
    from src.domain.futures.strategy.family_lifecycle import FAMILY_TF_RETIREMENT

    for tf in ("1h", "2h", "4h", "6h", "8h", "12h"):
        assert ("sparse_breakout_retest_v2", tf) in FAMILY_TF_RETIREMENT, (
            f"sparse_breakout_retest_v2 not retired for {tf}"
        )


def test_sparse_breakout_retest_liquidity_retired_for_all_tfs() -> None:
    from src.domain.futures.strategy.family_lifecycle import FAMILY_TF_RETIREMENT

    for tf in ("1h", "2h", "4h", "6h", "8h", "12h"):
        assert ("sparse_breakout_retest_liquidity", tf) in FAMILY_TF_RETIREMENT, (
            f"sparse_breakout_retest_liquidity not retired for {tf}"
        )


def test_apply_tf_gate_overrides_1d_does_not_mutate_original_cfg() -> None:
    base_cfg = CandidateStrategyConfig()

    patched_cfg = apply_tf_gate_overrides(base_cfg, "1d")

    assert base_cfg.l1_pair_min_effective_obs != patched_cfg.l1_pair_min_effective_obs
    with pytest.raises(dataclasses.FrozenInstanceError):
        base_cfg.l1_pair_min_effective_obs = 999.0  # type: ignore[misc]
