"""L0 Readiness Hardening — 4 fixes (Fix 1-4) unit tests."""

from __future__ import annotations

import logging
from typing import Any

import pytest

from src.domain.futures.strategy.config import (
    _DEFAULT_PER_TF_FAMILIES,
    DEPRIORITIZED_FAMILY_PRIOR,
)
from src.domain.futures.strategy.tiered_workflow.pipeline import _log_pertf_registry_diag

# ─── Fix 1: L1-PERTF-REGISTRY-DIAG blockers logging ─────────────────────


def test_fix1_logs_blockers_when_gate_fails(caplog: Any) -> None:
    from unittest.mock import MagicMock

    from src.domain.futures.strategy.candidate_contracts import Layer1GateReport
    from src.domain.futures.strategy.config import PerTfL1Result
    from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer1Result

    caplog.set_level(logging.DEBUG)
    gate_report = Layer1GateReport(
        checks=(), passed=False, blockers=("fold_ratio:0.400", "probe_lcb_bps:-1.200")
    )
    l1_result = MagicMock(spec=Layer1Result)
    l1_result.gate_report = gate_report
    l1_result.gate_passed = False
    l1_result.deployment_registry = None
    l1_result.strategy_panel = ()
    per_tf = {"4h": PerTfL1Result(tf="4h", l1_result=l1_result, n_winning_signals=0)}

    _log_pertf_registry_diag(per_tf, l2_tf_resolved="12h")

    assert "blockers=fold_ratio:0.400,probe_lcb_bps:-1.200" in caplog.text


def test_fix1_logs_blockers_none_when_all_pass(caplog: Any) -> None:
    from unittest.mock import MagicMock

    from src.domain.futures.strategy.candidate_contracts import Layer1GateReport
    from src.domain.futures.strategy.config import PerTfL1Result
    from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer1Result

    caplog.set_level(logging.DEBUG)
    gate_report = Layer1GateReport(checks=(), passed=True, blockers=())
    l1_result = MagicMock(spec=Layer1Result)
    l1_result.gate_report = gate_report
    l1_result.gate_passed = True
    l1_result.deployment_registry = None
    l1_result.strategy_panel = ()
    per_tf = {"4h": PerTfL1Result(tf="4h", l1_result=l1_result, n_winning_signals=0)}

    _log_pertf_registry_diag(per_tf, l2_tf_resolved="12h")

    assert "blockers=none" in caplog.text


def test_fix1_logs_each_tf_once(caplog: Any) -> None:
    from unittest.mock import MagicMock

    from src.domain.futures.strategy.candidate_contracts import Layer1GateReport
    from src.domain.futures.strategy.config import PerTfL1Result
    from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer1Result

    caplog.set_level(logging.DEBUG)
    gate_report = Layer1GateReport(checks=(), passed=True, blockers=())
    per_tf: dict[str, PerTfL1Result] = {}
    for tf in ("4h", "6h", "8h", "12h"):
        l1_result = MagicMock(spec=Layer1Result)
        l1_result.gate_report = gate_report
        l1_result.gate_passed = True
        l1_result.deployment_registry = None
        l1_result.strategy_panel = ()
        per_tf[tf] = PerTfL1Result(tf=tf, l1_result=l1_result, n_winning_signals=0)

    _log_pertf_registry_diag(per_tf, l2_tf_resolved="12h")

    lines = [r.message for r in caplog.records if "gate_passed=True" in r.message and "blockers=" in r.message]
    assert len(lines) == 4, f"Expected 4 lines, got {len(lines)}: {caplog.text}"
    for tf in ("4h", "6h", "8h", "12h"):
        assert any(f"tf={tf}" in msg for msg in lines)


# ─── Fix 2: DEPRIORITIZED_FAMILY_PRIOR evidence-contradicted removals ────


def test_fix2_excludes_evidence_contradicted_families() -> None:
    assert "vol_term_structure_gate" not in DEPRIORITIZED_FAMILY_PRIOR
    assert "trend_donchian" not in DEPRIORITIZED_FAMILY_PRIOR


def test_fix2_retains_negative_prior_for_persistently_poor_families() -> None:
    assert DEPRIORITIZED_FAMILY_PRIOR["carry_net_of_funding"] == pytest.approx(-0.5)
    assert DEPRIORITIZED_FAMILY_PRIOR["taker_imbalance_momentum"] == pytest.approx(-0.5)
    assert DEPRIORITIZED_FAMILY_PRIOR["supertrend"] == pytest.approx(-0.5)
    assert DEPRIORITIZED_FAMILY_PRIOR["funding_flow_carry"] == pytest.approx(-0.3)


def test_fix2_all_remaining_prior_scores_negative() -> None:
    assert all(v < 0 for v in DEPRIORITIZED_FAMILY_PRIOR.values())


# ─── Fix 3: vol_breakout added to untested TFs ──────────────────────────


@pytest.mark.parametrize("tf", ["6h", "8h", "12h"])
def test_fix3_vol_breakout_added_to_untested_timeframes(tf: str) -> None:
    assert "vol_breakout" in _DEFAULT_PER_TF_FAMILIES[tf]


def test_fix3_vol_breakout_absent_from_retired_4h() -> None:
    assert "vol_breakout" not in _DEFAULT_PER_TF_FAMILIES["4h"]
    from src.domain.futures.strategy.family_lifecycle import is_family_tf_retired
    assert is_family_tf_retired("vol_breakout", "4h") is True


# ─── Fix 4: L0_CROSS_TF_PRUNING default to True (opt-out) ────────────────


def test_fix4_pruning_default_enabled(monkeypatch: Any) -> None:
    monkeypatch.delenv("L0_CROSS_TF_PRUNING", raising=False)
    from src.application.futures.runner.config import _l0_cross_tf_pruning_enabled
    assert _l0_cross_tf_pruning_enabled() is True


@pytest.mark.parametrize("val", ["0", "false", "False"])
def test_fix4_pruning_explicit_opt_out(monkeypatch: Any, val: str) -> None:
    monkeypatch.setenv("L0_CROSS_TF_PRUNING", val)
    from src.application.futures.runner.config import _l0_cross_tf_pruning_enabled
    assert _l0_cross_tf_pruning_enabled() is False


def test_fix4_pruning_truthy_values_enable(monkeypatch: Any) -> None:
    monkeypatch.setenv("L0_CROSS_TF_PRUNING", "1")
    from src.application.futures.runner.config import _l0_cross_tf_pruning_enabled
    assert _l0_cross_tf_pruning_enabled() is True
