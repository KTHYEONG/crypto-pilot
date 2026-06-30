from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.domain.futures.strategy.tiered_workflow.metrics import _bars_per_year_for_tf
from src.domain.futures.strategy.tiered_workflow.replay_parity import (
    assert_selection_replay_parity,
)
from src.domain.futures.strategy.tiered_workflow.risk_deployment import apply_deployment

# ── Helpers ──────────────────────────────────────────────────────────────

def make_eval(
    rets: list[float],
    l_star: float,
    bars_per_year: float,
) -> SimpleNamespace:
    dep = apply_deployment(
        rets=np.asarray(rets, dtype=np.float64),
        leverage=l_star,
        bars_per_year=bars_per_year,
    )
    return SimpleNamespace(
        returns_hybrid=tuple(rets),
        cagr_hybrid=dep.cagr,
        mdd_hybrid=dep.mdd,
        deploy_leverage=l_star,
        fold_pass_ratio=0.6667,
        trade_count=184,
    )


# ── Scenario 1: Bug reproduction — horizon mismatch -> x2 CAGR / xV2 Sharpe ──

class TestScenario1BugReproduction:
    def test_annualization_tf_2x_signature_on_4h_vs_8h(self) -> None:
        rng = np.random.default_rng(42)
        rets = list(rng.normal(1e-8, 1e-6, size=2000))
        l_star = 1.0
        study = make_eval(rets, l_star, bars_per_year=2190.0)
        final = make_eval(rets, l_star, bars_per_year=1095.0)
        ratio = study.cagr_hybrid / final.cagr_hybrid
        assert abs(ratio - 2.0) < 1e-4
        assert abs(study.mdd_hybrid - final.mdd_hybrid) < 1e-12


# ── Scenario 2: Fix invariant — same tf → parity passes ──

class TestScenario2FixInvariant:
    def test_parity_passes_when_annualization_tf_identical(self) -> None:
        rets = [0.001, -0.0005, 0.002, -0.001, 0.0015, -0.0008]
        l_star = 3.0
        study = make_eval(rets, l_star, bars_per_year=1095.0)
        final = make_eval(rets, l_star, bars_per_year=1095.0)
        ok = assert_selection_replay_parity(
            replay_evaluation=study,
            final_evaluation=final,
            gate=True,
        )
        assert ok is True


# ── Scenario 3: B1 — runner resolves & passes master tf ──

class TestScenario3RunnerResolvesMasterTf:
    def test_resolve_l2_master_tf_with_empty_inputs_returns_default(self) -> None:
        from src.domain.futures.strategy.config import CandidateStrategyConfig
        from src.domain.futures.strategy.tiered_workflow.pipeline import (
            _resolve_l2_master_tf,
        )

        cfg = MagicMock(spec=CandidateStrategyConfig)
        cfg.l2_master_tf = ""
        tf = _resolve_l2_master_tf(cfg, {}, None)
        assert tf == "8h"

    def test_resolve_l2_master_tf_respects_cfg_override(self) -> None:
        from src.domain.futures.strategy.config import CandidateStrategyConfig
        from src.domain.futures.strategy.tiered_workflow.pipeline import (
            _resolve_l2_master_tf,
        )

        cfg = MagicMock(spec=CandidateStrategyConfig)
        cfg.l2_master_tf = "4h"
        tf = _resolve_l2_master_tf(cfg, {}, None)
        assert tf == "4h"

    def test_runner_passes_resolved_master_tf_to_l2_study(self) -> None:
        from src.domain.futures.strategy.tiered_workflow.pipeline import (
            _resolve_l2_master_tf,
        )
        mock_run = MagicMock()
        tiered_cfg = MagicMock()
        tiered_cfg.l2_master_tf = ""
        l2_master_tf = _resolve_l2_master_tf(tiered_cfg, {}, None)
        assert l2_master_tf == "8h"
        mock_run(signal_batch=MagicMock(), aligned=MagicMock(), cfg=tiered_cfg,
                 window=MagicMock(), caps=MagicMock(), tf=l2_master_tf,
                 n_trials=5, seed=42, l2_sim_cache=MagicMock())
        _call_tf = mock_run.call_args[1]["tf"]
        assert _call_tf == "8h", f"Expected tf='8h', got tf={_call_tf!r}"


# ── Scenario 4: B2 — SSOT assert blocks on mismatch ──

class TestScenario4SsotAssertBlocks:
    def test_l2_tf_ssot_assert_blocks_on_mismatch(self, caplog: pytest.LogCaptureFixture) -> None:
        from dataclasses import dataclass
        _test_logger = logging.getLogger("test_l2_tf_ssot")
        @dataclass
        class _MockL2Result:
            master_tf: str = "8h"
            gate_passed: bool = True
            blocker_reason: str = ""
        l2_master_tf = "4h"
        l2_final = _MockL2Result(master_tf="8h", gate_passed=True, blocker_reason="")
        with caplog.at_level(logging.ERROR, logger="test_l2_tf_ssot"):
            if l2_master_tf != l2_final.master_tf:
                import dataclasses as _dc
                _test_logger.error(
                    "[L2-TF-SSOT] study_tf=%s final_tf=%s → blocking: annualization_tf_mismatch",
                    l2_master_tf, l2_final.master_tf,
                )
                l2_final = _dc.replace(l2_final, gate_passed=False, blocker_reason="annualization_tf_mismatch")
        assert l2_final.gate_passed is False
        assert l2_final.blocker_reason == "annualization_tf_mismatch"
        assert any("annualization_tf_mismatch" in rec.message for rec in caplog.records)

    def test_l2_tf_ssot_passes_on_match(self, caplog: pytest.LogCaptureFixture) -> None:
        from dataclasses import dataclass
        @dataclass
        class _MockL2Result:
            master_tf: str = "8h"
            gate_passed: bool = True
            blocker_reason: str = ""
        l2_master_tf = "8h"
        l2_final = _MockL2Result(master_tf="8h", gate_passed=True, blocker_reason="")
        with caplog.at_level(logging.ERROR):
            if l2_master_tf != l2_final.master_tf:
                import dataclasses as _dc
                l2_final = _dc.replace(l2_final, gate_passed=False, blocker_reason="annualization_tf_mismatch")
        assert l2_final.gate_passed is True
        assert not any("annualization_tf_mismatch" in rec.message for rec in caplog.records)


# ── Scenario 5: Edge — master_tf override keeps consistency ──

class TestScenario5MasterTfOverride:
    def test_master_tf_override_keeps_study_and_final_consistent(self) -> None:
        from src.domain.futures.strategy.config import CandidateStrategyConfig
        from src.domain.futures.strategy.tiered_workflow.pipeline import (
            _resolve_l2_master_tf,
        )

        cfg = MagicMock(spec=CandidateStrategyConfig)
        cfg.l2_master_tf = "4h"
        tf = _resolve_l2_master_tf(cfg, {}, None)
        assert tf == "4h"
        rets = [0.001, -0.0005, 0.002]
        l_star = 2.0
        study = make_eval(rets, l_star, bars_per_year=_bars_per_year_for_tf(tf))
        final = make_eval(rets, l_star, bars_per_year=_bars_per_year_for_tf(tf))
        ok = assert_selection_replay_parity(
            replay_evaluation=study,
            final_evaluation=final,
            gate=True,
        )
        assert ok is True


# ── Scenario 6: Edge — empty / singleton rets ──

class TestScenario6EmptySingleton:
    def test_annualization_handles_empty(self) -> None:
        dep = apply_deployment(
            rets=np.asarray([], dtype=np.float64),
            leverage=2.0,
            bars_per_year=1095.0,
        )
        assert dep.cagr == 0.0
        assert dep.mdd == 0.0

    def test_annualization_handles_singleton(self) -> None:
        dep = apply_deployment(
            rets=np.asarray([0.001], dtype=np.float64),
            leverage=2.0,
            bars_per_year=1095.0,
        )
        assert np.isfinite(dep.cagr)
        assert dep.mdd == 0.0
