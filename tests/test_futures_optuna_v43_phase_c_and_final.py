from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import optuna

from src.domain.futures.optimization.candidate_selector import (
    select_v43_phase_b_top_candidates,
)
from src.domain.futures.optimization.final_evaluator import (
    _build_ensemble_evaluation_summary,
    _passes_champion_swap_4conditions,
)
from src.domain.futures.optimization.phase_c_robustness import (
    evaluate_phase_c_robustness,
)
from src.domain.futures.optimization.phase_runner import run_v43_phase_optimization_skeleton
from src.domain.futures.validation.unified_gates import (
    FuturesResearchGateInput,
    evaluate_research_gates,
)


def test_phase_c_robustness_scaffold_schema() -> None:
    study = optuna.create_study(direction="minimize")

    def _obj(trial):
        x = trial.suggest_float("x", 0.0, 1.0)
        return (x - 0.2) ** 2

    study.optimize(_obj, n_trials=4)

    diag = evaluate_phase_c_robustness(study_b=study, target_seeds=[7, 11], top_k=3)
    assert diag["phase"] == "phase_c"
    assert 0.0 <= float(diag["robustness_score"]) <= 1.0
    assert float(diag["stability_cv"]) >= 0.0
    assert 0.0 <= float(diag["pbo_candidate"]) <= 1.0
    assert 0.0 <= float(diag["dsr_proxy"]) <= 1.0
    stress = diag["stress_diagnostics"]
    assert stress["schema_version"] == "v43.phase_c.1"
    assert stress["method"] in {
        "salib_sobol",
        "deterministic_perturbation_fallback",
    }
    assert int(stress["candidate_count"]) >= 1
    assert int(stress["seed_count"]) == 2
    assert "stress" in stress
    assert stress["stress"]["status"] == "placeholder_structured"


@dataclass
class _DummyStudy:
    study_name: str


def test_phase_runner_includes_phase_c_diagnostics(monkeypatch) -> None:
    def _fake_loop(**kwargs):
        return _DummyStudy(study_name=kwargs["study_name"])

    def _fake_phase_c(*, study_b, target_seeds, top_k):
        assert target_seeds == [101, 202]
        assert top_k == 5
        return {"phase": "phase_c", "robustness_score": 0.5, "stability_cv": 0.1}

    monkeypatch.setattr(
        "src.domain.futures.optimization.phase_runner.run_optimization_loop", _fake_loop
    )
    monkeypatch.setattr(
        "src.domain.futures.optimization.phase_runner.evaluate_phase_c_robustness",
        _fake_phase_c,
    )

    bundle = run_v43_phase_optimization_skeleton(
        base_ctx=SimpleNamespace(run_id="r-phase-c"),
        base_study_name="base_phase",
        storage_url="sqlite:///tmp.db",
        storage=None,
        n_trials=2,
        seed=42,
        resume=False,
        n_workers=1,
        enqueue_seeds=None,
        target_seeds=[101, 202],
    )
    assert bundle.phase_c_diagnostics is not None
    assert bundle.phase_c_diagnostics["phase"] == "phase_c"


def test_expectancy_retention_gate_blocks_when_below_floor() -> None:
    gate_ok, failures = evaluate_research_gates(
        FuturesResearchGateInput(
            phase3_enabled=False,
            pbo_max=0.6,
            dsr_min=0.0,
            is_precision=0.55,
            oos_port={},
            pbo_obs=0.1,
            dsr_obs=0.1,
            wf_failures=(),
            min_is_net_alpha_pct=-999.0,
            is_net_alpha_pct=0.0,
            min_long_pf=0.0,
            min_short_pf=0.0,
            oos_long_pf=2.0,
            oos_short_pf=2.0,
            is_cagr_pct=30.0,
            is_sharpe=2.0,
            is_survival_min_cagr=0.0,
            is_survival_min_sharpe=0.0,
            worst_leg_log_tw=1.0,
            awf_p10_log_tw_floor=0.0,
            oos_mdd_duration=30.0,
            max_mdd_duration=180.0,
            oos_expectancy=0.20,
            min_expectancy=0.10,
            is_expectancy=0.50,
            min_oos_retention_expectancy_pct=50.0,
            oos_cagr_pct=100.0,   # Auxiliary only: should not bypass expectancy retention.
            is_cagr_ref_pct=10.0,
        )
    )
    assert gate_ok is False
    assert "OOS_RETENTION_EXPECTANCY_GATE" in failures


def test_champion_swap_4conditions_are_strict() -> None:
    ok, failures = _passes_champion_swap_4conditions(
        gate_ok=True,
        new_m={
            "cagr": 20.0,
            "calmar": 1.8,
            "sortino": 2.2,
            "tw": 1.2,
            "mdd": 10.0,
            "pbo": 8.0,
            "avg_pnl": 0.70,
            "oos_retention_expectancy_pct": 70.0,
        },
        champ_m={
            "cagr": 20.0,
            "calmar": 1.8,
            "sortino": 2.2,
            "tw": 1.2,
            "mdd": 10.0,
            "pbo": 8.0,  # equal is not superior
            "avg_pnl": 0.70,
            "oos_retention_expectancy_pct": 70.0,
        },
        cand_ev_cost_ratio=0.40,
        champ_ev_cost_ratio=0.40,
    )
    assert ok is False
    assert "CHAMP_SWAP_PBO_NOT_SUPERIOR" in failures


def test_ensemble_summary_includes_meta_and_members() -> None:
    members = [
        {"trial": SimpleNamespace(number=3)},
        {"trial": SimpleNamespace(number=7)},
    ]
    ports = [
        {"cagr_pct": 10.0, "mdd_pct": 5.0, "terminal_wealth_ratio": 1.1, "avg_trade_pnl_pct": 0.6},
        {"cagr_pct": 12.0, "mdd_pct": 6.0, "terminal_wealth_ratio": 1.2, "avg_trade_pnl_pct": 0.7},
    ]
    meta = {"cagr_pct": 11.0, "mdd_pct": 4.5, "terminal_wealth_ratio": 1.15, "avg_trade_pnl_pct": 0.65}
    summary = _build_ensemble_evaluation_summary(
        selected_ensemble_results=members,
        ensemble_ports=ports,
        meta_port=meta,
    )
    assert summary["selected_count"] == 2
    assert len(summary["members"]) == 2
    assert summary["members"][0]["trial_number"] == 3
    assert summary["ensemble_meta"]["cagr_pct"] == 11.0


def test_phase_b_top_candidates_prioritize_calmar_and_constraints(monkeypatch) -> None:
    study = optuna.create_study(direction="maximize", study_name="demo_phase_b")

    def _obj(trial):
        trial.suggest_float("x", 0.0, 1.0)
        return float(trial.number)

    study.optimize(_obj, n_trials=4)
    trials = study.get_trials(deepcopy=False)

    # t0: feasible (calmar 1.6)
    trials[0].set_user_attr("phase", "phase_b")
    trials[0].set_user_attr("calmar_lcb", 1.6)
    trials[0].set_user_attr("ev_cost_ratio", 3.2)
    trials[0].set_user_attr("funding_drag_ratio", 0.20)
    trials[0].set_user_attr("cagr_lcb", 35.0)
    trials[0].set_user_attr("sortino_lcb", 2.0)
    trials[0].set_user_attr("mdd_ucb", 18.0)
    trials[0].set_user_attr("mdd_duration", 120.0)
    trials[0].set_user_attr("cvar", 18.0)
    trials[0].set_user_attr("minority_side_ratio", 0.20)

    # t1: infeasible (calmar 1.4 < 1.5) even though trial.value is high
    trials[1].set_user_attr("phase", "phase_b")
    trials[1].set_user_attr("calmar_lcb", 1.4)
    trials[1].set_user_attr("ev_cost_ratio", 3.2)
    trials[1].set_user_attr("funding_drag_ratio", 0.20)
    trials[1].set_user_attr("cagr_lcb", 35.0)
    trials[1].set_user_attr("sortino_lcb", 2.0)
    trials[1].set_user_attr("mdd_ucb", 18.0)
    trials[1].set_user_attr("mdd_duration", 120.0)
    trials[1].set_user_attr("cvar", 18.0)
    trials[1].set_user_attr("minority_side_ratio", 0.20)

    # t2: feasible and best calmar (1.9)
    trials[2].set_user_attr("phase", "phase_b")
    trials[2].set_user_attr("calmar_lcb", 1.9)
    trials[2].set_user_attr("ev_cost_ratio", 3.5)
    trials[2].set_user_attr("funding_drag_ratio", 0.18)
    trials[2].set_user_attr("cagr_lcb", 40.0)
    trials[2].set_user_attr("sortino_lcb", 2.1)
    trials[2].set_user_attr("mdd_ucb", 17.0)
    trials[2].set_user_attr("mdd_duration", 100.0)
    trials[2].set_user_attr("cvar", 20.0)
    trials[2].set_user_attr("minority_side_ratio", 0.22)

    # t3: not phase_b, should be ignored for v43 phase-b ranking
    trials[3].set_user_attr("phase", "phase_a2")
    trials[3].set_user_attr("calmar_lcb", 2.5)

    def _fake_replay(_ctx, _params):
        return 0.0, {"awf_pos_frac": 0.5}

    monkeypatch.setattr(
        "src.domain.futures.optimization.candidate_selector.replay_robust_awf_for_trial_params",
        _fake_replay,
    )

    top, summary = select_v43_phase_b_top_candidates(
        study,
        base_ctx=SimpleNamespace(),
        cfg={},
        top_k=2,
    )
    assert summary["selected_by"] == "v43_phase_b_calmar_lcb"
    assert summary["feasible_count"] == 2
    assert summary["selected_count"] == 2
    assert len(top) == 2
    assert str(top[0]["trial"].user_attrs.get("phase", "")).lower() == "phase_b"
    assert str(top[1]["trial"].user_attrs.get("phase", "")).lower() == "phase_b"
    assert float(top[0]["trial"].user_attrs["calmar_lcb"]) >= float(top[1]["trial"].user_attrs["calmar_lcb"])
