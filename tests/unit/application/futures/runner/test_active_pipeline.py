from __future__ import annotations

import contextlib
import inspect
from dataclasses import dataclass
from datetime import date
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from src.application.futures.runner.active_pipeline import (
    _has_l1_delivery_candidates,
    _resolve_effective_evidence_start,
    _run_strategy_stage,
)


@dataclass
class _FakeDatetime:
    _d: date

    def date(self) -> date:
        return self._d


@dataclass
class _FakeWindow:
    effective_from: _FakeDatetime
    active_symbols: frozenset[str]


def _win(d: date, syms: frozenset[str]) -> _FakeWindow:
    return _FakeWindow(effective_from=_FakeDatetime(d), active_symbols=syms)


def test_effective_evidence_start_requires_two_consecutive_stable_quarters() -> None:
    windows = [
        _win(date(2023, 4, 1), frozenset()),
        _win(date(2023, 7, 1), frozenset(f"SYM{i}" for i in range(60))),
        _win(date(2023, 10, 1), frozenset()),
        _win(date(2024, 1, 1), frozenset(f"SYM{i}" for i in range(55))),
        _win(date(2024, 4, 1), frozenset(f"SYM{i}" for i in range(58))),
    ]

    result = _resolve_effective_evidence_start(
        tf="4h",
        timeline_windows=windows,
        data_start=date(2023, 4, 29),
        regime_floor=date(2023, 1, 1),
        min_universe_size=50,
        membership_warmup_days=10,
    )

    assert result == date(2024, 1, 1)


def test_effective_evidence_start_respects_membership_warmup_days() -> None:
    windows = [
        _win(date(2023, 4, 1), frozenset(f"SYM{i}" for i in range(60))),
        _win(date(2023, 7, 1), frozenset(f"SYM{i}" for i in range(60))),
    ]

    result = _resolve_effective_evidence_start(
        tf="4h",
        timeline_windows=windows,
        data_start=date(2023, 4, 29),
        regime_floor=date(2023, 1, 1),
        min_universe_size=50,
        membership_warmup_days=90,
    )

    assert result > date(2023, 7, 1)


def test_effective_evidence_start_raises_when_never_stable() -> None:
    windows = [_win(date(2023, 4, 1), frozenset()), _win(date(2023, 7, 1), frozenset({"BTCUSDT"}))]

    with pytest.raises(ValueError, match="never reaches"):
        _resolve_effective_evidence_start(
            tf="4h",
            timeline_windows=windows,
            data_start=date(2023, 4, 29),
            regime_floor=date(2023, 1, 1),
            min_universe_size=50,
            membership_warmup_days=10,
        )


def test_strategy_stage_wires_causal_cutoff_and_delivery_manifest() -> None:
    """The runner must forward both sides of the L0→L1 delivery contract."""
    source = inspect.getsource(_run_strategy_stage)

    assert "l0_evidence_end=l0_evidence_end" in source
    assert "l0_delivery_manifest=l0_delivery_manifest" in source
    assert "consume_candidate_output_for_tiered(" in source


def test_has_l1_delivery_candidates_uses_multi_tf_manifest_not_base_report() -> None:
    """HTF L0 candidates must not be discarded when the base-TF report is empty."""
    output = SimpleNamespace(
        l0_delivery_manifest=SimpleNamespace(final_selected_recipe_ids=("recipe:12h",)),
    )

    assert _has_l1_delivery_candidates(output)
    assert not _has_l1_delivery_candidates(SimpleNamespace(l0_delivery_manifest=None))


def test_tiered_labeled_events_marks_unrouted_events_with_empty_l0_recipe_id() -> None:
    """Unrouted events must be filtered by the L0 manifest, not crash L1."""
    from src.application.futures.runner.active_pipeline import _tiered_labeled_events

    source = SimpleNamespace(labeled_unfiltered=pd.DataFrame({"native_tf": ["4h"]}))

    labeled = _tiered_labeled_events(cast(Any, source))

    assert labeled["l0_recipe_id"].tolist() == [""]


def test_build_l1_swf_folds_unaffected_by_l2_override() -> None:
    """_run_tiered_l2_study does not call build_l1_swf_folds."""
    import inspect
    from src.application.futures.runner.active_pipeline import _run_tiered_l2_study
    source = inspect.getsource(_run_tiered_l2_study)
    assert "build_l1_swf_folds" not in source


class TestRunTieredL2StudyFoldOverride:
    def test_run_tiered_l2_study_uses_override_fold_cfg_not_original_cfg(self, mocker) -> None:
        from src.application.futures.runner.active_pipeline import _run_tiered_l2_study
        from src.domain.futures.strategy.config import CandidateStrategyConfig

        cfg = CandidateStrategyConfig(wf_n_folds=4)
        spy = mocker.patch(
            "src.domain.futures.strategy.walk_forward.build_walk_forward_folds",
            return_value=(),
        )
        mocker.patch(
            "src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache",
            return_value=mocker.MagicMock(),
        )
        mocker.patch(
            "src.domain.futures.strategy.tiered_workflow.awf_sim.compute_per_tf_fit_edge",
            return_value={},
        )
        mocker.patch(
            "src.domain.futures.strategy.market_regime.compute_market_regime_context",
            return_value=mocker.MagicMock(),
        )
        mocker.patch(
            "src.domain.futures.strategy.tiered_workflow.l2_meta.build_regime_routing_plan",
            return_value=mocker.MagicMock(),
        )
        mocker.patch(
            "src.domain.futures.strategy.tiered_workflow.diagnostics.build_layer_universe_audit",
            return_value=mocker.MagicMock(warnings=()),
        )
        mocker.patch(
            "src.domain.futures.strategy.market_regime.compute_risk_severity_code",
            return_value=mocker.MagicMock(),
        )
        mocker.patch(
            "src.domain.futures.optimization.workflow.objective_l2_growth",
            return_value=0.0,
        )
        mocker.patch(
            "src.domain.futures.strategy.tiered_workflow.selection.select_layer2_champion",
            return_value=mocker.MagicMock(
                best_params={}, best_trial_number=None, completed_trials=0,
            ),
        )
        mocker.patch(
            "src.application.futures.runner.active_pipeline._get_rss_mb",
            return_value=100.0,
        )
        import dataclasses
        mocker.patch.object(
            dataclasses,
            "replace",
            side_effect=lambda obj, **kw: obj,
        )

        from datetime import date
        window = SimpleNamespace(
            holdout_start=date(2025, 6, 1),
            l2_start=date(2024, 6, 1),
        )
        aligned = SimpleNamespace(
            symbols=("BTCUSDT",),
            close_2d=mocker.MagicMock(),
            datetimes=pd.date_range("2024-01-01", periods=500, freq="h"),
        )
        caps = SimpleNamespace(trial_number=0)
        signal_batch = mocker.MagicMock()
        signal_batch.start_idx = 0
        signal_batch.end_idx = 500
        signal_batch.registry_version = "v1"
        signal_batch.model_version = "v1"
        signal_batch.events = ()

        _run_tiered_l2_study(
            signal_batch=signal_batch,
            aligned=aligned,
            cfg=cfg,
            window=window,
            caps=caps,
            tf="1h",
            n_trials=2,
            seed=42,
            l2_sim_cache=mocker.MagicMock(),
            l2_wf_n_folds=8,
        )

        assert spy.called
        called_cfg = spy.call_args.kwargs["cfg"]
        assert called_cfg.wf_n_folds == 8
        assert cfg.wf_n_folds == 4

    def test_run_tiered_l2_study_defaults_to_layer2_allocation_config_value(self, mocker) -> None:
        from src.application.futures.runner.active_pipeline import _run_tiered_l2_study
        from src.domain.futures.strategy.config import CandidateStrategyConfig
        from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer2AllocationConfig

        cfg = CandidateStrategyConfig(wf_n_folds=4)
        spy = mocker.patch(
            "src.domain.futures.strategy.walk_forward.build_walk_forward_folds",
            return_value=(),
        )
        mocker.patch(
            "src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache",
            return_value=mocker.MagicMock(),
        )
        mocker.patch(
            "src.domain.futures.strategy.tiered_workflow.awf_sim.compute_per_tf_fit_edge",
            return_value={},
        )
        mocker.patch(
            "src.domain.futures.strategy.market_regime.compute_market_regime_context",
            return_value=mocker.MagicMock(),
        )
        mocker.patch(
            "src.domain.futures.strategy.tiered_workflow.l2_meta.build_regime_routing_plan",
            return_value=mocker.MagicMock(),
        )
        mocker.patch(
            "src.domain.futures.strategy.tiered_workflow.diagnostics.build_layer_universe_audit",
            return_value=mocker.MagicMock(warnings=()),
        )
        mocker.patch(
            "src.domain.futures.strategy.market_regime.compute_risk_severity_code",
            return_value=mocker.MagicMock(),
        )
        mocker.patch(
            "src.domain.futures.optimization.workflow.objective_l2_growth",
            return_value=0.0,
        )
        mocker.patch(
            "src.domain.futures.strategy.tiered_workflow.selection.select_layer2_champion",
            return_value=mocker.MagicMock(
                best_params={}, best_trial_number=None, completed_trials=0,
            ),
        )
        mocker.patch(
            "src.application.futures.runner.active_pipeline._get_rss_mb",
            return_value=100.0,
        )
        import dataclasses
        mocker.patch.object(
            dataclasses,
            "replace",
            side_effect=lambda obj, **kw: obj,
        )

        from datetime import date
        window = SimpleNamespace(
            holdout_start=date(2025, 6, 1),
            l2_start=date(2024, 6, 1),
        )
        aligned = SimpleNamespace(
            symbols=("BTCUSDT",),
            close_2d=mocker.MagicMock(),
            datetimes=pd.date_range("2024-01-01", periods=500, freq="h"),
        )
        caps = SimpleNamespace(trial_number=0)
        signal_batch = mocker.MagicMock()
        signal_batch.start_idx = 0
        signal_batch.end_idx = 500
        signal_batch.registry_version = "v1"
        signal_batch.model_version = "v1"
        signal_batch.events = ()

        _run_tiered_l2_study(
            signal_batch=signal_batch,
            aligned=aligned,
            cfg=cfg,
            window=window,
            caps=caps,
            tf="1h",
            n_trials=2,
            seed=42,
            l2_sim_cache=mocker.MagicMock(),
            l2_wf_n_folds=None,
        )

        assert spy.called
        called_cfg = spy.call_args.kwargs["cfg"]
        assert called_cfg.wf_n_folds == Layer2AllocationConfig().l2_wf_n_folds
        assert cfg.wf_n_folds == 4

    def test_active_pipeline_scoped_fold_override_falls_back_to_layer2_config_default(self, mocker) -> None:
        from src.application.futures.runner.active_pipeline import _run_tiered_l2_study
        from src.domain.futures.strategy.config import CandidateStrategyConfig

        cfg = CandidateStrategyConfig(wf_n_folds=4)
        routing_spy = mocker.patch(
            "src.domain.futures.strategy.tiered_workflow.l2_meta.build_regime_routing_plan",
            return_value=mocker.MagicMock(),
        )
        mocker.patch(
            "src.domain.futures.strategy.tiered_workflow.diagnostics.build_layer_universe_audit",
            return_value=mocker.MagicMock(warnings=()),
        )
        mocker.patch(
            "src.domain.futures.strategy.market_regime.compute_market_regime_context",
            return_value=mocker.MagicMock(),
        )
        mocker.patch(
            "src.domain.futures.strategy.market_regime.compute_risk_severity_code",
            return_value=mocker.MagicMock(),
        )
        mocker.patch(
            "src.domain.futures.strategy.tiered_workflow.awf_sim.compute_per_tf_fit_edge",
            return_value={},
        )
        mocker.patch(
            "src.domain.futures.strategy.walk_forward.build_walk_forward_folds",
            return_value=(),
        )
        mocker.patch(
            "src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache",
            return_value=mocker.MagicMock(),
        )
        mocker.patch(
            "src.domain.futures.optimization.workflow.objective_l2_growth",
            return_value=0.0,
        )
        mocker.patch(
            "src.domain.futures.strategy.tiered_workflow.selection.select_layer2_champion",
            return_value=mocker.MagicMock(
                best_params={}, best_trial_number=None, completed_trials=0,
            ),
        )
        mocker.patch(
            "src.application.futures.runner.active_pipeline._get_rss_mb",
            return_value=100.0,
        )
        import dataclasses
        mocker.patch.object(
            dataclasses,
            "replace",
            side_effect=lambda obj, **kw: obj,
        )

        from datetime import date
        window = SimpleNamespace(
            holdout_start=date(2025, 6, 1),
            l2_start=date(2024, 6, 1),
        )
        aligned = SimpleNamespace(
            symbols=("BTCUSDT",),
            close_2d=mocker.MagicMock(),
            datetimes=pd.date_range("2024-01-01", periods=500, freq="h"),
        )
        caps = SimpleNamespace(trial_number=0)
        signal_batch = mocker.MagicMock()
        signal_batch.start_idx = 0
        signal_batch.end_idx = 500
        signal_batch.registry_version = "v1"
        signal_batch.model_version = "v1"
        signal_batch.events = ()

        _run_tiered_l2_study(
            signal_batch=signal_batch,
            aligned=aligned,
            cfg=cfg,
            window=window,
            caps=caps,
            tf="1h",
            n_trials=2,
            seed=42,
            l2_sim_cache=mocker.MagicMock(),
            l2_wf_n_folds=8,
        )

        result = _run_tiered_l2_study(
            signal_batch=signal_batch,
            aligned=aligned,
            cfg=cfg,
            window=window,
            caps=caps,
            tf="1h",
            n_trials=2,
            seed=42,
            l2_sim_cache=mocker.MagicMock(),
            l2_wf_n_folds=8,
        )

        assert result.blocker_reason == "insufficient_causal_l2_span"
        assert not routing_spy.called

    def test_run_tiered_l2_study_forwards_crisis_data_to_champion_selection(self, mocker) -> None:
        """[SPEC_L2_CHAMPION_SELECTION_CRISIS_BLINDNESS_FIX][S4] _run_tiered_l2_study가
        자신의 crisis_rets/crisis_replay_ctx를 select_layer2_champion에 그대로 전달함을 검증."""
        from src.application.futures.runner.active_pipeline import _run_tiered_l2_study
        from src.domain.futures.strategy.config import CandidateStrategyConfig

        cfg = CandidateStrategyConfig(wf_n_folds=4)
        _folds = (
            SimpleNamespace(fit_start=0, fit_end=80, cal_start=40, cal_end=80, oos_start=100, oos_end=200),
            SimpleNamespace(fit_start=0, fit_end=160, cal_start=120, cal_end=160, oos_start=200, oos_end=300),
            SimpleNamespace(fit_start=0, fit_end=260, cal_start=220, cal_end=260, oos_start=300, oos_end=400),
        )
        mocker.patch(
            "src.domain.futures.strategy.walk_forward.build_walk_forward_folds",
            return_value=_folds,
        )
        mocker.patch(
            "src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache",
            return_value=mocker.MagicMock(),
        )
        mocker.patch(
            "src.domain.futures.strategy.tiered_workflow.awf_sim.compute_per_tf_fit_edge",
            return_value={},
        )
        mocker.patch(
            "src.domain.futures.strategy.market_regime.compute_market_regime_context",
            return_value=mocker.MagicMock(),
        )
        mocker.patch(
            "src.domain.futures.strategy.tiered_workflow.l2_meta.build_regime_routing_plan",
            return_value=mocker.MagicMock(),
        )
        mocker.patch(
            "src.domain.futures.strategy.tiered_workflow.diagnostics.build_layer_universe_audit",
            return_value=mocker.MagicMock(warnings=()),
        )
        mocker.patch(
            "src.domain.futures.strategy.market_regime.compute_risk_severity_code",
            return_value=mocker.MagicMock(),
        )
        mocker.patch(
            "src.domain.futures.optimization.workflow.objective_l2_growth",
            return_value=0.0,
        )
        # 순차(n_jobs=1, in-process) 경로를 config로 강제 — subprocess pickling으로
        # 인해 objective_l2_growth mock이 미적용되는 것을 방지.
        from src.domain.futures.optimization.opt_config import OPT_FUTURES_CONFIG
        mocker.patch.dict(OPT_FUTURES_CONFIG, {"L2_OPTUNA_BATCH_SIZE": 1})
        champion_spy = mocker.patch(
            "src.domain.futures.strategy.tiered_workflow.selection.select_layer2_champion",
            return_value=mocker.MagicMock(
                best_params={}, best_trial_number=None, completed_trials=0,
            ),
        )
        mocker.patch(
            "src.application.futures.runner.active_pipeline._get_rss_mb",
            return_value=100.0,
        )
        import dataclasses
        mocker.patch.object(
            dataclasses,
            "replace",
            side_effect=lambda obj, **kw: obj,
        )
        # [l2-crisis-replay-routing-parity] transfer is a no-op in this test context
        mocker.patch(
            "src.domain.futures.strategy.tiered_workflow.l2_meta.transfer_routing_plan_to_crisis_cache",
            side_effect=lambda crisis_cache, study_cache, **kw: crisis_cache,
        )

        from datetime import date
        window = SimpleNamespace(
            holdout_start=date(2024, 1, 19),
            l2_start=date(2024, 1, 5),
        )
        aligned = SimpleNamespace(
            symbols=("BTCUSDT",),
            close_2d=mocker.MagicMock(),
            datetimes=pd.date_range("2024-01-01", periods=500, freq="h"),
        )
        caps = SimpleNamespace(trial_number=0)
        signal_batch = mocker.MagicMock()
        signal_batch.start_idx = 0
        signal_batch.end_idx = 500
        signal_batch.registry_version = "v1"
        signal_batch.model_version = "v1"
        signal_batch.events = ()

        fake_crisis_rets = mocker.MagicMock()
        fake_crisis_replay_ctx = mocker.MagicMock()

        _run_tiered_l2_study(
            signal_batch=signal_batch,
            aligned=aligned,
            cfg=cfg,
            window=window,
            caps=caps,
            tf="1h",
            n_trials=2,
            seed=42,
            l2_sim_cache=mocker.MagicMock(),
            crisis_rets=fake_crisis_rets,
            crisis_replay_ctx=fake_crisis_replay_ctx,
        )

        assert champion_spy.called
        assert champion_spy.call_args.kwargs["crisis_rets"] is fake_crisis_rets
        assert champion_spy.call_args.kwargs["crisis_replay_ctx"] is fake_crisis_replay_ctx

    def test_run_tiered_l2_study_enriches_crisis_ctx_with_routed_cache(self, mocker) -> None:
        """[SPEC l2-crisis-replay-routing-parity][S4] parity flag(기본 True) 시
        transfer_routing_plan_to_crisis_cache가 호출되고 crisis_replay_ctx.cache가
        교체됨을 검증."""
        from src.application.futures.runner.active_pipeline import _run_tiered_l2_study
        from src.domain.futures.strategy.config import CandidateStrategyConfig

        cfg = CandidateStrategyConfig(wf_n_folds=4)
        _folds = (
            SimpleNamespace(fit_start=0, fit_end=80, cal_start=40, cal_end=80, oos_start=100, oos_end=200),
            SimpleNamespace(fit_start=0, fit_end=160, cal_start=120, cal_end=160, oos_start=200, oos_end=300),
            SimpleNamespace(fit_start=0, fit_end=260, cal_start=220, cal_end=260, oos_start=300, oos_end=400),
        )
        mocker.patch(
            "src.domain.futures.strategy.walk_forward.build_walk_forward_folds",
            return_value=_folds,
        )
        mocker.patch(
            "src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache",
            return_value=mocker.MagicMock(),
        )
        mocker.patch(
            "src.domain.futures.strategy.tiered_workflow.awf_sim.compute_per_tf_fit_edge",
            return_value={},
        )
        mocker.patch(
            "src.domain.futures.strategy.market_regime.compute_market_regime_context",
            return_value=mocker.MagicMock(),
        )
        mocker.patch(
            "src.domain.futures.strategy.tiered_workflow.l2_meta.build_regime_routing_plan",
            return_value=mocker.MagicMock(),
        )
        mocker.patch(
            "src.domain.futures.strategy.tiered_workflow.diagnostics.build_layer_universe_audit",
            return_value=mocker.MagicMock(warnings=()),
        )
        mocker.patch(
            "src.domain.futures.strategy.market_regime.compute_risk_severity_code",
            return_value=mocker.MagicMock(),
        )
        mocker.patch(
            "src.domain.futures.optimization.workflow.objective_l2_growth",
            return_value=0.0,
        )
        from src.domain.futures.optimization.opt_config import OPT_FUTURES_CONFIG
        mocker.patch.dict(OPT_FUTURES_CONFIG, {"L2_OPTUNA_BATCH_SIZE": 1})
        mocker.patch(
            "src.domain.futures.strategy.tiered_workflow.selection.select_layer2_champion",
            return_value=mocker.MagicMock(
                best_params={}, best_trial_number=None, completed_trials=0,
            ),
        )
        mocker.patch(
            "src.application.futures.runner.active_pipeline._get_rss_mb",
            return_value=100.0,
        )
        import dataclasses
        mocker.patch.object(
            dataclasses,
            "replace",
            side_effect=lambda obj, **kw: obj,
        )
        routed_cache_sentinel = mocker.MagicMock(
            regime_policy_by_fold=({},), bucket_edges_by_fold=({},),
            regime_routing_diagnostics=None,
        )
        transfer_spy = mocker.patch(
            "src.domain.futures.strategy.tiered_workflow.l2_meta.transfer_routing_plan_to_crisis_cache",
            return_value=routed_cache_sentinel,
        )

        from datetime import date
        window = SimpleNamespace(
            holdout_start=date(2024, 1, 19),
            l2_start=date(2024, 1, 5),
        )
        aligned = SimpleNamespace(
            symbols=("BTCUSDT",),
            close_2d=mocker.MagicMock(),
            datetimes=pd.date_range("2024-01-01", periods=500, freq="h"),
        )
        caps = SimpleNamespace(trial_number=0)
        signal_batch = mocker.MagicMock()
        signal_batch.start_idx = 0
        signal_batch.end_idx = 500
        signal_batch.registry_version = "v1"
        signal_batch.model_version = "v1"
        signal_batch.events = ()

        fake_crisis_ctx = mocker.MagicMock(awf_folds=(mocker.MagicMock(),))

        _run_tiered_l2_study(
            signal_batch=signal_batch,
            aligned=aligned,
            cfg=cfg,
            window=window,
            caps=caps,
            tf="1h",
            n_trials=2,
            seed=42,
            l2_sim_cache=mocker.MagicMock(),
            crisis_rets=mocker.MagicMock(),
            crisis_replay_ctx=fake_crisis_ctx,
        )

        assert transfer_spy.called
        assert transfer_spy.call_args.kwargs["crisis_cache"] is fake_crisis_ctx.cache
        assert transfer_spy.call_args.kwargs["n_crisis_folds"] == 1

    def test_run_tiered_l2_study_parity_flag_off_skips_transfer(self, mocker) -> None:
        """[SPEC l2-crisis-replay-routing-parity][S3] l2_crisis_replay_routing_parity_enabled=False
        시 transfer_routing_plan_to_crisis_cache가 호출되지 않음(회귀)."""
        from src.application.futures.runner.active_pipeline import _run_tiered_l2_study
        from src.domain.futures.strategy.config import CandidateStrategyConfig
        from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer2AllocationConfig

        _fake_default = Layer2AllocationConfig(l2_crisis_replay_routing_parity_enabled=False)
        mocker.patch(
            "src.domain.futures.strategy.tiered_workflow.dataclasses.Layer2AllocationConfig",
            return_value=_fake_default,
        )
        cfg = CandidateStrategyConfig(wf_n_folds=4)
        mocker.patch(
            "src.domain.futures.strategy.walk_forward.build_walk_forward_folds",
            return_value=(),
        )
        mocker.patch(
            "src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache",
            return_value=mocker.MagicMock(),
        )
        mocker.patch(
            "src.domain.futures.strategy.tiered_workflow.awf_sim.compute_per_tf_fit_edge",
            return_value={},
        )
        mocker.patch(
            "src.domain.futures.strategy.market_regime.compute_market_regime_context",
            return_value=mocker.MagicMock(),
        )
        mocker.patch(
            "src.domain.futures.strategy.tiered_workflow.l2_meta.build_regime_routing_plan",
            return_value=mocker.MagicMock(),
        )
        mocker.patch(
            "src.domain.futures.strategy.tiered_workflow.diagnostics.build_layer_universe_audit",
            return_value=mocker.MagicMock(warnings=()),
        )
        mocker.patch(
            "src.domain.futures.strategy.market_regime.compute_risk_severity_code",
            return_value=mocker.MagicMock(),
        )
        mocker.patch(
            "src.domain.futures.optimization.workflow.objective_l2_growth",
            return_value=0.0,
        )
        from src.domain.futures.optimization.opt_config import OPT_FUTURES_CONFIG
        mocker.patch.dict(OPT_FUTURES_CONFIG, {"L2_OPTUNA_BATCH_SIZE": 1})
        mocker.patch(
            "src.domain.futures.strategy.tiered_workflow.selection.select_layer2_champion",
            return_value=mocker.MagicMock(
                best_params={}, best_trial_number=None, completed_trials=0,
            ),
        )
        mocker.patch(
            "src.application.futures.runner.active_pipeline._get_rss_mb",
            return_value=100.0,
        )
        import dataclasses
        mocker.patch.object(
            dataclasses,
            "replace",
            side_effect=lambda obj, **kw: obj,
        )
        transfer_spy = mocker.patch(
            "src.domain.futures.strategy.tiered_workflow.l2_meta.transfer_routing_plan_to_crisis_cache",
        )

        from datetime import date
        window = SimpleNamespace(
            holdout_start=date(2025, 6, 1),
            l2_start=date(2024, 6, 1),
        )
        aligned = SimpleNamespace(
            symbols=("BTCUSDT",),
            close_2d=mocker.MagicMock(),
            datetimes=pd.date_range("2024-01-01", periods=500, freq="h"),
        )
        caps = SimpleNamespace(trial_number=0)
        signal_batch = mocker.MagicMock()
        signal_batch.start_idx = 0
        signal_batch.end_idx = 500
        signal_batch.registry_version = "v1"
        signal_batch.model_version = "v1"
        signal_batch.events = ()

        fake_crisis_ctx = mocker.MagicMock(awf_folds=(mocker.MagicMock(),))

        _run_tiered_l2_study(
            signal_batch=signal_batch,
            aligned=aligned,
            cfg=cfg,
            window=window,
            caps=caps,
            tf="1h",
            n_trials=2,
            seed=42,
            l2_sim_cache=mocker.MagicMock(),
            crisis_rets=mocker.MagicMock(),
            crisis_replay_ctx=fake_crisis_ctx,
        )

        assert not transfer_spy.called


def test_l2_batch_size_invariant_to_available_memory(mocker) -> None:
    """[l2-optuna-batch-determinism-fix] batch_size는 RAM 상태와 무관 — ask() 호출 횟수 불변."""
    from src.application.futures.runner.active_pipeline import _run_tiered_l2_study
    from src.domain.futures.strategy.config import CandidateStrategyConfig
    from src.domain.futures.optimization.opt_config import OPT_FUTURES_CONFIG

    cfg = CandidateStrategyConfig(wf_n_folds=4)
    ask_call_counts: list[int] = []
    max_workers_seen: list[int] = []

    _folds = (
        SimpleNamespace(fit_start=0, fit_end=80, cal_start=40, cal_end=80, oos_start=100, oos_end=200),
        SimpleNamespace(fit_start=0, fit_end=160, cal_start=120, cal_end=160, oos_start=200, oos_end=300),
        SimpleNamespace(fit_start=0, fit_end=260, cal_start=220, cal_end=260, oos_start=300, oos_end=400),
    )
    mocker.patch(
        "src.domain.futures.strategy.walk_forward.build_walk_forward_folds",
        return_value=_folds,
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache",
        return_value=mocker.MagicMock(),
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.awf_sim.compute_per_tf_fit_edge",
        return_value={},
    )
    mocker.patch(
        "src.domain.futures.strategy.market_regime.compute_market_regime_context",
        return_value=mocker.MagicMock(),
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.l2_meta.build_regime_routing_plan",
        return_value=mocker.MagicMock(),
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.diagnostics.build_layer_universe_audit",
        return_value=mocker.MagicMock(warnings=()),
    )
    mocker.patch(
        "src.domain.futures.strategy.market_regime.compute_risk_severity_code",
        return_value=mocker.MagicMock(),
    )
    mocker.patch(
        "src.domain.futures.optimization.workflow.objective_l2_growth",
        return_value=0.0,
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection.select_layer2_champion",
        return_value=mocker.MagicMock(best_params={}, best_trial_number=0, completed_trials=0),
    )
    mocker.patch(
        "src.application.futures.runner.active_pipeline._get_rss_mb",
        return_value=100.0,
    )
    import dataclasses
    mocker.patch.object(
        dataclasses,
        "replace",
        side_effect=lambda obj, **kw: obj,
    )
    mocker.patch.dict(OPT_FUTURES_CONFIG, {"L2_OPTUNA_BATCH_SIZE": 6}, clear=False)

    from datetime import date
    window = SimpleNamespace(
        holdout_start=date(2024, 1, 19),
        l2_start=date(2024, 1, 5),
    )
    aligned = SimpleNamespace(
        symbols=("BTCUSDT",),
        close_2d=mocker.MagicMock(),
        datetimes=pd.date_range("2024-01-01", periods=500, freq="h"),
    )
    caps = SimpleNamespace(trial_number=0)
    signal_batch = mocker.MagicMock()
    signal_batch.start_idx = 0
    signal_batch.end_idx = 500
    signal_batch.registry_version = "v1"
    signal_batch.model_version = "v1"
    signal_batch.events = ()

    for avail_gb in (0.5, 32.0):
        mocker.patch(
            "psutil.virtual_memory",
            return_value=SimpleNamespace(available=avail_gb * (1024.0**3)),
        )

        def _make_executor_spy(**kwargs):
            max_workers_seen.append(kwargs.get("max_workers"))
            mock_executor = mocker.MagicMock()
            mock_executor.__enter__.return_value = mock_executor
            mock_future = mocker.MagicMock()
            mock_future.result.return_value = (0.1, {}, 0.01)
            mock_executor.submit.return_value = mock_future
            return mock_executor

        mocker.patch(
            "concurrent.futures.ProcessPoolExecutor",
            side_effect=_make_executor_spy,
        )

        mock_study = mocker.MagicMock()
        mock_study.trials = []
        mock_study.ask.side_effect = [mocker.MagicMock(number=i) for i in range(6)]
        mock_study._stop_flag = False
        mocker.patch(
            "src.application.futures.runner.active_pipeline.get_or_create_study",
            return_value=mock_study,
        )

        _run_tiered_l2_study(
            signal_batch=signal_batch,
            aligned=aligned,
            cfg=cfg,
            window=window,
            caps=caps,
            tf="1h",
            n_trials=6,
            seed=42,
            l2_sim_cache=mocker.MagicMock(),
            l2_wf_n_folds=None,
        )

        ask_call_counts.append(mock_study.ask.call_count)

    assert ask_call_counts[0] == ask_call_counts[1] == 6
    assert max_workers_seen[0] <= max_workers_seen[1]


def test_run_tiered_l2_study_wires_probe_span_around_batch_loop(mocker) -> None:
    """[WS1][pipeline-runtime-memory-optimization] L2_optuna 배치 루프가
    L2RuntimeProbe.span('l2_optuna_batch', ...)로 감싸져 RSS/PSS peak를
    귀속 가능하게 계측하는지 검증."""
    from src.application.futures.runner.active_pipeline import _run_tiered_l2_study
    from src.domain.futures.strategy.config import CandidateStrategyConfig
    from src.domain.futures.optimization.opt_config import OPT_FUTURES_CONFIG

    cfg = CandidateStrategyConfig(wf_n_folds=4)

    _folds = (
        SimpleNamespace(fit_start=0, fit_end=80, cal_start=40, cal_end=80, oos_start=100, oos_end=200),
        SimpleNamespace(fit_start=0, fit_end=160, cal_start=120, cal_end=160, oos_start=200, oos_end=300),
        SimpleNamespace(fit_start=0, fit_end=260, cal_start=220, cal_end=260, oos_start=300, oos_end=400),
    )
    mocker.patch(
        "src.domain.futures.strategy.walk_forward.build_walk_forward_folds",
        return_value=_folds,
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache",
        return_value=mocker.MagicMock(),
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.awf_sim.compute_per_tf_fit_edge",
        return_value={},
    )
    mocker.patch(
        "src.domain.futures.strategy.market_regime.compute_market_regime_context",
        return_value=mocker.MagicMock(),
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.l2_meta.build_regime_routing_plan",
        return_value=mocker.MagicMock(),
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.diagnostics.build_layer_universe_audit",
        return_value=mocker.MagicMock(warnings=()),
    )
    mocker.patch(
        "src.domain.futures.strategy.market_regime.compute_risk_severity_code",
        return_value=mocker.MagicMock(),
    )
    mocker.patch(
        "src.domain.futures.optimization.workflow.objective_l2_growth",
        return_value=0.0,
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection.select_layer2_champion",
        return_value=mocker.MagicMock(best_params={}, best_trial_number=0, completed_trials=0),
    )
    mocker.patch(
        "src.application.futures.runner.active_pipeline._get_rss_mb",
        return_value=100.0,
    )
    import dataclasses

    mocker.patch.object(dataclasses, "replace", side_effect=lambda obj, **kw: obj)
    mocker.patch.dict(OPT_FUTURES_CONFIG, {"L2_OPTUNA_BATCH_SIZE": 2}, clear=False)
    mocker.patch("psutil.virtual_memory", return_value=SimpleNamespace(available=32.0 * (1024.0**3)))

    mock_executor = mocker.MagicMock()
    mock_executor.__enter__.return_value = mock_executor
    mock_future = mocker.MagicMock()
    mock_future.result.return_value = (0.1, {}, 0.01)
    mock_executor.submit.return_value = mock_future
    mocker.patch("concurrent.futures.ProcessPoolExecutor", return_value=mock_executor)

    mock_study = mocker.MagicMock()
    mock_study.trials = []
    mock_study.ask.side_effect = [mocker.MagicMock(number=i) for i in range(2)]
    mock_study._stop_flag = False
    mocker.patch(
        "src.application.futures.runner.active_pipeline.get_or_create_study",
        return_value=mock_study,
    )

    span_calls: list[tuple[Any, dict[str, Any]]] = []
    mock_probe = mocker.MagicMock()
    mock_probe.enabled = True

    def _record_span(stage: str, **fields: Any):
        span_calls.append((stage, fields))
        return contextlib.nullcontext()

    mock_probe.span.side_effect = _record_span
    mocker.patch(
        "src.application.futures.runner.active_pipeline.L2RuntimeProbe.from_environment",
        return_value=mock_probe,
    )

    from datetime import date

    window = SimpleNamespace(holdout_start=date(2024, 1, 19), l2_start=date(2024, 1, 5))
    aligned = SimpleNamespace(
        symbols=("BTCUSDT",),
        close_2d=mocker.MagicMock(),
        datetimes=pd.date_range("2024-01-01", periods=500, freq="h"),
    )
    caps = SimpleNamespace(trial_number=0)
    signal_batch = mocker.MagicMock()
    signal_batch.start_idx = 0
    signal_batch.end_idx = 500
    signal_batch.registry_version = "v1"
    signal_batch.model_version = "v1"
    signal_batch.events = ()

    _run_tiered_l2_study(
        signal_batch=signal_batch,
        aligned=aligned,
        cfg=cfg,
        window=window,
        caps=caps,
        tf="1h",
        n_trials=2,
        seed=42,
        l2_sim_cache=mocker.MagicMock(),
        l2_wf_n_folds=None,
    )

    assert any(stage == "l2_optuna_batch" for stage, _ in span_calls)
    batch_fields = next(fields for stage, fields in span_calls if stage == "l2_optuna_batch")
    assert "batch_num" in batch_fields
    assert "n_workers" in batch_fields


def test_run_tiered_l2_study_batch_loop_without_probe(mocker) -> None:
    """[WS1][pipeline-runtime-memory-optimization] L2RuntimeProbe가 비활성(기본,
    DEBUG 미설정) 상태여도 배치 루프가 예외 없이 정상 완료돼야 한다."""
    from src.application.futures.runner.active_pipeline import _run_tiered_l2_study
    from src.domain.futures.strategy.config import CandidateStrategyConfig
    from src.domain.futures.optimization.opt_config import OPT_FUTURES_CONFIG

    cfg = CandidateStrategyConfig(wf_n_folds=4)

    mocker.patch(
        "src.domain.futures.strategy.walk_forward.build_walk_forward_folds",
        return_value=(),
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache",
        return_value=mocker.MagicMock(),
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.awf_sim.compute_per_tf_fit_edge",
        return_value={},
    )
    mocker.patch(
        "src.domain.futures.strategy.market_regime.compute_market_regime_context",
        return_value=mocker.MagicMock(),
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.l2_meta.build_regime_routing_plan",
        return_value=mocker.MagicMock(),
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.diagnostics.build_layer_universe_audit",
        return_value=mocker.MagicMock(warnings=()),
    )
    mocker.patch(
        "src.domain.futures.strategy.market_regime.compute_risk_severity_code",
        return_value=mocker.MagicMock(),
    )
    mocker.patch(
        "src.domain.futures.optimization.workflow.objective_l2_growth",
        return_value=0.0,
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection.select_layer2_champion",
        return_value=mocker.MagicMock(best_params={}, best_trial_number=0, completed_trials=0),
    )
    mocker.patch(
        "src.application.futures.runner.active_pipeline._get_rss_mb",
        return_value=100.0,
    )
    import dataclasses

    mocker.patch.object(dataclasses, "replace", side_effect=lambda obj, **kw: obj)
    mocker.patch.dict(OPT_FUTURES_CONFIG, {"L2_OPTUNA_BATCH_SIZE": 2}, clear=False)
    mocker.patch("psutil.virtual_memory", return_value=SimpleNamespace(available=32.0 * (1024.0**3)))

    mock_executor = mocker.MagicMock()
    mock_executor.__enter__.return_value = mock_executor
    mock_future = mocker.MagicMock()
    mock_future.result.return_value = (0.1, {}, 0.01)
    mock_executor.submit.return_value = mock_future
    mocker.patch("concurrent.futures.ProcessPoolExecutor", return_value=mock_executor)

    mock_study = mocker.MagicMock()
    mock_study.trials = []
    mock_study.ask.side_effect = [mocker.MagicMock(number=i) for i in range(2)]
    mock_study._stop_flag = False
    mocker.patch(
        "src.application.futures.runner.active_pipeline.get_or_create_study",
        return_value=mock_study,
    )

    # L2RuntimeProbe.from_environment은 mock되지 않음 — 실제 기본(비활성) 인스턴스 사용.
    from datetime import date

    window = SimpleNamespace(holdout_start=date(2025, 6, 1), l2_start=date(2024, 6, 1))
    aligned = SimpleNamespace(
        symbols=("BTCUSDT",),
        close_2d=mocker.MagicMock(),
        datetimes=pd.date_range("2024-01-01", periods=500, freq="h"),
    )
    caps = SimpleNamespace(trial_number=0)
    signal_batch = mocker.MagicMock()
    signal_batch.start_idx = 0
    signal_batch.end_idx = 500
    signal_batch.registry_version = "v1"
    signal_batch.model_version = "v1"
    signal_batch.events = ()

    result = _run_tiered_l2_study(
        signal_batch=signal_batch,
        aligned=aligned,
        cfg=cfg,
        window=window,
        caps=caps,
        tf="1h",
        n_trials=2,
        seed=42,
        l2_sim_cache=mocker.MagicMock(),
        l2_wf_n_folds=None,
    )

    assert result is not None


def test_run_tiered_l2_study_logs_child_peak_rss(mocker: Any) -> None:
    from src.application.futures.runner.active_pipeline import _run_tiered_l2_study
    from src.domain.futures.strategy.config import CandidateStrategyConfig
    from src.domain.futures.optimization.opt_config import OPT_FUTURES_CONFIG

    cfg = CandidateStrategyConfig(wf_n_folds=4)

    _folds = (
        SimpleNamespace(fit_start=0, fit_end=80, cal_start=40, cal_end=80, oos_start=100, oos_end=200),
        SimpleNamespace(fit_start=0, fit_end=160, cal_start=120, cal_end=160, oos_start=200, oos_end=300),
        SimpleNamespace(fit_start=0, fit_end=260, cal_start=220, cal_end=260, oos_start=300, oos_end=400),
    )
    mocker.patch(
        "src.domain.futures.strategy.walk_forward.build_walk_forward_folds",
        return_value=_folds,
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache",
        return_value=mocker.MagicMock(),
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.awf_sim.compute_per_tf_fit_edge",
        return_value={},
    )
    mocker.patch(
        "src.domain.futures.strategy.market_regime.compute_market_regime_context",
        return_value=mocker.MagicMock(),
    )
    _routing_mock = mocker.MagicMock()
    _routing_mock.diagnostics.policy_diagnostics.sign_consistency_ratio = 1.0
    _routing_mock.diagnostics.policy_diagnostics.mean_cal_lift_bps = 0.0
    _routing_mock.diagnostics.policy_diagnostics.mean_confidence = 1.0
    _routing_mock.diagnostics.policy_diagnostics.mode = "test"
    _routing_mock.diagnostics.policy_diagnostics.global_reliable = True
    _routing_mock.diagnostics.policy_diagnostics.n_allow = 1
    _routing_mock.diagnostics.policy_diagnostics.n_downweight = 0
    _routing_mock.diagnostics.policy_diagnostics.n_block = 0
    _routing_mock.diagnostics.policy_diagnostics.n_pooled = 0
    _routing_mock.diagnostics.policy_diagnostics.n_unstable = 0
    _routing_mock.diagnostics.policy_diagnostics.n_hard_block_eligible = 0
    _routing_mock.diagnostics.policy_diagnostics.hard_block_enabled = False
    _routing_mock.diagnostics.compression_enabled = False
    _routing_mock.diagnostics.conditioning_path = "direct"
    _routing_mock.diagnostics.proof_passed = True
    _routing_mock.diagnostics.active_state_count = 1
    _routing_mock.diagnostics.debug_diagnostics = None
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.l2_meta.build_regime_routing_plan",
        return_value=_routing_mock,
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.diagnostics.build_layer_universe_audit",
        return_value=mocker.MagicMock(warnings=()),
    )
    mocker.patch(
        "src.domain.futures.strategy.market_regime.compute_risk_severity_code",
        return_value=mocker.MagicMock(),
    )
    mocker.patch(
        "src.domain.futures.optimization.workflow.objective_l2_growth",
        return_value=0.0,
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection.select_layer2_champion",
        return_value=mocker.MagicMock(best_params={}, best_trial_number=0, completed_trials=0),
    )
    mocker.patch(
        "src.application.futures.runner.active_pipeline._get_rss_mb",
        return_value=100.0,
    )
    import dataclasses

    mocker.patch.object(dataclasses, "replace", side_effect=lambda obj, **kw: obj)
    mocker.patch.dict(OPT_FUTURES_CONFIG, {"L2_OPTUNA_BATCH_SIZE": 2}, clear=False)
    mocker.patch("psutil.virtual_memory", return_value=SimpleNamespace(available=32.0 * (1024.0**3)))

    mock_executor = mocker.MagicMock()
    mock_executor.__enter__.return_value = mock_executor
    mock_future = mocker.MagicMock()
    mock_future.result.return_value = (0.1, {}, 0.01)
    mock_executor.submit.return_value = mock_future
    mocker.patch("concurrent.futures.ProcessPoolExecutor", return_value=mock_executor)

    mock_study = mocker.MagicMock()
    mock_study.trials = []
    mock_study.ask.side_effect = [mocker.MagicMock(number=i) for i in range(2)]
    mock_study._stop_flag = False
    mocker.patch(
        "src.application.futures.runner.active_pipeline.get_or_create_study",
        return_value=mock_study,
    )

    import src.application.futures.runner.active_pipeline as _ap_mod

    spy = mocker.spy(_ap_mod, "_get_child_peak_rss_mb")

    from datetime import date

    window = SimpleNamespace(holdout_start=date(2024, 1, 19), l2_start=date(2024, 1, 5))
    aligned = SimpleNamespace(
        symbols=("BTCUSDT",),
        close_2d=mocker.MagicMock(),
        datetimes=pd.date_range("2024-01-01", periods=500, freq="h"),
    )
    caps = SimpleNamespace(trial_number=0)
    signal_batch = mocker.MagicMock()
    signal_batch.start_idx = 0
    signal_batch.end_idx = 500
    signal_batch.registry_version = "v1"
    signal_batch.model_version = "v1"
    signal_batch.events = ()

    _run_tiered_l2_study(
        signal_batch=signal_batch,
        aligned=aligned,
        cfg=cfg,
        window=window,
        caps=caps,
        tf="1h",
        n_trials=2,
        seed=42,
        l2_sim_cache=mocker.MagicMock(),
        l2_wf_n_folds=None,
    )

    assert spy.call_count == 2
    for ret in spy.spy_return_list:
        assert isinstance(ret, float)
        assert ret >= 0.0 or ret == -1.0


def _setup_l2_study_mocks(
    mocker: Any,
    n_trials: int = 2,
    batch_size_val: int = 2,
    routing_diag: Any = None,
) -> tuple[Any, Any, Any, Any]:
    from src.application.futures.runner.active_pipeline import _run_tiered_l2_study
    from src.domain.futures.strategy.config import CandidateStrategyConfig
    from src.domain.futures.optimization.opt_config import OPT_FUTURES_CONFIG

    cfg = CandidateStrategyConfig(wf_n_folds=4)

    _folds = (
        SimpleNamespace(fit_start=0, fit_end=80, cal_start=40, cal_end=80, oos_start=100, oos_end=200),
        SimpleNamespace(fit_start=0, fit_end=160, cal_start=120, cal_end=160, oos_start=200, oos_end=300),
        SimpleNamespace(fit_start=0, fit_end=260, cal_start=220, cal_end=260, oos_start=300, oos_end=400),
    )
    mocker.patch(
        "src.domain.futures.strategy.walk_forward.build_walk_forward_folds",
        return_value=_folds,
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache",
        return_value=mocker.MagicMock(),
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.awf_sim.compute_per_tf_fit_edge",
        return_value={},
    )
    mocker.patch(
        "src.domain.futures.strategy.market_regime.compute_market_regime_context",
        return_value=mocker.MagicMock(),
    )
    if routing_diag is None:
        _routing_mock = mocker.MagicMock()
        _routing_mock.diagnostics.policy_diagnostics.sign_consistency_ratio = 1.0
        _routing_mock.diagnostics.policy_diagnostics.mean_cal_lift_bps = 0.0
        _routing_mock.diagnostics.policy_diagnostics.mean_confidence = 1.0
        _routing_mock.diagnostics.policy_diagnostics.mode = "test"
        _routing_mock.diagnostics.policy_diagnostics.global_reliable = True
        _routing_mock.diagnostics.policy_diagnostics.n_allow = 1
        _routing_mock.diagnostics.policy_diagnostics.n_downweight = 0
        _routing_mock.diagnostics.policy_diagnostics.n_block = 0
        _routing_mock.diagnostics.policy_diagnostics.n_pooled = 0
        _routing_mock.diagnostics.policy_diagnostics.n_unstable = 0
        _routing_mock.diagnostics.policy_diagnostics.n_hard_block_eligible = 0
        _routing_mock.diagnostics.policy_diagnostics.hard_block_enabled = False
        _routing_mock.diagnostics.compression_enabled = False
        _routing_mock.diagnostics.conditioning_path = "direct"
        _routing_mock.diagnostics.proof_passed = True
        _routing_mock.diagnostics.active_state_count = 1
        _routing_mock.diagnostics.debug_diagnostics = None
        routing_diag = _routing_mock
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.l2_meta.build_regime_routing_plan",
        return_value=routing_diag,
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.diagnostics.build_layer_universe_audit",
        return_value=mocker.MagicMock(warnings=()),
    )
    mocker.patch(
        "src.domain.futures.strategy.market_regime.compute_risk_severity_code",
        return_value=mocker.MagicMock(),
    )
    mocker.patch(
        "src.domain.futures.optimization.workflow.objective_l2_growth",
        return_value=0.0,
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.selection.select_layer2_champion",
        return_value=mocker.MagicMock(best_params={}, best_trial_number=0, completed_trials=0),
    )
    mocker.patch(
        "src.application.futures.runner.active_pipeline._get_rss_mb",
        return_value=100.0,
    )
    import dataclasses

    mocker.patch.object(dataclasses, "replace", side_effect=lambda obj, **kw: obj)
    mocker.patch.dict(OPT_FUTURES_CONFIG, {"L2_OPTUNA_BATCH_SIZE": batch_size_val}, clear=False)
    mocker.patch("psutil.virtual_memory", return_value=SimpleNamespace(available=32.0 * (1024.0**3)))

    mock_executor = mocker.MagicMock()
    mock_executor.__enter__.return_value = mock_executor
    mock_future = mocker.MagicMock()
    mock_future.result.return_value = (0.1, {}, 0.01)
    mock_executor.submit.return_value = mock_future
    mocker.patch("concurrent.futures.ProcessPoolExecutor", return_value=mock_executor)

    mock_study = mocker.MagicMock()
    mock_study.trials = []
    mock_study.ask.side_effect = [mocker.MagicMock(number=i) for i in range(n_trials)]
    mock_study._stop_flag = False
    mocker.patch(
        "src.application.futures.runner.active_pipeline.get_or_create_study",
        return_value=mock_study,
    )

    return cfg, mock_study, _run_tiered_l2_study, mock_executor


def _call_l2_study(
    run_fn: Any,
    cfg: Any,
    n_trials: int = 2,
    tf: str = "1h",
    seed: int = 42,
) -> Any:
    from datetime import date

    window = SimpleNamespace(holdout_start=date(2024, 1, 19), l2_start=date(2024, 1, 5))
    aligned = SimpleNamespace(
        symbols=("BTCUSDT",),
        close_2d=SimpleNamespace(),
        datetimes=pd.date_range("2024-01-01", periods=500, freq="h"),
    )
    caps = SimpleNamespace(trial_number=0)
    signal_batch = SimpleNamespace()
    signal_batch.start_idx = 0
    signal_batch.end_idx = 500
    signal_batch.registry_version = "v1"
    signal_batch.model_version = "v1"
    signal_batch.events = ()
    signal_batch.symbols = ("BTCUSDT",)

    return run_fn(
        signal_batch=signal_batch,
        aligned=aligned,
        cfg=cfg,
        window=window,
        caps=caps,
        tf=tf,
        n_trials=n_trials,
        seed=seed,
        l2_sim_cache=SimpleNamespace(),
        l2_wf_n_folds=None,
    )


def test_run_tiered_l2_study_logs_worker_private_per_batch(mocker: Any) -> None:
    MB = 1024 * 1024
    from src.domain.futures.strategy.tiered_workflow.memory import ProcessTreeMemory

    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.memory.snapshot_process_tree_memory",
        side_effect=[
            ProcessTreeMemory(parent_rss_bytes=0, tree_pss_bytes=6000 * MB, tree_uss_bytes=None, available_bytes=0),
            ProcessTreeMemory(parent_rss_bytes=0, tree_pss_bytes=6600 * MB, tree_uss_bytes=None, available_bytes=0),
            ProcessTreeMemory(parent_rss_bytes=0, tree_pss_bytes=6600 * MB, tree_uss_bytes=None, available_bytes=0),
            ProcessTreeMemory(parent_rss_bytes=0, tree_pss_bytes=6800 * MB, tree_uss_bytes=None, available_bytes=0),
            ProcessTreeMemory(parent_rss_bytes=0, tree_pss_bytes=6800 * MB, tree_uss_bytes=None, available_bytes=0),
            ProcessTreeMemory(parent_rss_bytes=0, tree_pss_bytes=7000 * MB, tree_uss_bytes=None, available_bytes=0),
        ],
    )

    import logging

    captured_logs: list[str] = []

    class _CaptureHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured_logs.append(record.getMessage())

    logger = logging.getLogger("opt_main_futures")
    saved_level = logger.level
    saved_handlers = list(logger.handlers)
    saved_propagate = logger.propagate
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.addHandler(_CaptureHandler())
    logger.propagate = False

    cfg, _study, run_fn, _executor = _setup_l2_study_mocks(mocker, n_trials=6, batch_size_val=2)
    _call_l2_study(run_fn, cfg, n_trials=6)

    logger.setLevel(saved_level)
    logger.handlers.clear()
    logger.handlers.extend(saved_handlers)
    logger.propagate = saved_propagate

    wp_logs = [m for m in captured_logs if "worker_private_measured" in m]
    assert len(wp_logs) == 3
    assert "batch_num=1" in wp_logs[0]
    assert "batch_num=2" in wp_logs[1]
    assert "batch_num=3" in wp_logs[2]
    assert "measured_worker_private_mb=" in wp_logs[0]


def test_run_tiered_l2_study_logs_worker_private_na_when_pss_unavailable(mocker: Any) -> None:
    from src.domain.futures.strategy.tiered_workflow.memory import ProcessTreeMemory

    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.memory.snapshot_process_tree_memory",
        return_value=ProcessTreeMemory(parent_rss_bytes=0, tree_pss_bytes=None, tree_uss_bytes=None, available_bytes=0),
    )

    import logging

    captured_logs: list[str] = []

    class _CaptureHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured_logs.append(record.getMessage())

    logger = logging.getLogger("opt_main_futures")
    saved_level = logger.level
    saved_handlers = list(logger.handlers)
    saved_propagate = logger.propagate
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.addHandler(_CaptureHandler())
    logger.propagate = False

    cfg, _study, run_fn, _executor = _setup_l2_study_mocks(mocker, n_trials=2, batch_size_val=2)
    _call_l2_study(run_fn, cfg, n_trials=2)

    logger.setLevel(saved_level)
    logger.handlers.clear()
    logger.handlers.extend(saved_handlers)
    logger.propagate = saved_propagate

    wp_logs = [m for m in captured_logs if "worker_private_measured" in m]
    assert wp_logs
    assert "measured_worker_private_mb=n/a" in wp_logs[0]


def test_run_tiered_l2_study_batch_determinism_unaffected_by_memory_snapshot(mocker: Any) -> None:
    from src.domain.futures.strategy.tiered_workflow.memory import ProcessTreeMemory

    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.memory.snapshot_process_tree_memory",
        side_effect=[
            ProcessTreeMemory(parent_rss_bytes=0, tree_pss_bytes=6000 * 1024 * 1024, tree_uss_bytes=None, available_bytes=0),
            ProcessTreeMemory(parent_rss_bytes=0, tree_pss_bytes=6600 * 1024 * 1024, tree_uss_bytes=None, available_bytes=0),
        ],
    )

    cfg, mock_study, run_fn, _executor = _setup_l2_study_mocks(mocker, n_trials=2, batch_size_val=2)

    _call_l2_study(run_fn, cfg, n_trials=2)

    assert mock_study.ask.call_count == 2
    assert mock_study.tell.call_count == 2


def test_l2_batch_size_defaults_when_config_missing() -> None:
    """[l2-optuna-batch-determinism-fix] L2_OPTUNA_BATCH_SIZE 키 없을 때 기본값 2."""
    from src.domain.futures.optimization.opt_config import OPT_FUTURES_CONFIG

    saved = OPT_FUTURES_CONFIG.pop("L2_OPTUNA_BATCH_SIZE")
    try:
        assert int(OPT_FUTURES_CONFIG.get("L2_OPTUNA_BATCH_SIZE", 2)) == 2
    finally:
        OPT_FUTURES_CONFIG["L2_OPTUNA_BATCH_SIZE"] = saved


def test_l2_study_trial_sequence_reproducible_across_memory_states(mocker) -> None:
    """[l2-optuna-batch-determinism-fix] 동일 seed면 저RAM/고RAM 모두 tell() 순서·값이 동일해야 한다.

    tests/unit/execution/test_tiered_l2_optuna_integration.py의 TestS14는
    `src.execution.opt_main_futures._run_tiered_l2_study`를 import하나 해당 심볼이
    그 모듈에 존재하지 않는 기존(스코프 밖) 깨진 참조라 이 spec의 대상이 아니다 —
    실제 구현체가 있는 이 모듈에서 종단 재현성을 검증한다.
    """
    import optuna

    from src.application.futures.runner.active_pipeline import _run_tiered_l2_study
    from src.domain.futures.strategy.config import CandidateStrategyConfig
    from src.domain.futures.optimization.opt_config import OPT_FUTURES_CONFIG

    cfg = CandidateStrategyConfig(wf_n_folds=4)

    _folds = (
        SimpleNamespace(fit_start=0, fit_end=80, cal_start=40, cal_end=80, oos_start=100, oos_end=200),
        SimpleNamespace(fit_start=0, fit_end=160, cal_start=120, cal_end=160, oos_start=200, oos_end=300),
        SimpleNamespace(fit_start=0, fit_end=260, cal_start=220, cal_end=260, oos_start=300, oos_end=400),
    )
    mocker.patch(
        "src.domain.futures.strategy.walk_forward.build_walk_forward_folds",
        return_value=_folds,
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache",
        return_value=mocker.MagicMock(),
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.awf_sim.compute_per_tf_fit_edge",
        return_value={},
    )
    mocker.patch(
        "src.domain.futures.strategy.market_regime.compute_market_regime_context",
        return_value=mocker.MagicMock(),
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.l2_meta.build_regime_routing_plan",
        return_value=mocker.MagicMock(),
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.diagnostics.build_layer_universe_audit",
        return_value=mocker.MagicMock(warnings=()),
    )
    mocker.patch(
        "src.domain.futures.strategy.market_regime.compute_risk_severity_code",
        return_value=mocker.MagicMock(),
    )
    mocker.patch(
        "src.application.futures.runner.active_pipeline._get_rss_mb",
        return_value=100.0,
    )
    import dataclasses
    mocker.patch.object(
        dataclasses,
        "replace",
        side_effect=lambda obj, **kw: obj,
    )
    mocker.patch.dict(OPT_FUTURES_CONFIG, {"L2_OPTUNA_BATCH_SIZE": 6}, clear=False)

    from datetime import date
    window = SimpleNamespace(
        holdout_start=date(2024, 1, 19),
        l2_start=date(2024, 1, 5),
    )
    aligned = SimpleNamespace(
        symbols=("BTCUSDT",),
        close_2d=mocker.MagicMock(),
        datetimes=pd.date_range("2024-01-01", periods=500, freq="h"),
    )
    caps = SimpleNamespace(trial_number=0)
    signal_batch = mocker.MagicMock()
    signal_batch.start_idx = 0
    signal_batch.end_idx = 500
    signal_batch.registry_version = "v1"
    signal_batch.model_version = "v1"
    signal_batch.events = ()

    told_sequences: list[list[tuple[int, float]]] = []

    for avail_gb in (0.5, 32.0):
        mocker.patch(
            "psutil.virtual_memory",
            return_value=SimpleNamespace(available=avail_gb * (1024.0**3)),
        )

        told: list[tuple[int, float]] = []

        def _make_executor_spy(**kwargs):
            mock_executor = mocker.MagicMock()
            mock_executor.__enter__.return_value = mock_executor

            def _submit(fn, params):
                mock_future = mocker.MagicMock()
                mock_future.result.return_value = (0.1, {}, 0.01)
                return mock_future

            mock_executor.submit.side_effect = _submit
            return mock_executor

        mocker.patch(
            "concurrent.futures.ProcessPoolExecutor",
            side_effect=_make_executor_spy,
        )

        champion_spy = mocker.patch(
            "src.domain.futures.strategy.tiered_workflow.selection.select_layer2_champion",
            return_value=mocker.MagicMock(best_params={"seed_check": "ok"}, best_trial_number=5, completed_trials=6),
        )

        mocker.patch(
            "src.domain.futures.strategy.tiered_workflow.awf_sim.compute_per_tf_fit_edge",
            return_value={},
        )

        mock_study = mocker.MagicMock()
        mock_study.trials = []
        _ask_trials = [mocker.MagicMock(number=i, params={"_trial_number": i}) for i in range(6)]
        mock_study.ask.side_effect = _ask_trials
        mock_study._stop_flag = False

        def _tell(
            trial: Any,
            value: Any,
            _told: list[tuple[int, float]] = told,
            _study: Any = mock_study,
        ) -> None:
            _told.append((trial.number, float(value)))
            trial.value = value
            trial.state = optuna.trial.TrialState.COMPLETE
            if trial not in _study.trials:
                _study.trials.append(trial)

        mock_study.tell.side_effect = _tell
        mocker.patch(
            "src.application.futures.runner.active_pipeline.get_or_create_study",
            return_value=mock_study,
        )

        _run_tiered_l2_study(
            signal_batch=signal_batch,
            aligned=aligned,
            cfg=cfg,
            window=window,
            caps=caps,
            tf="1h",
            n_trials=6,
            seed=42,
            l2_sim_cache=mocker.MagicMock(),
            l2_wf_n_folds=None,
        )

        told_sequences.append(told)
        assert champion_spy.called

    assert told_sequences[0] == told_sequences[1], (
        f"tell() sequence diverged across memory states: {told_sequences[0]} != {told_sequences[1]}"
    )


@pytest.mark.slow
def test_run_tiered_l2_study_wall_time_within_perf_budget() -> None:
    """L2 wf_n_folds=8 실행 시 wall-time이 baseline 대비 15% 이내인지 확인.
    수동/스크립트 성격 테스트 — 실제 full 실행 필요."""
    pass


def test_active_pipeline_helpers() -> None:
    from src.application.futures.runner.active_pipeline import (
        _btc_index_if_present,
        _get_rss_mb,
        _get_child_peak_rss_mb,
        _get_peak_rss_mb,
        _fit_table_cell,
        _select_probe_source_tf,
    )
    
    # 1. _btc_index_if_present
    assert _btc_index_if_present(("ETHUSDT", "BTCUSDT", "SOLUSDT")) == 1
    assert _btc_index_if_present(("ETHUSDT", "SOLUSDT")) == -1

    # 2. _get_rss_mb, _get_child_peak_rss_mb, _get_peak_rss_mb (execution test)
    _get_rss_mb()
    _get_child_peak_rss_mb()
    _get_peak_rss_mb()

    # 3. _fit_table_cell
    assert _fit_table_cell("hello", 10) == "hello"
    assert _fit_table_cell("hello world", 5) == "he..."
    assert _fit_table_cell("hi", 2) == "hi"

    # 4. _select_probe_source_tf
    sym_maps = {"BTCUSDT": {"candles_1h": None}}
    _select_probe_source_tf(sym_maps, "1h")


# ─── Multi-Seed Robustness Consensus tests ───────────────────────────────


import numpy as np
import pytest

from src.application.futures.runner.active_pipeline import (
    SeedRobustnessOutcome,
    _run_multi_seed_robustness_consensus,
)


def _make_outcome(seed: int, *, passed: bool, cagr: float) -> SeedRobustnessOutcome:
    l3_final = type("L3", (), {"gate_passed": passed, "cagr": cagr})()
    l2_final = type("L2", (), {"gate_passed": True, "cagr_hybrid": cagr, "window_bottleneck_covered": True})()
    return SeedRobustnessOutcome(
        seed=seed,
        l1_result=None,  # type: ignore[arg-type]
        l2_study_result=None,  # type: ignore[arg-type]
        l2_final=l2_final,
        l3_final=l3_final,
        passed=passed,
        blocker_reason="" if passed else "negative_return",
    )


def _dummy_regime_code() -> np.ndarray:
    return np.array([0, 0, 1, 1, 2, 0], dtype=np.int8)


def test_run_multi_seed_robustness_consensus_admits_when_majority_passes(mocker) -> None:
    outcomes_by_seed = {
        42: _make_outcome(42, passed=True, cagr=0.10),
        43: _make_outcome(43, passed=True, cagr=0.05),
        44: _make_outcome(44, passed=False, cagr=-0.20),
    }
    mocker.patch(
        "src.application.futures.runner.active_pipeline._run_single_seed_outcome",
        side_effect=lambda seed, **kw: outcomes_by_seed[seed],
    )

    result = _run_multi_seed_robustness_consensus(
        signal_batch=mocker.Mock(), aligned=mocker.Mock(), cfg=mocker.Mock(),
        window=mocker.Mock(), caps=mocker.Mock(), tf="8h", n_trials=120,
        base_seed=42, target_phase="l3", l1_res=mocker.Mock(),
        labeled_events=mocker.Mock(), per_tf_data_maps=None, labeled_events_by_tf=None,
        crisis_rets=None, crisis_replay_ctx=None, l2_sim_cache=None,
        probe_manifest=None, l3_regime_code_1d=_dummy_regime_code(),
    )

    assert result.admitted is True
    assert result.pass_count == 2
    assert result.selected is not None
    assert result.selected.seed == 43  # cagr=0.05, lower than 0.10 among the passing pair


def test_run_multi_seed_robustness_consensus_blocks_when_majority_fails(mocker) -> None:
    outcomes_by_seed = {
        42: _make_outcome(42, passed=False, cagr=-0.10),
        43: _make_outcome(43, passed=True, cagr=0.31),
        44: _make_outcome(44, passed=False, cagr=-0.20),
    }
    mocker.patch(
        "src.application.futures.runner.active_pipeline._run_single_seed_outcome",
        side_effect=lambda seed, **kw: outcomes_by_seed[seed],
    )

    result = _run_multi_seed_robustness_consensus(
        signal_batch=mocker.Mock(), aligned=mocker.Mock(), cfg=mocker.Mock(),
        window=mocker.Mock(), caps=mocker.Mock(), tf="8h", n_trials=120,
        base_seed=42, target_phase="l3", l1_res=mocker.Mock(),
        labeled_events=mocker.Mock(), per_tf_data_maps=None, labeled_events_by_tf=None,
        crisis_rets=None, crisis_replay_ctx=None, l2_sim_cache=None,
        probe_manifest=None, l3_regime_code_1d=_dummy_regime_code(),
    )

    assert result.admitted is False
    assert result.blocker_reason == "seed_consensus_blocked:1/3"
    assert result.selected is None


def test_run_multi_seed_robustness_consensus_selects_most_conservative_not_best(mocker) -> None:
    outcomes_by_seed = {
        42: _make_outcome(42, passed=True, cagr=0.20),
        43: _make_outcome(43, passed=True, cagr=0.05),
        44: _make_outcome(44, passed=True, cagr=0.10),
    }
    mocker.patch(
        "src.application.futures.runner.active_pipeline._run_single_seed_outcome",
        side_effect=lambda seed, **kw: outcomes_by_seed[seed],
    )

    result = _run_multi_seed_robustness_consensus(
        signal_batch=mocker.Mock(), aligned=mocker.Mock(), cfg=mocker.Mock(),
        window=mocker.Mock(), caps=mocker.Mock(), tf="8h", n_trials=120,
        base_seed=42, target_phase="l3", l1_res=mocker.Mock(),
        labeled_events=mocker.Mock(), per_tf_data_maps=None, labeled_events_by_tf=None,
        crisis_rets=None, crisis_replay_ctx=None, l2_sim_cache=None,
        probe_manifest=None, l3_regime_code_1d=_dummy_regime_code(),
    )

    assert result.admitted is True
    assert result.pass_count == 3
    assert result.selected is not None
    assert result.selected.seed == 43
    assert result.selected.l3_final.cagr == 0.05


def test_run_multi_seed_robustness_consensus_l2_phase_uses_l2_cagr(mocker) -> None:
    l3_final_passed = type("L3", (), {"gate_passed": True, "cagr": 0.99})()
    l2_final_42 = type("L2", (), {"gate_passed": True, "cagr_hybrid": 0.08, "window_bottleneck_covered": True})()
    l2_final_43 = type("L2", (), {"gate_passed": True, "cagr_hybrid": 0.03, "window_bottleneck_covered": True})()
    l2_final_44 = type("L2", (), {"gate_passed": True, "cagr_hybrid": 0.15, "window_bottleneck_covered": True})()
    outcomes_by_seed = {
        42: SeedRobustnessOutcome(
            seed=42, l1_result=None, l2_study_result=None,
            l2_final=l2_final_42, l3_final=l3_final_passed,
            passed=True, blocker_reason="",
        ),
        43: SeedRobustnessOutcome(
            seed=43, l1_result=None, l2_study_result=None,
            l2_final=l2_final_43, l3_final=l3_final_passed,
            passed=True, blocker_reason="",
        ),
        44: SeedRobustnessOutcome(
            seed=44, l1_result=None, l2_study_result=None,
            l2_final=l2_final_44, l3_final=l3_final_passed,
            passed=True, blocker_reason="",
        ),
    }
    mocker.patch(
        "src.application.futures.runner.active_pipeline._run_single_seed_outcome",
        side_effect=lambda seed, **kw: outcomes_by_seed[seed],
    )

    result = _run_multi_seed_robustness_consensus(
        signal_batch=mocker.Mock(), aligned=mocker.Mock(), cfg=mocker.Mock(),
        window=mocker.Mock(), caps=mocker.Mock(), tf="8h", n_trials=120,
        base_seed=42, target_phase="l2", l1_res=mocker.Mock(),
        labeled_events=mocker.Mock(), per_tf_data_maps=None, labeled_events_by_tf=None,
        crisis_rets=None, crisis_replay_ctx=None, l2_sim_cache=None,
        probe_manifest=None, l3_regime_code_1d=_dummy_regime_code(),
    )

    assert result.admitted is True
    assert result.pass_count == 3
    assert result.selected is not None
    assert result.selected.seed == 43  # lowest l2 cagr_hybrid=0.03


def test_active_pipeline_l3_blocked_when_robust_candidate_fails_returns_exit_code_1(mocker) -> None:
    from src.application.futures.runner.active_pipeline import (
        RobustSearchOutcome,
        _run_strategy_stage,
    )
    from src.application.futures.runner.models import RunnerResult

    _failed_outcome = RobustSearchOutcome(
        l2_study_result=None, l2_final=None, l3_final=None,
        passed=False, blocker_reason="no_robust_candidate",
    )
    mocker.patch(
        "src.application.futures.runner.active_pipeline._run_portfolio_causal_robust_outcome",
        return_value=_failed_outcome,
    )

    _tiered_window = mocker.Mock()
    _tiered_window.fetch_start = mocker.Mock()
    _tiered_window.fetch_start.isoformat.return_value = "2024-01-01"
    _tiered_window.holdout_start = mocker.Mock()
    _tiered_window.holdout_start.isoformat.return_value = "2024-06-01"
    _tiered_window.holdout_end = "2024-09-01"
    _tiered_window.l1_start = mocker.Mock()
    _tiered_window.l1_start.isoformat.return_value = "2024-01-01"

    _cfg = mocker.Mock()
    _cfg.phase = "l3"
    _cfg.l1_tfs = ("4h",)
    _cfg.seed = 42

    _window = mocker.Mock(
        fetch_start="2024-01-01", is_start="2024-03-01", oos_start="2024-06-01",
        end_date="2024-09-01", fetch_start_date=mocker.Mock(), is_start_date=mocker.Mock(),
        oos_start_date=mocker.Mock(), end_date_value=mocker.Mock(),
    )
    _data_stage = mocker.Mock()
    _data_stage.valid_symbols = ["BTCUSDT"]
    _data_stage.effective_l0_evidence_end = None
    _data_stage.data_maps = {"BTCUSDT": {}}
    _data_stage.oos_data_maps = {"BTCUSDT": {}}

    mocker.patch(
        "src.application.futures.runner.active_pipeline.pick_strategy_data_maps",
        return_value={"BTCUSDT": {"4h": mocker.Mock()}},
    )
    mocker.patch(
        "src.application.futures.runner.active_pipeline.run_active_strategy_output_bridge",
        return_value=mocker.Mock(
            labeled_unfiltered=mocker.Mock(),
            alpha_foundry_report=mocker.Mock(mode="pass", n_passed=1),
        ),
    )
    mocker.patch(
        "src.application.futures.runner.active_pipeline._tiered_labeled_events",
        return_value=mocker.Mock(),
    )
    mocker.patch(
        "src.application.futures.runner.active_pipeline._resolve_layered_window",
        return_value=_tiered_window,
    )
    mocker.patch(
        "src.application.futures.runner.active_pipeline._resolve_base_symbol_scope",
        return_value=("BTCUSDT",),
    )
    mocker.patch(
        "src.application.futures.runner.active_pipeline._resolve_tradeable_scope",
        return_value=mocker.Mock(admitted=["BTCUSDT"], dropped_by_reason={}),
    )
    mocker.patch(
        "src.application.futures.runner.tiered_handoff.consume_candidate_output_for_tiered",
        return_value=mocker.Mock(
            aligned=mocker.Mock(datetimes=mocker.Mock(
                __getitem__=mocker.Mock(return_value=pd.Timestamp("2024-01-01")),
                __len__=mocker.Mock(return_value=100),
            )),
            labeled_events=mocker.Mock(),
            aligned_by_tf={},
            labeled_events_by_tf={},
            l0_delivery_manifest=None,
        ),
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.pipeline.run_tiered_pipeline",
        return_value=(mocker.Mock(gate_passed=True), None, None),
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.awf_sim.build_l2_simulation_cache",
        return_value=mocker.Mock(),
    )
    mocker.patch(
        "src.application.futures.runner.active_pipeline._build_l2_signal_batch",
        return_value=mocker.Mock(),
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.pipeline._resolve_l2_master_tf_from_prior",
        return_value="8h",
    )
    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.pipeline._load_crisis_replay_context",
        return_value=None,
    )
    mocker.patch(
        "src.domain.futures.strategy.market_regime.compress_regime_codes",
        return_value=np.array([0, 0, 1, 1], dtype=np.int8),
    )
    mocker.patch(
        "src.domain.futures.strategy.market_regime.compute_market_regime_context",
        return_value=mocker.Mock(code_1d=np.array([0, 0, 1, 1], dtype=np.int8)),
    )

    _output = _run_strategy_stage(
        run_config=_cfg,
        window=_window,
        data_stage=_data_stage,
        trading_symbols=("BTCUSDT",),
        layered_window=_tiered_window,
    )

    assert isinstance(_output, RunnerResult)
    assert _output.exit_code == 1
    assert _output.reason == "no_robust_candidate"


# ─── Recency Holdout Gate: window_covered log ─────────────────────────────


class TestMultiSeedWindowCoverageLogging:
    def test_run_single_seed_outcome_logs_window_covered(self, mocker):
        from src.application.futures.runner import active_pipeline as _ap
        l2_final = type("L2", (), {"gate_passed": True, "cagr_hybrid": 0.05, "window_bottleneck_covered": False})()
        outcome = SeedRobustnessOutcome(
            seed=42, l1_result=None, l2_study_result=None,
            l2_final=l2_final, l3_final=None, passed=True, blocker_reason="",
        )
        mocker.patch.object(_ap, "_run_single_seed_outcome", return_value=outcome)
        mock_logger = mocker.patch.object(_ap, "_logger")
        _ap._run_multi_seed_robustness_consensus(
            signal_batch=mocker.Mock(), aligned=mocker.Mock(), cfg=mocker.Mock(),
            window=mocker.Mock(), caps=mocker.Mock(), tf="8h", n_trials=120,
            base_seed=42, target_phase="l2", l1_res=mocker.Mock(),
            labeled_events=mocker.Mock(), per_tf_data_maps=None, labeled_events_by_tf=None,
            crisis_rets=None, crisis_replay_ctx=None, l2_sim_cache=None,
            probe_manifest=None, l3_regime_code_1d=mocker.Mock(),
        )
        pass_count_call = next(
            (c for c in mock_logger.info.call_args_list if "pass_count=" in str(c.args)),
            None,
        )
        assert pass_count_call is not None
        _, kwargs = pass_count_call
        # kwargs가 없으면 args[0]이 format string, args[-1]이 window_covered value
        assert pass_count_call.args[-1] is False


# ─── Fix A: crisis_rets wiring (L2-CRISIS-WIRING-AND-TF-SIGNAL-LOSS-FIX) ──


def test_run_robust_l2_l3_outcome_threads_crisis_rets_symmetrically(mocker) -> None:
    from src.application.futures.runner.active_pipeline import _run_robust_l2_l3_outcome

    captured: dict = {}

    def _fake_l2_study(**kwargs):
        captured.update(kwargs)
        return mocker.Mock(blocker_reason="stub_blocked", best_params={})

    mocker.patch(
        "src.application.futures.runner.active_pipeline._run_tiered_l2_study",
        side_effect=_fake_l2_study,
    )
    _crisis_rets = np.array([0.001, -0.002, 0.0005])
    _crisis_ctx = mocker.Mock()

    _run_robust_l2_l3_outcome(
        signal_batch=mocker.Mock(), aligned=mocker.Mock(), cfg=mocker.Mock(),
        window=mocker.Mock(), caps=mocker.Mock(), tf="4h", n_trials=1,
        target_phase="l2", l1_res=mocker.Mock(deployment_registry=mocker.Mock()),
        labeled_events=mocker.Mock(), per_tf_data_maps=None, labeled_events_by_tf=None,
        crisis_replay_ctx=_crisis_ctx, crisis_rets=_crisis_rets,
        l2_sim_cache=None, probe_manifest=None,
    )

    assert captured["crisis_rets"] is _crisis_rets
    assert captured["crisis_replay_ctx"] is _crisis_ctx


def test_run_robust_l2_l3_outcome_both_absent_stays_symmetric(mocker) -> None:
    from src.application.futures.runner.active_pipeline import _run_robust_l2_l3_outcome

    captured: dict = {}
    mocker.patch(
        "src.application.futures.runner.active_pipeline._run_tiered_l2_study",
        side_effect=lambda **kw: (captured.update(kw), mocker.Mock(blocker_reason="stub", best_params={}))[1],
    )

    _run_robust_l2_l3_outcome(
        signal_batch=mocker.Mock(), aligned=mocker.Mock(), cfg=mocker.Mock(),
        window=mocker.Mock(), caps=mocker.Mock(), tf="4h", n_trials=1,
        target_phase="l2", l1_res=mocker.Mock(deployment_registry=mocker.Mock()),
        labeled_events=mocker.Mock(), per_tf_data_maps=None, labeled_events_by_tf=None,
        crisis_replay_ctx=None, crisis_rets=None,
        l2_sim_cache=None, probe_manifest=None,
    )

    assert captured["crisis_rets"] is None
    assert captured["crisis_replay_ctx"] is None
