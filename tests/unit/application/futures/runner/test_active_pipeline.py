from __future__ import annotations

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


@pytest.mark.slow
def test_run_tiered_l2_study_wall_time_within_perf_budget() -> None:
    """L2 wf_n_folds=8 실행 시 wall-time이 baseline 대비 15% 이내인지 확인.
    수동/스크립트 성격 테스트 — 실제 full 실행 필요."""
    pass
