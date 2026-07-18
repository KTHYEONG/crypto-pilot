from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.strategy.candidate_contracts import (
    Layer1GateReport,
    QualifiedSignalRegistry,
    SignalSourceKey,
    SymbolStrategyEvidence,
)
from src.domain.futures.strategy.config import CandidateStrategyConfig, PerTfL1Result
from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer1Result, StrategySignal
from src.domain.futures.strategy.tiered_workflow.pipeline import (
    TieredPipelineError,
    _aggregate_per_tf_l1,
    _is_deployable_per_tf_result,
    _log_pertf_registry_diag,
    _resolve_l2_master_tf,
    _resolve_l2_master_tf_from_prior,
    _resolve_labeled_events_for_tf,
    _resolve_layer1_deployment_passed,
    _resolve_selected_l1_tf,
    _select_representative_l1_registry,
    run_per_tf_l1,
)


def _gate_report(*, structural: bool, strict: bool) -> Layer1GateReport:
    return Layer1GateReport(
        checks=(),
        passed=strict,
        blockers=() if strict else ("fold_ratio:0.250",),
        structural_passed=structural,
        advisory_checks=(),
    )


def _registry(*, ready_symbols: tuple[str, ...]) -> QualifiedSignalRegistry:
    return QualifiedSignalRegistry(
        by_symbol=dict.fromkeys(ready_symbols, ()),
        ready_symbols=ready_symbols,
        trade_scope_count=len(ready_symbols),
        registry_version="test",
    )


def _mock_per_tf(
    *,
    tf: str = "4h",
    gate_passed: bool = False,
    registry: QualifiedSignalRegistry | None = None,
    n_winning_signals: int = 0,
    strategy_panel_edge_bps: float = 0.0,
) -> PerTfL1Result:
    from src.domain.futures.strategy.tiered_workflow.dataclasses import StrategySignal

    panel: tuple[StrategySignal, ...] = ()
    if strategy_panel_edge_bps > 0.0:
        panel = (
            MagicMock(
                spec=StrategySignal,
                valid=True,
                oos_edge_bps=strategy_panel_edge_bps,
                quality_weight=1.0,
            ),
        )
    l1 = Layer1Result(
        signals_per_fold=(),
        oos_stacked={},
        pooled_ic=0.0,
        pooled_tstat=0.0,
        breadth=0.0,
        valid_coverage=0.0,
        fold_pass_ratio=0.0,
        gate_passed=gate_passed,
        n_valid=0,
        n_total=0,
        deployment_registry=registry,
        strategy_panel=panel,
    )
    m = MagicMock(spec=PerTfL1Result, tf=tf, l1_result=l1, n_winning_signals=n_winning_signals)
    m.l1_result = l1
    return m


def _registry_with_family_diversity(
    *,
    items_by_symbol: Mapping[str, tuple[str, ...]],
) -> QualifiedSignalRegistry:
    by_symbol = {
        sym: tuple(
            SymbolStrategyEvidence(
                key=SignalSourceKey(symbol=sym, strategy_id=sid, activation_context="all"),
                mean_gross_bps=50.0,
                mean_incremental_bps=20.0,
                p_value=0.02,
                q_value=0.08,
                positive_fold_ratio=0.8,
                n_obs=200,
                effective_n=150.0,
                n_folds=4,
                quality_weight=0.8,
                lcb_net_bps=15.0,
            )
            for sid in strategy_ids
        )
        for sym, strategy_ids in items_by_symbol.items()
    }
    return QualifiedSignalRegistry(
        by_symbol=by_symbol,
        ready_symbols=tuple(items_by_symbol.keys()),
        trade_scope_count=len(items_by_symbol),
        registry_version="test",
    )


# ── _resolve_layer1_deployment_passed ────────────────────────────────


class TestResolveLayer1DeploymentPassed:
    def test_s1_structural_only_and_registry_ready_returns_true(self) -> None:
        report = _gate_report(structural=True, strict=False)
        registry = _registry(ready_symbols=("BTCUSDT",))

        passed = _resolve_layer1_deployment_passed(
            gate_report=report,
            deployment_registry=registry,
            structural_gate_only=True,
        )

        assert passed is True
        assert report.passed is False

    def test_s2_conditional_path(self) -> None:
        report = _gate_report(structural=True, strict=False)
        registry = _registry(ready_symbols=("BTCUSDT",))

        passed = _resolve_layer1_deployment_passed(
            gate_report=report,
            deployment_registry=registry,
            structural_gate_only=True,
        )

        assert passed is True
        assert report.passed is False

    def test_s3_strict_mode_advisory_fails_returns_false(self) -> None:
        report = _gate_report(structural=True, strict=False)
        registry = _registry(ready_symbols=("BTCUSDT",))

        passed = _resolve_layer1_deployment_passed(
            gate_report=report,
            deployment_registry=registry,
            structural_gate_only=False,
        )

        assert passed is False

    def test_s4_empty_registry_returns_false(self) -> None:
        report = _gate_report(structural=True, strict=True)
        registry = _registry(ready_symbols=())

        passed = _resolve_layer1_deployment_passed(
            gate_report=report,
            deployment_registry=registry,
            structural_gate_only=True,
        )

        assert passed is False

    def test_s5_missing_registry_returns_false(self) -> None:
        report = _gate_report(structural=True, strict=True)

        passed = _resolve_layer1_deployment_passed(
            gate_report=report,
            deployment_registry=None,
            structural_gate_only=True,
        )

        assert passed is False

    def test_s6_structural_failure_returns_false(self) -> None:
        report = _gate_report(structural=False, strict=False)
        registry = _registry(ready_symbols=("BTCUSDT",))

        passed = _resolve_layer1_deployment_passed(
            gate_report=report,
            deployment_registry=registry,
            structural_gate_only=True,
        )

        assert passed is False


# ── _is_deployable_per_tf_result ──────────────────────────────────


class TestIsDeployablePerTfResult:
    def test_deployable_when_gate_passed_and_registry_with_ready(self) -> None:
        registry = _registry(ready_symbols=("BTCUSDT",))
        result = _mock_per_tf(gate_passed=True, registry=registry)

        assert _is_deployable_per_tf_result(result) is True

    def test_not_deployable_when_gate_failed(self) -> None:
        registry = _registry(ready_symbols=("BTCUSDT",))
        result = _mock_per_tf(gate_passed=False, registry=registry)

        assert _is_deployable_per_tf_result(result) is False

    def test_not_deployable_when_registry_none(self) -> None:
        result = _mock_per_tf(gate_passed=True, registry=None)

        assert _is_deployable_per_tf_result(result) is False

    def test_not_deployable_when_ready_empty(self) -> None:
        registry = _registry(ready_symbols=())
        result = _mock_per_tf(gate_passed=True, registry=registry)

        assert _is_deployable_per_tf_result(result) is False


# ── _resolve_selected_l1_tf ────────────────────────────────────────


class TestResolveSelectedL1Tf:
    def test_returns_preferred_tf_when_given_and_deployable(self) -> None:
        registry = _registry(ready_symbols=("BTCUSDT",))
        per_tf = {
            "4h": _mock_per_tf(tf="4h", gate_passed=True, registry=registry),
            "8h": _mock_per_tf(tf="8h", gate_passed=False, registry=None),
        }

        selected = _resolve_selected_l1_tf(per_tf, preferred_tf="4h")

        assert selected == "4h"

    def test_returns_none_when_preferred_tf_not_in_map(self) -> None:
        per_tf = {
            "4h": _mock_per_tf(tf="4h", gate_passed=True, registry=_registry(ready_symbols=("BTCUSDT",))),
        }

        selected = _resolve_selected_l1_tf(per_tf, preferred_tf="8h")

        assert selected is None

    def test_returns_min_tf_hours_among_deployable(self) -> None:
        registry_4h = _registry(ready_symbols=("BTCUSDT",))
        registry_8h = _registry(ready_symbols=("ETHUSDT",))
        per_tf = {
            "8h": _mock_per_tf(tf="8h", gate_passed=True, registry=registry_8h),
            "4h": _mock_per_tf(tf="4h", gate_passed=True, registry=registry_4h),
            "12h": _mock_per_tf(tf="12h", gate_passed=False, registry=None),
        }

        selected = _resolve_selected_l1_tf(per_tf, preferred_tf=None)

        assert selected == "4h"

    def test_returns_none_when_no_deployable_tf(self) -> None:
        per_tf = {
            "4h": _mock_per_tf(tf="4h", gate_passed=False, registry=None),
            "8h": _mock_per_tf(tf="8h", gate_passed=False, registry=None),
        }

        selected = _resolve_selected_l1_tf(per_tf, preferred_tf=None)

        assert selected is None

    def test_s10_empty_map_returns_none(self) -> None:
        selected = _resolve_selected_l1_tf({}, preferred_tf=None)

        assert selected is None


# ── _resolve_l2_master_tf ──────────────────────────────────────────


class TestResolveL2MasterTf:
    def test_s7_auto_master_selects_deployable_tf(self) -> None:
        cfg = MagicMock(
            spec=CandidateStrategyConfig,
            l2_master_tf=None,
            l2_master_min_ready_symbols=1,
            l2_master_min_source_families=1,
        )
        registry_8h = _registry_with_family_diversity(
            items_by_symbol={
                "ETHUSDT": ("family_a:v1",),
            }
        )
        per_tf = {
            "4h": _mock_per_tf(tf="4h", gate_passed=False, registry=None, n_winning_signals=100),
            "8h": _mock_per_tf(
                tf="8h",
                gate_passed=True,
                registry=registry_8h,
                n_winning_signals=20,
                strategy_panel_edge_bps=15.0,
            ),
        }

        master = _resolve_l2_master_tf(cfg, per_tf)

        assert master == "8h"

    def test_s8_explicit_override_must_be_deployable(self) -> None:
        cfg = MagicMock(spec=CandidateStrategyConfig, l2_master_tf="4h")
        registry_8h = _registry(ready_symbols=("ETHUSDT",))
        per_tf = {
            "4h": _mock_per_tf(tf="4h", gate_passed=False, registry=None, n_winning_signals=100),
            "8h": _mock_per_tf(tf="8h", gate_passed=True, registry=registry_8h, n_winning_signals=20),
        }

        with pytest.raises(TieredPipelineError, match="not deployable"):
            _resolve_l2_master_tf(cfg, per_tf)

    def test_auto_fails_closed_when_no_eligible(self) -> None:
        cfg = MagicMock(spec=CandidateStrategyConfig, l2_master_tf=None)
        per_tf = {
            "4h": _mock_per_tf(tf="4h", gate_passed=False, registry=None),
            "8h": _mock_per_tf(tf="8h", gate_passed=False, registry=None),
        }

        with pytest.raises(TieredPipelineError, match=r"deployable.*timeframe"):
            _resolve_l2_master_tf(cfg, per_tf)


# ── _select_representative_l1_registry ─────────────────────────────


class TestSelectRepresentativeL1Registry:
    def test_returns_registry_from_deployable_tf(self) -> None:
        registry_4h = _registry(ready_symbols=("BTCUSDT",))
        per_tf = {
            "4h": _mock_per_tf(tf="4h", gate_passed=True, registry=registry_4h),
            "8h": _mock_per_tf(tf="8h", gate_passed=False, registry=None),
        }

        reg = _select_representative_l1_registry(per_tf_l1=per_tf, preferred_tf=None)

        assert reg is registry_4h

    def test_returns_none_when_no_deployable_tf(self) -> None:
        per_tf = {
            "4h": _mock_per_tf(tf="4h", gate_passed=False, registry=None),
            "8h": _mock_per_tf(tf="8h", gate_passed=False, registry=None),
        }

        reg = _select_representative_l1_registry(per_tf_l1=per_tf, preferred_tf=None)

        assert reg is None

    def test_s10_empty_map_returns_none(self) -> None:
        reg = _select_representative_l1_registry(per_tf_l1={}, preferred_tf=None)

        assert reg is None


# ── _aggregate_per_tf_l1 ───────────────────────────────────────────


class TestAggregatePerTfL1:
    def test_s9_atomicity_aggregate_uses_master_tf_registry(self) -> None:
        registry_4h = _registry_with_family_diversity(
            items_by_symbol={
                "BTCUSDT": ("family_a:v1",),
            }
        )
        registry_8h = _registry_with_family_diversity(
            items_by_symbol={
                "ETHUSDT": ("family_a:v1",),
            }
        )
        per_tf = {
            "4h": _mock_per_tf(
                tf="4h",
                gate_passed=True,
                registry=registry_4h,
                n_winning_signals=10,
                strategy_panel_edge_bps=5.0,
            ),
            "8h": _mock_per_tf(
                tf="8h",
                gate_passed=True,
                registry=registry_8h,
                n_winning_signals=100,
                strategy_panel_edge_bps=50.0,
            ),
        }
        cfg = MagicMock(
            spec=CandidateStrategyConfig,
            l2_master_tf=None,
            l2_master_min_ready_symbols=1,
            l2_master_min_source_families=1,
        )
        master_tf = _resolve_l2_master_tf(cfg, per_tf)
        assert master_tf == "8h"

        agg = _aggregate_per_tf_l1(per_tf, preferred_tf=master_tf)

        assert agg.gate_passed is True
        assert agg.deployment_registry is registry_8h

    def test_s10_empty_map_returns_blocked(self) -> None:
        agg = _aggregate_per_tf_l1({})

        assert agg.gate_passed is False
        assert agg.deployment_registry is None

    def test_aggregate_blocked_when_preferred_tf_not_deployable(self) -> None:
        registry_8h = _registry(ready_symbols=("ETHUSDT",))
        per_tf = {
            "4h": _mock_per_tf(tf="4h", gate_passed=False, registry=None),
            "8h": _mock_per_tf(tf="8h", gate_passed=True, registry=registry_8h),
        }

        agg = _aggregate_per_tf_l1(per_tf, preferred_tf="4h")

        assert agg.gate_passed is False
        assert agg.deployment_registry is None


def test_s12_registry_diag_logs_strict_structural_and_advisory_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Conditional deployment status is distinguishable from strict audit status."""
    import src.domain.futures.strategy.tiered_workflow.pipeline as pipeline_module

    logger = MagicMock()
    monkeypatch.setattr(pipeline_module, "logger", logger)
    report = Layer1GateReport(
        checks=(),
        passed=False,
        blockers=("fold_ratio:0.250",),
        structural_passed=True,
        advisory_checks=(),
    )
    l1 = MagicMock(spec=Layer1Result)
    l1.gate_report = report
    l1.gate_passed = True
    l1.deployment_registry = _registry(ready_symbols=("BTCUSDT",))
    per_tf = {"4h": PerTfL1Result(tf="4h", l1_result=l1, n_winning_signals=1)}

    _log_pertf_registry_diag(per_tf, l2_tf_resolved="4h")

    args = logger.debug.call_args.args
    assert "strict_gate_passed=%s" in args[0]
    assert args[3] is False
    assert args[4] is True
    assert args[10] == "none"


# ── _resolve_labeled_events_for_tf ──────────────────────────────────


def test_resolve_labeled_events_for_tf_prefers_native_dict() -> None:
    pooled = pd.DataFrame({"entry_idx": [100, 200]})
    native_4h = pd.DataFrame({"entry_idx": [5, 10]})
    by_tf = {"4h": native_4h}

    result = _resolve_labeled_events_for_tf("4h", pooled, by_tf)

    assert result is native_4h


def test_resolve_labeled_events_for_tf_raises_when_missing() -> None:
    from src.domain.futures.strategy.event_grid_contracts import MissingNativeTfEventsError

    pooled = pd.DataFrame({"entry_idx": [100, 200]})
    by_tf = {"4h": pd.DataFrame({"entry_idx": [5, 10]})}

    with pytest.raises(MissingNativeTfEventsError, match="tf=8h"):
        _resolve_labeled_events_for_tf("8h", pooled, by_tf, require_native=True)


def test_resolve_labeled_events_for_tf_falls_back_when_require_native_false() -> None:
    pooled = pd.DataFrame({"entry_idx": [100, 200]})

    result = _resolve_labeled_events_for_tf("4h", pooled, None, require_native=False)

    assert result is pooled


def test_run_per_tf_l1_fails_on_oob_mismatch_entry_idx() -> None:
    from src.domain.futures.strategy.event_grid_contracts import EventGridContractError

    aligned = MagicMock()
    aligned.datetimes = np.array(
        ["2024-01-01T00:00", "2024-01-01T06:00", "2024-01-01T12:00"],
        dtype="datetime64[ns]",
    )

    cfg = CandidateStrategyConfig()
    labeled = pd.DataFrame(
        {
            "event_id": [1, 2],
            "datetime": [pd.Timestamp("2024-01-01T00:00Z"), pd.Timestamp("2024-01-01T12:00Z")],
            "entry_idx": [1, 5],
            "native_tf": ["6h", "6h"],
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "strategy_id": ["r1", "r1"],
        }
    )
    outer_folds: tuple[Any, ...] = ()

    with pytest.raises(EventGridContractError, match="timeframe=6h event_id=2"):
        run_per_tf_l1(
            tf="6h",
            labeled_events=labeled,
            aligned=aligned,
            outer_folds=outer_folds,
            cfg=cfg,
            seed=42,
        )


class TestResolveL2MasterTfMasterEligibleGate:
    def test_resolve_l2_master_tf_selects_highest_edge_quality_among_master_eligible(self) -> None:
        cfg = MagicMock(
            spec=CandidateStrategyConfig,
            l2_master_tf=None,
            l2_master_min_ready_symbols=5,
            l2_master_min_source_families=2,
        )
        registry_4h = _registry_with_family_diversity(
            items_by_symbol={f"SYM{i}": ("family_a:v1",) for i in range(5)} | {"SYM5": ("family_b:v2",)}
        )
        registry_8h = _registry_with_family_diversity(
            items_by_symbol={f"SYM{i}": ("family_a:v1",) for i in range(5)} | {"SYM5": ("family_b:v2",)}
        )
        per_tf = {
            "4h": _mock_per_tf(tf="4h", gate_passed=True, registry=registry_4h, strategy_panel_edge_bps=5.0),
            "8h": _mock_per_tf(tf="8h", gate_passed=True, registry=registry_8h, strategy_panel_edge_bps=15.0),
        }

        master = _resolve_l2_master_tf(cfg, per_tf)

        assert master == "8h"

    def test_resolve_l2_master_tf_rejects_narrow_tf_favors_diversified_master_eligible_tf(self) -> None:
        cfg = MagicMock(
            spec=CandidateStrategyConfig,
            l2_master_tf=None,
            l2_master_min_ready_symbols=5,
            l2_master_min_source_families=2,
        )
        registry_1d = _registry_with_family_diversity(
            items_by_symbol={
                "BTCUSDT": ("btc_regime_pullback:slow",),
                "ETHUSDT": ("btc_regime_pullback:slow",),
                "SOLUSDT": ("trend_donchian:72",),
            }
        )
        registry_4h = _registry_with_family_diversity(
            items_by_symbol={f"SYM{i}": ("family_a:v1",) for i in range(4)}
            | {"SYM4": ("family_b:v2",), "SYM5": ("family_b:v2",)}
        )
        per_tf = {
            "1d": _mock_per_tf(tf="1d", gate_passed=True, registry=registry_1d, strategy_panel_edge_bps=100.0),
            "4h": _mock_per_tf(tf="4h", gate_passed=True, registry=registry_4h, strategy_panel_edge_bps=5.0),
        }

        master = _resolve_l2_master_tf(cfg, per_tf)

        assert master == "4h"

    def test_resolve_l2_master_tf_raises_with_rejection_reasons_when_none_master_eligible(self) -> None:
        cfg = MagicMock(
            spec=CandidateStrategyConfig,
            l2_master_tf=None,
            l2_master_min_ready_symbols=5,
            l2_master_min_source_families=2,
        )
        registry_1sym = _registry_with_family_diversity(items_by_symbol={"BTCUSDT": ("family_a:v1",)})
        per_tf = {
            "4h": _mock_per_tf(tf="4h", gate_passed=True, registry=registry_1sym, strategy_panel_edge_bps=5.0),
            "8h": _mock_per_tf(tf="8h", gate_passed=True, registry=registry_1sym, strategy_panel_edge_bps=5.0),
        }

        with pytest.raises(TieredPipelineError, match=r"deployable.*timeframe") as exc_info:
            _resolve_l2_master_tf(cfg, per_tf)
        assert "ready_symbols 1 < 5" in str(exc_info.value)


class TestResolveL2MasterTfFromPrior:
    def test_reuses_prior_selected_timeframe_without_empty_dict_recompute(self) -> None:
        cfg = MagicMock(spec=CandidateStrategyConfig, l2_master_tf=None)
        prior = MagicMock(spec=Layer1Result, selected_timeframe="8h")

        result = _resolve_l2_master_tf_from_prior(prior, cfg)

        assert result == "8h"

    def test_falls_back_to_fail_closed_when_prior_has_no_selected_timeframe(self) -> None:
        cfg = MagicMock(spec=CandidateStrategyConfig, l2_master_tf=None)
        prior = MagicMock(spec=Layer1Result, selected_timeframe=None)

        with pytest.raises(TieredPipelineError, match=r"deployable.*timeframe"):
            _resolve_l2_master_tf_from_prior(prior, cfg)


class TestRunTieredPipelineSelectedTimeframe:
    def test_run_tiered_pipeline_l1_only_populates_selected_timeframe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """S4b: L1-only phase populates Layer1Result.selected_timeframe with the
        auto-resolved master TF (was always None before this fix)."""
        import datetime as _dt

        from src.domain.futures.optimization.opt_config import LayeredWindow
        from src.domain.futures.strategy.common.alignment import AlignedMarketData
        from src.domain.futures.strategy.tiered_workflow.pipeline import run_tiered_pipeline

        n_bars = 200
        datetimes = (np.datetime64("2024-01-01T00:00", "h") + np.arange(n_bars).astype("timedelta64[h]")).astype(
            "datetime64[ns]"
        )
        aligned = AlignedMarketData(
            datetimes=datetimes,
            symbols=("BTCUSDT",),
            open_2d=np.ones((n_bars, 1)),
            high_2d=np.ones((n_bars, 1)),
            low_2d=np.ones((n_bars, 1)),
            close_2d=np.ones((n_bars, 1)),
            volume_2d=np.ones((n_bars, 1)),
            funding_2d=np.zeros((n_bars, 1)),
            active_mask=np.ones(n_bars, dtype=bool),
            warm_mask=np.ones(n_bars, dtype=bool),
            entry_block_mask=np.zeros(n_bars, dtype=bool),
            kill_mask=np.zeros(n_bars, dtype=bool),
        )
        window = LayeredWindow(
            fetch_start=_dt.date(2024, 1, 1),
            l1_start=_dt.date(2024, 1, 1),
            l2_start=_dt.date(2024, 1, 5),
            holdout_start=_dt.date(2024, 1, 6),
            holdout_end=_dt.date(2024, 1, 8),
            regime_floor=_dt.date(2024, 1, 1),
        )
        cfg = CandidateStrategyConfig(l2_master_tf=None, l2_master_min_ready_symbols=5, l2_master_min_source_families=2)

        registry_8h = _registry_with_family_diversity(
            items_by_symbol={f"SYM{i}": ("family_a:v1",) for i in range(5)} | {"SYM5": ("family_b:v2",)}
        )
        per_tf_results = {
            "4h": _mock_per_tf(tf="4h", gate_passed=False, registry=None),
            "8h": _mock_per_tf(tf="8h", gate_passed=True, registry=registry_8h, strategy_panel_edge_bps=15.0),
        }

        def _fake_run_per_tf_l1(*, tf: str, **_kwargs: Any) -> PerTfL1Result:
            return per_tf_results[tf]

        import src.domain.futures.strategy.tiered_workflow.pipeline as pipeline_module

        monkeypatch.setattr(pipeline_module, "run_per_tf_l1", _fake_run_per_tf_l1)
        monkeypatch.setattr(
            "src.domain.futures.strategy.tiered_workflow.build_l1_nested_swf_folds",
            lambda **_kwargs: (),
        )

        empty_events = pd.DataFrame({"entry_idx": []})
        l1, l2, l3 = run_tiered_pipeline(
            labeled_events=empty_events,
            aligned=aligned,
            cfg=cfg,
            window=window,
            l1_params={},
            l2_params={},
            target_phase="l1",
            l1_tfs=("4h", "8h"),
            labeled_events_by_tf={"4h": empty_events, "8h": empty_events},
            verbose=False,
        )

        assert l1.selected_timeframe == "8h"
        assert l2 is None
        assert l3 is None


# ---------------------------------------------------------------------------
# S14: compute_crisis_unit_returns
# ---------------------------------------------------------------------------
class TestComputeCrisisUnitReturns:
    def test_compute_crisis_unit_returns_returns_empty_on_load_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """[S3] load_futures_data_maps_for_symbols 예외 시 빈 배열 반환."""
        from src.domain.futures.strategy.tiered_workflow.pipeline import compute_crisis_unit_returns

        registry = _registry(ready_symbols=("BTCUSDT",))

        def _mock_load_fail(*, symbols: Any, **kwargs: Any) -> Any:
            raise RuntimeError("mock data load failure")

        monkeypatch.setattr(
            "src.domain.futures.optimization.opt_data_utils.load_futures_data_maps_for_symbols",
            _mock_load_fail,
        )

        result = compute_crisis_unit_returns(
            deployment_registry=registry,
            strategy_cfg=CandidateStrategyConfig(),
            tf="4h",
        )
        assert isinstance(result, np.ndarray)
        assert result.size == 0
