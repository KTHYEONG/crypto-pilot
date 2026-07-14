"[LIMIT-01] cross-TF HTF labeled events not discarded on empty raw_events."
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from src.domain.futures.alpha_foundry.bridge_helpers import AlphaFoundryL0Result
from src.domain.futures.alpha_foundry.contracts import L0SignalCandidate
from src.domain.futures.strategy.candidate_contracts import (
    CandidateSignalPanel,
)
from src.domain.futures.strategy.config import StrategyConfig
from src.domain.futures.strategy_runtime.bridge import run_candidate_strategy_for_universe


def _minimal_ohlc_bar() -> pd.DataFrame:
    """Return a single-row OHLC DataFrame that passes _build_virtual_probe_tf_maps."""
    return pd.DataFrame(
        {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5], "volume": [1000.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-01-01")], name="datetime"),
    )


def _make_panel(
    variant: str = "ema_12_72", native_tf: str = "8h",
) -> CandidateSignalPanel:
    return CandidateSignalPanel(
        family="trend_ma",
        variant=variant,
        params={},
        datetimes=np.asarray([np.datetime64("2026-01-01T00:00:00")], dtype="datetime64[ns]"),
        symbols=("BTCUSDT",),
        signed_score_2d=np.ones((1, 1), dtype=np.float64),
        side_hint_2d=np.ones((1, 1), dtype=np.int8),
        expected_holding_bars=1,
        min_holding_bars=1,
        stop_atr_mult=50.0,
        take_profit_atr_mult=50.0,
        turnover_proxy_2d=np.zeros((1, 1), dtype=np.float64),
        valid_mask_2d=np.ones((1, 1), dtype=bool),
        metadata={"native_tf": native_tf},
        archetype="trend",
    )


def _make_l0_candidate(recipe_id: str = "r1") -> L0SignalCandidate:
    return L0SignalCandidate(
        run_id="test", timeframe="8h", family="trend_ma", variant="ema_12_72",
        recipe_id=recipe_id, archetype="trend", source="synthetic_recipe",
        n_events=100, effective_n=50.0, mean_net_bps=1.0, block_lcb_bps=0.5,
        nw_tstat=1.5, bootstrap_lcb_bps=0.0, bootstrap_agree=True,
        cost_drag_ratio=0.3, turnover_per_year=50.0, max_abs_corr_in_bucket=0.0,
        tf_coverage_count=0, sign_agreement_ratio=0.0,
        corroboration_tier="single_tf_strict", discovery_tier="candidate",
        l1_priority_score=1.0, l1_budget_units=1,
        hard_reject_reasons=(), soft_flags=(),
    )


def _make_aligned() -> SimpleNamespace:
    return SimpleNamespace(
        datetimes=np.asarray([np.datetime64("2026-01-01T00:00:00")], dtype="datetime64[ns]"),
        symbols=("BTCUSDT",),
        close_2d=np.ones((1, 1), dtype=np.float64),
        open_2d=np.ones((1, 1), dtype=np.float64),
        high_2d=np.ones((1, 1), dtype=np.float64),
        low_2d=np.ones((1, 1), dtype=np.float64),
        volume_2d=np.ones((1, 1), dtype=np.float64),
        funding_2d=np.zeros((1, 1), dtype=np.float64),
        active_mask=np.ones((1, 1), dtype=bool),
        warm_mask=np.ones((1, 1), dtype=bool),
        entry_block_mask=np.zeros((1, 1), dtype=bool),
        kill_mask=np.zeros((1, 1), dtype=bool),
        execution_cost_bps_2d=np.zeros((1, 1), dtype=np.float64),
    )


def _make_multi_result(
    base_tf: str,
    htf_tfs: tuple[str, ...],
) -> dict[str, AlphaFoundryL0Result]:
    results: dict[str, AlphaFoundryL0Result] = {}
    for tf_k in (base_tf, *htf_tfs):
        panels: tuple[CandidateSignalPanel, ...]
        cands: tuple[L0SignalCandidate, ...]
        if tf_k == base_tf:
            panels = ()
            cands = ()
        else:
            panels = (_make_panel(variant=f"var_{tf_k}", native_tf=tf_k),)
            cands = (_make_l0_candidate(recipe_id=f"r_{tf_k}"),)
        results[tf_k] = AlphaFoundryL0Result(
            panels_for_l1=panels,
            summary_report=SimpleNamespace(n_passed=0),
            gate_results=(),
            panel_bindings=(),
            candidates_for_l1=cands,
        )
    return results


def _make_htf_projected_panels(htf_tfs: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(
        _make_panel(variant=f"var_{t}", native_tf=t)
        for t in htf_tfs
    )


def _mock_panels_to_events(
    panels: Any, *args: Any, **kwargs: Any,
) -> pd.DataFrame:
    if panels:
        rows = []
        for p in panels:
            variant = getattr(p, "variant", "var")
            rows.append({"score": 1.0, "variant": variant, "expected_holding_bars": 1})
        return pd.DataFrame(rows)
    return pd.DataFrame()


def _mock_label_events(events: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
    df = events.copy()
    df["entry_idx"] = 0
    df["exit_idx"] = 1
    df["entry_price"] = 100.0
    df["exit_price"] = 101.0
    df["gross_bps"] = 10.0
    df["net_bps"] = 9.5
    df["label"] = 1
    return df


def _setup_htf_mocks(
    monkeypatch: Any,
    aligned: Any,
    htf_tfs: tuple[str, ...],
    htf_projected: tuple[Any, ...],
    multi_results: dict[str, AlphaFoundryL0Result],
    label_mock: Any = _mock_label_events,
) -> None:
    monkeypatch.setattr(
        "src.domain.futures.strategy.common.alignment.align_data_maps",
        lambda *_, **__: aligned,
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.rule_signals.build_rule_signal_panels",
        lambda *_, **__: (_make_panel(native_tf="4h"),),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.rule_signals.candidate_panels_to_events",
        _mock_panels_to_events,
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.candidate_labels._compute_yang_zhang_vol_2d",
        lambda *_, **__: np.ones((1, 1), dtype=np.float64),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.candidate_labels.label_candidate_events",
        label_mock,
    )
    monkeypatch.setattr(
        "src.domain.futures.alpha_foundry.recipes.build_alpha_recipe_catalog",
        lambda *_, **__: [SimpleNamespace(recipe_id="r1")],
    )
    monkeypatch.setattr(
        "src.domain.futures.alpha_foundry.bridge_helpers.bind_panels_to_alpha_recipes",
        lambda *_, **__: [SimpleNamespace(recipe_id="r1", panel_index=0)],
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy_runtime.bridge.build_native_htf_panels",
        lambda *_, **__: {
            t: (aligned, (_make_panel(variant=f"native_{t}", native_tf=t),))
            for t in htf_tfs
        },
    )
    monkeypatch.setattr(
        "src.domain.futures.alpha_foundry.bridge_helpers.run_alpha_foundry_l0_gate_multi_tf",
        lambda *_, **__: multi_results,
    )
    monkeypatch.setattr(
        "src.domain.futures.alpha_foundry.bridge_helpers.assemble_l0_strategy_delivery_manifest",
        lambda *_, **__: (
            multi_results,
            SimpleNamespace(
                routes=(),
                independence_audit=None,
                pruning_status="disabled",
                pruning_reason="",
            ),
        ),
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy_runtime.bridge.project_htf_panels_to_base",
        lambda *_, **__: htf_projected,
    )
    monkeypatch.setattr(
        "src.domain.futures.strategy.candidate_portfolio.build_candidate_alpha_panel",
        lambda *_, **__: SimpleNamespace(score_2d=np.zeros((1, 1))),
    )


class TestBridgeCrossTFLabeledEvents:
    """[LIMIT-01] raw_events.empty early return and HTF-labeled events consistency."""

    def test_bridge_returns_htf_labeled_events_when_base_tf_raw_events_empty(
        self, monkeypatch: Any,
    ) -> None:
        base_tf = "4h"
        htf_tfs = ("8h", "12h")
        aligned = _make_aligned()
        htf_projected = _make_htf_projected_panels(htf_tfs)
        multi_results = _make_multi_result(base_tf, htf_tfs)

        _setup_htf_mocks(monkeypatch, aligned, htf_tfs, htf_projected, multi_results)

        af_config = SimpleNamespace(
            mode="gate",
            max_recipes_per_family=10,
            include_families=(),
            exclude_families=(),
            enable_synthetic_recipes=True,
            use_all_timeframes_in_l0=True,
            enable_cross_tf_pruning=False,
            enable_cross_tf_diversity_audit=False,
            enable_correlation_audit=False,
            l0_parallel_max_workers=1,
            l0_max_rss_mb=10_240,
            l0_memory_fraction_cap=0.60,
        )

        strat_cfg = StrategyConfig()
        object.__setattr__(
            strat_cfg,
            "candidate",
            replace(
                strat_cfg.candidate,
                l1_tfs=(base_tf, *htf_tfs),
            ),
        )

        result = run_candidate_strategy_for_universe(
            symbols=["BTCUSDT"],
            tf=base_tf,
            strategy_cfg=strat_cfg,
            preloaded_data_maps={"BTCUSDT": {"4h": _minimal_ohlc_bar()}},
            alpha_foundry_config=af_config,
        )

        assert result.labeled_unfiltered is not None
        assert not result.labeled_unfiltered.empty, (
            "base TF raw_events is empty, but cross-TF HTF panels exist: "
            "labeled_unfiltered must be non-empty"
        )
        assert "native_tf" in result.labeled_unfiltered.columns
        native_tfs = set(result.labeled_unfiltered["native_tf"])
        assert native_tfs & set(htf_tfs), (
            f"labeled_unfiltered must contain events from HTF TFs {htf_tfs}, "
            f"got {native_tfs}"
        )

    def test_bridge_labeled_unfiltered_stays_empty_when_both_base_and_htf_events_empty(
        self, monkeypatch: Any,
    ) -> None:
        aligned = _make_aligned()

        monkeypatch.setattr(
            "src.domain.futures.strategy.common.alignment.align_data_maps",
            lambda *_, **__: aligned,
        )
        monkeypatch.setattr(
            "src.domain.futures.strategy.rule_signals.build_rule_signal_panels",
            lambda *_, **__: (),
        )
        monkeypatch.setattr(
            "src.domain.futures.strategy.rule_signals.candidate_panels_to_events",
            lambda *_, **__: pd.DataFrame(),
        )
        monkeypatch.setattr(
            "src.domain.futures.strategy.candidate_portfolio.build_candidate_alpha_panel",
            lambda *_, **__: SimpleNamespace(score_2d=np.zeros((1, 1))),
        )

        result = run_candidate_strategy_for_universe(
            symbols=["BTCUSDT"],
            tf="4h",
            strategy_cfg=StrategyConfig(),
            preloaded_data_maps={"BTCUSDT": {"4h": _minimal_ohlc_bar()}},
        )

        assert result.labeled_unfiltered is not None
        assert result.labeled_unfiltered.empty

    def test_bridge_htf_only_labeling_failure_is_suppressed_not_propagated(
        self, monkeypatch: Any,
    ) -> None:
        base_tf = "4h"
        htf_tfs = ("8h",)
        aligned = _make_aligned()
        htf_projected = _make_htf_projected_panels(htf_tfs)
        multi_results = _make_multi_result(base_tf, htf_tfs)

        def _raise_label_error(*args: Any, **kwargs: Any) -> pd.DataFrame:
            raise ValueError("simulated labeling failure")

        _setup_htf_mocks(
            monkeypatch, aligned, htf_tfs, htf_projected, multi_results,
            label_mock=_raise_label_error,
        )

        af_config = SimpleNamespace(
            mode="gate",
            max_recipes_per_family=10,
            include_families=(),
            exclude_families=(),
            enable_synthetic_recipes=True,
            use_all_timeframes_in_l0=True,
            enable_cross_tf_pruning=False,
            enable_cross_tf_diversity_audit=False,
            enable_correlation_audit=False,
            l0_parallel_max_workers=1,
            l0_max_rss_mb=10_240,
            l0_memory_fraction_cap=0.60,
        )

        strat_cfg = StrategyConfig()
        object.__setattr__(
            strat_cfg,
            "candidate",
            replace(
                strat_cfg.candidate,
                l1_tfs=(base_tf, *htf_tfs),
            ),
        )

        result = run_candidate_strategy_for_universe(
            symbols=["BTCUSDT"],
            tf=base_tf,
            strategy_cfg=strat_cfg,
            preloaded_data_maps={"BTCUSDT": {"4h": _minimal_ohlc_bar()}},
            alpha_foundry_config=af_config,
        )

        assert result.labeled_unfiltered is not None
        assert result.labeled_unfiltered.empty
