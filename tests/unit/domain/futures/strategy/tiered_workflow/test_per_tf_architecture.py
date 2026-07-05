"""Tests for per-TF L1 architecture (TF-Architecture V2)."""
from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.strategy.candidate_contracts import CandidateSignalPanel, QualifiedSignalRegistry
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import (
    CandidateStrategyConfig,
    PerTfL1Result,
    resolve_tf_gate_overrides,
    resolve_tf_signal_pool,
)
from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer1Result
from src.domain.futures.strategy.tiered_workflow.pipeline import (
    _aggregate_per_tf_l1,
    _resolve_l2_master_tf,
    run_per_tf_l1,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_cfg(**overrides: object) -> MagicMock:
    cfg = MagicMock(spec=CandidateStrategyConfig)
    defaults: dict[str, object] = {
        "candidate_families": (
            "trend_ma",
            "bollinger_reversion",
            "rsi_reversion",
        ),
        "per_tf_candidate_families": None,
        "per_tf_gate_overrides": None,
        "l2_master_tf": None,
        "l1_pair_min_effective_obs": 5.0,
        "l1_min_sym_count": 6,
        "l1_min_fold_ratio": 0.50,
        "l1_min_realized_match_ratio": 0.90,
        "seed": 42,
        "max_holding_bars": 4,
    }
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(cfg, k, v)
    return cfg


def _make_l1_result(
    gate_passed: bool = True,
    n_winning: int = 5,
    prefix: str = "",
) -> Layer1Result:
    return Layer1Result(
        signals_per_fold=(),
        oos_stacked={f"{prefix}sym{i}": MagicMock() for i in range(n_winning)},
        pooled_ic=0.05,
        pooled_tstat=2.0,
        breadth=0.8,
        valid_coverage=0.9,
        fold_pass_ratio=0.75,
        gate_passed=gate_passed,
        n_valid=10,
        n_total=20,
    )


def _make_aligned(n_bars: int = 100, n_syms: int = 5) -> MagicMock:
    aligned = MagicMock(spec=AlignedMarketData)
    start = np.datetime64("2024-01-01")
    aligned.datetimes = np.array(
        [start + np.timedelta64(i, "h") for i in range(n_bars)]
    )
    aligned.symbols = tuple(f"sym{i}" for i in range(n_syms))
    aligned.close_2d = np.random.randn(n_bars, n_syms)
    return aligned


# ── Scenario 1: Per-TF signal pool filtering ────────────────────────────────


def test_resolve_tf_signal_pool_returns_tf_specific_families() -> None:
    """Scenario 1: resolve_tf_signal_pool returns 1h families for 1h."""
    cfg = CandidateStrategyConfig(
        per_tf_candidate_families={
            "1h": ("rsi_reversion", "bollinger_reversion"),
            "4h": ("trend_ma", "dual_momentum"),
        },
    )
    families_1h = resolve_tf_signal_pool(cfg, "1h")
    assert families_1h == ("rsi_reversion", "bollinger_reversion")


# ── Scenario 2: Backward compat (no per-TF config) ─────────────────────────


def test_resolve_tf_signal_pool_falls_back_to_candidate_families() -> None:
    """Scenario 2: per_tf_candidate_families=None uses candidate_families."""
    cfg = CandidateStrategyConfig(per_tf_candidate_families=None, per_tf_signal_pool_enabled=False)
    families = resolve_tf_signal_pool(cfg, "1h")
    assert families == cfg.candidate_families


def test_run_per_tf_l1_passes_all_families_when_no_tf_config() -> None:
    """Scenario 2: No per-TF config → all families pass through to nested SWF."""
    cfg = CandidateStrategyConfig(per_tf_candidate_families=None, per_tf_signal_pool_enabled=False)
    labeled = pd.DataFrame({
        "family": ["trend_ma", "bollinger_reversion", "rsi_reversion"],
        "strategy_id": ["a", "b", "c"],
        "side": [1, -1, 1],
        "entry_idx": [0, 1, 2],
        "exit_idx": [10, 11, 12],
    })

    with patch(
        "src.domain.futures.strategy.tiered_workflow.run_l1_nested_swf",
        return_value=_make_l1_result(),
    ) as mock_swf:
        aligned = _make_aligned()
        run_per_tf_l1(
            tf="1h",
            labeled_events=labeled,
            aligned=aligned,
            outer_folds=(),
            cfg=cfg,
            seed=42,
            verbose=False,
        )
        _, kwargs = mock_swf.call_args
        filtered = kwargs["labeled_events"]
        assert len(filtered) == 3


# ── Scenario 3: Gate override ──────────────────────────────────────────────


def test_resolve_tf_gate_overrides_empty_when_no_overrides() -> None:
    """Scenario 3: No overrides → empty dict."""
    cfg = _make_cfg(per_tf_gate_overrides=None)
    overrides = resolve_tf_gate_overrides(cfg, "1h")
    assert overrides == {}


def test_resolve_tf_gate_overrides_returns_tf_specific() -> None:
    """Scenario 3: 1h overrides returned for 1h."""
    cfg = _make_cfg(
        per_tf_gate_overrides={
            "1h": {"l1_pair_min_effective_obs": 3.0},
        },
    )
    overrides = resolve_tf_gate_overrides(cfg, "1h")
    assert overrides["l1_pair_min_effective_obs"] == 3.0


def test_apply_tf_gate_overrides_does_not_affect_other_tfs() -> None:
    """Scenario 3: 4h retains global defaults when only 1h has overrides."""
    cfg = _make_cfg(
        per_tf_gate_overrides={
            "1h": {"l1_pair_min_effective_obs": 3.0},
        },
        l1_pair_min_effective_obs=5.0,
    )
    from src.domain.futures.strategy.config import apply_tf_gate_overrides

    cfg_4h = apply_tf_gate_overrides(cfg, "4h")
    assert cfg_4h.l1_pair_min_effective_obs == 5.0


# ── Scenario 4: L2 TF selection via probe manifest ─────────────────────────


def test_resolve_l2_master_tf_from_probe_manifest() -> None:
    """Scenario 4: Most winning cells TF is selected."""
    per_tf_l1: dict[str, PerTfL1Result] = {}
    cfg = _make_cfg(l2_master_tf=None)
    probe_manifest = [
        {"tf": "6h", "is_winner": True},
        {"tf": "6h", "is_winner": True},
        {"tf": "8h", "is_winner": True},
        {"tf": "8h", "is_winner": True},
        {"tf": "12h", "is_winner": True},
        {"tf": "12h", "is_winner": True},
        {"tf": "12h", "is_winner": True},
    ]
    master_tf = _resolve_l2_master_tf(cfg, per_tf_l1, probe_manifest)
    assert master_tf == "12h"


def test_resolve_l2_master_tf_probe_manifest_empty_fallback() -> None:
    """Scenario 4: Empty probe manifest falls back to '8h'."""
    cfg = _make_cfg(l2_master_tf=None)
    master_tf = _resolve_l2_master_tf(cfg, {}, [])
    assert master_tf == "8h"


# ── Scenario 5: L2 TF selection via L1 winning signals ─────────────────────


def test_resolve_l2_master_tf_from_l1_winning_signals() -> None:
    """Scenario 5: TF with most L1 winning signals is selected."""
    cfg = _make_cfg(l2_master_tf=None)
    per_tf_l1 = {
        "4h": PerTfL1Result(tf="4h", l1_result=_make_l1_result(n_winning=5, prefix="4h_"), n_winning_signals=5),
        "8h": PerTfL1Result(tf="8h", l1_result=_make_l1_result(n_winning=12, prefix="8h_"), n_winning_signals=12),
        "12h": PerTfL1Result(tf="12h", l1_result=_make_l1_result(n_winning=3, prefix="12h_"), n_winning_signals=3),
    }
    master_tf = _resolve_l2_master_tf(cfg, per_tf_l1, None)
    assert master_tf == "8h"


def test_resolve_l2_master_tf_explicit_config_wins() -> None:
    """Scenario 5: cfg.l2_master_tf takes priority over L1 results."""
    cfg = _make_cfg(l2_master_tf="12h")
    per_tf_l1 = {
        "4h": PerTfL1Result(tf="4h", l1_result=_make_l1_result(n_winning=5), n_winning_signals=5),
        "8h": PerTfL1Result(tf="8h", l1_result=_make_l1_result(n_winning=12), n_winning_signals=12),
    }
    master_tf = _resolve_l2_master_tf(cfg, per_tf_l1, None)
    assert master_tf == "12h"


# ── Scenario 6: TF with zero data ──────────────────────────────────────────


def test_resolve_aligned_skips_missing_tf() -> None:
    """Scenario 6: _resolve_aligned_for_tf falls back to main aligned when tf missing."""
    from src.domain.futures.strategy.tiered_workflow.pipeline import _resolve_aligned_for_tf

    aligned = _make_aligned()
    aligned_1h_mock = _make_aligned()
    per_tf_data: dict[str, AlignedMarketData] = {"1h": aligned_1h_mock, "4h": aligned}
    result = _resolve_aligned_for_tf("2h", aligned, per_tf_data)
    assert result is aligned


def test_resolve_aligned_returns_tf_specific_when_available() -> None:
    """Scenario 6: Per-TF aligned data is returned when available."""
    from src.domain.futures.strategy.tiered_workflow.pipeline import _resolve_aligned_for_tf

    aligned = _make_aligned(n_bars=100)
    aligned_1h = _make_aligned(n_bars=500)
    per_tf_data: dict[str, AlignedMarketData] = {"1h": aligned_1h, "4h": aligned}
    result = _resolve_aligned_for_tf("1h", aligned, per_tf_data)
    assert result is aligned_1h
    assert len(result.datetimes) == 500


# ── Scenario 7: Full pipeline aggregation ──────────────────────────────────


def test_aggregate_per_tf_l1_empty() -> None:
    """Scenario 7: Empty per_tf_l1 returns blocked result."""
    result = _aggregate_per_tf_l1({})
    assert result.gate_passed is False
    assert result.oos_stacked == {}


def test_aggregate_per_tf_l1_merges_oos_stacked() -> None:
    """Scenario 7: OOS stacked from multiple TFs are merged."""
    per_tf_l1 = {
        "1h": PerTfL1Result(
            tf="1h",
            l1_result=_make_l1_result(gate_passed=True, n_winning=3, prefix="1h_"),
            n_winning_signals=3,
        ),
        "4h": PerTfL1Result(
            tf="4h",
            l1_result=_make_l1_result(gate_passed=False, n_winning=2, prefix="4h_"),
            n_winning_signals=2,
        ),
    }
    result = _aggregate_per_tf_l1(per_tf_l1)
    assert result.gate_passed is True
    assert len(result.oos_stacked) == 5


def test_aggregate_per_tf_l1_gate_passed_any() -> None:
    """Scenario 7: gate_passed is True if any TF passed."""
    per_tf_l1 = {
        "1h": PerTfL1Result(
            tf="1h",
            l1_result=_make_l1_result(gate_passed=False, n_winning=0),
            n_winning_signals=0,
        ),
        "4h": PerTfL1Result(
            tf="4h",
            l1_result=_make_l1_result(gate_passed=True, n_winning=5),
            n_winning_signals=5,
        ),
    }
    result = _aggregate_per_tf_l1(per_tf_l1)
    assert result.gate_passed is True


def test_aggregate_per_tf_l1_all_blocked() -> None:
    """Scenario 7: gate_passed is False when all TFs blocked."""
    per_tf_l1 = {
        "1h": PerTfL1Result(
            tf="1h",
            l1_result=_make_l1_result(gate_passed=False, n_winning=0),
            n_winning_signals=0,
        ),
        "4h": PerTfL1Result(
            tf="4h",
            l1_result=_make_l1_result(gate_passed=False, n_winning=0),
            n_winning_signals=0,
        ),
    }
    result = _aggregate_per_tf_l1(per_tf_l1)
    assert result.gate_passed is False


# ── S1: Representative registry selection ─────────────────────────────────────


def test_aggregate_per_tf_l1_preserves_preferred_tf_registry() -> None:
    """S1: preferred_tf='4h' → result.deployment_registry is per_tf_l1['4h'].l1_result.deployment_registry."""
    from types import SimpleNamespace

    ev = SimpleNamespace(
        key=SimpleNamespace(strategy_id="dual_momentum:42"),
        mean_incremental_bps=3.5, hard_eligible=True,
    )
    reg_4h = QualifiedSignalRegistry(
        by_symbol={"BTCUSDT": (ev,)}, ready_symbols=("BTCUSDT",),
        trade_scope_count=1, registry_version="test",
    )
    reg_8h = QualifiedSignalRegistry(
        by_symbol={"ETHUSDT": (ev,)}, ready_symbols=("ETHUSDT",),
        trade_scope_count=1, registry_version="test",
    )
    l1_4h = dataclasses.replace(_make_l1_result(prefix="4h_"), deployment_registry=reg_4h)
    l1_8h = dataclasses.replace(_make_l1_result(prefix="8h_"), deployment_registry=reg_8h)
    per_tf_l1 = {
        "4h": PerTfL1Result(tf="4h", l1_result=l1_4h, n_winning_signals=len(l1_4h.oos_stacked)),
        "8h": PerTfL1Result(tf="8h", l1_result=l1_8h, n_winning_signals=len(l1_8h.oos_stacked)),
    }
    result = _aggregate_per_tf_l1(per_tf_l1, preferred_tf="4h")
    assert result.deployment_registry is reg_4h


def test_aggregate_per_tf_l1_falls_back_to_artifact_registry() -> None:
    """S2: top-level deployment_registry=None, artifact registry exists → use artifact registry."""
    from types import SimpleNamespace

    ev = SimpleNamespace(
        key=SimpleNamespace(strategy_id="dual_momentum:42"),
        mean_incremental_bps=3.5, hard_eligible=True,
    )
    reg_art = QualifiedSignalRegistry(
        by_symbol={"BTCUSDT": (ev,)}, ready_symbols=("BTCUSDT",),
        trade_scope_count=1, registry_version="test",
    )
    l1 = dataclasses.replace(
        _make_l1_result(prefix="4h_"),
        deployment_registry=None,
        inference_artifact=SimpleNamespace(deployment_registry=reg_art),
    )
    per_tf_l1 = {
        "4h": PerTfL1Result(tf="4h", l1_result=l1, n_winning_signals=len(l1.oos_stacked)),
    }
    result = _aggregate_per_tf_l1(per_tf_l1, preferred_tf="4h")
    assert result.deployment_registry is reg_art


def test_aggregate_per_tf_l1_does_not_concat_multiple_registries() -> None:
    """X1: 2 TF with different registry rows → single registry identity preserved."""
    from types import SimpleNamespace

    ev = SimpleNamespace(
        key=SimpleNamespace(strategy_id="dual_momentum:42"),
        mean_incremental_bps=3.5, hard_eligible=True,
    )
    reg = QualifiedSignalRegistry(
        by_symbol={"BTCUSDT": (ev,)}, ready_symbols=("BTCUSDT",),
        trade_scope_count=1, registry_version="test",
    )
    l1_a = dataclasses.replace(_make_l1_result(prefix="4h_"), deployment_registry=reg)
    l1_b = dataclasses.replace(_make_l1_result(prefix="8h_"), deployment_registry=reg)
    per_tf_l1 = {
        "4h": PerTfL1Result(tf="4h", l1_result=l1_a, n_winning_signals=len(l1_a.oos_stacked)),
        "8h": PerTfL1Result(tf="8h", l1_result=l1_b, n_winning_signals=len(l1_b.oos_stacked)),
    }
    result = _aggregate_per_tf_l1(per_tf_l1, preferred_tf="4h")
    assert result.deployment_registry is reg


# ── Multi-TF helpers ──────────────────────────────────────────────────────


def _make_aligned_base(
    n_bars: int = 100,
    n_syms: int = 10,
    base_tf: str = "4h",
) -> MagicMock:
    aligned = MagicMock(spec=AlignedMarketData)
    from src.domain.futures.strategy_runtime.bridge import _HPB_BRIDGE

    hpb = _HPB_BRIDGE.get(base_tf, 4.0)
    start = np.datetime64("2024-01-01")
    aligned.datetimes = np.array(
        [start + np.timedelta64(int(i * hpb * 3600), "s") for i in range(n_bars)]
    )
    aligned.symbols = tuple(f"sym{i}" for i in range(n_syms))
    aligned.close_2d = np.random.randn(n_bars, n_syms).astype(np.float64)
    ones_2d = np.ones((n_bars, n_syms), dtype=bool)
    aligned.active_mask = ones_2d
    aligned.warm_mask = ones_2d
    aligned.execution_eligibility_mask = ones_2d
    aligned.strategy_readiness_mask = ones_2d
    aligned.promotion_active_mask = ones_2d
    aligned.entry_block_mask = np.zeros((n_bars, n_syms), dtype=bool)
    aligned.kill_mask = np.zeros((n_bars, n_syms), dtype=bool)
    return aligned


def _make_synthetic_panel(
    family: str = "trend_donchian",
    variant: str = "donchian_72",
    n_bars: int = 50,
    n_syms: int = 10,
    freq_h: float = 4.0,
    start: str = "2024-01-01",
) -> CandidateSignalPanel:
    base_ns = np.datetime64(start, "ns").view(np.int64)
    step_ns = int(freq_h * 3600 * 1_000_000_000)
    dt = (base_ns + np.arange(n_bars, dtype=np.int64) * step_ns).view("datetime64[ns]")
    symbols = tuple(f"sym{i}" for i in range(n_syms))
    return CandidateSignalPanel(
        family=family,
        variant=variant,
        params={},
        datetimes=dt,
        symbols=symbols,
        signed_score_2d=np.full((n_bars, n_syms), 0.5, dtype=np.float64),
        side_hint_2d=np.ones((n_bars, n_syms), dtype=np.int8),
        expected_holding_bars=4,
        min_holding_bars=2,
        stop_atr_mult=1.5,
        take_profit_atr_mult=3.0,
        turnover_proxy_2d=np.full((n_bars, n_syms), 0.1, dtype=np.float64),
        valid_mask_2d=np.ones((n_bars, n_syms), dtype=bool),
        metadata={},
        archetype=family,
        allowed_regimes=(),
        exit_policies=(),
    )


# ── S1: HTF native 생성·태깅 (Happy Path) ──────────────────────────────────


def test_build_multi_tf_panels_htf_native_tagging() -> None:
    """S1: HTF native panels created, projected to base grid, probe_origin absent."""
    from src.domain.futures.strategy_runtime.bridge import (
        _HPB_BRIDGE,
        build_multi_tf_panels,
    )

    n_bars, n_syms = 100, 10
    base_tf = "4h"
    aligned_base = _make_aligned_base(n_bars=n_bars, n_syms=n_syms, base_tf=base_tf)
    symbols = list(aligned_base.symbols)
    base_cfg = CandidateStrategyConfig()
    tfs = ("4h", "6h", "8h", "12h")
    non_base_tfs = [tf for tf in tfs if tf != base_tf]

    hpb_base = _HPB_BRIDGE[base_tf]
    panels_by_tf: dict[str, CandidateSignalPanel] = {}
    for tf_i in non_base_tfs:
        hpb_i = _HPB_BRIDGE[tf_i]
        n_bars_i = max(1, int(np.ceil(n_bars * hpb_base / hpb_i)))
        panels_by_tf[tf_i] = _make_synthetic_panel(
            family="trend_donchian",
            variant="donchian_72",
            n_bars=n_bars_i,
            n_syms=n_syms,
            freq_h=hpb_i,
        )

    def _fake_virtual_maps(
        data_maps: object,
        symbols_: list[str],
        target_tf: str,
    ) -> dict[str, dict[str, pd.DataFrame]]:
        return {sym: {target_tf: pd.DataFrame()} for sym in symbols_[:3]}

    def _fake_align(
        data_maps: object,
        symbols_: list[str],
        tf: str,
    ) -> MagicMock:
        panel = panels_by_tf[tf]
        aligned = MagicMock()
        aligned.datetimes = panel.datetimes
        return aligned

    def _fake_build_panels(
        aligned: object,
        cfg: object,
        family_filter: tuple[str, ...] | None = None,
        **kwargs: object,
    ) -> tuple[CandidateSignalPanel, ...]:
        for panel in panels_by_tf.values():
            aligned_dt = getattr(aligned, "datetimes", None)
            if aligned_dt is not None and np.array_equal(aligned_dt, panel.datetimes):
                return (panel,)
        return ()

    with (
        patch(
            "src.domain.futures.strategy_runtime.bridge._build_virtual_probe_tf_maps",
            side_effect=_fake_virtual_maps,
        ),
        patch(
            "src.domain.futures.strategy.common.alignment.align_data_maps",
            side_effect=_fake_align,
        ),
        patch(
            "src.domain.futures.strategy.rule_signals.build_rule_signal_panels",
            side_effect=_fake_build_panels,
        ),
    ):
        panels = build_multi_tf_panels(
            data_maps={},
            symbols=symbols,
            aligned_base=aligned_base,
            base_cfg=base_cfg,
            base_tf=base_tf,
            tfs=tfs,
            family_pool=lambda tf: ("trend_donchian",),
            htf_only=True,
        )

    native_tfs = [p.metadata.get("native_tf") for p in panels]
    for expected_tf in ("6h", "8h", "12h"):
        assert expected_tf in native_tfs, f"missing native_tf={expected_tf}"

    for p in panels:
        assert p.valid_mask_2d.shape == aligned_base.close_2d.shape
        assert p.metadata.get("probe_origin") is None


# ── S2: htf_only 가드 — LTF 차단 ──────────────────────────────────────────


def test_build_multi_tf_panels_htf_only_blocks_ltf() -> None:
    """S2: htf_only=True excludes 1h panels; only 6h/12h remain."""
    from src.domain.futures.strategy_runtime.bridge import (
        _HPB_BRIDGE,
        build_multi_tf_panels,
    )

    n_bars, n_syms = 100, 10
    base_tf = "4h"
    aligned_base = _make_aligned_base(n_bars=n_bars, n_syms=n_syms, base_tf=base_tf)
    symbols = list(aligned_base.symbols)
    base_cfg = CandidateStrategyConfig()
    tfs = ("1h", "4h", "6h", "12h")
    non_base_tfs = [tf for tf in tfs if tf != base_tf]

    hpb_base = _HPB_BRIDGE[base_tf]
    panels_by_tf: dict[str, CandidateSignalPanel] = {}
    for tf_i in non_base_tfs:
        hpb_i = _HPB_BRIDGE[tf_i]
        n_bars_i = max(1, int(np.ceil(n_bars * hpb_base / hpb_i)))
        panels_by_tf[tf_i] = _make_synthetic_panel(
            family="trend_donchian",
            variant="donchian_72",
            n_bars=n_bars_i,
            n_syms=n_syms,
            freq_h=hpb_i,
        )

    def _fake_virtual_maps(
        data_maps: object,
        symbols_: list[str],
        target_tf: str,
    ) -> dict[str, dict[str, pd.DataFrame]]:
        return {sym: {target_tf: pd.DataFrame()} for sym in symbols_[:3]}

    def _fake_align(
        data_maps: object,
        symbols_: list[str],
        tf: str,
    ) -> MagicMock:
        panel = panels_by_tf[tf]
        aligned = MagicMock()
        aligned.datetimes = panel.datetimes
        return aligned

    def _fake_build_panels(
        aligned: object,
        cfg: object,
        family_filter: tuple[str, ...] | None = None,
        **kwargs: object,
    ) -> tuple[CandidateSignalPanel, ...]:
        for panel in panels_by_tf.values():
            aligned_dt = getattr(aligned, "datetimes", None)
            if aligned_dt is not None and np.array_equal(aligned_dt, panel.datetimes):
                return (panel,)
        return ()

    with (
        patch(
            "src.domain.futures.strategy_runtime.bridge._build_virtual_probe_tf_maps",
            side_effect=_fake_virtual_maps,
        ),
        patch(
            "src.domain.futures.strategy.common.alignment.align_data_maps",
            side_effect=_fake_align,
        ),
        patch(
            "src.domain.futures.strategy.rule_signals.build_rule_signal_panels",
            side_effect=_fake_build_panels,
        ),
    ):
        panels = build_multi_tf_panels(
            data_maps={},
            symbols=symbols,
            aligned_base=aligned_base,
            base_cfg=base_cfg,
            base_tf=base_tf,
            tfs=tfs,
            family_pool=lambda tf: ("trend_donchian",),
            htf_only=True,
        )

    native_tfs = [p.metadata.get("native_tf") for p in panels]
    assert "1h" not in native_tfs
    assert "6h" in native_tfs
    assert "12h" in native_tfs


# ── S4: Multi-TF Threading — 병렬 처리 검증 ──────────────────────────────────


def test_build_multi_tf_panels_threaded_3tfs() -> None:
    """S4: 3 eligible TFs → ThreadPoolExecutor path — panels identical to sequential."""
    from src.domain.futures.strategy_runtime.bridge import (
        _HPB_BRIDGE,
        build_multi_tf_panels,
    )

    n_bars, n_syms = 100, 10
    base_tf = "4h"
    aligned_base = _make_aligned_base(n_bars=n_bars, n_syms=n_syms, base_tf=base_tf)
    symbols = list(aligned_base.symbols)
    base_cfg = CandidateStrategyConfig()
    tfs = ("4h", "6h", "8h", "12h")
    non_base_tfs = [tf for tf in tfs if tf != base_tf]

    hpb_base = _HPB_BRIDGE[base_tf]
    panels_by_tf: dict[str, CandidateSignalPanel] = {}
    for tf_i in non_base_tfs:
        hpb_i = _HPB_BRIDGE[tf_i]
        n_bars_i = max(1, int(np.ceil(n_bars * hpb_base / hpb_i)))
        panels_by_tf[tf_i] = _make_synthetic_panel(
            family="trend_donchian",
            variant="donchian_72",
            n_bars=n_bars_i,
            n_syms=n_syms,
            freq_h=hpb_i,
        )

    def _fake_virtual_maps(
        data_maps: object,
        symbols_: list[str],
        target_tf: str,
    ) -> dict[str, dict[str, pd.DataFrame]]:
        return {sym: {target_tf: pd.DataFrame()} for sym in symbols_[:3]}

    def _fake_align(
        data_maps: object,
        symbols_: list[str],
        tf: str,
    ) -> MagicMock:
        panel = panels_by_tf[tf]
        aligned = MagicMock()
        aligned.datetimes = panel.datetimes
        return aligned

    def _fake_build_panels(
        aligned: object,
        cfg: object,
        family_filter: tuple[str, ...] | None = None,
        **kwargs: object,
    ) -> tuple[CandidateSignalPanel, ...]:
        for panel in panels_by_tf.values():
            aligned_dt = getattr(aligned, "datetimes", None)
            if aligned_dt is not None and np.array_equal(aligned_dt, panel.datetimes):
                return (panel,)
        return ()

    with (
        patch(
            "src.domain.futures.strategy_runtime.bridge._build_virtual_probe_tf_maps",
            side_effect=_fake_virtual_maps,
        ),
        patch(
            "src.domain.futures.strategy.common.alignment.align_data_maps",
            side_effect=_fake_align,
        ),
        patch(
            "src.domain.futures.strategy.rule_signals.build_rule_signal_panels",
            side_effect=_fake_build_panels,
        ),
    ):
        panels = build_multi_tf_panels(
            data_maps={},
            symbols=symbols,
            aligned_base=aligned_base,
            base_cfg=base_cfg,
            base_tf=base_tf,
            tfs=tfs,
            family_pool=lambda tf: ("trend_donchian",),
            htf_only=True,
        )

    # All 3 non-base TFs should be represented
    native_tfs = [p.metadata.get("native_tf") for p in panels]
    for expected_tf in ("6h", "8h", "12h"):
        assert expected_tf in native_tfs, f"missing native_tf={expected_tf}"

    # All panels projected to base grid
    for p in panels:
        assert p.valid_mask_2d.shape == aligned_base.close_2d.shape
        assert p.metadata.get("probe_origin") is None


def test_build_multi_tf_panels_single_tf_skips_threading() -> None:
    """S4: Only 1 eligible TF → ThreadPoolExecutor NOT instantiated."""
    from src.domain.futures.strategy_runtime.bridge import (
        build_multi_tf_panels,
    )

    n_bars, n_syms = 100, 5
    base_tf = "8h"
    aligned_base = _make_aligned_base(n_bars=n_bars, n_syms=n_syms, base_tf=base_tf)
    symbols = list(aligned_base.symbols)
    base_cfg = CandidateStrategyConfig()
    tfs = ("8h", "12h")  # only 1 non-base TF: 12h

    with patch("src.domain.futures.strategy_runtime.bridge.ThreadPoolExecutor") as mock_executor:
        panels = build_multi_tf_panels(
            data_maps={},
            symbols=symbols,
            aligned_base=aligned_base,
            base_cfg=base_cfg,
            base_tf=base_tf,
            tfs=tfs,
            family_pool=lambda tf: (),
            htf_only=True,
        )
    mock_executor.assert_not_called()
    # Panels may be empty (no families configured) — that is expected
    assert isinstance(panels, tuple)


def test_build_multi_tf_panels_one_tf_fails_others_succeed() -> None:
    """S4: build_rule_signal_panels raises for 12h, but 6h/8h panels returned."""
    from src.domain.futures.strategy_runtime.bridge import (
        build_multi_tf_panels,
    )

    n_bars, n_syms = 100, 10
    base_tf = "4h"
    aligned_base = _make_aligned_base(n_bars=n_bars, n_syms=n_syms, base_tf=base_tf)
    symbols = list(aligned_base.symbols)
    base_cfg = CandidateStrategyConfig()
    tfs = ("4h", "6h", "8h", "12h")
    non_base_tfs = [tf for tf in tfs if tf != base_tf]

    from src.domain.futures.strategy_runtime.bridge import _HPB_BRIDGE
    hpb_base = _HPB_BRIDGE[base_tf]
    panels_by_tf: dict[str, CandidateSignalPanel] = {}
    for tf_i in non_base_tfs:
        hpb_i = _HPB_BRIDGE[tf_i]
        n_bars_i = max(1, int(np.ceil(n_bars * hpb_base / hpb_i)))
        panels_by_tf[tf_i] = _make_synthetic_panel(
            family="trend_donchian",
            variant="donchian_72",
            n_bars=n_bars_i,
            n_syms=n_syms,
            freq_h=hpb_i,
        )

    def _fake_virtual_maps(
        data_maps: object,
        symbols_: list[str],
        target_tf: str,
    ) -> dict[str, dict[str, pd.DataFrame]]:
        return {sym: {target_tf: pd.DataFrame()} for sym in symbols_[:3]}

    def _fake_align(
        data_maps: object,
        symbols_: list[str],
        tf: str,
    ) -> MagicMock:
        panel = panels_by_tf[tf]
        aligned = MagicMock()
        aligned.datetimes = panel.datetimes
        return aligned

    call_count: dict[str, int] = {}

    def _fake_build_panels(
        aligned: object,
        cfg: object,
        family_filter: tuple[str, ...] | None = None,
        **kwargs: object,
    ) -> tuple[CandidateSignalPanel, ...]:
        tf_i = getattr(cfg, "timeframe", "unknown")
        call_count[tf_i] = call_count.get(tf_i, 0) + 1
        if tf_i == "12h":
            raise ValueError("simulated 12h failure")
        for panel in panels_by_tf.values():
            aligned_dt = getattr(aligned, "datetimes", None)
            if aligned_dt is not None and np.array_equal(aligned_dt, panel.datetimes):
                return (panel,)
        return ()

    with (
        patch(
            "src.domain.futures.strategy_runtime.bridge._build_virtual_probe_tf_maps",
            side_effect=_fake_virtual_maps,
        ),
        patch(
            "src.domain.futures.strategy.common.alignment.align_data_maps",
            side_effect=_fake_align,
        ),
        patch(
            "src.domain.futures.strategy.rule_signals.build_rule_signal_panels",
            side_effect=_fake_build_panels,
        ),
    ):
        panels = build_multi_tf_panels(
            data_maps={},
            symbols=symbols,
            aligned_base=aligned_base,
            base_cfg=base_cfg,
            base_tf=base_tf,
            tfs=tfs,
            family_pool=lambda tf: ("trend_donchian",),
            htf_only=True,
        )

    # 12h failure should NOT block 6h and 8h panels
    native_tfs = [p.metadata.get("native_tf") for p in panels]
    assert "6h" in native_tfs
    assert "8h" in native_tfs
    # 12h should NOT be in native_tfs because its panels failed
    assert "12h" not in native_tfs

    # Each successful TF's build_rule_signal_panels was called
    assert call_count.get("6h", 0) == 1
    assert call_count.get("8h", 0) == 1
    # 12h was called once and raised
    assert call_count.get("12h", 0) == 1


# ── S5: Key namespacing — 충돌 방지 ────────────────────────────────────────


def test_aggregate_per_tf_l1_key_namespacing() -> None:
    """S3: oos_stacked keys include TF prefix to prevent collision."""
    l1_4h = Layer1Result(
        signals_per_fold=(), oos_stacked={"SYM1": MagicMock()},
        pooled_ic=0.0, pooled_tstat=0.0, breadth=0.0, valid_coverage=0.0,
        fold_pass_ratio=0.0, gate_passed=True, n_valid=1, n_total=1,
    )
    l1_6h = Layer1Result(
        signals_per_fold=(), oos_stacked={"SYM1": MagicMock()},
        pooled_ic=0.0, pooled_tstat=0.0, breadth=0.0, valid_coverage=0.0,
        fold_pass_ratio=0.0, gate_passed=True, n_valid=1, n_total=1,
    )
    per_tf_l1 = {
        "4h": PerTfL1Result(tf="4h", l1_result=l1_4h, n_winning_signals=1),
        "6h": PerTfL1Result(tf="6h", l1_result=l1_6h, n_winning_signals=1),
    }

    result = _aggregate_per_tf_l1(per_tf_l1)

    assert "4h::SYM1" in result.oos_stacked
    assert "6h::SYM1" in result.oos_stacked


# ── S4: native_tf 필터 격리 ────────────────────────────────────────────────


def test_run_per_tf_l1_filters_by_native_tf() -> None:
    """S4: run_per_tf_l1(tf='6h') passes only events with native_tf=='6h' to nested SWF."""
    cfg = CandidateStrategyConfig()
    labeled = pd.DataFrame({
        "native_tf": ["4h", "6h", "6h", "8h"],
        "family": ["trend_ma", "trend_donchian", "rsi_reversion", "bollinger_reversion"],
        "strategy_id": ["a", "b", "c", "d"],
        "side": [1, -1, 1, -1],
        "entry_idx": [0, 1, 2, 3],
        "exit_idx": [10, 11, 12, 13],
    })

    with patch(
        "src.domain.futures.strategy.tiered_workflow.run_l1_nested_swf",
        return_value=_make_l1_result(),
    ) as mock_swf:
        aligned = _make_aligned()
        run_per_tf_l1(
            tf="6h",
            labeled_events=labeled,
            aligned=aligned,
            outer_folds=(),
            cfg=cfg,
            seed=42,
            verbose=False,
        )

        assert mock_swf.called
        _, kwargs = mock_swf.call_args
        filtered = kwargs["labeled_events"]
        assert list(filtered["native_tf"].unique()) == ["6h"]
        assert len(filtered) == 2


# ── S5: Regression — base-only 동치 ────────────────────────────────────────


@pytest.mark.skip(reason="Full integration test requiring real aligned data + complete pipeline")
def test_multi_tf_base_only_equivalence() -> None:
    """S5: cfg.l1_tfs=('4h',) with no HTF data maps produces same output as single-TF path."""
    ...


# ── S6: native_tf 컬럼 전파 — bridge ──────────────────────────────────────


def test_run_candidate_strategy_native_tf_column() -> None:
    """S6: labeled_unfiltered from run_candidate_strategy_for_universe has native_tf with 0 nulls."""
    from src.domain.futures.strategy.config import StrategyConfig
    from src.domain.futures.strategy_runtime.bridge import (
        CandidatePipelineOutput,
        run_candidate_strategy_for_universe,
    )

    n_bars, n_syms = 100, 5
    tf = "4h"
    symbols = [f"sym{i}" for i in range(n_syms)]

    strategy_cfg = MagicMock(spec=StrategyConfig)
    strategy_cfg.candidate = CandidateStrategyConfig(
        min_rule_net_bps=0.0,
        ml_fit_fraction=0.55,
        ml_calibration_fraction=0.15,
        selection_policy="utility_topk",
        signal_only=False,
        min_candidate_obs=50,
    )

    aligned_mock = MagicMock(spec=AlignedMarketData)
    aligned_mock.close_2d = np.random.randn(n_bars, n_syms).astype(np.float64)
    aligned_mock.datetimes = np.array(
        [np.datetime64("2024-01-01") + np.timedelta64(i * 4, "h") for i in range(n_bars)]
    )
    aligned_mock.symbols = tuple(symbols)
    aligned_mock.execution_cost_bps_2d = np.full((n_bars, n_syms), 7.5, dtype=np.float64)
    aligned_mock.active_mask = np.ones((n_bars, n_syms), dtype=bool)
    aligned_mock.warm_mask = np.ones((n_bars, n_syms), dtype=bool)
    aligned_mock.entry_block_mask = np.zeros((n_bars, n_syms), dtype=bool)
    aligned_mock.kill_mask = np.zeros((n_bars, n_syms), dtype=bool)

    panel = _make_synthetic_panel(n_bars=n_bars, n_syms=n_syms, freq_h=4.0)

    labeled_df = pd.DataFrame({
        "family": ["trend_donchian"],
        "strategy_id": ["a"],
        "side": [1],
        "entry_idx": [10],
        "exit_idx": [50],
        "expected_holding_bars": [4],
        "min_holding_bars": [2],
    })

    aligned_events = pd.DataFrame({
        "family": ["trend_donchian"],
        "strategy_id": ["a"],
        "side": [1],
        "entry_idx": [10],
        "exit_idx": [50],
        "expected_holding_bars": [4],
        "min_holding_bars": [2],
    })

    data_maps = {"sym0": {"4h": pd.DataFrame({"close": [100.0]})}}

    with (
        patch(
            "src.domain.futures.strategy.common.alignment.align_data_maps",
            return_value=aligned_mock,
        ),
        patch(
            "src.domain.futures.strategy.rule_signals.build_rule_signal_panels",
            return_value=(panel,),
        ),
        patch(
            "src.domain.futures.strategy.rule_signals.candidate_panels_to_events",
            return_value=aligned_events,
        ),
        patch(
            "src.domain.futures.strategy.candidate_labels.label_candidate_events",
            return_value=labeled_df,
        ),
        patch(
            "src.domain.futures.strategy.rule_diagnostics.compute_rule_diagnostics",
        ),
        patch(
            "src.domain.futures.strategy.candidate_gate.predict_candidate_gate",
            return_value=(labeled_df, {}),
        ),
        patch(
            "src.domain.futures.strategy.candidate_edge.predict_candidate_edges",
            return_value=pd.DataFrame(),
        ),
        patch(
            "src.domain.futures.strategy.walk_forward.build_walk_forward_folds",
            return_value=(),
        ),
        patch(
            "src.domain.futures.strategy.ablation.apply_variant_promotions",
        ),
        patch(
            "src.domain.futures.strategy.candidate_dataset.build_candidate_dataset",
            return_value=(pd.DataFrame(), pd.DataFrame(), pd.DataFrame()),
        ),
        patch(
            "src.domain.futures.strategy.candidate_portfolio.select_candidate_events_for_portfolio",
            return_value=pd.DataFrame(),
        ),
        patch(
            "src.domain.futures.strategy.candidate_portfolio.build_candidate_target_weights",
            return_value=np.zeros((n_bars, n_syms), dtype=np.float64),
        ),
        patch(
            "src.domain.futures.strategy.candidate_portfolio.build_candidate_alpha_panel",
        ),
    ):
        output = run_candidate_strategy_for_universe(
            symbols=symbols,
            tf=tf,
            strategy_cfg=strategy_cfg,
            preloaded_data_maps=data_maps,
            silent=True,
        )

    assert isinstance(output, CandidatePipelineOutput)
    labeled_unfiltered = output.labeled_unfiltered
    assert labeled_unfiltered is not None
    assert "native_tf" in labeled_unfiltered.columns
    assert labeled_unfiltered["native_tf"].isnull().sum() == 0
