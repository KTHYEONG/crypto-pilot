"""Tests for per-TF L1 architecture (TF-Architecture V2)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

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
    cfg = _make_cfg(
        per_tf_candidate_families={
            "1h": ("rsi_reversion", "bollinger_reversion"),
            "4h": ("trend_ma", "dual_momentum"),
        },
    )
    families_1h = resolve_tf_signal_pool(cfg, "1h")
    assert families_1h == ("rsi_reversion", "bollinger_reversion")


def test_resolve_tf_signal_pool_filters_labeled_events() -> None:
    """Scenario 1: run_per_tf_l1 passes only matching families to nested SWF."""
    cfg = _make_cfg(
        per_tf_candidate_families={
            "1h": ("rsi_reversion",),
        },
    )
    labeled = pd.DataFrame({
        "family": ["rsi_reversion", "trend_ma", "rsi_reversion", "bollinger_reversion"],
        "strategy_id": ["a", "b", "c", "d"],
        "side": [1, -1, 1, -1],
        "entry_idx": [0, 1, 2, 3],
        "exit_idx": [10, 11, 12, 13],
    })

    with patch(
        "src.domain.futures.strategy.tiered_workflow.run_l1_nested_swf"
    ) as mock_swf:
        mock_swf.return_value = _make_l1_result()
        aligned = _make_aligned()
        outer_folds = ()
        run_per_tf_l1(
            tf="1h",
            labeled_events=labeled,
            aligned=aligned,
            outer_folds=outer_folds,
            cfg=cfg,
            seed=42,
            verbose=False,
        )
        assert mock_swf.called
        _, kwargs = mock_swf.call_args
        filtered = kwargs["labeled_events"]
        assert list(filtered["family"].unique()) == ["rsi_reversion"]


# ── Scenario 2: Backward compat (no per-TF config) ─────────────────────────


def test_resolve_tf_signal_pool_falls_back_to_candidate_families() -> None:
    """Scenario 2: per_tf_candidate_families=None uses candidate_families."""
    cfg = _make_cfg(per_tf_candidate_families=None)
    families = resolve_tf_signal_pool(cfg, "1h")
    assert families == cfg.candidate_families


def test_run_per_tf_l1_passes_all_families_when_no_tf_config() -> None:
    """Scenario 2: No per-TF config → all families pass through to nested SWF."""
    cfg = _make_cfg(per_tf_candidate_families=None)
    labeled = pd.DataFrame({
        "family": ["trend_ma", "bollinger_reversion", "rsi_reversion"],
        "strategy_id": ["a", "b", "c"],
        "side": [1, -1, 1],
        "entry_idx": [0, 1, 2],
        "exit_idx": [10, 11, 12],
    })

    with patch(
        "src.domain.futures.strategy.tiered_workflow.run_l1_nested_swf"
    ) as mock_swf:
        mock_swf.return_value = _make_l1_result()
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
