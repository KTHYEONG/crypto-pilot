from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import numpy as np
import pytest

from src.domain.futures.alpha_foundry.bridge_helpers import (
    AlphaFoundryL0Result,
    bind_panels_to_alpha_recipes,
    build_cheap_gate_evidence_frame,
    run_alpha_foundry_l0_gate_multi_tf,
)
from src.domain.futures.alpha_foundry.contracts import (
    AlphaFoundryRuntimeConfig,
    AlphaGateConfig,
    AlphaRecipe,
)
from src.domain.futures.signals.contracts import CandidateSignalPanel
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.execution_cost import ExecutionCostModel


def make_aligned_for_tf(
    *, tf_hours: int, bars: int = 200, symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")
) -> AlignedMarketData:
    dt = np.arange(
        np.datetime64("2026-01-01T00:00:00"),
        np.datetime64("2026-01-01T00:00:00") + np.timedelta64(tf_hours * bars, "h"),
        np.timedelta64(tf_hours, "h"),
        dtype="datetime64[ns]",
    )
    t, n = dt.shape[0], len(symbols)
    close = 100.0 * np.exp(0.001 * np.arange(t, dtype=np.float64))[:, None] * np.ones((1, n))
    mask = np.ones((t, n), dtype=np.bool_)
    return AlignedMarketData(
        datetimes=dt, symbols=symbols,
        open_2d=close.copy(), high_2d=close * 1.01, low_2d=close * 0.99, close_2d=close,
        volume_2d=np.full((t, n), 1000.0), funding_2d=np.full((t, n), 0.00005),
        active_mask=mask, warm_mask=mask,
        entry_block_mask=np.zeros((t, n), dtype=np.bool_), kill_mask=np.zeros((t, n), dtype=np.bool_),
    )


def make_panel_for_tf(
    *,
    tf: str,
    family: str = "fam",
    variant: str = "var",
    bars: int = 200,
    n_symbols: int = 2,
    sign: float = 1.0,
) -> CandidateSignalPanel:
    score = np.full((bars, n_symbols), 0.0)
    side = np.zeros((bars, n_symbols), dtype=np.int8)
    for start in range(10, bars, 20):
        score[start, :] = sign * 0.8
        side[start:start + 3, :] = int(sign)
    return CandidateSignalPanel(
        family=family, variant=f"{variant}_{tf}",
        params={"lookback": 20}, datetimes=np.arange(bars, dtype=np.int64),
        symbols=tuple(f"S{i}" for i in range(n_symbols)),
        signed_score_2d=score, side_hint_2d=side,
        expected_holding_bars=3, min_holding_bars=1,
        stop_atr_mult=2.0, take_profit_atr_mult=4.0,
        turnover_proxy_2d=np.abs(np.diff(score, axis=0, prepend=0.0)),
        valid_mask_2d=np.ones((bars, n_symbols), dtype=np.bool_),
        metadata={"recipe_id": f"{family}:{variant}:{tf}"},
    )


def make_recipe_for_tf(*, tf: str, family: str = "fam", variant: str = "var") -> AlphaRecipe:
    return AlphaRecipe(
        recipe_id=f"{family}:{variant}:{tf}", family=family, variant=f"{variant}_{tf}",
        timeframe=tf, archetype="trend", indicator_params={"lookback": 20},
        side_rule_id="trend_follow", exit_policy_id="atr_trail_2",
        required_fields=("close",), causal_lag_bars=1, max_turnover_per_year=365.0,
    )


def make_gate_config() -> AlphaGateConfig:
    return AlphaGateConfig(
        min_events=1, min_effective_n=1.0, min_lcb_net_bps=-1000.0, min_nw_tstat=0.0,
        max_cost_drag_ratio=100.0, max_turnover_per_year=10000.0, min_candidate_rank_ic_tstat=0.0,
    )


RUNTIME_CONFIG = AlphaFoundryRuntimeConfig(
    mode="gate",
    cheap_gate=make_gate_config(),
    max_recipes_per_family=10,
)


def _build_panels_recipes_aligned(
    tfs: tuple[str, ...],
    *,
    sign_by_tf: dict[str, float] | None = None,
) -> tuple[
    dict[str, list[CandidateSignalPanel]],
    dict[str, MutableMapping[str, AlphaRecipe]],
    dict[str, AlignedMarketData],
    dict[str, list[Any]],
]:
    """Build panels/recipes/aligned per TF AND real bindings via
    bind_panels_to_alpha_recipes() (never empty bindings — an empty binding
    list makes run_alpha_foundry_l0_gate() bind zero panels, which silently
    produces empty evidence regardless of the gate logic under test)."""
    panels_by_tf: dict[str, list[CandidateSignalPanel]] = {}
    recipes_by_tf: dict[str, MutableMapping[str, AlphaRecipe]] = {}
    aligned_by_tf: dict[str, AlignedMarketData] = {}
    bindings_by_tf: dict[str, list[Any]] = {}
    for tf in tfs:
        tf_hours = int(tf.replace("h", ""))
        sign = sign_by_tf[tf] if sign_by_tf else 1.0
        aligned = make_aligned_for_tf(tf_hours=tf_hours, bars=200)
        panel = make_panel_for_tf(tf=tf, bars=200, sign=sign)
        recipe = make_recipe_for_tf(tf=tf)
        panels_by_tf[tf] = [panel]
        recipes_by_tf[tf] = {recipe.recipe_id: recipe}
        aligned_by_tf[tf] = aligned
        bindings_by_tf[tf] = list(
            bind_panels_to_alpha_recipes(
                panels=[panel],
                recipes=recipes_by_tf[tf],
                timeframe=tf,
                max_recipes_per_family=10,
                include_families=(),
                exclude_families=(),
                enable_synthetic_recipes=True,
            )
        )
        assert bindings_by_tf[tf], f"panel for tf={tf} failed to bind to its own recipe"
    return panels_by_tf, recipes_by_tf, aligned_by_tf, bindings_by_tf


class TestBuildCheapGateEvidenceFrame:
    """S1-3: build_cheap_gate_evidence_frame schema."""

    def test_returns_required_schema(self) -> None:
        aligned = make_aligned_for_tf(tf_hours=4, bars=200)
        panel = make_panel_for_tf(tf="4h", bars=200)
        recipe = make_recipe_for_tf(tf="4h")
        df = build_cheap_gate_evidence_frame(
            panels=[panel],
            recipes={recipe.recipe_id: recipe},
            aligned=aligned,
            cost_model=ExecutionCostModel(),
            cheap_gate_config=make_gate_config(),
            timeframe="4h",
        )
        _cols = {"family", "variant", "timeframe", "recipe_id", "reject_reasons", "mean_net_bps", "block_lcb_bps"}
        assert _cols.issubset(set(df.columns)), f"missing cols: {_cols - set(df.columns)}"
        assert len(df) == 1
        assert df["family"].iloc[0] == "fam"
        assert df["variant"].iloc[0] == "var_4h"
        assert df["timeframe"].iloc[0] == "4h"

    def test_empty_panels_returns_empty_df(self) -> None:
        aligned = make_aligned_for_tf(tf_hours=4, bars=200)
        df = build_cheap_gate_evidence_frame(
            panels=[],
            recipes={},
            aligned=aligned,
            cost_model=ExecutionCostModel(),
            cheap_gate_config=make_gate_config(),
            timeframe="4h",
        )
        assert df.empty
        _cols = {"family", "variant", "timeframe", "recipe_id", "reject_reasons", "mean_net_bps", "block_lcb_bps"}
        assert _cols.issubset(set(df.columns))

    def test_unknown_recipe_id_excluded(self) -> None:
        """E2-4: recipe_id without matching recipe is excluded."""
        aligned = make_aligned_for_tf(tf_hours=4, bars=200)
        panel = make_panel_for_tf(tf="4h", bars=200, family="unknown", variant="unk")
        recipe = make_recipe_for_tf(tf="4h")
        df = build_cheap_gate_evidence_frame(
            panels=[panel],
            recipes={recipe.recipe_id: recipe},
            aligned=aligned,
            cost_model=ExecutionCostModel(),
            cheap_gate_config=make_gate_config(),
            timeframe="4h",
        )
        assert len(df) == 0


class TestRunAlphaFoundryL0GateMultiTf:
    """S1-4: multi-TF gate basic behavior."""

    def test_returns_all_requested_tf_keys(self) -> None:
        tfs = ("4h", "6h", "8h", "12h")
        panels_by_tf, recipes_by_tf, aligned_by_tf, bindings_by_tf = _build_panels_recipes_aligned(tfs)

        results = run_alpha_foundry_l0_gate_multi_tf(
            panels_by_tf=panels_by_tf,
            bindings_by_tf=bindings_by_tf,
            recipes_by_tf=recipes_by_tf,
            aligned_by_tf=aligned_by_tf,
            cost_model=ExecutionCostModel(),
            runtime_config=RUNTIME_CONFIG,
            run_id_prefix="test_s1_4",
        )
        assert set(results) == set(tfs)
        for tf in tfs:
            assert isinstance(results[tf], AlphaFoundryL0Result)
            assert len(results[tf].evidence_rows) == 1, f"tf={tf} should bind and gate exactly 1 recipe"

    def test_raises_on_key_mismatch(self) -> None:
        """X3-1: panels_by_tf and aligned_by_tf key mismatch."""
        tfs = ("4h", "6h")
        panels_by_tf, recipes_by_tf, aligned_by_tf, bindings_by_tf = _build_panels_recipes_aligned(tfs)
        aligned_by_tf.pop("6h")

        with pytest.raises(ValueError, match="aligned_by_tf missing timeframe"):
            run_alpha_foundry_l0_gate_multi_tf(
                panels_by_tf=panels_by_tf,
                bindings_by_tf=bindings_by_tf,
                recipes_by_tf=recipes_by_tf,
                aligned_by_tf=aligned_by_tf,
                cost_model=ExecutionCostModel(),
                runtime_config=RUNTIME_CONFIG,
                run_id_prefix="test_x3_1",
            )

    def test_empty_tf_panels_returns_empty_result(self) -> None:
        """E2-1: specific TF with empty panels produces empty result, others unaffected."""
        tfs = ("4h", "6h")
        panels_by_tf, recipes_by_tf, aligned_by_tf, bindings_by_tf = _build_panels_recipes_aligned(tfs)
        panels_by_tf["6h"] = []
        bindings_by_tf["6h"] = []

        results = run_alpha_foundry_l0_gate_multi_tf(
            panels_by_tf=panels_by_tf,
            bindings_by_tf=bindings_by_tf,
            recipes_by_tf=recipes_by_tf,
            aligned_by_tf=aligned_by_tf,
            cost_model=ExecutionCostModel(),
            runtime_config=RUNTIME_CONFIG,
            run_id_prefix="test_e2_1",
        )
        assert "4h" in results
        assert "6h" in results
        assert len(results["6h"].panels_for_l1) == 0
        assert len(results["4h"].evidence_rows) == 1, "TF unaffected by sibling TF's empty panels"

    def test_cross_tf_corroborated_reaches_real_tf_corroboration(self) -> None:
        """S1-5: 4 TF all same sign positive -> corroboration -> tf_corroboration/corroboration_tier
        reflect real cross-TF agreement (previously always 0.0/"insufficient_coverage" due to the
        2-tuple vs 3-tuple tf_fusion_index key bug)."""
        tfs = ("4h", "6h", "8h", "12h")
        panels_by_tf, recipes_by_tf, aligned_by_tf, bindings_by_tf = _build_panels_recipes_aligned(
            tfs, sign_by_tf=dict.fromkeys(tfs, 1.0)
        )

        results = run_alpha_foundry_l0_gate_multi_tf(
            panels_by_tf=panels_by_tf,
            bindings_by_tf=bindings_by_tf,
            recipes_by_tf=recipes_by_tf,
            aligned_by_tf=aligned_by_tf,
            cost_model=ExecutionCostModel(),
            runtime_config=RUNTIME_CONFIG,
            run_id_prefix="test_s1_5",
        )
        for tf in tfs:
            rows = results[tf].evidence_rows
            assert len(rows) == 1, f"tf={tf} expected exactly 1 evidence row"
            row = rows[0]
            assert row.tf_corroboration == pytest.approx(1.0), (
                f"tf={tf}: same-sign panels across all 4 TFs must yield full corroboration, got {row.tf_corroboration}"
            )
            assert row.corroboration_tier == "corroborated", (
                f"tf={tf}: expected 'corroborated', got {row.corroboration_tier!r}"
            )
            assert row.handoff_tier != "blocked", f"tf={tf}: corroborated candidate must not be blocked"

    def test_mixed_sign_gives_contradicted_tier(self) -> None:
        """E2-3: 2 positive, 2 negative -> each TF's 3 siblings agree only 1/3 of the
        time (sign_agreement_ratio=1/3 <= 0.50 contradiction threshold) -> "contradicted",
        which forces tf_corroboration=0.0 (compute_tf_corroboration hard-zeros contradicted)."""
        tfs = ("4h", "6h", "8h", "12h")
        sign_by_tf = {tf: (1.0 if i < 2 else -1.0) for i, tf in enumerate(tfs)}
        panels_by_tf, recipes_by_tf, aligned_by_tf, bindings_by_tf = _build_panels_recipes_aligned(
            tfs, sign_by_tf=sign_by_tf
        )

        results = run_alpha_foundry_l0_gate_multi_tf(
            panels_by_tf=panels_by_tf,
            bindings_by_tf=bindings_by_tf,
            recipes_by_tf=recipes_by_tf,
            aligned_by_tf=aligned_by_tf,
            cost_model=ExecutionCostModel(),
            runtime_config=RUNTIME_CONFIG,
            run_id_prefix="test_e2_3",
        )
        for tf in tfs:
            rows = results[tf].evidence_rows
            assert len(rows) == 1, f"tf={tf} expected exactly 1 evidence row"
            row = rows[0]
            assert row.sign_agreement_ratio == pytest.approx(1.0 / 3.0), (
                f"tf={tf}: 1-of-3 sibling sign agreement expected, got {row.sign_agreement_ratio}"
            )
            assert row.corroboration_tier == "contradicted", (
                f"tf={tf}: expected 'contradicted', got {row.corroboration_tier!r}"
            )
            assert row.tf_corroboration == pytest.approx(0.0), (
                f"tf={tf}: contradicted tier must force tf_corroboration=0.0, got {row.tf_corroboration}"
            )
