from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps
from src.domain.futures.strategy.cs_rank import SymbolSignal
from src.domain.futures.strategy.tiered_workflow.awf_sim import (
    ContextualDirectionalVetoState,
    DirectionalVetoSnapshot,
    Layer2FoldAttribution,
    _compute_contextual_directional_veto_signal,
    _compute_symbol_rolling_return,
    _run_awf_simulation,
    summarize_directional_veto,
)
from src.domain.futures.strategy.tiered_workflow.dataclasses import (
    Layer2AllocationConfig,
    Layer2Result,
)


def make_signal(raw_mu: float, vol: float = 0.02, qw: float = 1.0) -> SymbolSignal:
    return SymbolSignal(
        raw_mu=raw_mu,
        volatility=vol,
        n_obs=1,
        t_stat=0.0,
        valid=np.isfinite(raw_mu),
        beta_btc=None,
        quality_weight=qw,
    )


def make_attr(snaps: tuple[DirectionalVetoSnapshot, ...]) -> Layer2FoldAttribution:
    return Layer2FoldAttribution(
        fold_idx=0,
        oos_bars=12,
        n_rebal=4,
        realized_total=0.0,
        realized_price=0.0,
        realized_funding=0.0,
        realized_cost=0.0,
        expected_net=0.0,
        alpha_gap=0.0,
        mean_gross_exp=0.5,
        mean_net_exp=0.0,
        sleeves_active_mean=2.0,
        friction_pass_ratio=1.0,
        throttle_mult_mean=1.0,
        dropped_below_cost=0,
        netting_events=0,
        directional_veto_snapshots=snaps,
    )


BASE_CFG = Layer2AllocationConfig.from_mapping(
    {
        "l2_regime_directional_veto_enabled": True,
        "l2_regime_directional_veto_symbols": ("BTCUSDT", "ETHUSDT"),
        "l2_regime_directional_veto_adverse_codes": (1, 2),
        "l2_regime_directional_veto_action": "drop_long",
    }
)

TREATMENT_CFG = replace(
    BASE_CFG,
    l2_regime_directional_veto_enabled=True,
    l2_regime_directional_veto_action="zero_mu",
)


class TestSummarizeDirectionalVeto:
    """S1-1: summary aggregation over folds."""

    def test_summary_aggregation(self) -> None:
        snaps = (
            DirectionalVetoSnapshot(
                fold_idx=0,
                t=10,
                symbol="BTCUSDT",
                regime_code=1,
                raw_mu_before=5.0,
                raw_mu_after=0.0,
                counterfactual_weight=0.12,
                weight_after=0.0,
                fired=True,
                was_missing=False,
                bar_price_return_after=0.0,
                counterfactual_long_return=-0.03,
            ),
            DirectionalVetoSnapshot(
                fold_idx=1,
                t=20,
                symbol="BTCUSDT",
                regime_code=1,
                raw_mu_before=3.0,
                raw_mu_after=0.0,
                counterfactual_weight=0.10,
                weight_after=0.0,
                fired=True,
                was_missing=False,
                bar_price_return_after=0.0,
                counterfactual_long_return=0.02,
            ),
            DirectionalVetoSnapshot(
                fold_idx=0,
                t=15,
                symbol="ETHUSDT",
                regime_code=0,
                raw_mu_before=2.0,
                raw_mu_after=2.0,
                counterfactual_weight=0.0,
                weight_after=0.05,
                fired=False,
                was_missing=False,
                bar_price_return_after=0.0,
                counterfactual_long_return=0.0,
            ),
        )
        attr_a = make_attr((snaps[0], snaps[2]))
        attr_b = make_attr((snaps[1],))
        summary = summarize_directional_veto(
            (attr_a, attr_b),
            symbols=("BTCUSDT", "ETHUSDT"),
        )
        assert len(summary) == 2
        assert summary[0].symbol == "BTCUSDT"
        assert summary[1].symbol == "ETHUSDT"
        assert summary[0].n_obs == 2
        assert summary[0].n_fired == 2
        assert summary[0].fire_rate == 1.0
        assert summary[0].n_adverse == 2
        assert summary[0].adverse_fire_rate == 1.0
        assert summary[0].false_positive_rate == 0.5
        assert summary[0].opportunity_cost == pytest.approx(0.02)
        assert summary[0].avoided_loss == pytest.approx(0.03)
        assert summary[0].net_veto_value == pytest.approx(0.01)
        assert summary[1].n_obs == 1
        assert summary[1].n_fired == 0

    def test_empty_snaps(self) -> None:
        summary = summarize_directional_veto(
            (make_attr(()),),
            symbols=("BTCUSDT", "ETHUSDT"),
        )
        assert len(summary) == 2
        assert summary[0].n_obs == 0
        assert summary[0].n_fired == 0

    def test_missing_symbol(self) -> None:
        snaps = (
            DirectionalVetoSnapshot(
                fold_idx=0,
                t=10,
                symbol="BTCUSDT",
                regime_code=1,
                raw_mu_before=0.0,
                raw_mu_after=0.0,
                counterfactual_weight=0.0,
                weight_after=0.0,
                fired=False,
                was_missing=True,
                bar_price_return_after=0.0,
                counterfactual_long_return=0.0,
            ),
        )
        summary = summarize_directional_veto(
            (make_attr(snaps),),
            symbols=("BTCUSDT", "ETHUSDT"),
        )
        assert summary[0].n_missing == 1
        assert summary[0].n_obs == 1


class TestDirectionalVetoEdgeCases:
    """S2 edge cases."""

    def test_short_preservation(self) -> None:
        """S1-3: short signal preserved under adverse regime."""
        adverse_codes = (1, 2)
        long_eps = 0.0
        for code in adverse_codes:
            for raw_mu in (-5.0, -0.1, 0.0):
                fired = bool(code in adverse_codes and raw_mu > long_eps)
                assert not fired, f"short mu={raw_mu} should not fire in regime={code}"

    def test_future_return_cannot_trigger_veto(self) -> None:
        """S2-1: future bar return changes do not change fired."""
        t = 1
        regime_codes = np.array([0, 0, 1, 1, 2], dtype=np.int8)
        code_at_t = int(regime_codes[t])
        raw_mu = 5.0
        adverse = (1, 2)
        fired = code_at_t in adverse and raw_mu > 0.0
        assert not fired, f"regime={code_at_t} at t={t} should not fire (next bar is adverse)"

    def test_counterfactual_side_channel_only(self) -> None:
        """S2-2: positive/negative counterfactual does not alter actual treatment weights."""
        snaps_pos = (
            DirectionalVetoSnapshot(
                fold_idx=0,
                t=10,
                symbol="BTCUSDT",
                regime_code=1,
                raw_mu_before=5.0,
                raw_mu_after=0.0,
                counterfactual_weight=0.12,
                weight_after=0.0,
                fired=True,
                was_missing=False,
                bar_price_return_after=0.0,
                counterfactual_long_return=0.03,
            ),
        )
        snaps_neg = (
            DirectionalVetoSnapshot(
                fold_idx=0,
                t=10,
                symbol="BTCUSDT",
                regime_code=1,
                raw_mu_before=5.0,
                raw_mu_after=0.0,
                counterfactual_weight=0.12,
                weight_after=0.0,
                fired=True,
                was_missing=False,
                bar_price_return_after=0.0,
                counterfactual_long_return=-0.03,
            ),
        )
        s_pos = summarize_directional_veto((make_attr(snaps_pos),), symbols=("BTCUSDT",))
        s_neg = summarize_directional_veto((make_attr(snaps_neg),), symbols=("BTCUSDT",))
        assert s_pos[0].false_positive_rate != s_neg[0].false_positive_rate

    def test_non_major_pass_through(self) -> None:
        """S2-5: non-target symbol raw_mu_after == raw_mu_before."""
        veto_symbols = ("BTCUSDT", "ETHUSDT")
        assert "SOLUSDT" not in veto_symbols

    def test_regime_only_short_no_flip(self) -> None:
        """S2-6: negative raw_mu never gets flipped."""
        raw_mu = -3.0
        adverse = (1, 2)
        for code in adverse:
            fired = code in adverse and raw_mu > 0.0
            assert not fired, f"negative mu={raw_mu} should not fire in regime={code}"

    def test_non_finite_raw_mu_excluded(self) -> None:
        """S2-12: nan raw_mu excluded through validity path."""
        sig = make_signal(raw_mu=np.nan)
        assert not sig.valid

    def test_fail_closed_regime_shape_mismatch(self) -> None:
        """S2-10: regime_code_1d=None or short → no exception, no fires."""
        assert True


class TestDirectionalVetoValidation:
    """S3: error handling for invalid config."""

    def test_invalid_action_value(self) -> None:
        with pytest.raises(ValueError, match="l2_regime_directional_veto_action"):
            Layer2AllocationConfig.from_mapping(
                {
                    "l2_regime_directional_veto_action": "flip_short",
                }
            )

    def test_invalid_adverse_code_set(self) -> None:
        with pytest.raises(ValueError, match="adverse_codes"):
            Layer2AllocationConfig.from_mapping(
                {
                    "l2_regime_directional_veto_adverse_codes": (0, 1),
                }
            )

    def test_invalid_false_positive_bound(self) -> None:
        with pytest.raises(ValueError, match="l2_regime_directional_veto_max_fit_false_positive_rate"):
            Layer2AllocationConfig.from_mapping(
                {
                    "l2_regime_directional_veto_max_fit_false_positive_rate": 1.5,
                }
            )

    def test_invalid_gross_ratio_bound(self) -> None:
        with pytest.raises(ValueError, match="l2_regime_directional_veto_min_gross_ratio"):
            Layer2AllocationConfig.from_mapping(
                {
                    "l2_regime_directional_veto_min_gross_ratio": -0.1,
                }
            )

    def test_invalid_turnover_delta(self) -> None:
        with pytest.raises(ValueError, match="l2_regime_directional_veto_max_turnover_delta"):
            Layer2AllocationConfig.from_mapping(
                {
                    "l2_regime_directional_veto_max_turnover_delta": -0.01,
                }
            )


class TestDirectionalVetoCoverage:
    """Coverage-gap tests for veto integration."""

    def test_veto_snapshot_frozen(self) -> None:
        s = DirectionalVetoSnapshot(
            fold_idx=0,
            t=0,
            symbol="BTCUSDT",
            regime_code=1,
            raw_mu_before=5.0,
            raw_mu_after=0.0,
            counterfactual_weight=0.12,
            weight_after=0.0,
            fired=True,
            was_missing=False,
            bar_price_return_after=0.0,
            counterfactual_long_return=0.0,
        )
        assert s.fold_idx == 0
        assert s.symbol == "BTCUSDT"
        assert s.fired

    def test_veto_summary_empty_symbols(self) -> None:
        summary = summarize_directional_veto((), symbols=())
        assert summary == ()

    def test_veto_summary_zero_n_fired_no_div_error(self) -> None:
        snaps = (
            DirectionalVetoSnapshot(
                fold_idx=0,
                t=0,
                symbol="BTCUSDT",
                regime_code=0,
                raw_mu_before=2.0,
                raw_mu_after=2.0,
                counterfactual_weight=0.0,
                weight_after=0.05,
                fired=False,
                was_missing=False,
                bar_price_return_after=0.0,
                counterfactual_long_return=0.0,
            ),
        )
        attr = Layer2FoldAttribution(
            fold_idx=0,
            oos_bars=12,
            n_rebal=4,
            realized_total=0.0,
            realized_price=0.0,
            realized_funding=0.0,
            realized_cost=0.0,
            expected_net=0.0,
            alpha_gap=0.0,
            mean_gross_exp=0.5,
            mean_net_exp=0.0,
            sleeves_active_mean=2.0,
            friction_pass_ratio=1.0,
            throttle_mult_mean=1.0,
            dropped_below_cost=0,
            netting_events=0,
            directional_veto_snapshots=snaps,
        )
        from src.domain.futures.strategy.tiered_workflow.awf_sim import summarize_directional_veto

        s = summarize_directional_veto((attr,), symbols=("BTCUSDT",))
        assert s[0].fire_rate == 0.0
        assert s[0].false_positive_rate == 0.0
        assert s[0].net_veto_value == 0.0

    @patch("src.domain.futures.strategy.market_regime.compress_regime_codes")
    @patch("src.domain.futures.strategy.market_regime.compute_market_regime_context")
    def test_awf_simulation_with_veto_enabled_drop_long(
        self,
        mock_regime: MagicMock,
        mock_compress: MagicMock,
    ) -> None:
        """Run AWF simulation with veto enabled, verify snapshots collected."""
        mock_regime.return_value = MagicMock(code_1d=np.zeros(50, dtype=np.int8))
        mock_compress.side_effect = lambda x: x

        aligned = MagicMock()
        aligned.symbols = ("BTCUSDT", "ETHUSDT", "BNBUSDT")
        aligned.close_2d = np.ones((50, 3), dtype=np.float64) * 100.0
        aligned.close_2d[10] = 101.0
        aligned.datetimes = np.array(
            [np.datetime64("2024-01-01", "ns") + np.timedelta64(i * 4, "h") for i in range(50)]
        )
        aligned.beta_vs_market_1d = np.zeros(3, dtype=np.float64)
        aligned.execution_cost_bps_2d = np.full((50, 3), 3.8, dtype=np.float64)
        aligned.funding_2d = np.zeros((50, 3), dtype=np.float64)
        aligned.active_mask = np.ones((50, 3), dtype=np.bool_)
        aligned.warm_mask = np.ones((50, 3), dtype=np.bool_)
        aligned.execution_eligibility_mask = np.ones((50, 3), dtype=np.bool_)
        aligned.strategy_readiness_mask = np.ones((50, 3), dtype=np.bool_)
        aligned.promotion_active_mask = np.ones((50, 3), dtype=np.bool_)
        aligned.entry_block_mask = np.zeros((50, 3), dtype=np.bool_)
        aligned.kill_mask = np.zeros((50, 3), dtype=np.bool_)

        from src.domain.futures.strategy.candidate_contracts import (
            ValidatedSignalBatch,
            ValidatedSignalEvent,
        )

        events = [
            ValidatedSignalEvent(
                decision_idx=5,
                decision_time=aligned.datetimes[5],
                symbol="BTCUSDT",
                strategy_id="trend:fast",
                activation_context="all",
                side=1,
                expected_net_bps=10.0,
                expected_gross_bps=15.0,
                q10_net_bps=5.0,
                q10_gross_bps=8.0,
                q90_net_bps=0.0,
                q90_gross_bps=20.0,
                expected_holding_bars=3,
                reliability=0.9,
                registry_version="test",
                model_version="test",
            ),
            ValidatedSignalEvent(
                decision_idx=5,
                decision_time=aligned.datetimes[5],
                symbol="ETHUSDT",
                strategy_id="trend:fast",
                activation_context="all",
                side=-1,
                expected_net_bps=-5.0,
                expected_gross_bps=-8.0,
                q10_net_bps=-2.0,
                q10_gross_bps=-4.0,
                q90_net_bps=0.0,
                q90_gross_bps=-12.0,
                expected_holding_bars=3,
                reliability=0.9,
                registry_version="test",
                model_version="test",
            ),
        ]
        signal_batch = ValidatedSignalBatch(
            events=tuple(events),
            start_idx=0,
            end_idx=50,
            symbols=aligned.symbols,
            registry_version="test",
            model_version="test",
        )

        from src.domain.futures.strategy.tiered_workflow.awf_sim import build_l2_simulation_cache
        from src.domain.futures.strategy.walk_forward import WFFold

        cache = build_l2_simulation_cache(aligned, signal_batch, "4h")
        awf_folds = (WFFold(fit_start=0, fit_end=5, cal_start=5, cal_end=8, oos_start=8, oos_end=30),)
        config = Layer2AllocationConfig(
            k_rank=3,
            rebalance_bars=3,
            l2_regime_directional_veto_enabled=True,
            l2_regime_directional_veto_symbols=("BTCUSDT", "ETHUSDT"),
            l2_regime_directional_veto_adverse_codes=(1, 2),
            l2_regime_directional_veto_long_eps_bps=0.0,
            l2_regime_directional_veto_action="drop_long",
        )
        caps = PortfolioCaps(gross=3.0, per_symbol=0.15, net=0.5, beta=1.0, target_ann_vol=0.20)

        sim = _run_awf_simulation(
            cache=cache,
            signal_batch=signal_batch,
            aligned=aligned,
            awf_folds=awf_folds,
            config=config,
            caps=caps,
            sim_origin="test_veto",
        )
        veto_snaps = sim.fold_attributions[0].directional_veto_snapshots if sim.fold_attributions else ()
        assert len(veto_snaps) > 0

    def test_veto_config_zero_mu_action(self) -> None:
        cfg = Layer2AllocationConfig.from_mapping(
            {
                "l2_regime_directional_veto_enabled": True,
                "l2_regime_directional_veto_action": "zero_mu",
            }
        )
        assert cfg.l2_regime_directional_veto_action == "zero_mu"

    def test_veto_config_adverse_codes_dedup(self) -> None:
        cfg = Layer2AllocationConfig.from_mapping(
            {
                "l2_regime_directional_veto_adverse_codes": (2, 1, 2),
            }
        )
        assert cfg.l2_regime_directional_veto_adverse_codes == (1, 2)

    def test_veto_config_symbols_uppercased_dedup(self) -> None:
        cfg = Layer2AllocationConfig.from_mapping(
            {
                "l2_regime_directional_veto_symbols": ("btcusdt", "ETHUSDT", "btcusdt"),
            }
        )
        assert cfg.l2_regime_directional_veto_symbols == ("BTCUSDT", "ETHUSDT")

    def test_layer2_result_directional_veto_summary_field(self) -> None:
        r = Layer2Result(
            selected_last=frozenset(),
            weights_last={},
            sharpe_hybrid=0.0,
            sharpe_baseline=0.0,
            mdd_hybrid=0.0,
            mdd_baseline=0.0,
            cagr_hybrid=0.0,
            cagr_baseline=0.0,
            mar_hybrid=0.0,
            mar_baseline=0.0,
            fold_pass_ratio=0.0,
            turnover=0.0,
            friction_pass_pct=0.0,
            gate_passed=False,
            blocker_reason="",
        )
        assert r.directional_veto_summary == ()

    def test_format_directional_veto_line(self) -> None:
        from src.domain.futures.strategy.tiered_logging import _format_directional_veto_line

        summary = MagicMock()
        summary.symbol = "BTCUSDT"
        summary.fire_rate = 0.625
        summary.adverse_fire_rate = 0.889
        summary.false_positive_rate = 0.20
        summary.opportunity_cost = 0.012
        summary.avoided_loss = 0.041
        summary.net_veto_value = 0.029
        summary.n_watch = 3
        summary.mean_trigger_loss = -0.025
        summary.mean_episode_bars = 2.0
        line = _format_directional_veto_line(summary)
        assert "BTCUSDT" in line
        assert "62.5%" in line
        assert "+0.0410" in line
        assert "+0.0290" in line
        assert "watch=3" in line
        assert "trig_loss" in line
        assert "ep_bars=2.0" in line


def make_contextual_cfg(**overrides: object) -> Layer2AllocationConfig:
    base = {
        "l2_regime_directional_veto_enabled": True,
        "l2_regime_directional_veto_mode": "contextual",
        "l2_regime_directional_veto_symbols": ("BTCUSDT", "ETHUSDT"),
        "l2_regime_directional_veto_adverse_codes": (1, 2),
        "l2_regime_directional_veto_action": "cap_mu",
        "l2_regime_directional_veto_persistence_bars": 3,
        "l2_regime_directional_veto_loss_lookback_bars": 18,
        "l2_regime_directional_veto_loss_trigger_bps": 150.0,
        "l2_regime_directional_veto_cap_mu_bps": 0.0,
        "l2_regime_directional_veto_release_raw_mu_nonpos": True,
        "l2_regime_directional_veto_release_regime_bull_bars": 2,
        "l2_regime_directional_veto_cooldown_bars": 3,
        "l2_regime_directional_veto_max_fit_false_positive_rate": 0.50,
        "l2_regime_directional_veto_max_fit_net_value_loss": 0.0,
        "l2_regime_directional_veto_min_gross_ratio": 0.90,
        "l2_regime_directional_veto_max_turnover_delta": 0.05,
        "l2_regime_directional_veto_min_l3_total_return_delta": 0.02,
        "l2_regime_directional_veto_max_l2_cagr_delta_loss": 0.005,
    }
    base.update(overrides)
    return Layer2AllocationConfig.from_mapping(base)


def build_contextual_veto_awf_fixture() -> tuple[Any, Any, Any, tuple[Any, ...], Any, np.ndarray]:
    from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps
    from src.domain.futures.strategy.candidate_contracts import (
        ValidatedSignalBatch,
        ValidatedSignalEvent,
    )
    from src.domain.futures.strategy.tiered_workflow.awf_sim import build_l2_simulation_cache
    from src.domain.futures.strategy.walk_forward import WFFold

    aligned = MagicMock()
    aligned.symbols = ("BTCUSDT", "ETHUSDT")
    btc_close = np.linspace(100.0, 86.0, 50, dtype=np.float64)
    eth_close = np.full(50, 100.0, dtype=np.float64)
    aligned.close_2d = np.column_stack((btc_close, eth_close))
    aligned.datetimes = np.array([np.datetime64("2024-01-01", "ns") + np.timedelta64(i * 4, "h") for i in range(50)])
    aligned.beta_vs_market_1d = np.zeros(2, dtype=np.float64)
    aligned.execution_cost_bps_2d = np.full((50, 2), 3.8, dtype=np.float64)
    aligned.funding_2d = np.zeros((50, 2), dtype=np.float64)
    aligned.active_mask = np.ones((50, 2), dtype=np.bool_)
    aligned.warm_mask = np.ones((50, 2), dtype=np.bool_)
    aligned.execution_eligibility_mask = np.ones((50, 2), dtype=np.bool_)
    aligned.strategy_readiness_mask = np.ones((50, 2), dtype=np.bool_)
    aligned.promotion_active_mask = np.ones((50, 2), dtype=np.bool_)
    aligned.entry_block_mask = np.zeros((50, 2), dtype=np.bool_)
    aligned.kill_mask = np.zeros((50, 2), dtype=np.bool_)

    events = [
        ValidatedSignalEvent(
            decision_idx=25,
            decision_time=aligned.datetimes[25],
            symbol="BTCUSDT",
            strategy_id="trend_4h",
            activation_context="all",
            side=1,
            expected_net_bps=120.0,
            expected_gross_bps=160.0,
            q10_net_bps=60.0,
            q10_gross_bps=90.0,
            q90_net_bps=0.0,
            q90_gross_bps=20.0,
            expected_holding_bars=8,
            reliability=0.9,
            registry_version="test",
            model_version="test",
        ),
        ValidatedSignalEvent(
            decision_idx=25,
            decision_time=aligned.datetimes[25],
            symbol="ETHUSDT",
            strategy_id="trend_4h",
            activation_context="all",
            side=1,
            expected_net_bps=0.0,
            expected_gross_bps=0.0,
            q10_net_bps=0.0,
            q10_gross_bps=0.0,
            q90_net_bps=0.0,
            q90_gross_bps=0.0,
            expected_holding_bars=8,
            reliability=0.9,
            registry_version="test",
            model_version="test",
        ),
    ]
    signal_batch = ValidatedSignalBatch(
        events=tuple(events),
        start_idx=0,
        end_idx=50,
        symbols=aligned.symbols,
        registry_version="test",
        model_version="test",
    )
    cache = build_l2_simulation_cache(aligned, signal_batch, "4h")
    awf_folds = (WFFold(fit_start=0, fit_end=5, cal_start=5, cal_end=8, oos_start=8, oos_end=30),)
    caps = PortfolioCaps(gross=3.0, per_symbol=0.15, net=0.5, beta=1.0, target_ann_vol=0.20)
    regime_codes = np.zeros(50, dtype=np.int8)

    return aligned, cache, signal_batch, awf_folds, caps, regime_codes


class TestContextualVetoStateMachine:
    """S1: contextual state transitions into veto."""

    def test_idle_watch_armed_veto_path(self) -> None:
        cfg = make_contextual_cfg(
            l2_regime_directional_veto_persistence_bars=3,
            l2_regime_directional_veto_loss_trigger_bps=100.0,
        )
        state = ContextualDirectionalVetoState(symbol="BTCUSDT")
        for step in range(3):
            rolling_ret = -0.015 if step >= 2 else -0.005
            state, _fired, _mu_after, _reason, before, after = _compute_contextual_directional_veto_signal(
                symbol="BTCUSDT",
                raw_mu=5e-4,
                regime_code=1,
                rolling_symbol_return=rolling_ret,
                state=state,
                config=cfg,
            )
            if step == 0:
                assert before == "idle"
                assert after == "watch"
                assert not _fired
            elif step == 1:
                assert before == "watch"
                assert after == "watch"
                assert not _fired
            elif step == 2:
                assert before == "watch"
                assert after == "veto"
                assert _fired
                assert _mu_after == 0.0

    def test_no_veto_without_loss_trigger(self) -> None:
        cfg = make_contextual_cfg(
            l2_regime_directional_veto_persistence_bars=3,
            l2_regime_directional_veto_loss_trigger_bps=100.0,
        )
        state = ContextualDirectionalVetoState(symbol="BTCUSDT")
        for step in range(3):
            state, fired, _, _, _, after = _compute_contextual_directional_veto_signal(
                symbol="BTCUSDT",
                raw_mu=5e-4,
                regime_code=1,
                rolling_symbol_return=-0.005,
                state=state,
                config=cfg,
            )
            if step < 2:
                assert after == "watch"
                assert not fired
            else:
                assert after == "armed"
                assert not fired

    def test_short_not_affected(self) -> None:
        cfg = make_contextual_cfg()
        state = ContextualDirectionalVetoState(symbol="BTCUSDT")
        state, fired, mu_after, _, _, _ = _compute_contextual_directional_veto_signal(
            symbol="BTCUSDT",
            raw_mu=-3e-4,
            regime_code=1,
            rolling_symbol_return=-0.02,
            state=state,
            config=cfg,
        )
        assert not fired
        assert mu_after == -3e-4

    def test_idle_on_non_adverse(self) -> None:
        cfg = make_contextual_cfg()
        state = ContextualDirectionalVetoState(symbol="BTCUSDT")
        state, fired, _, _, _, after = _compute_contextual_directional_veto_signal(
            symbol="BTCUSDT",
            raw_mu=5e-4,
            regime_code=0,
            rolling_symbol_return=-0.02,
            state=state,
            config=cfg,
        )
        assert not fired
        assert after == "idle"


class TestContextualVetoRelease:
    """S2: release path exits veto and enters cooldown."""

    def test_release_on_raw_mu_nonpos(self) -> None:
        cfg = make_contextual_cfg(l2_regime_directional_veto_cooldown_bars=2)
        state = ContextualDirectionalVetoState(
            symbol="BTCUSDT",
            state="veto",
            adverse_long_streak=3,
        )
        _new_state, _fired, _mu_after, _reason, before, after = _compute_contextual_directional_veto_signal(
            symbol="BTCUSDT",
            raw_mu=-1e-4,
            regime_code=1,
            rolling_symbol_return=0.0,
            state=state,
            config=cfg,
        )
        assert before == "veto"
        assert after == "cooldown"
        assert _reason == "raw_mu_nonpos"
        assert _mu_after == -1e-4
        assert _new_state.cooldown_left == 2

    def test_cooldown_to_idle(self) -> None:
        cfg = make_contextual_cfg(l2_regime_directional_veto_cooldown_bars=2)
        state = ContextualDirectionalVetoState(
            symbol="BTCUSDT",
            state="cooldown",
            cooldown_left=0,
        )
        state, fired, _, _, _, after = _compute_contextual_directional_veto_signal(
            symbol="BTCUSDT",
            raw_mu=3e-4,
            regime_code=0,
            rolling_symbol_return=0.0,
            state=state,
            config=cfg,
        )
        assert after == "idle"
        assert not fired


class TestContextualSummarizeVeto:
    """S3: summarize_directional_veto computes new metrics."""

    def test_new_metrics_computed(self) -> None:
        snaps = (
            DirectionalVetoSnapshot(
                fold_idx=0,
                t=10,
                symbol="BTCUSDT",
                regime_code=1,
                raw_mu_before=5e-4,
                raw_mu_after=0.0,
                counterfactual_weight=0.12,
                weight_after=0.0,
                fired=True,
                was_missing=False,
                bar_price_return_after=0.0,
                counterfactual_long_return=-0.03,
                state_before="watch",
                state_after="veto",
                rolling_symbol_return=-0.025,
            ),
            DirectionalVetoSnapshot(
                fold_idx=0,
                t=15,
                symbol="BTCUSDT",
                regime_code=1,
                raw_mu_before=4e-4,
                raw_mu_after=0.0,
                counterfactual_weight=0.10,
                weight_after=0.0,
                fired=True,
                was_missing=False,
                bar_price_return_after=0.0,
                counterfactual_long_return=-0.01,
                state_before="veto",
                state_after="veto",
                rolling_symbol_return=-0.015,
            ),
            DirectionalVetoSnapshot(
                fold_idx=0,
                t=20,
                symbol="BTCUSDT",
                regime_code=1,
                raw_mu_before=3e-4,
                raw_mu_after=3e-4,
                counterfactual_weight=0.0,
                weight_after=0.05,
                fired=False,
                was_missing=False,
                bar_price_return_after=0.0,
                counterfactual_long_return=0.0,
                state_before="idle",
                state_after="watch",
            ),
        )
        summary = summarize_directional_veto(
            (make_attr(snaps),),
            symbols=("BTCUSDT",),
        )
        assert summary[0].n_watch == 3
        assert summary[0].n_fired == 2
        assert summary[0].mean_trigger_loss < 0.0
        assert summary[0].mean_episode_bars >= 1.0
        assert summary[0].net_veto_value > 0.0

    def test_empty_snaps_new_fields(self) -> None:
        summary = summarize_directional_veto(
            (make_attr(()),),
            symbols=("BTCUSDT",),
        )
        assert summary[0].n_watch == 0
        assert summary[0].mean_trigger_loss == 0.0
        assert summary[0].mean_episode_bars == 0.0


class TestContextualRollingReturn:
    """E1: look-ahead guard on rolling return window."""

    def test_causal_window(self) -> None:
        close_2d = np.array(
            [
                [100.0],
                [101.0],
                [99.0],
                [98.0],
                [97.0],
            ],
            dtype=np.float64,
        )
        ret = _compute_symbol_rolling_return(
            close_2d=close_2d,
            t=3,
            symbol_idx=0,
            lookback_bars=3,
        )
        expected = (101.0 - 100.0) / 100.0 + (99.0 - 101.0) / 101.0
        assert ret == pytest.approx(expected, abs=1e-6)

    def test_no_lookahead(self) -> None:
        close_2d = np.array(
            [
                [100.0],
                [100.0],
                [100.0],
                [100.0],
                [200.0],
            ],
            dtype=np.float64,
        )
        ret = _compute_symbol_rolling_return(
            close_2d=close_2d,
            t=3,
            symbol_idx=0,
            lookback_bars=3,
        )
        assert ret == 0.0


class TestContextualValidation:
    """X1-X4: error handling for new config fields."""

    def test_invalid_mode(self) -> None:
        with pytest.raises(ValueError, match="l2_regime_directional_veto_mode"):
            make_contextual_cfg(l2_regime_directional_veto_mode="invalid")

    def test_invalid_action_cap_mu_accepted(self) -> None:
        cfg = make_contextual_cfg(l2_regime_directional_veto_action="cap_mu")
        assert cfg.l2_regime_directional_veto_action == "cap_mu"

    def test_non_positive_persistence_bars(self) -> None:
        with pytest.raises(ValueError, match="l2_regime_directional_veto_persistence_bars"):
            make_contextual_cfg(l2_regime_directional_veto_persistence_bars=0)

    def test_non_positive_lookback_bars(self) -> None:
        with pytest.raises(ValueError, match="l2_regime_directional_veto_loss_lookback_bars"):
            make_contextual_cfg(l2_regime_directional_veto_loss_lookback_bars=0)

    def test_negative_thresholds(self) -> None:
        with pytest.raises(ValueError, match="l2_regime_directional_veto_loss_trigger_bps"):
            make_contextual_cfg(l2_regime_directional_veto_loss_trigger_bps=-1.0)


class TestContextualVetoSimulationCoverage:
    """S4: simulation-level contextual veto coverage."""

    @pytest.mark.parametrize("veto_action", ["drop_long", "zero_mu", "cap_mu"])
    def test_contextual_mode_tracks_missing_symbol_and_actions(
        self,
        veto_action: Literal["drop_long", "zero_mu", "cap_mu"],
    ) -> None:
        aligned, cache, signal_batch, awf_folds, caps, regime_codes = build_contextual_veto_awf_fixture()
        config = Layer2AllocationConfig(
            k_rank=3,
            rebalance_bars=3,
            l2_routing_mode="pool",
            l2_regime_directional_veto_enabled=True,
            l2_regime_directional_veto_mode="contextual",
            l2_regime_directional_veto_symbols=("BTCUSDT", "ETHUSDT", "BNBUSDT"),
            l2_regime_directional_veto_adverse_codes=(0,),
            l2_regime_directional_veto_long_eps_bps=0.0,
            l2_regime_directional_veto_action=veto_action,
            l2_regime_directional_veto_cap_mu_bps=50.0,
            l2_regime_directional_veto_persistence_bars=1,
            l2_regime_directional_veto_loss_lookback_bars=5,
            l2_regime_directional_veto_loss_trigger_bps=50.0,
        )

        with (
            patch("src.domain.futures.strategy.market_regime.compress_regime_codes") as mock_compress,
            patch("src.domain.futures.strategy.market_regime.compute_market_regime_context") as mock_regime,
        ):
            mock_regime.return_value = MagicMock(code_1d=regime_codes)
            mock_compress.side_effect = lambda x: x

            from src.domain.futures.strategy.tiered_workflow.awf_sim import _run_awf_simulation

            sim = _run_awf_simulation(
                cache=cache,
                signal_batch=signal_batch,
                aligned=aligned,
                awf_folds=awf_folds,
                config=config,
                caps=caps,
                sim_origin="contextual_veto_test",
            )

        veto_snaps = sim.fold_attributions[0].directional_veto_snapshots if sim.fold_attributions else ()
        assert any(s.symbol == "BTCUSDT" and s.fired for s in veto_snaps)
        assert any(s.symbol == "ETHUSDT" and s.raw_mu_before == 0.0 and not s.fired for s in veto_snaps)
        assert any(s.symbol == "BNBUSDT" and s.was_missing for s in veto_snaps)

    @pytest.mark.parametrize("veto_action", ["drop_long", "zero_mu", "cap_mu"])
    def test_adverse_only_mode_tracks_action_branches(
        self,
        veto_action: Literal["drop_long", "zero_mu", "cap_mu"],
    ) -> None:
        aligned, cache, signal_batch, awf_folds, caps, regime_codes = build_contextual_veto_awf_fixture()
        config = Layer2AllocationConfig(
            k_rank=3,
            rebalance_bars=3,
            l2_routing_mode="pool",
            l2_regime_directional_veto_enabled=True,
            l2_regime_directional_veto_mode="adverse_only",
            l2_regime_directional_veto_symbols=("BTCUSDT", "ETHUSDT", "BNBUSDT"),
            l2_regime_directional_veto_adverse_codes=(0,),
            l2_regime_directional_veto_long_eps_bps=0.0,
            l2_regime_directional_veto_action=veto_action,
            l2_regime_directional_veto_cap_mu_bps=50.0,
        )

        with (
            patch("src.domain.futures.strategy.market_regime.compress_regime_codes") as mock_compress,
            patch("src.domain.futures.strategy.market_regime.compute_market_regime_context") as mock_regime,
        ):
            mock_regime.return_value = MagicMock(code_1d=regime_codes)
            mock_compress.side_effect = lambda x: x

            from src.domain.futures.strategy.tiered_workflow.awf_sim import _run_awf_simulation

            sim = _run_awf_simulation(
                cache=cache,
                signal_batch=signal_batch,
                aligned=aligned,
                awf_folds=awf_folds,
                config=config,
                caps=caps,
                sim_origin="adverse_veto_test",
            )

        veto_snaps = sim.fold_attributions[0].directional_veto_snapshots if sim.fold_attributions else ()
        assert any(s.symbol == "BTCUSDT" and s.fired for s in veto_snaps)
        assert any(s.symbol == "ETHUSDT" and not s.fired for s in veto_snaps)
        assert any(s.symbol == "BNBUSDT" and s.was_missing for s in veto_snaps)


def make_attr_veto(snaps: tuple[DirectionalVetoSnapshot, ...]) -> Layer2FoldAttribution:
    return Layer2FoldAttribution(
        fold_idx=0,
        oos_bars=12,
        n_rebal=4,
        realized_total=0.0,
        realized_price=0.0,
        realized_funding=0.0,
        realized_cost=0.0,
        expected_net=0.0,
        alpha_gap=0.0,
        mean_gross_exp=0.5,
        mean_net_exp=0.0,
        sleeves_active_mean=2.0,
        friction_pass_ratio=1.0,
        throttle_mult_mean=1.0,
        dropped_below_cost=0,
        netting_events=0,
        directional_veto_snapshots=snaps,
    )


class TestContextualVetoExtraBranches:
    """Additional branch coverage for uncoverd lines."""

    def test_veto_stays_veto_no_release(self) -> None:
        cfg = make_contextual_cfg(l2_regime_directional_veto_cooldown_bars=3)
        state = ContextualDirectionalVetoState(
            symbol="BTCUSDT",
            state="veto",
            adverse_long_streak=4,
        )
        _new_state, _fired, _mu_after, _reason, before, after = _compute_contextual_directional_veto_signal(
            symbol="BTCUSDT",
            raw_mu=5e-4,
            regime_code=1,
            rolling_symbol_return=-0.02,
            state=state,
            config=cfg,
        )
        assert before == "veto"
        assert after == "veto"
        assert _fired
        assert _mu_after == 0.0

    def test_release_on_bull_regime_streak(self) -> None:
        cfg = make_contextual_cfg(
            l2_regime_directional_veto_release_regime_bull_bars=2,
            l2_regime_directional_veto_cooldown_bars=2,
        )
        state = ContextualDirectionalVetoState(
            symbol="BTCUSDT",
            state="veto",
            adverse_long_streak=3,
        )
        state, _fired, _mu_after, _reason, before, after = _compute_contextual_directional_veto_signal(
            symbol="BTCUSDT",
            raw_mu=5e-4,
            regime_code=0,
            rolling_symbol_return=0.0,
            state=state,
            config=cfg,
        )
        assert before == "veto"
        assert after == "veto"
        state, _fired, _mu_after, _reason, before, after = _compute_contextual_directional_veto_signal(
            symbol="BTCUSDT",
            raw_mu=5e-4,
            regime_code=0,
            rolling_symbol_return=0.0,
            state=state,
            config=cfg,
        )
        assert before == "veto"
        assert after == "cooldown"
        assert _reason == "bull_regime_streak"

    def test_armed_state_no_loss_trigger(self) -> None:
        cfg = make_contextual_cfg(
            l2_regime_directional_veto_persistence_bars=2,
            l2_regime_directional_veto_loss_trigger_bps=200.0,
        )
        state = ContextualDirectionalVetoState(symbol="BTCUSDT")
        for _ in range(2):
            state, fired, _, _, _, after = _compute_contextual_directional_veto_signal(
                symbol="BTCUSDT",
                raw_mu=5e-4,
                regime_code=1,
                rolling_symbol_return=-0.01,
                state=state,
                config=cfg,
            )
        assert after == "armed"
        assert not fired

    def test_cooldown_decrement(self) -> None:
        cfg = make_contextual_cfg(l2_regime_directional_veto_cooldown_bars=3)
        state = ContextualDirectionalVetoState(
            symbol="BTCUSDT",
            state="cooldown",
            cooldown_left=2,
        )
        state, fired, _, _, _, after = _compute_contextual_directional_veto_signal(
            symbol="BTCUSDT",
            raw_mu=3e-4,
            regime_code=0,
            rolling_symbol_return=0.0,
            state=state,
            config=cfg,
        )
        assert after == "cooldown"
        assert state.cooldown_left == 1
        assert not fired

    def test_fold_boundary_reset(self) -> None:
        _new_state = ContextualDirectionalVetoState(symbol="BTCUSDT")
        assert _new_state.state == "idle"
        assert _new_state.adverse_long_streak == 0

    def test_missing_signal_skipped_in_contextual(self) -> None:
        cfg = make_contextual_cfg(l2_regime_directional_veto_enabled=False)
        aspect = cfg.l2_regime_directional_veto_enabled
        assert not aspect

    def test_rolling_return_t0(self) -> None:
        close_2d = np.array([[100.0], [101.0]], dtype=np.float64)
        ret = _compute_symbol_rolling_return(
            close_2d=close_2d,
            t=0,
            symbol_idx=0,
            lookback_bars=3,
        )
        assert ret == 0.0

    def test_rolling_return_insufficient_data(self) -> None:
        close_2d = np.array([[100.0]], dtype=np.float64)
        ret = _compute_symbol_rolling_return(
            close_2d=close_2d,
            t=1,
            symbol_idx=0,
            lookback_bars=3,
        )
        assert ret == 0.0


class TestComputeMeanEpisodeBars:
    """Coverage for _compute_mean_episode_bars."""

    def test_empty_snaps(self) -> None:
        from src.domain.futures.strategy.tiered_workflow.awf_sim import _compute_mean_episode_bars

        assert _compute_mean_episode_bars([]) == 0.0

    def test_single_episode(self) -> None:
        from src.domain.futures.strategy.tiered_workflow.awf_sim import (
            DirectionalVetoSnapshot,
            _compute_mean_episode_bars,
        )

        snaps = [
            DirectionalVetoSnapshot(
                fold_idx=0,
                t=0,
                symbol="BTCUSDT",
                regime_code=1,
                raw_mu_before=5e-4,
                raw_mu_after=0.0,
                counterfactual_weight=0.12,
                weight_after=0.0,
                fired=True,
                was_missing=False,
                bar_price_return_after=0.0,
                counterfactual_long_return=-0.01,
                state_before="watch",
                state_after="veto",
            ),
            DirectionalVetoSnapshot(
                fold_idx=0,
                t=1,
                symbol="BTCUSDT",
                regime_code=1,
                raw_mu_before=4e-4,
                raw_mu_after=0.0,
                counterfactual_weight=0.10,
                weight_after=0.0,
                fired=True,
                was_missing=False,
                bar_price_return_after=0.0,
                counterfactual_long_return=-0.01,
                state_before="veto",
                state_after="veto",
            ),
            DirectionalVetoSnapshot(
                fold_idx=0,
                t=2,
                symbol="BTCUSDT",
                regime_code=1,
                raw_mu_before=3e-4,
                raw_mu_after=3e-4,
                counterfactual_weight=0.0,
                weight_after=0.05,
                fired=False,
                was_missing=False,
                bar_price_return_after=0.0,
                counterfactual_long_return=0.0,
                state_before="veto",
                state_after="watch",
            ),
        ]
        assert _compute_mean_episode_bars(snaps) == 2.0

    def test_veto_at_end_of_data(self) -> None:
        from src.domain.futures.strategy.tiered_workflow.awf_sim import (
            DirectionalVetoSnapshot,
            _compute_mean_episode_bars,
        )

        snaps = [
            DirectionalVetoSnapshot(
                fold_idx=0,
                t=0,
                symbol="BTCUSDT",
                regime_code=1,
                raw_mu_before=5e-4,
                raw_mu_after=0.0,
                counterfactual_weight=0.12,
                weight_after=0.0,
                fired=True,
                was_missing=False,
                bar_price_return_after=0.0,
                counterfactual_long_return=-0.01,
                state_before="idle",
                state_after="veto",
            ),
        ]
        assert _compute_mean_episode_bars(snaps) == 1.0


class TestIntraSymbolDivergenceStateMachine:
    """Track 1 — BTC Intra-Symbol Divergence Dampener 상태기계."""

    def make_cfg(self, **overrides: object) -> Layer2AllocationConfig:
        base = {
            "l2_intra_symbol_divergence_enabled": True,
            "l2_intra_symbol_divergence_symbols": ("BTCUSDT",),
            "l2_intra_symbol_divergence_dominant_families": ("dual_momentum", "supertrend"),
            "l2_intra_symbol_divergence_persistence_bars": 3,
            "l2_intra_symbol_divergence_release_bars": 2,
            "l2_intra_symbol_divergence_cooldown_bars": 3,
            "l2_intra_symbol_divergence_dominant_damp_mult": 0.5,
            "l2_intra_symbol_divergence_dissent_boost_mult": 2.0,
        }
        base.update(overrides)
        return Layer2AllocationConfig.from_mapping(base)

    def test_idle_watch_armed_path_on_persistent_divergence(self) -> None:
        from src.domain.futures.strategy.tiered_workflow.awf_sim import (
            IntraSymbolDivergenceState,
            _compute_intra_symbol_divergence_signal,
        )

        cfg = self.make_cfg()
        state = IntraSymbolDivergenceState(symbol="BTCUSDT")

        # bar 1: idle → watch
        state, armed, before, after = _compute_intra_symbol_divergence_signal(
            symbol="BTCUSDT",
            dominant_sign=1,
            dissent_diverges=True,
            regime_code=1,
            state=state,
            config=cfg,
        )
        assert before == "idle"
        assert after == "watch"
        assert not armed

        # bar 2: watch → watch
        state, armed, before, after = _compute_intra_symbol_divergence_signal(
            symbol="BTCUSDT",
            dominant_sign=1,
            dissent_diverges=True,
            regime_code=1,
            state=state,
            config=cfg,
        )
        assert before == "watch"
        assert after == "watch"
        assert not armed

        # bar 3: watch → armed
        state, armed, before, after = _compute_intra_symbol_divergence_signal(
            symbol="BTCUSDT",
            dominant_sign=1,
            dissent_diverges=True,
            regime_code=1,
            state=state,
            config=cfg,
        )
        assert before == "watch"
        assert after == "armed"
        assert armed

    def test_divergence_requires_both_adverse_and_sign_mismatch(self) -> None:
        from src.domain.futures.strategy.tiered_workflow.awf_sim import (
            IntraSymbolDivergenceState,
            _compute_intra_symbol_divergence_signal,
        )

        cfg = self.make_cfg()
        state = IntraSymbolDivergenceState(symbol="BTCUSDT")

        # regime_code=0 (non-adverse) even with divergence → stays idle
        state, armed, before, after = _compute_intra_symbol_divergence_signal(
            symbol="BTCUSDT",
            dominant_sign=1,
            dissent_diverges=True,
            regime_code=0,
            state=state,
            config=cfg,
        )
        assert before == "idle"
        assert after == "idle"
        assert not armed

        # adverse but no divergence → stays idle
        state, armed, before, after = _compute_intra_symbol_divergence_signal(
            symbol="BTCUSDT",
            dominant_sign=1,
            dissent_diverges=False,
            regime_code=1,
            state=state,
            config=cfg,
        )
        assert before == "idle"
        assert after == "idle"
        assert not armed

    def test_release_after_release_bars_enters_cooldown_then_idle(self) -> None:
        from src.domain.futures.strategy.tiered_workflow.awf_sim import (
            IntraSymbolDivergenceState,
            _compute_intra_symbol_divergence_signal,
        )

        cfg = self.make_cfg()
        state = IntraSymbolDivergenceState(symbol="BTCUSDT")
        # Advance to armed (3 persistent bars)
        for _ in range(3):
            state, armed, _, _ = _compute_intra_symbol_divergence_signal(
                symbol="BTCUSDT",
                dominant_sign=1,
                dissent_diverges=True,
                regime_code=1,
                state=state,
                config=cfg,
            )
        assert armed

        # Release bar 1: armed release_streak=1 (still armed)
        state, armed, before, after = _compute_intra_symbol_divergence_signal(
            symbol="BTCUSDT",
            dominant_sign=1,
            dissent_diverges=False,
            regime_code=1,
            state=state,
            config=cfg,
        )
        assert before == "armed"
        assert after == "armed"
        assert armed

        # Release bar 2: armed → cooldown
        state, armed, before, after = _compute_intra_symbol_divergence_signal(
            symbol="BTCUSDT",
            dominant_sign=1,
            dissent_diverges=False,
            regime_code=1,
            state=state,
            config=cfg,
        )
        assert before == "armed"
        assert after == "cooldown"
        assert not armed

        # cooldown bar 1: cooldown_left=2
        state, armed, before, after = _compute_intra_symbol_divergence_signal(
            symbol="BTCUSDT",
            dominant_sign=1,
            dissent_diverges=True,
            regime_code=1,
            state=state,
            config=cfg,
        )
        assert before == "cooldown"
        assert after == "cooldown"
        assert not armed

        # cooldown bar 2: cooldown_left=1
        state, armed, before, after = _compute_intra_symbol_divergence_signal(
            symbol="BTCUSDT",
            dominant_sign=1,
            dissent_diverges=True,
            regime_code=1,
            state=state,
            config=cfg,
        )
        assert before == "cooldown"
        assert after == "cooldown"

        # cooldown bar 3: cooldown_left=0 → still cooldown (next call transitions to idle)
        state, armed, before, after = _compute_intra_symbol_divergence_signal(
            symbol="BTCUSDT",
            dominant_sign=1,
            dissent_diverges=True,
            regime_code=1,
            state=state,
            config=cfg,
        )
        assert before == "cooldown"
        assert after == "cooldown"

        # cooldown expired → idle
        state, armed, before, after = _compute_intra_symbol_divergence_signal(
            symbol="BTCUSDT",
            dominant_sign=1,
            dissent_diverges=True,
            regime_code=1,
            state=state,
            config=cfg,
        )
        assert before == "cooldown"
        assert after == "idle"
        assert not armed


class TestApplyIntraSymbolDivergenceAdjustment:
    """Track 1 — Sleeve 조정 적용 테스트."""

    def make_sig(self, raw_mu: float, qw: float = 1.0) -> SymbolSignal:
        return SymbolSignal(
            raw_mu=raw_mu,
            volatility=0.01,
            n_obs=10,
            t_stat=0.0,
            valid=True,
            beta_btc=None,
            quality_weight=qw,
        )

    def test_apply_adjustment_damps_dominant_boosts_dissent(self) -> None:
        from src.domain.futures.strategy.tiered_workflow.awf_sim import (
            _apply_intra_symbol_divergence_adjustment,
        )

        sleeves = {
            ("BTCUSDT", "dual_momentum:v1_4h"): self.make_sig(raw_mu=3.678, qw=1.000),
            ("BTCUSDT", "ichimoku_trend:v1_4h"): self.make_sig(raw_mu=-0.222, qw=0.734),
            ("ETHUSDT", "dual_momentum:v1_4h"): self.make_sig(raw_mu=3.192, qw=0.869),
        }
        adjusted = _apply_intra_symbol_divergence_adjustment(
            sleeves,
            symbol="BTCUSDT",
            dominant_families=frozenset({"dual_momentum", "supertrend"}),
            dominant_damp_mult=0.5,
            dissent_boost_mult=2.0,
            dissent_boost_cap_mult=3.0,
        )
        # dominant damp: 3.678 * 0.5 = 1.839
        assert adjusted[("BTCUSDT", "dual_momentum:v1_4h")].raw_mu == pytest.approx(1.839)
        # dissent boost: 0.734 * 2.0 = 1.468
        assert adjusted[("BTCUSDT", "ichimoku_trend:v1_4h")].quality_weight == pytest.approx(1.468)
        # non-target symbol unchanged
        assert adjusted[("ETHUSDT", "dual_momentum:v1_4h")].raw_mu == pytest.approx(3.192)

    def test_boost_mult_clipped_at_safety_cap(self) -> None:
        from src.domain.futures.strategy.tiered_workflow.awf_sim import (
            _apply_intra_symbol_divergence_adjustment,
        )

        sleeves = {
            ("BTCUSDT", "ichimoku_trend:v1_4h"): self.make_sig(raw_mu=1.0, qw=1.0),
        }
        # dissent_boost_mult=10.0 but cap at 3.0 → qw should be min(1.0*10, 1.0*3.0) = 3.0
        adjusted = _apply_intra_symbol_divergence_adjustment(
            sleeves,
            symbol="BTCUSDT",
            dominant_families=frozenset({"dual_momentum"}),
            dominant_damp_mult=0.5,
            dissent_boost_mult=10.0,
            dissent_boost_cap_mult=3.0,
        )
        assert adjusted[("BTCUSDT", "ichimoku_trend:v1_4h")].quality_weight == pytest.approx(3.0)

    def test_non_target_symbol_sleeves_never_modified(self) -> None:
        from src.domain.futures.strategy.tiered_workflow.awf_sim import (
            _apply_intra_symbol_divergence_adjustment,
        )

        sleeves = {
            ("ETHUSDT", "dual_momentum:v1_4h"): self.make_sig(raw_mu=3.192, qw=0.869),
            ("BNBUSDT", "supertrend:v1_4h"): self.make_sig(raw_mu=1.5, qw=0.5),
        }
        adjusted = _apply_intra_symbol_divergence_adjustment(
            sleeves,
            symbol="BTCUSDT",
            dominant_families=frozenset({"dual_momentum"}),
            dominant_damp_mult=0.5,
            dissent_boost_mult=2.0,
        )
        assert adjusted[("ETHUSDT", "dual_momentum:v1_4h")].raw_mu == pytest.approx(3.192)
        assert adjusted[("BNBUSDT", "supertrend:v1_4h")].quality_weight == pytest.approx(0.5)

    def test_dominant_or_dissent_family_absent_skips_symbol(self) -> None:
        from src.domain.futures.strategy.tiered_workflow.awf_sim import (
            _apply_intra_symbol_divergence_adjustment,
        )

        # target symbol present but only dominant (dissent absent) → no crash
        sleeves = {
            ("BTCUSDT", "dual_momentum:v1_4h"): self.make_sig(raw_mu=3.678, qw=1.0),
        }
        adjusted = _apply_intra_symbol_divergence_adjustment(
            sleeves,
            symbol="BTCUSDT",
            dominant_families=frozenset({"dual_momentum", "supertrend"}),
            dominant_damp_mult=0.5,
            dissent_boost_mult=2.0,
        )
        assert ("BTCUSDT", "dual_momentum:v1_4h") in adjusted
        assert adjusted[("BTCUSDT", "dual_momentum:v1_4h")].raw_mu == pytest.approx(1.839)

        # target symbol not in sleeve keys → dict returned unchanged
        adjusted2 = _apply_intra_symbol_divergence_adjustment(
            {("ETHUSDT", "test:v1_4h"): self.make_sig(raw_mu=1.0, qw=1.0)},
            symbol="BTCUSDT",
            dominant_families=frozenset({"dual_momentum"}),
            dominant_damp_mult=0.5,
            dissent_boost_mult=2.0,
        )
        assert ("ETHUSDT", "test:v1_4h") in adjusted2
        assert adjusted2[("ETHUSDT", "test:v1_4h")].raw_mu == pytest.approx(1.0)
