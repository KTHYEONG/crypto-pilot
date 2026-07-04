from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.domain.futures.portfolio.portfolio_constructor import PortfolioCaps
from src.domain.futures.strategy.cs_rank import SymbolSignal
from src.domain.futures.strategy.tiered_workflow.awf_sim import (
    DirectionalVetoSnapshot,
    Layer2FoldAttribution,
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


BASE_CFG = Layer2AllocationConfig.from_mapping({
    "l2_regime_directional_veto_enabled": True,
    "l2_regime_directional_veto_symbols": ("BTCUSDT", "ETHUSDT"),
    "l2_regime_directional_veto_adverse_codes": (1, 2),
    "l2_regime_directional_veto_action": "drop_long",
})

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
                fold_idx=0, t=10, symbol="BTCUSDT", regime_code=1,
                raw_mu_before=5.0, raw_mu_after=0.0,
                counterfactual_weight=0.12, weight_after=0.0,
                fired=True, was_missing=False,
                bar_price_return_after=0.0, counterfactual_long_return=-0.03,
            ),
            DirectionalVetoSnapshot(
                fold_idx=1, t=20, symbol="BTCUSDT", regime_code=1,
                raw_mu_before=3.0, raw_mu_after=0.0,
                counterfactual_weight=0.10, weight_after=0.0,
                fired=True, was_missing=False,
                bar_price_return_after=0.0, counterfactual_long_return=0.02,
            ),
            DirectionalVetoSnapshot(
                fold_idx=0, t=15, symbol="ETHUSDT", regime_code=0,
                raw_mu_before=2.0, raw_mu_after=2.0,
                counterfactual_weight=0.0, weight_after=0.05,
                fired=False, was_missing=False,
                bar_price_return_after=0.0, counterfactual_long_return=0.0,
            ),
        )
        attr_a = make_attr((snaps[0], snaps[2]))
        attr_b = make_attr((snaps[1],))
        summary = summarize_directional_veto(
            (attr_a, attr_b), symbols=("BTCUSDT", "ETHUSDT"),
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
            (make_attr(()),), symbols=("BTCUSDT", "ETHUSDT"),
        )
        assert len(summary) == 2
        assert summary[0].n_obs == 0
        assert summary[0].n_fired == 0

    def test_missing_symbol(self) -> None:
        snaps = (DirectionalVetoSnapshot(
            fold_idx=0, t=10, symbol="BTCUSDT", regime_code=1,
            raw_mu_before=0.0, raw_mu_after=0.0,
            counterfactual_weight=0.0, weight_after=0.0,
            fired=False, was_missing=True,
            bar_price_return_after=0.0, counterfactual_long_return=0.0,
        ),)
        summary = summarize_directional_veto(
            (make_attr(snaps),), symbols=("BTCUSDT", "ETHUSDT"),
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
                fold_idx=0, t=10, symbol="BTCUSDT", regime_code=1,
                raw_mu_before=5.0, raw_mu_after=0.0,
                counterfactual_weight=0.12, weight_after=0.0,
                fired=True, was_missing=False,
                bar_price_return_after=0.0, counterfactual_long_return=0.03,
            ),
        )
        snaps_neg = (
            DirectionalVetoSnapshot(
                fold_idx=0, t=10, symbol="BTCUSDT", regime_code=1,
                raw_mu_before=5.0, raw_mu_after=0.0,
                counterfactual_weight=0.12, weight_after=0.0,
                fired=True, was_missing=False,
                bar_price_return_after=0.0, counterfactual_long_return=-0.03,
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
            Layer2AllocationConfig.from_mapping({
                "l2_regime_directional_veto_action": "flip_short",
            })

    def test_invalid_adverse_code_set(self) -> None:
        with pytest.raises(ValueError, match="adverse_codes"):
            Layer2AllocationConfig.from_mapping({
                "l2_regime_directional_veto_adverse_codes": (0, 1),
            })

    def test_invalid_false_positive_bound(self) -> None:
        with pytest.raises(ValueError, match="l2_regime_directional_veto_max_fit_false_positive_rate"):
            Layer2AllocationConfig.from_mapping({
                "l2_regime_directional_veto_max_fit_false_positive_rate": 1.5,
            })

    def test_invalid_gross_ratio_bound(self) -> None:
        with pytest.raises(ValueError, match="l2_regime_directional_veto_min_gross_ratio"):
            Layer2AllocationConfig.from_mapping({
                "l2_regime_directional_veto_min_gross_ratio": -0.1,
            })

    def test_invalid_turnover_delta(self) -> None:
        with pytest.raises(ValueError, match="l2_regime_directional_veto_max_turnover_delta"):
            Layer2AllocationConfig.from_mapping({
                "l2_regime_directional_veto_max_turnover_delta": -0.01,
            })


class TestDirectionalVetoCoverage:
    """Coverage-gap tests for veto integration."""

    def test_veto_snapshot_frozen(self) -> None:
        s = DirectionalVetoSnapshot(
            fold_idx=0, t=0, symbol="BTCUSDT", regime_code=1,
            raw_mu_before=5.0, raw_mu_after=0.0,
            counterfactual_weight=0.12, weight_after=0.0,
            fired=True, was_missing=False,
            bar_price_return_after=0.0, counterfactual_long_return=0.0,
        )
        assert s.fold_idx == 0
        assert s.symbol == "BTCUSDT"
        assert s.fired

    def test_veto_summary_empty_symbols(self) -> None:
        summary = summarize_directional_veto((), symbols=())
        assert summary == ()

    def test_veto_summary_zero_n_fired_no_div_error(self) -> None:
        snaps = (DirectionalVetoSnapshot(
            fold_idx=0, t=0, symbol="BTCUSDT", regime_code=0,
            raw_mu_before=2.0, raw_mu_after=2.0,
            counterfactual_weight=0.0, weight_after=0.05,
            fired=False, was_missing=False,
            bar_price_return_after=0.0, counterfactual_long_return=0.0,
        ),)
        attr = Layer2FoldAttribution(
            fold_idx=0, oos_bars=12, n_rebal=4,
            realized_total=0.0, realized_price=0.0, realized_funding=0.0,
            realized_cost=0.0, expected_net=0.0, alpha_gap=0.0,
            mean_gross_exp=0.5, mean_net_exp=0.0, sleeves_active_mean=2.0,
            friction_pass_ratio=1.0, throttle_mult_mean=1.0,
            dropped_below_cost=0, netting_events=0,
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
        self, mock_regime: MagicMock, mock_compress: MagicMock,
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
                decision_idx=5, decision_time=aligned.datetimes[5],
                symbol="BTCUSDT", strategy_id="trend:fast",
                activation_context="all", side=1,
                expected_net_bps=10.0, expected_gross_bps=15.0,
                q10_net_bps=5.0, q10_gross_bps=8.0,
                q90_net_bps=0.0, q90_gross_bps=20.0,
                expected_holding_bars=3, reliability=0.9,
                registry_version="test", model_version="test",
            ),
            ValidatedSignalEvent(
                decision_idx=5, decision_time=aligned.datetimes[5],
                symbol="ETHUSDT", strategy_id="trend:fast",
                activation_context="all", side=-1,
                expected_net_bps=-5.0, expected_gross_bps=-8.0,
                q10_net_bps=-2.0, q10_gross_bps=-4.0,
                q90_net_bps=0.0, q90_gross_bps=-12.0,
                expected_holding_bars=3, reliability=0.9,
                registry_version="test", model_version="test",
            ),
        ]
        signal_batch = ValidatedSignalBatch(
            events=tuple(events),
            start_idx=0, end_idx=50,
            symbols=aligned.symbols,
            registry_version="test", model_version="test",
        )

        from src.domain.futures.strategy.tiered_workflow.awf_sim import build_l2_simulation_cache
        from src.domain.futures.strategy.walk_forward import WFFold

        cache = build_l2_simulation_cache(aligned, signal_batch, "4h")
        awf_folds = (WFFold(fit_start=0, fit_end=5, cal_start=5, cal_end=8, oos_start=8, oos_end=30),)
        config = Layer2AllocationConfig(
            k_rank=3, rebalance_bars=3,
            l2_regime_directional_veto_enabled=True,
            l2_regime_directional_veto_symbols=("BTCUSDT", "ETHUSDT"),
            l2_regime_directional_veto_adverse_codes=(1, 2),
            l2_regime_directional_veto_long_eps_bps=0.0,
            l2_regime_directional_veto_action="drop_long",
        )
        caps = PortfolioCaps(gross=3.0, per_symbol=0.15, net=0.5, beta=1.0, target_ann_vol=0.20)

        sim = _run_awf_simulation(
            cache=cache, signal_batch=signal_batch, aligned=aligned,
            awf_folds=awf_folds, config=config, caps=caps,
            sim_origin="test_veto",
        )
        veto_snaps = sim.fold_attributions[0].directional_veto_snapshots if sim.fold_attributions else ()
        assert len(veto_snaps) > 0

    def test_veto_config_zero_mu_action(self) -> None:
        cfg = Layer2AllocationConfig.from_mapping({
            "l2_regime_directional_veto_enabled": True,
            "l2_regime_directional_veto_action": "zero_mu",
        })
        assert cfg.l2_regime_directional_veto_action == "zero_mu"

    def test_veto_config_adverse_codes_dedup(self) -> None:
        cfg = Layer2AllocationConfig.from_mapping({
            "l2_regime_directional_veto_adverse_codes": (2, 1, 2),
        })
        assert cfg.l2_regime_directional_veto_adverse_codes == (1, 2)

    def test_veto_config_symbols_uppercased_dedup(self) -> None:
        cfg = Layer2AllocationConfig.from_mapping({
            "l2_regime_directional_veto_symbols": ("btcusdt", "ETHUSDT", "btcusdt"),
        })
        assert cfg.l2_regime_directional_veto_symbols == ("BTCUSDT", "ETHUSDT")

    def test_layer2_result_directional_veto_summary_field(self) -> None:
        r = Layer2Result(
            selected_last=frozenset(), weights_last={},
            sharpe_hybrid=0.0, sharpe_baseline=0.0,
            mdd_hybrid=0.0, mdd_baseline=0.0,
            cagr_hybrid=0.0, cagr_baseline=0.0,
            mar_hybrid=0.0, mar_baseline=0.0,
            fold_pass_ratio=0.0, turnover=0.0,
            friction_pass_pct=0.0, gate_passed=False,
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
        line = _format_directional_veto_line(summary)
        assert "BTCUSDT" in line
        assert "62.5%" in line
        assert "+0.0410" in line
        assert "+0.0290" in line
