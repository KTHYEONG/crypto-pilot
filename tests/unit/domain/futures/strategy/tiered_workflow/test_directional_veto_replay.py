from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from src.domain.futures.strategy.tiered_workflow.awf_sim import DirectionalVetoSummary
from src.domain.futures.strategy.tiered_workflow.dataclasses import (
    Layer2AllocationConfig,
    Layer2Result,
    Layer3Result,
)
from src.domain.futures.strategy.tiered_workflow.pipeline import (
    DirectionalVetoReplayResult,
    _directional_veto_replay_adoption_verdict,
    _directional_veto_replay_variants,
    run_directional_veto_economic_replay,
)


class TestDirectionalVetoReplayVariants:
    """S4: replay variants include new contextual candidates."""

    def test_variants_return_five_arms(self) -> None:
        variants = _directional_veto_replay_variants()
        names = [v.name for v in variants]
        assert len(variants) == 5
        assert "baseline" in names
        assert "veto_adverse_only" in names
        assert "contextual_cap_mu" in names
        assert "contextual_zero_mu" in names
        assert "contextual_crisis_only" in names

    def test_baseline_disabled(self) -> None:
        variants = _directional_veto_replay_variants()
        by_name = {v.name: v for v in variants}
        assert not by_name["baseline"].directional_veto_enabled

    def test_contextual_mode_set(self) -> None:
        variants = _directional_veto_replay_variants()
        by_name = {v.name: v for v in variants}
        assert by_name["contextual_cap_mu"].directional_veto_mode == "contextual"
        assert by_name["contextual_zero_mu"].directional_veto_mode == "contextual"
        assert by_name["contextual_crisis_only"].directional_veto_mode == "contextual"
        assert by_name["veto_adverse_only"].directional_veto_mode == "adverse_only"

    def test_contextual_crisis_only_adverse_codes(self) -> None:
        variants = _directional_veto_replay_variants()
        crisis = next(v for v in variants if v.name == "contextual_crisis_only")
        assert crisis.directional_veto_adverse_codes == (2,)


class TestDirectionalVetoReplayAdoptionVerdict:
    """S5 + X5/X6: adoption gate tests with new rules."""

    def _make_result(
        self,
        *,
        variant: str = "contextual_cap_mu",
        baseline_parity: bool = True,
        l2_fp: float = 0.0,
        l2_net_veto: float = 0.01,
        l2_gross_exp: float = 0.95,
        l2_turnover: float = 0.5,
        l2_cagr: float = 0.15,
        l3_total_return: float = 0.05,
        l3_mdd: float = 0.10,
        l3_sharpe: float = 1.2,
        l3_long_loss: tuple[tuple[str, float], ...] = (("BTCUSDT", -0.02), ("ETHUSDT", -0.01)),
    ) -> DirectionalVetoReplayResult:
        l2_veto_summary = (
            DirectionalVetoSummary(
                symbol="BTCUSDT",
                n_obs=10,
                n_missing=0,
                n_adverse=5,
                n_fired=4,
                fire_rate=0.4,
                adverse_fire_rate=0.8,
                false_positive_rate=l2_fp,
                opportunity_cost=0.01,
                avoided_loss=max(l2_net_veto + 0.01, 0.02),
                net_veto_value=l2_net_veto,
            ),
        )
        return DirectionalVetoReplayResult(
            variant=variant,
            baseline_parity=baseline_parity,
            l2_cagr=l2_cagr,
            l2_mdd=0.08,
            l2_turnover=l2_turnover,
            l2_average_gross_exposure=l2_gross_exp,
            l2_gate_passed=True,
            l2_blocker_reason="",
            l2_directional_veto_summary=l2_veto_summary,
            l3_cagr=0.12,
            l3_mdd=l3_mdd,
            l3_sharpe=l3_sharpe,
            l3_total_return=l3_total_return,
            l3_gate_passed=True,
            l3_blocker_reason="",
            l3_realized_price_long_by_symbol=l3_long_loss,
            l3_directional_veto_summary=l2_veto_summary,
            adoption_passed=False,
            blocker_reason="",
        )

    def _make_baseline(
        self,
        l3_total_return: float = 0.04,
        l3_mdd: float = 0.12,
        l3_sharpe: float = 1.0,
        l2_cagr: float = 0.14,
        l3_long_loss: tuple[tuple[str, float], ...] | None = None,
    ) -> DirectionalVetoReplayResult:
        if l3_long_loss is None:
            l3_long_loss = (("BTCUSDT", -0.04), ("ETHUSDT", -0.02))
        return self._make_result(
            variant="baseline",
            l3_total_return=l3_total_return,
            l3_mdd=l3_mdd,
            l3_sharpe=l3_sharpe,
            l2_cagr=l2_cagr,
            l3_long_loss=l3_long_loss,
        )

    def test_adoption_passed(self) -> None:
        baseline = self._make_baseline(
            l3_total_return=0.02,
            l3_long_loss=(("BTCUSDT", -0.04), ("ETHUSDT", -0.02)),
        )
        candidate = self._make_result(
            l3_total_return=0.05,
            l3_long_loss=(("BTCUSDT", -0.02), ("ETHUSDT", -0.01)),
        )
        verdict, reason = _directional_veto_replay_adoption_verdict(
            baseline=baseline,
            candidate=candidate,
            max_fit_false_positive_rate=0.50,
            min_gross_ratio=0.90,
            max_turnover_delta=0.05,
            max_fit_net_value_loss=0.0,
            min_l3_total_return_delta=0.02,
            max_l2_cagr_delta_loss=0.005,
        )
        assert verdict, reason

    def test_fp_budget_breach_blocks_adoption(self) -> None:
        baseline = self._make_baseline()
        candidate = self._make_result(l2_fp=0.75)
        verdict, reason = _directional_veto_replay_adoption_verdict(
            baseline=baseline,
            candidate=candidate,
            max_fit_false_positive_rate=0.50,
            min_gross_ratio=0.90,
            max_turnover_delta=0.05,
            max_fit_net_value_loss=0.0,
            min_l3_total_return_delta=0.02,
            max_l2_cagr_delta_loss=0.005,
        )
        assert not verdict
        assert "false_positive" in reason

    def test_baseline_parity_fail(self) -> None:
        baseline = self._make_baseline()
        candidate = self._make_result(baseline_parity=False)
        verdict, reason = _directional_veto_replay_adoption_verdict(
            baseline=baseline,
            candidate=candidate,
            max_fit_false_positive_rate=0.50,
            min_gross_ratio=0.90,
            max_turnover_delta=0.05,
            max_fit_net_value_loss=0.0,
            min_l3_total_return_delta=0.02,
            max_l2_cagr_delta_loss=0.005,
        )
        assert not verdict
        assert "baseline_parity" in reason

    def test_below_min_total_return_delta(self) -> None:
        baseline = self._make_baseline(
            l3_total_return=0.06,
            l3_long_loss=(("BTCUSDT", -0.04), ("ETHUSDT", -0.02)),
        )
        candidate = self._make_result(
            l3_total_return=0.04,
            l3_long_loss=(("BTCUSDT", -0.02), ("ETHUSDT", -0.01)),
        )
        verdict, reason = _directional_veto_replay_adoption_verdict(
            baseline=baseline,
            candidate=candidate,
            max_fit_false_positive_rate=0.50,
            min_gross_ratio=0.90,
            max_turnover_delta=0.05,
            max_fit_net_value_loss=0.0,
            min_l3_total_return_delta=0.02,
            max_l2_cagr_delta_loss=0.005,
        )
        assert not verdict
        assert "below_min_total_return_delta" in reason

    def test_fit_cagr_degradation_blocks(self) -> None:
        baseline = self._make_baseline(
            l2_cagr=0.20,
            l3_long_loss=(("BTCUSDT", -0.04), ("ETHUSDT", -0.02)),
        )
        candidate = self._make_result(
            l2_cagr=0.10,
            l3_long_loss=(("BTCUSDT", -0.02), ("ETHUSDT", -0.01)),
        )
        verdict, reason = _directional_veto_replay_adoption_verdict(
            baseline=baseline,
            candidate=candidate,
            max_fit_false_positive_rate=0.50,
            min_gross_ratio=0.90,
            max_turnover_delta=0.05,
            max_fit_net_value_loss=0.0,
            min_l3_total_return_delta=0.02,
            max_l2_cagr_delta_loss=0.005,
        )
        assert not verdict
        assert "fit_cagr_degradation" in reason

    def test_net_value_negative_blocks_adoption(self) -> None:
        baseline = self._make_baseline()
        candidate = self._make_result(l2_net_veto=-0.01)
        verdict, reason = _directional_veto_replay_adoption_verdict(
            baseline=baseline,
            candidate=candidate,
            max_fit_false_positive_rate=0.50,
            min_gross_ratio=0.90,
            max_turnover_delta=0.05,
            max_fit_net_value_loss=0.005,
            min_l3_total_return_delta=0.02,
            max_l2_cagr_delta_loss=0.005,
        )
        assert not verdict
        assert "fit_net_value_negative" in reason

    def test_gross_preservation_fail(self) -> None:
        baseline = self._make_baseline()
        candidate = self._make_result(l2_gross_exp=0.5)
        verdict, reason = _directional_veto_replay_adoption_verdict(
            baseline=baseline,
            candidate=candidate,
            max_fit_false_positive_rate=0.50,
            min_gross_ratio=0.90,
            max_turnover_delta=0.05,
            max_fit_net_value_loss=0.0,
            min_l3_total_return_delta=0.02,
            max_l2_cagr_delta_loss=0.005,
        )
        assert not verdict
        assert "gross_preservation" in reason

    def test_turnover_budget_exceeded(self) -> None:
        baseline = self._make_baseline()
        candidate = self._make_result(l2_turnover=1.0)
        verdict, reason = _directional_veto_replay_adoption_verdict(
            baseline=baseline,
            candidate=candidate,
            max_fit_false_positive_rate=0.50,
            min_gross_ratio=0.90,
            max_turnover_delta=0.05,
            max_fit_net_value_loss=0.0,
            min_l3_total_return_delta=0.02,
            max_l2_cagr_delta_loss=0.005,
        )
        assert not verdict
        assert "turnover_budget" in reason

    def test_major_long_loss_not_improved(self) -> None:
        baseline = self._make_baseline(
            l3_long_loss=(("BTCUSDT", -0.03), ("ETHUSDT", -0.01)),
        )
        candidate = self._make_result(
            l3_total_return=0.06,
            l3_long_loss=(("BTCUSDT", -0.05), ("ETHUSDT", -0.02)),
        )
        verdict, reason = _directional_veto_replay_adoption_verdict(
            baseline=baseline,
            candidate=candidate,
            max_fit_false_positive_rate=0.50,
            min_gross_ratio=0.90,
            max_turnover_delta=0.05,
            max_fit_net_value_loss=0.0,
            min_l3_total_return_delta=0.02,
            max_l2_cagr_delta_loss=0.005,
        )
        assert not verdict
        assert "major_long_loss" in reason

    def test_improved_long_loss_helps_adoption(self) -> None:
        baseline = self._make_baseline(
            l3_total_return=0.02,
            l3_long_loss=(("BTCUSDT", -0.04), ("ETHUSDT", -0.02)),
        )
        candidate = self._make_result(
            l3_total_return=0.05,
            l3_long_loss=(("BTCUSDT", -0.02), ("ETHUSDT", -0.01)),
        )
        verdict, reason = _directional_veto_replay_adoption_verdict(
            baseline=baseline,
            candidate=candidate,
            max_fit_false_positive_rate=0.50,
            min_gross_ratio=0.90,
            max_turnover_delta=0.05,
            max_fit_net_value_loss=0.0,
            min_l3_total_return_delta=0.02,
            max_l2_cagr_delta_loss=0.005,
        )
        assert verdict, reason


class TestDirectionalVetoReplayRun:
    """Test the replay runner with mocked L2/L3."""

    @patch("src.domain.futures.strategy.tiered_workflow.pipeline.run_l2_awf")
    @patch("src.domain.futures.strategy.tiered_workflow.pipeline.run_l3_holdout")
    def test_returns_five_variants(
        self,
        mock_l3: MagicMock,
        mock_l2: MagicMock,
    ) -> None:
        mock_l2.return_value = MagicMock(
            spec=Layer2Result,
            cagr_hybrid=0.15,
            mdd_hybrid=0.08,
            turnover=0.5,
            average_gross_exposure=0.95,
            gate_passed=True,
            blocker_reason="",
            directional_veto_summary=(),
        )
        mock_l3.return_value = MagicMock(
            spec=Layer3Result,
            cagr=0.12,
            mdd=0.10,
            sharpe=1.2,
            total_return=0.05,
            gate_passed=True,
            blocker_reason="",
            realized_price_long_by_symbol=(),
            directional_veto_summary=(),
        )
        from types import SimpleNamespace

        aligned = SimpleNamespace(symbols=("BTCUSDT", "ETHUSDT", "BNBUSDT"))
        config = Layer2AllocationConfig.from_mapping({})
        caps = SimpleNamespace(gross=3.0, per_symbol=0.15, net=0.5, beta=1.0, target_ann_vol=0.20)
        folds = (SimpleNamespace(oos_start=0, oos_end=100),)
        results = run_directional_veto_economic_replay(
            l2_signal_batch=MagicMock(),
            l3_signal_batch=MagicMock(),
            aligned=aligned,
            awf_folds=folds,
            holdout_span=(0, 100),
            config=config,
            caps=caps,
            tf="4h",
            deploy_leverage=None,
        )
        assert len(results) == 5
        assert results[0].variant == "baseline"
        assert results[1].variant == "veto_adverse_only"
        assert results[2].variant == "contextual_cap_mu"
        assert results[3].variant == "contextual_zero_mu"
        assert results[4].variant == "contextual_crisis_only"


class TestDirectionalVetoReplayCoverage:
    """Coverage-gap tests for replay helpers."""

    def test_replay_variants_structure(self) -> None:
        variants = _directional_veto_replay_variants()
        assert len(variants) == 5
        assert variants[0].name == "baseline"
        assert not variants[0].directional_veto_enabled
        assert variants[1].name == "veto_adverse_only"
        assert variants[1].directional_veto_enabled
        assert variants[1].directional_veto_action == "drop_long"
        assert variants[1].directional_veto_symbols == ("BTCUSDT", "ETHUSDT")

    def test_write_replay_csv(self, tmp_path: Any) -> None:
        from src.domain.futures.strategy.tiered_workflow.pipeline import (
            _write_directional_veto_replay_csv,
        )

        summary = DirectionalVetoSummary(
            symbol="BTCUSDT",
            n_obs=10,
            n_missing=0,
            n_adverse=5,
            n_fired=4,
            fire_rate=0.4,
            adverse_fire_rate=0.8,
            false_positive_rate=0.25,
            opportunity_cost=0.01,
            avoided_loss=0.03,
            net_veto_value=0.02,
        )
        result = DirectionalVetoReplayResult(
            variant="contextual_cap_mu",
            baseline_parity=True,
            l2_cagr=0.15,
            l2_mdd=0.08,
            l2_turnover=0.5,
            l2_average_gross_exposure=0.95,
            l2_gate_passed=True,
            l2_blocker_reason="",
            l2_directional_veto_summary=(summary,),
            l3_cagr=0.12,
            l3_mdd=0.10,
            l3_sharpe=1.2,
            l3_total_return=0.05,
            l3_gate_passed=True,
            l3_blocker_reason="",
            l3_realized_price_long_by_symbol=(("BTCUSDT", -0.01),),
            l3_directional_veto_summary=(summary,),
            adoption_passed=True,
            blocker_reason="",
        )
        path = tmp_path / "veto_replay.csv"
        _write_directional_veto_replay_csv((result,), path=path)
        assert path.exists()
        content = path.read_text()
        assert "contextual_cap_mu" in content
        assert "0.15" in content

    def test_write_replay_detail_csv(self, tmp_path: Any) -> None:
        from src.domain.futures.strategy.tiered_workflow.pipeline import (
            _write_directional_veto_replay_detail_csv,
        )

        summary = DirectionalVetoSummary(
            symbol="BTCUSDT",
            n_obs=10,
            n_missing=0,
            n_adverse=5,
            n_fired=4,
            fire_rate=0.4,
            adverse_fire_rate=0.8,
            false_positive_rate=0.25,
            opportunity_cost=0.01,
            avoided_loss=0.03,
            net_veto_value=0.02,
            n_watch=3,
            mean_trigger_loss=-0.025,
            mean_episode_bars=2.0,
        )
        result = DirectionalVetoReplayResult(
            variant="contextual_cap_mu",
            baseline_parity=True,
            l2_cagr=0.15,
            l2_mdd=0.08,
            l2_turnover=0.5,
            l2_average_gross_exposure=0.95,
            l2_gate_passed=True,
            l2_blocker_reason="",
            l2_directional_veto_summary=(summary,),
            l3_cagr=0.12,
            l3_mdd=0.10,
            l3_sharpe=1.2,
            l3_total_return=0.05,
            l3_gate_passed=True,
            l3_blocker_reason="",
            l3_realized_price_long_by_symbol=(("BTCUSDT", -0.01),),
            l3_directional_veto_summary=(summary,),
            adoption_passed=True,
            blocker_reason="",
        )
        path = tmp_path / "veto_replay_detail.csv"
        _write_directional_veto_replay_detail_csv((result,), path=path)
        assert path.exists()
        content = path.read_text()
        assert "contextual_cap_mu" in content
        assert "n_watch" in content
        assert "mean_episode_bars" in content

    def test_format_replay_table(self) -> None:
        from src.domain.futures.strategy.tiered_workflow.pipeline import (
            format_directional_veto_replay_table,
        )

        summary = DirectionalVetoSummary(
            symbol="BTCUSDT",
            n_obs=10,
            n_missing=0,
            n_adverse=5,
            n_fired=4,
            fire_rate=0.4,
            adverse_fire_rate=0.8,
            false_positive_rate=0.25,
            opportunity_cost=0.01,
            avoided_loss=0.03,
            net_veto_value=0.02,
        )
        result = DirectionalVetoReplayResult(
            variant="contextual_cap_mu",
            baseline_parity=True,
            l2_cagr=0.15,
            l2_mdd=0.08,
            l2_turnover=0.5,
            l2_average_gross_exposure=0.95,
            l2_gate_passed=True,
            l2_blocker_reason="",
            l2_directional_veto_summary=(summary,),
            l3_cagr=0.12,
            l3_mdd=0.10,
            l3_sharpe=1.2,
            l3_total_return=0.05,
            l3_gate_passed=True,
            l3_blocker_reason="",
            l3_realized_price_long_by_symbol=(("BTCUSDT", -0.01),),
            l3_directional_veto_summary=(summary,),
            adoption_passed=True,
            blocker_reason="",
        )
        table = format_directional_veto_replay_table((result,))
        assert "contextual_cap_mu" in table
        assert "CAGR" in table
        assert "PASS" in table

    def test_replay_with_baseline_parity(self) -> None:
        from src.domain.futures.strategy.tiered_workflow.pipeline import (
            run_directional_veto_economic_replay,
        )

        with (
            patch(
                "src.domain.futures.strategy.tiered_workflow.pipeline.run_l2_awf",
            ) as mock_l2,
            patch(
                "src.domain.futures.strategy.tiered_workflow.pipeline.run_l3_holdout",
            ) as mock_l3,
        ):
            l2_res = MagicMock(spec=Layer2Result)
            l2_res.cagr_hybrid = 0.15
            l2_res.mdd_hybrid = 0.08
            l2_res.turnover = 0.5
            l2_res.average_gross_exposure = 0.95
            l2_res.gate_passed = True
            l2_res.blocker_reason = ""
            l2_res.directional_veto_summary = ()
            mock_l2.return_value = l2_res
            l3_res = MagicMock(spec=Layer3Result)
            l3_res.cagr = 0.12
            l3_res.mdd = 0.10
            l3_res.sharpe = 1.2
            l3_res.total_return = 0.05
            l3_res.gate_passed = True
            l3_res.blocker_reason = ""
            l3_res.realized_price_long_by_symbol = ()
            l3_res.directional_veto_summary = ()
            mock_l3.return_value = l3_res
            aligned = MagicMock(symbols=("BTCUSDT", "ETHUSDT", "BNBUSDT"))
            config = Layer2AllocationConfig.from_mapping({})
            caps = MagicMock(gross=3.0, per_symbol=0.15, net=0.5, beta=1.0, target_ann_vol=0.20)
            folds = (MagicMock(oos_start=0, oos_end=100),)
            results = run_directional_veto_economic_replay(
                l2_signal_batch=MagicMock(),
                l3_signal_batch=MagicMock(),
                aligned=aligned,
                awf_folds=folds,
                holdout_span=(0, 100),
                config=config,
                caps=caps,
                tf="4h",
                deploy_leverage=None,
                baseline_l2=l2_res,
                baseline_l3=l3_res,
            )
            assert len(results) == 5
            assert results[0].variant == "baseline"


class TestL2CrisisDeriskStack:
    """Phase 0 Parity Fix — Scenario 1~3 TDD tests for prebuilt_cache/eval_memo forwarding & L2 parity."""

    @patch("src.domain.futures.strategy.tiered_workflow.pipeline.run_l2_awf")
    @patch("src.domain.futures.strategy.tiered_workflow.pipeline.run_l3_holdout")
    def test_prebuilt_cache_and_eval_memo_forwarded_to_every_variant(
        self,
        mock_l3: MagicMock,
        mock_l2: MagicMock,
    ) -> None:
        mock_l2.return_value = MagicMock(
            spec=Layer2Result,
            cagr_hybrid=0.582,
            mdd_hybrid=0.176,
            turnover=0.5,
            average_gross_exposure=0.95,
            gate_passed=True,
            blocker_reason="",
            directional_veto_summary=(),
            fold_pass_ratio=1.0,
            trade_count=214,
            deploy_leverage=2.0341,
            sharpe_hac_hybrid=1.5,
            sortino_hybrid=1.8,
        )
        mock_l3.return_value = MagicMock(
            spec=Layer3Result,
            cagr=-0.171,
            mdd=0.268,
            sharpe=-1.21,
            total_return=-0.169,
            gate_passed=False,
            blocker_reason="negative_return",
            realized_price_long_by_symbol=(),
            directional_veto_summary=(),
        )
        from types import SimpleNamespace

        sentinel_cache = SimpleNamespace(tag="prebuilt")
        sentinel_memo: dict = {}
        aligned = SimpleNamespace(symbols=("BTCUSDT", "ETHUSDT", "BNBUSDT"))
        config = Layer2AllocationConfig.from_mapping({})
        caps = SimpleNamespace(gross=3.0, per_symbol=0.15, net=0.5, beta=1.0, target_ann_vol=0.20)
        folds = (SimpleNamespace(oos_start=0, oos_end=100),)

        run_directional_veto_economic_replay(
            l2_signal_batch=MagicMock(),
            l3_signal_batch=MagicMock(),
            aligned=aligned,
            awf_folds=folds,
            holdout_span=(0, 100),
            config=config,
            caps=caps,
            tf="4h",
            deploy_leverage=2.0341,
            baseline_l2=mock_l2.return_value,
            baseline_l3=mock_l3.return_value,
            prebuilt_cache=sentinel_cache,
            eval_memo=sentinel_memo,
        )

        for call in mock_l2.call_args_list:
            assert call.kwargs["prebuilt_cache"] is sentinel_cache
            assert call.kwargs["eval_memo"] is sentinel_memo

    @patch("src.domain.futures.strategy.tiered_workflow.pipeline.run_l2_awf")
    @patch("src.domain.futures.strategy.tiered_workflow.pipeline.run_l3_holdout")
    def test_baseline_parity_true_when_l2_and_l3_metrics_match(
        self,
        mock_l3: MagicMock,
        mock_l2: MagicMock,
    ) -> None:
        l2_res = MagicMock(
            spec=Layer2Result,
            cagr_hybrid=0.582,
            mdd_hybrid=0.176,
            turnover=0.5,
            average_gross_exposure=0.95,
            gate_passed=True,
            blocker_reason="",
            directional_veto_summary=(),
            fold_pass_ratio=1.0,
            trade_count=214,
            deploy_leverage=2.0341,
            sharpe_hac_hybrid=1.5,
            sortino_hybrid=1.8,
        )
        l3_res = MagicMock(
            spec=Layer3Result,
            cagr=-0.171,
            mdd=0.268,
            sharpe=-1.21,
            total_return=-0.169,
            gate_passed=False,
            blocker_reason="negative_return",
            realized_price_long_by_symbol=(),
            directional_veto_summary=(),
        )
        mock_l2.return_value = l2_res
        mock_l3.return_value = l3_res
        from types import SimpleNamespace

        aligned = SimpleNamespace(symbols=("BTCUSDT", "ETHUSDT", "BNBUSDT"))
        config = Layer2AllocationConfig.from_mapping({})
        caps = SimpleNamespace(gross=3.0, per_symbol=0.15, net=0.5, beta=1.0, target_ann_vol=0.20)
        folds = (SimpleNamespace(oos_start=0, oos_end=100),)

        results = run_directional_veto_economic_replay(
            l2_signal_batch=MagicMock(),
            l3_signal_batch=MagicMock(),
            aligned=aligned,
            awf_folds=folds,
            holdout_span=(0, 100),
            config=config,
            caps=caps,
            tf="4h",
            deploy_leverage=2.0341,
            baseline_l2=l2_res,
            baseline_l3=l3_res,
        )
        assert results[0].baseline_parity is True
        for r in results[1:]:
            assert r.baseline_parity is True

    def test_prebuilt_cache_none_preserves_legacy_behavior(self) -> None:
        """기존 test_returns_five_variants가 수정 없이 통과해야 함 — regression guard."""
        from tests.unit.domain.futures.strategy.tiered_workflow.test_directional_veto_replay import (
            TestDirectionalVetoReplayRun,
        )

        tester = TestDirectionalVetoReplayRun()
        tester.test_returns_five_variants()

    @patch("src.domain.futures.strategy.tiered_workflow.pipeline.run_l2_awf")
    @patch("src.domain.futures.strategy.tiered_workflow.pipeline.run_l3_holdout")
    def test_baseline_parity_false_propagates_to_all_variants_on_l2_mismatch(
        self,
        mock_l3: MagicMock,
        mock_l2: MagicMock,
    ) -> None:
        l2_replay = MagicMock(
            spec=Layer2Result,
            cagr_hybrid=0.582,
            mdd_hybrid=0.176,
            turnover=0.5,
            average_gross_exposure=0.95,
            gate_passed=True,
            blocker_reason="",
            directional_veto_summary=(),
            fold_pass_ratio=1.0,
            trade_count=214,
            deploy_leverage=2.0341,
            sharpe_hac_hybrid=1.5,
            sortino_hybrid=1.8,
        )
        l2_baseline = MagicMock(
            spec=Layer2Result,
            cagr_hybrid=0.583,
            mdd_hybrid=0.176,
            turnover=0.5,
            average_gross_exposure=0.95,
            gate_passed=True,
            blocker_reason="",
            directional_veto_summary=(),
            fold_pass_ratio=1.0,
            trade_count=214,
            deploy_leverage=2.0341,
            sharpe_hac_hybrid=1.5,
            sortino_hybrid=1.8,
        )
        l3_res = MagicMock(
            spec=Layer3Result,
            cagr=-0.171,
            mdd=0.268,
            sharpe=-1.21,
            total_return=-0.169,
            gate_passed=False,
            blocker_reason="negative_return",
            realized_price_long_by_symbol=(),
            directional_veto_summary=(),
        )
        mock_l2.return_value = l2_replay
        mock_l3.return_value = l3_res
        from types import SimpleNamespace

        aligned = SimpleNamespace(symbols=("BTCUSDT", "ETHUSDT", "BNBUSDT"))
        config = Layer2AllocationConfig.from_mapping({})
        caps = SimpleNamespace(gross=3.0, per_symbol=0.15, net=0.5, beta=1.0, target_ann_vol=0.20)
        folds = (SimpleNamespace(oos_start=0, oos_end=100),)

        results = run_directional_veto_economic_replay(
            l2_signal_batch=MagicMock(),
            l3_signal_batch=MagicMock(),
            aligned=aligned,
            awf_folds=folds,
            holdout_span=(0, 100),
            config=config,
            caps=caps,
            tf="4h",
            deploy_leverage=2.0341,
            baseline_l2=l2_baseline,
            baseline_l3=l3_res,
        )
        assert results[0].baseline_parity is False
        for r in results[1:]:
            assert r.baseline_parity is False

    @patch("src.domain.futures.strategy.tiered_workflow.pipeline.run_l2_awf")
    @patch("src.domain.futures.strategy.tiered_workflow.pipeline.run_l3_holdout")
    def test_baseline_parity_false_when_only_l3_mismatches(
        self,
        mock_l3: MagicMock,
        mock_l2: MagicMock,
    ) -> None:
        l2_res = MagicMock(
            spec=Layer2Result,
            cagr_hybrid=0.582,
            mdd_hybrid=0.176,
            turnover=0.5,
            average_gross_exposure=0.95,
            gate_passed=True,
            blocker_reason="",
            directional_veto_summary=(),
            fold_pass_ratio=1.0,
            trade_count=214,
            deploy_leverage=2.0341,
            sharpe_hac_hybrid=1.5,
            sortino_hybrid=1.8,
        )
        l3_replay = MagicMock(
            spec=Layer3Result,
            cagr=-0.171,
            mdd=0.268,
            sharpe=-1.21,
            total_return=-0.169,
            gate_passed=False,
            blocker_reason="negative_return",
            realized_price_long_by_symbol=(),
            directional_veto_summary=(),
        )
        l3_baseline = MagicMock(
            spec=Layer3Result,
            cagr=-0.172,
            mdd=0.268,
            sharpe=-1.21,
            total_return=-0.169,
            gate_passed=False,
            blocker_reason="negative_return",
            realized_price_long_by_symbol=(),
            directional_veto_summary=(),
        )
        mock_l2.return_value = l2_res
        mock_l3.return_value = l3_replay
        from types import SimpleNamespace

        aligned = SimpleNamespace(symbols=("BTCUSDT", "ETHUSDT", "BNBUSDT"))
        config = Layer2AllocationConfig.from_mapping({})
        caps = SimpleNamespace(gross=3.0, per_symbol=0.15, net=0.5, beta=1.0, target_ann_vol=0.20)
        folds = (SimpleNamespace(oos_start=0, oos_end=100),)

        results = run_directional_veto_economic_replay(
            l2_signal_batch=MagicMock(),
            l3_signal_batch=MagicMock(),
            aligned=aligned,
            awf_folds=folds,
            holdout_span=(0, 100),
            config=config,
            caps=caps,
            tf="4h",
            deploy_leverage=2.0341,
            baseline_l2=l2_res,
            baseline_l3=l3_baseline,
        )
        assert results[0].baseline_parity is False

    @patch("src.domain.futures.strategy.tiered_workflow.pipeline.run_l2_awf")
    @patch("src.domain.futures.strategy.tiered_workflow.pipeline.run_l3_holdout")
    def test_baseline_parity_defaults_true_without_baseline_refs(
        self,
        mock_l3: MagicMock,
        mock_l2: MagicMock,
    ) -> None:
        mock_l2.return_value = MagicMock(
            spec=Layer2Result,
            cagr_hybrid=0.582,
            mdd_hybrid=0.176,
            turnover=0.5,
            average_gross_exposure=0.95,
            gate_passed=True,
            blocker_reason="",
            directional_veto_summary=(),
            fold_pass_ratio=1.0,
            trade_count=214,
            deploy_leverage=2.0341,
            sharpe_hac_hybrid=1.5,
            sortino_hybrid=1.8,
        )
        mock_l3.return_value = MagicMock(
            spec=Layer3Result,
            cagr=-0.171,
            mdd=0.268,
            sharpe=-1.21,
            total_return=-0.169,
            gate_passed=False,
            blocker_reason="negative_return",
            realized_price_long_by_symbol=(),
            directional_veto_summary=(),
        )
        from types import SimpleNamespace

        aligned = SimpleNamespace(symbols=("BTCUSDT", "ETHUSDT", "BNBUSDT"))
        config = Layer2AllocationConfig.from_mapping({})
        caps = SimpleNamespace(gross=3.0, per_symbol=0.15, net=0.5, beta=1.0, target_ann_vol=0.20)
        folds = (SimpleNamespace(oos_start=0, oos_end=100),)

        results = run_directional_veto_economic_replay(
            l2_signal_batch=MagicMock(),
            l3_signal_batch=MagicMock(),
            aligned=aligned,
            awf_folds=folds,
            holdout_span=(0, 100),
            config=config,
            caps=caps,
            tf="4h",
            deploy_leverage=None,
        )
        assert len(results) == 5
        for r in results:
            assert r.baseline_parity is True

    @patch("src.domain.futures.strategy.tiered_workflow.pipeline.run_l2_awf")
    @patch("src.domain.futures.strategy.tiered_workflow.pipeline.run_l3_holdout")
    def test_l2_parity_mismatch_does_not_raise_only_warns(
        self,
        mock_l3: MagicMock,
        mock_l2: MagicMock,
        caplog: Any,
    ) -> None:
        import logging

        caplog.set_level(logging.WARNING)
        l2_replay = MagicMock(
            spec=Layer2Result,
            cagr_hybrid=0.582,
            mdd_hybrid=0.176,
            turnover=0.5,
            average_gross_exposure=0.95,
            gate_passed=True,
            blocker_reason="",
            directional_veto_summary=(),
            fold_pass_ratio=1.0,
            trade_count=214,
            deploy_leverage=2.0341,
            sharpe_hac_hybrid=1.5,
            sortino_hybrid=1.8,
        )
        l2_baseline = MagicMock(
            spec=Layer2Result,
            cagr_hybrid=0.999,
            mdd_hybrid=0.999,
            turnover=0.5,
            average_gross_exposure=0.95,
            gate_passed=True,
            blocker_reason="",
            directional_veto_summary=(),
            fold_pass_ratio=1.0,
            trade_count=214,
            deploy_leverage=2.0341,
            sharpe_hac_hybrid=1.5,
            sortino_hybrid=1.8,
        )
        l3_res = MagicMock(
            spec=Layer3Result,
            cagr=-0.171,
            mdd=0.268,
            sharpe=-1.21,
            total_return=-0.169,
            gate_passed=False,
            blocker_reason="negative_return",
            realized_price_long_by_symbol=(),
            directional_veto_summary=(),
        )
        mock_l2.return_value = l2_replay
        mock_l3.return_value = l3_res
        from types import SimpleNamespace

        aligned = SimpleNamespace(symbols=("BTCUSDT", "ETHUSDT", "BNBUSDT"))
        config = Layer2AllocationConfig.from_mapping({})
        caps = SimpleNamespace(gross=3.0, per_symbol=0.15, net=0.5, beta=1.0, target_ann_vol=0.20)
        folds = (SimpleNamespace(oos_start=0, oos_end=100),)

        results = run_directional_veto_economic_replay(
            l2_signal_batch=MagicMock(),
            l3_signal_batch=MagicMock(),
            aligned=aligned,
            awf_folds=folds,
            holdout_span=(0, 100),
            config=config,
            caps=caps,
            tf="4h",
            deploy_leverage=2.0341,
            baseline_l2=l2_baseline,
            baseline_l3=l3_res,
        )
        assert results[0].baseline_parity is False
        assert "[L2-PARITY-DIAG]" in caplog.text

    @patch("src.domain.futures.strategy.tiered_workflow.pipeline.run_l2_awf")
    @patch("src.domain.futures.strategy.tiered_workflow.pipeline.run_l3_holdout")
    def test_l2_parity_with_missing_optional_attrs_does_not_crash(
        self,
        mock_l3: MagicMock,
        mock_l2: MagicMock,
    ) -> None:
        from types import SimpleNamespace

        l2_replay = SimpleNamespace(
            cagr_hybrid=0.582,
            mdd_hybrid=0.176,
            turnover=0.5,
            average_gross_exposure=0.95,
            gate_passed=True,
            blocker_reason="",
            directional_veto_summary=(),
        )
        l2_baseline = SimpleNamespace(
            cagr_hybrid=0.582,
            mdd_hybrid=0.176,
            turnover=0.5,
            average_gross_exposure=0.95,
            gate_passed=True,
            blocker_reason="",
            directional_veto_summary=(),
        )
        l3_res = MagicMock(
            spec=Layer3Result,
            cagr=-0.171,
            mdd=0.268,
            sharpe=-1.21,
            total_return=-0.169,
            gate_passed=False,
            blocker_reason="negative_return",
            realized_price_long_by_symbol=(),
            directional_veto_summary=(),
        )
        mock_l2.return_value = l2_replay
        mock_l3.return_value = l3_res
        aligned = SimpleNamespace(symbols=("BTCUSDT", "ETHUSDT", "BNBUSDT"))
        config = Layer2AllocationConfig.from_mapping({})
        caps = SimpleNamespace(gross=3.0, per_symbol=0.15, net=0.5, beta=1.0, target_ann_vol=0.20)
        folds = (SimpleNamespace(oos_start=0, oos_end=100),)

        results = run_directional_veto_economic_replay(
            l2_signal_batch=MagicMock(),
            l3_signal_batch=MagicMock(),
            aligned=aligned,
            awf_folds=folds,
            holdout_span=(0, 100),
            config=config,
            caps=caps,
            tf="4h",
            deploy_leverage=2.0341,
            baseline_l2=l2_baseline,
            baseline_l3=l3_res,
        )
        assert isinstance(results[0].baseline_parity, bool)
