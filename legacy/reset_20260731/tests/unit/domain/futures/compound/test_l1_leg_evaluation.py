from __future__ import annotations

import numpy as np

from src.domain.futures.compound.config import L1LegConfig
from src.domain.futures.compound.contracts import LegBook, SignalConceptSpec
from src.domain.futures.compound.l1_leg_evaluation import (
    compute_breakeven_cost_bps,
    evaluate_leg_alpha,
)


def _make_spec() -> SignalConceptSpec:
    return SignalConceptSpec(
        concept_id="test", member_signal_ids=("sig_a",),
        mode="xs", horizon_band_bars=(6,), declared_orientation=1,
    )


class TestComputeBreakevenCostBps:
    def test_compute_breakeven_cost_bps_zero_turnover(self) -> None:
        assert compute_breakeven_cost_bps(0.0, 0.0, 2190.0) == 0.0
        assert compute_breakeven_cost_bps(0.05, 0.0, 2190.0) == 0.0
        assert compute_breakeven_cost_bps(0.0, 0.01, 2190.0) == 0.0

    def test_compute_breakeven_cost_bps_known_value(self) -> None:
        result = compute_breakeven_cost_bps(0.05, 0.01, 2190.0)
        expected = 0.05 / (0.01 * 2190.0) * 1e4
        assert abs(result - expected) < 1e-10

    def test_compute_breakeven_cost_bps_sign_follows_alpha(self) -> None:
        result = compute_breakeven_cost_bps(-0.2190, 0.01, 2190.0)
        assert abs(result - (-100.0)) < 1e-6


class TestEvaluateLegAlpha:
    def test_evaluate_leg_alpha_strips_market_beta(self) -> None:
        T, S = 300, 5
        rng = np.random.default_rng(42)
        market = rng.standard_normal(T).astype(np.float64) * 0.02
        noise = rng.standard_normal(T).astype(np.float64) * 0.0005
        gross_return = 2.0 * market + noise
        book = np.full((T, S), 1.0 / S, dtype=np.float64)
        turnover = np.zeros(T, dtype=np.float64)
        leg = LegBook(
            spec=_make_spec(), book_2d=book,
            gross_return_1d=gross_return, turnover_1d=turnover,
        )
        oos = (slice(T // 2, T),)
        evidence = evaluate_leg_alpha(
            leg, market, oos, 8.0, L1LegConfig(n_bootstrap=100),
        )
        assert abs(evidence.beta_market - 2.0) < 0.2
        assert abs(evidence.alpha_ann) < 0.5

    def test_evaluate_leg_alpha_uses_net_returns_for_fold_and_tstat_evidence(self) -> None:
        T, S = 120, 3
        rng = np.random.default_rng(42)
        gross_return = rng.standard_normal(T).astype(np.float64) * 0.001 + 0.002
        turnover = np.full(T, 0.5, dtype=np.float64)
        book = np.full((T, S), 1.0 / S, dtype=np.float64)
        leg = LegBook(
            spec=_make_spec(), book_2d=book,
            gross_return_1d=gross_return, turnover_1d=turnover,
        )
        market = np.zeros(T, dtype=np.float64)
        oos = (slice(0, T),)
        evidence = evaluate_leg_alpha(
            leg, market, oos, 8.0, L1LegConfig(n_bootstrap=100),
            n_tested_hypotheses=1,
        )
        cost_per_bar = 8.0 * 1e-4 * 0.5
        expected_net_ann = float(np.mean(gross_return - cost_per_bar)) * 2190.0
        assert evidence.net_alpha_ann < evidence.alpha_ann
        assert abs(evidence.net_alpha_ann - expected_net_ann) < 0.1
        assert evidence.n_folds == 1

    def test_evaluate_leg_alpha_emits_leg_diagnostics_under_l1_debug(
        self, tmp_path, monkeypatch,
    ) -> None:
        import json

        log_path = tmp_path / "l1_admission.jsonl"
        monkeypatch.setenv("L1_DEBUG", "1")
        def _mock_init(self, path=None):
            self._enabled = True
            self._path = log_path

        monkeypatch.setattr(
            "src.domain.futures.compound.l1_diagnostics.L1AdmissionRecorder.__init__",
            _mock_init,
        )
        T, S = 50, 3
        rng = np.random.default_rng(1)
        market = rng.standard_normal(T).astype(np.float64) * 0.01
        book = np.full((T, S), 1.0 / S, dtype=np.float64)
        leg = LegBook(
            spec=_make_spec(), book_2d=book,
            gross_return_1d=rng.standard_normal(T).astype(np.float64) * 0.01,
            turnover_1d=np.zeros(T, dtype=np.float64),
        )
        evaluate_leg_alpha(leg, market, (slice(0, T),), 8.0, L1LegConfig(n_bootstrap=50))
        assert log_path.exists()
        records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        assert any(r.get("tag") == "LEG" and r.get("concept_id") == "test" for r in records)

    def test_record_leg_is_noop_when_disabled(self, tmp_path, monkeypatch) -> None:
        from src.domain.futures.compound.l1_diagnostics import L1AdmissionRecorder

        monkeypatch.delenv("L1_DEBUG", raising=False)
        log_path = tmp_path / "l1_admission.jsonl"
        recorder = L1AdmissionRecorder(path=log_path)
        assert recorder.enabled is False
        recorder.record_leg(
            concept_id="c", mode="xs", alpha_ann=0.0, beta_market=0.0,
            alpha_sharpe=0.0, t_alpha=0.0, breakeven_cost_bps=0.0,
            mean_turnover_per_bar=0.0, positive_folds=0, n_folds=0,
            posterior_positive=0.0, evidence_weight=0.0, reasons=(),
        )
        assert not log_path.exists()

    def test_evaluate_leg_alpha_no_oos_slices(self) -> None:
        T, S = 20, 3
        book = np.zeros((T, S), dtype=np.float64)
        leg = LegBook(
            spec=_make_spec(), book_2d=book,
            gross_return_1d=np.zeros(T), turnover_1d=np.zeros(T),
        )
        evidence = evaluate_leg_alpha(
            leg, np.zeros(T), (), 8.0, L1LegConfig(),
        )
        assert evidence.reasons == ("no_oos_folds",)
        assert evidence.evidence_weight == 0.0

    def test_evaluate_leg_alpha_gross_positive_net_negative(self) -> None:
        T, S = 120, 3
        rng = np.random.default_rng(42)
        gross_return = rng.standard_normal(T).astype(np.float64) * 0.001 + 0.005
        turnover = np.full(T, 1.0, dtype=np.float64)
        book = np.full((T, S), 1.0 / S, dtype=np.float64)
        leg = LegBook(
            spec=_make_spec(), book_2d=book,
            gross_return_1d=gross_return, turnover_1d=turnover,
        )
        market = np.zeros(T, dtype=np.float64)
        oos = (slice(0, T),)
        evidence = evaluate_leg_alpha(
            leg, market, oos, 8.0, L1LegConfig(n_bootstrap=100),
            n_tested_hypotheses=1,
        )
        assert evidence.alpha_ann > 0
        assert evidence.net_alpha_ann < evidence.alpha_ann
