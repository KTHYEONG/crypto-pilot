from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


from src.domain.futures.compound.config import L1LegConfig
from src.domain.futures.compound.l1_diagnostics import L1AdmissionRecorder


class TestL1AdmissionRecorder:
    def test_disabled_by_default(self) -> None:
        old = os.environ.pop("L1_DEBUG", None)
        try:
            rec = L1AdmissionRecorder()
            assert not rec.enabled
        finally:
            if old is not None:
                os.environ["L1_DEBUG"] = old

    def test_enabled_when_debug_is_1(self) -> None:
        old = os.environ.get("L1_DEBUG")
        os.environ["L1_DEBUG"] = "1"
        try:
            rec = L1AdmissionRecorder()
            assert rec.enabled
        finally:
            if old is not None:
                os.environ["L1_DEBUG"] = old
            else:
                os.environ.pop("L1_DEBUG", None)

    def test_record_sleeve_noop_when_disabled(self) -> None:
        old = os.environ.pop("L1_DEBUG", None)
        try:
            rec = L1AdmissionRecorder()
            rec.record_sleeve(signal_id="s", fold=0, cluster=0, beta=0.0, se_hac=0.1, se_ols_ratio=1.0, prob=0.5, n_obs=100, n_blocks=1, admitted=True)
        finally:
            if old is not None:
                os.environ["L1_DEBUG"] = old

    def test_record_gate_noop_when_disabled(self) -> None:
        old = os.environ.pop("L1_DEBUG", None)
        try:
            rec = L1AdmissionRecorder()
            rec.record_gate(admitted_sleeves=1, distinct_series=1, oos_bars=10, ann_growth=0.0, ann_lcb90=0.0, pw_block=5.0, turnover=0.0, cost_drag=0.0, admitted=False)
        finally:
            if old is not None:
                os.environ["L1_DEBUG"] = old

    def test_record_gate_writes_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "l1_admission.jsonl"
        old = os.environ.get("L1_DEBUG")
        os.environ["L1_DEBUG"] = "1"
        try:
            rec = L1AdmissionRecorder(path=path)
            assert rec.enabled
            rec.record_gate(admitted_sleeves=2, distinct_series=1, oos_bars=50, ann_growth=0.05, ann_lcb90=0.01, pw_block=5.0, turnover=0.1, cost_drag=0.0002, admitted=True)
            lines = path.read_text().strip().split("\n")
            assert len(lines) == 1
            parsed = json.loads(lines[0])
            assert parsed["tag"] == "EVAL"
            assert parsed["admitted"] is True
        finally:
            if old is not None:
                os.environ["L1_DEBUG"] = old
            else:
                os.environ.pop("L1_DEBUG", None)

    def test_record_sleeve_writes_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "l1_admission.jsonl"
        old = os.environ.get("L1_DEBUG")
        os.environ["L1_DEBUG"] = "1"
        try:
            rec = L1AdmissionRecorder(path=path)
            rec.record_sleeve(signal_id="trend:fast", fold=1, cluster=2, beta=0.5, se_hac=0.2, se_ols_ratio=2.5, prob=0.95, n_obs=500, n_blocks=12, admitted=True)
            lines = path.read_text().strip().split("\n")
            assert len(lines) == 1
            parsed = json.loads(lines[0])
            assert parsed["tag"] == "ALGO"
            assert parsed["signal_id"] == "trend:fast"
        finally:
            if old is not None:
                os.environ["L1_DEBUG"] = old
            else:
                os.environ.pop("L1_DEBUG", None)

    def test_unwritable_directory_does_not_raise(self) -> None:
        old = os.environ.get("L1_DEBUG")
        os.environ["L1_DEBUG"] = "1"
        try:
            rec = L1AdmissionRecorder(path=Path("/nonexistent_dir/out.jsonl"))
            rec.record_gate(admitted_sleeves=1, distinct_series=1, oos_bars=10, ann_growth=0.0, ann_lcb90=0.0, pw_block=5.0, turnover=0.0, cost_drag=0.0, admitted=False)
            assert not rec.enabled
        finally:
            if old is not None:
                os.environ["L1_DEBUG"] = old
            else:
                os.environ.pop("L1_DEBUG", None)

    def test_record_regime_evidence_writes_jsonl(self, tmp_path: Path) -> None:
        old = os.environ.get("L1_DEBUG")
        os.environ["L1_DEBUG"] = "1"
        try:
            log_path = tmp_path / "regime_test.jsonl"
            rec = L1AdmissionRecorder(path=log_path)
            rec.record_regime_evidence(
                signal_id="mom:fast", outer_fold_id=0, regime_code=2,
                effective_blocks=25, posterior_probability=0.95,
                growth_lcb90=0.05, growth_2x_cost=0.03,
                robust_inner_growth=0.02, positive_inner_folds=3,
                scale=0.8, admitted=True, reasons=(),
            )
            lines = log_path.read_text().strip().splitlines()
            assert len(lines) == 1
            row = json.loads(lines[0])
            assert row["tag"] == "REGIME"
            assert row["signal_id"] == "mom:fast"
            assert row["admitted"] is True
            assert row["scale"] == 0.8
        finally:
            if old is not None:
                os.environ["L1_DEBUG"] = old
            else:
                os.environ.pop("L1_DEBUG", None)

    def test_record_regime_evidence_disabled_when_no_debug(self, tmp_path: Path) -> None:
        old = os.environ.pop("L1_DEBUG", None)
        try:
            log_path = tmp_path / "disabled_regime.jsonl"
            rec = L1AdmissionRecorder(path=log_path)
            rec.record_regime_evidence(
                signal_id="mom:fast", outer_fold_id=0, regime_code=2,
                effective_blocks=25, posterior_probability=0.95,
                growth_lcb90=0.05, growth_2x_cost=0.03,
                robust_inner_growth=0.02, positive_inner_folds=3,
                scale=0.8, admitted=True, reasons=(),
            )
            assert not log_path.exists()
        finally:
            if old is not None:
                os.environ["L1_DEBUG"] = old
            else:
                os.environ.pop("L1_DEBUG", None)

    def test_record_family_screen_writes_jsonl(self, tmp_path: Path) -> None:
        old = os.environ.get("L1_DEBUG")
        os.environ["L1_DEBUG"] = "1"
        try:
            log_path = tmp_path / "family_screen.jsonl"
            rec = L1AdmissionRecorder(path=log_path)
            rec.record_family_screen(
                family="xs_reversal", n_signals=2, n_ic_bars=1360,
                mean_ic=0.0384, t_newey_west=3.53, sidak_alpha=0.0073,
                declared_orientation=-1, admitted=True, reasons=(),
            )
            lines = log_path.read_text().strip().splitlines()
            assert len(lines) == 1
            row = json.loads(lines[0])
            assert row["tag"] == "SCREEN"
            assert row["family"] == "xs_reversal"
            assert row["admitted"] is True
        finally:
            if old is not None:
                os.environ["L1_DEBUG"] = old
            else:
                os.environ.pop("L1_DEBUG", None)

    def test_record_family_screen_disabled_when_no_debug(self, tmp_path: Path) -> None:
        old = os.environ.pop("L1_DEBUG", None)
        try:
            log_path = tmp_path / "disabled_family_screen.jsonl"
            rec = L1AdmissionRecorder(path=log_path)
            rec.record_family_screen(
                family="momentum_ts", n_signals=5, n_ic_bars=1360,
                mean_ic=-0.0062, t_newey_west=-0.37, sidak_alpha=0.0073,
                declared_orientation=1, admitted=False,
                reasons=("insufficient_edge",),
            )
            assert not log_path.exists()
        finally:
            if old is not None:
                os.environ["L1_DEBUG"] = old

    def test_record_leg_emits_net_familywise_gate_diagnostics(self, tmp_path: Path, monkeypatch) -> None:
        import json

        from src.domain.futures.compound.contracts import LegBook, SignalConceptSpec
        from src.domain.futures.compound.l1_leg_evaluation import evaluate_leg_alpha

        log_path = tmp_path / "l1_admission.jsonl"
        monkeypatch.setenv("L1_DEBUG", "1")

        def _mock_init(self, path=None):
            self._enabled = True
            self._path = log_path

        monkeypatch.setattr(
            "src.domain.futures.compound.l1_diagnostics.L1AdmissionRecorder.__init__",
            _mock_init,
        )

        T, S = 60, 3
        rng = np.random.default_rng(1)
        market = rng.standard_normal(T).astype(np.float64) * 0.01
        book = np.full((T, S), 1.0 / S, dtype=np.float64)
        spec = SignalConceptSpec(
            concept_id="test_c", member_signal_ids=("sig_a",),
            mode="xs", horizon_band_bars=(6,), declared_orientation=1,
        )
        leg = LegBook(
            spec=spec, book_2d=book,
            gross_return_1d=rng.standard_normal(T).astype(np.float64) * 0.01,
            turnover_1d=np.zeros(T, dtype=np.float64),
        )
        evaluate_leg_alpha(
            leg, market, (slice(0, T),), 8.0,
            L1LegConfig(n_bootstrap=100),
            n_tested_hypotheses=11,
        )

        assert log_path.exists()
        records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        leg_records = [r for r in records if r.get("tag") == "LEG"]
        assert len(leg_records) >= 1
        last = leg_records[-1]
        required = {"net_alpha_ann", "t_net_alpha", "critical_t", "n_tested_hypotheses", "reasons"}
        assert required <= set(last), f"missing keys: {required - set(last)}"
        assert last["n_tested_hypotheses"] == 11
