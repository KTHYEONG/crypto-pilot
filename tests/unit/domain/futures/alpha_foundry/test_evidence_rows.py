from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from src.domain.futures.alpha_foundry.bridge_helpers import (
    _write_alpha_foundry_report,
)
from src.domain.futures.alpha_foundry.contracts import AlphaFoundryEvidenceRow


@dataclass
class FakeReport:
    run_id: str
    mode: str
    timeframe: str
    symbols: tuple[str, ...]
    n_bars: int
    n_panels_in: int
    n_bound_panels: int
    n_evidence: int
    n_passed: int
    n_rejected: int
    reject_reason_counts: dict[str, int]
    elapsed_sec: float


class TestWriteAlphaFoundryReport:
    def _make_evidence_rows(self, n: int) -> list[AlphaFoundryEvidenceRow]:
        return [
            AlphaFoundryEvidenceRow(
                run_id="test_run",
                timeframe="4h",
                family="trend_ma",
                variant=f"v{i}",
                recipe_id=f"r{i}",
                archetype="trend",
                n_events=100,
                effective_n=50.0,
                mean_net_bps=5.0 + i,
                nw_tstat=2.0,
                block_lcb_bps=2.0 + i * 0.5,
                rank_ic=0.05,
                incremental_rank_ic=0.02,
                cost_drag_ratio=0.3,
                turnover_per_year=100.0,
                compute_cost_score=0.5,
                gate_passed=True,
                reject_reasons="",
                bucket_key="trend_ma:4h",
                bucket_rank=i,
                selected_for_l1=True,
                redundant_with="",
                bucket_eff_test_count=2.0,
                global_eff_test_count=3.0,
                bootstrap_lcb_bps=1.0 + i * 0.5,
                bootstrap_agree=True,
                created_at_ms=1000,
            )
            for i in range(n)
        ]

    # Scenario 1.5: evidence_rows 3건 → parquet 기록 확인
    def test_writes_evidence_rows_to_parquet(self, tmp_path: Path) -> None:
        report = FakeReport(
            run_id="test_run",
            mode="gate",
            timeframe="4h",
            symbols=("BTCUSDT",),
            n_bars=100,
            n_panels_in=10,
            n_bound_panels=8,
            n_evidence=5,
            n_passed=3,
            n_rejected=2,
            reject_reason_counts={},
            elapsed_sec=0.1,
        )
        rows = self._make_evidence_rows(3)
        _json_path, parquet_path = _write_alpha_foundry_report(
            report=report,
            evidence_rows=rows,
            report_dir=tmp_path,
            run_id="test_run",
        )
        parquet_file = Path(parquet_path)
        assert parquet_file.exists()
        df = pd.read_parquet(parquet_path)
        assert len(df) == 3
        for col in ("mean_net_bps", "block_lcb_bps", "cost_drag_ratio", "turnover_per_year"):
            assert col in df.columns

    # Scenario 2.8: 빈 evidence → 빈 DataFrame + 스키마 존재
    def test_empty_evidence_rows_writes_schema_only(self, tmp_path: Path) -> None:
        report = FakeReport(
            run_id="empty_run",
            mode="audit",
            timeframe="4h",
            symbols=(),
            n_bars=0,
            n_panels_in=0,
            n_bound_panels=0,
            n_evidence=0,
            n_passed=0,
            n_rejected=0,
            reject_reason_counts={},
            elapsed_sec=0.0,
        )
        _json_path, parquet_path = _write_alpha_foundry_report(
            report=report,
            evidence_rows=(),
            report_dir=tmp_path,
            run_id="empty_run",
        )
        parquet_file = Path(parquet_path)
        assert parquet_file.exists()
        df = pd.read_parquet(parquet_path)
        assert len(df) == 0
        assert "mean_net_bps" in df.columns
        assert "reject_reasons" in df.columns

    # Scenario 3.5: OSError 전파
    def test_raises_on_write_error(self, tmp_path: Path) -> None:
        report = FakeReport(
            run_id="fail_run",
            mode="gate",
            timeframe="4h",
            symbols=(),
            n_bars=0,
            n_panels_in=0,
            n_bound_panels=0,
            n_evidence=0,
            n_passed=0,
            n_rejected=0,
            reject_reason_counts={},
            elapsed_sec=0.0,
        )
        rows = self._make_evidence_rows(1)
        read_only_dir = tmp_path / "readonly"
        read_only_dir.mkdir(parents=True, exist_ok=True)
        read_only_dir.chmod(0o444)
        with pytest.raises(OSError, match=r"[Pp]ermission"):
            _write_alpha_foundry_report(
                report=report,
                evidence_rows=rows,
                report_dir=read_only_dir,
                run_id="fail_run",
            )
