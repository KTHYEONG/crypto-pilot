from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.application.research.technical import trend_screen as ts
from src.common.errors import DataIntegrityError
from src.research.contracts import CostModel
from src.research.technical_experts.backtest import run_technical_expert_backtest


def synthetic_market(start: str = "2022-01-01", end: str = "2025-12-31 23:59:59"):
    """Deterministic oscillating 4h market plus a zero funding stream."""
    idx = pd.date_range(start, end, freq="4h", tz="UTC")
    t = np.arange(len(idx), dtype=np.float64)
    close = 100.0 + 0.02 * t + 30.0 * np.sin(t / 40.0) + 20.0 * np.cos(t / 150.0)
    frame = pd.DataFrame({
        "open": close - 0.2,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": 1000.0 + 500.0 * np.abs(np.sin(t / 5.0)),
    }, index=idx)
    funding = pd.Series(0.0, index=idx, dtype=np.float64)
    return frame, funding


@dataclasses.dataclass(frozen=True, slots=True)
class _FastGateConfig(ts.ReliabilityGateConfig):
    """Small bootstrap/null-draw config to keep screen tests fast."""
    n_bootstrap: int = 100
    fold_null_draws: int = 1000
    block_size: int = 1


def _install_fast_config(monkeypatch) -> None:
    monkeypatch.setattr(ts, "ReliabilityGateConfig", _FastGateConfig)
    monkeypatch.setattr(ts, "effective_worker_count", lambda *args, **kwargs: 1)


def _install_synthetic_data(monkeypatch) -> None:
    frame, funding = synthetic_market()
    monkeypatch.setattr(
        ts,
        "_load_symbol_data",
        lambda symbol, s, e: (
            frame.copy(), funding.copy(), {"perp_ohlcv": f"fp-{symbol}"}, 1.0,
        ),
    )


def _single_candidate_set():
    return tuple(
        c for c in ts.TREND_SCREEN_CANDIDATES
        if c.return_source == "technical_ema_alignment_long_v1"
    )


class TestTrendScreenScale:
    def test_exactly_450_cells_defined(self) -> None:
        assert len(ts.TREND_SCREEN_CANDIDATES) == 30
        assert len(ts.TREND_SCREEN_SYMBOLS) == 15
        assert len(ts.TREND_SCREEN_CANDIDATES) * len(ts.TREND_SCREEN_SYMBOLS) == 450


class TestHolmAdjustment:
    def test_holm_strongest_survives_when_very_small(self) -> None:
        p_values = [0.5] * 449 + [1e-6]
        passed = ts._holm_adjusted_pass(p_values, alpha=0.10)
        assert passed[-1] is True
        assert sum(passed) == 1

    def test_holm_adjusts_across_all_hypotheses(self) -> None:
        # 0.01 > 0.10/100 -> none survive the family-wise adjustment.
        assert not any(ts._holm_adjusted_pass([0.01] * 100, alpha=0.10))

    def test_data_invalid_p_one_never_passes(self) -> None:
        assert not any(ts._holm_adjusted_pass([1.0] * 450, alpha=0.10))


class TestDiscoveryRequirements:
    def _run(self, monkeypatch, *, net_cagr, t_stat, trade_count):
        _install_fast_config(monkeypatch)
        idx = pd.date_range("2022-04-01", periods=4000, freq="4h", tz="UTC")
        cell = ts.TrendScreenCell(
            return_source="technical_ema_alignment_long_v1",
            family="ema_alignment", side="LONG", symbol="BTCUSDT",
            data_valid=True, funding_coverage=1.0, trade_count=trade_count,
            net_cagr=net_cagr, mdd=-0.1, lcb90=0.05, t_stat=t_stat, fold_score=0.4,
            stress_verdict="PASS", p_negative=0.0, fingerprint="f",
            discovery_pass=False, rejected_reason=None,
        )
        run = ts._CellRun(cell)
        run.disc_equity = pd.Series(np.linspace(100.0, 120.0, len(idx)), index=idx)
        return run

    def test_min_trades_fails_closed(self, monkeypatch) -> None:
        run = self._run(monkeypatch, net_cagr=0.30, t_stat=3.0, trade_count=10)
        ts._apply_discovery_requirements([run], [True], ts.ReliabilityGateConfig())
        assert run.cell.discovery_pass is False
        assert "min_trades" in run.cell.rejected_reason

    def test_negative_cagr_and_low_tstat_fail_closed(self, monkeypatch) -> None:
        run = self._run(monkeypatch, net_cagr=-0.02, t_stat=1.2, trade_count=40)
        ts._apply_discovery_requirements([run], [True], ts.ReliabilityGateConfig())
        assert run.cell.discovery_pass is False
        assert "net_cagr" in run.cell.rejected_reason
        assert "t_stat" in run.cell.rejected_reason

    def test_holm_failure_fails_closed(self, monkeypatch) -> None:
        run = self._run(monkeypatch, net_cagr=0.30, t_stat=3.0, trade_count=40)
        ts._apply_discovery_requirements([run], [False], ts.ReliabilityGateConfig())
        assert run.cell.discovery_pass is False
        assert "holm_lcb90_not_above_zero" in run.cell.rejected_reason

    def test_all_discovery_requirements_mark_cell_eligible(self, monkeypatch) -> None:
        run = self._run(monkeypatch, net_cagr=0.30, t_stat=3.0, trade_count=40)
        ts._apply_discovery_requirements([run], [True], ts.ReliabilityGateConfig())
        assert run.cell.discovery_pass is True
        assert run.cell.rejected_reason is None


class TestScreenPipeline:
    def test_end_to_end_rejects_failed_candidate(self, monkeypatch) -> None:
        # BGP end-to-end fixture: on synthetic oscillating data the screen
        # retains CASH and records a binding constraint rather than promoting.
        _install_fast_config(monkeypatch)
        _install_synthetic_data(monkeypatch)
        monkeypatch.setattr(ts, "TREND_SCREEN_CANDIDATES", _single_candidate_set())

        report = ts.run_trend_screen()

        assert report.profile == "baseline_gate_performance_v1"
        assert len(report.cells) == len(_single_candidate_set()) * 15
        assert all(cell.data_valid for cell in report.cells)
        assert report.qualification.admitted is False
        assert report.qualification.binding_constraint is not None

    def test_report_is_byte_deterministic(self, monkeypatch) -> None:
        _install_fast_config(monkeypatch)
        _install_synthetic_data(monkeypatch)
        monkeypatch.setattr(ts, "TREND_SCREEN_CANDIDATES", _single_candidate_set())

        first = ts.run_trend_screen()
        second = ts.run_trend_screen()
        assert first.to_json() == second.to_json()
        payload = first.to_payload()
        assert len(payload["report_fingerprint"]) == 64
        assert all(
            cell["fingerprint"]["perp_ohlcv"].startswith("fp-")
            for cell in payload["cells"]
        )

    def test_missing_funding_invalidates_cells_fail_closed(self, monkeypatch) -> None:
        # BGP-03 at the screen level: an unloadable funding stream invalidates
        # the symbol's cells; funding is never replaced by a zero-cost fallback.
        _install_fast_config(monkeypatch)
        monkeypatch.setattr(
            ts,
            "_load_symbol_data",
            lambda symbol, s, e: (_ for _ in ()).throw(
                DataIntegrityError(f"no settled funding events in window for {symbol}")
            ),
        )
        monkeypatch.setattr(ts, "TREND_SCREEN_CANDIDATES", _single_candidate_set())

        report = ts.run_trend_screen()
        assert len(report.cells) == len(_single_candidate_set()) * 15
        assert all(not cell.data_valid for cell in report.cells)
        assert all("funding" in (cell.rejected_reason or "") for cell in report.cells)

    def test_load_symbol_data_missing_funding_fails_closed(self, monkeypatch) -> None:
        frame, _funding = synthetic_market()
        monkeypatch.setattr(
            ts, "load_ohlcv_1h_as", lambda *a, **k: frame,
        )
        monkeypatch.setattr(
            ts, "load_funding_rates", lambda path: pd.Series(
                [0.001], index=pd.DatetimeIndex([frame.index[0] - pd.Timedelta(days=7)]),
            ),
        )
        with pytest.raises(DataIntegrityError, match="funding"):
            ts._load_symbol_data("BTCUSDT", None, None)

    def test_no_discovery_qualification_cross_read(self, monkeypatch) -> None:
        # Selection uses only discovery evidence: mutating qualification bars
        # cannot change a cell's discovery metrics or pass verdict.
        _install_fast_config(monkeypatch)
        frame, funding = synthetic_market()
        candidate = ts._candidate_by_source("technical_ema_alignment_long_v1")

        base = ts._run_cell(
            frame, funding, candidate, "BTCUSDT", {"perp_ohlcv": "f"}, 1.0,
        )
        mutated = frame.copy()
        mutated.loc[mutated.index >= ts.QUALIFICATION_START, "close"] *= 1.5
        mutated.loc[mutated.index >= ts.QUALIFICATION_START, "open"] *= 1.5
        after = ts._run_cell(
            mutated, funding, candidate, "BTCUSDT", {"perp_ohlcv": "f"}, 1.0,
        )

        for field in ("trade_count", "net_cagr", "mdd", "lcb90", "t_stat", "fold_score"):
            assert getattr(base.cell, field) == pytest.approx(getattr(after.cell, field))
        assert base.cell.discovery_pass == after.cell.discovery_pass

    def test_discovery_cell_defers_stress_replay(self, monkeypatch) -> None:
        _install_fast_config(monkeypatch)
        frame, funding = synthetic_market()
        candidate = ts._candidate_by_source("technical_ema_alignment_long_v1")
        original = ts.run_technical_expert_backtest
        calls = 0

        def counted(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(ts, "run_technical_expert_backtest", counted)
        run = ts._run_cell(
            frame, funding, candidate, "BTCUSDT", {"perp_ohlcv": "f"}, 1.0,
        )

        assert calls == 1
        assert run.cell.stress_verdict == "PENDING"


class TestQualification:
    def _selected_runs(self) -> list[ts._CellRun]:
        frame, funding = synthetic_market()
        candidates = [
            ts._candidate_by_source("technical_ema_alignment_long_v1"),
            ts._candidate_by_source("technical_donchian_breakout_short_v1"),
        ]
        runs: list[ts._CellRun] = []
        for candidate in candidates:
            result = run_technical_expert_backtest(
                frame, candidate, CostModel(), funding,
                signal_delay_bars=ts._BASE_DELAY_BARS,
            )
            disc_equity = result.equity[
                (result.equity.index >= ts.DISCOVERY_START)
                & (result.equity.index <= ts.DISCOVERY_END)
            ]
            cell = ts.TrendScreenCell(
                return_source=candidate.return_source,
                family=candidate.family,
                side=candidate.side,
                symbol="BTCUSDT",
                data_valid=True, funding_coverage=1.0, trade_count=5,
                net_cagr=0.1, mdd=-0.1, lcb90=0.01, t_stat=2.1, fold_score=0.3,
                stress_verdict="PASS", p_negative=0.0, fingerprint="f",
                discovery_pass=True, rejected_reason=None,
            )
            run = ts._CellRun(cell, result)
            run.disc_equity = disc_equity
            runs.append(run)
        return runs

    def test_zero_pre_warmup_exposure_and_same_schedule_under_stress(
        self, monkeypatch,
    ) -> None:
        _install_fast_config(monkeypatch)
        frame, funding = synthetic_market()
        data = {"BTCUSDT": (frame, funding)}
        runs = self._selected_runs()
        weights = ts._equal_risk_weights(runs)

        schedule, scheduled, stress_scheduled, qualification = ts._qualify(
            runs, weights, data, unseal_holdout=False,
        )

        assert len(schedule) == len(scheduled) == len(stress_scheduled)
        lookback_bars = round(pd.Timedelta(days=365) / pd.Timedelta(hours=4))
        assert schedule.iloc[:lookback_bars].eq(0.0).all()
        # The stressed replay shares the identical frozen schedule.
        assert qualification.admitted is False
        assert qualification.binding_constraint is not None
