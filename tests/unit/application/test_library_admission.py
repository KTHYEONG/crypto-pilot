from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.application.research.expert_portfolio import admission as app
from src.application.research.expert_portfolio.admission import run_technical_library_admission
from src.research.expert_portfolio.catalog import default_catalog
from src.research.expert_portfolio.admission_types import (
    LibraryAdmissionConfig,
    TechnicalLibraryAdmissionRequest,
)
from src.research.expert_portfolio.models import ContextualRouterSpec


def _frame(n: int = 4400) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    t = np.arange(n, dtype=np.float64)
    close = 100.0 + 0.01 * t + 5.0 * np.sin(t / 20.0)
    open_ = close - 0.1
    return pd.DataFrame({
        "open": open_,
        "high": np.maximum(open_, close) + 0.5,
        "low": np.minimum(open_, close) - 0.5,
        "close": close,
        "volume": 1000.0,
    }, index=index)


def _admission_config(max_workers: int | None = 1) -> LibraryAdmissionConfig:
    return LibraryAdmissionConfig(
        min_experts=1,
        max_experts=2,
        min_closed_trades=0,
        min_active_return_bars=0,
        max_abs_pairwise_log_return_correlation=0.8,
        max_joint_negative_return_rate=0.5,
        min_context_covered_states=1,
        max_combinations=100,
        max_workers=max_workers,
    )


def _request(max_workers: int | None = 1) -> TechnicalLibraryAdmissionRequest:
    return TechnicalLibraryAdmissionRequest(
        candidate_sources=(
            "technical_macd_histogram_regime_long_v1",
            "technical_rsi_trend_pullback_long_v1",
        ),
        symbols=("BTCUSDT", "ETHUSDT"),
        router=ContextualRouterSpec("BTCUSDT", 60, 20, 30),
        admission=_admission_config(max_workers=max_workers),
        start="2024-01-01",
    )


def _patch_environment(monkeypatch) -> pd.DataFrame:
    frame = _frame()
    funding = pd.Series(0.0, index=frame.index, dtype=float)
    monkeypatch.setattr(
        app, "_load_technical_market_data",
        lambda symbol, start, end: (frame, funding),
    )
    monkeypatch.setattr(
        app, "load_ohlcv_4h", lambda path, *, start=None, end=None: frame,
    )
    monkeypatch.setattr(
        app, "compute_code_hash", lambda *args, **kwargs: "c" * 64,
    )
    monkeypatch.setattr(
        app, "technical_data_hashes",
        lambda symbol: {"perp_ohlcv": "a" * 64, "funding": "b" * 64},
    )
    return frame


def test_application_runs_sealed_universe_and_records_provenance(
    monkeypatch, capsys,
) -> None:
    """LAE-07: one sealed window, exact-aligned router context, provenance hashes."""
    frame = _patch_environment(monkeypatch)
    report = run_technical_library_admission(_request(max_workers=1))

    assert report.status == "COMPLETE"
    assert report.window_start == str(frame.index[0])
    assert report.window_end == str(frame.index[-1])
    assert report.code_hash == "c" * 64
    assert report.data_hashes == {
        "BTCUSDT": {"perp_ohlcv": "a" * 64, "funding": "b" * 64},
        "ETHUSDT": {"perp_ohlcv": "a" * 64, "funding": "b" * 64},
    }
    assert [e.expert_id for e in report.experts] == sorted(
        e.expert_id for e in report.experts
    )
    assert len(report.experts) == 4
    assert report.fingerprint()["code_hash"] == "c" * 64
    assert "max_workers" not in json.dumps(report.fingerprint())
    assert default_catalog().blueprints == {}


def test_application_sealed_end_past_cutoff_is_rejected() -> None:
    with __import__("pytest").raises(RuntimeError, match="Holdout sealed"):
        TechnicalLibraryAdmissionRequest(
            candidate_sources=("technical_macd_histogram_regime_long_v1",),
            symbols=("BTCUSDT",),
            router=ContextualRouterSpec("BTCUSDT", 60, 20, 30),
            admission=_admission_config(),
            end="2026-06-01",
        )


class _FakeFuture:
    def __init__(self, fn, *args):
        self._fn = fn
        self._args = args

    def result(self):
        return self._fn(*self._args)

    def cancel(self) -> bool:
        return True


class _FakeExecutor:
    def __init__(self, max_workers: int):
        self.max_workers = max_workers
        self.submits: list[tuple[object, ...]] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def submit(self, fn, *args):
        self.submits.append(args)
        return _FakeFuture(fn, *args)


def test_parallel_determinism_and_one_task_per_symbol(monkeypatch) -> None:
    """LAE-07A: 1-worker and 2-worker runs agree; one task per distinct symbol."""
    index = pd.date_range("2024-01-01", periods=40, freq="4h", tz="UTC")

    def _canned_series(symbol: str, source: str) -> pd.Series:
        seed = int(np.frombuffer(f"{symbol}:{source}".encode(), dtype=np.uint8).sum())
        rng = np.random.default_rng(seed)
        values = rng.normal(0.001, 0.01, size=len(index))
        values[0] = np.nan
        return pd.Series(values, index=index)

    def fake_worker(symbol, sources, start, end):
        return {source: (_canned_series(symbol, source), 5) for source in sources}

    monkeypatch.setattr(app, "_symbol_admission_worker", fake_worker)
    monkeypatch.setattr(
        app, "_build_admission_context",
        lambda router, idx, start, end: pd.Series(["up_low_vol"] * len(idx), index=idx),
    )
    monkeypatch.setattr(app, "compute_code_hash", lambda *args, **kwargs: "c" * 64)
    monkeypatch.setattr(
        app, "technical_data_hashes",
        lambda symbol: {"perp_ohlcv": "a" * 64, "funding": "b" * 64},
    )

    sequential = run_technical_library_admission(_request(max_workers=1))
    fake_executor = _FakeExecutor(max_workers=2)
    monkeypatch.setattr(app, "ProcessPoolExecutor", lambda **kwargs: fake_executor)
    parallel = run_technical_library_admission(_request(max_workers=2))

    assert fake_executor.max_workers == 2
    assert len(fake_executor.submits) == 2
    submitted_symbols = sorted(args[0] for args in fake_executor.submits)
    assert submitted_symbols == ["BTCUSDT", "ETHUSDT"]

    assert sequential.fingerprint() == parallel.fingerprint()
    assert sequential.to_report_dict()["fingerprint"] == parallel.to_report_dict()["fingerprint"]
    assert [e.expert_id for e in sequential.experts] == [e.expert_id for e in parallel.experts]
    assert sequential.execution_workers == 1
    assert parallel.execution_workers == 2
