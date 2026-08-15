from __future__ import annotations

import dataclasses
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.application.research.universe import candidate_scan as universe_scan
from src.research.universe.candidate_scan import (
    UniverseCandidateSpec,
    evaluate_universe_candidate,
)


def _frame(n: int = 200) -> pd.DataFrame:
    idx = pd.date_range("2022-04-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame(
        {
            "close": np.full(n, 100.0),
            "taker_buy_ratio": np.full(n, 0.5),
            "quote_vol": np.full(n, 1_000_000.0),
        },
        index=idx,
    )


def _load(path: Path, timeframe: str, *, start, end) -> pd.DataFrame:
    """Fake loader honoring the real loader's [start, end] restriction."""
    frame = _frame()
    lo = pd.Timestamp(start) if start is not None else frame.index[0]
    hi = pd.Timestamp(end) if end is not None else frame.index[-1]
    return frame.loc[(frame.index >= lo) & (frame.index <= hi)]


def _install_lake(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the orchestrator's I/O at a temp lake and stub the loader."""
    monkeypatch.setattr(
        universe_scan, "ohlcv_path",
        lambda symbol, timeframe: tmp_path / f"{symbol}.parquet",
    )
    monkeypatch.setattr(
        universe_scan, "funding_path",
        lambda symbol: tmp_path / f"{symbol}.funding",
    )
    monkeypatch.setattr(universe_scan, "load_ohlcv_1h_as", _load)


_WINDOW = (_frame().index[0], _frame().index[-1])


class TestRunUniverseCandidateScan:
    # UCS-04-SCAN-SKIPS-MISSING-SYMBOLS-WITHOUT-RAISING
    def test_ucs_04_scan_skips_missing_symbols_without_raising(self, monkeypatch, tmp_path) -> None:
        (tmp_path / "AUSDT.parquet").write_text("x", encoding="utf-8")
        (tmp_path / "AUSDT.funding").write_text("x", encoding="utf-8")
        _install_lake(monkeypatch, tmp_path)

        results = universe_scan.run_universe_candidate_scan(
            ("AUSDT", "MISSINGUSDT"), *_WINDOW,
        )

        assert {r.symbol for r in results} == {"AUSDT"}
        assert all(r.qualifies for r in results)

    def test_ucs_04_load_failure_is_skipped_not_raised(self, monkeypatch, tmp_path) -> None:
        (tmp_path / "AUSDT.parquet").write_text("x", encoding="utf-8")
        (tmp_path / "BROKENUSDT.parquet").write_text("x", encoding="utf-8")
        (tmp_path / "AUSDT.funding").write_text("x", encoding="utf-8")
        _install_lake(monkeypatch, tmp_path)

        def failing_load(path, timeframe, *, start, end):
            if "BROKENUSDT" in str(path):
                raise ValueError("corrupt frame")
            return _load(path, timeframe, start=start, end=end)

        monkeypatch.setattr(universe_scan, "load_ohlcv_1h_as", failing_load)

        results = universe_scan.run_universe_candidate_scan(
            ("AUSDT", "BROKENUSDT"), *_WINDOW,
        )
        assert {r.symbol for r in results} == {"AUSDT"}

    def test_deterministic_given_same_lake_state(self, monkeypatch, tmp_path) -> None:
        (tmp_path / "AUSDT.parquet").write_text("x", encoding="utf-8")
        (tmp_path / "AUSDT.funding").write_text("x", encoding="utf-8")
        _install_lake(monkeypatch, tmp_path)
        kwargs = {"symbols": ("AUSDT",), "discovery_start": _WINDOW[0], "end": _WINDOW[1]}
        first = universe_scan.run_universe_candidate_scan(**kwargs)
        second = universe_scan.run_universe_candidate_scan(**kwargs)
        assert [dataclasses.asdict(r) for r in first] == [dataclasses.asdict(r) for r in second]

    def test_entry_point_uses_existing_loader_and_pure_evaluator(self) -> None:
        source = inspect.getsource(universe_scan.run_universe_candidate_scan)
        assert "load_ohlcv_1h_as" in source
        assert "evaluate_universe_candidate" in source
        assert "funding_path" in source


class TestPersistUniverseCandidateScan:
    # UCS-05-PERSIST-LEDGER-ROUNDTRIP
    def test_ucs_05_persist_ledger_roundtrip(self, tmp_path) -> None:
        frame = _frame()
        results = (
            evaluate_universe_candidate(
                "AUSDT", frame, True, frame.index[0], frame.index[-1],
                UniverseCandidateSpec(),
            ),
            evaluate_universe_candidate(
                "BUSDT", frame, True, frame.index[0], frame.index[-1],
                UniverseCandidateSpec(),
            ),
        )
        path = tmp_path / "xs_alpha_universe_candidate_scan.json"

        universe_scan.persist_universe_candidate_scan(results, path)

        # No 'reliability' key -> the ledger's documented non-gated fallback.
        fail_ledger = tmp_path / "xs_alpha_reliability_fail.json"
        assert fail_ledger.exists()
        ledger = json.loads(fail_ledger.read_text(encoding="utf-8"))
        assert set(ledger) == {path.stem}
        entry = ledger[path.stem]
        assert set(entry) == {"AUSDT", "BUSDT"}
        for symbol in ("AUSDT", "BUSDT"):
            assert entry[symbol]["qualifies"] is True
            assert entry[symbol]["coverage"] == 1.0
            assert entry[symbol]["taker_ratio_valid"] is True
            assert entry[symbol]["avg_daily_quote_vol_recent"] == pytest.approx(6_000_000.0)
            assert entry[symbol]["first_bar"] == frame.index[0].isoformat()
            assert entry[symbol]["last_bar"] == frame.index[-1].isoformat()

        # Deterministic roundtrip: re-persisting produces an identical ledger.
        universe_scan.persist_universe_candidate_scan(results, path)
        assert json.loads(fail_ledger.read_text(encoding="utf-8")) == ledger

    def test_ucs_05_float_fields_are_rounded_and_finite(self, tmp_path) -> None:
        idx = pd.date_range("2022-04-01", periods=200, freq="4h", tz="UTC")
        frame = pd.DataFrame(
            {
                "close": np.full(200, 100.0),
                "taker_buy_ratio": np.full(200, 0.5),
                "quote_vol": np.linspace(1000.0, 2_000_000.0, 200),
            },
            index=idx,
        )
        result = evaluate_universe_candidate(
            "AUSDT", frame, True, idx[0], idx[-1], UniverseCandidateSpec(),
        )
        path = tmp_path / "xs_alpha_universe_candidate_scan.json"
        universe_scan.persist_universe_candidate_scan((result,), path)
        ledger = json.loads((tmp_path / "xs_alpha_reliability_fail.json").read_text(encoding="utf-8"))
        coverage = ledger[path.stem]["AUSDT"]["coverage"]
        vol = ledger[path.stem]["AUSDT"]["avg_daily_quote_vol_recent"]
        assert isinstance(coverage, float)
        assert coverage == round(coverage, 8)
        assert isinstance(vol, float)
        assert vol == round(vol, 8)

    def test_persistence_reuses_existing_ledger_entry_point(self) -> None:
        source = inspect.getsource(universe_scan.persist_universe_candidate_scan)
        assert "persist_reliability_ledger_entry" in source
