from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Project Root Setup
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.domain.futures.ml_pipeline.pipeline_runner import MLPipelineOutput
from src.execution import opt_main_futures


def _dummy_data_maps(symbols: list[str]) -> tuple[dict, dict]:
    data_maps = {s: {"4h": pd.DataFrame({"datetime": [], "close": []})} for s in symbols}
    return data_maps, dict(data_maps)


def test_hmm_only_skips_optimization_and_writes_snapshot(monkeypatch):
    symbols = ["BTCUSDT"]
    data_maps, oos_data_maps = _dummy_data_maps(symbols)
    optimization_called = False

    def _fake_run_opt(*args, **kwargs):
        nonlocal optimization_called
        optimization_called = True
        raise AssertionError("optimization loop must not run in --hmm-only mode")

    def _fake_ml(*args, **kwargs):
        out = MLPipelineOutput()
        out.hmm_report = {"hmm_prob_crisis_mean": 0.12}
        return out

    monkeypatch.setattr(opt_main_futures, "get_quarterly_window", lambda _ref: ("2025-01-01", "2025-03-01", "2025-04-01", "2025-05-01"))
    monkeypatch.setattr(opt_main_futures, "load_futures_data_maps_for_symbols", lambda *a, **k: (data_maps, oos_data_maps, symbols))
    monkeypatch.setattr(opt_main_futures, "run_ml_pipeline_for_universe", _fake_ml)
    monkeypatch.setattr(opt_main_futures, "run_v43_phase_optimization_skeleton", _fake_run_opt)
    monkeypatch.setattr(opt_main_futures, "resolve_futures_parallel_policy", lambda _n: 1)
    monkeypatch.setattr(sys, "argv", ["opt_main_futures.py", "--skip-universe", "--hmm-only", "--symbols", "BTCUSDT"])

    opt_main_futures.main()

    assert optimization_called is False


def test_alpha_only_skips_optimization_and_writes_snapshot(monkeypatch):
    symbols = ["BTCUSDT"]
    data_maps, oos_data_maps = _dummy_data_maps(symbols)
    optimization_called = False

    alpha_idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2025-01-01T00:00:00Z"), "BTCUSDT"),
            (pd.Timestamp("2025-01-01T04:00:00Z"), "BTCUSDT"),
        ],
        names=["datetime", "symbol"],
    )
    alpha_panel = pd.DataFrame({"score": [0.1, 0.2]}, index=alpha_idx)

    def _fake_run_opt(*args, **kwargs):
        nonlocal optimization_called
        optimization_called = True
        raise AssertionError("optimization loop must not run in --alpha-only mode")

    def _fake_ml(*args, **kwargs):
        out = MLPipelineOutput(alpha_panel=alpha_panel)
        out.hmm_report = {"hmm_prob_crisis_mean": 0.08}
        return out

    monkeypatch.setattr(opt_main_futures, "get_quarterly_window", lambda _ref: ("2025-01-01", "2025-03-01", "2025-04-01", "2025-05-01"))
    monkeypatch.setattr(opt_main_futures, "load_futures_data_maps_for_symbols", lambda *a, **k: (data_maps, oos_data_maps, symbols))
    monkeypatch.setattr(opt_main_futures, "run_ml_pipeline_for_universe", _fake_ml)
    monkeypatch.setattr(opt_main_futures, "run_v43_phase_optimization_skeleton", _fake_run_opt)
    monkeypatch.setattr(opt_main_futures, "resolve_futures_parallel_policy", lambda _n: 1)
    monkeypatch.setattr(sys, "argv", ["opt_main_futures.py", "--skip-universe", "--alpha-only", "--symbols", "BTCUSDT"])

    opt_main_futures.main()

    assert optimization_called is False
