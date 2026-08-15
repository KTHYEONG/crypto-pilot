from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from src.application.research.technical import trend_screen as ts
from src.cli.main import main as cli_main


@dataclasses.dataclass(frozen=True, slots=True)
class _FastGateConfig(ts.ReliabilityGateConfig):
    n_bootstrap: int = 100
    fold_null_draws: int = 1000
    block_size: int = 1


def _install_fast_screen(monkeypatch: pytest.MonkeyPatch, report_path) -> None:
    idx = pd.date_range("2022-01-01", "2025-12-31 23:59:59", freq="4h", tz="UTC")
    t = np.arange(len(idx), dtype=np.float64)
    close = 100.0 + 0.02 * t + 30.0 * np.sin(t / 40.0) + 20.0 * np.cos(t / 150.0)
    frame = pd.DataFrame({
        "open": close - 0.2,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": 1000.0 + 500.0 * np.abs(np.sin(t / 5.0)),
    }, index=idx)
    funding = pd.Series(0.0, index=idx)

    monkeypatch.setattr(
        ts, "_load_symbol_data",
        lambda symbol, s, e: (
            frame.copy(), funding.copy(), {"perp_ohlcv": f"fp-{symbol}"}, 1.0,
        ),
    )
    monkeypatch.setattr(
        ts, "TREND_SCREEN_CANDIDATES",
        tuple(c for c in ts.TREND_SCREEN_CANDIDATES
              if c.return_source == "technical_ema_alignment_long_v1"),
    )
    monkeypatch.setattr(ts, "ReliabilityGateConfig", _FastGateConfig)
    monkeypatch.setattr(ts, "effective_worker_count", lambda *args, **kwargs: 1)
    monkeypatch.setattr(ts, "trend_screen_report_path", lambda: report_path)


def test_trend_screen_cli_leaf_runs_and_persists_deterministic_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    # CLI-01: the no-tuning trend-screen leaf accepts the source-controlled
    # profile, executes the screen, and persists a byte-deterministic report.
    report_path = tmp_path / "trend_screen_baseline_gate_performance_v1.json"
    _install_fast_screen(monkeypatch, report_path)

    cli_main(["research", "run", "single", "trend-screen"])

    assert report_path.exists()
    payload = json.loads(report_path.read_text())
    assert payload["profile"] == "baseline_gate_performance_v1"
    assert len(payload["report_fingerprint"]) == 64
    assert payload["qualification"]["admitted"] is False
    assert payload["qualification"]["binding_constraint"] is not None

    cli_main(["research", "run", "single", "trend-screen"])
    assert report_path.read_text() == report_path.read_text()


def test_trend_screen_cli_rejects_unknown_profile() -> None:
    from src.cli.main import build_root_parser

    parser = build_root_parser()
    args = parser.parse_args(
        ["research", "run", "single", "trend-screen", "--profile", "not_a_profile"]
    )
    with pytest.raises(ValueError, match="unknown trend-screen profile"):
        args.handler(args)
