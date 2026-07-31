from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest



def _breakout_frame(symbol: str, jump: float, signal_bar: int = 260) -> pd.DataFrame:
    """Flat base with isolated breakout cycles across two calendar years."""
    n = 4400
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    o = np.full(n, 100.0)
    h = np.full(n, 101.0)
    l_ = np.full(n, 99.0)
    c = np.full(n, 100.0)

    def cycle(start: int, bump: float) -> None:
        c[start] = 100.0 + bump
        h[start] = 100.0 + bump + 1.0
        l_[start] = 100.0 + bump - 1.0
        o[start + 1 : start + 8] = 100.0 + bump
        h[start + 1 : start + 8] = 100.0 + bump + 1.0
        l_[start + 1 : start + 8] = 100.0 + bump - 1.0
        c[start + 1 : start + 8] = 100.0 + bump
        o[start + 8 : start + 20] = 100.0 + bump - 2.0
        h[start + 8 : start + 20] = 100.0 + bump - 1.4
        l_[start + 8 : start + 20] = 100.0 + bump - 2.6
        c[start + 8 : start + 20] = 100.0 + bump - 2.0

    cycle(800, 6.0)
    cycle(2200, 9.0)
    quote_vol = np.full(n, 1000.0)
    return pd.DataFrame({
        "open": o, "high": h, "low": l_, "close": c,
        "quote_vol": quote_vol, "volume": 1000.0,
    }, index=idx)


@pytest.fixture
def portfolio_frames() -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series]]:
    """Five data-complete symbols with a full trailing liquidity window.

    Five symbols satisfy the frozen ``universe_size=5`` requirement while the
    frames span two calendar years so fold distribution is computable. Every
    symbol has a completed 30-day liquidity window before the first breakout,
    so the portfolio ledger opens positions under the frozen 2.5% aggregate
    initial-risk invariant.
    """
    frames = {
        "AAAUSDT": _breakout_frame("AAAUSDT", jump=6.0, signal_bar=800),
        "BBBUSDT": _breakout_frame("BBBUSDT", jump=9.0, signal_bar=2200),
        "CCCUSDT": _breakout_frame("CCCUSDT", jump=5.0, signal_bar=1400),
        "DDDUSDT": _breakout_frame("DDDUSDT", jump=8.0, signal_bar=2500),
        "EEEUSDT": _breakout_frame("EEEUSDT", jump=7.0, signal_bar=1800),
    }
    funding = {}
    for symbol, frame in frames.items():
        ts = frame.index[200]
        funding[symbol] = pd.Series([0.0], index=[ts], dtype=float)
    return frames, funding


@pytest.fixture
def temporary_source_units(tmp_path: Path) -> dict[str, Path]:
    """Canonical logical-unit -> source-file mapping on tmp files.

    Used to verify that ``compute_code_hash`` is ordered by logical-unit ID and
    sensitive to canonical unit bytes without touching real repository files.
    """
    units = {
        "unit.a": tmp_path / "a.py",
        "unit.b": tmp_path / "b.py",
        "unit.c": tmp_path / "c.py",
    }
    for unit_id, path in units.items():
        path.write_text(f"# {unit_id}\nVALUE = 1\n", encoding="utf-8")
    return units
