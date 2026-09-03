from __future__ import annotations

import pandas as pd

from src.market_data.services import mhs_execution as mc


def test_mhs_execution_plan_is_pit_and_dry_run_writes_manifest(tmp_path, monkeypatch) -> None:
    """MHS-COLLECT-01-DRY-RUN: plan before network collection."""
    idx = pd.date_range("2025-01-01", periods=2200, freq="1h", tz="UTC")
    quote = pd.DataFrame({f"S{i:02d}": float(i + 1) for i in range(16)}, index=idx)
    close = pd.DataFrame(
        {symbol: 100.0 + (i + 1) * pd.Series(range(len(idx)), index=idx)
         for i, symbol in enumerate(quote.columns)},
    )
    monkeypatch.setattr(mc, "load_base_panel", lambda *args, **kwargs: {"close": close, "quote_vol": quote})
    monkeypatch.setattr(mc, "funding_path", lambda symbol: tmp_path / f"{symbol}.parquet")
    for symbol in quote.columns:
        (tmp_path / f"{symbol}.parquet").touch()
    monkeypatch.setattr(mc, "_manifest_path", lambda *args: tmp_path / "plan.json")

    plan = mc.build_mhs_execution_plan("2025-01-01", "2025-03-30", execution_universe_size=8)
    result = mc.collect_mhs_execution_data(plan)

    assert plan.symbols
    assert len(plan.symbols) <= 8
    assert result["mode"] == "dry_run"
    assert (tmp_path / "plan.json").exists()
