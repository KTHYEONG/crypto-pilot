"""Fail-closed wiring of the settlement-registry audit at every replay entry point."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

import src.mhs.evaluation.guards as guards_mod
import src.mhs.pipeline.stages.panel as panel_stage
from src.common.errors import DataIntegrityError
from src.mhs.contracts import MhsDiagnosticRequest
from src.mhs.pipeline.context import PipelineContext
from src.mhs.telemetry import StageTelemetry

T0 = pd.Timestamp("2025-01-01T00:00:00Z")
STEP = pd.Timedelta(minutes=3)
GRID = pd.date_range("2025-01-01", periods=3, freq="1h", tz="UTC")


def _write_3m(root, symbol: str, start: pd.Timestamp, count: int, quote_vol: float = 10.0) -> None:
    path = root / "3m" / f"{symbol}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    stamps = [int((start + STEP * i).value // 1_000_000) for i in range(count)]
    pd.DataFrame({
        "timestamp": stamps,
        "open": [1.0] * count, "high": [1.0] * count, "low": [1.0] * count,
        "close": [1.0] * count, "quote_vol": [quote_vol] * count,
    }).to_parquet(path, index=False)


def _panel_context(data_root: str, start: pd.Timestamp, end: pd.Timestamp) -> PipelineContext:
    return PipelineContext(
        config=MhsDiagnosticRequest(data_root=data_root),
        resolved_end=None,
        start=start,
        end=end,
        rss_budget_bytes=None,
        rss_reserve_bytes=None,
        root="",
        grid_1h=pd.DatetimeIndex([]),
        close=pd.DataFrame(),
        opens=pd.DataFrame(),
        quote_vol=pd.DataFrame(),
        taker_buy_quote=None,
        symbols=[],
    )


def _mock_panel_stage(monkeypatch: pytest.MonkeyPatch, symbols: list[str]) -> None:
    monkeypatch.setattr(panel_stage, "_resolve_ram_budget", lambda *_a, **_k: (None, None), raising=False)
    monkeypatch.setattr(
        panel_stage,
        "load_base_panel",
        lambda *_a, **_k: {
            "close": pd.DataFrame(1.0, index=GRID, columns=symbols),
            "open": pd.DataFrame(1.0, index=GRID, columns=symbols),
            "quote_vol": pd.DataFrame(1.0, index=GRID, columns=symbols),
            "taker_buy_quote": pd.DataFrame(1.0, index=GRID, columns=symbols),
        },
    )
    monkeypatch.setattr(panel_stage, "_guard_stage_or_breach", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(guards_mod, "_guard_stage_or_breach", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(
        panel_stage, "_load_funding_series",
        lambda names: ({s: pd.Series(0.0001, index=GRID) for s in names}, []), raising=False,
    )
    monkeypatch.setattr(
        panel_stage, "bar_funding_panel",
        lambda funding_window, grid: pd.DataFrame(0.0001, index=grid, columns=list(funding_window)),
    )


def test_panel_stage_fails_closed_on_incomplete_registry(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_3m(tmp_path, "AAAUSDT", T0, 20)
    _mock_panel_stage(monkeypatch, ["AAAUSDT"])
    ctx = _panel_context(str(tmp_path), T0, pd.Timestamp("2025-06-01T00:00:00Z"))
    with pytest.raises(DataIntegrityError, match="AAAUSDT"):
        panel_stage.load_panel(ctx, StageTelemetry(log_run=False))


def test_panel_audit_horizon_covers_fold_calendar(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.core.settlement_evidence import audit_settlement_registry
    from src.core.instrument_settlements import EMPTY_SETTLEMENT_REGISTRY

    _write_3m(tmp_path, "AAAUSDT", pd.Timestamp("2025-05-01T00:00:00Z"), 20)
    end = pd.Timestamp("2025-01-01T00:00:00Z")
    alone = audit_settlement_registry(
        tmp_path, ["AAAUSDT"], audit_end=end, registry=EMPTY_SETTLEMENT_REGISTRY,
    )
    assert alone.complete
    _mock_panel_stage(monkeypatch, ["AAAUSDT"])
    ctx = _panel_context(str(tmp_path), T0, end)
    with pytest.raises(DataIntegrityError, match="AAAUSDT"):
        panel_stage.load_panel(ctx, StageTelemetry(log_run=False))


def test_frozen_source_fails_closed(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.engine.strategy_backtest as frozen_mod
    from src.engine.strategy_backtest import _load_frozen_source
    from src.core.resources import resolve_mhs_memory_budget
    from tests.unit.engine.test_frozen_research_run import _request

    _write_3m(tmp_path, "AAAUSDT", T0, 20)
    grid = pd.date_range("2025-01-01", periods=10, freq="1h", tz="UTC")
    frame = pd.DataFrame(1.0, index=grid, columns=["AAAUSDT"])
    monkeypatch.setattr(
        frozen_mod, "load_base_panel",
        lambda *_a, **_k: {"close": frame, "quote_vol": frame, "taker_buy_quote": frame},
    )

    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("candidate construction must not run after a failed audit")

    monkeypatch.setattr(frozen_mod, "_load_funding_series", _boom, raising=False)
    request = _request(
        data_root=tmp_path,
        source_start=pd.Timestamp("2025-01-01", tz="UTC"),
        evaluation_start=pd.Timestamp("2025-01-05", tz="UTC"),
        evaluation_end=pd.Timestamp("2025-01-10", tz="UTC"),
    )
    with pytest.raises(DataIntegrityError, match="AAAUSDT"):
        _load_frozen_source(request, resolve_mhs_memory_budget(None), None)


def test_process_backtest_fails_closed(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.mhs.backtest.inventory as inventory_mod
    from src.mhs.backtest.contracts import ProcessInventoryBacktestError
    from src.mhs.process import ProcessExecutionPolicy

    _write_3m(tmp_path, "AAAUSDT", T0, 20)
    index = pd.date_range("2025-01-01", periods=5, freq="1h", tz="UTC")
    targets = pd.DataFrame(0.1, index=index, columns=["AAAUSDT"])
    proxy = SimpleNamespace(
        base=SimpleNamespace(
            target_weights=targets,
            signal_available_at=None,
            execution_policy=ProcessExecutionPolicy(),
        ),
    )
    monkeypatch.setattr(inventory_mod, "evaluate_process_backtest", lambda *_a, **_k: proxy)
    monkeypatch.setattr(inventory_mod, "_load_funding_series", lambda names: ({}, {}))

    def _boom(*_a: object, **_k: object) -> None:
        raise AssertionError("replay must not start after a failed audit")

    monkeypatch.setattr(inventory_mod, "_inventory_window_stream", _boom, raising=False)
    with pytest.raises(ProcessInventoryBacktestError) as excinfo:
        inventory_mod.evaluate_process_inventory_backtest(
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-02-01", tz="UTC"),
            data_root=str(tmp_path),
        )
    assert excinfo.value.report.error_code == "DATA_INTEGRITY"
    assert "AAAUSDT" in excinfo.value.report.error_message
    assert "settlement registry incomplete" in excinfo.value.report.error_message


def test_complete_registry_is_behaviour_neutral(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_3m(tmp_path, "AAAUSDT", pd.Timestamp("2025-01-01T00:00:00Z"), 200_000)
    _mock_panel_stage(monkeypatch, ["AAAUSDT"])
    ctx = _panel_context(
        str(tmp_path), pd.Timestamp("2025-06-01T00:00:00Z"), pd.Timestamp("2025-07-01T00:00:00Z"),
    )
    panel_stage.load_panel(ctx, StageTelemetry(log_run=False))
    assert ctx._terminal_report is None
    assert ctx.symbols == ["AAAUSDT"]
    assert ctx.aligned_symbols == ["AAAUSDT"]
