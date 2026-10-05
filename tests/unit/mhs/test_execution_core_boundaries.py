"""Execution core owners: re-export identity and owner behaviour guards."""

from __future__ import annotations

from tests.fixtures.mhs_requests import research_baseline
import ast
from pathlib import Path

import pandas as pd
import pytest

from src.mhs.types import ExecutionSpec

_REEXPORTS: tuple[tuple[str, str, str], ...] = (
    ("src.mhs.evaluation.windows", "src.mhs.execution.window_stream", "_iter_mhs_execution_windows"),
    ("src.mhs.evaluation.windows", "src.mhs.execution.window_stream", "_resolve_ns_vectorized"),
    ("src.mhs.evaluation.windows", "src.mhs.execution.window_stream", "_estimate_mhs_execution_allocation"),
    ("src.mhs.evaluation.windows", "src.mhs.execution.window_stream", "_minimum_mhs_execution_bars"),
    ("src.mhs.evaluation.windows", "src.mhs.execution.window_stream", "_materialize_execution_piece"),
    ("src.mhs.evaluation.windows", "src.mhs.execution.window_stream", "MhsExecutionWindow"),
    ("src.mhs.evaluation.integrity", "src.mhs.execution.integrity", "replay_ledger_certified"),
    ("src.mhs.evaluation.integrity", "src.mhs.execution.integrity", "ledger_terminal_only"),
    ("src.mhs.evaluation.integrity", "src.mhs.execution.integrity", "_funding_gap_terminal_symbols"),
    ("src.mhs.evaluation.integrity", "src.mhs.data_policy", "SOURCE_GAP_EXCLUDED_SYMBOLS"),
    ("src.mhs.evaluation.specs", "src.mhs.execution.specs", "_stress_cost_execution_spec"),
)


@pytest.mark.parametrize(("facade_module", "owner_module", "name"), _REEXPORTS)
def test_evaluation_facade_reexports_execution_owner_object(facade_module: str, owner_module: str, name: str) -> None:
    """The evaluation namespaces re-export the execution owners, not copies.

    Callers and monkeypatch seams use ``src.mhs.evaluation.*``; identity
    guarantees there is a single implementation, so behaviour is tested once on
    the owner and parity between "legacy" and "new" can never drift.
    """
    import importlib

    facade = importlib.import_module(facade_module)
    owner = importlib.import_module(owner_module)
    assert getattr(facade, name) is getattr(owner, name)


def _write_3m_ohlcv(root: Path, symbol: str, labels: pd.DatetimeIndex) -> None:
    ms = (labels - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)
    close = 100.0 + 0.01 * (labels.asi8 // 180_000_000_000)
    pd.DataFrame(
        {
            "timestamp": ms.to_numpy(dtype="int64"),
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "quote_vol": 1000.0,
        }
    ).to_parquet(root / f"{symbol}.parquet")


def _fixture(tmp_path: Path, *, days: int = 2):
    start = pd.Timestamp("2022-01-01", tz="UTC")
    decisions = pd.date_range(start, periods=days, freq="24h", tz="UTC")
    end = start + pd.Timedelta(days=days)
    grid = pd.date_range(start, end, freq="3min", tz="UTC")
    lake = tmp_path / "ohlcv" / "3m"
    lake.mkdir(parents=True, exist_ok=True)
    for sym in ("AUSDT", "BUSDT"):
        _write_3m_ohlcv(lake, sym, grid)
    funding = {s: pd.Series(0.0, index=grid) for s in ("AUSDT", "BUSDT")}
    targets = pd.DataFrame(0.0, index=decisions, columns=["AUSDT", "BUSDT"])
    targets.iloc[:, 0] = 0.05
    return start, end, decisions, funding, targets, ExecutionSpec()


def test_iter_live_required_symbols_present_in_every_window(tmp_path) -> None:
    """Live requirements: every window carries the required symbols."""
    from src.mhs.execution.window_stream import _iter_mhs_execution_windows

    start, end, decisions, funding, targets, spec = _fixture(tmp_path)
    signals = decisions + pd.Timedelta(hours=1)
    root = str(tmp_path / "ohlcv")
    windows = list(
        _iter_mhs_execution_windows(
            targets, signals, root, "3m", start, end, funding, spec,
            required_symbols=lambda: frozenset({"AUSDT"}),
        )
    )
    assert all("AUSDT" in w.symbols for w in windows)


def test_iter_keeps_half_open_fence(tmp_path) -> None:
    """Half open range: fence bars stay unpublished in the owner."""
    from src.mhs.execution.window_stream import _iter_mhs_execution_windows

    start, end, decisions, funding, targets, spec = _fixture(tmp_path)
    signals = decisions + pd.Timedelta(hours=1)
    root = str(tmp_path / "ohlcv")
    windows = list(_iter_mhs_execution_windows(targets, signals, root, "3m", start, end, funding, spec))
    assert all(end not in w.minute_grid for w in windows)
    assert all((w.bar_available_at <= end).all() for w in windows)


def test_certification_verdicts() -> None:
    """Certification verdicts on every replay class."""
    from src.mhs.execution.contracts import ExecutionDataGap
    from src.mhs.execution.integrity import ledger_terminal_only, replay_ledger_certified

    stamp = pd.Timestamp("2022-01-01", tz="UTC")

    class _Ledger:
        def __init__(self, valid, gaps):
            self.primary_valid = valid
            self.invalid_reasons = ()
            self.data_gaps = gaps

    class _Replay:
        def __init__(self, valid, gaps, fills):
            self.ledger = _Ledger(valid, gaps)
            self.simulated_fills = fills
            self.terminal_positions = ()

    fills = pd.DataFrame({"symbol": [], "timestamp": []})
    gap = ExecutionDataGap(code="UNKNOWN_TERMINATION", symbol="AUSDT", timestamp=stamp)
    assert replay_ledger_certified(_Replay(True, [], fills)) is True
    assert ledger_terminal_only([gap], fills) is True
    assert replay_ledger_certified(_Replay(False, [gap], fills)) is False
    assert replay_ledger_certified(object()) is False
    assert replay_ledger_certified(None) is False
    assert replay_ledger_certified(_Replay(False, [], fills)) is False


def test_later_fill_classification() -> None:
    """Later fill classification: recovery versus delist settlement."""
    from src.mhs.execution.contracts import ExecutionDataGap
    from src.mhs.execution.integrity import _funding_gap_terminal_symbols

    stamp = pd.Timestamp("2022-01-01", tz="UTC")
    later = pd.Timestamp("2022-01-02", tz="UTC")
    gaps = [ExecutionDataGap(code="MISSING_HELD_FUNDING", symbol="AUSDT", timestamp=stamp)]
    fills = pd.DataFrame({"symbol": ["AUSDT"], "timestamp": [later], "reason": ["fill"]})
    assert _funding_gap_terminal_symbols(gaps, fills) == frozenset()
    settlement = pd.DataFrame({"symbol": ["AUSDT"], "timestamp": [later], "reason": ["delist_settlement"]})
    assert _funding_gap_terminal_symbols(gaps, settlement) == frozenset({"AUSDT"})
    assert _funding_gap_terminal_symbols([], fills) == frozenset()


def test_exception_roster_values() -> None:
    """Exception roster preservation: the single registry holds the frozen set."""
    from src.mhs.data_policy import SOURCE_GAP_EXCLUDED_SYMBOLS

    assert set(SOURCE_GAP_EXCLUDED_SYMBOLS) == frozenset(
        {
            "AERGOUSDT", "CTKUSDT", "CVCUSDT", "MAVIAUSDT", "LITUSDT", "PUMPUSDT",
            "CVXUSDT", "SLPUSDT", "BNXUSDT", "AIAUSDT", "ICPUSDT", "BNTUSDT",
            "BTCSTUSDT", "BDXNUSDT", "LUNAUSDT", "MANAUSDT", "NEARUSDT",
        }
    )


def test_moved_defaults_match_legacy_values() -> None:
    """Defaults parity: production CLI defaults keep values; research baseline stays frozen."""
    from src.mhs import params as _params

    assert _params.CLI_GROWTH_ENVELOPE_DEFAULT == "growth_extreme_budgeted"
    assert _params.CLI_EXECUTION_UNIVERSE_SIZE_DEFAULT == 60
    assert research_baseline().execution_universe_size == 30


def test_iter_rejects_invalid_coverage(tmp_path) -> None:
    """Invalid coverage guards: misaligned signals and reversed ranges fail closed."""
    import pytest

    from src.common.errors import DataIntegrityError
    from src.mhs.execution.window_stream import _iter_mhs_execution_windows

    start, end, decisions, funding, targets, spec = _fixture(tmp_path)
    with pytest.raises(DataIntegrityError, match="align"):
        list(_iter_mhs_execution_windows(targets, decisions[:1], str(tmp_path / "ohlcv"), "3m", start, end, funding, spec))
    with pytest.raises(DataIntegrityError, match="precede"):
        list(_iter_mhs_execution_windows(targets, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"), "3m", end, start, funding, spec))


def test_iter_covers_non_last_bound_fallback_and_clamp(tmp_path) -> None:
    """Non-last bound fallback: off-grid timeouts reuse decision span and clamp to fence."""
    from src.mhs.execution.window_stream import _iter_mhs_execution_windows
    from src.mhs.types import ExecutionSpec

    start = pd.Timestamp("2022-01-01", tz="UTC")
    decisions = pd.DatetimeIndex([start, start + pd.Timedelta(days=40)])
    end = start + pd.Timedelta(minutes=30)
    grid = pd.date_range(start, end, freq="3min", tz="UTC")
    lake = tmp_path / "ohlcv" / "3m"
    lake.mkdir(parents=True, exist_ok=True)
    for sym in ("AUSDT", "BUSDT"):
        _write_3m_ohlcv(lake, sym, grid)
    funding = {s: pd.Series(0.0, index=grid) for s in ("AUSDT", "BUSDT")}
    targets = pd.DataFrame(0.0, index=decisions, columns=["AUSDT", "BUSDT"])
    occupied = pd.DataFrame(0.0, index=decisions, columns=["AUSDT", "BUSDT"])
    occupied.iloc[0, 0] = 0.05
    spec = ExecutionSpec(passive_timeout_minutes=7)
    windows = list(
        _iter_mhs_execution_windows(
            occupied, decisions + pd.Timedelta(hours=1), str(tmp_path / "ohlcv"), "3m", start, end, funding, spec
        )
    )
    assert len(windows) >= 1
    assert all(end not in w.minute_grid for w in windows)


def test_ops_migrate_wiring_stays_available(tmp_path) -> None:
    """Ops wiring: backtests-migrate leaf previews without mutating sources."""
    import argparse

    from src.cli.commands.ops import add_ops_commands

    history = tmp_path / "history"
    history.mkdir()
    (history / "active.jsonl").write_text('{"status": "COMPLETE"}\n', encoding="utf-8")
    parser = argparse.ArgumentParser()
    add_ops_commands(parser)
    args = parser.parse_args(
        ["backtests-migrate", "--registry-path", str(tmp_path / "registry.sqlite3"), "--history-directory", str(history)]
    )
    args.handler(args)
    assert not (tmp_path / "registry.sqlite3").exists()
    import src.mhs.pipeline.stages.assemble as _assemble

    assert callable(_assemble.assemble_report)


def test_no_recovery_is_not_proof() -> None:
    """No recovery is not proof: held gaps without later fills never certify."""
    from src.mhs.execution.contracts import ExecutionDataGap
    from src.mhs.execution.integrity import ledger_terminal_only, replay_ledger_certified

    stamp = pd.Timestamp("2022-01-01", tz="UTC")

    class _Ledger:
        def __init__(self, valid, reasons, gaps):
            self.primary_valid = valid
            self.invalid_reasons = reasons
            self.data_gaps = gaps

    class _Replay:
        def __init__(self, valid, reasons, gaps, fills, positions=()):
            self.ledger = _Ledger(valid, reasons, gaps)
            self.simulated_fills = fills
            self.terminal_positions = positions

    fills = pd.DataFrame({"symbol": [], "timestamp": []})
    gaps = [
        ExecutionDataGap(code="MISSING_HELD_MARK", symbol="AUSDT", timestamp=stamp),
        ExecutionDataGap(code="MISSING_HELD_FUNDING", symbol="BUSDT", timestamp=stamp),
    ]
    assert ledger_terminal_only(gaps, fills) is True
    assert replay_ledger_certified(_Replay(False, ("MISSING_DATA",), gaps, fills)) is False


def test_forged_validity_cannot_rescue_gaps() -> None:
    """Forged validity cannot rescue gaps: flag/reason/position conflicts fail closed."""
    from src.mhs.execution.contracts import ExecutionDataGap, TerminalPositionEvidence
    from src.mhs.execution.integrity import replay_ledger_certified

    stamp = pd.Timestamp("2022-01-01", tz="UTC")
    fills = pd.DataFrame({"symbol": [], "timestamp": []})

    class _Ledger:
        def __init__(self, valid, reasons, gaps):
            self.primary_valid = valid
            self.invalid_reasons = reasons
            self.data_gaps = gaps

    class _Replay:
        def __init__(self, valid, reasons, gaps, positions=()):
            self.ledger = _Ledger(valid, reasons, gaps)
            self.simulated_fills = fills
            self.terminal_positions = positions

    gap = ExecutionDataGap(code="MISSING_HELD_FUNDING", symbol="AUSDT", timestamp=stamp)
    assert replay_ledger_certified(_Replay(True, ("MISSING_DATA",), [], ())) is False
    assert replay_ledger_certified(_Replay(True, (), [gap], ())) is False
    unresolved = TerminalPositionEvidence(
        symbol="AUSDT", quantity=1.0, cutoff=stamp, status="unresolved",
        mark=None, mark_available_at=None, funding_complete=False,
        reason_codes=("STALE_MARK", "MISSING_HELD_FUNDING"),
    )
    assert replay_ledger_certified(_Replay(True, (), [], (unresolved,))) is False


def test_certification_terminal_evidence_classes() -> None:
    """Canonical terminal evidence classes certify exactly the priced, settled and gap cases."""
    from src.mhs.execution.contracts import ExecutionDataGap, TerminalPositionEvidence
    from src.mhs.execution.integrity import replay_ledger_certified

    stamp = pd.Timestamp("2022-01-01", tz="UTC")
    fills = pd.DataFrame({"symbol": [], "timestamp": []})

    class _Ledger:
        def __init__(self, valid, reasons, gaps):
            self.primary_valid = valid
            self.invalid_reasons = reasons
            self.data_gaps = gaps

    class _Replay:
        def __init__(self, valid, reasons, gaps, positions=()):
            self.ledger = _Ledger(valid, reasons, gaps)
            self.simulated_fills = fills
            self.terminal_positions = positions

    priced_open = TerminalPositionEvidence(
        symbol="AUSDT", quantity=1.0, cutoff=stamp, status="open_marked",
        mark=100.0, mark_available_at=stamp, funding_complete=True,
        reason_codes=("FRESH_MARK", "FUNDING_COMPLETE"),
    )
    priced = _Replay(True, (), [], (priced_open,))
    unresolved = _Replay(
        False, ("MISSING_DATA",),
        [ExecutionDataGap(code="MISSING_HELD_MARK", symbol="AUSDT", timestamp=stamp)],
        (),
    )
    settled = _Replay(True, (), [], ())
    assert replay_ledger_certified(priced) is True
    assert replay_ledger_certified(unresolved) is False
    assert replay_ledger_certified(settled) is True


def test_recovery_cannot_erase_unknown_economics() -> None:
    """Recovery cannot erase unknown economics: later fills never backfill financing."""
    from src.mhs.execution.contracts import ExecutionDataGap, TerminalPositionEvidence
    from src.mhs.execution.integrity import replay_ledger_certified

    stamp = pd.Timestamp("2022-01-01", tz="UTC")
    later = pd.Timestamp("2022-01-02", tz="UTC")

    class _Ledger:
        def __init__(self, valid, reasons, gaps):
            self.primary_valid = valid
            self.invalid_reasons = reasons
            self.data_gaps = gaps

    class _Replay:
        def __init__(self, valid, reasons, gaps, fills, positions=()):
            self.ledger = _Ledger(valid, reasons, gaps)
            self.simulated_fills = fills
            self.terminal_positions = positions

    gap = ExecutionDataGap(code="MISSING_HELD_FUNDING", symbol="AUSDT", timestamp=stamp)
    resumed = pd.DataFrame({"symbol": ["AUSDT"], "timestamp": [later], "reason": ["timeout_taker"]})
    settlement = pd.DataFrame({"symbol": ["AUSDT"], "timestamp": [later], "reason": ["delist_settlement"]})
    settled = TerminalPositionEvidence(
        symbol="AUSDT", quantity=0.0, cutoff=later, status="settled",
        mark=100.0, mark_available_at=later, funding_complete=False,
        reason_codes=("SETTLEMENT_EVENT",),
    )
    assert replay_ledger_certified(_Replay(True, (), [gap], resumed, ())) is False
    assert replay_ledger_certified(_Replay(True, (), [gap], settlement, (settled,))) is False


def test_core_imports_avoid_research_facade() -> None:
    """Core import boundary: top-level helper imports stay out of research modules."""
    forbidden = ("src.mhs.evaluation", "src.mhs.pipeline.config", "src.mhs.report", "src.mhs.reporting")
    for module in ("src/mhs/execution/window_stream.py", "src/mhs/execution/integrity.py", "src/mhs/execution/specs.py"):
        tree = ast.parse((Path(__file__).resolve().parents[3] / module).read_text(encoding="utf-8"))
        top_imports = [
            node.names[0].name if isinstance(node, ast.Import) else (node.module or "")
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        assert not any(name.startswith(bad) for name in top_imports for bad in forbidden), module
