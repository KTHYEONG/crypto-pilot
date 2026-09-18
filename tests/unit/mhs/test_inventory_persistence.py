"""Invariant guards for compact inventory evidence with lossless details."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import uuid
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.backtests.contracts import ArtifactReference, RetentionPolicy, RunFinalization, RunRegistration
from src.backtests.registry import finalize_run, initialize_registry, register_run
from src.backtests.retention import plan_retention
from src.common.errors import DataIntegrityError
from src.mhs.contracts import MhsResourceMeasurement
from src.mhs.deploy_gate import DeployGateResult
from src.mhs.execution.contracts import (
    ExecutionDataGap,
    SimulatedInventoryLedgerResult,
    StrategyExecutionReplayResult,
)
from src.mhs.process import ProcessExecutionPolicy
import src.mhs.backtest.inventory as bt_inventory
import src.mhs.reporting.inventory as rep_inventory
from src.mhs.backtest.contracts import (
    ProcessBacktestReport,
    ProcessInventoryReport,
    ProcessPath,
)
from src.mhs.reporting.inventory import PROCESS_INVENTORY_CERTIFICATION_LEVEL

from src.mhs.reporting.inventory import export_inventory_json, persist_inventory_evidence
from src.mhs.resources import ProcessTreeMemoryStats


def _targets(n_days: int = 3, symbols: tuple[str, ...] = ("AUSDT", "BUSDT")) -> pd.DataFrame:
    index = pd.date_range("2022-01-01", periods=n_days, freq="24h", tz="UTC")
    weights = [[0.5 if j == 0 else -0.5 if j == 1 else 0.0 for j in range(len(symbols))] for _ in range(n_days)]
    return pd.DataFrame(weights, index=index, columns=list(symbols), dtype="float64")


def _path(targets: pd.DataFrame) -> ProcessPath:
    hourly = pd.date_range(targets.index[0], targets.index[-1] + pd.Timedelta(hours=23), freq="1h", tz="UTC")
    return ProcessPath(
        one_way_bps=8.0,
        daily_returns=pd.Series(0.01, index=targets.index),
        unit_daily_returns=pd.Series(0.01, index=targets.index),
        exposure=pd.Series(1.0, index=targets.index),
        refits=(),
        leverage_cap=2.0,
        execution_policy=ProcessExecutionPolicy(None),
        unit_target_weights=targets,
        target_weights=targets,
        turnover_1h=pd.Series(0.01, index=hourly),
    )


def _proxy(targets: pd.DataFrame) -> ProcessBacktestReport:
    path = _path(targets)
    return ProcessBacktestReport(
        start=targets.index[0],
        end=targets.index[-1],
        certification_level=PROCESS_INVENTORY_CERTIFICATION_LEVEL,
        n_candidates=len(targets.columns),
        base=path,
        stress=path,
        gate=DeployGateResult(go=False, reason_codes=("X",), metrics={"n_folds": 1.0}),
    )


def _equity_grid(periods: int = 1500) -> pd.DatetimeIndex:
    return pd.date_range("2022-01-01", periods=periods, freq="3min", tz="UTC")


def _gap(
    code: str = "MISSING_DECISION_MARK",
    symbol: str = "AUSDT",
    timestamp: pd.Timestamp | None = None,
    decision_time: pd.Timestamp | None = None,
    signal_time: pd.Timestamp | None = None,
    execution_bound: str = "OHLCV_IMMEDIATE_TAKER",
) -> ExecutionDataGap:
    stamp = timestamp if timestamp is not None else pd.Timestamp("2022-01-02T00:03:00", tz="UTC")
    return ExecutionDataGap(
        code=cast(Any, code),
        symbol=symbol,
        timestamp=stamp,
        decision_time=decision_time,
        signal_time=signal_time,
        execution_bound=execution_bound,
    )


def _ledger(
    equity: pd.Series, *, valid: bool = True, gaps: tuple[ExecutionDataGap, ...] = ()
) -> SimulatedInventoryLedgerResult:
    zeros = pd.Series(0.0, index=equity.index, dtype="float64")
    return SimulatedInventoryLedgerResult(
        equity=equity,
        net_returns=equity.pct_change().dropna(),
        simulated_units=None,
        mark_to_market_pnl=zeros,
        funding_charge=zeros,
        fee_charge=zeros,
        fill_turnover=zeros,
        fill_source="OHLCV_IMMEDIATE_TAKER",
        mark_source="MARK_PRICE",
        primary_valid=valid,
        invalid_reasons=() if valid else ("MISSING_DATA",),
        data_gaps=gaps,
    )


def _fills(grid: pd.DatetimeIndex, n_fills: int = 4) -> pd.DataFrame:
    stamps = grid[:n_fills]
    return pd.DataFrame(
        {
            "timestamp": list(stamps),
            "symbol": ["AUSDT", "BUSDT", "AUSDT", "BUSDT"][:n_fills],
            "quantity_delta": [1.0, -1.0, 0.5, 0.25][:n_fills],
            "fill_price": [100.0] * n_fills,
            "fee_bps": [8.0] * n_fills,
            "reason": ["immediate_taker"] * n_fills,
            "pre_trade_equity": [1.0] * n_fills,
        }
    )


def _result(
    equity: pd.Series, *, valid: bool = True, gaps: tuple[ExecutionDataGap, ...] = (),
    n_fills: int = 4,
) -> StrategyExecutionReplayResult:
    fills = _fills(pd.DatetimeIndex(equity.index), n_fills) if n_fills else pd.DataFrame(
        {
            "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
            "symbol": pd.Series(dtype="object"),
            "quantity_delta": pd.Series(dtype="float64"),
            "fill_price": pd.Series(dtype="float64"),
            "fee_bps": pd.Series(dtype="float64"),
            "reason": pd.Series(dtype="object"),
            "pre_trade_equity": pd.Series(dtype="float64"),
        }
    )
    return StrategyExecutionReplayResult(
        simulated_fills=fills,
        ledger=_ledger(equity, valid=valid, gaps=gaps),
        simulated_units=pd.DataFrame(columns=["AUSDT"]),
        simulated_notional_weights=pd.DataFrame(columns=["AUSDT"]),
        fill_source="OHLCV_IMMEDIATE_TAKER",
        mark_source="MARK_PRICE",
        submit_times=pd.Series(dtype="float64"),
        fill_times=pd.Series(dtype="float64"),
        fill_count=n_fills,
        unfilled_count=1,
        fallback_count=0,
        all_intent_shortfall_bps=0.0,
        forced_exit_count=0,
        forced_exit_notional=0.0,
        termination_counts={"exhausted": 2},
        unsupported_assumptions=(),
        elapsed_seconds=0.0,
    )


def _report(
    targets: pd.DataFrame | None = None, *,
    base_gaps: tuple[ExecutionDataGap, ...] = (),
    stress_gaps: tuple[ExecutionDataGap, ...] = (),
    stress_valid: bool = True,
) -> ProcessInventoryReport:
    grid = _equity_grid()
    levels = 1.0 + 0.0001 * pd.Series(range(len(grid)), index=grid, dtype="float64")
    proxy_targets = targets if targets is not None else _targets()
    return ProcessInventoryReport(
        proxy=_proxy(proxy_targets),
        base=_result(levels, gaps=base_gaps),
        stress=_result(levels * 0.999999, valid=stress_valid, gaps=stress_gaps),
        gate=DeployGateResult(go=False, reason_codes=("X",), metrics={"n_folds": 1.0}),
        resource_measurements=(MhsResourceMeasurement(stage="replay", elapsed_ms=3, rss_bytes=8),),
        memory_stats=ProcessTreeMemoryStats(
            tree_pss_peak_bytes=10, tree_uss_peak_bytes=9, min_system_available_bytes=7,
            max_concurrent_procs=2, samples_taken=1,
        ),
    )


def _persist_ok(tmp_path: Path, report: ProcessInventoryReport, name: str = "summary.json") -> tuple[Path, str, Path]:
    root = tmp_path / "evidence"
    out = tmp_path / name
    summary_path, identity = persist_inventory_evidence(report, out, evidence_root=root)
    return summary_path, identity, root


def _manifest_bundle(root: Path, identity: str) -> tuple[Path, dict[str, Any]]:
    bundle = root / identity
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    return bundle, manifest


def test_summary_matches_legacy_financials(tmp_path: Path) -> None:

    report = _report()
    summary_path, identity, _root = _persist_ok(tmp_path, report)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["schema_version"] == 1
    assert summary["status"] == "completed"
    assert summary["execution_timeframe"] == "3m"
    assert summary["evidence_id"] == identity
    assert summary["evidence"]["manifest"] == "manifest.json"
    for tier_key in ("base", "stress"):
        result = report.base if tier_key == "base" else report.stress
        legacy = rep_inventory._inventory_result_payload(result)
        for field in (
            "cagr", "max_drawdown", "annualized_turnover", "total_fees", "total_funding",
            "total_fills", "passive_fills", "unfilled_count", "fallback_count",
            "forced_exit_count", "forced_exit_notional", "termination_counts",
        ):
            assert summary[tier_key][field] == legacy[field]
        for flag in ("primary_valid", "terminal_certified", "invalid_reasons", "open_inventory", "unpriced_terminal_symbols"):
            assert summary[tier_key]["terminal"][flag] == legacy["terminal"][flag]
        assert summary[tier_key]["daily_returns"]["evidence_id"] == identity
        assert summary[tier_key]["daily_returns"]["count"] == len(bt_inventory._inventory_daily_returns(result))
        assert summary[tier_key]["terminal"]["data_gaps"]["evidence_id"] == identity
    assert summary["gate"] == {"go": False, "reason_codes": ["X"], "metrics": {"n_folds": 1.0}}
    assert summary["proxy"]["scope"] == "hourly_proxy_comparison"
    assert summary["certification_level"] == PROCESS_INVENTORY_CERTIFICATION_LEVEL


def test_gaps_round_trip_with_nanosecond_precision(tmp_path: Path) -> None:
    gaps = (
        _gap(timestamp=pd.Timestamp("2022-01-02T00:03:00.123456789", tz="UTC"), decision_time=None, signal_time=None),
        _gap(
            code="MISSING_HELD_FUNDING", symbol="BUSDT",
            timestamp=pd.Timestamp("2022-01-02T00:06:00.000000001", tz="UTC"),
            decision_time=pd.Timestamp("2022-01-01T23:00:00.5", tz="UTC"),
            signal_time=pd.Timestamp("2022-01-02T00:00:00", tz="UTC"),
            execution_bound="OHLCV_STRICT_PROXY",
        ),
    )
    report = _report(base_gaps=gaps)
    _summary_path, identity, root = _persist_ok(tmp_path, report)
    table = pq.read_table(root / identity / "base_gaps.parquet")
    assert table.num_rows == 2
    assert str(table.schema.field("timestamp").type) == "timestamp[ns, tz=UTC]"
    frame = table.to_pandas().sort_values("ordinal", kind="stable")
    assert frame["timestamp"].tolist() == [gaps[0].timestamp, gaps[1].timestamp]
    assert frame["decision_time"].isna().tolist() == [True, False]
    assert frame["signal_time"].isna().tolist() == [True, False]
    assert frame["decision_time"].iloc[1] == gaps[1].decision_time
    assert frame["execution_bound"].tolist() == ["OHLCV_IMMEDIATE_TAKER", "OHLCV_STRICT_PROXY"]


def test_duplicate_gaps_keep_count_and_order(tmp_path: Path) -> None:
    gap = _gap()
    report = _report(base_gaps=(gap, gap, _gap(symbol="BUSDT")))
    _summary_path, identity, root = _persist_ok(tmp_path, report)
    frame = pq.read_table(root / identity / "base_gaps.parquet").to_pandas().sort_values("ordinal", kind="stable")
    assert len(frame) == 3
    assert frame["ordinal"].tolist() == [0, 1, 2]
    assert frame["symbol"].tolist() == ["AUSDT", "AUSDT", "BUSDT"]


def test_tiers_keep_independent_values_and_flags(tmp_path: Path) -> None:
    report = _report(stress_valid=False, stress_gaps=(_gap(symbol="BUSDT"),))
    summary_path, identity, root = _persist_ok(tmp_path, report)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["base"]["terminal"]["primary_valid"] is True
    assert summary["stress"]["terminal"]["primary_valid"] is False
    assert summary["stress"]["terminal"]["invalid_reasons"] == ["MISSING_DATA"]
    assert summary["base"]["cagr"] != summary["stress"]["cagr"]
    base_daily = pq.read_table(root / identity / "base_daily.parquet").column("daily_return").to_pylist()
    stress_daily = pq.read_table(root / identity / "stress_daily.parquet").column("daily_return").to_pylist()
    assert base_daily != stress_daily


def test_daily_returns_round_trip_float64(tmp_path: Path) -> None:

    report = _report()
    _summary_path, identity, root = _persist_ok(tmp_path, report)
    expected = bt_inventory._inventory_daily_returns(report.base)
    table = pq.read_table(root / identity / "base_daily.parquet")
    assert str(table.schema.field("daily_return").type) == "double"
    frame = table.to_pandas()
    assert frame["timestamp"].tolist() == list(expected.index)
    assert frame["daily_return"].dtype == "float64"
    assert frame["daily_return"].tolist() == list(expected.to_numpy(dtype="float64"))


def test_bundle_preserves_fills_ledger_and_termination(tmp_path: Path) -> None:
    report = _report(base_gaps=(_gap(code="MISSING_ACTIVE_FUNDING"),))
    _summary_path, identity, root = _persist_ok(tmp_path, report)
    bundle = root / identity
    restored_fills = pq.read_table(bundle / "base_fills.parquet").to_pandas()
    pd.testing.assert_frame_equal(restored_fills, report.base.simulated_fills)
    for field in ("equity", "net_returns", "mark_to_market_pnl", "funding_charge", "fee_charge", "fill_turnover"):
        expected = getattr(report.base.ledger, field)
        frame = pq.read_table(bundle / f"base_ledger_{field}.parquet").to_pandas().set_index("timestamp")
        assert frame.index.tolist() == list(expected.index)
        assert frame.columns.tolist() == [field]
        assert str(frame.dtypes.iloc[0]) == str(expected.dtype)
        assert frame.iloc[:, 0].tolist() == list(expected.to_numpy())
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["base"]["termination_counts"] == {"exhausted": 2}
    assert summary["base"]["terminal"]["invalid_reasons"] == []
    assert set(summary["base"]["terminal"]["open_inventory"]) == {"AUSDT", "BUSDT"}


def test_repeated_persist_shares_evidence_identity(tmp_path: Path) -> None:
    report = _report(base_gaps=(_gap(),))
    first, identity, root = _persist_ok(tmp_path, report, name="first.json")
    second, rerun_identity, _rerun_root = _persist_ok(tmp_path, report, name="second.json")
    assert rerun_identity == identity
    before = {path.name: path.read_bytes() for path in sorted((root / identity).iterdir())}
    third, _third_identity, _third_root = _persist_ok(tmp_path, report, name="third.json")
    after = {path.name: path.read_bytes() for path in sorted((root / identity).iterdir())}
    assert before == after
    assert json.loads(first.read_text(encoding="utf-8"))["evidence_id"] == identity
    assert json.loads(second.read_text(encoding="utf-8"))["evidence_id"] == identity
    assert json.loads(third.read_text(encoding="utf-8"))["evidence_id"] == identity


def test_changed_bound_yields_different_identity(tmp_path: Path) -> None:
    _summary_path, identity, _root = _persist_ok(tmp_path, _report(base_gaps=(_gap(),)))
    altered = _report(base_gaps=(_gap(execution_bound="OHLCV_STRICT_PROXY"),))
    _other_path, other_identity, _other_root = _persist_ok(tmp_path, altered, name="other.json")
    assert other_identity != identity


def test_changed_row_order_yields_different_identity(tmp_path: Path) -> None:
    first_gap, second_gap = _gap(symbol="AUSDT"), _gap(symbol="BUSDT")
    _summary_path, identity, _root = _persist_ok(tmp_path, _report(base_gaps=(first_gap, second_gap)))
    _other_path, other_identity, _other_root = _persist_ok(tmp_path, _report(base_gaps=(second_gap, first_gap)), name="other.json")
    assert other_identity != identity


def test_interrupted_persist_leaves_no_completed_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.mhs.reporting.inventory as inventory

    report = _report()
    _summary_path, identity, root = _persist_ok(tmp_path, report)
    manifest_bytes = (root / identity / "manifest.json").read_bytes()

    def _boom(_path: Path, _table: Any) -> None:
        raise OSError("injected staging failure")

    monkeypatch.setattr(inventory.pq, "ParquetWriter", _boom_raiser(_boom))
    with pytest.raises(OSError, match=r".+"):
        persist_inventory_evidence(report, tmp_path / "broken.json", evidence_root=root)
    assert not (tmp_path / "broken.json").exists()
    assert (root / identity / "manifest.json").read_bytes() == manifest_bytes
    assert [child for child in root.iterdir() if child.name.startswith(".staging_")] == []


def _boom_raiser(handler: Any) -> Any:
    class _Writer:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            handler(None, None)

        def __enter__(self) -> _Writer:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

    return _Writer


def test_corrupt_bundle_reuse_fails_closed(tmp_path: Path) -> None:
    report = _report(base_gaps=(_gap(),))
    _summary_path, identity, root = _persist_ok(tmp_path, report)
    bundle, _manifest = _manifest_bundle(root, identity)
    target = bundle / "base_gaps.parquet"
    tampered = target.read_bytes() + b"\x00"
    target.write_bytes(tampered)
    with pytest.raises(DataIntegrityError, match=r".+"):
        persist_inventory_evidence(report, tmp_path / "second.json", evidence_root=root)
    assert target.read_bytes() == tampered
    assert not (tmp_path / "second.json").exists()


def test_export_restores_full_json_and_preserves_originals(tmp_path: Path) -> None:

    gaps = (
        _gap(),
        _gap(code="MISSING_HELD_FUNDING", symbol="BUSDT", decision_time=pd.Timestamp("2022-01-01T23:00:00", tz="UTC")),
    )
    report = _report(base_gaps=gaps)
    summary_path, identity, root = _persist_ok(tmp_path, report)
    bundle, _manifest = _manifest_bundle(root, identity)
    before_files = {path.name: path.read_bytes() for path in sorted(bundle.iterdir())}
    before_summary = summary_path.read_bytes()
    exported = export_inventory_json(summary_path, tmp_path / "full.json")
    payload = json.loads(exported.read_text(encoding="utf-8"))
    assert "schema_version" not in payload
    assert "evidence" not in payload
    assert "evidence_id" not in payload
    for tier_key in ("base", "stress"):
        result = report.base if tier_key == "base" else report.stress
        legacy = rep_inventory._inventory_result_payload(result)
        assert payload[tier_key]["daily_returns"] == legacy["daily_returns"]
        assert payload[tier_key]["terminal"]["data_gaps"] == legacy["terminal"]["data_gaps"]
        for field in ("cagr", "total_fees", "termination_counts"):
            assert payload[tier_key][field] == legacy[field]
    assert {path.name: path.read_bytes() for path in sorted(bundle.iterdir())} == before_files
    assert summary_path.read_bytes() == before_summary


def test_bounded_gap_conversion_limits_row_groups(tmp_path: Path) -> None:
    grid = _equity_grid(periods=300)
    stamps = pd.date_range("2022-01-01", periods=100_000, freq="1s", tz="UTC")
    gaps = tuple(
        _gap(
            code="MISSING_DECISION_MARK" if i % 2 == 0 else "MISSING_ACTIVE_FUNDING",
            symbol="AUSDT" if i % 3 else "BUSDT",
            timestamp=stamps[i],
            decision_time=None if i % 5 else stamps[i] - pd.Timedelta(hours=1),
            signal_time=None,
        )
        for i in range(100_000)
    )
    levels = pd.Series(1.0, index=grid, dtype="float64")
    report = ProcessInventoryReport(
        proxy=_proxy(_targets()),
        base=_result(levels, gaps=gaps, n_fills=0),
        stress=_result(levels, gaps=(), n_fills=0),
        gate=DeployGateResult(go=False, reason_codes=("X",), metrics={"n_folds": 1.0}),
        resource_measurements=(),
        memory_stats=ProcessTreeMemoryStats(
            tree_pss_peak_bytes=1, tree_uss_peak_bytes=1, min_system_available_bytes=1,
            max_concurrent_procs=1, samples_taken=0,
        ),
    )
    _summary_path, identity, root = _persist_ok(tmp_path, report)
    metadata = pq.ParquetFile(root / identity / "base_gaps.parquet").metadata
    assert metadata.num_rows == 100_000
    assert metadata.num_row_groups == 10
    assert max(metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)) <= 10_000


def test_managed_persist_records_publication_lease(tmp_path: Path) -> None:
    db = tmp_path / "registry.sqlite3"
    initialize_registry(db)
    run_id = uuid.uuid4().hex
    register_run(
        db,
        RunRegistration(
            run_id=run_id, strategy_id="strat-a", registered_at="2026-01-01T00:00:00+00:00",
            request={"window": "3m"}, managed_directory=None,
        ),
    )
    root = tmp_path / "evidence"
    _summary_path, identity = persist_inventory_evidence(
        _report(), tmp_path / "summary.json", evidence_root=root, registry_path=db, run_id=run_id
    )
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute(
            "SELECT role, managed, evidence_id, retained FROM artifacts WHERE run_id = ?", (run_id,)
        ).fetchall()
    finally:
        conn.close()
    assert rows == [("publication_lease", 1, identity, 1)]


def test_leased_bundle_survives_collection_before_finalization(tmp_path: Path) -> None:
    db = tmp_path / "registry.sqlite3"
    initialize_registry(db)
    run_id = uuid.uuid4().hex
    register_run(
        db,
        RunRegistration(
            run_id=run_id, strategy_id="strat-a", registered_at="2026-01-01T00:00:00+00:00",
            request={"window": "3m"}, managed_directory=None,
        ),
    )
    root = tmp_path / "evidence"
    _summary_path, identity = persist_inventory_evidence(
        _report(), tmp_path / "summary.json", evidence_root=root, registry_path=db, run_id=run_id
    )
    plan = plan_retention(db, root, RetentionPolicy(max_detail_bytes=1))
    assert plan.evidence_ids == ()
    assert (root / identity).is_dir()


def test_lease_converts_to_artifact_reference_at_finalization(tmp_path: Path) -> None:
    db = tmp_path / "registry.sqlite3"
    initialize_registry(db)
    run_id = uuid.uuid4().hex
    register_run(
        db,
        RunRegistration(
            run_id=run_id, strategy_id="strat-a", registered_at="2026-01-01T00:00:00+00:00",
            request={"window": "3m"}, managed_directory=None,
        ),
    )
    root = tmp_path / "evidence"
    _summary_path, identity = persist_inventory_evidence(
        _report(), tmp_path / "summary.json", evidence_root=root, registry_path=db, run_id=run_id
    )
    finalize_run(
        db,
        RunFinalization(
            run_id=run_id, status="completed", finalized_at="2026-01-01T01:00:00+00:00",
            primary_valid=True, terminal_certified=True, outcome={},
        ),
        (
            ArtifactReference(
                run_id=run_id, role="detail", path=Path(f"/evidence/{identity}/detail.bin"),
                sha256="ab" * 32, byte_count=10, managed=True, evidence_id=identity,
            ),
        ),
    )
    conn = sqlite3.connect(str(db))
    try:
        roles = sorted(
            row[0] for row in conn.execute("SELECT role FROM artifacts WHERE run_id = ?", (run_id,)).fetchall()
        )
    finally:
        conn.close()
    assert roles == ["detail"]


def test_managed_persist_rejects_unknown_run(tmp_path: Path) -> None:
    db = tmp_path / "registry.sqlite3"
    initialize_registry(db)
    with pytest.raises(sqlite3.Error):
        persist_inventory_evidence(
            _report(), tmp_path / "summary.json", evidence_root=tmp_path / "evidence",
            registry_path=db, run_id=uuid.uuid4().hex,
        )


def test_managed_persist_rejects_bad_run_id(tmp_path: Path) -> None:
    db = tmp_path / "registry.sqlite3"
    initialize_registry(db)
    with pytest.raises(ValueError, match=r".+"):
        persist_inventory_evidence(
            _report(), tmp_path / "summary.json", evidence_root=tmp_path / "evidence",
            registry_path=db, run_id="not-a-run",
        )


def test_persist_rejects_split_ownership_context(tmp_path: Path) -> None:
    db = tmp_path / "registry.sqlite3"
    initialize_registry(db)
    with pytest.raises(ValueError, match=r".+"):
        persist_inventory_evidence(
            _report(), tmp_path / "summary.json", evidence_root=tmp_path / "evidence", registry_path=db,
        )
    with pytest.raises(ValueError, match=r".+"):
        persist_inventory_evidence(
            _report(), tmp_path / "other.json", evidence_root=tmp_path / "evidence", run_id=uuid.uuid4().hex,
        )


def test_persist_rejects_occupied_or_unsupported_destination(tmp_path: Path) -> None:
    report = _report()
    occupied = tmp_path / "summary.json"
    occupied.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match=r".+"):
        persist_inventory_evidence(report, occupied, evidence_root=tmp_path / "evidence")
    with pytest.raises(ValueError, match=r".+"):
        persist_inventory_evidence(report, tmp_path / "summary.csv", evidence_root=tmp_path / "evidence")


def test_persist_rejects_symlinked_root(tmp_path: Path) -> None:
    target = tmp_path / "real"
    target.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match=r".+"):
        persist_inventory_evidence(_report(), tmp_path / "summary.json", evidence_root=link)


def test_publish_rejects_occupied_evidence_slot(tmp_path: Path) -> None:
    report = _report()
    root = tmp_path / "evidence"
    _summary_path, identity, _bundle_root = _persist_ok(tmp_path, report)
    shutil.rmtree(root / identity)
    (root / identity).symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(DataIntegrityError, match=r".+"):
        persist_inventory_evidence(report, tmp_path / "second.json", evidence_root=root)


def test_export_rejects_occupied_destination(tmp_path: Path) -> None:
    summary_path, _identity, _root = _persist_ok(tmp_path, _report())
    occupied = tmp_path / "full.json"
    occupied.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match=r".+"):
        export_inventory_json(summary_path, occupied)


def test_export_rejects_missing_summary(tmp_path: Path) -> None:
    with pytest.raises(DataIntegrityError, match=r".+"):
        export_inventory_json(tmp_path / "absent.json", tmp_path / "full.json")


def test_export_rejects_legacy_summary_without_reference(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    with pytest.raises(DataIntegrityError, match=r".+"):
        export_inventory_json(legacy, tmp_path / "full.json")


def test_export_rejects_missing_manifest(tmp_path: Path) -> None:
    summary_path, identity, root = _persist_ok(tmp_path, _report())
    (root / identity / "manifest.json").unlink()
    with pytest.raises(DataIntegrityError, match=r".+"):
        export_inventory_json(summary_path, tmp_path / "full.json")


def test_export_rejects_corrupt_manifest(tmp_path: Path) -> None:
    summary_path, identity, root = _persist_ok(tmp_path, _report())
    (root / identity / "manifest.json").write_text("not json", encoding="utf-8")
    with pytest.raises(DataIntegrityError, match=r".+"):
        export_inventory_json(summary_path, tmp_path / "full.json")


def test_export_rejects_manifest_identity_mismatch(tmp_path: Path) -> None:
    summary_path, identity, root = _persist_ok(tmp_path, _report())
    manifest_path = root / identity / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["evidence_id"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    with pytest.raises(DataIntegrityError, match=r".+"):
        export_inventory_json(summary_path, tmp_path / "full.json")


def test_export_rejects_manifest_without_files(tmp_path: Path) -> None:
    summary_path, identity, root = _persist_ok(tmp_path, _report())
    manifest_path = root / identity / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = "nope"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    with pytest.raises(DataIntegrityError, match=r".+"):
        export_inventory_json(summary_path, tmp_path / "full.json")


def test_export_rejects_missing_detail_file(tmp_path: Path) -> None:
    summary_path, identity, root = _persist_ok(tmp_path, _report())
    (root / identity / "base_gaps.parquet").unlink()
    with pytest.raises(DataIntegrityError, match=r".+"):
        export_inventory_json(summary_path, tmp_path / "full.json")


def test_export_rejects_tampered_detail_file(tmp_path: Path) -> None:
    summary_path, identity, root = _persist_ok(tmp_path, _report())
    target = root / identity / "base_daily.parquet"
    target.write_bytes(target.read_bytes() + b"\x00")
    with pytest.raises(DataIntegrityError, match=r".+"):
        export_inventory_json(summary_path, tmp_path / "full.json")


def test_export_rejects_manifest_covering_tampered_file(tmp_path: Path) -> None:
    summary_path, identity, root = _persist_ok(tmp_path, _report())
    bundle, _manifest = _manifest_bundle(root, identity)
    target = bundle / "base_daily.parquet"
    target.write_bytes(target.read_bytes() + b"\x00")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    for entry in manifest["files"]:
        if entry["name"] == "base_daily.parquet":
            entry["sha256"] = digest
            entry["size"] = target.stat().st_size
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    with pytest.raises(DataIntegrityError, match=r".+"):
        export_inventory_json(summary_path, tmp_path / "full.json")


def test_atomic_write_failure_propagates_without_partial_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.mhs.reporting.inventory as inventory

    summary_path, _identity, _root = _persist_ok(tmp_path, _report())
    real_dumps = inventory.json.dumps

    def _boom(payload: Any, *args: Any, **kwargs: Any) -> str:
        if isinstance(payload, dict) and "status" in payload:
            raise OSError("injected serialization failure")
        return real_dumps(payload, *args, **kwargs)

    monkeypatch.setattr(inventory.json, "dumps", _boom)
    with pytest.raises(OSError, match=r".+"):
        export_inventory_json(summary_path, tmp_path / "full.json")
    assert not (tmp_path / "full.json").exists()
    assert [child for child in tmp_path.iterdir() if child.suffix == ".tmp"] == []


def test_restore_daily_failure_fails_closed_without_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.mhs.reporting.inventory as inventory

    summary_path, _identity, _root = _persist_ok(tmp_path, _report())
    real_read = inventory.pq.read_table

    def _boom(path: Any, *args: Any, **kwargs: Any) -> Any:
        if Path(path).name.endswith("_daily.parquet"):
            raise OSError("injected restore failure")
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(inventory.pq, "read_table", _boom)
    with pytest.raises(DataIntegrityError, match=r".+"):
        export_inventory_json(summary_path, tmp_path / "full.json")
    assert not (tmp_path / "full.json").exists()


def test_restore_gaps_failure_fails_closed_without_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.mhs.reporting.inventory as inventory

    summary_path, _identity, _root = _persist_ok(tmp_path, _report())
    real_read = inventory.pq.read_table

    def _boom(path: Any, *args: Any, **kwargs: Any) -> Any:
        if Path(path).name.endswith("_gaps.parquet"):
            raise OSError("injected restore failure")
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(inventory.pq, "read_table", _boom)
    with pytest.raises(DataIntegrityError, match=r".+"):
        export_inventory_json(summary_path, tmp_path / "full.json")
    assert not (tmp_path / "full.json").exists()


def test_export_rejects_manifest_with_bad_entry(tmp_path: Path) -> None:
    summary_path, identity, root = _persist_ok(tmp_path, _report())
    manifest_path = root / identity / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = ["nope"]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    with pytest.raises(DataIntegrityError, match=r".+"):
        export_inventory_json(summary_path, tmp_path / "full.json")


def test_export_rejects_non_mapping_summary(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(DataIntegrityError, match=r".+"):
        export_inventory_json(summary_path, tmp_path / "full.json")


def test_export_rejects_non_string_bundle_path(tmp_path: Path) -> None:
    summary_path, _identity, _root = _persist_ok(tmp_path, _report())
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["evidence"]["bundle_path"] = 123
    summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2), encoding="utf-8")
    with pytest.raises(DataIntegrityError, match=r".+"):
        export_inventory_json(summary_path, tmp_path / "full.json")


def test_wrapper_uses_standalone_default_root(tmp_path: Path) -> None:

    report = _report()
    out = rep_inventory.persist_process_inventory_report(report, tmp_path / "inventory.json")
    assert out == tmp_path / "inventory.json"
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["evidence_id"]
    bundle = tmp_path / ".evidence" / payload["evidence_id"]
    assert (bundle / "manifest.json").is_file()
    assert payload["evidence"]["bundle_path"] == str(bundle)


def test_wrapper_rejects_occupied_destination_and_format_checks(tmp_path: Path) -> None:

    hourly_path = tmp_path / "hourly.json"
    hourly_path.write_text('{"hourly": true}', encoding="utf-8")
    report = _report()
    with pytest.raises(ValueError, match=r".+"):
        rep_inventory.persist_process_inventory_report(report, hourly_path)
    with pytest.raises(ValueError, match=r".+"):
        rep_inventory.persist_process_inventory_report(report, tmp_path / "inventory.csv")
    assert hourly_path.read_text(encoding="utf-8") == '{"hourly": true}'
