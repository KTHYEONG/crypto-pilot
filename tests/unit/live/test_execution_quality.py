# ruff: noqa
"""SCENARIO Live Execution Quality Recording contract tests."""

from __future__ import annotations

import dataclasses
import inspect
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from src.live.execution_quality import (
    EXECUTION_QUALITY_MIN_EVIDENCE_DAYS,
    append_execution_quality,
    build_execution_quality_records,
    summarize_execution_quality,
)
from src.live.planner import OrderIntent
from src.live.executor import ExecutionOutcome
from src.live.audit import AUDIT_LOG_RETENTION_DAYS
from src.mhs.params import MEASURED_EXECUTION_COST_TIERS_BPS


def _intent(symbol: str, side: str, qty: str = "1.0", *, leg_index: int = 0, client_order_prefix: str = "20260101") -> OrderIntent:
    return OrderIntent(
        symbol=symbol,
        side=side,
        quantity=Decimal(qty),
        reduce_only=False,
        target_qty=Decimal(qty),
        current_qty=Decimal("0"),
        client_order_prefix=client_order_prefix,
        leg_index=leg_index,
        decision_price=Decimal("100"),
    )


def test_SCENARIO_LIVE_EXECUTION_QUALITY_SLIPPAGE_SIGN_CONVENTION() -> None:
    dt = pd.Timestamp("2026-01-01 00:00Z")
    weights = pd.Series({"BTCUSDT": 0.1})
    # BUY unfavorable: mark 100, fill 100.5 => +50 bps
    marks = {"BTCUSDT": Decimal("100")}
    intent_buy = _intent("BTCUSDT", "BUY")
    outcome_buy_unfav = ExecutionOutcome(symbol="BTCUSDT", filled_qty=Decimal("1"), unfilled_qty=Decimal("0"), avg_fill_price=Decimal("100.5"), chases=0, status="FILLED")
    recs = build_execution_quality_records(dt, "paper", weights, marks, [intent_buy], [outcome_buy_unfav])
    assert recs[0].slippage_bps == pytest.approx(50.0)

    # SELL unfavorable: mark 100, fill 99.5 => +50 bps
    intent_sell = _intent("BTCUSDT", "SELL")
    outcome_sell_unfav = ExecutionOutcome(symbol="BTCUSDT", filled_qty=Decimal("1"), unfilled_qty=Decimal("0"), avg_fill_price=Decimal("99.5"), chases=0, status="FILLED")
    recs2 = build_execution_quality_records(dt, "paper", weights, marks, [intent_sell], [outcome_sell_unfav])
    assert recs2[0].slippage_bps == pytest.approx(50.0)

    # Favorable BUY: fill 99.5 => -50 bps
    outcome_buy_fav = ExecutionOutcome(symbol="BTCUSDT", filled_qty=Decimal("1"), unfilled_qty=Decimal("0"), avg_fill_price=Decimal("99.5"), chases=0, status="FILLED")
    recs3 = build_execution_quality_records(dt, "paper", weights, marks, [intent_buy], [outcome_buy_fav])
    assert recs3[0].slippage_bps == pytest.approx(-50.0)

    # None fill => None
    outcome_none = ExecutionOutcome(symbol="BTCUSDT", filled_qty=Decimal("0"), unfilled_qty=Decimal("1"), avg_fill_price=None, chases=0, status="SHADOW")
    recs4 = build_execution_quality_records(dt, "shadow", weights, marks, [intent_buy], [outcome_none])
    assert recs4[0].slippage_bps is None


def test_SCENARIO_LIVE_EXECUTION_QUALITY_MODE_BLIND_SCHEMA() -> None:
    dt = pd.Timestamp("2026-01-01 00:00Z")
    weights = pd.Series({"A": 0.1, "B": 0.1})
    marks = {"A": Decimal("100"), "B": Decimal("100")}
    intents = [_intent("A", "BUY"), _intent("B", "BUY")]
    outcomes = [
        ExecutionOutcome(symbol="A", filled_qty=Decimal("0"), unfilled_qty=Decimal("1"), avg_fill_price=None, chases=0, status="SHADOW"),
        ExecutionOutcome(symbol="B", filled_qty=Decimal("1"), unfilled_qty=Decimal("0"), avg_fill_price=Decimal("100.5"), chases=1, status="FILLED"),
    ]
    recs = build_execution_quality_records(dt, "shadow", weights, marks, intents, outcomes)
    assert len(recs) == 2
    fields1 = {f.name for f in dataclasses.fields(recs[0])}
    fields2 = {f.name for f in dataclasses.fields(recs[1])}
    assert fields1 == fields2
    shadow_rec = recs[0]
    filled_rec = recs[1]
    assert shadow_rec.avg_fill_price is None
    assert shadow_rec.filled_qty == Decimal(0)
    assert filled_rec.avg_fill_price is not None


def test_SCENARIO_LIVE_EXECUTION_QUALITY_NO_NETWORK_IO() -> None:
    sig = inspect.signature(build_execution_quality_records)
    param_names = set(sig.parameters.keys())
    assert "client" not in param_names
    assert "session" not in param_names
    assert "socket" not in param_names
    # Ensure source has no network calls
    src = inspect.getsource(build_execution_quality_records)
    assert "REST" not in src
    assert "requests" not in src.lower()
    # Functional prove: call with fixtures produces records without needing network
    dt = pd.Timestamp("2026-01-01 00:00Z")
    weights = pd.Series({"X": 0.1})
    marks = {"X": Decimal("10")}
    intent = _intent("X", "BUY")
    outcome = ExecutionOutcome(symbol="X", filled_qty=Decimal("1"), unfilled_qty=Decimal("0"), avg_fill_price=Decimal("10.1"), chases=0, status="FILLED")
    recs = build_execution_quality_records(dt, "paper", weights, marks, [intent], [outcome])
    assert len(recs) == 1


def test_SCENARIO_LIVE_EXECUTION_QUALITY_ARCHIVE_NEVER_DELETES(monkeypatch, tmp_path: Path) -> None:
    import src.live.execution_quality as eq_mod

    # Force rotation with tiny byte budget
    monkeypatch.setattr(eq_mod, "EXECUTION_QUALITY_SHARD_MAX_BYTES", 900)
    monkeypatch.setattr(eq_mod, "EXECUTION_QUALITY_MAX_SHARDS", 12)
    history_dir = tmp_path / "eq_hist"
    dt = pd.Timestamp("2026-01-01 00:00Z")
    total = 0
    # Append enough batches to force at least one rotation
    for batch_idx in range(6):
        weights = pd.Series({f"S{i}USDT": 0.01 for i in range(5)})
        marks = {f"S{i}USDT": Decimal("100") for i in range(5)}
        intents = [_intent(f"S{i}USDT", "BUY") for i in range(5)]
        outcomes = [ExecutionOutcome(symbol=f"S{i}USDT", filled_qty=Decimal("1"), unfilled_qty=Decimal("0"), avg_fill_price=Decimal("100"), chases=0, status="FILLED") for i in range(5)]
        # vary decision_time per batch to create distinct cycles
        batch_dt = dt + pd.Timedelta(days=batch_idx)
        recs = build_execution_quality_records(batch_dt, "paper", weights, marks, intents, outcomes)
        path = append_execution_quality(recs, history_dir)
        assert path is not None
        total += len(recs)

    shards = sorted(history_dir.glob("*.parquet"))
    # Typed monthly partitions: at least 1 shard, and no deletion (combined == total)
    assert len(shards) >= 1
    # Combined row count equals total appended
    combined = 0
    for shard in shards:
        df = pd.read_parquet(shard)
        combined += len(df)
    assert combined == total


def test_SCENARIO_LIVE_EXECUTION_QUALITY_SUMMARY_INSUFFICIENT_EVIDENCE(tmp_path: Path) -> None:
    # Empty / missing history
    empty_dir = tmp_path / "empty_hist"
    res = summarize_execution_quality(empty_dir)
    assert res["n_cycles"] == 0
    assert res["sufficient_evidence"] is False

    missing_dir = tmp_path / "does_not_exist_123"
    res2 = summarize_execution_quality(missing_dir)
    assert res2["n_cycles"] == 0
    assert res2["sufficient_evidence"] is False

    # Records spanning fewer days than threshold => insufficient
    history_dir = tmp_path / "hist_short"
    dt0 = pd.Timestamp("2026-01-01 00:00Z")
    weights = pd.Series({"A": 0.1})
    marks = {"A": Decimal("100")}
    intent = _intent("A", "BUY")
    outcome = ExecutionOutcome(symbol="A", filled_qty=Decimal("1"), unfilled_qty=Decimal("0"), avg_fill_price=Decimal("100"), chases=0, status="FILLED")
    recs = build_execution_quality_records(dt0, "paper", weights, marks, [intent], [outcome])
    append_execution_quality(recs, history_dir)
    # second day still within <90
    dt1 = dt0 + pd.Timedelta(days=5)
    recs2 = build_execution_quality_records(dt1, "paper", weights, marks, [intent], [outcome])
    append_execution_quality(recs2, history_dir)
    res_short = summarize_execution_quality(history_dir)
    assert res_short["sufficient_evidence"] is False
    assert set(res_short["vs_measured_cost_tiers"].keys()) == set(MEASURED_EXECUTION_COST_TIERS_BPS.keys())

    # Spanning at least threshold => sufficient
    history_dir2 = tmp_path / "hist_long"
    for offset in [0, EXECUTION_QUALITY_MIN_EVIDENCE_DAYS]:
        dtx = dt0 + pd.Timedelta(days=int(offset))
        r = build_execution_quality_records(dtx, "paper", weights, marks, [intent], [outcome])
        append_execution_quality(r, history_dir2)
    res_long = summarize_execution_quality(history_dir2)
    assert res_long["sufficient_evidence"] is True
    assert res_long["n_days_span"] >= EXECUTION_QUALITY_MIN_EVIDENCE_DAYS
    assert set(res_long["vs_measured_cost_tiers"].keys()) == set(MEASURED_EXECUTION_COST_TIERS_BPS.keys())


def test_SCENARIO_LIVE_DEPLOYMENT_READINESS_GATES_UNCHANGED(tmp_path: Path) -> None:
    from src.mhs.evidence import compute_deployment_readiness

    # Create sufficient evidence dir but gates should remain False
    history_dir = tmp_path / "hist_gate"
    dt0 = pd.Timestamp("2026-01-01 00:00Z")
    weights = pd.Series({"A": 0.1})
    marks = {"A": Decimal("100")}
    intent = _intent("A", "BUY")
    outcome = ExecutionOutcome(symbol="A", filled_qty=Decimal("1"), unfilled_qty=Decimal("0"), avg_fill_price=Decimal("100"), chases=0, status="FILLED")
    for offset in [0, EXECUTION_QUALITY_MIN_EVIDENCE_DAYS]:
        dtx = dt0 + pd.Timedelta(days=int(offset))
        r = build_execution_quality_records(dtx, "paper", weights, marks, [intent], [outcome])
        append_execution_quality(r, history_dir)
    summ = summarize_execution_quality(history_dir)
    assert summ["sufficient_evidence"] is True

    # compute_deployment_readiness still hardcoded False
    equity = pd.Series([100.0, 101.0, 102.0, 103.0], index=pd.date_range("2026-01-01", periods=4, tz="UTC"))
    result = compute_deployment_readiness(equity, periods_per_year=252.0)
    assert result.execution_go_eligible is False
    assert result.pilot_go_eligible is False
    assert result.scale_go_eligible is False

    # Even without evidence
    empty_summ = summarize_execution_quality(tmp_path / "nope")
    assert empty_summ["sufficient_evidence"] is False
    result2 = compute_deployment_readiness(equity, periods_per_year=252.0)
    assert result2.execution_go_eligible is False
    assert result2.pilot_go_eligible is False
    assert result2.scale_go_eligible is False


def test_execution_quality_min_evidence_references_audit_retention() -> None:
    assert EXECUTION_QUALITY_MIN_EVIDENCE_DAYS == AUDIT_LOG_RETENTION_DAYS


def test_execution_quality_shard_constants_reference_run_history() -> None:
    from src.mhs.run_history import RUN_HISTORY_SHARD_MAX_BYTES, RUN_HISTORY_MAX_SHARDS
    import src.live.execution_quality as eq_mod

    # Check source literally references the run_history constants (not literal copy)
    import pathlib

    src_text = pathlib.Path(eq_mod.__file__).read_text(encoding="utf-8")
    assert "RUN_HISTORY_SHARD_MAX_BYTES" in src_text
    assert "RUN_HISTORY_MAX_SHARDS" in src_text
    assert eq_mod.EXECUTION_QUALITY_SHARD_MAX_BYTES == RUN_HISTORY_SHARD_MAX_BYTES
    assert eq_mod.EXECUTION_QUALITY_MAX_SHARDS == RUN_HISTORY_MAX_SHARDS

def test_SCENARIO_EXECUTOR_REVERSAL_LEGS_HAVE_DISTINCT_LEG_INDEX() -> None:
    """반전(reversal) 분해로 같은 심볼에 두 intent(청산=0, 신규=1)가 생겨도
    leg_index 로 항상 구분 가능해야 한다(I-LEG-DISAMBIGUATION)."""
    dt = pd.Timestamp("2026-01-01 00:00Z")
    weights = pd.Series({"BTCUSDT": 0.1})
    marks = {"BTCUSDT": Decimal("100")}
    close_leg = _intent("BTCUSDT", "SELL", leg_index=0)
    open_leg = _intent("BTCUSDT", "SELL", leg_index=1)
    outcomes = [
        ExecutionOutcome(symbol="BTCUSDT", filled_qty=Decimal("1"), unfilled_qty=Decimal("0"), avg_fill_price=Decimal("100"), chases=0, status="FILLED"),
        ExecutionOutcome(symbol="BTCUSDT", filled_qty=Decimal("1"), unfilled_qty=Decimal("0"), avg_fill_price=Decimal("100"), chases=0, status="FILLED"),
    ]
    records = build_execution_quality_records(dt, "paper", weights, marks, [close_leg, open_leg], outcomes)
    assert len(records) == 2
    assert records[0].leg_index == 0
    assert records[1].leg_index == 1
    keys = {(r.symbol, r.leg_index) for r in records}
    assert len(keys) == 2


def test_SCENARIO_EXECUTOR_RUN_ID_MATCHES_CLIENT_ORDER_PREFIX() -> None:
    """ExecutionQualityRecord.run_id 는 intent.client_order_prefix 를 그대로 옮긴다
    (raw 감사로그 조인 키)."""
    dt = pd.Timestamp("2026-01-01 00:00Z")
    weights = pd.Series({"BTCUSDT": 0.1})
    marks = {"BTCUSDT": Decimal("100")}
    intent = _intent("BTCUSDT", "BUY", client_order_prefix="20260101")
    outcome = ExecutionOutcome(symbol="BTCUSDT", filled_qty=Decimal("1"), unfilled_qty=Decimal("0"), avg_fill_price=Decimal("100"), chases=0, status="FILLED")
    records = build_execution_quality_records(dt, "paper", weights, marks, [intent], [outcome])
    assert records[0].run_id == "20260101"


def test_SCENARIO_EXECUTION_QUALITY_SCHEMA_EVOLUTION_NO_MIGRATION(tmp_path: Path) -> None:
    """leg_index/run_id/latency_seconds 컬럼이 없는 구 스키마 샤드와 신규 스키마
    레코드가 섞여도 summarize_execution_quality 는 예외 없이 동작하고 반환 dict의
    최상위 키 집합이 이번 변경 전과 동일해야 한다."""
    history_dir = tmp_path / "hist_schema_evolution"
    history_dir.mkdir(parents=True)
    old_schema_df = pd.DataFrame([
        {
            "decision_time": "2026-01-01T00:00:00+00:00",
            "symbol": "OLDUSDT",
            "mode": "paper",
            "target_weight": 0.1,
            "mark_price_at_decision": "100",
            "avg_fill_price": "100.1",
            "filled_qty": "1",
            "unfilled_qty": "0",
            "status": "FILLED",
            "chases": 0,
            "slippage_bps": 10.0,
            # leg_index/run_id/latency_seconds 컬럼 부재 -> 구 스키마 시뮬레이션
        }
    ])
    old_schema_df.to_parquet(history_dir / "active.parquet", index=False, compression="snappy")

    dt = pd.Timestamp("2026-04-01 00:00Z")
    weights = pd.Series({"NEWUSDT": 0.1})
    marks = {"NEWUSDT": Decimal("100")}
    intent = _intent("NEWUSDT", "BUY")
    outcome = ExecutionOutcome(symbol="NEWUSDT", filled_qty=Decimal("1"), unfilled_qty=Decimal("0"), avg_fill_price=Decimal("100.2"), chases=0, status="FILLED")
    records = build_execution_quality_records(dt, "paper", weights, marks, [intent], [outcome])
    append_execution_quality(records, history_dir)

    summary = summarize_execution_quality(history_dir)
    expected_keys = {
        "n_cycles", "n_fill_records", "fill_rate",
        "slippage_bps_mean", "slippage_bps_median", "slippage_bps_p90",
        "latency_seconds_mean", "latency_seconds_median", "latency_seconds_p90",
        "by_mode", "vs_measured_cost_tiers", "n_days_span", "sufficient_evidence",
    }
    assert set(summary.keys()) == expected_keys
    assert summary["n_cycles"] == 2


def test_SCENARIO_REC_01_typed_execution_quality(tmp_path: Path) -> None:
    dt = pd.Timestamp("2026-01-01 00:00Z")
    weights = pd.Series({"BTCUSDT": 0.1})
    marks = {"BTCUSDT": Decimal("100")}
    intent = OrderIntent(symbol="BTCUSDT", side="BUY", quantity=Decimal("1"), reduce_only=False, target_qty=Decimal("1"), current_qty=Decimal("0"), client_order_prefix="20260101", leg_index=0, decision_price=Decimal("100"))
    outcome = ExecutionOutcome(symbol="BTCUSDT", filled_qty=Decimal("1"), unfilled_qty=Decimal("0"), avg_fill_price=Decimal("100.5"), chases=0, status="FILLED")
    recs = build_execution_quality_records(dt, "paper", weights, marks, [intent], [outcome])
    # 100 records
    many_recs = []
    for i in range(100):
        many_recs.extend(recs)
    history_dir = tmp_path / "eq_typed"
    path = append_execution_quality(many_recs, history_dir)
    assert path is not None
    # find parquet file
    import glob

    # read via load
    from src.live.execution_quality import _load_all_frames

    combined = _load_all_frames(history_dir)
    assert combined is not None
    assert str(combined["decision_time"].dtype) == "datetime64[ns, UTC]"
    assert str(combined["mark_price_at_decision"].dtype) == "float64"
    assert str(combined["avg_fill_price"].dtype) == "float64"
    assert str(combined["filled_qty"].dtype) == "float64"
    assert str(combined["unfilled_qty"].dtype) == "float64"
    object_cols = [c for c in ["decision_time", "mark_price_at_decision", "avg_fill_price", "filled_qty", "unfilled_qty"] if combined[c].dtype == object]
    assert len(object_cols) == 0


def test_SCENARIO_REC_11_legacy_schema_coexists(tmp_path: Path) -> None:
    history_dir = tmp_path / "hist_mixed"
    history_dir.mkdir(parents=True)
    old_df = pd.DataFrame([
        {"decision_time": "2026-01-01T00:00:00+00:00", "symbol": "OLDUSDT", "mode": "paper", "target_weight": "0.1", "mark_price_at_decision": "100", "avg_fill_price": "100.1", "filled_qty": "1", "unfilled_qty": "0", "status": "FILLED", "chases": "0", "slippage_bps": "10.0", "leg_index": "0", "run_id": "20260101", "latency_seconds": "0.1", "sizing_anchor": "book_mid", "maker_fill_fraction": "0.5"}
    ])
    old_df.to_parquet(history_dir / "execution_quality_202601.parquet", index=False, compression="snappy")
    dt = pd.Timestamp("2026-02-01 00:00Z")
    weights = pd.Series({"NEWUSDT": 0.1})
    marks = {"NEWUSDT": Decimal("100")}
    intent = OrderIntent(symbol="NEWUSDT", side="BUY", quantity=Decimal("1"), reduce_only=False, target_qty=Decimal("1"), current_qty=Decimal("0"), client_order_prefix="20260201", leg_index=0, decision_price=Decimal("100"))
    outcome = ExecutionOutcome(symbol="NEWUSDT", filled_qty=Decimal("1"), unfilled_qty=Decimal("0"), avg_fill_price=Decimal("100.2"), chases=0, status="FILLED")
    recs = build_execution_quality_records(dt, "paper", weights, marks, [intent], [outcome])
    append_execution_quality(recs, history_dir)
    summary = summarize_execution_quality(history_dir)
    assert summary["n_cycles"] == 2
    assert summary["slippage_bps_mean"] is not None


#: lean_check tracking
COVERED_SCENARIOS: tuple[str, ...] = (
    "SCENARIO_LIVE_EXECUTION_QUALITY_SLIPPAGE_SIGN_CONVENTION",
    "SCENARIO_LIVE_EXECUTION_QUALITY_MODE_BLIND_SCHEMA",
    "SCENARIO_LIVE_EXECUTION_QUALITY_NO_NETWORK_IO",
    "SCENARIO_LIVE_EXECUTION_QUALITY_ARCHIVE_NEVER_DELETES",
    "SCENARIO_LIVE_EXECUTION_QUALITY_SUMMARY_INSUFFICIENT_EVIDENCE",
    "SCENARIO_LIVE_DEPLOYMENT_READINESS_GATES_UNCHANGED",
    "SCENARIO_EXECUTOR_REVERSAL_LEGS_HAVE_DISTINCT_LEG_INDEX",
    "SCENARIO_EXECUTOR_RUN_ID_MATCHES_CLIENT_ORDER_PREFIX",
    "SCENARIO_EXECUTION_QUALITY_SCHEMA_EVOLUTION_NO_MIGRATION",
    "SCENARIO_REC_01",
    "SCENARIO_REC_11",
)
# SCENARIO_REC_01-typed-execution-quality
# SCENARIO_REC_11-legacy-schema-coexists
