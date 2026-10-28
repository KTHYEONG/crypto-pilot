"""Exhaustive branch coverage for the fill parity monitor."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.application import fill_parity as parity
from src.common.errors import DataIntegrityError


def _eq_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _fills_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _base_eq(ts: pd.Timestamp, symbol: str = "BTCUSDT", leg: int = 0, ref: float = 100.0, avg: float = 100.0, qty: float = 1.0):
    return {
        "decision_time": ts, "symbol": symbol, "leg_index": leg,
        "mark_price_at_decision": ref, "avg_fill_price": avg, "filled_qty": qty,
    }


def _base_fill(ts: pd.Timestamp, symbol: str = "BTCUSDT", leg: int = 0, delta: float = 1.0, price: float = 100.0, fee: float = 2.0):
    return {
        "decision_time": ts, "symbol": symbol, "leg_index": leg,
        "quantity_delta": delta, "fill_price": price, "fee_bps": fee,
    }


def test_load_branches(tmp_path: Path) -> None:
    assert parity._load_eq_frames(tmp_path / "missing").empty
    assert parity._load_fills_frames(tmp_path / "missing").empty
    empty_eq = tmp_path / "r1"
    (empty_eq / "execution_quality").mkdir(parents=True)
    (empty_eq / "fills").mkdir(parents=True)
    assert parity._load_eq_frames(empty_eq).empty
    assert parity._load_fills_frames(empty_eq).empty
    bad = tmp_path / "r2"
    (bad / "execution_quality").mkdir(parents=True)
    (bad / "fills").mkdir(parents=True)
    (bad / "execution_quality" / "a.parquet").write_text("not parquet", encoding="utf-8")
    (bad / "fills" / "a.parquet").write_text("not parquet", encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        parity._load_eq_frames(bad)
    with pytest.raises(DataIntegrityError):
        parity._load_fills_frames(bad)


def test_key_parsing_errors() -> None:
    with pytest.raises(DataIntegrityError):
        parity._day_key("not-a-date-at-all!!!")
    with pytest.raises(DataIntegrityError):
        parity._day_key(pd.NaT)
    with pytest.raises(DataIntegrityError):
        parity._as_ms("not-a-date-at-all!!!")
    with pytest.raises(DataIntegrityError):
        parity._as_ms(pd.NaT)


def test_collect_intent_errors() -> None:
    ts = pd.Timestamp("2023-01-02", tz="UTC")
    with pytest.raises(DataIntegrityError):
        parity._collect_intents(pd.DataFrame([{"symbol": "BTCUSDT"}]), pd.DataFrame())
    with pytest.raises(DataIntegrityError):
        parity._collect_intents(
            _eq_frame([_base_eq(ts)]),
            _fills_frame([{"decision_time": ts}]),
        )
    with pytest.raises(DataIntegrityError):
        parity._collect_intents(
            _eq_frame([_base_eq(ts)]),
            _fills_frame([dict(_base_fill(ts), leg_index="bad-leg")]),
        )
    with pytest.raises(DataIntegrityError):
        parity._collect_intents(
            _eq_frame([dict(_base_eq(ts), avg_fill_price="bad")]),
            _fills_frame([_base_fill(ts)]),
        )
    with pytest.raises(DataIntegrityError):
        parity._collect_intents(
            _eq_frame([dict(_base_eq(ts), mark_price_at_decision=float("nan"))]),
            _fills_frame([_base_fill(ts)]),
        )
    with pytest.raises(DataIntegrityError):
        parity._collect_intents(
            _eq_frame([dict(_base_eq(ts), filled_qty=float("nan"))]),
            _fills_frame([_base_fill(ts)]),
        )
    skipped = parity._collect_intents(
        _eq_frame([dict(_base_eq(ts), filled_qty=0.0)]),
        _fills_frame([_base_fill(ts)]),
    )
    assert skipped == []
    with pytest.raises(DataIntegrityError):
        parity._collect_intents(
            _eq_frame([dict(_base_eq(ts), ref=0.0, mark_price_at_decision=0.0)]),
            _fills_frame([_base_fill(ts)]),
        )
    with pytest.raises(DataIntegrityError):
        parity._collect_intents(
            _eq_frame([dict(_base_eq(ts), leg_index="bad")]),
            _fills_frame([_base_fill(ts)]),
        )
    with pytest.raises(DataIntegrityError):
        parity._collect_intents(
            _eq_frame([_base_eq(ts)]),
            _fills_frame([dict(_base_fill(ts, symbol="OTHER", leg=9))]),
        )
    with pytest.raises(DataIntegrityError):
        parity._collect_intents(
            _eq_frame([_base_eq(ts)]),
            _fills_frame([dict(_base_fill(ts), quantity_delta="bad")]),
        )
    with pytest.raises(DataIntegrityError):
        parity._collect_intents(
            _eq_frame([_base_eq(ts)]),
            _fills_frame([dict(_base_fill(ts), quantity_delta=float("nan"))]),
        )
    with pytest.raises(DataIntegrityError):
        parity._collect_intents(
            _eq_frame([_base_eq(ts)]),
            _fills_frame([dict(_base_fill(ts), fill_price=-1.0)]),
        )
    net_zero = parity._collect_intents(
        _eq_frame([dict(_base_eq(ts), filled_qty=1.0)]),
        pd.DataFrame([_base_fill(ts, delta=1.0), _base_fill(ts, delta=-1.0)]),
    )
    assert net_zero == []
    with pytest.raises(DataIntegrityError):
        parity._collect_intents(
            _eq_frame([dict(_base_eq(ts), filled_qty=1.0, avg_fill_price=100.0)]),
            _fills_frame([dict(_base_fill(ts, delta=1.0, price=0.0))]),
        )
    with pytest.raises(DataIntegrityError):
        parity._collect_intents(
            _eq_frame([_base_eq(ts)]),
            _fills_frame([dict(_base_fill(ts), fee_bps=float("inf"))]),
        )
    with pytest.raises(DataIntegrityError):
        parity._collect_intents(
            _eq_frame([dict(_base_eq(ts), mark_price_at_decision=100.0, avg_fill_price=float("inf"))]),
            _fills_frame([_base_fill(ts, price=100.0)]),
        )
    with pytest.raises(DataIntegrityError):
        parity._collect_intents(
            _eq_frame([dict(_base_eq(ts), filled_qty=float("inf"), avg_fill_price=100.0)]),
            _fills_frame([_base_fill(ts)]),
        )


def test_measure_invalid_and_empty(tmp_path: Path) -> None:
    with pytest.raises(DataIntegrityError):
        parity.measure_fill_parity("2023-01-02", paper_run_dir=tmp_path, archive_root=tmp_path, live_capture_root=tmp_path)  # type: ignore[arg-type]
    day = date(2023, 1, 2)
    run_dir = tmp_path / "run1"
    (run_dir / "execution_quality").mkdir(parents=True)
    row = parity.measure_fill_parity(day, paper_run_dir=run_dir, archive_root=tmp_path, live_capture_root=tmp_path)
    assert row.status == "SKIPPED"
    assert row.detail == "no_intents"
    run2 = tmp_path / "run2"
    (run2 / "execution_quality").mkdir(parents=True)
    (run2 / "fills").mkdir(parents=True)
    (run2 / "run_manifest.json").write_text('{"mode": "paper"}', encoding="utf-8")
    (run2 / "audit").mkdir(parents=True)
    row2 = parity.measure_fill_parity(day, paper_run_dir=run2, archive_root=tmp_path, live_capture_root=tmp_path)
    assert row2.status == "SKIPPED"


def test_score_through_agreement_with_posts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.application.fill_parity as mod

    def _fake_score(request, exclusions):
        rows = [
            {"rule": "through_only", "paper_filled": True, "fill": 1.0},
            {"rule": "through_only", "paper_filled": False, "fill": 0.0},
            {"rule": "through_only", "paper_filled": True, "fill": 0.0},
            {"rule": "other", "paper_filled": True, "fill": 1.0},
        ]
        return rows, {}

    monkeypatch.setattr("src.application.fill_calibration_paper.score_paper", _fake_score)
    day = date(2023, 1, 2)
    agreement, n = mod._score_through_agreement(day, paper_run_dir=tmp_path, archive_root=tmp_path, live_capture_root=tmp_path)
    assert n == 3
    assert agreement == pytest.approx(2 / 3)


def test_trailing_file_branches(tmp_path: Path) -> None:
    assert parity.trailing_breach(tmp_path / "missing.jsonl", parity.FillParityThresholds()) is None
    path = tmp_path / "p.jsonl"
    path.write_text("\n", encoding="utf-8")
    assert parity.trailing_breach(path, parity.FillParityThresholds()) is None
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not json\n", encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        parity.trailing_breach(bad, parity.FillParityThresholds())
    bad2 = tmp_path / "bad2.jsonl"
    bad2.write_text('{"day": "2023-01-02"}\n', encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        parity.trailing_breach(bad2, parity.FillParityThresholds())
    bad3 = tmp_path / "bad3.jsonl"
    bad3.write_text('{"day": "2023-01-02", "status": "WEIRD", "run_id": "r", "n_intents": 1, "n_posts": 1}\n', encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        parity.trailing_breach(bad3, parity.FillParityThresholds())
    skipped_only = tmp_path / "sk.jsonl"
    row = parity.FillParityRow(day=date(2023, 1, 2), run_id="r", n_intents=0, n_posts=0, through_agreement=None, paper_is_bps=None, rule_is_bps=None, gap_bps=None, status="SKIPPED", detail="no_intents")
    parity.append_parity_row(row, skipped_only)
    assert parity.trailing_breach(skipped_only, parity.FillParityThresholds()) is None
    agree_path = tmp_path / "agree.jsonl"
    for i in range(7):
        parity.append_parity_row(
            parity.FillParityRow(day=date(2023, 1, 2 + i), run_id="r", n_intents=30, n_posts=10, through_agreement=0.5, paper_is_bps=2.0, rule_is_bps=2.0, gap_bps=0.0, status="OK", detail=None),
            agree_path,
        )
    breach = parity.trailing_breach(agree_path, parity.FillParityThresholds())
    assert breach is not None
    assert "agreement" in breach


def test_append_chmod_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    row = parity.FillParityRow(day=date(2023, 1, 2), run_id="r", n_intents=1, n_posts=0, through_agreement=None, paper_is_bps=None, rule_is_bps=None, gap_bps=None, status="OK", detail=None)
    path = tmp_path / "p.jsonl"

    def _boom(p, mode):
        raise OSError("chmod denied")

    monkeypatch.setattr("src.application.fill_parity.os.chmod", _boom)
    with pytest.raises(DataIntegrityError):
        parity.append_parity_row(row, path)


def test_rule_insufficient_bars_and_no_included(tmp_path: Path) -> None:
    from src.market_data.public_archive import store

    day = date(2023, 1, 2)
    archive = tmp_path / "archive"
    month = day.strftime("%Y%m")
    release = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=23)
    rows = [(int(release.value // 10**6) + 60_000, 1, 100.0, 1.0, True)]
    frame = pd.DataFrame({
        "transact_time_ms": [r[0] for r in rows],
        "agg_trade_id": [r[1] for r in rows],
        "price": [r[2] for r in rows],
        "quantity": [r[3] for r in rows],
        "is_buyer_maker": [r[4] for r in rows],
    })
    path = store.partition_path(archive, "aggTrades", "BTCUSDT", month, kind="slice")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    intents = [{"symbol": "BTCUSDT", "side": "BUY", "ref": 100.0, "notional": 1.0, "paper_is": 2.0}]
    rule_mean, n_included, detail, included = parity._rule_outcome(intents, archive, day)
    assert n_included == 0
    assert rule_mean is None
    assert detail is not None


def test_collect_tiny_and_huge_values_trigger_guards() -> None:
    ts = pd.Timestamp("2023-01-02", tz="UTC")
    with pytest.raises(DataIntegrityError):
        parity._collect_intents(
            _eq_frame([_base_eq(ts, qty=1e-300, avg=1e-300)]),
            _fills_frame([_base_fill(ts, delta=1e-300, price=1e-300)]),
        )
    with pytest.raises(DataIntegrityError):
        parity._collect_intents(
            _eq_frame([_base_eq(ts, qty=1.0, avg=100.0)]),
            _fills_frame([_base_fill(ts, delta=1e308, price=1.0, fee=1e308)]),
        )
    with pytest.raises(DataIntegrityError):
        parity._collect_intents(
            _eq_frame([dict(_base_eq(ts), mark_price_at_decision=1e-308, avg_fill_price=1e308)]),
            _fills_frame([_base_fill(ts)]),
        )
    with pytest.raises(DataIntegrityError):
        parity._collect_intents(
            _eq_frame([dict(_base_eq(ts), filled_qty=1e308, avg_fill_price=100.0)]),
            _fills_frame([_base_fill(ts, delta=1.0, price=100.0)]),
        )


def test_collect_malformed_fills_row_via_monkeypatch(monkeypatch: pytest.MonkeyPatch) -> None:
    ts = pd.Timestamp("2023-01-02", tz="UTC")

    def _boom(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(pd, "to_numeric", _boom)
    with pytest.raises(DataIntegrityError):
        parity._collect_intents(_eq_frame([_base_eq(ts)]), _fills_frame([_base_fill(ts)]))


def test_rule_nonfinite_via_inf_tape(tmp_path: Path) -> None:
    from src.market_data.public_archive import store

    day = date(2023, 1, 2)
    archive = tmp_path / "archive_inf"
    month = day.strftime("%Y%m")
    release = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=23)
    prices = [float("inf")] * 10
    rows = [(int(release.value // 10**6) + i * 180_000 + 60_000, i + 1, p, 1.0, True) for i, p in enumerate(prices)]
    frame = pd.DataFrame({
        "transact_time_ms": [r[0] for r in rows],
        "agg_trade_id": [r[1] for r in rows],
        "price": [r[2] for r in rows],
        "quantity": [r[3] for r in rows],
        "is_buyer_maker": [r[4] for r in rows],
    })
    path = store.partition_path(archive, "aggTrades", "BTCUSDT", month, kind="slice")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    intents = [{"symbol": "BTCUSDT", "side": "BUY", "ref": 100.0, "notional": 1.0, "paper_is": 2.0}]
    with pytest.raises(DataIntegrityError):
        parity._rule_outcome(intents, archive, day)
    frame["price"] = 100.0
    frame.to_parquet(path)
    intents[0]["ref"] = 1e-308
    with pytest.raises(DataIntegrityError, match="rule shortfall"):
        parity._rule_outcome(intents, archive, day)


def test_missing_decision_column_is_integrity_error(tmp_path: Path) -> None:
    directory = tmp_path / "execution_quality"
    directory.mkdir()
    pd.DataFrame([{"symbol": "BTCUSDT"}]).to_parquet(directory / "rows.parquet")
    with pytest.raises(DataIntegrityError, match="decision_time"):
        parity.measure_fill_parity(date(2023, 1, 2), paper_run_dir=tmp_path, archive_root=tmp_path, live_capture_root=tmp_path)


def test_measure_collect_empty_and_tape_missing_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    day = date(2023, 1, 2)
    ts = pd.Timestamp(day, tz="UTC")
    run_dir = tmp_path / "run_bad"
    eq_dir = run_dir / "execution_quality"
    eq_dir.mkdir(parents=True)
    pd.DataFrame([_base_eq(ts)]).to_parquet(eq_dir / "a.parquet")
    with pytest.raises(DataIntegrityError):
        parity.measure_fill_parity(day, paper_run_dir=run_dir, archive_root=tmp_path, live_capture_root=tmp_path)

    def _raise_data(*args, **kwargs):
        raise DataIntegrityError("boom decision")

    monkeypatch.setattr(parity, "_day_key", _raise_data)
    run_ok = tmp_path / "run_ok"
    (run_ok / "execution_quality").mkdir(parents=True)
    pd.DataFrame([_base_eq(ts)]).to_parquet(run_ok / "execution_quality" / "a.parquet")
    with pytest.raises(DataIntegrityError):
        parity.measure_fill_parity(day, paper_run_dir=run_ok, archive_root=tmp_path, live_capture_root=tmp_path)

    def _raise_other(*args, **kwargs):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(parity, "_day_key", _raise_other)
    with pytest.raises(DataIntegrityError):
        parity.measure_fill_parity(day, paper_run_dir=run_ok, archive_root=tmp_path, live_capture_root=tmp_path)


def test_measure_no_included_and_score_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.market_data.public_archive import store

    day = date(2023, 1, 2)
    ts = pd.Timestamp(day, tz="UTC")
    run_dir = tmp_path / "run_inc"
    eq_dir = run_dir / "execution_quality"
    fills_dir = run_dir / "fills"
    eq_dir.mkdir(parents=True)
    fills_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text('{"mode": "paper"}', encoding="utf-8")
    (run_dir / "audit").mkdir(parents=True)
    (run_dir / "audit" / "a.jsonl").write_text("", encoding="utf-8")
    pd.DataFrame([_base_eq(ts, qty=0.0)]).to_parquet(eq_dir / "a.parquet")
    row = parity.measure_fill_parity(day, paper_run_dir=run_dir, archive_root=tmp_path, live_capture_root=tmp_path)
    assert row.status == "SKIPPED"

    run_dir2 = tmp_path / "run_inc2"
    (run_dir2 / "execution_quality").mkdir(parents=True)
    (run_dir2 / "fills").mkdir(parents=True)
    (run_dir2 / "run_manifest.json").write_text('{"mode": "paper"}', encoding="utf-8")
    (run_dir2 / "audit").mkdir(parents=True)
    (run_dir2 / "audit" / "a.jsonl").write_text("", encoding="utf-8")
    pd.DataFrame([_base_eq(ts)]).to_parquet(run_dir2 / "execution_quality" / "a.parquet")
    pd.DataFrame([_base_fill(ts)]).to_parquet(run_dir2 / "fills" / "a.parquet")
    archive = tmp_path / "arch2"
    month = day.strftime("%Y%m")
    release = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=23)
    frame = pd.DataFrame({
        "transact_time_ms": [int(release.value // 10**6) + 60_000],
        "agg_trade_id": [1],
        "price": [100.0],
        "quantity": [1.0],
        "is_buyer_maker": [True],
    })
    p = store.partition_path(archive, "aggTrades", "BTCUSDT", month, kind="slice")
    p.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(p)

    def _fake_score(day_, *, paper_run_dir, archive_root, live_capture_root):
        return None, 0

    monkeypatch.setattr(parity, "_score_through_agreement", _fake_score)
    row2 = parity.measure_fill_parity(day, paper_run_dir=run_dir2, archive_root=archive, live_capture_root=tmp_path / "live2")
    assert row2.status == "OK"
    assert row2.n_intents == 0

    def _raise_tape_missing(*args, **kwargs):
        raise DataIntegrityError("tape partition missing for covered")

    monkeypatch.setattr(parity, "_score_through_agreement", _raise_tape_missing)
    archive_full = tmp_path / "arch_full"
    prices = [99.0] * 10
    rows = [(int(release.value // 10**6) + i * 180_000 + 60_000, i + 1, pr, 1.0, True) for i, pr in enumerate(prices)]
    full = pd.DataFrame({
        "transact_time_ms": [r[0] for r in rows],
        "agg_trade_id": [r[1] for r in rows],
        "price": [r[2] for r in rows],
        "quantity": [r[3] for r in rows],
        "is_buyer_maker": [r[4] for r in rows],
    })
    pf = store.partition_path(archive_full, "aggTrades", "BTCUSDT", month, kind="slice")
    pf.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(pf)
    row3 = parity.measure_fill_parity(day, paper_run_dir=run_dir2, archive_root=archive_full, live_capture_root=tmp_path / "live3")
    assert row3.status == "SKIPPED"
    assert row3.detail == "tape_missing"

    def _raise_other_tape(*args, **kwargs):
        raise DataIntegrityError("other integrity failure")

    monkeypatch.setattr(parity, "_score_through_agreement", _raise_other_tape)
    with pytest.raises(DataIntegrityError):
        parity.measure_fill_parity(day, paper_run_dir=run_dir2, archive_root=archive_full, live_capture_root=tmp_path / "live3")


def test_parse_and_trailing_oserror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad = tmp_path / "non_dict.jsonl"
    bad.write_text("[]\n", encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        parity.trailing_breach(bad, parity.FillParityThresholds())

    def _raise_oserror(path_arg, *args, **kwargs):
        raise OSError("denied")

    real_read = Path.read_text

    def _fake_read(self, *args, **kwargs):
        if str(self).endswith("unreadable.jsonl"):
            raise OSError("denied")
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _fake_read)
    (tmp_path / "unreadable.jsonl").write_text("x", encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        parity.trailing_breach(tmp_path / "unreadable.jsonl", parity.FillParityThresholds())
