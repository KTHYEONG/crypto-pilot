"""Invariant scenarios for the backtest cohort of the fill calibration report."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.application.fill_calibration_backtest as fcb
from src.application.fill_calibration import run_fill_calibration
from src.common.errors import DataIntegrityError
from src.engine.account_ledger import AccountMarkPanels, _maker_fill_outcome
from tests.unit.application._fill_calibration_helpers import (
    _ts,
    _request,
    _toy_parts,
    _install_backtest,
    _install_tape,
    _slice_frame,
    _coverage_row,
    _day_ms,
    _paper_evidence,
)


def test_ohlcv_rule_matches_account_ledger_maker_decision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parts = _toy_parts()
    raw_marks = parts[1]
    parts = (parts[0], AccountMarkPanels(raw_marks.close, raw_marks.close, raw_marks.low), *parts[2:])
    _install_backtest(monkeypatch, parts)
    weights, marks, *_ = parts
    closes = marks.close["AAAUSDT"].to_numpy(dtype="float64")
    pos = marks.close.index.get_loc(pd.Timestamp("2023-01-02T23:00:00Z"))
    day = date(2023, 1, 2)
    trade_ms = _day_ms(day, 23.05)
    frame = _slice_frame(day, [(trade_ms, 1, 99.0, 1.0, True), (_day_ms(day, 23.06), 2, 101.0, 1.0, False)])
    _install_tape(
        monkeypatch,
        tmp_path / "archive",
        {("AAAUSDT", "202301"): frame, ("BBBUSDT", "202301"): frame},
        {(s, day): _coverage_row(s, day, "ok") for s in ("AAAUSDT", "BBBUSDT")}
        | {(s, date(2023, 1, 3)): _coverage_row(s, date(2023, 1, 3), "ok") for s in ("AAAUSDT", "BBBUSDT")},
    )
    report = run_fill_calibration(_request(tmp_path, k_bars=(10,)), output_dir=tmp_path / "out")
    assert report.exclusions.get("no_tape_coverage", 0) == 0
    intents = pd.read_parquet(report.intents_path)
    assert set(intents["cohort"]) == {"backtest"}
    delta = np.array([1.0])
    price = np.array([100.0])
    anchor = pos
    _, maker_fill, _ = _maker_fill_outcome(
        delta,
        price,
        anchor,
        10,
        len(closes),
        closes.reshape(-1, 1),
        (closes * 0.999).reshape(-1, 1),
        (closes * 1.001).reshape(-1, 1),
        0.0006,
        0.0002,
    )
    assert bool(maker_fill[0]) is True
    got = intents[(intents["symbol"] == "AAAUSDT") & (intents["rule"] == "ohlcv_strict")]
    assert len(got) > 0
    assert set(got["fill"].unique()) <= {0.0, 1.0}
    assert bool(got.iloc[0]["fill"]) == bool(maker_fill[0])
    import dataclasses
    import src.engine.account_ledger as ledger
    from src.market_data.binance.venue_rules import VenueBracket, VenueRuleSnapshot, VenueSymbolRules
    from src.strategy.sizing import account_growth_policy

    observed = []
    def maker(*args):
        result = _maker_fill_outcome(*args)
        observed.append(result[1])
        return result
    monkeypatch.setattr(ledger, "_maker_fill_outcome", maker)
    rules = VenueRuleSnapshot(
        captured_at=_ts("2023-01-01T00:00:00Z"),
        symbols={symbol: VenueSymbolRules(symbol=symbol, brackets=(VenueBracket(0, 1e12, 0.01, 0, 10),), step_size=0.001, min_notional=5) for symbol in weights.columns},
    )
    ledger.replay_account(
        *parts[:5], rules, dataclasses.replace(account_growth_policy(impact_y=0), kind="fixed", exposure_max=1),
        anchor_times=parts[-1], capital=100000, taker_fee_bps=6, apply_order_filters=False,
        execution="maker", maker_fee_bps=2, passive_window_bars=10,
    )
    for day_index, entry in enumerate(weights.index):
        for position, symbol in enumerate(weights.columns):
            selected = intents[(intents["entry"] == entry) & (intents["symbol"] == symbol) & (intents["rule"] == "ohlcv_strict")]
            if len(selected):
                assert bool(selected.iloc[0]["fill"]) == bool(observed[day_index][position])


def test_marketable_anchor_detected_from_tape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entries = pd.DatetimeIndex([_ts("2023-01-02T00:00:00Z")])
    anchors = pd.DatetimeIndex([_ts("2023-01-02T23:00:00Z")])
    grid = pd.date_range(anchors[0], entries[-1] + pd.Timedelta(days=1), freq="3min", inclusive="left")
    weights = pd.DataFrame([[1.0]], index=entries, columns=["AAAUSDT"])
    close = pd.DataFrame({"AAAUSDT": np.full(len(grid), 100.0)}, index=grid)
    marks = AccountMarkPanels(close=close, high=close * 1.001, low=close * 0.999)
    parts = (
        weights,
        marks,
        pd.DataFrame(0.0, index=entries, columns=["AAAUSDT"]),
        pd.DataFrame(1e9, index=entries, columns=["AAAUSDT"]),
        pd.DataFrame(0.01, index=entries, columns=["AAAUSDT"]),
        anchors,
    )
    _install_backtest(monkeypatch, parts)
    day = date(2023, 1, 2)
    a_ms = _day_ms(day, 23.0) + 3 * 60_000
    ask_ms = a_ms - 10_000
    bid_ms = a_ms - 20_000
    frame = _slice_frame(
        day, [(bid_ms, 1, 99.0, 5.0, True), (ask_ms, 2, 100.0, 5.0, False), (_day_ms(day, 23.05), 3, 99.0, 5.0, True)]
    )
    _install_tape(
        monkeypatch,
        tmp_path / "archive",
        {("AAAUSDT", "202301"): frame},
        {("AAAUSDT", day): _coverage_row("AAAUSDT", day, "ok")},
    )
    report = run_fill_calibration(
        _request(tmp_path, evaluation_end=_ts("2023-01-03T00:00:00Z")), output_dir=tmp_path / "out"
    )
    frame_out = pd.read_parquet(report.intents_path)
    feas = frame_out[frame_out["rule"] == "tape_through_feasible"]
    assert len(feas) > 0
    assert bool(feas.iloc[0]["marketable_anchor"]) is True
    assert float(feas.iloc[0]["limit"]) == pytest.approx(99.0)
    strict = frame_out[frame_out["rule"] == "ohlcv_strict"].iloc[0]
    anchor_row = frame_out[frame_out["rule"] == "tape_through_anchor"].iloc[0]
    assert strict["fill"] == anchor_row["fill"] == 1.0
    assert float(feas.iloc[0]["fill"]) == 0.0


def test_intents_without_coverage_excluded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parts = _toy_parts()
    _install_backtest(monkeypatch, parts)
    day_ok, day_absent = date(2023, 1, 2), date(2023, 1, 3)
    frame = _slice_frame(day_ok, [(_day_ms(day_ok, 23.1), 1, 100.0, 1.0, True)])
    _install_tape(
        monkeypatch,
        tmp_path / "archive",
        {("AAAUSDT", "202301"): frame, ("BBBUSDT", "202301"): frame},
        {
            ("AAAUSDT", day_ok): _coverage_row("AAAUSDT", day_ok, "ok"),
            ("BBBUSDT", day_ok): _coverage_row("BBBUSDT", day_ok, "ok"),
            ("AAAUSDT", day_absent): _coverage_row("AAAUSDT", day_absent, "absent"),
            ("BBBUSDT", day_absent): _coverage_row("BBBUSDT", day_absent, "absent"),
        },
    )
    report = run_fill_calibration(_request(tmp_path), output_dir=tmp_path / "out")
    assert report.exclusions.get("no_tape_coverage", 0) > 0
    intents = pd.read_parquet(report.intents_path)
    assert (intents["day"] != pd.Timestamp("2023-01-03T00:00:00Z")).all()


def test_design_and_holdout_never_pool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.strategy.targets import FLOW_MOM_TOP20

    cutoff = FLOW_MOM_TOP20.design_data_cutoff
    start = (cutoff - pd.Timedelta(days=2)).tz_convert("UTC")
    end = (cutoff + pd.Timedelta(days=2)).tz_convert("UTC")
    entries = pd.DatetimeIndex([start + pd.Timedelta(days=1), end - pd.Timedelta(days=1)])
    anchors = pd.DatetimeIndex(
        [pd.Timestamp("2023-01-02T23:00:00Z", tz="UTC"), pd.Timestamp("2026-07-02T23:00:00Z", tz="UTC")]
    )
    grid = pd.date_range(anchors[0], entries[-1] + pd.Timedelta(days=1), freq="3min", inclusive="left")
    weights = pd.DataFrame([[1.0], [2.0]], index=entries, columns=["AAAUSDT"])
    close = pd.DataFrame({"AAAUSDT": np.full(len(grid), 100.0)}, index=grid)
    marks = AccountMarkPanels(close=close, high=close, low=close)
    parts = (
        weights,
        marks,
        pd.DataFrame(0.0, index=entries, columns=["AAAUSDT"]),
        pd.DataFrame(1e9, index=entries, columns=["AAAUSDT"]),
        pd.DataFrame(0.01, index=entries, columns=["AAAUSDT"]),
        anchors,
    )
    _install_backtest(monkeypatch, parts)
    day_a, day_b = date(2023, 1, 2), date(2026, 7, 2)
    f = _slice_frame(day_a, [(_day_ms(day_a, 23.1), 1, 100.0, 1.0, True)])
    fb = _slice_frame(day_b, [(_day_ms(day_b, 23.1), 1, 100.0, 1.0, True)])
    _install_tape(
        monkeypatch,
        tmp_path / "archive",
        {("AAAUSDT", "202301"): f, ("AAAUSDT", "202607"): fb},
        {
            ("AAAUSDT", day_a): _coverage_row("AAAUSDT", day_a, "ok"),
            ("AAAUSDT", day_b): _coverage_row("AAAUSDT", day_b, "ok"),
        },
    )
    req = _request(tmp_path, evaluation_start=start, evaluation_end=end)
    report = run_fill_calibration(req, output_dir=tmp_path / "out")
    assert set(report.backtest["periods"]) == {"design", "holdout"}
    import src.application.fill_calibration as cal

    design_rows = pd.read_parquet(report.intents_path)
    design_rows = design_rows[design_rows["period"] == "design"]
    reduced = fcb.summarize_backtest(design_rows.to_dict("records"), req)
    assert cal._round_json(reduced["periods"]["design"]) == cal._round_json(report.backtest["periods"]["design"])


def test_bootstrap_ci_deterministic_and_brackets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parts = _toy_parts()
    _install_backtest(monkeypatch, parts)
    day = date(2023, 1, 2)
    frame = _slice_frame(day, [(_day_ms(day, 23.02), 1, 101.0, 2.0, False), (_day_ms(day, 23.03), 2, 99.0, 2.0, True)])
    cov = {(s, day): _coverage_row(s, day, "ok") for s in ("AAAUSDT", "BBBUSDT")} | {
        (s, date(2023, 1, 3)): _coverage_row(s, date(2023, 1, 3), "ok") for s in ("AAAUSDT", "BBBUSDT")
    }
    _install_tape(monkeypatch, tmp_path / "archive", {("AAAUSDT", "202301"): frame, ("BBBUSDT", "202301"): frame}, cov)
    req = _request(tmp_path)
    r1 = run_fill_calibration(req, output_dir=tmp_path / "o1")
    r2 = run_fill_calibration(req, output_dir=tmp_path / "o2")
    b1 = (r1.report_path).read_bytes()
    b2 = (r2.report_path).read_bytes()
    assert b1 == b2
    for node in r1.backtest["periods"].values():
        for entry in node["rules"].values():
            if "diff_fill_vs_ohlcv" in entry:
                lo, hi = entry["diff_fill_ci95"]
                assert lo <= entry["diff_fill_vs_ohlcv"] <= hi


def test_is_formula_fallback_when_maker_never_fills(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.core.params import ACCOUNT_TAKER_FEE_BPS

    parts = _toy_parts()
    _install_backtest(monkeypatch, parts)
    day = date(2023, 1, 2)
    frame = _slice_frame(day, [(_day_ms(day, 23.5), 1, 100.0, 0.5, True)])
    cov = {(s, day): _coverage_row(s, day, "ok") for s in ("AAAUSDT", "BBBUSDT")} | {
        (s, date(2023, 1, 3)): _coverage_row(s, date(2023, 1, 3), "ok") for s in ("AAAUSDT", "BBBUSDT")
    }
    _install_tape(monkeypatch, tmp_path / "archive", {("AAAUSDT", "202301"): frame, ("BBBUSDT", "202301"): frame}, cov)
    report = run_fill_calibration(_request(tmp_path, k_bars=(1,)), output_dir=tmp_path / "out")
    intents = pd.read_parquet(report.intents_path)
    for _, row in intents[intents["rule"].str.startswith("tape_")].iterrows():
        if row["fill"] == 0.0:
            a = float(row["anchor_price"])
            c = float(row["close_k"])
            side = 1.0 if row["side"] == "BUY" else -1.0
            assert float(row["is_bps"]) == pytest.approx(side * (c - a) / a * 1e4 + ACCOUNT_TAKER_FEE_BPS)


def test_paired_bootstrap_excludes_unavailable_rule_rows(tmp_path: Path) -> None:

    rows = []
    for index, (symbol, fill, notional) in enumerate((("AAA", 1.0, 2.0), ("BBB", 0.0, 7.0), ("CCC", 1.0, 1.0))):
        common = {
            "symbol": symbol, "entry": _ts("2023-01-02T00:00:00Z"), "day": _ts("2023-01-02T00:00:00Z"),
            "period": "design", "year": 2023, "k": 1, "notional": notional, "marketable_anchor": False,
            "fill_time_min": 1.0, "is_bps": 2.0,
        }
        rows.append({**common, "rule": "ohlcv_strict", "fill": fill})
        if index != 1:
            rows.append({**common, "rule": "tape_through_feasible", "fill": 0.0 if index == 0 else 1.0})
    report = fcb.summarize_backtest(rows, _request(tmp_path))
    metric = report["periods"]["design"]["rules"]["tape_through_feasible@K1"]
    assert metric["n_intents"] == 2
    assert metric["diff_fill_vs_ohlcv"] == pytest.approx(-2 / 3)
    assert metric["diff_fill_ci95"] == pytest.approx([-2 / 3, -2 / 3])


def test_anchor_absent_and_malformed_marks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.application.fill_calibration as cal

    parts = _toy_parts()
    weights, marks, funding, adv, sigma, anchors = parts
    bad_anchors = pd.DatetimeIndex([_ts("2020-01-01T00:00:00Z"), _ts("2023-01-03T23:00:00Z")])
    monkeypatch.setattr(cal, "_load_backtest_inputs", lambda request: (weights, marks, funding, adv, sigma, bad_anchors))
    with pytest.raises(DataIntegrityError, match="anchor bar absent"):
        run_fill_calibration(_request(tmp_path), output_dir=tmp_path / "out")
    narrow = marks.close[["AAAUSDT"]]
    from src.engine.account_ledger import AccountMarkPanels

    bad_marks = AccountMarkPanels(close=narrow, high=narrow, low=narrow)
    monkeypatch.setattr(cal, "_load_backtest_inputs", lambda request: (weights, bad_marks, funding, adv, sigma, anchors))
    with pytest.raises(DataIntegrityError, match="malformed mark panels"):
        run_fill_calibration(_request(tmp_path), output_dir=tmp_path / "out2")
    nan_close = marks.close.copy()
    nan_close.iloc[:, :] = np.nan
    nan_marks = AccountMarkPanels(close=nan_close, high=marks.high, low=marks.low)
    monkeypatch.setattr(cal, "_load_backtest_inputs", lambda request: (weights, nan_marks, funding, adv, sigma, anchors))
    cov = {(s, date(2023, 1, 2)): _coverage_row(s, date(2023, 1, 2), "ok") for s in ("AAAUSDT", "BBBUSDT")}
    cov.update({(s, date(2023, 1, 3)): _coverage_row(s, date(2023, 1, 3), "ok") for s in ("AAAUSDT", "BBBUSDT")})
    _install_tape(monkeypatch, tmp_path / "archive", {("AAAUSDT", "202301"): _slice_frame(date(2023, 1, 2), []), ("BBBUSDT", "202301"): _slice_frame(date(2023, 1, 2), [])}, cov)
    report = run_fill_calibration(_request(tmp_path), output_dir=tmp_path / "out3")
    assert report.backtest == {"periods": {}}


def test_mark_alignment_and_full_window_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    parts = _toy_parts()
    weights, marks, *rest = parts
    invalid_marks = AccountMarkPanels(close=marks.close, high=marks.high.iloc[1:], low=marks.low)
    with pytest.raises(DataIntegrityError, match="aligned indexes"):
        fcb.build_backtest_intents((weights, invalid_marks, *rest))
    repeated = pd.concat([marks.close, marks.close])
    with pytest.raises(DataIntegrityError, match="unique and ordered"):
        fcb.build_backtest_intents((weights, AccountMarkPanels(repeated, repeated, repeated), *rest))
    with pytest.raises(DataIntegrityError, match="aligned and ordered"):
        fcb.build_backtest_intents((*parts[:-1], parts[-1][:1]))
    bad_weights = weights.copy()
    bad_weights.iloc[0, 0] = np.nan
    with pytest.raises(DataIntegrityError, match="finite"):
        fcb.build_backtest_intents((bad_weights, *parts[1:]))
    _, live = _paper_evidence(tmp_path, monkeypatch)
    missing = marks.close.index[2]
    shortened = AccountMarkPanels(*(panel.drop(missing) for panel in (marks.close, marks.high, marks.low)))
    _install_backtest(monkeypatch, (weights, shortened, *rest))
    with pytest.raises(DataIntegrityError, match="complete calibration window"):
        run_fill_calibration(_request(tmp_path, k_bars=(3,)), output_dir=tmp_path / "incomplete")
    assert live.is_dir()


def test_tape_error_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.application.fill_calibration as cal
    from src.market_data.public_archive import store

    parts = _toy_parts()
    monkeypatch.setattr(cal, "_load_backtest_inputs", lambda request: parts)
    day = date(2023, 1, 2)
    frame = _slice_frame(day, [(_day_ms(day, 23.1), 1, 100.0, 1.0, True)])
    cov = {(s, day): _coverage_row(s, day, "ok") for s in ("AAAUSDT", "BBBUSDT")} | {
        (s, date(2023, 1, 3)): _coverage_row(s, date(2023, 1, 3), "ok") for s in ("AAAUSDT", "BBBUSDT")
    }
    _install_tape(monkeypatch, tmp_path / "archive", {("AAAUSDT", "202301"): frame, ("BBBUSDT", "202301"): frame}, cov)
    real_partition = store.partition_path
    monkeypatch.setattr(store, "partition_path", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad key")))
    with pytest.raises(DataIntegrityError, match="partition key"):
        run_fill_calibration(_request(tmp_path), output_dir=tmp_path / "o1")
    monkeypatch.setattr(store, "partition_path", real_partition)
    bad_path = store.partition_path(tmp_path / "archive", "aggTrades", "AAAUSDT", "202301", kind="slice")
    bad_path.write_bytes(b"not a parquet file at all" * 10)
    with pytest.raises(DataIntegrityError, match="unreadable tape"):
        run_fill_calibration(_request(tmp_path), output_dir=tmp_path / "o2")
    frame_bad = pd.DataFrame({"transact_time_ms": [1], "agg_trade_id": [1]})
    frame_bad.to_parquet(bad_path)
    with pytest.raises(DataIntegrityError, match="missing columns"):
        run_fill_calibration(_request(tmp_path), output_dir=tmp_path / "o3")


def test_window_outside_slice_and_missing_partition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.application.fill_calibration as cal

    entries = pd.DatetimeIndex([_ts("2023-01-02T00:00:00Z")])
    anchors = pd.DatetimeIndex([_ts("2023-01-02T22:00:00Z")])
    grid = pd.date_range(anchors[0], entries[-1] + pd.Timedelta(days=1), freq="3min", inclusive="left")
    weights = pd.DataFrame([[1.0]], index=entries, columns=["AAAUSDT"])
    close = pd.DataFrame({"AAAUSDT": np.full(len(grid), 100.0)}, index=grid)
    from src.engine.account_ledger import AccountMarkPanels

    marks = AccountMarkPanels(close=close, high=close, low=close)
    parts = (weights, marks, pd.DataFrame(0.0, index=entries, columns=["AAAUSDT"]), pd.DataFrame(1e9, index=entries, columns=["AAAUSDT"]), pd.DataFrame(0.01, index=entries, columns=["AAAUSDT"]), anchors)
    monkeypatch.setattr(cal, "_load_backtest_inputs", lambda request: parts)
    day = date(2023, 1, 2)
    _install_tape(monkeypatch, tmp_path / "archive", {}, {(symbol, day): _coverage_row(symbol, day, "ok") for symbol in ("AAAUSDT",)})
    report = run_fill_calibration(_request(tmp_path, evaluation_end=_ts("2023-01-03T00:00:00Z")), output_dir=tmp_path / "out")
    assert report.exclusions.get("window_outside_slice", 0) > 0
    anchors2 = pd.DatetimeIndex([_ts("2023-01-02T23:00:00Z")])
    grid2 = pd.date_range(anchors2[0], entries[-1] + pd.Timedelta(days=1), freq="3min", inclusive="left")
    close2 = pd.DataFrame({"AAAUSDT": np.full(len(grid2), 100.0)}, index=grid2)
    marks2 = AccountMarkPanels(close=close2, high=close2, low=close2)
    monkeypatch.setattr(cal, "_load_backtest_inputs", lambda request: (weights, marks2, parts[2], parts[3], parts[4], anchors2))
    with pytest.raises(DataIntegrityError, match="tape partition missing"):
        run_fill_calibration(_request(tmp_path, evaluation_end=_ts("2023-01-03T00:00:00Z")), output_dir=tmp_path / "out2")


def test_malformed_slice_and_far_lookback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.application.fill_calibration as cal

    parts = _toy_parts()
    monkeypatch.setattr(cal, "_load_backtest_inputs", lambda request: parts)
    day = date(2023, 1, 2)
    bad = pd.DataFrame({"transact_time_ms": [_day_ms(day, 23.5)], "agg_trade_id": [1], "price": [float("nan")], "quantity": [1.0], "is_buyer_maker": [True]})
    cov = {(s, day): _coverage_row(s, day, "ok") for s in ("AAAUSDT", "BBBUSDT")} | {
        (s, date(2023, 1, 3)): _coverage_row(s, date(2023, 1, 3), "ok") for s in ("AAAUSDT", "BBBUSDT")
    }
    _install_tape(monkeypatch, tmp_path / "archive", {("AAAUSDT", "202301"): bad, ("BBBUSDT", "202301"): bad}, cov)
    with pytest.raises(DataIntegrityError, match="malformed tape slice"):
        run_fill_calibration(_request(tmp_path), output_dir=tmp_path / "o1")
    good = _slice_frame(day, [(_day_ms(day, 23.5), 1, 100.0, 1.0, True)])
    _install_tape(monkeypatch, tmp_path / "archive", {("AAAUSDT", "202301"): good, ("BBBUSDT", "202301"): good}, cov)
    report = run_fill_calibration(_request(tmp_path, touch_lookback_s=7200.0), output_dir=tmp_path / "o2")
    assert report.exclusions.get("no_touch_proxy", 0) > 0
