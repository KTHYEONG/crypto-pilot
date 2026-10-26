"""Invariant scenarios for the tape-based fill calibration report."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.application.fill_calibration import FillCalibrationRequest, run_fill_calibration
from src.common.errors import DataIntegrityError
from src.engine.account_ledger import AccountMarkPanels, _maker_fill_outcome


def _ts(raw: str) -> pd.Timestamp:
    return pd.Timestamp(raw, tz="UTC")


def _request(tmp_path: Path, **overrides) -> FillCalibrationRequest:
    base = {
        "evaluation_start": _ts("2023-01-02T00:00:00Z"),
        "evaluation_end": _ts("2023-01-04T00:00:00Z"),
        "archive_root": tmp_path / "archive",
        "k_bars": (1,),
        "bootstrap_samples": 100,
        "seed": 7,
    }
    base.update(overrides)
    return FillCalibrationRequest(**base)


def _toy_parts(symbols=("AAAUSDT", "BBBUSDT"), closes=(100.0, 200.0)):
    entries = pd.DatetimeIndex([_ts("2023-01-02T00:00:00Z"), _ts("2023-01-03T00:00:00Z")])
    anchors = pd.DatetimeIndex([_ts("2023-01-02T23:00:00Z"), _ts("2023-01-03T23:00:00Z")])
    grid = pd.date_range(anchors[0], entries[-1] + pd.Timedelta(days=1), freq="3min", inclusive="left")
    weights = pd.DataFrame([[1.0, 0.0], [0.5, 0.5]], index=entries, columns=list(symbols))
    close = pd.DataFrame(
        {s: np.full(len(grid), c, dtype="float64") for s, c in zip(symbols, closes, strict=True)}, index=grid
    )
    high = close * 1.001
    low = close * 0.999
    marks = AccountMarkPanels(close=close, high=high, low=low)
    funding = pd.DataFrame(0.0, index=entries, columns=list(symbols))
    adv = pd.DataFrame(1e9, index=entries, columns=list(symbols))
    sigma = pd.DataFrame(0.01, index=entries, columns=list(symbols))
    return (weights, marks, funding, adv, sigma, anchors)


def _install_backtest(monkeypatch: pytest.MonkeyPatch, parts) -> None:
    import src.application.fill_calibration as cal

    monkeypatch.setattr(cal, "_load_backtest_inputs", lambda request: parts)


def _install_tape(
    monkeypatch: pytest.MonkeyPatch, archive_root: Path, frames: dict[tuple[str, str], pd.DataFrame], coverage: dict
) -> None:
    from src.market_data.public_archive import store

    monkeypatch.setattr(store, "load_coverage", lambda root, dataset: coverage)
    for (symbol, month), frame in frames.items():
        path = store.partition_path(archive_root, "aggTrades", symbol, month, kind="slice")
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)


def _slice_frame(day: date, rows: list[tuple[int, int, float, float, bool]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transact_time_ms": [r[0] for r in rows],
            "agg_trade_id": [r[1] for r in rows],
            "price": [r[2] for r in rows],
            "quantity": [r[3] for r in rows],
            "is_buyer_maker": [r[4] for r in rows],
        }
    )


def _coverage_row(symbol: str, day: date, status: str):
    from src.market_data.public_archive.store import CoverageRow

    return CoverageRow(
        dataset="aggTrades",
        symbol=symbol,
        day=day,
        status=status,
        raw_sha256=None,
        rows=10,
        recorded_at=_ts("2023-01-05T00:00:00Z"),
        detail=None,
    )


def _day_ms(day: date, hour: float) -> int:
    return int((pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=hour)).value // 10**6)


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
    reduced = cal._summarize_backtest(design_rows.to_dict("records"), req, {})
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


def test_paper_confusion_uses_observed_queue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.application.fill_calibration as cal

    parts = _toy_parts()
    monkeypatch.setattr(cal, "_load_backtest_inputs", lambda request: parts)
    day = date(2023, 1, 2)
    paper_dir = tmp_path / "run"
    (paper_dir / "audit").mkdir(parents=True)
    post_ts = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=23, minutes=1)
    base_ms = int(post_ts.value // 10**6)
    tape_frame = _slice_frame(day, [(base_ms + (i + 1) * 10_000, i + 1, 100.0, 3.0, True) for i in range(4)])
    (paper_dir / "audit" / "a.jsonl").write_text(
        json.dumps(
            {
                "event": "order_posted",
                "client_order_id": "c1",
                "symbol": "AAAUSDT",
                "side": "BUY",
                "time_in_force": "GTX",
                "price": "100",
                "quantity": "5",
                "ts": post_ts.isoformat(),
            }
        )
        + "\n"
        + json.dumps(
            {
                "event": "fill",
                "qty": "5",
                "client_order_id": "c1",
                "symbol": "AAAUSDT",
                "ts": (post_ts + pd.Timedelta(minutes=1)).isoformat(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (paper_dir / "run_manifest.json").write_text(json.dumps({"mode": "paper"}), encoding="utf-8")
    live_root = tmp_path / "live"
    depth_dir = live_root / "exec_depth" / "20230102"
    depth_dir.mkdir(parents=True)
    depth_row = pd.DataFrame(
        {
            "symbol": ["AAAUSDT"],
            "transact_time_ms": [int(post_ts.value // 10**6)],
            "bid_px_0": [100.0],
            "bid_qty_0": [10.0],
            "ask_px_0": [101.0],
            "ask_qty_0": [10.0],
        }
    )
    depth_row.to_parquet(depth_dir / "part_230000_0.parquet")
    from src.market_data.public_archive import store

    monkeypatch.setattr(
        store,
        "load_coverage",
        lambda root, dataset: {
            ("AAAUSDT", day): _coverage_row("AAAUSDT", day, "ok"),
            ("BBBUSDT", day): _coverage_row("BBBUSDT", day, "ok"),
            ("AAAUSDT", date(2023, 1, 3)): _coverage_row("AAAUSDT", date(2023, 1, 3), "ok"),
            ("BBBUSDT", date(2023, 1, 3)): _coverage_row("BBBUSDT", date(2023, 1, 3), "ok"),
        },
    )
    for sym in ("AAAUSDT", "BBBUSDT"):
        p = store.partition_path(tmp_path / "archive", "aggTrades", sym, "202301", kind="slice")
        p.parent.mkdir(parents=True, exist_ok=True)
        tape_frame.to_parquet(p)
    req = _request(tmp_path, paper_run_dir=paper_dir, live_capture_root=live_root)
    report = run_fill_calibration(req, output_dir=tmp_path / "out")
    assert report.paper is not None
    through = report.paper["variants"]["through_only"]["confusion"]
    queued = report.paper["variants"]["queue_observed"]["confusion"]
    front = report.paper["variants"]["front_of_queue"]["confusion"]
    assert through["paper_only"] == 1
    assert queued["both_filled"] == 1
    assert front["both_filled"] == 1
    scored = pd.read_parquet(report.intents_path)
    paper_fills = scored[scored["cohort"] == "paper"].set_index("rule")["fill"].to_dict()
    assert paper_fills == pytest.approx({"through_only": 0.0, "queue_observed": 2 / 5, "front_of_queue": 1.0})


def test_non_paper_run_dir_rejected(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text(json.dumps({"mode": "live"}), encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        run_fill_calibration(
            _request(tmp_path, paper_run_dir=run_dir, live_capture_root=tmp_path), output_dir=tmp_path / "out"
        )


def test_existing_output_dir_never_overwritten(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.application.fill_calibration as cal

    parts = _toy_parts()
    monkeypatch.setattr(cal, "_load_backtest_inputs", lambda request: parts)
    monkeypatch.setattr(cal, "_utc_now_stamp", lambda: "20230101T000000Z")
    out = tmp_path / "out"
    (out / "20230101T000000Z").mkdir(parents=True)
    sentinel = out / "20230101T000000Z" / "report.json"
    sentinel.write_text("{}", encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        run_fill_calibration(_request(tmp_path), output_dir=out)
    assert sentinel.read_text(encoding="utf-8") == "{}"


def test_holdout_gate_before_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.cli.commands.backtest as cli

    called: list = []
    import src.application.fill_calibration as cal

    def _boom(request):
        called.append(request)
        raise AssertionError("must not load")

    monkeypatch.setattr(cal, "_load_backtest_inputs", _boom)
    import src.evaluation.holdout as holdout_mod

    monkeypatch.setattr(holdout_mod, "holdout_covered", lambda *a, **k: False)
    args = type("A", (), {})()
    args.start = "2026-08-01"
    args.end = "2026-09-01"
    args.archive_root = None
    args.paper_run_dir = None
    args.live_capture_root = None
    args.k_bars = None
    args.touch_lookback_s = None
    args.queue_ahead_fraction = None
    args.bootstrap_samples = None
    args.seed = None
    args.output_dir = str(tmp_path / "out")
    args.data_root = None
    args.total_tree_pss_bytes = None
    args.replay_tree_pss_bytes = None
    args.min_available_bytes = None
    with pytest.raises(SystemExit):
        cli.run_fill_calibration_command(args)
    assert called == []


def _args(**overrides):
    args = type("A", (), {})()
    args.start = "2023-01-01"
    args.end = "2023-02-01"
    args.archive_root = None
    args.paper_run_dir = None
    args.live_capture_root = None
    args.k_bars = None
    args.touch_lookback_s = None
    args.queue_ahead_fraction = None
    args.bootstrap_samples = None
    args.seed = None
    args.output_dir = None
    args.data_root = None
    args.total_tree_pss_bytes = None
    args.replay_tree_pss_bytes = None
    args.min_available_bytes = None
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_request_validation_matrix(tmp_path: Path) -> None:
    good = {"evaluation_start": _ts("2023-01-02T00:00:00Z"), "evaluation_end": _ts("2023-01-04T00:00:00Z"), "archive_root": tmp_path}
    bads = [
        {"evaluation_start": pd.Timestamp("2023-01-02")},
        {"evaluation_start": pd.Timestamp("2023-01-02 09:00:00+09:00")},
        {"evaluation_start": _ts("2023-01-05T00:00:00Z")},
        {"k_bars": ()},
        {"k_bars": [1, 3]},
        {"k_bars": (3, 1)},
        {"k_bars": (1, 1)},
        {"k_bars": (0,)},
        {"k_bars": (True,)},
        {"touch_lookback_s": 0.0},
        {"touch_lookback_s": float("nan")},
        {"bootstrap_samples": 50},
        {"bootstrap_samples": True},
        {"paper_run_dir": tmp_path},
        {"live_capture_root": tmp_path},
        {"archive_root": "nope"},
        {"data_root": 3},
        {"seed": -1},
        {"touch_lookback_s": 0.0001},
        {"queue_ahead_fraction": 1.1},
    ]
    for overrides in bads:
        kwargs = {**good, **overrides}
        with pytest.raises(ValueError, match=r".+"):
            FillCalibrationRequest(**kwargs)


def test_helper_branches() -> None:
    import src.application.fill_calibration as cal

    assert cal._kind_of(0.0, 1.0) == "open"
    assert cal._kind_of(1.0, 0.0) == "close"
    assert cal._kind_of(1.0, -1.0) == "flip"
    assert cal._kind_of(1.0, 2.0) == "increase"
    assert cal._kind_of(2.0, 1.0) == "reduce"
    assert cal._round_json(float("nan")) is None
    rounded = cal._round_json({"a": [1, 2.123456789, True, None], "b": (1.5,)})
    assert rounded == {"a": [1, 2.123457, True, None], "b": [1.5]}
    assert cal._round_json("x") == "x"
    assert np.isnan(cal._weighted_mean(np.array([1.0]), np.array([0.0])))
    assert np.isnan(cal._weighted_mean(np.array([np.nan]), np.array([1.0])))
    assert np.isnan(cal._percentile(np.array([np.nan]), 50.0))
    assert cal._percentile(np.array([1.0, 2.0]), 50.0) == pytest.approx(1.5)
    pt, lo, hi = cal._bootstrap_diff_ci(np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), samples=100, seed=7)
    assert np.isnan(pt)
    assert np.isnan(lo)
    assert np.isnan(hi)
    pt, lo, hi = cal._bootstrap_diff_ci(
        np.array([1]), np.array([1, 1]), np.array([1.0, 2.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0]), samples=100, seed=7
    )
    assert np.isnan(lo)
    assert np.isnan(hi)


def test_paired_bootstrap_excludes_unavailable_rule_rows(tmp_path: Path) -> None:
    import src.application.fill_calibration as cal

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
    report = cal._summarize_backtest(rows, _request(tmp_path), {})
    metric = report["periods"]["design"]["rules"]["tape_through_feasible@K1"]
    assert metric["n_intents"] == 2
    assert metric["diff_fill_vs_ohlcv"] == pytest.approx(-2 / 3)
    assert metric["diff_fill_ci95"] == pytest.approx([-2 / 3, -2 / 3])


def _paper_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trades=None):
    _install_backtest(monkeypatch, _toy_parts())
    day = date(2023, 1, 2)
    stamp = _ts("2023-01-02T23:01:00Z")
    frame = _slice_frame(day, trades if trades is not None else [(int(stamp.value // 10**6) + 20_000, 1, 99.0, 100.0, True)])
    cov = {(s, d): _coverage_row(s, d, "ok") for s in ("AAAUSDT", "BBBUSDT") for d in (day, date(2023, 1, 3))}
    _install_tape(monkeypatch, tmp_path / "archive", {(s, "202301"): frame for s in ("AAAUSDT", "BBBUSDT")}, cov)
    live = tmp_path / "capture"
    live.mkdir()
    return stamp, live


def test_paper_partial_fill_does_not_terminate_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stamp = _ts("2023-01-02T23:01:00Z")
    base = int(stamp.value // 10**6)
    _, live = _paper_evidence(tmp_path, monkeypatch, [(base + 40_000, 1, 99.0, 10.0, True)])
    records = [
        {"event": "order_posted", "client_order_id": "partial", "side": "BUY", "symbol": "AAAUSDT", "price": "100", "quantity": "5", "time_in_force": "GTX", "ts": stamp.isoformat()},
        {"event": "fill", "client_order_id": "partial", "qty": "2", "ts": (stamp + pd.Timedelta(seconds=10)).isoformat()},
        {"event": "fill", "client_order_id": "partial", "qty": "3", "ts": (stamp + pd.Timedelta(minutes=1)).isoformat()},
    ]
    run = _paper_run(tmp_path / "paper", [json.dumps(row) + "\n" for row in records])
    report = run_fill_calibration(_request(tmp_path, paper_run_dir=run, live_capture_root=live), output_dir=tmp_path / "out")
    assert report.paper["variants"]["through_only"]["confusion"]["both_filled"] == 1
    frame = pd.read_parquet(report.intents_path)
    paper = frame[(frame["cohort"] == "paper") & (frame["rule"] == "through_only")].iloc[0]
    assert paper["fill"] == 1.0
    assert paper["paper_fill_time_min"] == pytest.approx(1 / 6)
    assert paper["fill_time_min"] == pytest.approx(2 / 3)


def test_paper_cancel_ignores_late_fill_and_gap_is_notional_weighted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stamp, live = _paper_evidence(tmp_path, monkeypatch)
    rows = []
    for cid, price, qty in (("filled", 100, 5), ("missed", 90, 50)):
        rows.extend([
            {"event": "order_posted", "client_order_id": cid, "side": "BUY", "symbol": "AAAUSDT", "price": str(price), "quantity": str(qty), "time_in_force": "GTX", "ts": stamp.isoformat()},
            {"event": "fill", "client_order_id": cid, "qty": str(qty), "ts": (stamp + pd.Timedelta(minutes=1)).isoformat()},
        ])
    rows.extend([
        {"event": "order_posted", "client_order_id": "late", "side": "BUY", "symbol": "AAAUSDT", "price": "100", "quantity": "5", "time_in_force": "GTX", "ts": stamp.isoformat()},
        {"event": "order_cancelled", "client_order_id": "late", "ts": (stamp + pd.Timedelta(seconds=5)).isoformat()},
        {"event": "fill", "client_order_id": "late", "qty": "5", "ts": (stamp + pd.Timedelta(minutes=2)).isoformat()},
    ])
    run = _paper_run(tmp_path / "paper", [json.dumps(row) + "\n" for row in rows])
    report = run_fill_calibration(_request(tmp_path, paper_run_dir=run, live_capture_root=live), output_dir=tmp_path / "out")
    assert report.paper["variants"]["through_only"]["fill_rate_gap"] == pytest.approx(-4500 / 5500)
    assert report.paper["variants"]["through_only"]["confusion"] == {"both_filled": 1, "paper_only": 1, "neither": 1, "tape_only": 0}


def test_real_paper_post_side_comes_from_read_only_journal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stamp = _ts("2023-01-02T23:01:00Z")
    base = int(stamp.value // 10**6)
    _, live = _paper_evidence(tmp_path, monkeypatch, [(base + 20_000, 1, 101.0, 10.0, False)])
    run = _paper_run(tmp_path / "paper", [])
    events = [
        {"event": "order_posted", "client_order_id": "sell", "symbol": "AAAUSDT", "price": "100", "quantity": "5", "time_in_force": "GTX", "ts": stamp.isoformat()},
        {"event": "fill", "client_order_id": "sell", "qty": "5", "ts": (stamp + pd.Timedelta(minutes=1)).isoformat()},
    ]
    (run / "audit" / "a.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events))
    with pytest.raises(DataIntegrityError, match="no side"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=run, live_capture_root=live), output_dir=tmp_path / "missing")
    journal = run / "order_journal.jsonl"
    journal.write_text(json.dumps({"event": "submit", "client_order_id": "sell", "side": "SELL"}) + "\n")
    before = journal.read_bytes()
    report = run_fill_calibration(_request(tmp_path, paper_run_dir=run, live_capture_root=live), output_dir=tmp_path / "out")
    assert report.paper["variants"]["through_only"]["confusion"]["both_filled"] == 1
    assert journal.read_bytes() == before


def test_queue_snapshot_cannot_arrive_after_post(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stamp, live = _paper_evidence(tmp_path, monkeypatch)
    directory = live / "exec_depth" / "20230102"
    directory.mkdir(parents=True)
    pd.DataFrame({
        "symbol": ["AAAUSDT"], "received_at": [stamp + pd.Timedelta(milliseconds=1)],
        "transact_time_ms": [int(stamp.value // 10**6) - 100],
        "bid_px_0": [100.0], "bid_qty_0": [1000.0], "ask_px_0": [101.0], "ask_qty_0": [1.0],
    }).to_parquet(directory / "part.parquet")
    run = _paper_run(tmp_path / "paper", [
        json.dumps({"event": "order_posted", "client_order_id": "p", "side": "BUY", "symbol": "AAAUSDT", "price": "100", "quantity": "5", "time_in_force": "GTX", "ts": stamp.isoformat()}) + "\n",
        json.dumps({"event": "order_cancelled", "client_order_id": "p", "ts": (stamp + pd.Timedelta(minutes=1)).isoformat()}) + "\n",
    ])
    report = run_fill_calibration(_request(tmp_path, paper_run_dir=run, live_capture_root=live), output_dir=tmp_path / "out")
    assert report.paper["queue_fallback_count"] == 1


def test_paper_periods_and_requested_window_never_pool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.application.fill_calibration as cal
    from src.market_data.public_archive import store

    design_stamp, live = _paper_evidence(tmp_path, monkeypatch)
    holdout_stamp = _ts("2026-07-02T23:01:00Z")
    day = holdout_stamp.date()
    frame = _slice_frame(day, [(int(holdout_stamp.value // 10**6) + 20_000, 1, 99.0, 10.0, True)])
    path = store.partition_path(tmp_path / "archive", "aggTrades", "AAAUSDT", "202607", kind="slice")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    coverage = store.load_coverage(tmp_path / "archive", "aggTrades")
    coverage[("AAAUSDT", day)] = _coverage_row("AAAUSDT", day, "ok")
    monkeypatch.setattr(store, "load_coverage", lambda *args: coverage)
    events = []
    for cid, stamp in (("d", design_stamp), ("h", holdout_stamp)):
        events.extend([
            {"event": "order_posted", "client_order_id": cid, "side": "BUY", "symbol": "AAAUSDT", "price": "100", "quantity": "5", "time_in_force": "GTX", "ts": stamp.isoformat()},
            {"event": "fill", "client_order_id": cid, "qty": "5", "ts": (stamp + pd.Timedelta(minutes=1)).isoformat()},
        ])
    run = _paper_run(tmp_path / "paper", [json.dumps(row) + "\n" for row in events])
    full = run_fill_calibration(_request(tmp_path, evaluation_end=_ts("2026-07-03T00:00:00Z"), paper_run_dir=run, live_capture_root=live), output_dir=tmp_path / "full")
    assert set(full.paper["periods"]) == {"design", "holdout"}
    assert "variants" not in full.paper
    design = run_fill_calibration(_request(tmp_path, paper_run_dir=run, live_capture_root=live), output_dir=tmp_path / "design")
    assert set(design.paper["periods"]) == {"design"}
    assert cal._round_json(design.paper["periods"]["design"]) == cal._round_json(full.paper["periods"]["design"])


def test_cli_preserves_zero_controls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.application.fill_calibration as cal
    import src.cli.commands.backtest as cli

    captured = []
    def fake(request, *, output_dir):
        captured.append(request)
        return cal.FillCalibrationReport({}, None, {}, tmp_path / "i", tmp_path / "r")
    monkeypatch.setattr(cal, "run_fill_calibration", fake)
    cli.run_fill_calibration_command(_args(seed=0, queue_ahead_fraction=0.0))
    assert captured[0].seed == 0
    assert captured[0].queue_ahead_fraction == 0.0
    with pytest.raises(SystemExit) as exc:
        cli.run_fill_calibration_command(_args(touch_lookback_s=0.0))
    assert exc.value.code == 2


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
    import src.application.fill_calibration as cal

    parts = _toy_parts()
    weights, marks, *rest = parts
    invalid_marks = AccountMarkPanels(close=marks.close, high=marks.high.iloc[1:], low=marks.low)
    with pytest.raises(DataIntegrityError, match="aligned indexes"):
        cal._build_backtest_intents((weights, invalid_marks, *rest))
    repeated = pd.concat([marks.close, marks.close])
    with pytest.raises(DataIntegrityError, match="unique and ordered"):
        cal._build_backtest_intents((weights, AccountMarkPanels(repeated, repeated, repeated), *rest))
    with pytest.raises(DataIntegrityError, match="aligned and ordered"):
        cal._build_backtest_intents((*parts[:-1], parts[-1][:1]))
    bad_weights = weights.copy()
    bad_weights.iloc[0, 0] = np.nan
    with pytest.raises(DataIntegrityError, match="finite"):
        cal._build_backtest_intents((bad_weights, *parts[1:]))
    _, live = _paper_evidence(tmp_path, monkeypatch)
    missing = marks.close.index[2]
    shortened = AccountMarkPanels(*(panel.drop(missing) for panel in (marks.close, marks.high, marks.low)))
    _install_backtest(monkeypatch, (weights, shortened, *rest))
    with pytest.raises(DataIntegrityError, match="complete calibration window"):
        run_fill_calibration(_request(tmp_path, k_bars=(3,)), output_dir=tmp_path / "incomplete")
    assert live.is_dir()


def test_journal_corruption_and_direction_conflicts_are_rejected(tmp_path: Path) -> None:
    import src.application.fill_calibration as cal

    stamp = _ts("2023-01-02T23:01:00Z")
    run = _paper_run(tmp_path, [json.dumps({"event": "order_posted", "client_order_id": "p", "side": "BUY", "symbol": "AAAUSDT", "price": "100", "quantity": "5", "time_in_force": "GTX", "ts": stamp.isoformat()}) + "\n"])
    journal = run / "order_journal.jsonl"
    journal.write_text("[]\n")
    with pytest.raises(DataIntegrityError, match="malformed paper order journal"):
        cal._load_paper_posts(run)
    journal.write_text(json.dumps({"event": "submit", "client_order_id": "p", "side": "SELL"}) + "\n")
    with pytest.raises(DataIntegrityError, match="disagrees with journal"):
        cal._load_paper_posts(run)
    journal.write_text("")
    with (run / "audit" / "a.jsonl").open("a") as handle:
        handle.write(json.dumps({"event": "fill", "client_order_id": "p", "qty": "0", "ts": stamp.isoformat()}) + "\n")
    with pytest.raises(DataIntegrityError, match="fill quantity"):
        cal._load_paper_posts(run)
    with pytest.raises(DataIntegrityError, match="NaT"):
        cal._parse_ts("NaT")
    with pytest.raises(DataIntegrityError, match="quantities must be non-negative"):
        cal._queue_ahead("BUY", 100, pd.Series({"bid_px_0": 100, "bid_qty_0": -1}), 1)
    with pytest.raises(DataIntegrityError, match="malformed exec_depth quantities"):
        cal._queue_ahead("BUY", 100, pd.Series({"bid_px_0": 100}), 1)


def test_partition_pruning_caching_and_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    import itertools
    import src.application.fill_calibration as cal

    stamp, live = _paper_evidence(tmp_path, monkeypatch)
    ignored = live / "exec_depth" / "20180101"
    ignored.mkdir(parents=True)
    (ignored / "corrupt.parquet").write_bytes(b"must never be read")
    posts = []
    for cid in ("a", "b"):
        posts.extend([
            {"event": "order_posted", "client_order_id": cid, "side": "BUY", "symbol": "AAAUSDT", "price": "100", "quantity": "5", "time_in_force": "GTX", "ts": stamp.isoformat()},
            {"event": "order_cancelled", "client_order_id": cid, "ts": (stamp + pd.Timedelta(minutes=1)).isoformat()},
        ])
    run = _paper_run(tmp_path / "paper", [json.dumps(row) + "\n" for row in posts])
    reads = []
    real_read = pd.read_parquet
    def read(path, **kwargs):
        if "aggTrades_slice" in str(path):
            reads.append((str(path), kwargs.get("columns")))
        return real_read(path, **kwargs)
    monkeypatch.setattr(pd, "read_parquet", read)
    clock = itertools.count(step=61)
    monkeypatch.setattr(cal.time, "monotonic", lambda: next(clock))
    with caplog.at_level("INFO"):
        report = run_fill_calibration(_request(tmp_path, paper_run_dir=run, live_capture_root=live), output_dir=tmp_path / "out")
    assert len(reads) == 3
    assert all(tuple(columns) == cal._SLICE_COLUMNS for _, columns in reads)
    assert "cohort=paper done=" in caplog.text
    assert "cohort=backtest done=" in caplog.text
    assert report.paper["variants"]["through_only"]["confusion"]["tape_only"] == 2


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


def _paper_run(tmp_path: Path, lines: list[str], mode: str = "paper") -> Path:
    run_dir = tmp_path / "run"
    (run_dir / "audit").mkdir(parents=True, exist_ok=True)
    synthetic = []
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            synthetic.append(line)
            continue
        if isinstance(record, dict):
            if record.get("event") == "order_posted":
                record.setdefault("side", "BUY")
            elif record.get("event") == "fill":
                record.setdefault("qty", "5")
        synthetic.append(json.dumps(record) + "\n")
    (run_dir / "audit" / "a.jsonl").write_text("".join(synthetic), encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(json.dumps({"mode": mode}), encoding="utf-8")
    return run_dir


def test_paper_audit_error_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.application.fill_calibration as cal

    parts = _toy_parts()
    monkeypatch.setattr(cal, "_load_backtest_inputs", lambda request: parts)
    day = date(2023, 1, 2)
    cov = {(s, day): _coverage_row(s, day, "ok") for s in ("AAAUSDT", "BBBUSDT")} | {
        (s, date(2023, 1, 3)): _coverage_row(s, date(2023, 1, 3), "ok") for s in ("AAAUSDT", "BBBUSDT")
    }
    frame = _slice_frame(day, [(_day_ms(day, 23.1), 1, 100.0, 1.0, True)])
    _install_tape(monkeypatch, tmp_path / "archive", {("AAAUSDT", "202301"): frame, ("BBBUSDT", "202301"): frame}, cov)
    live = tmp_path / "live"
    live.mkdir()
    with pytest.raises(DataIntegrityError, match="non-paper"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=_paper_run(tmp_path, [], mode="live"), live_capture_root=live), output_dir=tmp_path / "o0")
    run_dir = tmp_path / "run2"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text("not json", encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="manifest"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=run_dir, live_capture_root=live), output_dir=tmp_path / "o1")
    run_dir3 = tmp_path / "run3"
    run_dir3.mkdir()
    (run_dir3 / "run_manifest.json").write_text(json.dumps({"mode": "paper"}), encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="audit directory missing"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=run_dir3, live_capture_root=live), output_dir=tmp_path / "o2")
    (run_dir3 / "audit").mkdir()
    with pytest.raises(DataIntegrityError, match="no jsonl"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=run_dir3, live_capture_root=live), output_dir=tmp_path / "o3")
    post_ts = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=23, minutes=1)
    run_ok = _paper_run(tmp_path / "ok", ["\n", json.dumps({"event": "order_posted", "client_order_id": "c1", "symbol": "AAAUSDT", "time_in_force": "GTX", "price": "100", "quantity": "5", "ts": post_ts.isoformat()}) + "\n", "oops\n"])
    with pytest.raises(DataIntegrityError, match="malformed paper audit line"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=run_ok, live_capture_root=live), output_dir=tmp_path / "o4")
    run_ok2 = _paper_run(tmp_path / "ok2", ["[1]\n"])
    with pytest.raises(DataIntegrityError, match="malformed paper audit line"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=run_ok2, live_capture_root=live), output_dir=tmp_path / "o5")
    run_ok3 = _paper_run(
        tmp_path / "ok3",
        [
            json.dumps({"event": "order_posted", "client_order_id": "c1", "symbol": "AAAUSDT", "time_in_force": "IOC", "price": "100", "quantity": "5", "ts": post_ts.isoformat()}) + "\n",
            json.dumps({"event": "order_posted", "symbol": "AAAUSDT", "time_in_force": "GTX", "price": "100", "quantity": "5", "ts": post_ts.isoformat()}) + "\n",
            json.dumps({"event": "order_posted", "client_order_id": "c1", "symbol": "AAAUSDT", "time_in_force": "GTX", "price": "bad", "quantity": "5", "ts": post_ts.isoformat()}) + "\n",
        ],
    )
    with pytest.raises(DataIntegrityError, match="malformed order_posted"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=run_ok3, live_capture_root=live), output_dir=tmp_path / "o6")
    run_ok4 = _paper_run(
        tmp_path / "ok4",
        [
            json.dumps({"event": "order_posted", "client_order_id": "c1", "symbol": "AAAUSDT", "time_in_force": "GTX", "price": "100", "quantity": "0", "ts": post_ts.isoformat()}) + "\n",
        ],
    )
    with pytest.raises(DataIntegrityError, match="malformed order_posted"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=run_ok4, live_capture_root=live), output_dir=tmp_path / "o7")
    run_ok5 = _paper_run(
        tmp_path / "ok5",
        [
            json.dumps({"event": "order_posted", "client_order_id": "c1", "symbol": "AAAUSDT", "time_in_force": "GTX", "price": "100", "quantity": "5"}) + "\n",
        ],
    )
    with pytest.raises(DataIntegrityError, match="missing ts"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=run_ok5, live_capture_root=live), output_dir=tmp_path / "o8")
    run_ok6 = _paper_run(
        tmp_path / "ok6",
        [
            json.dumps({"event": "order_posted", "client_order_id": "c1", "symbol": "AAAUSDT", "time_in_force": "GTX", "price": "100", "quantity": "5", "ts": "not-a-time"}) + "\n",
        ],
    )
    with pytest.raises(DataIntegrityError, match="malformed audit ts"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=run_ok6, live_capture_root=live), output_dir=tmp_path / "o9")
    run_ok7 = _paper_run(
        tmp_path / "ok7",
        [
            json.dumps({"event": "order_posted", "client_order_id": "c1", "symbol": "AAAUSDT", "time_in_force": "GTX", "price": "100", "quantity": "5", "ts": "2023-01-02 23:01:00"}) + "\n",
            json.dumps({"event": "order_cancelled", "client_order_id": "c1", "ts": "2023-01-02 23:02:00"}) + "\n",
        ],
    )
    report = run_fill_calibration(_request(tmp_path, paper_run_dir=run_ok7, live_capture_root=live), output_dir=tmp_path / "o10")
    assert report.paper is not None
    run_ok8 = _paper_run(
        tmp_path / "ok8",
        [
            json.dumps({"event": "order_posted", "client_order_id": "c1", "symbol": "AAAUSDT", "time_in_force": "GTX", "price": "100", "quantity": "5", "ts": post_ts.isoformat()}) + "\n",
            json.dumps({"event": "order_cancelled", "client_order_id": "c1", "ts": "bad-ts"}) + "\n",
        ],
    )
    with pytest.raises(DataIntegrityError, match="malformed order_cancelled"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=run_ok8, live_capture_root=live), output_dir=tmp_path / "o11")
    run_ok9 = _paper_run(
        tmp_path / "ok9",
        [
            json.dumps({"event": "order_posted", "client_order_id": "c1", "symbol": "AAAUSDT", "time_in_force": "GTX", "price": "100", "quantity": "5", "ts": post_ts.isoformat()}) + "\n",
            json.dumps({"event": "fill", "client_order_id": "c1", "ts": "bad-ts"}) + "\n",
        ],
    )
    with pytest.raises(DataIntegrityError, match="malformed fill record"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=run_ok9, live_capture_root=live), output_dir=tmp_path / "o12")


def test_paper_depth_and_outcome_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.application.fill_calibration as cal
    from src.market_data.public_archive import store

    parts = _toy_parts()
    monkeypatch.setattr(cal, "_load_backtest_inputs", lambda request: parts)
    day = date(2023, 1, 2)
    post_ts = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=23, minutes=1)
    base_ms = int(post_ts.value // 10**6)
    tape_frame = _slice_frame(day, [(base_ms + 10_000, 1, 100.0, 50.0, False)])
    for sym in ("AAAUSDT", "BBBUSDT"):
        p = store.partition_path(tmp_path / "archive", "aggTrades", sym, "202301", kind="slice")
        p.parent.mkdir(parents=True, exist_ok=True)
        tape_frame.to_parquet(p)
    monkeypatch.setattr(store, "load_coverage", lambda root, dataset: {(s, day): _coverage_row(s, day, "ok") for s in ("AAAUSDT", "BBBUSDT")} | {(s, date(2023, 1, 3)): _coverage_row(s, date(2023, 1, 3), "ok") for s in ("AAAUSDT", "BBBUSDT")})
    live = tmp_path / "live"
    depth_dir = live / "exec_depth" / "20230102"
    depth_dir.mkdir(parents=True)
    depth_row = pd.DataFrame({"symbol": ["AAAUSDT"], "transact_time_ms": [base_ms], "bid_px_0": [100.0], "bid_qty_0": [4.0], "ask_px_0": [101.0], "ask_qty_0": [4.0]})
    depth_row.to_parquet(depth_dir / "part_0.parquet")
    run_cancel = _paper_run(
        tmp_path / "cancel",
        [
            json.dumps({"event": "order_posted", "client_order_id": "s1", "symbol": "AAAUSDT", "side": "SELL", "time_in_force": "GTX", "price": "101", "quantity": "5", "ts": post_ts.isoformat()}) + "\n",
            json.dumps({"event": "order_cancelled", "client_order_id": "s1", "ts": (post_ts + pd.Timedelta(minutes=5)).isoformat()}) + "\n",
            json.dumps({"event": "order_posted", "client_order_id": "u1", "symbol": "AAAUSDT", "side": "BUY", "time_in_force": "GTX", "price": "50", "quantity": "5", "ts": post_ts.isoformat()}) + "\n",
            json.dumps({"event": "order_posted", "client_order_id": "u2", "symbol": "AAAUSDT", "side": "BUY", "time_in_force": "GTX", "price": "50", "quantity": "5", "ts": (post_ts - pd.Timedelta(minutes=5)).isoformat()}) + "\n",
            json.dumps({"event": "order_cancelled", "client_order_id": "u2", "ts": post_ts.isoformat()}) + "\n",
        ],
    )
    report = run_fill_calibration(_request(tmp_path, paper_run_dir=run_cancel, live_capture_root=live), output_dir=tmp_path / "oc")
    assert report.paper is not None
    assert report.exclusions.get("paper_post_unresolved", 0) >= 1
    both = sum(v["confusion"]["tape_only"] + v["confusion"]["neither"] for v in report.paper["variants"].values())
    assert both >= 3
    assert report.paper["variants"]["queue_observed"]["confusion"]["neither"] == 1
    assert report.paper["optimistic_full_fill_share"] == 1.0
    run_stale = _paper_run(
        tmp_path / "stale",
        [json.dumps({"event": "order_posted", "client_order_id": "c1", "symbol": "AAAUSDT", "side": "BUY", "time_in_force": "GTX", "price": "100", "quantity": "5", "ts": post_ts.isoformat()}) + "\n", json.dumps({"event": "fill", "client_order_id": "c1", "ts": (post_ts + pd.Timedelta(minutes=1)).isoformat()}) + "\n"],
    )
    (depth_dir / "part_0.parquet").unlink()
    pd.DataFrame({"symbol": ["AAAUSDT"], "received_at": [post_ts - pd.Timedelta(seconds=30)], "bid_px_0": [100.0], "bid_qty_0": [10.0], "ask_px_0": [102.0], "ask_qty_0": [10.0]}).to_parquet(depth_dir / "part_1.parquet")
    report2 = run_fill_calibration(_request(tmp_path, paper_run_dir=run_stale, live_capture_root=live), output_dir=tmp_path / "os")
    assert report2.paper is not None
    assert report2.paper["queue_fallback_count"] >= 1


def test_paper_depth_error_tables_and_sell_books(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.application.fill_calibration as cal

    parts = _toy_parts()
    monkeypatch.setattr(cal, "_load_backtest_inputs", lambda request: parts)
    day = date(2023, 1, 2)
    live = tmp_path / "live"
    (live / "exec_depth" / "20230102").mkdir(parents=True)
    bad_depth = pd.DataFrame({"transact_time_ms": [1]})
    bad_depth.to_parquet(live / "exec_depth" / "20230102" / "part_0.parquet")
    post_ts = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=23, minutes=1)
    run_dir = _paper_run(tmp_path / "d1", [json.dumps({"event": "order_posted", "client_order_id": "c1", "symbol": "AAAUSDT", "time_in_force": "GTX", "price": "100", "quantity": "5", "ts": post_ts.isoformat()}) + "\n", json.dumps({"event": "order_cancelled", "client_order_id": "c1", "ts": (post_ts + pd.Timedelta(minutes=1)).isoformat()}) + "\n"])
    base_cov = {(s, day): _coverage_row(s, day, "ok") for s in ("AAAUSDT", "BBBUSDT")} | {
        (s, date(2023, 1, 3)): _coverage_row(s, date(2023, 1, 3), "ok") for s in ("AAAUSDT", "BBBUSDT")
    }
    tape_frame = _slice_frame(day, [(_day_ms(day, 23.1), 1, 100.0, 1.0, True)])
    _install_tape(monkeypatch, tmp_path / "archive", {("AAAUSDT", "202301"): tape_frame, ("BBBUSDT", "202301"): tape_frame}, base_cov)
    with pytest.raises(DataIntegrityError, match="missing symbol column"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=run_dir, live_capture_root=live), output_dir=tmp_path / "od1")
    for f in (live / "exec_depth" / "20230102").glob("*.parquet"):
        f.unlink()
    pd.DataFrame({"symbol": ["AAAUSDT"]}).to_parquet(live / "exec_depth" / "20230102" / "part_0.parquet")
    with pytest.raises(DataIntegrityError, match="missing time column"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=run_dir, live_capture_root=live), output_dir=tmp_path / "od2")
    for f in (live / "exec_depth" / "20230102").glob("*.parquet"):
        f.unlink()
    (live / "exec_depth" / "20230102" / "junk.txt").write_text("x", encoding="utf-8")
    (live / "exec_depth" / "20230102" / "junk.parquet").write_bytes(b"garbage-bytes-not-parquet")
    with pytest.raises(DataIntegrityError, match="unreadable exec_depth"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=run_dir, live_capture_root=live), output_dir=tmp_path / "od3")
    for f in (live / "exec_depth" / "20230102").glob("*"):
        f.unlink()
    assert cal._queue_ahead("BUY", 105.0, pd.Series({"bid_px_0": 100.0, "bid_qty_0": 3.0}), 1) == (0.0, None)
    assert cal._queue_ahead("BUY", 100.0, pd.Series({"bid_px_0": 100.0, "bid_qty_0": 3.0}), 1) == (3.0, 3.0)
    ask_row = pd.Series({"ask_px_0": 100.0, "ask_qty_0": 3.0, "ask_px_1": 101.0, "ask_qty_1": 4.0, "bid_px_0": 99.0, "bid_qty_0": 5.0, "bid_px_1": 98.0, "bid_qty_1": 6.0})
    assert cal._queue_ahead("SELL", 99.0, ask_row, 2) == (0.0, None)
    assert cal._queue_ahead("SELL", 100.0, ask_row, 2) == (3.0, 3.0)
    assert cal._queue_ahead("SELL", 100.5, ask_row, 2) == (0.0, None)
    assert cal._queue_ahead("SELL", 105.0, ask_row, 2) == (None, None)
    assert cal._queue_ahead("SELL", 100.0, pd.Series({"ask_px_0": float("nan"), "ask_qty_0": 1.0}), 1) == (None, None)
    assert cal._queue_ahead("BUY", 50.0, pd.Series({"bid_px_0": 100.0, "bid_qty_0": 3.0, "bid_px_1": 90.0, "bid_qty_1": 4.0}), 2) == (None, None)
    bid_row = pd.Series({"bid_px_0": 100.0, "bid_qty_0": 3.0, "bid_px_1": 90.0, "bid_qty_1": 4.0})
    assert cal._queue_ahead("BUY", 95.0, bid_row, 2) == (0.0, None)
    assert cal._queue_ahead("BUY", 100.0, pd.Series({"bid_px_0": float("nan"), "bid_qty_0": 1.0}), 1) == (None, None)


def test_run_guards_and_write_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.application.fill_calibration as cal

    parts = _toy_parts()
    monkeypatch.setattr(cal, "_load_backtest_inputs", lambda request: parts)
    day = date(2023, 1, 2)
    cov = {(s, day): _coverage_row(s, day, "ok") for s in ("AAAUSDT", "BBBUSDT")} | {(s, date(2023, 1, 3)): _coverage_row(s, date(2023, 1, 3), "ok") for s in ("AAAUSDT", "BBBUSDT")}
    frame = _slice_frame(day, [(_day_ms(day, 23.1), 1, 100.0, 1.0, True)])
    _install_tape(monkeypatch, tmp_path / "archive", {("AAAUSDT", "202301"): frame, ("BBBUSDT", "202301"): frame}, cov)
    with pytest.raises(DataIntegrityError, match="FillCalibrationRequest"):
        run_fill_calibration("nope", output_dir=tmp_path / "out")  # type: ignore[arg-type]
    with pytest.raises(DataIntegrityError, match="output_dir must be a Path"):
        run_fill_calibration(_request(tmp_path), output_dir="nope")  # type: ignore[arg-type]
    bad = _request(tmp_path)
    object.__setattr__(bad, "archive_root", "nope")
    with pytest.raises(DataIntegrityError, match="archive_root must be a Path"):
        run_fill_calibration(bad, output_dir=tmp_path / "out")
    monkeypatch.setattr(cal, "_load_backtest_inputs", lambda request: (_ for _ in ()).throw(DataIntegrityError("boom")))
    with pytest.raises(DataIntegrityError, match="malformed calibration inputs"):
        run_fill_calibration(_request(tmp_path), output_dir=tmp_path / "out2")

    def _boom_inputs(request):
        raise DataIntegrityError("boom")

    monkeypatch.setattr(cal, "_load_backtest_inputs", _boom_inputs)
    with pytest.raises(DataIntegrityError, match="malformed calibration inputs"):
        run_fill_calibration(_request(tmp_path), output_dir=tmp_path / "out3")
    monkeypatch.setattr(cal, "_load_backtest_inputs", lambda request: parts)
    import pathlib as _pl

    real_mkdir = _pl.Path.mkdir

    def _race(self, *args, **kwargs):
        raise FileExistsError("race")

    monkeypatch.setattr(_pl.Path, "mkdir", _race)
    with pytest.raises(DataIntegrityError, match="must be fresh"):
        run_fill_calibration(_request(tmp_path), output_dir=tmp_path / "out4")
    monkeypatch.setattr(_pl.Path, "mkdir", real_mkdir)
    monkeypatch.setattr(cal, "write_parquet_atomic", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(DataIntegrityError, match="intents write failed"):
        run_fill_calibration(_request(tmp_path), output_dir=tmp_path / "out5")
    import src.common.parquet_io as _pio

    monkeypatch.setattr(cal, "write_parquet_atomic", _pio.write_parquet_atomic)
    import os as _os

    real_replace = _os.replace

    def _report_only_replace(src, dst, *args, **kwargs):
        if str(dst).endswith("report.json"):
            raise OSError("disk full")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(cal.os, "replace", _report_only_replace)
    with pytest.raises(DataIntegrityError, match="report write failed"):
        run_fill_calibration(_request(tmp_path), output_dir=tmp_path / "out6")


def test_cli_happy_and_error_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    import src.application.fill_calibration as cal
    import src.cli.commands.backtest as cli

    seen = {}

    def _fake_run(request, *, output_dir):
        seen["request"] = request
        seen["output_dir"] = output_dir
        return cal.FillCalibrationReport(backtest={"periods": {"design": {"rules": {"ohlcv_strict@K1": {"n_intents": 2, "fill_fraction": 0.5}}, "marketable_anchor_share": 0.1}}}, paper=None, exclusions={}, intents_path=tmp_path / "i.parquet", report_path=tmp_path / "r.json")

    monkeypatch.setattr(cli, "run_fill_calibration_command", cli.run_fill_calibration_command)
    import src.application.fill_calibration as calmod

    monkeypatch.setattr(calmod, "run_fill_calibration", _fake_run)
    args = _args(output_dir=str(tmp_path / "out"))
    cli.run_fill_calibration_command(args)
    out = capsys.readouterr().out
    assert "fill-calibration period=design" in out
    assert seen["request"].k_bars == (1, 3, 5, 10)
    assert seen["request"].touch_lookback_s == 60.0
    with pytest.raises(SystemExit) as exc:
        cli.run_fill_calibration_command(_args(start="2023-02-01", end="2023-01-01"))
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        cli.run_fill_calibration_command(_args(k_bars=[0]))
    assert exc.value.code == 2

    def _boom_run(request, *, output_dir):
        from src.common.errors import DataIntegrityError as _D

        raise _D("bad tape")

    monkeypatch.setattr(calmod, "run_fill_calibration", _boom_run)
    with pytest.raises(SystemExit) as exc:
        cli.run_fill_calibration_command(_args(output_dir=str(tmp_path / "out2")))
    assert exc.value.code == 2
    assert "bad tape" in capsys.readouterr().err
    monkeypatch.setattr(calmod, "run_fill_calibration", _fake_run)
    import src.evaluation.holdout as holdout_mod

    def _integrity(*args, **kwargs):
        from src.common.errors import DataIntegrityError as _D

        raise _D("journal unreadable")

    monkeypatch.setattr(holdout_mod, "holdout_covered", _integrity)
    with pytest.raises(SystemExit) as exc:
        cli.run_fill_calibration_command(_args(start="2026-08-01", end="2026-09-01", output_dir=str(tmp_path / "out3")))
    assert exc.value.code == 2
    assert "journal unreadable" in capsys.readouterr().err

    def _nan_run(request, *, output_dir):
        return cal.FillCalibrationReport(backtest={"periods": {"design": {"rules": {}, "marketable_anchor_share": None}}}, paper=None, exclusions={}, intents_path=tmp_path / "i.parquet", report_path=tmp_path / "r.json")

    monkeypatch.setattr(calmod, "run_fill_calibration", _nan_run)
    cli.run_fill_calibration_command(_args(output_dir=str(tmp_path / "out4")))
    assert "nan" in capsys.readouterr().out


def test_paper_remaining_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.application.fill_calibration as cal
    from src.market_data.public_archive import store

    parts = _toy_parts()
    monkeypatch.setattr(cal, "_load_backtest_inputs", lambda request: parts)
    day = date(2023, 1, 2)
    post_ts = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=23, minutes=1)
    base_ms = int(post_ts.value // 10**6)
    tape_frame = _slice_frame(
        day,
        [
            (_day_ms(day, 23.1), 1, 100.0, 50.0, True),
            (base_ms + 30_000, 2, 101.0, 1.0, False),
        ],
    )
    feb_day = date(2023, 2, 6)
    feb_post = pd.Timestamp(feb_day, tz="UTC") + pd.Timedelta(hours=23, minutes=1)
    cov = {(s, day): _coverage_row(s, day, "ok") for s in ("AAAUSDT", "BBBUSDT")} | {
        (s, date(2023, 1, 3)): _coverage_row(s, date(2023, 1, 3), "ok") for s in ("AAAUSDT", "BBBUSDT")
    }
    cov[(("AAAUSDT", feb_day))] = _coverage_row("AAAUSDT", feb_day, "ok")
    _install_tape(monkeypatch, tmp_path / "archive", {("AAAUSDT", "202301"): tape_frame, ("BBBUSDT", "202301"): tape_frame}, cov)
    live = tmp_path / "live"
    live.mkdir()
    jan5 = pd.Timestamp(date(2023, 1, 5), tz="UTC") + pd.Timedelta(hours=23, minutes=1)
    run_cov = _paper_run(
        tmp_path / "cov",
        [
            json.dumps({"event": "order_posted", "client_order_id": "c1", "symbol": "AAAUSDT", "time_in_force": "GTX", "price": "100", "quantity": "5", "ts": jan5.isoformat()}) + "\n",
            json.dumps({"event": "order_cancelled", "client_order_id": "c1", "ts": (jan5 + pd.Timedelta(minutes=1)).isoformat()}) + "\n",
            json.dumps({"event": "order_posted", "client_order_id": "c2", "symbol": "AAAUSDT", "time_in_force": "GTX", "price": "100", "quantity": "5", "ts": post_ts.isoformat()}) + "\n",
            json.dumps({"event": "order_cancelled", "client_order_id": "c2", "ts": (post_ts - pd.Timedelta(minutes=10)).isoformat()}) + "\n",
            json.dumps({"event": "order_posted", "client_order_id": "s1", "symbol": "AAAUSDT", "side": "SELL", "time_in_force": "GTX", "price": "100", "quantity": "5", "ts": post_ts.isoformat()}) + "\n",
            json.dumps({"event": "order_cancelled", "client_order_id": "s1", "ts": (post_ts + pd.Timedelta(minutes=5)).isoformat()}) + "\n",
        ],
    )
    report = run_fill_calibration(_request(tmp_path, paper_run_dir=run_cov, live_capture_root=live, evaluation_end=_ts("2023-03-01T00:00:00Z")), output_dir=tmp_path / "ocov")
    assert report.paper is not None
    assert report.exclusions.get("paper_no_tape_coverage", 0) >= 1
    assert report.exclusions.get("paper_post_unresolved", 0) >= 1
    assert report.paper["variants"]["through_only"]["confusion"]["tape_only"] >= 1
    import pathlib as _pl

    real_read = _pl.Path.read_text

    def _boom_read(self, *args, **kwargs):
        if self.name == "a.jsonl":
            raise OSError("disk gone")
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(_pl.Path, "read_text", _boom_read)
    with pytest.raises(DataIntegrityError, match="unreadable paper audit"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=run_cov, live_capture_root=live), output_dir=tmp_path / "oord")
    monkeypatch.setattr(_pl.Path, "read_text", real_read)
    feb_lines = [
        json.dumps({"event": "order_posted", "client_order_id": "f1", "symbol": "AAAUSDT", "time_in_force": "GTX", "price": "100", "quantity": "5", "ts": feb_post.isoformat()}) + "\n",
        json.dumps({"event": "order_cancelled", "client_order_id": "f1", "ts": (feb_post + pd.Timedelta(minutes=1)).isoformat()}) + "\n",
    ]
    feb_month = feb_post.strftime("%Y%m")
    real_partition = store.partition_path

    def _key_boom(root, dataset, symbol, month, *, kind="derived"):
        if month == feb_month:
            raise ValueError("bad key")
        return real_partition(root, dataset, symbol, month, kind=kind)

    run_feb = _paper_run(tmp_path / "feb", feb_lines)
    monkeypatch.setattr(store, "partition_path", _key_boom)
    with pytest.raises(DataIntegrityError, match="malformed tape partition key"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=run_feb, live_capture_root=live, evaluation_end=_ts("2023-03-01T00:00:00Z")), output_dir=tmp_path / "ofeb0")
    monkeypatch.setattr(store, "partition_path", real_partition)
    with pytest.raises(DataIntegrityError, match="tape partition missing"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=run_feb, live_capture_root=live, evaluation_end=_ts("2023-03-01T00:00:00Z")), output_dir=tmp_path / "ofeb1")
    feb_path = real_partition(tmp_path / "archive", "aggTrades", "AAAUSDT", feb_month, kind="slice")
    feb_path.parent.mkdir(parents=True, exist_ok=True)
    feb_path.write_bytes(b"garbage-bytes-not-parquet")
    with pytest.raises(DataIntegrityError, match="unreadable tape partition"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=run_feb, live_capture_root=live, evaluation_end=_ts("2023-03-01T00:00:00Z")), output_dir=tmp_path / "ofeb2")
    pd.DataFrame({"transact_time_ms": [1]}).to_parquet(feb_path)
    with pytest.raises(DataIntegrityError, match="missing columns"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=run_feb, live_capture_root=live, evaluation_end=_ts("2023-03-01T00:00:00Z")), output_dir=tmp_path / "ofeb3")
    bad = pd.DataFrame({"transact_time_ms": [int(feb_post.value // 10**6) + 5_000], "agg_trade_id": [1], "price": [float("nan")], "quantity": [1.0], "is_buyer_maker": [True]})
    bad.to_parquet(feb_path)
    with pytest.raises(DataIntegrityError, match="malformed tape slice"):
        run_fill_calibration(_request(tmp_path, paper_run_dir=run_feb, live_capture_root=live, evaluation_end=_ts("2023-03-01T00:00:00Z")), output_dir=tmp_path / "ofeb4")
    run_empty = _paper_run(
        tmp_path / "empty",
        [json.dumps({"event": "order_cancelled", "client_order_id": "ghost", "ts": feb_post.isoformat()}) + "\n"],
    )
    empty_report = run_fill_calibration(_request(tmp_path, paper_run_dir=run_empty, live_capture_root=live), output_dir=tmp_path / "oempty")
    assert empty_report.paper is not None
    assert empty_report.paper["queue_fallback_count"] == 0
