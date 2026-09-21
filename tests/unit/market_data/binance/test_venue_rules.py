from __future__ import annotations

import argparse
import copy
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.market_data.binance.venue_rules import (
    fetch_venue_rules,
    latest_venue_rule_snapshot,
    load_venue_rule_snapshot,
    parse_venue_rules,
    write_venue_rule_snapshot,
)

CAPTURED_AT = pd.Timestamp("2026-09-21T00:00:00Z")


def _bracket_payload() -> Any:
    return [
        {
            "symbol": "BTCUSDT",
            "brackets": [
                {
                    "bracket": 1,
                    "initialLeverage": 125,
                    "notionalCap": 50000,
                    "notionalFloor": 0,
                    "maintMarginRatio": 0.004,
                    "cum": 0,
                },
                {
                    "bracket": 2,
                    "initialLeverage": 100,
                    "notionalCap": 250000,
                    "notionalFloor": 50000,
                    "maintMarginRatio": 0.005,
                    "cum": 50,
                },
            ],
        },
        {
            "symbol": "ETHUSDT",
            "brackets": [
                {
                    "bracket": 2,
                    "initialLeverage": 75,
                    "notionalCap": 100000,
                    "notionalFloor": 10000,
                    "maintMarginRatio": 0.01,
                    "cum": 50,
                },
                {
                    "bracket": 1,
                    "initialLeverage": 100,
                    "notionalCap": 10000,
                    "notionalFloor": 0,
                    "maintMarginRatio": 0.005,
                    "cum": 0,
                },
            ],
        },
    ]


def _exchange_info_payload() -> Any:
    return {
        "symbols": [
            {
                "symbol": "BTCUSDT",
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                    {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001", "maxQty": "1000"},
                    {"filterType": "MIN_NOTIONAL", "notional": 5},
                ],
            },
            {
                "symbol": "ETHUSDT",
                "filters": [
                    {"filterType": "LOT_SIZE", "stepSize": "0.01", "minQty": "0.01", "maxQty": "10000"},
                    {"filterType": "NOTIONAL", "minNotional": "5"},
                ],
            },
            {"symbol": "DOGEUSDT"},
        ]
    }


def test_parse_venue_rules_joins_brackets_and_filters() -> None:
    """Parse joins brackets and filters; brackets sorted with maint_amount from cum."""
    snapshot = parse_venue_rules(_bracket_payload(), _exchange_info_payload(), captured_at=CAPTURED_AT)

    assert snapshot.captured_at == pd.Timestamp("2026-09-21T00:00:00Z", tz="UTC")
    btc = snapshot.symbols["BTCUSDT"]
    assert [t.notional_floor for t in btc.brackets] == [0, 50000]
    assert btc.brackets[1].maint_amount == 50
    assert btc.brackets[1].maint_margin_ratio == 0.005
    assert btc.brackets[1].initial_leverage == 100
    assert btc.step_size == 0.001
    assert btc.min_notional == 5
    eth = snapshot.symbols["ETHUSDT"]
    assert [t.notional_floor for t in eth.brackets] == [0, 10000]
    assert eth.step_size == 0.01
    assert eth.min_notional == 5


def test_parse_venue_rules_keeps_null_filters_for_bracket_only_symbol() -> None:
    """Bracket-only symbol keeps null filters."""
    brackets = _bracket_payload()
    brackets.append(
        {
            "symbol": "SOLUSDT",
            "brackets": [
                {
                    "bracket": 1,
                    "initialLeverage": 50,
                    "notionalCap": 10000,
                    "notionalFloor": 0,
                    "maintMarginRatio": 0.01,
                    "cum": 0,
                }
            ],
        }
    )

    snapshot = parse_venue_rules(brackets, _exchange_info_payload(), captured_at=CAPTURED_AT)

    assert snapshot.symbols["SOLUSDT"].step_size is None
    assert snapshot.symbols["SOLUSDT"].min_notional is None


def test_parse_venue_rules_drops_exchange_info_only_symbol() -> None:
    """ExchangeInfo-only symbol dropped."""
    info = _exchange_info_payload()
    info["symbols"].append(
        {
            "symbol": "XRPUSDT",
            "filters": [
                {"filterType": "LOT_SIZE", "stepSize": "1"},
                {"filterType": "MIN_NOTIONAL", "notional": 5},
            ],
        }
    )

    snapshot = parse_venue_rules(_bracket_payload(), info, captured_at=CAPTURED_AT)

    assert "XRPUSDT" not in snapshot.symbols
    assert "DOGEUSDT" not in snapshot.symbols


def test_parse_venue_rules_rejects_non_contiguous_ladder() -> None:
    """Non-contiguous ladder rejected."""
    brackets = _bracket_payload()
    brackets[0]["brackets"][1]["notionalFloor"] = 6000

    with pytest.raises(DataIntegrityError):
        parse_venue_rules(brackets, _exchange_info_payload(), captured_at=CAPTURED_AT)


def test_parse_venue_rules_rejects_missing_maint_margin_ratio() -> None:
    """Missing maintMarginRatio rejected."""
    brackets = _bracket_payload()
    del brackets[0]["brackets"][0]["maintMarginRatio"]

    with pytest.raises(DataIntegrityError):
        parse_venue_rules(brackets, _exchange_info_payload(), captured_at=CAPTURED_AT)


def test_parse_venue_rules_rejects_malformed_payloads() -> None:
    """Malformed payloads, missing keys, and non-positive ratios/leverage rejected."""
    missing_cum = _bracket_payload()
    del missing_cum[0]["brackets"][0]["cum"]
    bad_leverage = _bracket_payload()
    bad_leverage[1]["brackets"][0]["initialLeverage"] = 0
    bad_ratio = _bracket_payload()
    bad_ratio[1]["brackets"][0]["maintMarginRatio"] = -0.01
    info = _exchange_info_payload()
    cases: list[tuple[Any, Any]] = [
        ("not-a-list", info),
        ([{"symbol": "BTCUSDT"}], info),
        ([{"symbol": "BTCUSDT", "brackets": []}], info),
        ([{"brackets": []}], info),
        (_bracket_payload(), {"symbols": [{"filters": []}]}),
        (_bracket_payload(), {"no-symbols": []}),
        (_bracket_payload(), "not-a-dict"),
        ([{"symbol": "BTCUSDT", "brackets": ["oops"]}], info),
        (["oops"], info),
        (missing_cum, info),
        (bad_leverage, info),
        (bad_ratio, info),
    ]
    for bracket, exchange in cases:
        with pytest.raises(DataIntegrityError):
            parse_venue_rules(bracket, exchange, captured_at=CAPTURED_AT)


def test_venue_rule_snapshot_round_trip(tmp_path: Path) -> None:
    """Snapshot round-trips."""
    brackets = _bracket_payload()
    brackets.append(
        {
            "symbol": "SOLUSDT",
            "brackets": [
                {
                    "bracket": 1,
                    "initialLeverage": 50,
                    "notionalCap": 10000,
                    "notionalFloor": 0,
                    "maintMarginRatio": 0.01,
                    "cum": 0,
                }
            ],
        }
    )
    snapshot = parse_venue_rules(brackets, _exchange_info_payload(), captured_at=CAPTURED_AT)

    loaded = load_venue_rule_snapshot(write_venue_rule_snapshot(snapshot, tmp_path))

    assert loaded == snapshot


def test_write_venue_rule_snapshot_rejects_same_day_rewrite(tmp_path: Path) -> None:
    """Daily snapshot fresh-only."""
    snapshot = parse_venue_rules(_bracket_payload(), _exchange_info_payload(), captured_at=CAPTURED_AT)
    write_venue_rule_snapshot(snapshot, tmp_path)

    with pytest.raises(FileExistsError):
        write_venue_rule_snapshot(snapshot, tmp_path)


def test_latest_venue_rule_snapshot_selects_newest(tmp_path: Path) -> None:
    """Latest snapshot selection."""
    first = parse_venue_rules(_bracket_payload(), _exchange_info_payload(), captured_at=CAPTURED_AT)
    second = parse_venue_rules(
        _bracket_payload(), _exchange_info_payload(), captured_at=pd.Timestamp("2026-09-22T00:00:00Z")
    )
    first_path = write_venue_rule_snapshot(first, tmp_path)
    second_path = write_venue_rule_snapshot(second, tmp_path)

    assert latest_venue_rule_snapshot(tmp_path) == second_path
    assert first_path.name == "20260921.json"


def test_latest_venue_rule_snapshot_missing(tmp_path: Path) -> None:
    """No snapshot exists."""
    with pytest.raises(FileNotFoundError):
        latest_venue_rule_snapshot(tmp_path)


class _StubResponse:
    def __init__(self, payload: Any) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> _StubResponse:
        return self

    def __exit__(self, *args: Any) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


def test_fetch_venue_rules_signs_bracket_request_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fetch signs only the bracket request."""
    calls: list[tuple[str, str, dict[str, str]]] = []
    brackets = _bracket_payload()
    info = _exchange_info_payload()

    def fake_urlopen(target: Any, timeout: float | None = None) -> _StubResponse:
        if isinstance(target, urllib.request.Request):
            calls.append(("signed", target.full_url, dict(target.header_items())))
            assert "signature=" in target.full_url
            return _StubResponse(brackets)
        calls.append(("public", str(target), {}))
        assert "signature=" not in str(target)
        return _StubResponse(info)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    credential = "test-credential"
    snapshot = fetch_venue_rules(api_key=credential, api_secret=credential)

    assert [kind for kind, _, _ in calls] == ["signed", "public"]
    assert any(key.lower() == "x-mbx-apikey" for key in calls[0][2])
    assert set(snapshot.symbols) == {"BTCUSDT", "ETHUSDT"}


def test_fetch_venue_rules_requires_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing credentials rejected."""
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_SECRET_KEY", raising=False)

    with pytest.raises(RuntimeError):
        fetch_venue_rules()


def test_fetch_venue_rules_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """HTTP failure rejected."""
    def fail_bracket(target: Any, timeout: float | None = None) -> _StubResponse:
        raise urllib.error.URLError("boom")

    monkeypatch.setattr(urllib.request, "urlopen", fail_bracket)
    credential = "test-credential"
    with pytest.raises(RuntimeError):
        fetch_venue_rules(api_key=credential, api_secret=credential)

    def fail_info(target: Any, timeout: float | None = None) -> _StubResponse:
        if isinstance(target, urllib.request.Request):
            return _StubResponse(_bracket_payload())
        raise urllib.error.URLError("boom")

    monkeypatch.setattr(urllib.request, "urlopen", fail_info)
    with pytest.raises(RuntimeError):
        fetch_venue_rules(api_key=credential, api_secret=credential)


def test_collect_venue_rules_registers_and_collects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI wiring: venue-rules subcommand fetches once and is idempotent within a day."""
    import src.cli.commands.data as data_mod
    import src.market_data.binance.venue_rules as venue_mod

    parser = argparse.ArgumentParser()
    data_mod.add_data_commands(parser.add_subparsers(dest="group", required=True).add_parser("data"))
    args = parser.parse_args(["data", "collect", "venue-rules"])
    assert args.handler is data_mod._venue_rules

    monkeypatch.setattr("src.common.paths.VENUE_RULES_DIR", tmp_path)
    snapshot = parse_venue_rules(_bracket_payload(), _exchange_info_payload(), captured_at=CAPTURED_AT)
    monkeypatch.setattr(venue_mod, "fetch_venue_rules", lambda **kwargs: snapshot)

    data_mod._venue_rules(argparse.Namespace())
    printed = capsys.readouterr().out
    assert "20260921.json" in printed
    assert (tmp_path / "20260921.json").exists()

    def _must_not_fetch(**kwargs: Any) -> Any:
        raise AssertionError("fetch must not run when today is already captured")

    monkeypatch.setattr(venue_mod, "fetch_venue_rules", _must_not_fetch)
    data_mod._venue_rules(argparse.Namespace())
    assert (tmp_path / "20260921.json").exists()


def test_venue_rules_snapshot_json_holds_parsed_fields_only(tmp_path: Path) -> None:
    """JSON on disk stores the parsed fields only, captured_at ISO UTC."""
    snapshot = parse_venue_rules(_bracket_payload(), _exchange_info_payload(), captured_at=CAPTURED_AT)
    path = write_venue_rule_snapshot(snapshot, tmp_path)

    raw = json.loads(path.read_text(encoding="utf-8"))

    assert raw["captured_at"] == snapshot.captured_at.isoformat()
    assert set(raw) == {"captured_at", "symbols"}
    assert set(raw["symbols"]["BTCUSDT"]) == {"brackets", "step_size", "min_notional"}
    assert set(raw["symbols"]["BTCUSDT"]["brackets"][0]) == {
        "notional_floor",
        "notional_cap",
        "maint_margin_ratio",
        "maint_amount",
        "initial_leverage",
    }


def test_parse_venue_rules_accepts_maint_amount_alias() -> None:
    """maintAmount alias accepted as maint_amount source."""
    brackets = copy.deepcopy(_bracket_payload())
    for row in brackets:
        for tier in row["brackets"]:
            tier["maintAmount"] = tier.pop("cum")

    snapshot = parse_venue_rules(brackets, _exchange_info_payload(), captured_at=pd.Timestamp("2026-09-21"))

    assert snapshot.symbols["BTCUSDT"].brackets[1].maint_amount == 50
