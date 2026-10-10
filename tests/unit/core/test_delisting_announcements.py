"""Invariant scenarios for delisting announcement evidence (spec 43 part 1)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.common.errors import DataIntegrityError
from src.core.delisting_announcements import (
    DelistingNotice,
    classify_notice,
    extract_contract_symbols,
    load_delisting_notices,
    parse_delisting_evidence,
    resolve_announcement,
)

_LAST = pd.Timestamp("2024-05-01T00:00:00Z")


def _notice(code: str, days_before: float, symbols: tuple[str, ...], kind: str = "delist") -> DelistingNotice:
    return DelistingNotice(
        code=code,
        title=f"{code} title",
        release_at=_LAST - pd.Timedelta(days=days_before),
        kind=kind,  # type: ignore[arg-type]
        symbols=symbols,
    )


def _row(code: str, release_ms: int, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "code": code,
        "title": f"title {code}",
        "release_ms": release_ms,
        "kind": "delist",
        "symbols": ["AAAUSDT"],
        "collected_at": "2024-05-02T00:00:00Z",
    }
    row.update(overrides)
    return row


def _write(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def test_earliest_qualifying_notice_wins() -> None:
    notices = (_notice("n-old", 6.0, ("AAAUSDT",)), _notice("n-new", 3.0, ("AAAUSDT",)))
    assert resolve_announcement("AAAUSDT", _LAST, notices) == notices[0]


def test_postponed_notice_never_resolves() -> None:
    notices = (_notice("n-post", 5.0, ("AAAUSDT",), kind="postponed"),)
    assert resolve_announcement("AAAUSDT", _LAST, notices) is None


@pytest.mark.parametrize("last_trade", [pd.NaT, pd.Timestamp("2024-05-01")])
def test_resolver_rejects_invalid_time(last_trade: pd.Timestamp) -> None:
    with pytest.raises(DataIntegrityError):
        resolve_announcement("AAAUSDT", last_trade, ())


def test_inclusive_window_and_code_tie_break() -> None:
    notices = (_notice("z", 21, ("AAAUSDT",)), _notice("a", 21, ("AAAUSDT",)))
    assert resolve_announcement("AAAUSDT", _LAST, notices) == notices[1]
    exact = _notice("now", 0, ("AAAUSDT",))
    assert resolve_announcement("AAAUSDT", _LAST, (exact,)) == exact


def test_notice_after_last_trade_is_rejected() -> None:
    late = DelistingNotice(
        code="n-late", title="late", release_at=_LAST + pd.Timedelta(days=20),
        kind="delist", symbols=("AAAUSDT",),
    )
    assert resolve_announcement("AAAUSDT", _LAST, (late,)) is None


def test_notice_older_than_21_days_is_rejected() -> None:
    assert resolve_announcement("AAAUSDT", _LAST, (_notice("n-old", 21.5, ("AAAUSDT",)),)) is None


def test_other_contract_of_same_notice_is_ignored() -> None:
    notices = (_notice("n-ab", 5.0, ("AAAUSDT", "BBBUSDT")),)
    assert resolve_announcement("AAAUSDT", _LAST, notices) == notices[0]
    assert resolve_announcement("BBBUSDT", _LAST, notices) == notices[0]
    assert resolve_announcement("CCCUSDT", _LAST, notices) is None


def test_release_time_rounds_up() -> None:
    (notice,) = parse_delisting_evidence(
        (json.dumps(_row("c1", 1_714_521_600_400)) + "\n").encode(),
        source="test",
    )
    assert notice.release_at == pd.Timestamp("2024-05-01T00:00:01Z")
    (exact,) = parse_delisting_evidence(
        (json.dumps(_row("c1", 1_714_521_600_000)) + "\n").encode(),
        source="test",
    )
    assert exact.release_at == pd.Timestamp("2024-05-01T00:00:00Z")


def test_symbol_extraction() -> None:
    title = "Binance Futures Will Delist USDⓈ-M XEMUSDT, ORBSUSDT and LOOMUSDT"
    body = "The 1000BONKUSDT perpetual contract will be settled and removed."
    assert extract_contract_symbols(title, body) == ("1000BONKUSDT", "LOOMUSDT", "ORBSUSDT", "XEMUSDT")


def test_classify_notice_kinds() -> None:
    assert classify_notice("Delisting postponed for AAAUSDT futures", "whatever") == "postponed"
    assert classify_notice(
        "Binance Futures Will Delist AAAUSDT", "The AAAUSDT perpetual contract settles soon.",
    ) == "delist"
    assert classify_notice("Binance Will Delist AAA on Spot", "Spot trading of AAA ends.") == "other"


def test_symbol_extraction_maps_base_tickers_in_delist_notice() -> None:
    title = "Binance Futures Will Delist CVC and AMP"
    body = "The CVC USDT and USDⓈ-M AMP perpetual contracts will be settled and removed."
    assert extract_contract_symbols(title, body) == ("AMPUSDT", "CVCUSDT")


def test_coin_m_quote_is_not_mapped_to_usdt() -> None:
    assert extract_contract_symbols("Futures Will Delist USDⓈ-M AAAUSDC", "") == ("AAAUSDC",)


def test_single_ticker_before_perpetual_contract() -> None:
    assert extract_contract_symbols(
        "Binance Futures Will Delist USDT-Margined ANC Perpetual Contract", "",
    ) == ("ANCUSDT",)


def test_ticker_list_before_usdt_margined_contracts() -> None:
    assert extract_contract_symbols(
        "Binance Futures Will Delist 1000BTTC and YFII USDT-Margined Contracts",
        "Automatic settlements on the 1000BTTC and YFII USDT-Margined Contracts.",
    ) == ("1000BTTCUSDT", "YFIIUSDT")


def test_mixed_full_symbol_and_bare_ticker() -> None:
    assert extract_contract_symbols(
        "Binance Futures Will Delist USDⓈ-M DEFIUSDT and MEMEFI Perpetual Contracts (2025-08-11)",
        "USDⓈ-M DEFIUSDT and MEMEFI perpetual contracts will be settled.",
    ) == ("DEFIUSDT", "MEMEFIUSDT")


def test_full_symbol_list_still_extracted() -> None:
    assert extract_contract_symbols(
        "Binance Futures Will Delist USDⓈ-M XEMUSDT, ORBSUSDT and LOOMUSDT Perpetual Contracts",
        "The XEMUSDT, ORBSUSDT and LOOMUSDT perpetual contracts will be settled.",
    ) == ("LOOMUSDT", "ORBSUSDT", "XEMUSDT")


def test_spot_notice_yields_nothing() -> None:
    assert extract_contract_symbols(
        "Binance Will Delist AERGO, AST, BURGER, COMBO, LINA on 2025-03-28",
        "Spot trading of AERGO, AST, BURGER, COMBO and LINA ends.",
    ) == ()


def test_vote_batch_yields_nothing() -> None:
    assert extract_contract_symbols(
        "First Batch of Vote to Delist Results and Will Delist BADGER, BAL, CREAM on 2025-04-01",
        "Spot trading of BADGER, BAL and CREAM ends following the vote.",
    ) == ()


def test_coin_m_only_notice_yields_nothing() -> None:
    assert extract_contract_symbols(
        "Binance Futures Will Delist COIN-M LUNA Perpetual Contracts",
        "The COIN-M LUNA perpetual contracts will be settled and removed.",
    ) == ()


def test_bare_margin_tag_without_ticker_yields_nothing() -> None:
    assert extract_contract_symbols(
        "Binance Futures Will Delist USDT-Margined Perpetual Contract",
        "The USDT-Margined perpetual contract will be settled.",
    ) == ()


def test_stop_word_never_becomes_ticker_and_usd_ticker_is_supported() -> None:
    assert extract_contract_symbols(
        "Binance Futures Will Delist COIN USDT-Margined Contracts",
        "The COIN USDT-Margined contracts will be settled.",
    ) == ()
    assert extract_contract_symbols(
        "Binance Futures Will Delist USDT-Margined ABCUSD Perpetual Contract",
        "The USDT-Margined ABCUSD perpetual contract will be settled.",
    ) == ("ABCUSDUSDT",)


@pytest.mark.parametrize("margin", ["USDT-Margined", "USDⓈ-M", "USDⓈ-Margined", "USDT-M"])
@pytest.mark.parametrize("separator", [", ", " and ", " & ", ", and "])
def test_margin_lists_preserve_long_full_symbols(margin: str, separator: str) -> None:
    title = f"Binance Futures Will Delist {margin} ABCDEFGHIJKLMNOUSDT{separator}ANC Perpetual Contracts"
    assert extract_contract_symbols(title, "") == ("ABCDEFGHIJKLMNOUSDT", "ANCUSDT")


def test_body_only_extraction_requires_action_in_same_sentence() -> None:
    title = "Binance Futures Will Delist Multiple Contracts"
    body = (
        "We will settle USDT-Margined ANC and SC perpetual contracts. "
        "USDⓈ-M OTHER perpetual contracts remain available. "
        "USDⓈ-M PROSE is unrelated. "
        "FAKE USDT balances remain available."
    )
    assert extract_contract_symbols(title, body) == ("ANCUSDT", "SCUSDT")


def test_margin_prefix_without_contract_phrase_is_ignored() -> None:
    assert extract_contract_symbols("Futures Will Delist USDⓈ-M PROSE", "") == ()


def test_coin_margin_and_overlong_ticker_do_not_create_usdt_symbols() -> None:
    assert extract_contract_symbols(
        "Futures Will Delist COIN-Margined ANC Perpetual Contracts", "",
    ) == ()
    assert extract_contract_symbols(
        "Futures Will Delist USDT-Margined ABCDEFGHIJKLMNOP Perpetual Contracts", "",
    ) == ()


def test_classification_combines_title_and_body_context() -> None:
    assert classify_notice("Binance Will Delist AAAUSDT", "Perpetual contracts end.") == "delist"
    assert classify_notice("Update", "x" * 2000 + " delist perpetual") == "other"
    assert classify_notice("Delisting postponed for AAAUSDT futures", "whatever") == "postponed"
    assert classify_notice(
        "Binance Futures Will Delist AAAUSDT", "The AAAUSDT perpetual contract settles soon.",
    ) == "delist"
    assert classify_notice("Binance Will Delist AAA on Spot", "Spot trading of AAA ends.") == "other"


def test_loader_accepts_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_bytes(b"")
    assert load_delisting_notices(path) == ()


def test_default_path_points_at_committed_policy() -> None:
    from src.core.delisting_announcements import default_delisting_evidence_path

    assert default_delisting_evidence_path().parts[-3:] == ("core", "policy", "delisting_announcements.jsonl")


def test_loader_rejects_malformed_rows(tmp_path: Path) -> None:
    base = _row("c1", 1000)
    cases = [
        {**base, "code": ""},
        {**base, "title": "  "},
        {**base, "release_ms": -1},
        {**base, "release_ms": True},
        {**base, "kind": "bogus"},
        {**base, "kind": []},
        {**base, "symbols": ["aaaUSDT"]},
        {**base, "symbols": ["BBBUSDT", "AAAUSDT"]},
        {**base, "symbols": ["AAAUSDT", "AAAUSDT"]},
        {**base, "symbols": ["USDT"]},
        {**base, "symbols": ["AAAUSDT", "USDC"]},
        {**base, "collected_at": ""},
        {**base, "collected_at": "not-a-time"},
        {**base, "collected_at": "NaT"},
        {**base, "collected_at": "2024-05-02T01:00:00+01:00"},
        {**base, "release_ms": 10**30},
        {**base, "extra": 1},
    ]
    for index, bad in enumerate(cases):
        path = tmp_path / f"bad{index}.jsonl"
        path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
        with pytest.raises(DataIntegrityError):
            load_delisting_notices(path)


def test_loader_rejects_non_object_and_bad_bytes(tmp_path: Path) -> None:
    non_object = tmp_path / "arr.jsonl"
    non_object.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        load_delisting_notices(non_object)
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text("{oops\n", encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        load_delisting_notices(malformed)
    binary = tmp_path / "binary.jsonl"
    binary.write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(DataIntegrityError):
        load_delisting_notices(binary)


def test_loader_rejects_multiple_objects_on_one_line(tmp_path: Path) -> None:
    path = tmp_path / "joined.jsonl"
    path.write_text(json.dumps(_row("c1", 1000)) + json.dumps(_row("c2", 2000)), encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        load_delisting_notices(path)


def test_loader_fails_closed(tmp_path: Path) -> None:
    dup = tmp_path / "dup.jsonl"
    _write(dup, [_row("c1", 1000), _row("c1", 2000)])
    with pytest.raises(DataIntegrityError):
        load_delisting_notices(dup)
    unsorted = tmp_path / "unsorted.jsonl"
    _write(unsorted, [_row("c2", 2000), _row("c1", 1000)])
    with pytest.raises(DataIntegrityError):
        load_delisting_notices(unsorted)
    missing_key = tmp_path / "missing.jsonl"
    row = _row("c1", 1000)
    del row["kind"]
    missing_key.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        load_delisting_notices(missing_key)
    with pytest.raises(DataIntegrityError):
        load_delisting_notices(tmp_path / "absent.jsonl")
