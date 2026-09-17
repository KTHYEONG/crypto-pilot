"""Invariant scenarios for universe gap detection."""

from __future__ import annotations

import pytest

from src.market_data.services.universe_gaps import (
    historical_universe_gaps,
    local_futures_symbols,
    non_crypto_symbols,
)


def test_local_futures_symbols_excludes_temp_artifacts(tmp_path) -> None:
    lake = tmp_path / "1h"
    lake.mkdir()
    (lake / "AAAUSDT.parquet").write_bytes(b"fake")
    (lake / "BBBUSDT.tmp.parquet").write_bytes(b"fake")
    assert local_futures_symbols(tmp_path, "1h") == frozenset({"AAAUSDT"})


def test_historical_universe_gaps_dev_only() -> None:
    from src.quant.universe.pit_universe import symbol_partition

    dev_syms = [s for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT") if symbol_partition(s) == "dev"]
    holdout_syms = [s for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT") if symbol_partition(s) == "holdout"]
    assert dev_syms
    assert holdout_syms
    vision = dev_syms + holdout_syms + ["BTCBUSD"]
    gaps = historical_universe_gaps(vision, [], partition="dev")
    assert gaps == tuple(sorted(set(dev_syms)))
    assert all(g.endswith("USDT") for g in gaps)


def test_historical_universe_gaps_empty_raises() -> None:
    with pytest.raises(ValueError, match=r".+"):
        historical_universe_gaps([], [], partition="dev")
    with pytest.raises(ValueError, match=r".+"):
        historical_universe_gaps(["BTCUSDT"], [], partition="nope")


def test_historical_universe_gaps_all_includes_holdout() -> None:
    from src.quant.universe.pit_universe import symbol_partition

    syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT", "ADAUSDT"]
    holdout = [s for s in syms if symbol_partition(s) == "holdout"]
    assert holdout
    gaps = historical_universe_gaps(syms, [], partition="all")
    assert set(gaps) >= set(holdout)
    assert gaps == tuple(sorted(gaps))


def test_non_crypto_symbols_flags_non_coin_underlyings() -> None:
    info = {
        "symbols": [
            {"symbol": "BTCUSDT", "underlyingType": "COIN"},
            {"symbol": "TSLAUSDT", "underlyingType": "EQUITY"},
            {"symbol": "XAUUSDT", "underlyingType": "COMMODITY"},
            {"symbol": "DEFIUSDT", "underlyingType": "INDEX"},
            {"symbol": "SKHYNIXUSDT", "underlyingType": "KR_EQUITY"},
            {"symbol": "ETHUSDT"},  # underlyingType absent -> treated as COIN
        ]
    }
    excluded = non_crypto_symbols(info)
    assert excluded == frozenset({"TSLAUSDT", "XAUUSDT", "DEFIUSDT", "SKHYNIXUSDT"})
    assert "BTCUSDT" not in excluded
    assert "ETHUSDT" not in excluded


def test_non_crypto_symbols_missing_symbols_list_raises() -> None:
    with pytest.raises(ValueError, match=r".+"):
        non_crypto_symbols({})


def test_historical_universe_gaps_exclude_removes_non_crypto() -> None:
    from src.quant.universe.pit_universe import symbol_partition

    assert symbol_partition("BTCUSDT") == "dev"
    vision = ["BTCUSDT", "TSLAUSDT"]
    gaps = historical_universe_gaps(vision, [], partition="dev", exclude={"TSLAUSDT"})
    assert gaps == ("BTCUSDT",)


def test_fetch_exchange_info_parses_json_response(monkeypatch) -> None:
    import io
    import src.market_data.services.universe_gaps as universe_gaps

    class _Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    payload = b'{"symbols": [{"symbol": "BTCUSDT", "underlyingType": "COIN"}]}'
    captured: dict = {}

    def _fake_urlopen(url, timeout=None):
        captured["url"] = url
        captured["timeout"] = timeout
        return _Response(payload)

    monkeypatch.setattr(universe_gaps.urllib.request, "urlopen", _fake_urlopen)
    result = universe_gaps.fetch_exchange_info(timeout=5)
    assert result == {"symbols": [{"symbol": "BTCUSDT", "underlyingType": "COIN"}]}
    assert captured == {"url": universe_gaps.EXCHANGE_INFO_URL, "timeout": 5}
