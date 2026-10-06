from __future__ import annotations

import hashlib
import inspect

import pytest

from src.quant.universe.pit_universe import symbol_partition


class TestSymbolPartition:
    # GEV2-04-HOLDOUT-STABLE
    def test_pre_registered_examples(self) -> None:
        assert symbol_partition("BTCUSDT") == "dev"
        assert symbol_partition("ETHUSDT") == "holdout"
        assert symbol_partition("SOLUSDT") == "dev"

    def test_deterministic_across_calls(self) -> None:
        for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"):
            assert symbol_partition(symbol) == symbol_partition(symbol)

    def test_hash_semantics(self) -> None:
        symbol = "AAAUSDT"
        bucket = int(hashlib.sha256(symbol.encode()).hexdigest()[:8], 16) % 100
        expected = "dev" if bucket < 80 else "holdout"
        assert symbol_partition(symbol) == expected

    def test_dev_share_is_close_to_dev_fraction(self) -> None:
        symbols = [f"SYM{i:03d}USDT" for i in range(200)]
        dev_share = sum(symbol_partition(s) == "dev" for s in symbols) / len(symbols)
        assert 0.70 <= dev_share <= 0.90

    def test_rejects_empty_symbol(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            symbol_partition("")


class TestNoLiteralStartDate:
    # GEV2-03-START-DERIVED: the module must derive its start, never hardcode one.
    def test_module_contains_no_literal_calendar_date(self) -> None:
        import re

        import src.quant.universe.pit_universe as pit_universe

        source = inspect.getsource(pit_universe)
        assert re.search(r"\b\d{4}-\d{2}-\d{2}\b", source) is None
