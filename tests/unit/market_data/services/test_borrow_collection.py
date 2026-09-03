"""P4 path-presence pin for the moved borrow-collection service.

Behavioral coverage lives in
``tests/unit/market_data/services/test_spot_collection.py``.
"""

from __future__ import annotations

import src.market_data.services.borrow_collection as borrow_collection


def test_borrow_collection_module_present() -> None:
    assert borrow_collection.__name__ == "src.market_data.services.borrow_collection"
    assert callable(borrow_collection.collect_binance_quote_borrow_history)
    assert callable(borrow_collection.import_quote_borrow_history)
