from __future__ import annotations

import hashlib
from typing import Literal


def symbol_partition(symbol: str, dev_fraction: float = 0.80) -> Literal["dev", "holdout"]:
    """Deterministic pre-registered symbol split driven only by the symbol string.

    The bucket is ``int(sha256(symbol)[:8], 16) % 100``; a symbol lands in
    ``"dev"`` exactly when its bucket is below ``dev_fraction * 100``.  No
    returns, performance, or market data is ever consulted, so the split cannot
    be tuned to hide a failing signal.
    """
    if not symbol:
        raise ValueError("symbol must not be empty")
    if not 0 < dev_fraction < 1:
        raise ValueError(f"dev_fraction must be in (0, 1), got {dev_fraction}")
    bucket = int(hashlib.sha256(symbol.encode()).hexdigest()[:8], 16) % 100
    return "dev" if bucket < dev_fraction * 100 else "holdout"


def _check_contract() -> None:
    """Executable assertions locking the pre-registered symbol partition at import."""
    assert symbol_partition("BTCUSDT") == "dev"
    assert symbol_partition("ETHUSDT") == "holdout"
    assert symbol_partition("SOLUSDT") == "dev"
    assert symbol_partition("BTCUSDT") == symbol_partition("BTCUSDT")


_check_contract()
