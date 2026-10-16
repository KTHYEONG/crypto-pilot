from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import Any

import pandas as pd

from src.common.errors import DataIntegrityError
from src.live.errors import LiveTradingError


class StaleOrderIdentity(LiveTradingError):  # noqa: N818 - contract pins the name
    """A venue order lookup returned an older order reusing the queried client order id."""


def assert_order_identity(
    payload: Mapping[str, Any],
    *,
    client_order_id: str,
    submitted_at: pd.Timestamp,
    tolerance_ms: int,
) -> None:
    """Fail closed when ``payload`` predates its journal submit beyond ``tolerance_ms``."""
    raw = payload.get("time")
    if isinstance(raw, bool) or raw is None:
        raise DataIntegrityError(f"order {client_order_id} query has invalid creation time")
    try:
        venue_ms = Decimal(str(raw))
    except (InvalidOperation, ValueError):
        raise DataIntegrityError(f"order {client_order_id} query has invalid creation time") from None
    if not venue_ms.is_finite() or venue_ms <= 0 or venue_ms != venue_ms.to_integral_value():
        raise DataIntegrityError(f"order {client_order_id} query has invalid creation time")
    if pd.isna(submitted_at) or submitted_at.tzinfo is None or tolerance_ms <= 0:
        raise DataIntegrityError(f"order {client_order_id} has invalid identity guard context")
    submitted_ms = int(submitted_at.value // 1_000_000)
    if venue_ms < submitted_ms - int(tolerance_ms):
        raw_order_id = payload.get("orderId")
        order_id = str(raw_order_id) if str(raw_order_id).isascii() and str(raw_order_id).isdigit() else "unknown"
        raise StaleOrderIdentity(
            f"order {client_order_id} identity stale: venue time {venue_ms} "
            f"precedes submit {submitted_ms} beyond tolerance {int(tolerance_ms)}ms "
            f"(orderId={order_id})"
        )
