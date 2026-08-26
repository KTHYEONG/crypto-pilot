from __future__ import annotations

import pandas as pd

# End of the observation window (spec section 3.2). Note the 23:59:59 boundary:
# load_ohlcv_4h filters "index <= end", and a bare "2025-12-31" parses to
# 00:00:00, which would drop the last 5 bars of that day.
HOLDOUT_CUTOFF = pd.Timestamp("2025-12-31 23:59:59", tz="UTC")


def resolve_evaluation_end(
    end: str | pd.Timestamp | None,
    *,
    unseal_holdout: bool,
    ceiling: pd.Timestamp = HOLDOUT_CUTOFF,
) -> pd.Timestamp:
    """Apply the single sealed-holdout policy shared by every evaluation CLI.

    Returns a tz-aware UTC ``pd.Timestamp`` on every branch -- never ``None``
    nor the raw input -- so every consumer sees one canonical window key. A
    missing ``end`` resolves to ``ceiling``; an explicit ``end`` past the
    active limit raises ``RuntimeError``. The limit is ``ceiling`` when
    ``unseal_holdout`` is true and the sealed ``HOLDOUT_CUTOFF`` otherwise,
    so the default path keeps its exact historical message.
    """
    if end is None:
        return ceiling
    end_ts = pd.Timestamp(end, tz="UTC")
    limit = ceiling if unseal_holdout else HOLDOUT_CUTOFF
    if end_ts > limit:
        raise RuntimeError(
            f"Holdout sealed: --end {end} > {limit}. "
            "Pass --unseal-holdout to override."
        )
    return end_ts
