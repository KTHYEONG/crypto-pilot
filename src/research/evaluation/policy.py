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
) -> str | pd.Timestamp | None:
    """Apply the single sealed-holdout policy shared by every evaluation CLI.

    Unless ``unseal_holdout`` is true, an explicit ``end`` past the sealed
    cutoff raises ``RuntimeError`` and a missing ``end`` defaults to the sealed
    cutoff. This is the only place the holdout decision is made, so every CLI
    uses the same sealed policy.
    """
    if unseal_holdout:
        return end
    if end is None:
        return HOLDOUT_CUTOFF
    end_ts = pd.Timestamp(end, tz="UTC")
    if end_ts > HOLDOUT_CUTOFF:
        raise RuntimeError(
            f"Holdout sealed: --end {end} > {HOLDOUT_CUTOFF}. "
            "Pass --unseal-holdout to override."
        )
    return end
