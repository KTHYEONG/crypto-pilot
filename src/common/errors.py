from __future__ import annotations


class DataIntegrityError(ValueError):
    """Fail-closed integrity error for missing, non-UTC, non-finite, or incomplete inputs.

    Raised whenever a canonical market-data input is incomplete or invalid.
    Missing costs (funding/borrow) are never replaced with zero-cost series.
    """

    pass
