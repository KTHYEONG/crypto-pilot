from __future__ import annotations


def intrabar_audit_required(
    *,
    competing_intrabar_exits: int,
    stop_atr_mult: float,
    gap_free_atr_mult: float = 1.5,
) -> bool:
    """Encode the measured intrabar-audit invariant, not a preference.

    1m replay is required only when a single 4h bar can contain more than one
    competing exit (``competing_intrabar_exits >= 2``) or when the stop sits
    inside the measured gap-through zone (``stop_atr_mult < gap_free_atr_mult``).
    The default ``gap_free_atr_mult = 1.5`` is the measured boundary at which the
    gap-through rate reaches 0.00 percent over 587 replayed stop events.
    """
    if competing_intrabar_exits < 0:
        raise ValueError(
            f"competing_intrabar_exits must be >= 0, got {competing_intrabar_exits}"
        )
    if stop_atr_mult <= 0:
        raise ValueError(f"stop_atr_mult must be > 0, got {stop_atr_mult}")
    return competing_intrabar_exits >= 2 or stop_atr_mult < gap_free_atr_mult


def _check_contract() -> None:
    """Executable assertions locking the frozen intrabar-audit contract surface."""
    assert intrabar_audit_required(competing_intrabar_exits=1, stop_atr_mult=3.0) is False
    assert intrabar_audit_required(competing_intrabar_exits=2, stop_atr_mult=3.0) is True
    assert intrabar_audit_required(competing_intrabar_exits=1, stop_atr_mult=1.0) is True


_check_contract()
