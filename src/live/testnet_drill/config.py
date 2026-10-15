"""Typed drill-run parameters; every scenario threshold lives here."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import Decimal

_SYMBOL_RE = re.compile(r"[A-Z0-9]+")

_MAX_RECV_WINDOW_MS = 60_000


def _as_decimal(value: Decimal | str | int | float, name: str) -> Decimal:
    """Coerce a decimal-like setting to Decimal, rejecting non-finite values."""
    try:
        amount = value if isinstance(value, Decimal) else Decimal(str(value))
    except Exception as exc:
        raise ValueError(f"{name} must be decimal-like, got {value!r}") from exc
    if not amount.is_finite():
        raise ValueError(f"{name} must be finite")
    return amount


def _as_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int, got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class DrillConfig:
    """Typed, validated parameters of one drill run."""

    symbols: tuple[str, ...]
    probe_notional_usdt: Decimal
    resting_distance_bps: int
    max_probe_notional_usdt: Decimal
    max_open_orders_probe: int
    scenario_timeout_s: float
    recv_window_ms: int
    funding_soak_min_funding_events: int
    qty_tolerance: Decimal
    price_rel_tolerance: Decimal
    fee_abs_tolerance_usdt: Decimal
    slate_max_attempts: int = 3
    settle_wait_s: float = 5.0
    settle_poll_s: float = 0.2
    counterparty_step_wait_s: float = 0.2
    race_repetitions: int = 20
    race_offsets_s: tuple[float, ...] = (-0.05, 0.0, 0.05)

    def __post_init__(self) -> None:
        for name in ("settle_wait_s", "settle_poll_s", "counterparty_step_wait_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if _as_int(self.race_repetitions, "race_repetitions") < 1:
            raise ValueError("race_repetitions must be positive")
        if not self.race_offsets_s or any(not math.isfinite(offset) for offset in self.race_offsets_s):
            raise ValueError("race_offsets_s must be non-empty and finite")
        symbols = tuple(self.symbols)
        if not symbols:
            raise ValueError("symbols must be non-empty")
        if len(set(symbols)) != len(symbols):
            raise ValueError("symbols must be unique")
        for symbol in symbols:
            if not isinstance(symbol, str) or _SYMBOL_RE.fullmatch(symbol) is None:
                raise ValueError(f"symbols must be uppercase ASCII, got {symbol!r}")
        object.__setattr__(self, "symbols", symbols)
        probe = _as_decimal(self.probe_notional_usdt, "probe_notional_usdt")
        cap = _as_decimal(self.max_probe_notional_usdt, "max_probe_notional_usdt")
        if probe <= 0 or cap <= 0:
            raise ValueError("probe_notional_usdt and max_probe_notional_usdt must be positive")
        if probe > cap:
            raise ValueError("probe_notional_usdt must not exceed max_probe_notional_usdt")
        object.__setattr__(self, "probe_notional_usdt", probe)
        object.__setattr__(self, "max_probe_notional_usdt", cap)
        for name in ("qty_tolerance", "price_rel_tolerance", "fee_abs_tolerance_usdt"):
            amount = _as_decimal(getattr(self, name), name)
            if amount < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, amount)
        if _as_int(self.resting_distance_bps, "resting_distance_bps") <= 0:
            raise ValueError("resting_distance_bps must be positive")
        if _as_int(self.max_open_orders_probe, "max_open_orders_probe") < 1:
            raise ValueError("max_open_orders_probe must be at least 1")
        try:
            timeout_s = float(self.scenario_timeout_s)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"scenario_timeout_s must be numeric, got {self.scenario_timeout_s!r}") from exc
        if not math.isfinite(timeout_s) or not timeout_s > 0:
            raise ValueError("scenario_timeout_s must be positive")
        object.__setattr__(self, "scenario_timeout_s", timeout_s)
        if _as_int(self.recv_window_ms, "recv_window_ms") <= 0 or self.recv_window_ms > _MAX_RECV_WINDOW_MS:
            raise ValueError(f"recv_window_ms must be in (0, {_MAX_RECV_WINDOW_MS}]")
        if _as_int(self.funding_soak_min_funding_events, "funding_soak_min_funding_events") < 1:
            raise ValueError("funding_soak_min_funding_events must be at least 1")
        if _as_int(self.slate_max_attempts, "slate_max_attempts") < 1:
            raise ValueError("slate_max_attempts must be at least 1")
