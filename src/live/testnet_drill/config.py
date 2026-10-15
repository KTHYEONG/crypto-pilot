"""Typed drill-run parameters; every scenario threshold lives here."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import Decimal

from src.live.settings import _MAX_RECV_WINDOW_MS

_SYMBOL_RE = re.compile(r"[A-Z0-9]+")



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
    race_repetitions: int = 20
    race_offsets_s: tuple[float, ...] = (-0.05, 0.0, 0.05)
    retry_after_probe_s: int = 2
    recv_window_overshoot_s: float = 1.0
    snapshot_trials: int = 10
    funding_soak_s: float = 0.0
    funding_poll_s: float = 60.0
    restart_kill_timeout_s: float = 600.0

    def __post_init__(self) -> None:
        self._validate_timing()
        self._validate_symbols()
        self._validate_amounts()
        self._validate_counts()

    def _validate_timing(self) -> None:
        for name in _POSITIVE_DURATIONS:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.funding_soak_s) or self.funding_soak_s < 0:
            raise ValueError("funding_soak_s must be finite and non-negative")
        if not self.race_offsets_s or any(not math.isfinite(offset) for offset in self.race_offsets_s):
            raise ValueError("race_offsets_s must be non-empty and finite")
        try:
            timeout_s = float(self.scenario_timeout_s)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"scenario_timeout_s must be numeric, got {self.scenario_timeout_s!r}") from exc
        if not math.isfinite(timeout_s) or not timeout_s > 0:
            raise ValueError("scenario_timeout_s must be positive")
        object.__setattr__(self, "scenario_timeout_s", timeout_s)
        if _as_int(self.recv_window_ms, "recv_window_ms") <= 0 or self.recv_window_ms > _MAX_RECV_WINDOW_MS:
            raise ValueError(f"recv_window_ms must be in (0, {_MAX_RECV_WINDOW_MS}]")

    def _validate_symbols(self) -> None:
        symbols = tuple(self.symbols)
        if not symbols:
            raise ValueError("symbols must be non-empty")
        if len(set(symbols)) != len(symbols):
            raise ValueError("symbols must be unique")
        for symbol in symbols:
            if not isinstance(symbol, str) or _SYMBOL_RE.fullmatch(symbol) is None:
                raise ValueError(f"symbols must be uppercase ASCII, got {symbol!r}")
        object.__setattr__(self, "symbols", symbols)

    def _validate_amounts(self) -> None:
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

    def _validate_counts(self) -> None:
        if _as_int(self.resting_distance_bps, "resting_distance_bps") <= 0:
            raise ValueError("resting_distance_bps must be positive")
        if _as_int(self.max_open_orders_probe, "max_open_orders_probe") < 1:
            raise ValueError("max_open_orders_probe must be at least 1")
        if _as_int(self.funding_soak_min_funding_events, "funding_soak_min_funding_events") < 1:
            raise ValueError("funding_soak_min_funding_events must be at least 1")
        if _as_int(self.slate_max_attempts, "slate_max_attempts") < 1:
            raise ValueError("slate_max_attempts must be at least 1")
        for name in ("race_repetitions", "retry_after_probe_s", "snapshot_trials"):
            if _as_int(getattr(self, name), name) < 1:
                raise ValueError(f"{name} must be positive")


_POSITIVE_DURATIONS = (
    "settle_wait_s",
    "settle_poll_s",
    "recv_window_overshoot_s",
    "restart_kill_timeout_s",
    "funding_poll_s",
)


def config_to_json(config: DrillConfig) -> dict[str, object]:
    """JSON-safe field map; Decimals travel as strings so the child rebuilds them exactly."""
    from dataclasses import fields

    payload: dict[str, object] = {}
    for item in fields(config):
        value = getattr(config, item.name)
        payload[item.name] = str(value) if isinstance(value, Decimal) else list(value) if isinstance(value, tuple) else value
    return payload


def config_from_json(payload: dict[str, object]) -> DrillConfig:
    """Inverse of ``config_to_json``; validation runs through ``DrillConfig.__post_init__``."""
    values = dict(payload)
    for name in ("symbols", "race_offsets_s"):
        if name in values:
            values[name] = tuple(values[name])  # type: ignore[arg-type]
    return DrillConfig(**values)  # type: ignore[arg-type]
