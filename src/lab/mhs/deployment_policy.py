"""Live-parity blocker registry and the typed exposure-sizing policy consumed by the scaling layer."""

from __future__ import annotations

from dataclasses import dataclass

from src.lab.mhs.contracts import MhsDiagnosticRequest

LIVE_UNSUPPORTED_REQUEST_FLAGS: dict[str, str] = {"name_drift_trim": "live daemon has no intraday trim loop"}


def live_parity_blockers(config: MhsDiagnosticRequest | None) -> tuple[str, ...]:
    """Return sorted live-parity blocker flag names set on ``config``."""
    if config is None:
        return ()
    return tuple(sorted(k for k in LIVE_UNSUPPORTED_REQUEST_FLAGS if bool(getattr(config, k, False))))


@dataclass(frozen=True, slots=True)
class SizingPolicy:
    """Typed exposure-sizing policy consumed by ``src.lab.mhs.scaling``; bounds are validated at construction."""
    mode: str
    target_annual_vol: float
    exposure_cap: float
    scale_floor: float
    kelly_enabled: bool
    kelly_window_days: int
    kelly_fraction: float
    kelly_lcb_z: float
    kelly_blend_weight: float
    drawdown_brake: bool

    def __post_init__(self) -> None:
        if not float(self.target_annual_vol) > 0:
            raise ValueError(f"target_annual_vol must be > 0, got {self.target_annual_vol}")
        if not float(self.exposure_cap) >= 1.0:
            raise ValueError(f"exposure_cap must be >= 1.0, got {self.exposure_cap}")
        if not 0.0 < float(self.scale_floor) <= 1.0:
            raise ValueError(f"scale_floor must be in (0, 1], got {self.scale_floor}")
        if not int(self.kelly_window_days) >= 1:
            raise ValueError(f"kelly_window_days must be >= 1, got {self.kelly_window_days}")
        if not 0.0 < float(self.kelly_fraction) <= 0.5:
            raise ValueError(f"kelly_fraction must be in (0, 0.5], got {self.kelly_fraction}")
        if not float(self.kelly_lcb_z) >= 0:
            raise ValueError(f"kelly_lcb_z must be >= 0, got {self.kelly_lcb_z}")
        if not 0.0 <= float(self.kelly_blend_weight) <= 1.0:
            raise ValueError(f"kelly_blend_weight must be in [0, 1], got {self.kelly_blend_weight}")
