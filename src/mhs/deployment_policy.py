"""Immutable deployable MHS policy: single typed conversion seam (I1)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from src.mhs.contracts import MhsDiagnosticRequest


@dataclass(frozen=True, slots=True)
class TargetWeightPolicy:
    execution_timeframe: str
    execution_universe_size: int
    fast_book_mode: str
    slow_book_mode: str
    rebalance_filter: str
    beta_neutralize: bool
    ensemble_signal: str
    trend_efficiency_overlay: bool
    trend_sleeve: bool
    trend_sleeve_gross: float
    crash_regime_tilt_alpha: float | None
    committee_capital: bool
    committee_member_set: str
    committee_tranche_smoothing: bool
    committee_regime_adaptive_tranche: bool
    committee_target_gross: float | None
    funding_carry_sleeve: bool
    funding_carry_weight: float
    fill_mark_parity_gate: bool

    def to_request(self) -> MhsDiagnosticRequest:
        from src.mhs.contracts import MhsDiagnosticRequest

        capital = bool(self.committee_capital)
        return MhsDiagnosticRequest(
            execution_timeframe=self.execution_timeframe,  # type: ignore[arg-type]
            execution_universe_size=int(self.execution_universe_size),
            fast_book_mode=self.fast_book_mode,  # type: ignore[arg-type]
            slow_book_mode=self.slow_book_mode,  # type: ignore[arg-type]
            rebalance_filter=self.rebalance_filter,  # type: ignore[arg-type]
            beta_neutralize=bool(self.beta_neutralize),
            ensemble_signal=self.ensemble_signal,  # type: ignore[arg-type]
            trend_efficiency_overlay=bool(self.trend_efficiency_overlay),
            trend_sleeve=bool(self.trend_sleeve),
            trend_sleeve_gross=float(self.trend_sleeve_gross),
            crash_regime_tilt_alpha=None if self.crash_regime_tilt_alpha is None else float(self.crash_regime_tilt_alpha),
            committee_capital=capital,
            committee_member_set=self.committee_member_set,  # type: ignore[arg-type]
            committee_tranche_smoothing=bool(self.committee_tranche_smoothing),
            committee_regime_adaptive_tranche=bool(self.committee_regime_adaptive_tranche),
            committee_target_gross=None if not capital else (None if self.committee_target_gross is None else float(self.committee_target_gross)),
            funding_carry_sleeve=bool(self.funding_carry_sleeve) and capital,
            funding_carry_weight=float(self.funding_carry_weight) if (bool(self.funding_carry_sleeve) and capital) else 0.0,
            fill_mark_parity_gate=bool(self.fill_mark_parity_gate),
        )


@dataclass(frozen=True, slots=True)
class SizingPolicy:
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


@dataclass(frozen=True, slots=True)
class SignalWindowPolicy:
    panel_window_days: int
    bootstrap_return_tail_days: int
    fold_panel_warmup_hours: int
    committee_purge_hours: int
    committee_oos_start: pd.Timestamp

    def __post_init__(self) -> None:
        if int(self.panel_window_days) <= 0:
            raise ValueError(f"panel_window_days must be > 0, got {self.panel_window_days}")
        if int(self.bootstrap_return_tail_days) <= 0:
            raise ValueError(f"bootstrap_return_tail_days must be > 0, got {self.bootstrap_return_tail_days}")
        if int(self.fold_panel_warmup_hours) <= 0:
            raise ValueError(f"fold_panel_warmup_hours must be > 0, got {self.fold_panel_warmup_hours}")
        if int(self.committee_purge_hours) <= 0:
            raise ValueError(f"committee_purge_hours must be > 0, got {self.committee_purge_hours}")
        ts = self.committee_oos_start if isinstance(self.committee_oos_start, pd.Timestamp) else pd.Timestamp(self.committee_oos_start)
        if ts.tzinfo is None:
            raise ValueError("committee_oos_start must be tz-aware UTC")
        utc = ts.tz_convert("UTC")
        object.__setattr__(self, "committee_oos_start", utc)


@dataclass(frozen=True, slots=True)
class MhsDeploymentPolicy:
    target_weights: TargetWeightPolicy
    sizing: SizingPolicy
    signal_window: SignalWindowPolicy
    slow_horizon_hours: int
    committee_member_weights: dict[str, float]
    admitted_members: tuple[str, ...]

    def __post_init__(self) -> None:
        admitted = tuple(self.admitted_members)
        if len(admitted) == 0:
            raise ValueError("admitted_members must be non-empty")
        object.__setattr__(self, "admitted_members", admitted)
        weights = dict(self.committee_member_weights)
        object.__setattr__(self, "committee_member_weights", weights)
        unknown = set(weights) - set(admitted)
        if unknown:
            raise ValueError(f"committee_member_weights has non-admitted members {sorted(unknown)}; admitted={sorted(admitted)}")
        total = 0.0
        for name, value in weights.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"committee_member_weights[{name!r}] must be finite and >= 0")
            total += float(value)
        if not total > 0.0:
            raise ValueError("committee_member_weights sum must be > 0")


def build_deployment_policy(
    request: MhsDiagnosticRequest,
    *,
    slow_horizon_hours: int,
    committee_member_weights: dict[str, float],
    admitted_members: tuple[str, ...],
    target_annual_vol: float,
    exposure_cap: float,
) -> MhsDeploymentPolicy:
    from src.mhs.params import (
        COMMITTEE_KELLY_FRACTION,
        COMMITTEE_KELLY_LCB_Z,
        COMMITTEE_KELLY_WINDOW_DAYS,
        COMMITTEE_OOS_START,
        COMMITTEE_PURGE_HOURS,
        FOLD_PANEL_WARMUP_HOURS,
        PNL_VOL_TARGET_SCALE_FLOOR,
        SIGNAL_PANEL_WINDOW_DAYS,
        SIGNAL_RETURN_TAIL_DAYS,
    )
    from src.mhs.research_go import _resolved_committee_target_gross

    target = TargetWeightPolicy(
        execution_timeframe=str(request.execution_timeframe),
        execution_universe_size=int(request.execution_universe_size),
        fast_book_mode=str(request.fast_book_mode),
        slow_book_mode=str(request.slow_book_mode),
        rebalance_filter=str(request.rebalance_filter),
        beta_neutralize=bool(request.beta_neutralize),
        ensemble_signal=str(request.ensemble_signal),
        trend_efficiency_overlay=bool(request.trend_efficiency_overlay),
        trend_sleeve=bool(request.trend_sleeve),
        trend_sleeve_gross=float(request.trend_sleeve_gross),
        crash_regime_tilt_alpha=None if request.crash_regime_tilt_alpha is None else float(request.crash_regime_tilt_alpha),
        committee_capital=bool(request.committee_capital),
        committee_member_set=str(request.committee_member_set),
        committee_tranche_smoothing=bool(request.committee_tranche_smoothing),
        committee_regime_adaptive_tranche=bool(request.committee_regime_adaptive_tranche),
        committee_target_gross=_resolved_committee_target_gross(request),
        funding_carry_sleeve=bool(request.funding_carry_sleeve),
        funding_carry_weight=float(request.funding_carry_weight),
        fill_mark_parity_gate=bool(request.fill_mark_parity_gate),
    )
    mode = str(request.pnl_vol_target_mode)
    kelly_enabled = bool(request.committee_capital and request.committee_kelly_sizing) and mode != "constant_risk"
    sizing = SizingPolicy(
        mode=mode,
        target_annual_vol=float(target_annual_vol),
        exposure_cap=float(exposure_cap),
        scale_floor=float(PNL_VOL_TARGET_SCALE_FLOOR),
        kelly_enabled=kelly_enabled,
        kelly_window_days=int(COMMITTEE_KELLY_WINDOW_DAYS),
        kelly_fraction=float(COMMITTEE_KELLY_FRACTION),
        kelly_lcb_z=float(COMMITTEE_KELLY_LCB_Z),
        kelly_blend_weight=0.5,
        drawdown_brake=bool(request.exposure_drawdown_brake),
    )
    window = SignalWindowPolicy(
        panel_window_days=int(SIGNAL_PANEL_WINDOW_DAYS),
        bootstrap_return_tail_days=int(SIGNAL_RETURN_TAIL_DAYS),
        fold_panel_warmup_hours=int(FOLD_PANEL_WARMUP_HOURS),
        committee_purge_hours=int(COMMITTEE_PURGE_HOURS),
        committee_oos_start=COMMITTEE_OOS_START,
    )
    return MhsDeploymentPolicy(
        target_weights=target,
        sizing=sizing,
        signal_window=window,
        slow_horizon_hours=int(slow_horizon_hours),
        committee_member_weights=dict(committee_member_weights),
        admitted_members=tuple(admitted_members),
    )
