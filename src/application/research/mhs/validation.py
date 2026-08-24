"""Metadata-driven request validator for ``MhsDiagnosticRequest`` (I3).

Validation rules derive from each field's ``cli_param`` metadata (``choices``,
``bounds``, ``requires``, ``excludes``) plus the field-specific predicates that
carry the exact historical ``ValueError`` message strings so the 56
``pytest.raises(ValueError, match=...)`` assertions keep passing verbatim.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.application.research.mhs.contracts import MhsDiagnosticRequest


def _choice_error(field: str, value: Any, choices: tuple[str, ...]) -> str:
    return f"unknown {field} '{value}'"


def _validate_field_choices(request: MhsDiagnosticRequest, field: str, choices: tuple[str, ...]) -> None:
    value = getattr(request, field)
    if value not in choices:
        raise ValueError(_choice_error(field, value, choices))


def _validate_field_bounds(request: MhsDiagnosticRequest, field: str, bounds: tuple[float, float]) -> None:
    value = getattr(request, field)
    if value is None:
        return
    lo, hi = bounds
    if not (lo <= value <= hi):
        raise ValueError(f"{field} must be in [{lo}, {hi}]")


def validate_request(request: MhsDiagnosticRequest, committee_target_gross_unset: object) -> None:
    """Run every declared validation rule over the request, fail closed.

    ``committee_target_gross_unset`` is the caller's ``COMMITTEE_TARGET_GROSS_UNSET``
    sentinel; its object identity distinguishes the registered default exposure
    from an explicit value (the sentinel is never resolved into the frozen
    field, so identity is preserved across ``dataclasses.replace``).
    """
    # Choice membership (closed sets).
    _validate_field_choices(request, "partition", ("dev", "holdout", "all"))
    _validate_field_choices(
        request, "mark_mode",
        ("cache_required", "cache_required_stale_carry", "ohlcv_close_fallback"),
    )
    _validate_field_choices(request, "execution_timeframe", ("1m", "3m", "5m"))
    if request.execution_universe_size < 8:
        raise ValueError("execution_universe_size must be >= 8")
    if request.max_rss_bytes is not None and request.max_rss_bytes <= 0:
        raise ValueError("max_rss_bytes must be > 0")
    if request.crash_regime_tilt_alpha is not None and not (
        0.0 < request.crash_regime_tilt_alpha <= 1.0
    ):
        raise ValueError(
            f"crash_regime_tilt_alpha must be in (0.0, 1.0] when set, "
            f"got {request.crash_regime_tilt_alpha}"
        )
    _validate_field_choices(request, "slow_book_mode", ("single_horizon", "horizon_ensemble"))
    _validate_field_choices(request, "fast_book_mode", ("single_horizon", "horizon_ensemble"))
    _validate_field_choices(
        request, "rebalance_filter", ("per_symbol_deadband", "portfolio_trigger"),
    )
    if request.discovery_gate_adjusted_net_t and not request.discovery_gate:
        raise ValueError("discovery_gate_adjusted_net_t requires discovery_gate=True")
    if request.discovery_gate_regime_scaled_net_t and not request.discovery_gate:
        raise ValueError("discovery_gate_regime_scaled_net_t requires discovery_gate=True")
    if not isinstance(request.beta_neutralize, bool):
        raise ValueError("beta_neutralize must be a bool")
    _validate_field_choices(request, "ensemble_signal", ("raw", "vol_normalized"))
    if not isinstance(request.trend_efficiency_overlay, bool):
        raise ValueError("trend_efficiency_overlay must be a bool")
    if not isinstance(request.pnl_vol_target, bool):
        raise ValueError("pnl_vol_target must be a bool")
    if not isinstance(request.trend_sleeve, bool):
        raise ValueError("trend_sleeve must be a bool")
    if not isinstance(request.multi_feature_book, bool):
        raise ValueError("multi_feature_book must be a bool")
    if not isinstance(request.committee_book, bool):
        raise ValueError("committee_book must be a bool")
    if not isinstance(request.committee_kelly_sizing, bool):
        raise ValueError("committee_kelly_sizing must be a bool")
    if request.committee_kelly_sizing and not (
        request.committee_book or request.committee_capital
    ):
        raise ValueError(
            "committee_kelly_sizing requires committee_book=True or committee_capital=True"
        )
    if not isinstance(request.committee_tranche_smoothing, bool):
        raise ValueError("committee_tranche_smoothing must be a bool")
    if request.committee_tranche_smoothing and not request.committee_capital:
        raise ValueError("committee_tranche_smoothing requires committee_capital=True")
    if not isinstance(request.committee_regime_adaptive_tranche, bool):
        raise ValueError("committee_regime_adaptive_tranche must be a bool")
    if request.committee_regime_adaptive_tranche:
        if not request.committee_capital:
            raise ValueError(
                "committee_regime_adaptive_tranche requires committee_capital=True"
            )
        if request.committee_tranche_smoothing:
            raise ValueError(
                "committee_regime_adaptive_tranche is mutually exclusive with "
                "committee_tranche_smoothing"
            )
    if not isinstance(request.committee_growth_diagnostic, bool):
        raise ValueError("committee_growth_diagnostic must be a bool")
    if request.committee_growth_diagnostic and not request.committee_book:
        raise ValueError("committee_growth_diagnostic requires committee_book=True")
    if not isinstance(request.committee_capital, bool):
        raise ValueError("committee_capital must be a bool")
    if not isinstance(request.committee_evidence_weighting, bool):
        raise ValueError("committee_evidence_weighting must be a bool")
    if request.committee_evidence_weighting and not request.committee_capital:
        raise ValueError("committee_evidence_weighting requires committee_capital=True")
    raw_target_gross = request.committee_target_gross
    if raw_target_gross is not committee_target_gross_unset and raw_target_gross is not None:
        if not (0.0 < raw_target_gross <= 2.0):
            raise ValueError("committee_target_gross must be in (0.0, 2.0] when set")
        if not request.committee_capital:
            raise ValueError("committee_target_gross requires committee_capital=True")
    if not isinstance(request.execution_coverage_gate, bool):
        raise ValueError("execution_coverage_gate must be a bool")
    if not isinstance(request.fill_mark_parity_gate, bool):
        raise ValueError("fill_mark_parity_gate must be a bool")
    if not isinstance(request.exposure_scale_two_sided, bool):
        raise ValueError("exposure_scale_two_sided must be a bool")
    if request.exposure_scale_two_sided and request.pnl_vol_target_mode not in (
        "exante_target", "growth_budget", "constant_risk",
    ):
        raise ValueError(
            "exposure_scale_two_sided requires pnl_vol_target_mode='exante_target', "
            "'growth_budget', or 'constant_risk'"
        )
    if not isinstance(request.exposure_drawdown_brake, bool):
        raise ValueError("exposure_drawdown_brake must be a bool")
    if request.exposure_drawdown_brake:
        if request.pnl_vol_target_mode != "constant_risk":
            raise ValueError(
                "exposure_drawdown_brake requires pnl_vol_target_mode='constant_risk'"
            )
        if not request.pnl_vol_target:
            raise ValueError("exposure_drawdown_brake requires pnl_vol_target=True")
    if not isinstance(request.ram_guard, bool):
        raise ValueError("ram_guard must be a bool")
    from src.mhs.params import GROWTH_RISK_ENVELOPES

    _validate_field_choices(
        request, "growth_envelope", tuple(sorted(GROWTH_RISK_ENVELOPES)),
    )
    if not isinstance(request.committee_member_attribution, bool):
        raise ValueError("committee_member_attribution must be a bool")
    if not (0.0 <= request.trend_sleeve_gross <= 1.0):
        raise ValueError("trend_sleeve_gross must be in [0.0, 1.0]")
    if request.trend_sleeve_gross > 0.0 and not request.trend_sleeve:
        raise ValueError("trend_sleeve_gross requires trend_sleeve=True")
    _validate_field_choices(
        request, "pnl_vol_target_mode",
        ("median_relative", "exante_target", "growth_budget", "constant_risk"),
    )
    _validate_field_choices(
        request, "committee_member_set", ("risk_premia", "flow_momentum"),
    )
    if not isinstance(request.funding_carry_sleeve, bool):
        raise ValueError("funding_carry_sleeve must be a bool")
    if request.funding_carry_sleeve and not request.committee_capital:
        raise ValueError("funding_carry_sleeve requires committee_capital=True")
    if request.funding_carry_sleeve and request.committee_target_gross is None:
        raise ValueError(
            "funding_carry_sleeve is mutually exclusive with "
            "committee_target_gross=None (the diluted book has no gross "
            "to normalize the mix against)"
        )
    if not (0.0 <= request.funding_carry_weight < 1.0):
        raise ValueError("funding_carry_weight must be in [0.0, 1.0)")
    if request.funding_carry_weight > 0.0 and not request.funding_carry_sleeve:
        raise ValueError("funding_carry_weight > 0.0 requires funding_carry_sleeve=True")
