# ruff: noqa
def test_build_deployment_policy_is_single_typed_conversion_seam() -> None:
    import dataclasses
    import pandas as pd
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.deployment_policy import build_deployment_policy
    from src.mhs.pipeline.config import MhsRunConfig

    request = MhsDiagnosticRequest(**dataclasses.asdict(MhsRunConfig()))
    policy = build_deployment_policy(request, slow_horizon_hours=168, committee_member_weights={"flow_imb_720h": 0.4, "flow_imb_168h": 0.6}, admitted_members=("flow_imb_720h", "flow_imb_168h"), target_annual_vol=0.35, exposure_cap=3.0)
    restored = policy.target_weights.to_request()
    assert restored.execution_universe_size == request.execution_universe_size
    assert restored.committee_member_set == request.committee_member_set
    assert restored.committee_target_gross == request.committee_target_gross
    assert policy.sizing.target_annual_vol == 0.35
    assert policy.sizing.kelly_window_days == 42
    assert policy.sizing.kelly_lcb_z == 0.0
    assert policy.signal_window.committee_oos_start == pd.Timestamp(policy.signal_window.committee_oos_start).tz_convert("UTC")



def test_sizing_and_deployment_policy_reject_invalid_financial_bounds() -> None:
    import pytest
    from src.mhs.deployment_policy import MhsDeploymentPolicy, SignalWindowPolicy, SizingPolicy, TargetWeightPolicy

    with pytest.raises(ValueError, match="kelly_lcb_z"):
        SizingPolicy(mode="growth_budget", target_annual_vol=0.35, exposure_cap=3.0, scale_floor=0.2, kelly_enabled=True, kelly_window_days=42, kelly_fraction=0.5, kelly_lcb_z=-0.1, kelly_blend_weight=0.5, drawdown_brake=False)
    target = TargetWeightPolicy(execution_timeframe="3m", execution_universe_size=60, fast_book_mode="single_horizon", slow_book_mode="single_horizon", rebalance_filter="per_symbol_deadband", beta_neutralize=False, ensemble_signal="raw", trend_efficiency_overlay=False, trend_sleeve=False, trend_sleeve_gross=0.0, crash_regime_tilt_alpha=None, committee_capital=True, committee_member_set="flow_momentum", committee_tranche_smoothing=False, committee_regime_adaptive_tranche=True, committee_target_gross=1.0, funding_carry_sleeve=True, funding_carry_weight=0.3, fill_mark_parity_gate=True)
    sizing = SizingPolicy(mode="growth_budget", target_annual_vol=0.35, exposure_cap=3.0, scale_floor=0.2, kelly_enabled=True, kelly_window_days=42, kelly_fraction=0.5, kelly_lcb_z=0.0, kelly_blend_weight=0.5, drawdown_brake=False)
    window = SignalWindowPolicy(panel_window_days=120, bootstrap_return_tail_days=400, fold_panel_warmup_hours=720, committee_purge_hours=24, committee_oos_start="2023-01-01T00:00:00+00:00")
    with pytest.raises(ValueError, match="admitted"):
        MhsDeploymentPolicy(target_weights=target, sizing=sizing, signal_window=window, slow_horizon_hours=168, committee_member_weights={"unknown": 1.0}, admitted_members=("member",))



def test_sizing_signal_and_deployment_policies_validate_all_bounds() -> None:
    import pandas as pd
    import pytest
    from src.mhs.deployment_policy import MhsDeploymentPolicy, SignalWindowPolicy, SizingPolicy, TargetWeightPolicy

    good = dict(mode="growth_budget", target_annual_vol=0.35, exposure_cap=3.0, scale_floor=0.2, kelly_enabled=True, kelly_window_days=42, kelly_fraction=0.5, kelly_lcb_z=0.0, kelly_blend_weight=0.5, drawdown_brake=False)
    for key, bad in [("target_annual_vol", 0.0), ("exposure_cap", 0.5), ("scale_floor", 0.0), ("kelly_window_days", 0), ("kelly_fraction", 0.75), ("kelly_blend_weight", 1.5)]:
        with pytest.raises(ValueError, match=key):
            SizingPolicy(**{**good, key: bad})
    target = TargetWeightPolicy(execution_timeframe="3m", execution_universe_size=60, fast_book_mode="single_horizon", slow_book_mode="single_horizon", rebalance_filter="per_symbol_deadband", beta_neutralize=False, ensemble_signal="raw", trend_efficiency_overlay=False, trend_sleeve=False, trend_sleeve_gross=0.0, crash_regime_tilt_alpha=None, committee_capital=True, committee_member_set="flow_momentum", committee_tranche_smoothing=False, committee_regime_adaptive_tranche=True, committee_target_gross=1.0, funding_carry_sleeve=True, funding_carry_weight=0.3, fill_mark_parity_gate=True)
    sizing = SizingPolicy(**good)
    base_window = dict(panel_window_days=120, bootstrap_return_tail_days=400, fold_panel_warmup_hours=720, committee_purge_hours=24, committee_oos_start="2023-01-01T00:00:00+00:00")
    for key, bad in [("panel_window_days", 0), ("bootstrap_return_tail_days", 0), ("fold_panel_warmup_hours", 0), ("committee_purge_hours", 0)]:
        with pytest.raises(ValueError, match=key):
            SignalWindowPolicy(**{**base_window, key: bad})
    with pytest.raises(ValueError, match="tz-aware"):
        SignalWindowPolicy(panel_window_days=120, bootstrap_return_tail_days=400, fold_panel_warmup_hours=720, committee_purge_hours=24, committee_oos_start="2023-01-01T00:00:00")
    window = SignalWindowPolicy(panel_window_days=120, bootstrap_return_tail_days=400, fold_panel_warmup_hours=720, committee_purge_hours=24, committee_oos_start="2023-01-01T00:00:00+00:00")
    assert window.committee_oos_start == pd.Timestamp("2023-01-01T00:00:00+00:00")
    with pytest.raises(ValueError, match="non-empty"):
        MhsDeploymentPolicy(target_weights=target, sizing=sizing, signal_window=window, slow_horizon_hours=168, committee_member_weights={}, admitted_members=())
    with pytest.raises(ValueError, match="finite"):
        MhsDeploymentPolicy(target_weights=target, sizing=sizing, signal_window=window, slow_horizon_hours=168, committee_member_weights={"m": -1.0}, admitted_members=("m",))
    with pytest.raises(ValueError, match="sum must be"):
        MhsDeploymentPolicy(target_weights=target, sizing=sizing, signal_window=window, slow_horizon_hours=168, committee_member_weights={"m": 0.0}, admitted_members=("m",))
