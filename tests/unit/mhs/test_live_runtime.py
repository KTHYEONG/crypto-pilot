# ruff: noqa
def test_bootstrap_runtime_from_params() -> None:
    import pandas as pd

    from src.mhs.live_runtime import bootstrap_runtime
    from src.mhs.live_strategy import LiveStrategyParams

    ref = pd.Series([0.01, -0.02, 0.005], index=pd.date_range("2026-06-27", periods=3, freq="1D", tz="UTC"))
    params = LiveStrategyParams(
        schema_version=1, strategy_digest="abc", backtest_window=(pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2026-06-30", tz="UTC")),
        created_at=pd.Timestamp("2026-08-31", tz="UTC"), slow_horizon_hours=168, committee_member_weights={"m1": 1.0}, admitted_members=("m1",),
        growth_budget_target_vol=0.35, exposure_cap=3.0, growth_envelope="growth_extreme_budgeted", execution_universe_size=60,
        pnl_vol_target_mode="constant_risk", deployed_flags={}, params_snapshot={}, bootstrap_held_row={"BTCUSDT": 0.3},
    )
    rt = bootstrap_runtime(params, ref)
    assert rt.last_decision_date == pd.Timestamp("2026-06-30", tz="UTC")
    assert rt.held_target_row == {"BTCUSDT": 0.3}
    assert len(rt.reference_daily_returns) == 3
    assert rt.params_digest == "abc"

def test_adopt_params_keeps_deadband_on_compatible_roster() -> None:
    import pandas as pd

    from src.mhs.live_runtime import LiveRuntime, adopt_params
    from src.mhs.live_strategy import LiveStrategyParams

    ref = pd.Series([0.01], index=pd.date_range("2026-06-30", periods=1, freq="1D", tz="UTC"))
    rt = LiveRuntime(schema_version=1, params_digest="old", last_decision_date=pd.Timestamp("2026-07-05", tz="UTC"),
                     held_target_row={"m1": 0.5}, reference_daily_returns=ref)

    def _mk(members, digest):
        return LiveStrategyParams(schema_version=1, strategy_digest=digest, backtest_window=(pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2026-06-30", tz="UTC")),
            created_at=pd.Timestamp("2026-08-31", tz="UTC"), slow_horizon_hours=168, committee_member_weights={m: 1.0 for m in members}, admitted_members=tuple(members),
            growth_budget_target_vol=0.35, exposure_cap=3.0, growth_envelope="g", execution_universe_size=60, pnl_vol_target_mode="constant_risk",
            deployed_flags={}, params_snapshot={}, bootstrap_held_row={"m9": 0.9})

    new_rt, reason = adopt_params(rt, _mk(["m1", "m2"], "new"), ref)
    assert reason == "soft_swap"
    assert new_rt.held_target_row == {"m1": 0.5}
    assert new_rt.params_digest == "new"

    new_rt2, reason2 = adopt_params(rt, _mk(["m2", "m3"], "new2"), ref)
    assert reason2 == "soft_swap"
    assert new_rt2.held_target_row == {"m1": 0.5}

def test_load_or_bootstrap_runtime_missing_is_not_error(tmp_path) -> None:
    import pandas as pd

    from src.mhs.live_runtime import load_or_bootstrap_runtime, save_runtime
    from src.mhs.live_strategy import LiveStrategyParams

    ref = pd.Series([0.01, 0.02], index=pd.date_range("2026-06-29", periods=2, freq="1D", tz="UTC"))
    params = LiveStrategyParams(schema_version=1, strategy_digest="d", backtest_window=(pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2026-06-30", tz="UTC")),
        created_at=pd.Timestamp("2026-08-31", tz="UTC"), slow_horizon_hours=168, committee_member_weights={"m1": 1.0}, admitted_members=("m1",),
        growth_budget_target_vol=0.35, exposure_cap=3.0, growth_envelope="g", execution_universe_size=60, pnl_vol_target_mode="constant_risk",
        deployed_flags={}, params_snapshot={}, bootstrap_held_row={"BTCUSDT": 0.1})
    p = tmp_path / "runtime.json"
    rt = load_or_bootstrap_runtime(p, params, ref)
    assert rt.params_digest == "d"
    assert p.exists()


# --- auto appended from contract ---
def test_adopt_params_soft_swap_preserves_holdings_and_refreshes_anchor() -> None:
    import pandas as pd
    from src.mhs.live_runtime import LiveRuntime, adopt_params
    from src.mhs.live_strategy import LiveStrategyParams

    old_ref = pd.Series([0.01], index=pd.date_range("2026-06-30", periods=1, freq="1D", tz="UTC"))
    new_ref = pd.Series(
        [0.02, 0.03], index=pd.date_range("2026-07-01", periods=2, freq="1D", tz="UTC")
    )
    rt = LiveRuntime(
        schema_version=1, params_digest="old",
        last_decision_date=pd.Timestamp("2026-07-05", tz="UTC"),
        held_target_row={"BTCUSDT": 0.5}, reference_daily_returns=old_ref,
    )
    params = LiveStrategyParams(
        schema_version=1, strategy_digest="newdigest",
        backtest_window=(pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2026-06-30", tz="UTC")),
        created_at=pd.Timestamp("2026-08-31", tz="UTC"), slow_horizon_hours=168,
        committee_member_weights={"m1": 1.0}, admitted_members=("m1",),
        growth_budget_target_vol=0.35, exposure_cap=3.0, growth_envelope="g",
        execution_universe_size=60, pnl_vol_target_mode="constant_risk",
        deployed_flags={}, params_snapshot={}, bootstrap_held_row={"m9": 0.9},
    )

    new_rt, reason = adopt_params(rt, params, new_ref)

    assert reason == "soft_swap"
    assert new_rt.held_target_row == {"BTCUSDT": 0.5}
    assert new_rt.last_decision_date == pd.Timestamp("2026-07-05", tz="UTC")
    assert new_rt.params_digest == "newdigest"
    assert len(new_rt.reference_daily_returns) == 2


def test_adopt_params_bootstrap_reason_when_no_holdings() -> None:
    import pandas as pd
    from src.mhs.live_runtime import LiveRuntime, adopt_params
    from src.mhs.live_strategy import LiveStrategyParams

    ref = pd.Series(dtype="float64")
    rt = LiveRuntime(
        schema_version=1, params_digest="old",
        last_decision_date=pd.Timestamp("2026-07-05", tz="UTC"),
        held_target_row={}, reference_daily_returns=ref,
    )
    params = LiveStrategyParams(
        schema_version=1, strategy_digest="nd",
        backtest_window=(pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2026-06-30", tz="UTC")),
        created_at=pd.Timestamp("2026-08-31", tz="UTC"), slow_horizon_hours=168,
        committee_member_weights={"m1": 1.0}, admitted_members=("m1",),
        growth_budget_target_vol=0.35, exposure_cap=3.0, growth_envelope="g",
        execution_universe_size=60, pnl_vol_target_mode="constant_risk",
        deployed_flags={}, params_snapshot={}, bootstrap_held_row={"m9": 0.9},
    )

    new_rt, reason = adopt_params(rt, params, ref)

    assert reason == "bootstrap"
    assert new_rt.params_digest == "nd"


def test_reconcile_runtime_params_noop_on_matching_digest() -> None:
    import pandas as pd
    from src.mhs.live_runtime import LiveRuntime, reconcile_runtime_params
    from src.mhs.live_strategy import LiveStrategyParams

    ref = pd.Series(dtype="float64")
    rt = LiveRuntime(
        schema_version=1, params_digest="same",
        last_decision_date=pd.Timestamp("2026-07-05", tz="UTC"),
        held_target_row={"BTCUSDT": 0.5}, reference_daily_returns=ref,
    )
    params = LiveStrategyParams(
        schema_version=1, strategy_digest="same",
        backtest_window=(pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2026-06-30", tz="UTC")),
        created_at=pd.Timestamp("2026-08-31", tz="UTC"), slow_horizon_hours=168,
        committee_member_weights={"m1": 1.0}, admitted_members=("m1",),
        growth_budget_target_vol=0.35, exposure_cap=3.0, growth_envelope="g",
        execution_universe_size=60, pnl_vol_target_mode="constant_risk",
        deployed_flags={}, params_snapshot={}, bootstrap_held_row={},
    )

    out_rt, reason = reconcile_runtime_params(rt, params, ref)

    assert reason is None
    assert out_rt is rt


def test_reconcile_runtime_params_swaps_on_digest_change() -> None:
    import pandas as pd
    from src.mhs.live_runtime import LiveRuntime, reconcile_runtime_params
    from src.mhs.live_strategy import LiveStrategyParams

    ref = pd.Series(dtype="float64")
    rt = LiveRuntime(
        schema_version=1, params_digest="old",
        last_decision_date=pd.Timestamp("2026-07-05", tz="UTC"),
        held_target_row={"BTCUSDT": 0.5}, reference_daily_returns=ref,
    )
    params = LiveStrategyParams(
        schema_version=1, strategy_digest="brandnew",
        backtest_window=(pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2026-06-30", tz="UTC")),
        created_at=pd.Timestamp("2026-08-31", tz="UTC"), slow_horizon_hours=168,
        committee_member_weights={"m1": 1.0}, admitted_members=("m1",),
        growth_budget_target_vol=0.35, exposure_cap=3.0, growth_envelope="g",
        execution_universe_size=60, pnl_vol_target_mode="constant_risk",
        deployed_flags={}, params_snapshot={}, bootstrap_held_row={},
    )

    out_rt, reason = reconcile_runtime_params(rt, params, ref)

    assert reason == "soft_swap"
    assert out_rt.params_digest == "brandnew"
    assert out_rt.held_target_row == {"BTCUSDT": 0.5}


