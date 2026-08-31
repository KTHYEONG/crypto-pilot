# ruff: noqa
def test_analytic_net_daily_return_terms() -> None:
    import pandas as pd

    from src.mhs.live_signal_step import analytic_net_daily_return

    d0 = pd.Timestamp("2026-08-24", tz="UTC")
    d1 = pd.Timestamp("2026-08-25", tz="UTC")
    close_1d = pd.DataFrame({"BTCUSDT": [100.0, 110.0], "ETHUSDT": [50.0, 45.0]}, index=[d0, d1])
    held = pd.Series({"BTCUSDT": 0.5, "ETHUSDT": -0.5})
    new = pd.Series({"BTCUSDT": 0.5, "ETHUSDT": -0.5})
    funding_1d = pd.Series({d1: 0.0})

    r = analytic_net_daily_return(held, new, close_1d, funding_1d, d1, taker_cost_bps=0.0)
    # BTC +10% * 0.5 + ETH -10% * -0.5 = 0.05 + 0.05 = 0.10
    assert abs(r - 0.10) < 1e-9

    new2 = pd.Series({"BTCUSDT": 0.7, "ETHUSDT": -0.5})
    r2 = analytic_net_daily_return(held, new2, close_1d, funding_1d, d1, taker_cost_bps=100.0)
    # turnover = 0.5*|0.7-0.5| = 0.1 ; cost = 0.1 * 0.01 = 0.001
    assert abs(r2 - (0.10 - 0.001)) < 1e-9

def test_analytic_net_daily_return_missing_price_is_zero() -> None:
    import pandas as pd

    from src.mhs.live_signal_step import analytic_net_daily_return

    d0 = pd.Timestamp("2026-08-24", tz="UTC")
    d1 = pd.Timestamp("2026-08-25", tz="UTC")
    close_1d = pd.DataFrame({"BTCUSDT": [100.0, 110.0]}, index=[d0, d1])
    held = pd.Series({"BTCUSDT": 0.5, "GHOSTUSDT": 0.5})
    funding_1d = pd.Series({d1: 0.0})

    r = analytic_net_daily_return(held, held, close_1d, funding_1d, d1, taker_cost_bps=0.0)
    assert abs(r - 0.05) < 1e-9

def test_advance_to_date_scores_missing_days(monkeypatch, tmp_path) -> None:
    import pandas as pd

    import src.mhs.live_signal_step as step
    from src.mhs.live_runtime import LiveRuntime
    from src.mhs.live_strategy import LiveStrategyParams

    params = LiveStrategyParams(schema_version=1, strategy_digest="d", backtest_window=(pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2026-08-20", tz="UTC")),
        created_at=pd.Timestamp("2026-08-31", tz="UTC"), slow_horizon_hours=168, committee_member_weights={"m1": 1.0}, admitted_members=("m1",),
        growth_budget_target_vol=0.35, exposure_cap=3.0, growth_envelope="g", execution_universe_size=60, pnl_vol_target_mode="constant_risk",
        deployed_flags={}, params_snapshot={"SIGNAL_RETURN_TAIL_DAYS": 400}, bootstrap_held_row={"BTCUSDT": 0.1})
    rt = LiveRuntime(schema_version=1, params_digest="d", last_decision_date=pd.Timestamp("2026-08-20", tz="UTC"),
                     held_target_row={"BTCUSDT": 0.1}, reference_daily_returns=pd.Series(dtype="float64"))

    def _fake_compute(p, r, root, date):
        return pd.Series({"BTCUSDT": 0.2}, name=date), pd.Series([0.01], index=pd.DatetimeIndex([date])), 1.0

    monkeypatch.setattr(step, "compute_signal_row", _fake_compute)
    path = tmp_path / "w.parquet"
    new_rt, n, _sc = step.advance_to_date(params, rt, path, "", pd.Timestamp("2026-08-23", tz="UTC"))
    assert n == 3
    assert new_rt.last_decision_date == pd.Timestamp("2026-08-23", tz="UTC")
    assert new_rt.held_target_row["BTCUSDT"] == 0.2
