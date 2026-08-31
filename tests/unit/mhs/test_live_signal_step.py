# ruff: noqa
import pytest


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

    def _fake_compute(p, r, root, date, **kw):
        return pd.Series({"BTCUSDT": 0.2}, name=date), pd.Series([0.01], index=pd.DatetimeIndex([date])), 1.0

    monkeypatch.setattr(step, "compute_signal_row", _fake_compute)
    path = tmp_path / "w.parquet"
    new_rt, n, _sc = step.advance_to_date(params, rt, path, "", pd.Timestamp("2026-08-23", tz="UTC"))
    assert n == 3
    assert new_rt.last_decision_date == pd.Timestamp("2026-08-23", tz="UTC")
    assert new_rt.held_target_row["BTCUSDT"] == 0.2


# --- auto appended from contract ---
def test_realized_daily_returns_pct_change_from_ledger(tmp_path) -> None:
    import pandas as pd
    from src.mhs.live_signal_step import realized_daily_returns

    df = pd.DataFrame(
        {
            "decision_time": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-03"], utc=True
            ),
            "mode": ["paper", "paper", "paper"],
            "equity_usdt": [2000.0, 2100.0, 2079.0],
        }
    )
    d = tmp_path / "live_portfolio_state"
    d.mkdir()
    df.to_parquet(d / "active.parquet", index=False)

    out = realized_daily_returns(d, "paper", bt_end=pd.Timestamp("2025-12-31", tz="UTC"))

    assert list(out.index) == list(
        pd.to_datetime(["2026-01-02", "2026-01-03"], utc=True)
    )
    assert out.iloc[0] == pytest.approx(0.05)
    assert out.iloc[1] == pytest.approx(-0.01)


def test_realized_daily_returns_filters_mode_and_nonfinite_and_dedupes(tmp_path) -> None:
    import numpy as np
    import pandas as pd
    from src.mhs.live_signal_step import realized_daily_returns

    df = pd.DataFrame(
        {
            "decision_time": pd.to_datetime(
                ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02", "2026-01-03"],
                utc=True,
            ),
            "mode": ["paper", "shadow", "paper", "paper", "paper"],
            "equity_usdt": [1000.0, 999.0, 900.0, 1100.0, np.nan],
        }
    )
    d = tmp_path / "ps"
    d.mkdir()
    df.to_parquet(d / "active.parquet", index=False)

    out = realized_daily_returns(d, "paper", bt_end=pd.Timestamp("2025-12-31", tz="UTC"))

    # 2026-01-01 -> 1000 (paper, first-of-dupe dropped), 2026-01-02 -> 1100 (last dupe),
    # 2026-01-03 dropped (NaN equity). One return: 1100/1000 - 1.
    assert len(out) == 1
    assert out.iloc[0] == pytest.approx(0.1)


def test_realized_daily_returns_empty_when_store_missing_or_thin(tmp_path) -> None:
    import pandas as pd
    from src.mhs.live_signal_step import realized_daily_returns

    missing = realized_daily_returns(
        tmp_path / "nope", "paper", bt_end=pd.Timestamp("2025-12-31", tz="UTC")
    )
    assert missing.empty and missing.dtype == "float64"

    d = tmp_path / "ps"
    d.mkdir()
    pd.DataFrame(
        {
            "decision_time": pd.to_datetime(["2026-01-02"], utc=True),
            "mode": ["paper"],
            "equity_usdt": [2000.0],
        }
    ).to_parquet(d / "active.parquet", index=False)
    thin = realized_daily_returns(d, "paper", bt_end=pd.Timestamp("2025-12-31", tz="UTC"))
    assert thin.empty


def test_realized_daily_returns_excludes_rows_at_or_before_bt_end(tmp_path) -> None:
    import pandas as pd
    from src.mhs.live_signal_step import realized_daily_returns

    df = pd.DataFrame(
        {
            "decision_time": pd.to_datetime(
                ["2025-12-30", "2025-12-31", "2026-01-01", "2026-01-02"], utc=True
            ),
            "mode": ["paper"] * 4,
            "equity_usdt": [10.0, 20.0, 100.0, 110.0],
        }
    )
    d = tmp_path / "ps"
    d.mkdir()
    df.to_parquet(d / "active.parquet", index=False)

    out = realized_daily_returns(d, "paper", bt_end=pd.Timestamp("2025-12-31", tz="UTC"))

    assert list(out.index) == [pd.Timestamp("2026-01-02", tz="UTC")]
    assert out.iloc[0] == pytest.approx(0.1)


def test_analytic_net_daily_return_is_removed() -> None:
    import src.mhs.live_signal_step as m

    assert not hasattr(m, "analytic_net_daily_return")


def test_compute_signal_row_scales_on_realized_forward_returns(tmp_path, monkeypatch) -> None:
    import numpy as np
    import pandas as pd
    import src.mhs.live_signal_step as m
    from src.mhs.live_runtime import LiveRuntime
    from src.mhs.live_strategy import LiveStrategyParams

    dt = pd.Timestamp("2026-08-31", tz="UTC")
    tw = pd.DataFrame(
        [[0.6, -0.4]], index=pd.DatetimeIndex([dt]), columns=["BTCUSDT", "ETHUSDT"]
    )
    monkeypatch.setattr(
        m, "_build_fold_target_weights",
        lambda *a, **k: (tw, pd.DatetimeIndex([dt]), [], pd.DatetimeIndex([dt])),
    )
    monkeypatch.setattr(m, "_load_funding_by_symbol", lambda *a, **k: {})
    # loud realized vol -> scalar should be pulled below the 3.0 cap
    rng = pd.date_range("2026-06-01", periods=60, freq="1D", tz="UTC")
    monkeypatch.setattr(
        m, "realized_daily_returns",
        lambda *a, **k: pd.Series(np.r_[np.full(30, 0.05), np.full(30, -0.05)], index=rng),
    )
    boot = pd.Series(
        np.full(120, 0.001),
        index=pd.date_range("2025-09-01", periods=120, freq="1D", tz="UTC"),
    )
    params = LiveStrategyParams(
        schema_version=1, strategy_digest="d",
        backtest_window=(pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2025-12-31", tz="UTC")),
        created_at=dt, slow_horizon_hours=168, committee_member_weights={"m": 1.0},
        admitted_members=("m",), growth_budget_target_vol=1.0, exposure_cap=3.0,
        growth_envelope="growth_extreme", execution_universe_size=60,
        pnl_vol_target_mode="growth_budget", deployed_flags={}, params_snapshot={},
        bootstrap_held_row={},
    )
    rt = LiveRuntime(
        schema_version=1, params_digest="d", last_decision_date=pd.Timestamp("2026-08-30", tz="UTC"),
        held_target_row={"BTCUSDT": 0.5}, reference_daily_returns=boot,
    )

    scaled, ref_out, scalar = m.compute_signal_row(
        params, rt, str(tmp_path), dt, portfolio_state_dir=tmp_path, mode="paper"
    )

    assert 0.0 < scalar <= 3.0
    assert scaled["BTCUSDT"] == pytest.approx(0.6 * scalar)
    assert ref_out is rt.reference_daily_returns


def test_compute_signal_row_day_one_uses_bootstrap_warmup_only(tmp_path, monkeypatch) -> None:
    import numpy as np
    import pandas as pd
    import src.mhs.live_signal_step as m
    from src.mhs.live_runtime import LiveRuntime
    from src.mhs.live_strategy import LiveStrategyParams

    dt = pd.Timestamp("2026-08-31", tz="UTC")
    tw = pd.DataFrame([[1.0]], index=pd.DatetimeIndex([dt]), columns=["BTCUSDT"])
    monkeypatch.setattr(
        m, "_build_fold_target_weights",
        lambda *a, **k: (tw, pd.DatetimeIndex([dt]), [], pd.DatetimeIndex([dt])),
    )
    monkeypatch.setattr(m, "_load_funding_by_symbol", lambda *a, **k: {})
    monkeypatch.setattr(m, "realized_daily_returns", lambda *a, **k: pd.Series(dtype="float64"))
    boot = pd.Series(
        np.full(150, 0.002),
        index=pd.date_range("2025-08-01", periods=150, freq="1D", tz="UTC"),
    )
    params = LiveStrategyParams(
        schema_version=1, strategy_digest="d",
        backtest_window=(pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2025-12-31", tz="UTC")),
        created_at=dt, slow_horizon_hours=168, committee_member_weights={"m": 1.0},
        admitted_members=("m",), growth_budget_target_vol=1.0, exposure_cap=3.0,
        growth_envelope="growth_extreme", execution_universe_size=60,
        pnl_vol_target_mode="growth_budget", deployed_flags={}, params_snapshot={},
        bootstrap_held_row={},
    )
    rt = LiveRuntime(
        schema_version=1, params_digest="d", last_decision_date=pd.Timestamp("2026-08-30", tz="UTC"),
        held_target_row={}, reference_daily_returns=boot,
    )

    scaled, _ref, scalar = m.compute_signal_row(
        params, rt, str(tmp_path), dt, portfolio_state_dir=tmp_path, mode="paper"
    )

    assert 0.0 < scalar <= 3.0
    assert np.isfinite(scaled["BTCUSDT"])


