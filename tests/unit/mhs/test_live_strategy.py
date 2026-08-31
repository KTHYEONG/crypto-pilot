# ruff: noqa
def test_strategy_params_roundtrip_sealed(tmp_path) -> None:
    import pandas as pd
    from pydantic import SecretStr

    from src.mhs.live_strategy import LiveStrategyParams, load_strategy_params, save_strategy_params

    params = LiveStrategyParams(
        schema_version=1,
        strategy_digest="",
        backtest_window=(pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2026-06-30", tz="UTC")),
        created_at=pd.Timestamp("2026-08-31", tz="UTC"),
        slow_horizon_hours=168,
        committee_member_weights={"m1": 0.5, "m2": 0.5},
        admitted_members=("m1", "m2"),
        growth_budget_target_vol=0.35,
        exposure_cap=3.0,
        growth_envelope="growth_extreme_budgeted",
        execution_universe_size=60,
        pnl_vol_target_mode="constant_risk",
        deployed_flags={"committee_capital": True, "committee_kelly_sizing": True},
        params_snapshot={"SIGNAL_PANEL_WINDOW_DAYS": 120},
        bootstrap_held_row={"BTCUSDT": 0.2, "ETHUSDT": -0.1},
    )
    key = SecretStr("A" * 43 + "=")
    dest = save_strategy_params(tmp_path / "strategy_params.json", params, artifact_key=key)
    loaded = load_strategy_params(dest, artifact_key=key)
    assert loaded.slow_horizon_hours == 168
    assert loaded.admitted_members == ("m1", "m2")
    assert loaded.bootstrap_held_row["BTCUSDT"] == 0.2
    assert loaded.strategy_digest and loaded.strategy_digest == load_strategy_params(dest, artifact_key=key).strategy_digest

def test_load_strategy_params_tamper_detected(tmp_path) -> None:
    import json

    import pytest

    from src.common.errors import DataIntegrityError
    from src.mhs.live_strategy import load_strategy_params

    payload = {
        "schema_version": 1,
        "strategy_digest": "deadbeef" * 4,
        "backtest_window": ["2021-01-01T00:00:00+00:00", "2026-06-30T00:00:00+00:00"],
        "created_at": "2026-08-31T00:00:00+00:00",
        "slow_horizon_hours": 168,
        "committee_member_weights": {"m1": 1.0},
        "admitted_members": ["m1"],
        "growth_budget_target_vol": 0.35,
        "exposure_cap": 3.0,
        "growth_envelope": "growth_extreme_budgeted",
        "execution_universe_size": 60,
        "pnl_vol_target_mode": "constant_risk",
        "deployed_flags": {},
        "params_snapshot": {},
        "bootstrap_held_row": {"BTCUSDT": 0.2},
    }
    path = tmp_path / "strategy_params.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DataIntegrityError):
        load_strategy_params(path)

def test_assert_deployment_eligible_rejects_research_go_fail() -> None:
    import types

    import pandas as pd
    import pytest

    from src.common.errors import DataIntegrityError
    from src.mhs.live_strategy import assert_deployment_eligible

    tw = pd.DataFrame({"BTCUSDT": [0.1]}, index=pd.DatetimeIndex([pd.Timestamp("2026-08-30", tz="UTC")]))
    report = types.SimpleNamespace(status="OK", research_go=types.SimpleNamespace(eligible=False), blend=types.SimpleNamespace(target_weights=tw))
    with pytest.raises(DataIntegrityError) as exc:
        assert_deployment_eligible(report)
    assert "deployment ineligible" in str(exc.value)
