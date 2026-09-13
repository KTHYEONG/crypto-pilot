# ruff: noqa
import dataclasses as _dc2

import pandas as _pd

from src.mhs.contracts import MhsDiagnosticRequest as _Req2
from src.mhs.deployment_policy import build_deployment_policy as _build2
from src.mhs.pipeline.config import MhsRunConfig as _Cfg2


def _v2_sig_params(mode="growth_budget", tv=1.0, cc=False, ks=False):
    _req = _Req2(pnl_vol_target_mode=mode, committee_capital=cc, committee_kelly_sizing=ks)
    _pol = _build2(_req, slow_horizon_hours=168, committee_member_weights={"m": 1.0}, admitted_members=("m",), target_annual_vol=tv, exposure_cap=3.0)
    return __import__("src.mhs.live_strategy", fromlist=["LiveStrategyParams"]).LiveStrategyParams(schema_version=2, strategy_digest="d", backtest_window=(_pd.Timestamp("2021-01-01", tz="UTC"), _pd.Timestamp("2025-12-31", tz="UTC")), created_at=_pd.Timestamp("2026-08-31", tz="UTC"), policy=_pol, bootstrap_sha256="a" * 64, bootstrap_held_row={})


import pytest


def test_advance_to_date_scores_missing_days(monkeypatch, tmp_path) -> None:
    import pandas as pd

    import src.mhs.live_signal_step as step
    from src.mhs.live_runtime import LiveRuntime
    from src.mhs.live_strategy import LiveStrategyParams

    params = _v2_sig_params("constant_risk", 0.35, False, False)
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
    params = _v2_sig_params("growth_budget", 1.0, False, False)
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
    params = _v2_sig_params("growth_budget", 1.0, False, False)
    rt = LiveRuntime(
        schema_version=1, params_digest="d", last_decision_date=pd.Timestamp("2026-08-30", tz="UTC"),
        held_target_row={}, reference_daily_returns=boot,
    )

    scaled, _ref, scalar = m.compute_signal_row(
        params, rt, str(tmp_path), dt, portfolio_state_dir=tmp_path, mode="paper"
    )

    assert 0.0 < scalar <= 3.0
    assert np.isfinite(scaled["BTCUSDT"])


def test_scenario_kelly_lcb_06_live_signal_scalar_uses_recalibrated_defaults(tmp_path, monkeypatch) -> None:
    # Given: 위원회 자본 + Kelly 사이징이 켜진 배포 플래그와 양의 에지 부트스트랩 레퍼런스
    import numpy as np
    import pandas as pd

    import src.mhs.live_signal_step as m
    from src.mhs import scaling
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
        np.random.default_rng(20260912).normal(0.0025, 0.010, 400),
        index=pd.date_range("2025-08-01", periods=400, freq="1D", tz="UTC"),
    )
    params = _v2_sig_params("growth_budget", 1.0, True, True)
    rt = LiveRuntime(
        schema_version=1, params_digest="d",
        last_decision_date=pd.Timestamp("2026-08-30", tz="UTC"),
        held_target_row={}, reference_daily_returns=boot,
    )
    original_kelly = scaling._committee_kelly_scale

    # When: 재보정 기본값으로 1회, 이어서 구 상수를 주입해 1회 산출
    _scaled_new, _ref, scalar_new = m.compute_signal_row(
        params, rt, str(tmp_path), dt, portfolio_state_dir=tmp_path, mode="paper",
    )
    monkeypatch.setattr(
        scaling, "_committee_kelly_scale",
        lambda r, **kw: original_kelly(
            r, window_days=21, fraction=0.25, z=1.0, cap=kw.get("cap", 1.0),
        ),
    )
    _scaled_legacy, _ref_legacy, scalar_legacy = m.compute_signal_row(
        params, rt, str(tmp_path), dt, portfolio_state_dir=tmp_path, mode="paper",
    )

    # Then: 라이브 경로가 재보정 값을 실제로 사용하며 cap 을 넘지 않는다
    assert scalar_new > scalar_legacy
    assert 0.0 < scalar_new <= 3.0
    assert _scaled_new["BTCUSDT"] == pytest.approx(scalar_new)





def test_compute_signal_row_wires_policy_bootstrap_and_target_constants(tmp_path, monkeypatch) -> None:
    import dataclasses
    import pandas as pd
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.deployment_policy import build_deployment_policy
    from src.mhs.live_runtime import LiveRuntime
    from src.mhs.live_strategy import LiveStrategyParams
    from src.mhs.pipeline.config import MhsRunConfig
    import src.mhs.live_signal_step as module

    dt = pd.Timestamp("2026-08-31", tz="UTC")
    request = MhsDiagnosticRequest(**dataclasses.asdict(MhsRunConfig()))
    policy = build_deployment_policy(request, slow_horizon_hours=168, committee_member_weights={"m": 1.0}, admitted_members=("m",), target_annual_vol=0.35, exposure_cap=3.0)
    params = LiveStrategyParams(schema_version=2, strategy_digest="d", backtest_window=(pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2025-12-31", tz="UTC")), created_at=dt, policy=policy, bootstrap_sha256="a" * 64, bootstrap_held_row={})
    warm = pd.Series([0.01, 0.02], index=pd.date_range("2025-12-29", periods=2, freq="1D", tz="UTC"), dtype="float64")
    runtime = LiveRuntime(schema_version=1, params_digest="d", last_decision_date=dt - pd.Timedelta(days=1), held_target_row={}, reference_daily_returns=warm)
    target = pd.DataFrame({"BTCUSDT": [1.0]}, index=pd.DatetimeIndex([dt]))
    captured = {}
    def fake_builder(*args, **kwargs):
        captured.update(request=args[2], panel=kwargs["panel_warmup_hours"], oos=kwargs["committee_oos_start"])
        return target, pd.DatetimeIndex([dt]), [], pd.DatetimeIndex([dt])
    def fake_scale(reference, sizing, *, warmup_returns=None):
        captured.update(sizing=sizing, warmup=warmup_returns)
        return pd.Series(2.0, index=reference.index, dtype="float64")
    monkeypatch.setattr(module, "_build_fold_target_weights", fake_builder)
    monkeypatch.setattr(module, "_load_funding_by_symbol", lambda *_: {})
    monkeypatch.setattr(module, "realized_daily_returns", lambda *_a, **_k: pd.Series([0.01], index=pd.DatetimeIndex([pd.Timestamp("2026-08-30", tz="UTC")]), dtype="float64"))
    monkeypatch.setattr(module, "compute_exposure_scale", fake_scale)
    scaled, _, scalar = module.compute_signal_row(params, runtime, str(tmp_path), dt, portfolio_state_dir=tmp_path, mode="paper")
    assert captured["request"].execution_universe_size == policy.target_weights.execution_universe_size
    assert captured["panel"] == policy.signal_window.fold_panel_warmup_hours
    assert captured["oos"] == policy.signal_window.committee_oos_start
    assert captured["sizing"] is policy.sizing
    pd.testing.assert_series_equal(captured["warmup"], warm)
    assert scalar == 2.0 and scaled["BTCUSDT"] == 2.0



def test_compute_signal_row_sorts_nonmonotonic_bootstrap_warmup(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import src.mhs.live_signal_step as module
    from src.mhs.live_runtime import LiveRuntime

    dt = pd.Timestamp("2026-08-31", tz="UTC")
    params = _v2_sig_params("growth_budget", 1.0, False, False)
    rev_idx = pd.DatetimeIndex([pd.Timestamp("2025-12-30", tz="UTC"), pd.Timestamp("2025-12-29", tz="UTC")])
    rev = pd.Series([0.02, 0.01], index=rev_idx, dtype="float64")
    runtime = LiveRuntime(schema_version=1, params_digest="d", last_decision_date=dt - pd.Timedelta(days=1), held_target_row={}, reference_daily_returns=rev)
    target = pd.DataFrame({"BTCUSDT": [1.0]}, index=pd.DatetimeIndex([dt]))
    captured = {}
    module._build_fold_target_weights.__name__  # keep linter calm about the seam below
    def fake_builder(*args, **kwargs):
        return target, pd.DatetimeIndex([dt]), [], pd.DatetimeIndex([dt])
    def fake_scale(reference, sizing, *, warmup_returns=None):
        captured["warmup"] = warmup_returns
        return pd.Series(2.0, index=reference.index, dtype="float64")
    monkeypatch.setattr(module, "_build_fold_target_weights", fake_builder)
    monkeypatch.setattr(module, "_load_funding_by_symbol", lambda *_: {})
    monkeypatch.setattr(module, "realized_daily_returns", lambda *_a, **_k: pd.Series([0.01], index=pd.DatetimeIndex([pd.Timestamp("2026-08-30", tz="UTC")]), dtype="float64"))
    monkeypatch.setattr(module, "compute_exposure_scale", fake_scale)
    module.compute_signal_row(params, runtime, str(tmp_path), dt, portfolio_state_dir=tmp_path, mode="paper")
    assert list(captured["warmup"].index) == sorted(captured["warmup"].index)
