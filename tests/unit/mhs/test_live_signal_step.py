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

    def _fake_compute(p, r, root, date, *, portfolio_state_dir=None, mode="shadow", applied_scale=None, quarantine=None):
        return pd.Series({"BTCUSDT": 0.4}, name=date), pd.Series({"BTCUSDT": 0.2}, name=date), 1.0

    monkeypatch.setattr(step, "compute_signal_row", _fake_compute)
    monkeypatch.setattr(step, "decision_mark_row", lambda symbols, date, mark_path_fn: pd.Series({"BTCUSDT": 101.0}, name=date, dtype="float64"))
    path = tmp_path / "deployed_target_weights.parquet"
    new_rt, n, _sc = step.advance_to_date(params, rt, path, "", pd.Timestamp("2026-08-23", tz="UTC"))
    assert n == 3
    assert new_rt.last_decision_date == pd.Timestamp("2026-08-23", tz="UTC")
    assert new_rt.held_target_row["BTCUSDT"] == 0.2


# --- auto appended from contract ---
def test_realized_daily_returns_pct_change_from_ledger(tmp_path) -> None:
    import pandas as pd
    from src.mhs.live_signal_step import realized_equity

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

    eq = realized_equity(d, "paper", bt_end=pd.Timestamp("2025-12-31", tz="UTC"))

    assert list(eq.index) == list(
        pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"], utc=True)
    )
    assert eq.iloc[0] == pytest.approx(2000.0)
    assert eq.iloc[1] == pytest.approx(2100.0)
    assert eq.iloc[2] == pytest.approx(2079.0)
    out = eq.pct_change().dropna()
    assert list(out.index) == list(
        pd.to_datetime(["2026-01-02", "2026-01-03"], utc=True)
    )
    assert out.iloc[0] == pytest.approx(0.05)
    assert out.iloc[1] == pytest.approx(-0.01)


def test_realized_daily_returns_filters_mode_and_nonfinite_and_dedupes(tmp_path) -> None:
    import numpy as np
    import pandas as pd
    from src.mhs.live_signal_step import realized_equity

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

    eq = realized_equity(d, "paper", bt_end=pd.Timestamp("2025-12-31", tz="UTC"))

    # 2026-01-01 -> 1000 (paper, first-of-dupe dropped), 2026-01-02 -> 1100 (last dupe),
    # 2026-01-03 dropped (NaN equity).
    assert list(eq.index) == list(
        pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True)
    )
    assert eq.iloc[0] == pytest.approx(1000.0)
    assert eq.iloc[1] == pytest.approx(1100.0)
    out = eq.pct_change().dropna()
    # One return: 1100/1000 - 1.
    assert len(out) == 1
    assert out.iloc[0] == pytest.approx(0.1)


def test_realized_daily_returns_empty_when_store_missing_or_thin(tmp_path) -> None:
    import pandas as pd
    from src.mhs.live_signal_step import realized_equity

    missing = realized_equity(
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
    thin = realized_equity(d, "paper", bt_end=pd.Timestamp("2025-12-31", tz="UTC"))
    assert list(thin.index) == list(pd.to_datetime(["2026-01-02"], utc=True))
    assert thin.iloc[0] == pytest.approx(2000.0)
    assert thin.pct_change().dropna().empty


def test_realized_daily_returns_excludes_rows_at_or_before_bt_end(tmp_path) -> None:
    import pandas as pd
    from src.mhs.live_signal_step import realized_equity

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

    eq = realized_equity(d, "paper", bt_end=pd.Timestamp("2025-12-31", tz="UTC"))

    assert list(eq.index) == list(
        pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True)
    )
    assert eq.iloc[0] == pytest.approx(100.0)
    assert eq.iloc[1] == pytest.approx(110.0)
    out = eq.pct_change().dropna()
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
    monkeypatch.setattr(m, "_assert_panel_history_available", lambda *a, **k: None)
    # loud realized vol -> scalar should be pulled below the 3.0 cap
    rng = pd.date_range("2026-06-01", periods=60, freq="1D", tz="UTC")
    fwd = pd.Series(np.r_[np.full(30, 0.05), np.full(30, -0.05)], index=rng)
    monkeypatch.setattr(m, "realized_equity", lambda *a, **k: pd.Series(dtype="float64"))
    monkeypatch.setattr(m, "descale_realized_returns", lambda equity, scale: fwd)
    boot = pd.Series(
        np.full(120, 0.001),
        index=pd.date_range("2025-09-01", periods=120, freq="1D", tz="UTC"),
    )
    params = _v2_sig_params("growth_budget", 1.0, False, False)
    rt = LiveRuntime(
        schema_version=1, params_digest="d", last_decision_date=pd.Timestamp("2026-08-30", tz="UTC"),
        held_target_row={"BTCUSDT": 0.5}, reference_daily_returns=boot,
    )

    scaled, prescale_out, scalar = m.compute_signal_row(
        params, rt, str(tmp_path), dt, portfolio_state_dir=tmp_path, mode="paper"
    )

    assert 0.0 < scalar <= 3.0
    assert scaled["BTCUSDT"] == pytest.approx(prescale_out["BTCUSDT"] * scalar)
    assert scaled["ETHUSDT"] == pytest.approx(prescale_out["ETHUSDT"] * scalar)
    assert prescale_out["BTCUSDT"] == pytest.approx(0.5)
    assert prescale_out["ETHUSDT"] == pytest.approx(-0.4)


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
    monkeypatch.setattr(m, "_assert_panel_history_available", lambda *a, **k: None)
    monkeypatch.setattr(m, "realized_equity", lambda *a, **k: pd.Series(dtype="float64"))
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
    monkeypatch.setattr(m, "_assert_panel_history_available", lambda *a, **k: None)
    monkeypatch.setattr(m, "realized_equity", lambda *a, **k: pd.Series(dtype="float64"))
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
    def fake_builder(*args, apply_rebalance_deadband=True, **kwargs):
        captured.update(request=args[2], panel=kwargs["panel_warmup_hours"], oos=kwargs["committee_oos_start"], apply_rebalance_deadband=apply_rebalance_deadband)
        return target, pd.DatetimeIndex([dt]), [], pd.DatetimeIndex([dt])
    def fake_scale(reference, sizing, *, warmup_returns=None):
        captured.update(sizing=sizing, warmup=warmup_returns)
        return pd.Series(2.0, index=reference.index, dtype="float64")
    monkeypatch.setattr(module, "_build_fold_target_weights", fake_builder)
    monkeypatch.setattr(module, "_load_funding_by_symbol", lambda *_: {})
    monkeypatch.setattr(module, "_assert_panel_history_available", lambda *_a, **_k: None)
    monkeypatch.setattr(module, "realized_equity", lambda *_a, **_k: pd.Series(dtype="float64"))
    monkeypatch.setattr(module, "compute_exposure_scale", fake_scale)
    scaled, _, scalar = module.compute_signal_row(params, runtime, str(tmp_path), dt, portfolio_state_dir=tmp_path, mode="paper")
    assert captured["apply_rebalance_deadband"] is False
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
    monkeypatch.setattr(module, "_assert_panel_history_available", lambda *_a, **_k: None)
    monkeypatch.setattr(module, "realized_equity", lambda *_a, **_k: pd.Series(dtype="float64"))
    monkeypatch.setattr(module, "compute_exposure_scale", fake_scale)
    module.compute_signal_row(params, runtime, str(tmp_path), dt, portfolio_state_dir=tmp_path, mode="paper")
    assert list(captured["warmup"].index) == sorted(captured["warmup"].index)


def test_descale_realized_returns_divides_by_prior_decision_scale() -> None:
    import numpy as np
    import pandas as pd
    from src.mhs.live_signal_step import descale_realized_returns

    idx = pd.to_datetime(["2026-09-01", "2026-09-02", "2026-09-03"], utc=True)
    equity = pd.Series([100.0, 110.0, 99.0], index=idx, dtype="float64")
    applied = pd.Series([2.0, 1.0], index=idx[:2], dtype="float64")
    out = descale_realized_returns(equity, applied)
    assert list(out.index) == list(idx[1:])
    assert out.dtype == np.float64
    np.testing.assert_allclose(out.to_numpy(), [0.10 / 2.0, (99.0 / 110.0 - 1.0) / 1.0], rtol=0.0, atol=1e-15)
    empty = descale_realized_returns(pd.Series(dtype="float64"), applied)
    assert empty.empty and isinstance(empty.index, pd.DatetimeIndex)


def test_descale_realized_returns_starts_at_first_scaled_prior_and_fails_closed() -> None:
    import numpy as np
    import pandas as pd
    import pytest
    from src.common.errors import DataIntegrityError
    from src.mhs.live_signal_step import descale_realized_returns

    idx = pd.to_datetime(["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04"], utc=True)
    equity = pd.Series([100.0, 101.0, 102.0, 103.0], index=idx, dtype="float64")
    started = descale_realized_returns(equity, pd.Series([2.0, 2.0, 2.0], index=idx[1:], dtype="float64"))
    assert list(started.index) == list(idx[2:])
    with pytest.raises(DataIntegrityError, match="applied scale"):
        descale_realized_returns(equity, pd.Series([2.0, 2.0], index=[idx[1], idx[3]], dtype="float64"))
    with pytest.raises(DataIntegrityError, match="applied scale"):
        descale_realized_returns(equity, pd.Series([2.0, 0.0, 2.0], index=idx[:3], dtype="float64"))
    with pytest.raises(DataIntegrityError, match="applied scale"):
        descale_realized_returns(equity, pd.Series([2.0, np.nan, 2.0], index=idx[:3], dtype="float64"))


def test_decision_mark_row_reads_prior_hour_close_and_omits_missing(tmp_path) -> None:
    import pandas as pd
    from src.mhs.live_signal_step import decision_mark_row

    dt = pd.Timestamp("2026-09-05", tz="UTC")
    prior = dt - pd.Timedelta(hours=1)
    pd.DataFrame({"timestamp": [prior.value // 1_000_000, dt.value // 1_000_000], "close": [101.5, 999.0]}).to_parquet(tmp_path / "AAAUSDT.parquet", index=False)
    pd.DataFrame({"timestamp": [dt.value // 1_000_000], "close": [55.0]}).to_parquet(tmp_path / "BBBUSDT.parquet", index=False)
    pd.DataFrame({"timestamp": [prior.value // 1_000_000], "close": [0.0]}).to_parquet(tmp_path / "DDDUSDT.parquet", index=False)
    row = decision_mark_row(["AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT"], dt, lambda symbol: tmp_path / f"{symbol}.parquet")
    assert row.to_dict() == {"AAAUSDT": 101.5}
    assert row.name == dt
    assert row.dtype == "float64"


def test_assert_panel_history_available_fails_closed_when_short(tmp_path) -> None:
    import pandas as pd
    import pytest
    from src.common.errors import DataIntegrityError
    from src.mhs.live_signal_step import PANEL_HISTORY_REFERENCE_SYMBOL, _assert_panel_history_available

    assert PANEL_HISTORY_REFERENCE_SYMBOL == "BTCUSDT"
    (tmp_path / "1h").mkdir()
    first = pd.Timestamp("2026-01-10", tz="UTC")
    pd.DataFrame({"timestamp": [first.value // 1_000_000], "close": [1.0]}).to_parquet(tmp_path / "1h" / "BTCUSDT.parquet", index=False)
    _assert_panel_history_available(str(tmp_path), first)
    with pytest.raises(DataIntegrityError, match="panel history"):
        _assert_panel_history_available(str(tmp_path), first - pd.Timedelta(hours=1))
    with pytest.raises(DataIntegrityError, match="panel history"):
        _assert_panel_history_available(str(tmp_path / "missing"), first)


def test_compute_signal_row_prescale_deadband_descaled_reference_and_placeholder(tmp_path, monkeypatch) -> None:
    import dataclasses
    import numpy as np
    import pandas as pd
    import pytest
    import src.mhs.live_signal_step as module
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.deployment_policy import build_deployment_policy
    from src.mhs.live_runtime import SCHEMA_VERSION, LiveRuntime
    from src.mhs.live_strategy import LiveStrategyParams
    from src.mhs.pipeline.config import MhsRunConfig

    request = MhsDiagnosticRequest(**dataclasses.asdict(MhsRunConfig()))
    policy = build_deployment_policy(request, slow_horizon_hours=168, committee_member_weights={"m": 1.0}, admitted_members=("m",), target_annual_vol=0.35, exposure_cap=3.0)
    params = LiveStrategyParams(schema_version=2, strategy_digest="d", backtest_window=(pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2025-12-31", tz="UTC")), created_at=pd.Timestamp("2026-09-01", tz="UTC"), policy=policy, bootstrap_sha256="a" * 64, bootstrap_held_row={})
    dt = pd.Timestamp("2026-09-05", tz="UTC")
    warm = pd.Series([0.01, -0.02], index=pd.date_range("2025-12-30", periods=2, freq="1D", tz="UTC"), dtype="float64")
    runtime = LiveRuntime(schema_version=SCHEMA_VERSION, params_digest="d", last_decision_date=dt - pd.Timedelta(days=1), held_target_row={"AAAUSDT": 0.10}, reference_daily_returns=warm)
    pre = pd.DataFrame({"AAAUSDT": [0.11], "BBBUSDT": [-0.50]}, index=pd.DatetimeIndex([dt]))
    equity = pd.Series([2000.0, 2200.0, 2090.0], index=pd.to_datetime(["2026-09-02", "2026-09-03", "2026-09-04"], utc=True), dtype="float64")
    applied = pd.Series([2.0, 1.0], index=pd.to_datetime(["2026-09-02", "2026-09-03"], utc=True), dtype="float64")
    captured: dict[str, object] = {}

    def fake_builder(*args, **kwargs):
        captured["apply_rebalance_deadband"] = kwargs["apply_rebalance_deadband"]
        captured["deadband_seed_row"] = kwargs.get("deadband_seed_row")
        return pre, pd.DatetimeIndex([dt + pd.Timedelta(hours=1)]), [], pd.DatetimeIndex([dt])

    def fake_scale(reference, sizing, *, warmup_returns=None):
        captured["reference"] = reference.copy()
        captured["warmup"] = warmup_returns
        return pd.Series(1.5, index=reference.index, dtype="float64")

    def fake_guard(data_root, panel_start, reference_symbol="BTCUSDT"):
        captured["panel_start"] = panel_start

    monkeypatch.setattr(module, "_build_fold_target_weights", fake_builder)
    monkeypatch.setattr(module, "_load_funding_by_symbol", lambda *_: {})
    monkeypatch.setattr(module, "_assert_panel_history_available", fake_guard)
    monkeypatch.setattr(module, "realized_equity", lambda *_a, **_k: equity)
    monkeypatch.setattr(module, "compute_exposure_scale", fake_scale)

    scaled, prescale, scalar = module.compute_signal_row(params, runtime, str(tmp_path), dt, portfolio_state_dir=tmp_path, mode="paper", applied_scale=applied)

    assert captured["apply_rebalance_deadband"] is False
    assert captured["deadband_seed_row"] is None
    assert captured["panel_start"] == dt - pd.Timedelta(days=policy.signal_window.panel_window_days)
    assert prescale["AAAUSDT"] == 0.10
    assert prescale["BBBUSDT"] == -0.50
    assert scalar == 1.5
    assert scaled["AAAUSDT"] == pytest.approx(0.15)
    assert scaled["BBBUSDT"] == pytest.approx(-0.75)
    reference = captured["reference"]
    assert reference.index[-1] == dt
    assert reference.iloc[-1] == 0.0
    assert list(reference.index[:-1]) == list(pd.to_datetime(["2026-09-03", "2026-09-04"], utc=True))
    np.testing.assert_allclose(reference.iloc[:-1].to_numpy(), [0.1 / 2.0, (2090.0 / 2200.0 - 1.0) / 1.0], rtol=0.0, atol=1e-15)
    pd.testing.assert_series_equal(captured["warmup"], warm)


def test_compute_signal_row_empty_forward_and_future_record_fail_closed(tmp_path, monkeypatch) -> None:
    import dataclasses
    import pandas as pd
    import pytest
    import src.mhs.live_signal_step as module
    from src.common.errors import DataIntegrityError
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.deployment_policy import build_deployment_policy
    from src.mhs.live_runtime import SCHEMA_VERSION, LiveRuntime
    from src.mhs.live_strategy import LiveStrategyParams
    from src.mhs.pipeline.config import MhsRunConfig

    request = MhsDiagnosticRequest(**dataclasses.asdict(MhsRunConfig()))
    policy = build_deployment_policy(request, slow_horizon_hours=168, committee_member_weights={"m": 1.0}, admitted_members=("m",), target_annual_vol=0.35, exposure_cap=3.0)
    params = LiveStrategyParams(schema_version=2, strategy_digest="d", backtest_window=(pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2025-12-31", tz="UTC")), created_at=pd.Timestamp("2026-09-01", tz="UTC"), policy=policy, bootstrap_sha256="a" * 64, bootstrap_held_row={})
    dt = pd.Timestamp("2026-09-05", tz="UTC")
    warm = pd.Series([0.01], index=pd.DatetimeIndex([pd.Timestamp("2025-12-31", tz="UTC")]), dtype="float64")
    runtime = LiveRuntime(schema_version=SCHEMA_VERSION, params_digest="d", last_decision_date=dt - pd.Timedelta(days=1), held_target_row={}, reference_daily_returns=warm)
    pre = pd.DataFrame({"AAAUSDT": [0.11]}, index=pd.DatetimeIndex([dt]))
    captured: dict[str, object] = {}

    def fake_scale(reference, sizing, *, warmup_returns=None):
        captured["reference"] = reference.copy()
        captured["warmup"] = warmup_returns
        return pd.Series(1.0, index=reference.index, dtype="float64")

    monkeypatch.setattr(module, "_build_fold_target_weights", lambda *a, **k: (pre, pd.DatetimeIndex([dt]), [], pd.DatetimeIndex([dt])))
    monkeypatch.setattr(module, "_load_funding_by_symbol", lambda *_: {})
    monkeypatch.setattr(module, "_assert_panel_history_available", lambda *_a, **_k: None)
    monkeypatch.setattr(module, "compute_exposure_scale", fake_scale)
    monkeypatch.setattr(module, "realized_equity", lambda *_a, **_k: pd.Series(dtype="float64"))

    scaled, prescale, scalar = module.compute_signal_row(params, runtime, str(tmp_path), dt, portfolio_state_dir=tmp_path, mode="paper")
    assert list(captured["reference"].index) == [dt]
    pd.testing.assert_series_equal(captured["warmup"], warm)
    assert prescale["AAAUSDT"] == 0.11
    assert scalar == 1.0

    future = pd.Series([2000.0, 2010.0], index=pd.DatetimeIndex([dt - pd.Timedelta(days=1), dt]), dtype="float64")
    monkeypatch.setattr(module, "realized_equity", lambda *_a, **_k: future)
    applied = pd.Series([1.0], index=pd.DatetimeIndex([dt - pd.Timedelta(days=1)]), dtype="float64")
    with pytest.raises(DataIntegrityError, match="precede"):
        module.compute_signal_row(params, runtime, str(tmp_path), dt, portfolio_state_dir=tmp_path, mode="paper", applied_scale=applied)


def test_advance_to_date_persists_prescale_held_scale_and_decision_marks(tmp_path, monkeypatch) -> None:
    import dataclasses
    import pandas as pd
    import src.mhs.live_signal_step as module
    from src.live.deployed_weights import EXPOSURE_SCALE_COLUMN, decision_marks_path, exposure_scale_path, load_weights_frame
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.deployment_policy import build_deployment_policy
    from src.mhs.live_runtime import SCHEMA_VERSION, LiveRuntime
    from src.mhs.live_strategy import LiveStrategyParams
    from src.mhs.pipeline.config import MhsRunConfig

    request = MhsDiagnosticRequest(**dataclasses.asdict(MhsRunConfig()))
    policy = build_deployment_policy(request, slow_horizon_hours=168, committee_member_weights={"m": 1.0}, admitted_members=("m",), target_annual_vol=0.35, exposure_cap=3.0)
    params = LiveStrategyParams(schema_version=2, strategy_digest="d", backtest_window=(pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2025-12-31", tz="UTC")), created_at=pd.Timestamp("2026-09-01", tz="UTC"), policy=policy, bootstrap_sha256="a" * 64, bootstrap_held_row={})
    d1 = pd.Timestamp("2026-09-05", tz="UTC")
    d2 = d1 + pd.Timedelta(days=1)
    runtime = LiveRuntime(schema_version=SCHEMA_VERSION, params_digest="d", last_decision_date=d1 - pd.Timedelta(days=1), held_target_row={}, reference_daily_returns=pd.Series(dtype="float64"))
    seen_scales: list[pd.Series] = []

    def fake_compute(p, rt, root, date, *, portfolio_state_dir=None, mode="shadow", applied_scale=None, quarantine=None):
        seen_scales.append(applied_scale.copy())
        return pd.Series({"AAAUSDT": 0.2}, name=date), pd.Series({"AAAUSDT": 0.1}, name=date), 2.0

    monkeypatch.setattr(module, "compute_signal_row", fake_compute)
    monkeypatch.setattr(module, "decision_mark_row", lambda symbols, date, mark_path_fn: pd.Series({"AAAUSDT": 101.0}, name=date, dtype="float64"))
    weights_path = tmp_path / "deployed_target_weights.parquet"

    new_rt, appended, scalar = module.advance_to_date(params, runtime, weights_path, "", d1)
    assert appended == 1 and scalar == 2.0
    assert new_rt.held_target_row == {"AAAUSDT": 0.1}
    scale_frame = load_weights_frame(exposure_scale_path(weights_path))
    assert float(scale_frame.loc[d1, EXPOSURE_SCALE_COLUMN]) == 2.0
    marks_frame = load_weights_frame(decision_marks_path(weights_path))
    assert float(marks_frame.loc[d1, "AAAUSDT"]) == 101.0
    assert seen_scales[0].empty

    module.advance_to_date(params, new_rt, weights_path, "", d2)
    assert float(seen_scales[1].loc[d1]) == 2.0


def test_assert_panel_history_empty_file_fails_closed(tmp_path) -> None:
    import pandas as pd
    import pytest
    from src.common.errors import DataIntegrityError
    from src.mhs.live_signal_step import _assert_panel_history_available

    (tmp_path / "1h").mkdir()
    pd.DataFrame({"timestamp": [], "close": []}).to_parquet(tmp_path / "1h" / "BTCUSDT.parquet", index=False)
    with pytest.raises(DataIntegrityError, match="panel history"):
        _assert_panel_history_available(str(tmp_path), pd.Timestamp("2026-01-10", tz="UTC"))


def test_descale_no_overlapping_scale_returns_empty() -> None:
    import pandas as pd
    from src.mhs.live_signal_step import descale_realized_returns

    idx = pd.to_datetime(["2026-09-01", "2026-09-02", "2026-09-03"], utc=True)
    equity = pd.Series([100.0, 101.0, 102.0], index=idx, dtype="float64")
    applied = pd.Series([2.0], index=pd.to_datetime(["2026-08-01"], utc=True), dtype="float64")
    out = descale_realized_returns(equity, applied)
    assert out.empty and isinstance(out.index, pd.DatetimeIndex)


def test_realized_equity_accepts_naive_bt_end(tmp_path) -> None:
    import pandas as pd
    from src.mhs.live_signal_step import realized_equity

    df = pd.DataFrame(
        {
            "decision_time": pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
            "mode": ["paper", "paper"],
            "equity_usdt": [2000.0, 2100.0],
        }
    )
    d = tmp_path / "ps"
    d.mkdir()
    df.to_parquet(d / "active.parquet", index=False)
    out = realized_equity(d, "paper", bt_end=pd.Timestamp("2025-12-31"))
    assert list(out.index) == list(pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True))


def test_advance_to_date_gap_branch_persists_prescale(tmp_path, monkeypatch) -> None:
    import dataclasses
    import pandas as pd
    import src.mhs.live_signal_step as module
    from src.live.deployed_weights import EXPOSURE_SCALE_COLUMN, exposure_scale_path, load_weights_frame
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.deployment_policy import build_deployment_policy
    from src.mhs.live_runtime import SCHEMA_VERSION, LiveRuntime
    from src.mhs.live_strategy import LiveStrategyParams
    from src.mhs.pipeline.config import MhsRunConfig

    request = MhsDiagnosticRequest(**dataclasses.asdict(MhsRunConfig()))
    policy = build_deployment_policy(request, slow_horizon_hours=168, committee_member_weights={"m": 1.0}, admitted_members=("m",), target_annual_vol=0.35, exposure_cap=3.0)
    params = LiveStrategyParams(schema_version=2, strategy_digest="d", backtest_window=(pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2025-12-31", tz="UTC")), created_at=pd.Timestamp("2026-09-01", tz="UTC"), policy=policy, bootstrap_sha256="a" * 64, bootstrap_held_row={})
    d1 = pd.Timestamp("2026-09-05", tz="UTC")
    far = d1 + pd.Timedelta(days=40)
    runtime = LiveRuntime(schema_version=SCHEMA_VERSION, params_digest="d", last_decision_date=d1 - pd.Timedelta(days=1), held_target_row={}, reference_daily_returns=pd.Series(dtype="float64"))

    def fake_compute(p, rt, root, date, *, portfolio_state_dir=None, mode="shadow", applied_scale=None, quarantine=None):
        return pd.Series({"AAAUSDT": 0.2}, name=date), pd.Series({"AAAUSDT": 0.1}, name=date), 2.0

    monkeypatch.setattr(module, "compute_signal_row", fake_compute)
    monkeypatch.setattr(module, "decision_mark_row", lambda symbols, date, mark_path_fn: pd.Series({"AAAUSDT": 101.0}, name=date, dtype="float64"))
    weights_path = tmp_path / "deployed_target_weights.parquet"

    new_rt, appended, scalar = module.advance_to_date(params, runtime, weights_path, "", far, max_catchup_days=30)
    assert appended == 1 and scalar == 2.0
    assert new_rt.last_decision_date == far
    assert new_rt.held_target_row == {"AAAUSDT": 0.1}
    scale_frame = load_weights_frame(exposure_scale_path(weights_path))
    assert float(scale_frame.loc[far, EXPOSURE_SCALE_COLUMN]) == 2.0


# --- auto appended from contract: signal_input_quarantine ---


def test_signal_quarantine_protects_held_and_reference_symbols() -> None:
    import pandas as pd
    from src.mhs.live_runtime import LiveRuntime
    from src.mhs.live_signal_step import _signal_quarantine

    runtime = LiveRuntime(
        schema_version=1, params_digest="d", last_decision_date=pd.Timestamp("2026-09-04", tz="UTC"),
        held_target_row={"ETHUSDT": -0.2, "XRPUSDT": 0.0}, reference_daily_returns=pd.Series(dtype="float64"),
    )

    quarantine = _signal_quarantine(runtime)

    assert quarantine.protected == frozenset({"ETHUSDT", "BTCUSDT"})
    assert quarantine.records == []


def test_compute_signal_row_threads_quarantine_to_funding_and_panel(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import src.mhs.live_signal_step as m
    from src.mhs.live_runtime import LiveRuntime
    from src.mhs.panel import PanelQuarantine

    dt = pd.Timestamp("2026-08-31", tz="UTC")
    tw = pd.DataFrame([[1.0]], index=pd.DatetimeIndex([dt]), columns=["BTCUSDT"])
    captured: dict[str, object] = {}

    def _funding(root, quarantine):
        captured["funding"] = quarantine
        return {}

    def _builder(*args, **kwargs):
        captured["panel"] = kwargs["panel_quarantine"]
        return tw, pd.DatetimeIndex([dt]), [], pd.DatetimeIndex([dt])

    monkeypatch.setattr(m, "_load_funding_by_symbol", _funding)
    monkeypatch.setattr(m, "_build_fold_target_weights", _builder)
    monkeypatch.setattr(m, "_assert_panel_history_available", lambda *a, **k: None)
    monkeypatch.setattr(m, "realized_equity", lambda *a, **k: pd.Series(dtype="float64"))
    monkeypatch.setattr(
        m, "compute_exposure_scale",
        lambda reference, sizing, *, warmup_returns=None: pd.Series(1.0, index=reference.index, dtype="float64"),
    )
    params = _v2_sig_params("growth_budget", 1.0, False, False)
    rt = LiveRuntime(
        schema_version=1, params_digest="d", last_decision_date=pd.Timestamp("2026-08-30", tz="UTC"),
        held_target_row={}, reference_daily_returns=pd.Series(dtype="float64"),
    )
    quarantine = PanelQuarantine(protected=frozenset({"BTCUSDT"}))

    m.compute_signal_row(params, rt, str(tmp_path), dt, portfolio_state_dir=tmp_path, mode="paper", quarantine=quarantine)

    assert captured["funding"] is quarantine
    assert captured["panel"] is quarantine


def test_load_funding_by_symbol_quarantines_unreadable_funding(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import pytest
    import src.common.paths as paths_mod
    import src.market_data.storage.loaders as loaders_mod
    from src.common.errors import DataIntegrityError
    from src.mhs.live_signal_step import _load_funding_by_symbol
    from src.mhs.panel import PanelQuarantine

    (tmp_path / "1h").mkdir()
    (tmp_path / "funding").mkdir()
    good = pd.Series([0.0001], index=pd.DatetimeIndex(["2026-09-01"], tz="UTC"), dtype="float64")

    def _loader(path):
        if "BAD" in str(path):
            raise ValueError("corrupt funding parquet")
        return good

    for sym in ("AAAUSDT", "BADUSDT"):
        pd.DataFrame({"timestamp": [0]}).to_parquet(tmp_path / "1h" / f"{sym}.parquet", index=False)
        pd.DataFrame({"timestamp": [0]}).to_parquet(tmp_path / "funding" / f"{sym}.parquet", index=False)
    monkeypatch.setattr(paths_mod, "funding_path", lambda sym: tmp_path / "funding" / f"{sym}.parquet")
    monkeypatch.setattr(loaders_mod, "load_funding_rates", _loader)
    quarantine = PanelQuarantine(protected=frozenset())

    out = _load_funding_by_symbol(str(tmp_path), quarantine)

    assert list(out) == ["AAAUSDT"]
    assert [(r.symbol, r.reason) for r in quarantine.records] == [("BADUSDT", "funding_unreadable:ValueError")]


def test_load_funding_by_symbol_without_quarantine_fails_closed(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import pytest
    import src.common.paths as paths_mod
    import src.market_data.storage.loaders as loaders_mod
    from src.common.errors import DataIntegrityError
    from src.mhs.live_signal_step import _load_funding_by_symbol
    from src.mhs.panel import PanelQuarantine

    (tmp_path / "1h").mkdir()
    (tmp_path / "funding").mkdir()
    good = pd.Series([0.0001], index=pd.DatetimeIndex(["2026-09-01"], tz="UTC"), dtype="float64")

    def _loader(path):
        if "BAD" in str(path):
            raise ValueError("corrupt funding parquet")
        return good

    pd.DataFrame({"timestamp": [0]}).to_parquet(tmp_path / "1h" / "BADUSDT.parquet", index=False)
    pd.DataFrame({"timestamp": [0]}).to_parquet(tmp_path / "funding" / "BADUSDT.parquet", index=False)
    monkeypatch.setattr(paths_mod, "funding_path", lambda sym: tmp_path / "funding" / f"{sym}.parquet")
    monkeypatch.setattr(loaders_mod, "load_funding_rates", _loader)

    with pytest.raises(DataIntegrityError, match="funding unreadable for BADUSDT"):
        _load_funding_by_symbol(str(tmp_path))


def test_load_funding_by_symbol_fallback_quarantines_unreadable_funding(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import pytest
    import src.common.paths as paths_mod
    import src.market_data.storage.loaders as loaders_mod
    from src.common.errors import DataIntegrityError
    from src.mhs.live_signal_step import _load_funding_by_symbol
    from src.mhs.panel import PanelQuarantine

    (tmp_path / "1h").mkdir()
    (tmp_path / "funding").mkdir()
    good = pd.Series([0.0001], index=pd.DatetimeIndex(["2026-09-01"], tz="UTC"), dtype="float64")

    def _loader(path):
        if "BAD" in str(path):
            raise ValueError("corrupt funding parquet")
        return good

    pd.DataFrame({"timestamp": [0]}).to_parquet(tmp_path / "1h" / "CCCUSDT.parquet", index=False)
    for sym in ("DDDUSDT", "BADUSDT"):
        pd.DataFrame({"timestamp": [0]}).to_parquet(tmp_path / "funding" / f"{sym}.parquet", index=False)
    monkeypatch.setattr(paths_mod, "funding_path", lambda sym: tmp_path / "nofunding" / f"{sym}.parquet")
    monkeypatch.setattr(loaders_mod, "load_funding_rates", _loader)
    quarantine = PanelQuarantine(protected=frozenset())

    out = _load_funding_by_symbol(str(tmp_path), quarantine)

    assert list(out) == ["DDDUSDT"]
    assert [(r.symbol, r.reason) for r in quarantine.records] == [("BADUSDT", "funding_unreadable:ValueError")]


def test_load_funding_by_symbol_fallback_without_quarantine_fails_closed(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import pytest
    import src.common.paths as paths_mod
    import src.market_data.storage.loaders as loaders_mod
    from src.common.errors import DataIntegrityError
    from src.mhs.live_signal_step import _load_funding_by_symbol
    from src.mhs.panel import PanelQuarantine

    (tmp_path / "1h").mkdir()
    (tmp_path / "funding").mkdir()
    good = pd.Series([0.0001], index=pd.DatetimeIndex(["2026-09-01"], tz="UTC"), dtype="float64")

    def _loader(path):
        if "BAD" in str(path):
            raise ValueError("corrupt funding parquet")
        return good

    pd.DataFrame({"timestamp": [0]}).to_parquet(tmp_path / "1h" / "CCCUSDT.parquet", index=False)
    pd.DataFrame({"timestamp": [0]}).to_parquet(tmp_path / "funding" / "BADUSDT.parquet", index=False)
    monkeypatch.setattr(paths_mod, "funding_path", lambda sym: tmp_path / "nofunding" / f"{sym}.parquet")
    monkeypatch.setattr(loaders_mod, "load_funding_rates", _loader)

    with pytest.raises(DataIntegrityError, match="funding unreadable for BADUSDT"):
        _load_funding_by_symbol(str(tmp_path))


def test_load_funding_by_symbol_fallback_uses_lake_funding_dir_when_root_has_none(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import pytest
    import src.common.paths as paths_mod
    import src.market_data.storage.loaders as loaders_mod
    from src.common.errors import DataIntegrityError
    from src.mhs.live_signal_step import _load_funding_by_symbol
    from src.mhs.panel import PanelQuarantine

    (tmp_path / "1h").mkdir()
    (tmp_path / "funding").mkdir()
    good = pd.Series([0.0001], index=pd.DatetimeIndex(["2026-09-01"], tz="UTC"), dtype="float64")

    def _loader(path):
        if "BAD" in str(path):
            raise ValueError("corrupt funding parquet")
        return good

    (tmp_path / "funding").rmdir()
    lake = tmp_path / "lake"
    (lake / "funding").mkdir(parents=True)
    pd.DataFrame({"timestamp": [0]}).to_parquet(tmp_path / "1h" / "CCCUSDT.parquet", index=False)
    pd.DataFrame({"timestamp": [0]}).to_parquet(lake / "funding" / "EEEUSDT.parquet", index=False)
    monkeypatch.setattr(paths_mod, "funding_path", lambda sym: tmp_path / "nofunding" / f"{sym}.parquet")
    monkeypatch.setattr(paths_mod, "FUTURES_DATA_DIR", lake)
    monkeypatch.setattr(loaders_mod, "load_funding_rates", _loader)

    out = _load_funding_by_symbol(str(tmp_path), PanelQuarantine(protected=frozenset()))

    assert list(out) == ["EEEUSDT"]


def test_realized_equity_unreadable_shard_fails_closed_with_named_error(tmp_path) -> None:
    import pandas as pd
    import pytest
    from src.common.errors import DataIntegrityError
    from src.mhs.live_signal_step import realized_equity

    d = tmp_path / "ps"
    d.mkdir()
    (d / "active.parquet").write_bytes(b"not a parquet")

    with pytest.raises(DataIntegrityError, match="portfolio state shard unreadable"):
        realized_equity(d, "paper", bt_end=pd.Timestamp("2025-12-31", tz="UTC"))


def test_decision_mark_row_logs_each_omission_reason(tmp_path, caplog) -> None:
    import logging
    import pandas as pd
    from src.mhs.live_signal_step import decision_mark_row

    dt = pd.Timestamp("2026-09-05", tz="UTC")
    prior_ms = (dt - pd.Timedelta(hours=1)).value // 1_000_000
    pd.DataFrame({"timestamp": [prior_ms], "close": [101.5]}).to_parquet(tmp_path / "OKUSDT.parquet", index=False)
    (tmp_path / "BADUSDT.parquet").write_bytes(b"not a parquet")
    pd.DataFrame({"timestamp": [dt.value // 1_000_000], "close": [55.0]}).to_parquet(tmp_path / "LATEUSDT.parquet", index=False)
    pd.DataFrame({"timestamp": [prior_ms], "close": [0.0]}).to_parquet(tmp_path / "ZEROUSDT.parquet", index=False)

    with caplog.at_level(logging.WARNING, logger="LiveSignalStep"):
        row = decision_mark_row(
            ["OKUSDT", "MISSINGUSDT", "BADUSDT", "LATEUSDT", "ZEROUSDT"], dt,
            lambda symbol: tmp_path / f"{symbol}.parquet",
        )

    assert row.to_dict() == {"OKUSDT": 101.5}
    text = caplog.text
    assert "symbol=MISSINGUSDT reason=file_missing" in text
    assert "symbol=BADUSDT reason=unreadable:ArrowInvalid" in text
    assert "symbol=LATEUSDT reason=bar_missing" in text
    assert "symbol=ZEROUSDT reason=invalid_close" in text


def test_advance_to_date_writes_quarantine_sidecar_with_protected_quarantine(tmp_path, monkeypatch) -> None:
    import dataclasses
    import json
    import pandas as pd
    import src.mhs.live_signal_step as module
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.deployment_policy import build_deployment_policy
    from src.mhs.live_runtime import SCHEMA_VERSION, LiveRuntime
    from src.mhs.live_strategy import LiveStrategyParams
    from src.mhs.pipeline.config import MhsRunConfig

    request = MhsDiagnosticRequest(**dataclasses.asdict(MhsRunConfig()))
    policy = build_deployment_policy(request, slow_horizon_hours=168, committee_member_weights={"m": 1.0}, admitted_members=("m",), target_annual_vol=0.35, exposure_cap=3.0)
    params = LiveStrategyParams(schema_version=2, strategy_digest="d", backtest_window=(pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2025-12-31", tz="UTC")), created_at=pd.Timestamp("2026-09-01", tz="UTC"), policy=policy, bootstrap_sha256="a" * 64, bootstrap_held_row={})
    d1 = pd.Timestamp("2026-09-05", tz="UTC")
    runtime = LiveRuntime(schema_version=SCHEMA_VERSION, params_digest="d", last_decision_date=d1 - pd.Timedelta(days=1), held_target_row={"AAAUSDT": 0.1, "FLATUSDT": 0.0}, reference_daily_returns=pd.Series(dtype="float64"))
    seen: dict[str, object] = {}

    def fake_compute(p, rt, root, date, *, portfolio_state_dir=None, mode="shadow", applied_scale=None, quarantine=None):
        seen["protected"] = quarantine.protected
        quarantine.add("ZZZUSDT", "decision_bar_missing")
        return pd.Series({"AAAUSDT": 0.2}, name=date), pd.Series({"AAAUSDT": 0.1}, name=date), 2.0

    monkeypatch.setattr(module, "compute_signal_row", fake_compute)
    monkeypatch.setattr(module, "decision_mark_row", lambda symbols, date, mark_path_fn: pd.Series({"AAAUSDT": 101.0}, name=date, dtype="float64"))
    weights_path = tmp_path / "deployed_target_weights.parquet"

    module.advance_to_date(params, runtime, weights_path, "", d1)

    sidecar = json.loads((tmp_path / "signal_quarantine.json").read_text(encoding="utf-8"))
    assert seen["protected"] == frozenset({"AAAUSDT", "BTCUSDT"})
    assert sidecar == {
        "decision_time": "2026-09-05T00:00:00+00:00",
        "records": [{"symbol": "ZZZUSDT", "reason": "decision_bar_missing"}],
    }
    assert module.quarantine_sidecar_path(weights_path) == tmp_path / "signal_quarantine.json"


def test_advance_to_date_gap_branch_writes_quarantine_sidecar(tmp_path, monkeypatch) -> None:
    import dataclasses
    import json
    import pandas as pd
    import src.mhs.live_signal_step as module
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.deployment_policy import build_deployment_policy
    from src.mhs.live_runtime import SCHEMA_VERSION, LiveRuntime
    from src.mhs.live_strategy import LiveStrategyParams
    from src.mhs.pipeline.config import MhsRunConfig

    request = MhsDiagnosticRequest(**dataclasses.asdict(MhsRunConfig()))
    policy = build_deployment_policy(request, slow_horizon_hours=168, committee_member_weights={"m": 1.0}, admitted_members=("m",), target_annual_vol=0.35, exposure_cap=3.0)
    params = LiveStrategyParams(schema_version=2, strategy_digest="d", backtest_window=(pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2025-12-31", tz="UTC")), created_at=pd.Timestamp("2026-09-01", tz="UTC"), policy=policy, bootstrap_sha256="a" * 64, bootstrap_held_row={})
    d1 = pd.Timestamp("2026-09-05", tz="UTC")
    runtime = LiveRuntime(schema_version=SCHEMA_VERSION, params_digest="d", last_decision_date=d1 - pd.Timedelta(days=1), held_target_row={"AAAUSDT": 0.1, "FLATUSDT": 0.0}, reference_daily_returns=pd.Series(dtype="float64"))
    seen: dict[str, object] = {}

    def fake_compute(p, rt, root, date, *, portfolio_state_dir=None, mode="shadow", applied_scale=None, quarantine=None):
        seen["protected"] = quarantine.protected
        quarantine.add("ZZZUSDT", "decision_bar_missing")
        return pd.Series({"AAAUSDT": 0.2}, name=date), pd.Series({"AAAUSDT": 0.1}, name=date), 2.0

    monkeypatch.setattr(module, "compute_signal_row", fake_compute)
    monkeypatch.setattr(module, "decision_mark_row", lambda symbols, date, mark_path_fn: pd.Series({"AAAUSDT": 101.0}, name=date, dtype="float64"))
    weights_path = tmp_path / "deployed_target_weights.parquet"

    far = d1 + pd.Timedelta(days=40)

    module.advance_to_date(params, runtime, weights_path, "", far, max_catchup_days=30)

    sidecar = json.loads((tmp_path / "signal_quarantine.json").read_text(encoding="utf-8"))
    assert seen["protected"] == frozenset({"AAAUSDT", "BTCUSDT"})
    assert sidecar["decision_time"] == far.isoformat()
    assert sidecar["records"] == [{"symbol": "ZZZUSDT", "reason": "decision_bar_missing"}]
    assert not list(tmp_path.glob("*.tmp"))


def test_compute_signal_row_threads_params_data_policy_to_fold_builder(tmp_path, monkeypatch) -> None:
    import dataclasses

    import numpy as np
    import pandas as pd

    import src.mhs.live_signal_step as m
    from src.mhs.live_runtime import LiveRuntime

    dt = pd.Timestamp("2026-08-31", tz="UTC")
    tw = pd.DataFrame([[0.6, -0.4]], index=pd.DatetimeIndex([dt]), columns=["BTCUSDT", "ETHUSDT"])
    captured: dict[str, object] = {}

    def _fake_builder(root, fold, request, *a, **k):
        captured["request"] = request
        return (tw, pd.DatetimeIndex([dt]), [], pd.DatetimeIndex([dt]))

    monkeypatch.setattr(m, "_build_fold_target_weights", _fake_builder)
    monkeypatch.setattr(m, "_load_funding_by_symbol", lambda *a, **k: {})
    monkeypatch.setattr(m, "_assert_panel_history_available", lambda *a, **k: None)
    rng = pd.date_range("2026-06-01", periods=60, freq="1D", tz="UTC")
    fwd = pd.Series(np.r_[np.full(30, 0.05), np.full(30, -0.05)], index=rng)
    monkeypatch.setattr(m, "realized_equity", lambda *a, **k: pd.Series(dtype="float64"))
    monkeypatch.setattr(m, "descale_realized_returns", lambda equity, scale: fwd)
    boot = pd.Series(np.full(120, 0.001), index=pd.date_range("2025-09-01", periods=120, freq="1D", tz="UTC"))
    params = dataclasses.replace(_v2_sig_params("growth_budget", 1.0, False, False), data_policy="zombie_mask_v1")
    rt = LiveRuntime(
        schema_version=1, params_digest="d", last_decision_date=pd.Timestamp("2026-08-30", tz="UTC"),
        held_target_row={"BTCUSDT": 0.5}, reference_daily_returns=boot,
    )

    m.compute_signal_row(params, rt, str(tmp_path), dt, portfolio_state_dir=tmp_path, mode="paper")

    assert captured["request"].data_policy == "zombie_mask_v1"


def test_descale_realized_returns_skips_non_daily_steps() -> None:
    import numpy as np
    import pandas as pd
    from src.mhs.live_signal_step import descale_realized_returns

    # Given: 09-02가 HALT로 빠진 실현 equity (09-01, 09-03, 09-04)
    idx = pd.to_datetime(["2026-09-01", "2026-09-03", "2026-09-04"], utc=True)
    equity = pd.Series([100.0, 121.0, 132.0], index=idx, dtype="float64")
    applied = pd.Series([2.0, 2.0, 2.0], index=idx, dtype="float64")

    # When
    out = descale_realized_returns(equity, applied)

    # Then: 09-01 -> 09-03 (2일)은 배제, 09-03 -> 09-04 (1일)만 방출
    assert list(out.index) == [pd.Timestamp("2026-09-04", tz="UTC")]
    np.testing.assert_allclose(
        out.to_numpy(), [(132.0 / 121.0 - 1.0) / 2.0], rtol=0.0, atol=1e-15
    )
    assert out.dtype == np.float64


def test_descale_realized_returns_preserves_contiguous_daily_series() -> None:
    import numpy as np
    import pandas as pd
    from src.mhs.live_signal_step import descale_realized_returns

    # Given: 결손 없는 일간 equity
    idx = pd.to_datetime(["2026-09-01", "2026-09-02", "2026-09-03"], utc=True)
    equity = pd.Series([100.0, 110.0, 99.0], index=idx, dtype="float64")
    applied = pd.Series([2.0, 1.0], index=idx[:2], dtype="float64")

    # When
    out = descale_realized_returns(equity, applied)

    # Then: 모든 후속 행 유지
    assert list(out.index) == list(idx[1:])
    np.testing.assert_allclose(
        out.to_numpy(),
        [0.10 / 2.0, (99.0 / 110.0 - 1.0) / 1.0],
        rtol=0.0,
        atol=1e-15,
    )


def test_descale_realized_returns_gap_does_not_raise_and_recovers() -> None:
    import pandas as pd
    from src.mhs.live_signal_step import descale_realized_returns

    # Given: 2025-12-31 이후 8개월 공백, 이후 3일 연속
    idx = pd.to_datetime(
        ["2025-12-31", "2026-09-01", "2026-09-02", "2026-09-03"], utc=True
    )
    equity = pd.Series([100.0, 150.0, 151.0, 152.0], index=idx, dtype="float64")
    applied = pd.Series([1.0, 1.0, 1.0, 1.0], index=idx, dtype="float64")

    # When: 예외 없이 통과
    out = descale_realized_returns(equity, applied)

    # Then: 공백 쌍만 배제되고 재개 구간은 살아 있다
    assert list(out.index) == [
        pd.Timestamp("2026-09-02", tz="UTC"),
        pd.Timestamp("2026-09-03", tz="UTC"),
    ]
    assert out.notna().all()


def test_warmup_reference_gap_days_measures_calendar_gap() -> None:
    import pandas as pd
    import pytest
    from src.mhs.live_signal_step import warmup_reference_gap_days

    # Given / When / Then: 빈 워밍업
    empty = pd.Series(dtype="float64", index=pd.DatetimeIndex([], tz="UTC"))
    assert warmup_reference_gap_days(empty, pd.Timestamp("2026-09-16", tz="UTC")) is None

    # Given: 2025-12-31에 끝나는 봉인 부트스트랩
    warm = pd.Series(
        [0.001, 0.002],
        index=pd.to_datetime(["2025-12-30", "2025-12-31"], utc=True),
        dtype="float64",
    )

    # Then: 인접 일간은 1일
    assert warmup_reference_gap_days(
        warm, pd.Timestamp("2026-01-01", tz="UTC")
    ) == pytest.approx(1.0)

    # Then: 실제 봉인 번들 공백
    assert warmup_reference_gap_days(
        warm, pd.Timestamp("2026-09-16", tz="UTC")
    ) == pytest.approx(259.0)

    # Then: tz-naive 참조는 UTC로 해석
    assert warmup_reference_gap_days(
        warm, pd.Timestamp("2026-01-01")
    ) == pytest.approx(1.0)


def test_compute_signal_row_discloses_warmup_gap_without_halting(tmp_path, monkeypatch, caplog) -> None:
    import logging

    import numpy as np
    import pandas as pd
    import src.mhs.live_signal_step as m
    from src.mhs.live_runtime import LiveRuntime

    # Given: 2025-12-28에 끝나는 부트스트랩, 결정일은 2026-08-31 (246일 공백)
    dt = pd.Timestamp("2026-08-31", tz="UTC")
    tw = pd.DataFrame([[1.0]], index=pd.DatetimeIndex([dt]), columns=["BTCUSDT"])
    monkeypatch.setattr(
        m,
        "_build_fold_target_weights",
        lambda *a, **k: (tw, pd.DatetimeIndex([dt]), [], pd.DatetimeIndex([dt])),
    )
    monkeypatch.setattr(m, "_load_funding_by_symbol", lambda *a, **k: {})
    monkeypatch.setattr(m, "_assert_panel_history_available", lambda *a, **k: None)
    monkeypatch.setattr(m, "realized_equity", lambda *a, **k: pd.Series(dtype="float64"))
    boot = pd.Series(
        np.full(150, 0.002),
        index=pd.date_range("2025-08-01", periods=150, freq="1D", tz="UTC"),
        dtype="float64",
    )
    params = _v2_sig_params("growth_budget", 1.0, False, False)
    rt = LiveRuntime(
        schema_version=1,
        params_digest="d",
        last_decision_date=pd.Timestamp("2026-08-30", tz="UTC"),
        held_target_row={},
        reference_daily_returns=boot,
    )

    # When
    with caplog.at_level(logging.WARNING, logger="LiveSignalStep"):
        scaled, _prescale, scalar = m.compute_signal_row(
            params, rt, str(tmp_path), dt, portfolio_state_dir=tmp_path, mode="paper"
        )

    # Then: 매매는 계속되고(예외 없음) 공백은 공시된다
    assert np.isfinite(scaled["BTCUSDT"])
    assert 0.0 < scalar <= 3.0
    assert any("warmup_gap_days" in rec.getMessage() for rec in caplog.records)


def test_compute_signal_row_silent_when_warmup_is_contiguous(tmp_path, monkeypatch, caplog) -> None:
    import logging

    import numpy as np
    import pandas as pd
    import src.mhs.live_signal_step as m
    from src.mhs.live_runtime import LiveRuntime

    # Given: 결정일 직전까지 이어지는 부트스트랩
    dt = pd.Timestamp("2026-08-31", tz="UTC")
    tw = pd.DataFrame([[1.0]], index=pd.DatetimeIndex([dt]), columns=["BTCUSDT"])
    monkeypatch.setattr(
        m,
        "_build_fold_target_weights",
        lambda *a, **k: (tw, pd.DatetimeIndex([dt]), [], pd.DatetimeIndex([dt])),
    )
    monkeypatch.setattr(m, "_load_funding_by_symbol", lambda *a, **k: {})
    monkeypatch.setattr(m, "_assert_panel_history_available", lambda *a, **k: None)
    monkeypatch.setattr(m, "realized_equity", lambda *a, **k: pd.Series(dtype="float64"))
    boot = pd.Series(
        np.full(150, 0.002),
        index=pd.date_range(end="2026-08-30", periods=150, freq="1D", tz="UTC"),
        dtype="float64",
    )
    params = _v2_sig_params("growth_budget", 1.0, False, False)
    rt = LiveRuntime(
        schema_version=1,
        params_digest="d",
        last_decision_date=pd.Timestamp("2026-08-30", tz="UTC"),
        held_target_row={},
        reference_daily_returns=boot,
    )

    # When
    with caplog.at_level(logging.WARNING, logger="LiveSignalStep"):
        _scaled, _prescale, scalar = m.compute_signal_row(
            params, rt, str(tmp_path), dt, portfolio_state_dir=tmp_path, mode="paper"
        )

    # Then: 경고 없음
    assert 0.0 < scalar <= 3.0
    assert not any("warmup_gap_days" in rec.getMessage() for rec in caplog.records)
