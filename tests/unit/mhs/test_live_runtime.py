# ruff: noqa
import dataclasses as _dc2

import pandas as _pd

from src.mhs.contracts import MhsDiagnosticRequest as _Req2
from src.mhs.deployment_policy import build_deployment_policy as _build2
from src.mhs.pipeline.config import MhsRunConfig as _Cfg2


def _v2_params(digest="d", held=None):
    _pol2 = _build2(_Req2(**_dc2.asdict(_Cfg2())), slow_horizon_hours=168, committee_member_weights={"m1": 1.0}, admitted_members=("m1",), target_annual_vol=0.35, exposure_cap=3.0)
    return __import__("src.mhs.live_strategy", fromlist=["LiveStrategyParams"]).LiveStrategyParams(schema_version=2, strategy_digest=digest, backtest_window=(_pd.Timestamp("2021-01-01", tz="UTC"), _pd.Timestamp("2026-06-30", tz="UTC")), created_at=_pd.Timestamp("2026-08-31", tz="UTC"), policy=_pol2, bootstrap_sha256="a" * 64, bootstrap_held_row=dict(held or {"BTCUSDT": 0.1}))

def test_bootstrap_runtime_from_params() -> None:
    import pandas as pd

    from src.mhs.live_runtime import bootstrap_runtime
    from src.mhs.live_strategy import LiveStrategyParams

    ref = pd.Series([0.01, -0.02, 0.005], index=pd.date_range("2026-06-27", periods=3, freq="1D", tz="UTC"))
    params = _v2_params("abc", {"BTCUSDT": 0.3})
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
        _polm = _build2(_Req2(**_dc2.asdict(_Cfg2())), slow_horizon_hours=168, committee_member_weights={m: 1.0 for m in members}, admitted_members=tuple(members), target_annual_vol=0.35, exposure_cap=3.0)
        from src.mhs.live_strategy import LiveStrategyParams as _LSP

        return _LSP(schema_version=2, strategy_digest=digest, backtest_window=(pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2026-06-30", tz="UTC")), created_at=pd.Timestamp("2026-08-31", tz="UTC"), policy=_polm, bootstrap_sha256="a" * 64, bootstrap_held_row={"m9": 0.9})

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
    params = _v2_params("d", {"BTCUSDT": 0.1})
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
    params = _v2_params("newdigest", {"m9": 0.9})

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
    params = _v2_params("nd", {"m9": 0.9})

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
    params = _v2_params("same", {})

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
    params = _v2_params("brandnew", {})

    out_rt, reason = reconcile_runtime_params(rt, params, ref)

    assert reason == "soft_swap"
    assert out_rt.params_digest == "brandnew"
    assert out_rt.held_target_row == {"BTCUSDT": 0.5}




def test_runtime_schema_v2_roundtrip_and_v1_migration_clears_held(tmp_path) -> None:
    import json
    import pandas as pd
    import pytest
    from src.common.errors import DataIntegrityError
    from src.mhs.live_runtime import SCHEMA_VERSION, LiveRuntime, load_or_bootstrap_runtime, save_runtime

    assert SCHEMA_VERSION == 2
    ref = pd.Series([0.01], index=pd.DatetimeIndex([pd.Timestamp("2025-12-31", tz="UTC")]), dtype="float64")
    runtime = LiveRuntime(schema_version=2, params_digest="d", last_decision_date=pd.Timestamp("2026-09-01", tz="UTC"), held_target_row={"AAAUSDT": 0.1}, reference_daily_returns=ref)
    path = save_runtime(tmp_path / "rt.json", runtime)
    loaded = load_or_bootstrap_runtime(path, None, pd.Series(dtype="float64"))
    assert loaded.schema_version == 2
    assert loaded.held_target_row == {"AAAUSDT": 0.1}

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["schema_version"] = 1
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps(raw), encoding="utf-8")
    migrated = load_or_bootstrap_runtime(legacy, None, pd.Series(dtype="float64"))
    assert migrated.schema_version == 2
    assert migrated.held_target_row == {}
    assert migrated.params_digest == "d"
    assert migrated.last_decision_date == runtime.last_decision_date
    pd.testing.assert_series_equal(migrated.reference_daily_returns, loaded.reference_daily_returns)

    raw["schema_version"] = 3
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="schema_version"):
        load_or_bootstrap_runtime(bad, None, pd.Series(dtype="float64"))


def test_runtime_corrupt_held_row_fails_closed(tmp_path) -> None:
    import json
    import pandas as pd
    import pytest
    from src.common.errors import DataIntegrityError
    from src.mhs.live_runtime import SCHEMA_VERSION, LiveRuntime, load_or_bootstrap_runtime, save_runtime

    ref = pd.Series([0.01], index=pd.DatetimeIndex([pd.Timestamp("2025-12-31", tz="UTC")]), dtype="float64")
    runtime = LiveRuntime(schema_version=SCHEMA_VERSION, params_digest="d", last_decision_date=pd.Timestamp("2026-09-01", tz="UTC"), held_target_row={"AAAUSDT": 0.1}, reference_daily_returns=ref)
    path = save_runtime(tmp_path / "rt.json", runtime)
    raw = json.loads(path.read_text(encoding="utf-8"))

    raw["held_target_row"] = {"AAAUSDT": "abc"}
    bad = tmp_path / "bad_held.json"
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="held_target_row"):
        load_or_bootstrap_runtime(bad, None, pd.Series(dtype="float64"))

    raw["held_target_row"] = {"AAAUSDT": True}
    bad_bool = tmp_path / "bad_held_bool.json"
    bad_bool.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="held_target_row"):
        load_or_bootstrap_runtime(bad_bool, None, pd.Series(dtype="float64"))
