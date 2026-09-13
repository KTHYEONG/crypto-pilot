# ruff: noqa
def test_strategy_params_roundtrip_sealed(tmp_path) -> None:
    import pandas as pd
    from pydantic import SecretStr

    from src.mhs.live_strategy import LiveStrategyParams, load_strategy_params, save_strategy_params

    import dataclasses as _dc
    from src.mhs.contracts import MhsDiagnosticRequest as _Req
    from src.mhs.deployment_policy import build_deployment_policy as _build
    from src.mhs.pipeline.config import MhsRunConfig as _Cfg

    _policy = _build(_Req(**_dc.asdict(_Cfg())), slow_horizon_hours=168, committee_member_weights={"m1": 0.5, "m2": 0.5}, admitted_members=("m1", "m2"), target_annual_vol=0.35, exposure_cap=3.0)
    params = LiveStrategyParams(
        schema_version=2,
        strategy_digest="",
        backtest_window=(pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2026-06-30", tz="UTC")),
        created_at=pd.Timestamp("2026-08-31", tz="UTC"),
        policy=_policy,
        bootstrap_sha256="a" * 64,
        bootstrap_held_row={"BTCUSDT": 0.2, "ETHUSDT": -0.1},
    )
    key = SecretStr("A" * 43 + "=")
    dest = save_strategy_params(tmp_path / "strategy_params.json", params, artifact_key=key)
    loaded = load_strategy_params(dest, artifact_key=key)
    assert loaded.policy.slow_horizon_hours == 168
    assert loaded.policy.admitted_members == ("m1", "m2")
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
    report = types.SimpleNamespace(status="COMPLETE", research_go=types.SimpleNamespace(eligible=False), blend=types.SimpleNamespace(target_weights=tw))
    with pytest.raises(DataIntegrityError) as exc:
        assert_deployment_eligible(report)
    assert "deployment ineligible" in str(exc.value)

def test_mhs_kelly_z0_live_snapshot_changes_seal_digest(monkeypatch) -> None:
    import dataclasses
    import pandas as pd

    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.deployment_policy import build_deployment_policy
    from src.mhs.live_strategy import LiveStrategyParams, _compute_strategy_digest
    from src.mhs.pipeline.config import MhsRunConfig

    request = MhsDiagnosticRequest(**dataclasses.asdict(MhsRunConfig()))
    policy = build_deployment_policy(request, slow_horizon_hours=168, committee_member_weights={"m": 1.0}, admitted_members=("m",), target_annual_vol=0.3, exposure_cap=3.0)
    assert policy.sizing.kelly_window_days == 42
    assert policy.sizing.kelly_fraction == 0.5
    assert policy.sizing.kelly_lcb_z == 0.0
    payload = {'schema_version': 2, 'backtest_window': (pd.Timestamp('2021-01-01', tz='UTC'), pd.Timestamp('2025-12-31', tz='UTC')), 'created_at': pd.Timestamp('2026-09-13T00:00:00+00:00', tz='UTC'), 'policy': policy, 'bootstrap_sha256': "a" * 64, 'bootstrap_held_row': {'BTCUSDT': 0.1}, 'strategy_digest': ''}
    baseline = _compute_strategy_digest(LiveStrategyParams(**payload))
    sealed_changed = dataclasses.replace(policy.sizing, kelly_lcb_z=0.5)
    policy_changed = dataclasses.replace(policy, sizing=sealed_changed)
    payload['policy'] = policy_changed
    changed = _compute_strategy_digest(LiveStrategyParams(**payload))
    assert changed != baseline



def test_strategy_params_v2_roundtrip_and_v1_rejected(tmp_path) -> None:
    import dataclasses
    import json
    import pandas as pd
    import pytest
    from src.common.errors import DataIntegrityError
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.deployment_policy import build_deployment_policy
    from src.mhs.live_strategy import LiveStrategyParams, load_strategy_params, save_strategy_params
    from src.mhs.pipeline.config import MhsRunConfig

    request = MhsDiagnosticRequest(**dataclasses.asdict(MhsRunConfig()))
    policy = build_deployment_policy(request, slow_horizon_hours=168, committee_member_weights={"m": 1.0}, admitted_members=("m",), target_annual_vol=0.35, exposure_cap=3.0)
    params = LiveStrategyParams(schema_version=2, strategy_digest="", backtest_window=(pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2026-06-30", tz="UTC")), created_at=pd.Timestamp("2026-09-13", tz="UTC"), policy=policy, bootstrap_sha256="a" * 64, bootstrap_held_row={"BTCUSDT": 0.2})
    path = save_strategy_params(tmp_path / "strategy_params.json", params)
    loaded = load_strategy_params(path)
    assert loaded.policy == policy
    assert loaded.bootstrap_sha256 == "a" * 64
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["schema_version"] = 1
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="schema_version"):
        load_strategy_params(legacy)




def test_strategy_bootstrap_hash_binds_exact_plaintext_artifact(tmp_path) -> None:
    import pandas as pd
    import pytest
    from src.common.errors import DataIntegrityError
    from src.mhs.live_strategy import load_strategy_bootstrap, save_strategy_bootstrap

    reference = pd.Series([0.01, -0.02, 0.03], index=pd.date_range("2026-01-01", periods=3, freq="1D", tz="UTC"), dtype="float64", name="reference_daily_return")
    path, digest = save_strategy_bootstrap(tmp_path / "strategy_bootstrap.parquet", reference)
    loaded = load_strategy_bootstrap(path, expected_sha256=digest)
    pd.testing.assert_series_equal(loaded, reference, check_exact=True)
    with pytest.raises(DataIntegrityError, match="bootstrap_sha256"):
        load_strategy_bootstrap(path, expected_sha256="0" * 64)



def _v2_valid_raw(tmp_path):
    import dataclasses
    import json
    import pandas as pd
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.deployment_policy import build_deployment_policy
    from src.mhs.live_strategy import LiveStrategyParams, save_strategy_params
    from src.mhs.pipeline.config import MhsRunConfig

    request = MhsDiagnosticRequest(**dataclasses.asdict(MhsRunConfig()))
    policy = build_deployment_policy(request, slow_horizon_hours=168, committee_member_weights={"m": 1.0}, admitted_members=("m",), target_annual_vol=0.35, exposure_cap=3.0)
    params = LiveStrategyParams(schema_version=2, strategy_digest="", backtest_window=(pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2026-06-30", tz="UTC")), created_at=pd.Timestamp("2026-09-13", tz="UTC"), policy=policy, bootstrap_sha256="a" * 64, bootstrap_held_row={"BTCUSDT": 0.2})
    path = save_strategy_params(tmp_path / "strategy_params.json", params)
    return json.loads(path.read_text(encoding="utf-8")), path


def test_strategy_params_v2_rejects_malformed_policy_and_params(tmp_path) -> None:
    import copy
    import json
    import pytest
    from src.common.errors import DataIntegrityError
    from src.mhs.live_strategy import load_strategy_params

    raw, _ = _v2_valid_raw(tmp_path)

    def _write(payload):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(payload), encoding="utf-8")
        return p

    bad = copy.deepcopy(raw)
    bad["extra_key"] = 1
    with pytest.raises(DataIntegrityError, match="unknown strategy params key"):
        load_strategy_params(_write(bad))
    bad = copy.deepcopy(raw)
    del bad["policy"]
    with pytest.raises(DataIntegrityError, match="missing keys"):
        load_strategy_params(_write(bad))
    bad = copy.deepcopy(raw)
    bad["bootstrap_sha256"] = "xyz"
    with pytest.raises(DataIntegrityError, match="bootstrap_sha256"):
        load_strategy_params(_write(bad))
    bad = copy.deepcopy(raw)
    bad["bootstrap_held_row"] = []
    with pytest.raises(DataIntegrityError, match="bootstrap_held_row"):
        load_strategy_params(_write(bad))
    bad = copy.deepcopy(raw)
    bad["policy"] = []
    with pytest.raises(DataIntegrityError, match="policy must be"):
        load_strategy_params(_write(bad))
    bad = copy.deepcopy(raw)
    bad["policy"]["unknown_section"] = 1
    with pytest.raises(DataIntegrityError, match="unknown policy key"):
        load_strategy_params(_write(bad))
    bad = copy.deepcopy(raw)
    del bad["policy"]["sizing"]
    with pytest.raises(DataIntegrityError, match="missing keys"):
        load_strategy_params(_write(bad))
    bad = copy.deepcopy(raw)
    bad["policy"]["sizing"] = []
    with pytest.raises(DataIntegrityError, match="sections must be objects"):
        load_strategy_params(_write(bad))
    with pytest.raises(DataIntegrityError, match="JSON object"):
        load_strategy_params(_write([1, 2]))


def test_strategy_params_v2_rejects_save_with_legacy_schema(tmp_path) -> None:
    import dataclasses
    import pandas as pd
    import pytest
    from src.common.errors import DataIntegrityError
    from src.mhs.contracts import MhsDiagnosticRequest
    from src.mhs.deployment_policy import build_deployment_policy
    from src.mhs.live_strategy import LiveStrategyParams, save_strategy_params
    from src.mhs.pipeline.config import MhsRunConfig

    request = MhsDiagnosticRequest(**dataclasses.asdict(MhsRunConfig()))
    policy = build_deployment_policy(request, slow_horizon_hours=168, committee_member_weights={"m": 1.0}, admitted_members=("m",), target_annual_vol=0.35, exposure_cap=3.0)
    params = LiveStrategyParams(schema_version=1, strategy_digest="", backtest_window=(pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2026-06-30", tz="UTC")), created_at=pd.Timestamp("2026-09-13", tz="UTC"), policy=policy, bootstrap_sha256="a" * 64, bootstrap_held_row={})
    with pytest.raises(DataIntegrityError, match="schema_version"):
        save_strategy_params(tmp_path / "strategy_params.json", params)


def test_strategy_params_v2_sealed_artifact_gates(tmp_path) -> None:
    import pytest
    from pydantic import SecretStr
    from src.common.errors import DataIntegrityError
    from src.live.errors import ArtifactSealError
    from src.mhs.live_strategy import load_strategy_params

    _, path = _v2_valid_raw(tmp_path)
    with pytest.raises(ArtifactSealError):
        load_strategy_params(path.with_name(path.name + ".enc"))
    key = SecretStr("A" * 43 + "=")
    with pytest.raises(DataIntegrityError, match="not found"):
        load_strategy_params(tmp_path / "missing.json.enc", artifact_key=key)


def test_strategy_bootstrap_rejects_invalid_series(tmp_path) -> None:
    import pandas as pd
    import pytest
    from src.mhs.live_strategy import save_strategy_bootstrap

    with pytest.raises(ValueError, match="must be a Series"):
        save_strategy_bootstrap(tmp_path / "b.parquet", [0.01])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        save_strategy_bootstrap(tmp_path / "b.parquet", pd.Series(dtype="float64"))
    naive = pd.Series([0.01], index=pd.DatetimeIndex(["2026-01-01"]), dtype="float64")
    with pytest.raises(ValueError, match="tz-aware"):
        save_strategy_bootstrap(tmp_path / "b.parquet", naive)
    dup_idx = pd.DatetimeIndex([pd.Timestamp("2026-01-01", tz="UTC")] * 2)
    with pytest.raises(ValueError, match="unique"):
        save_strategy_bootstrap(tmp_path / "b.parquet", pd.Series([0.01, 0.02], index=dup_idx, dtype="float64"))
    rev_idx = pd.DatetimeIndex([pd.Timestamp("2026-01-02", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC")])
    with pytest.raises(ValueError, match="increasing"):
        save_strategy_bootstrap(tmp_path / "b.parquet", pd.Series([0.01, 0.02], index=rev_idx, dtype="float64"))
    bad_idx = pd.date_range("2026-01-01", periods=2, freq="1D", tz="UTC")
    with pytest.raises(ValueError, match="finite"):
        save_strategy_bootstrap(tmp_path / "b.parquet", pd.Series([0.01, float("inf")], index=bad_idx, dtype="float64"))
    with pytest.raises(ValueError, match="DatetimeIndex"):
        save_strategy_bootstrap(tmp_path / "b.parquet", pd.Series([0.01], index=pd.RangeIndex(1), dtype="float64"))


def test_strategy_bootstrap_sealed_roundtrip_and_gates(tmp_path) -> None:
    import hashlib
    import pandas as pd
    import pytest
    from pydantic import SecretStr
    from src.common.errors import DataIntegrityError
    from src.live.errors import ArtifactSealError
    from src.mhs.live_strategy import load_strategy_bootstrap, save_strategy_bootstrap

    reference = pd.Series([0.01, -0.02, 0.03], index=pd.date_range("2026-01-01", periods=3, freq="1D", tz="UTC"), dtype="float64", name="reference_daily_return")
    key = SecretStr("A" * 43 + "=")
    enc_path, digest = save_strategy_bootstrap(tmp_path / "strategy_bootstrap.parquet", reference, artifact_key=key)
    assert str(enc_path).endswith(".enc")
    loaded = load_strategy_bootstrap(enc_path, expected_sha256=digest, artifact_key=key)
    pd.testing.assert_series_equal(loaded, reference, check_exact=True)
    with pytest.raises(ArtifactSealError):
        load_strategy_bootstrap(enc_path, expected_sha256=digest)
    with pytest.raises(DataIntegrityError, match="not found"):
        load_strategy_bootstrap(tmp_path / "missing.parquet", expected_sha256=digest)
    with pytest.raises(DataIntegrityError, match="64-char"):
        load_strategy_bootstrap(enc_path, expected_sha256="bogus", artifact_key=key)
    fallback = load_strategy_bootstrap(tmp_path / "strategy_bootstrap.parquet", expected_sha256=digest, artifact_key=key)
    pd.testing.assert_series_equal(fallback, reference, check_exact=True)
    raw_bytes = enc_path.read_bytes()
    (tmp_path / "copied.parquet").write_bytes(raw_bytes)
    copied = load_strategy_bootstrap(tmp_path / "copied.parquet", expected_sha256=digest, artifact_key=key)
    pd.testing.assert_series_equal(copied, reference, check_exact=True)
    (tmp_path / "garbage.parquet").write_bytes(b"not a parquet")
    with pytest.raises(DataIntegrityError, match="decode failed"):
        load_strategy_bootstrap(tmp_path / "garbage.parquet", expected_sha256=hashlib.sha256(b"not a parquet").hexdigest())


def test_strategy_bootstrap_rejects_corrupt_payloads(tmp_path) -> None:
    import hashlib
    import io
    import pandas as pd
    import pytest
    from src.common.errors import DataIntegrityError
    from src.mhs.live_strategy import load_strategy_bootstrap, save_strategy_bootstrap

    def _store(frame):
        buf = io.BytesIO()
        frame.to_parquet(buf, index=True)
        blob = buf.getvalue()
        p = tmp_path / "x.parquet"
        p.write_bytes(blob)
        return p, hashlib.sha256(blob).hexdigest()

    p, digest = _store(pd.DataFrame({"other": [1.0]}, index=pd.date_range("2026-01-01", periods=1, freq="1D", tz="UTC")))
    with pytest.raises(DataIntegrityError, match="missing column"):
        load_strategy_bootstrap(p, expected_sha256=digest)
    naive = pd.DataFrame({"reference_daily_return": [0.01]}, index=pd.DatetimeIndex(["2026-01-01"]))
    p, digest = _store(naive)
    with pytest.raises(DataIntegrityError, match="tz-aware"):
        load_strategy_bootstrap(p, expected_sha256=digest)
    bad_idx = pd.date_range("2026-01-01", periods=2, freq="1D", tz="UTC")
    p, digest = _store(pd.DataFrame({"reference_daily_return": [0.01, float("nan")]}, index=bad_idx))
    with pytest.raises(DataIntegrityError, match="finite"):
        load_strategy_bootstrap(p, expected_sha256=digest)
    ranged = pd.DataFrame({"reference_daily_return": [0.01]})
    p_ranged, digest_ranged = _store(ranged)
    with pytest.raises(DataIntegrityError, match="DatetimeIndex"):
        load_strategy_bootstrap(p_ranged, expected_sha256=digest_ranged)
    str_col = pd.DataFrame({"reference_daily_return": ["abc"]}, index=pd.date_range("2026-01-01", periods=1, freq="1D", tz="UTC"))
    p_str, digest_str = _store(str_col)
    with pytest.raises(DataIntegrityError, match="dtype invalid"):
        load_strategy_bootstrap(p_str, expected_sha256=digest_str)
    tiny = pd.Series([0.05, -0.01], index=pd.date_range("2026-01-01", periods=2, freq="1D", tz="UTC"), dtype="float64", name="reference_daily_return")
    path, tiny_digest = save_strategy_bootstrap(tmp_path / "tiny.parquet", tiny)
    pd.testing.assert_series_equal(load_strategy_bootstrap(path, expected_sha256=tiny_digest), tiny, check_exact=False, check_freq=False)


def test_assert_deployment_eligible_rejects_flags_drift(tmp_path) -> None:
    import json
    import types
    import pandas as pd
    import pytest
    from src.common.errors import DataIntegrityError
    from src.mhs.live_strategy import assert_deployment_eligible

    tw = pd.DataFrame({"BTCUSDT": [0.1]}, index=pd.DatetimeIndex([pd.Timestamp("2026-08-30", tz="UTC")]))
    report = types.SimpleNamespace(status="COMPLETE", research_go=types.SimpleNamespace(eligible=True), blend=types.SimpleNamespace(target_weights=tw), flags={"a": 1})
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"flags": {"a": 2}}), encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="flags digest drift"):
        assert_deployment_eligible(report, reference_report_path=ref)


def test_strategy_bootstrap_loader_fail_closed_branches(tmp_path) -> None:
    import pandas as pd
    import pytest
    from pydantic import SecretStr
    from src.common.errors import DataIntegrityError
    from src.live.errors import ArtifactSealError
    from src.mhs.live_strategy import load_strategy_bootstrap, save_strategy_bootstrap

    reference = pd.Series([0.01, -0.02, 0.03], index=pd.date_range("2026-01-01", periods=3, freq="1D", tz="UTC"), dtype="float64", name="reference_daily_return")
    key = SecretStr("A" * 43 + "=")
    wrong = SecretStr("B" * 43 + "=")
    enc_path, digest = save_strategy_bootstrap(tmp_path / "strategy_bootstrap.parquet", reference, artifact_key=key)
    with pytest.raises(DataIntegrityError, match="not found"):
        load_strategy_bootstrap(tmp_path / "absent.parquet.enc", expected_sha256=digest, artifact_key=key)
    with pytest.raises(ArtifactSealError):
        load_strategy_bootstrap(enc_path, expected_sha256=digest, artifact_key=wrong)
    enc_dir = tmp_path / "dir.parquet.enc"
    enc_dir.mkdir()
    with pytest.raises(DataIntegrityError, match="corrupt"):
        load_strategy_bootstrap(enc_dir, expected_sha256=digest, artifact_key=key)
    plain_dir = tmp_path / "plain.parquet"
    plain_dir.mkdir()
    with pytest.raises(DataIntegrityError, match="corrupt"):
        load_strategy_bootstrap(plain_dir, expected_sha256=digest)
    (tmp_path / "copied.parquet").write_bytes(enc_path.read_bytes())
    with pytest.raises(ArtifactSealError):
        load_strategy_bootstrap(tmp_path / "copied.parquet", expected_sha256=digest, artifact_key=wrong)
