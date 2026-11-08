"""evaluate ml-tilt-accept: the C1 evidence triple gates the ML tilt acceptance block (store fixtures only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from src.cli.commands.evaluate import run_ml_tilt_accept_command
from src.strategy.release import ML_TILT_PREREG_VERDICT_RELPATH, load_release, record_acceptance
from src.strategy.store import EXAMPLE_STORE_ROOT, strategy_values

FIXTURE_STORE = Path(__file__).resolve().parents[3] / "fixtures" / "strategy_store"
MODEL_SHA = "1" * 64
DEFINITION = "d" * 64
ACCOUNT_DIGEST = "f" * 64
EVAL_START = "2025-01-01T00:00:00+00:00"
EVAL_END = "2026-10-01T00:00:00+00:00"


def _write_bootstrap(path: Path, **metadata: str) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    tags = {
        "ml_tilt_digest": ACCOUNT_DIGEST,
        "execution": "maker",
        "strategy_id": "flow_mom_top20",
        "evaluation_start": EVAL_START,
        "evaluation_end": EVAL_END,
        **metadata,
    }
    table = pa.table({"weight": [0.5]}).replace_schema_metadata(tags)
    pq.write_table(table, path)


@pytest.fixture(autouse=True)
def _fresh_store_cache():
    strategy_values.cache_clear()
    yield
    strategy_values.cache_clear()


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    store = tmp_path / "store"
    shutil.copytree(FIXTURE_STORE, store)
    monkeypatch.setenv("CRYPTO_STRATEGY_STORE", str(store))
    strategy_values.cache_clear()
    release = load_release("flow_mom_top20", root=store)
    record_acceptance("flow_mom_top20", spec_digest=release.spec_digest, evaluation_digest="eval-1", root=store)

    verdict = store / ML_TILT_PREREG_VERDICT_RELPATH
    verdict.parent.mkdir(parents=True)
    verdict.write_text(json.dumps({"C1": {"PASS": True}}), encoding="utf-8")

    run = tmp_path / "run-c1"
    run.mkdir()
    account = {
        "strategy_id": "flow_mom_top20",
        "capital": 300.0,
        "execution": {"mode": "maker"},
        "policy": {"kind": "growth"},
        "liquidated_at": None,
        "stress_execution": {"liquidated_at": None},
        "initial_margin_breaches": 0,
        "ml_tilt_digest": ACCOUNT_DIGEST,
        "evaluation_start": EVAL_START,
        "evaluation_end": EVAL_END,
    }
    (run / "account.json").write_text(json.dumps(account), encoding="utf-8")
    bootstrap = tmp_path / "unit.parquet"
    _write_bootstrap(bootstrap)

    import src.strategy.release as release_module

    manifest = {
        "ml_tilt_digest": ACCOUNT_DIGEST,
        "ml_tilt_definition": DEFINITION,
        "ml_tilt_models": [["2026-10-01T00:00:00+00:00", MODEL_SHA]],
    }
    monkeypatch.setattr(release_module, "ml_tilt_manifest", lambda _models: manifest)
    return {"store": store, "run": run, "bootstrap": bootstrap, "verdict": verdict, "account": account}


def _args(evidence, **overrides) -> argparse.Namespace:
    fields = {
        "account_run": str(evidence["run"]),
        "models": str(evidence["run"].parent / "models"),
        "bootstrap": str(evidence["bootstrap"]),
        "execution_policy": "strict_passive_repeg",
        "write": False,
    }
    fields.update(overrides)
    return argparse.Namespace(**fields)


def _write_account(evidence, mutate) -> None:
    account = dict(evidence["account"])
    mutate(account)
    (evidence["run"] / "account.json").write_text(json.dumps(account), encoding="utf-8")


def test_accept_dry_run_prints_block_and_writes_nothing(evidence, capsys) -> None:
    release_file = evidence["store"] / "crypto-pilot" / "releases" / "flow_mom_top20.json"
    before = release_file.read_bytes()
    run_ml_tilt_accept_command(_args(evidence))
    block = json.loads(capsys.readouterr().out)
    assert block["definition_digest"] == DEFINITION
    assert block["target_capital_usdt"] == 300.0
    assert block["models"] == [["2026-10-01T00:00:00+00:00", MODEL_SHA]]
    assert block["unit_bootstrap_sha256"] == hashlib.sha256(evidence["bootstrap"].read_bytes()).hexdigest()
    assert block["account_run"] == "run-c1"
    assert release_file.read_bytes() == before


def test_accept_write_records_block(evidence) -> None:
    run_ml_tilt_accept_command(_args(evidence, write=True))
    release = load_release("flow_mom_top20", root=evidence["store"])
    acceptance = release.ml_tilt_acceptance
    assert acceptance is not None
    assert acceptance.target_capital_usdt == 300.0
    assert acceptance.account_sha256 == hashlib.sha256((evidence["run"] / "account.json").read_bytes()).hexdigest()
    assert acceptance.prereg_verdict_sha256 == hashlib.sha256(evidence["verdict"].read_bytes()).hexdigest()
    assert acceptance.account_ml_tilt_digest == ACCOUNT_DIGEST


@pytest.mark.parametrize(
    ("mutate", "bootstrap_name", "verdict_pass", "reason"),
    [
        (lambda a: a.update(ml_tilt_digest="0" * 64), "unit.parquet", True, "ml_tilt_digest"),
        (lambda a: a.update(execution={"mode": "taker"}), "unit.parquet", True, "execution mode"),
        (lambda a: a.update(policy={"kind": "fixed"}), "unit.parquet", True, "policy kind"),
        (lambda a: a.update(liquidated_at="2026-10-05T00:00:00+00:00"), "unit.parquet", True, "not liquidated"),
        (lambda a: a.update(initial_margin_breaches=2), "unit.parquet", True, "initial margin"),
        (lambda a: None, "unit.parquet", False, "C1 verdict"),
        (lambda a: None, "unit.parquet.enc", True, "sealed bootstrap"),
    ],
)
def test_accept_refuses_bad_evidence(evidence, mutate, bootstrap_name, verdict_pass, reason) -> None:
    _write_account(evidence, mutate)
    evidence["verdict"].write_text(json.dumps({"C1": {"PASS": verdict_pass}}), encoding="utf-8")
    bootstrap = evidence["bootstrap"].with_name(bootstrap_name)
    if bootstrap_name.endswith(".enc"):
        bootstrap.write_bytes(b"sealed")
    release_file = evidence["store"] / "crypto-pilot" / "releases" / "flow_mom_top20.json"
    before = release_file.read_bytes()
    with pytest.raises(SystemExit, match=reason):
        run_ml_tilt_accept_command(_args(evidence, bootstrap=str(bootstrap), write=True))
    assert release_file.read_bytes() == before
    assert load_release("flow_mom_top20", root=evidence["store"]).ml_tilt_acceptance is None


def test_accept_refuses_example_store(evidence, monkeypatch) -> None:
    monkeypatch.setenv("CRYPTO_STRATEGY_STORE", str(EXAMPLE_STORE_ROOT))
    strategy_values.cache_clear()
    with pytest.raises(SystemExit, match="ml-tilt-accept refused"):
        run_ml_tilt_accept_command(_args(evidence))


def test_accept_second_write_is_refused(evidence) -> None:
    run_ml_tilt_accept_command(_args(evidence, write=True))
    with pytest.raises(SystemExit, match="already recorded"):
        run_ml_tilt_accept_command(_args(evidence, write=True))


def test_accept_refuses_blank_execution_policy(evidence) -> None:
    with pytest.raises(SystemExit, match="execution policy"):
        run_ml_tilt_accept_command(_args(evidence, execution_policy="  "))


def test_accept_refuses_unaccepted_base_release(evidence) -> None:
    release_file = evidence["store"] / "crypto-pilot" / "releases" / "flow_mom_top20.json"
    raw = json.loads(release_file.read_text(encoding="utf-8"))
    raw["verdict"] = None
    release_file.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(SystemExit, match="base release is not accepted"):
        run_ml_tilt_accept_command(_args(evidence))


def test_accept_refuses_stress_liquidation(evidence) -> None:
    _write_account(evidence, lambda a: a.update(stress_execution={"liquidated_at": "2026-10-05T00:00:00+00:00"}))
    with pytest.raises(SystemExit, match="stress not liquidated"):
        run_ml_tilt_accept_command(_args(evidence))


def test_accept_refuses_policy_other_than_accepted(evidence) -> None:
    with pytest.raises(SystemExit, match="execution policy"):
        run_ml_tilt_accept_command(_args(evidence, execution_policy="taker_parity"))


@pytest.mark.parametrize(
    "metadata",
    [
        {"ml_tilt_digest": "0" * 64},
        {"execution": "taker"},
        {"strategy_id": "flow_mom_top40_control"},
        {"evaluation_start": "2024-01-01T00:00:00+00:00"},
        {"evaluation_end": "2026-11-01T00:00:00+00:00"},
    ],
)
def test_accept_refuses_bootstrap_from_other_digest_or_window(evidence, metadata) -> None:
    _write_bootstrap(evidence["bootstrap"], **metadata)
    with pytest.raises(SystemExit, match="bootstrap provenance"):
        run_ml_tilt_accept_command(_args(evidence))


def test_accept_refuses_bootstrap_without_parquet_metadata(evidence) -> None:
    evidence["bootstrap"].write_bytes(b"not a parquet file")
    with pytest.raises(SystemExit, match="bootstrap provenance"):
        run_ml_tilt_accept_command(_args(evidence))
