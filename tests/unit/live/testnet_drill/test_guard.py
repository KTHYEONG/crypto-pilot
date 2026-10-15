"""Isolation gate invariants for testnet drill runs."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr

from src.common.paths import DATA_DIR
from src.live.settings import ExecutionMode
from src.live.testnet_drill import guard as guard_module
from src.live.testnet_drill.guard import DrillRefused, assert_drill_allowed
from tests.unit.live.testnet_drill._stubs import MAIN_KEY, ORDER_CRED, ORDER_KEY, make_settings, patch_drill_root


@pytest.fixture
def drill_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return patch_drill_root(monkeypatch, tmp_path)


def test_allows_valid_isolated_drill(drill_root: Path) -> None:
    assert_drill_allowed(make_settings(), drill_root / "20261016a")


def test_allows_relative_workspace(drill_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert_drill_allowed(make_settings(), Path("scratch/testnet_drill/20261016a"))


@pytest.mark.parametrize("mode", [ExecutionMode.SHADOW, ExecutionMode.PAPER, ExecutionMode.LIVE_MAINNET])
def test_refuses_non_testnet_modes(drill_root: Path, mode: ExecutionMode) -> None:
    with pytest.raises(DrillRefused, match="mode"):
        assert_drill_allowed(make_settings(mode=mode), drill_root / "20261016a")


def test_refuses_mainnet_venue_url(drill_root: Path) -> None:
    with pytest.raises(DrillRefused, match="order_base_url"):
        assert_drill_allowed(make_settings(order_base_url="https://fapi.binance.com"), drill_root / "20261016a")


def test_refuses_market_data_url_mismatch(drill_root: Path) -> None:
    with pytest.raises(DrillRefused, match="market_data_base_url"):
        assert_drill_allowed(make_settings(market_data_base_url="https://fapi.binance.com"), drill_root / "20261016a")


def test_refuses_missing_order_credentials(drill_root: Path) -> None:
    with pytest.raises(DrillRefused, match="order_api_credentials"):
        assert_drill_allowed(make_settings(order_api_key=None, order_api_secret=None), drill_root / "20261016a")


def test_refuses_shared_key(drill_root: Path) -> None:
    with pytest.raises(DrillRefused, match="order_api_key_separation"):
        assert_drill_allowed(make_settings(order_api_key=SecretStr(MAIN_KEY)), drill_root / "20261016a")


def test_allows_absent_main_key(drill_root: Path) -> None:
    assert_drill_allowed(make_settings(api_key=None, api_secret=None), drill_root / "20261016a")


def test_refuses_production_workspace(drill_root: Path) -> None:
    workspace = DATA_DIR / "state" / "runs" / "x"
    assert not workspace.exists()
    with pytest.raises(DrillRefused, match="workspace_root"):
        assert_drill_allowed(make_settings(), workspace)
    assert not workspace.exists()


def test_refuses_record_run_id(drill_root: Path) -> None:
    with pytest.raises(DrillRefused, match="record_run_id"):
        assert_drill_allowed(make_settings(record_run_id="prod_run_1"), drill_root / "20261016a")


def test_refuses_production_paths(drill_root: Path) -> None:
    settings = make_settings(ledger_path=str(DATA_DIR / "state" / "live_position_ledger.json"))
    with pytest.raises(DrillRefused, match="production_paths"):
        assert_drill_allowed(settings, drill_root / "20261016a")


def test_refuses_workspace_under_state_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(guard_module, "DRILL_SCRATCH_ROOT", DATA_DIR / "state" / "drill")
    with pytest.raises(DrillRefused, match="workspace_root"):
        assert_drill_allowed(make_settings(), DATA_DIR / "state" / "drill" / "20261016a")


def test_aggregates_violations(drill_root: Path) -> None:
    settings = make_settings(mode=ExecutionMode.SHADOW, order_base_url="https://fapi.binance.com")
    with pytest.raises(DrillRefused) as excinfo:
        assert_drill_allowed(settings, DATA_DIR / "state" / "runs" / "x")
    message = str(excinfo.value)
    assert "mode" in message
    assert "order_base_url" in message
    assert "workspace_root" in message


def test_never_leaks_secrets(drill_root: Path) -> None:
    settings = make_settings(mode=ExecutionMode.SHADOW, order_base_url="https://fapi.binance.com")
    with pytest.raises(DrillRefused) as excinfo:
        assert_drill_allowed(settings, DATA_DIR / "state" / "runs" / "x")
    message = str(excinfo.value)
    assert ORDER_KEY not in message
    assert ORDER_CRED not in message
    assert MAIN_KEY not in message
