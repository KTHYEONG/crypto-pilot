"""Venue ledger directory resolution: account-scoped in live modes, run-scoped otherwise."""

from __future__ import annotations

from pathlib import Path

from src.live.settings import LiveSettings
from src.live.tax_ledger import (
    default_venue_tax_ledger_root,
    resolve_tax_ledger_dir,
)


def _live_settings(mode: str, run_id: str | None) -> LiveSettings:
    kwargs: dict = {"mode": mode}
    if run_id is not None:
        kwargs["record_run_id"] = run_id
    if mode in ("live_testnet", "live_mainnet"):
        kwargs.update(order_api_key="k", order_api_secret="s")  # noqa: S106 - hermetic test credential
    if mode == "live_mainnet":
        from src.live.settings import MAINNET_TRADING_ACK

        kwargs["mainnet_trading_ack"] = MAINNET_TRADING_ACK
    return LiveSettings(**kwargs)


def test_venue_ledger_account_scoped_across_run_ids(tmp_path: Path, monkeypatch) -> None:
    import src.live.tax_ledger as tax_mod

    monkeypatch.setattr(tax_mod, "default_venue_tax_ledger_root", lambda: tmp_path / "v")
    first = resolve_tax_ledger_dir(_live_settings("live_testnet", "run_a_12345678"))
    second = resolve_tax_ledger_dir(_live_settings("live_testnet", "run_b_12345678"))
    assert first == tmp_path / "v" / "testnet"
    assert second == first


def test_testnet_and_mainnet_never_share_a_ledger(tmp_path: Path, monkeypatch) -> None:
    import src.live.tax_ledger as tax_mod

    monkeypatch.setattr(tax_mod, "default_venue_tax_ledger_root", lambda: tmp_path / "v")
    testnet = resolve_tax_ledger_dir(_live_settings("live_testnet", "run_a_12345678"))
    mainnet = resolve_tax_ledger_dir(_live_settings("live_mainnet", "run_a_12345678"))
    assert testnet == tmp_path / "v" / "testnet"
    assert mainnet == tmp_path / "v" / "mainnet"
    assert testnet != mainnet


def test_explicit_directory_wins(tmp_path: Path) -> None:
    for mode in ("shadow", "paper", "live_testnet", "live_mainnet"):
        settings = _live_settings(mode, "run_a_12345678")
        settings.tax_ledger_dir = str(tmp_path / "x")
        assert resolve_tax_ledger_dir(settings) == tmp_path / "x"


def test_suppressed_modes_stay_run_scoped(tmp_path: Path, monkeypatch) -> None:
    import src.common.paths as paths_mod
    import src.live.settings as settings_mod
    import src.live.tax_ledger as tax_mod

    monkeypatch.setattr(paths_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(settings_mod, "DATA_DIR", tmp_path)
    settings = _live_settings("paper", "run_a_12345678")
    expected = tmp_path / "state" / "runs" / "run_a_12345678" / "tax_ledger"
    assert resolve_tax_ledger_dir(settings) == expected
    naked = _live_settings("paper", None)
    assert resolve_tax_ledger_dir(naked) == naked.resolved_tax_ledger_dir(tax_mod.default_tax_ledger_dir)


def test_resolver_has_no_side_effects(tmp_path: Path, monkeypatch) -> None:
    import src.live.tax_ledger as tax_mod

    root = tmp_path / "no_such_root"
    monkeypatch.setattr(tax_mod, "default_venue_tax_ledger_root", lambda: root)
    resolved = resolve_tax_ledger_dir(_live_settings("live_testnet", "run_a_12345678"))
    assert resolved == root / "testnet"
    assert not root.exists()


def test_venue_root_default_location() -> None:
    from src.common.paths import DATA_DIR

    assert default_venue_tax_ledger_root() == DATA_DIR / "state" / "tax_ledger"
