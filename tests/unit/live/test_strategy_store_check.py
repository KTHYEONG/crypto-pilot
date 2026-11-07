from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.live.preflight import strategy_store_check
from src.strategy.store import EXAMPLE_STORE_ROOT, strategy_values

FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "strategy_store"


@pytest.fixture(autouse=True)
def _fresh_store_cache():
    strategy_values.cache_clear()
    yield
    strategy_values.cache_clear()


def test_strategy_store_check_fails_on_example(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRYPTO_STRATEGY_STORE", str(EXAMPLE_STORE_ROOT))
    check = strategy_store_check()
    assert check.passed is False
    assert "example" in check.detail


def test_strategy_store_check_fails_on_broken_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRYPTO_STRATEGY_STORE", str(tmp_path))
    check = strategy_store_check()
    assert check.passed is False
    assert check.name == "strategy_store"


def test_strategy_store_check_passes_without_revision_and_no_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import shutil

    root = tmp_path / "store"
    shutil.copytree(FIXTURE_ROOT, root)
    monkeypatch.setenv("CRYPTO_STRATEGY_STORE", str(root))
    check = strategy_store_check()
    assert check.passed is True
    assert check.detail == "revision=none"
    assert "mom_168h" not in check.detail


def test_strategy_store_check_detail_carries_revision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil

    root = tmp_path / "store"
    shutil.copytree(FIXTURE_ROOT, root)
    (root / "STORE_REVISION").write_text("rev42\n", encoding="utf-8")
    monkeypatch.setenv("CRYPTO_STRATEGY_STORE", str(root))
    check = strategy_store_check()
    assert check.passed is True
    assert check.detail == "revision=rev42"


def test_evaluate_refuses_example_store(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.cli.commands.evaluate import run_evaluate_strategy_command
    from src.common.errors import DataIntegrityError

    monkeypatch.setenv("CRYPTO_STRATEGY_STORE", str(EXAMPLE_STORE_ROOT))
    with pytest.raises(DataIntegrityError, match="private strategy store"):
        run_evaluate_strategy_command(SimpleNamespace(strategy="flow_mom_top20"))


def test_public_tree_has_no_real_member_block() -> None:
    repo = Path(__file__).resolve().parents[3]
    offenders = [
        str(path.relative_to(repo))
        for path in (repo / "src").rglob("*.py")
        if "FeatureMember(name=" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
    assert not (repo / "src" / "strategy" / "releases").exists()


def test_preflight_logs_critical_when_store_unusable(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    import logging

    from src.live.preflight import _append_strategy_store_check

    monkeypatch.setenv("CRYPTO_STRATEGY_STORE", str(EXAMPLE_STORE_ROOT))
    checks: list = []
    with caplog.at_level(logging.CRITICAL):
        _append_strategy_store_check(checks)
    assert [c.name for c in checks] == ["strategy_store"]
    assert checks[0].passed is False
    assert any("[RISK] preflight" in r.message for r in caplog.records)


def test_daemon_refuses_to_start_on_example_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    import logging

    import src.live.scheduler as sched
    from src.live.errors import LiveTradingError
    from src.live.settings import LiveSettings

    monkeypatch.setenv("CRYPTO_STRATEGY_STORE", str(EXAMPLE_STORE_ROOT))
    calls: list = []
    with (
        caplog.at_level(logging.CRITICAL, logger="LiveScheduler"),
        pytest.raises(LiveTradingError, match="daemon refused to start"),
    ):
        sched.run_daemon(
            LiveSettings(mode="paper"), tmp_path / "w.parquet", tmp_path / "state.json",
            sleep_fn=lambda s: None, max_iterations=1,
            refresh_fn=lambda *a, **k: calls.append("refresh"),
            signal_step_fn=lambda t: calls.append("signal"),
            venue_fn=lambda t: calls.append("venue"),
            prefetch_fn=lambda t: calls.append("prefetch"),
            prune_fn=lambda: calls.append("prune"),
        )
    assert calls == []
    assert any("reason=strategy_store" in r.message for r in caplog.records)
