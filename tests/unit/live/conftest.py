"""tests/unit/live 공통 픽스처. 어떤 단위 테스트도 실제 네트워크를 열지 않는다."""

from __future__ import annotations

import contextlib
import os
import socket

import pytest


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch):
    """socket.connect를 예외 스텁으로 대체해 네트워크 사용을 구조적으로 차단한다."""

    def _raise(*args: object, **kwargs: object) -> None:
        raise AssertionError("unit tests must not open network sockets")

    monkeypatch.setattr(socket.socket, "connect", _raise)


@pytest.fixture(autouse=True)
def _isolate_exchange_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep developer-shell exchange credentials out of hermetic unit tests.

    ``LiveSettings`` intentionally accepts legacy ``BINANCE_*`` aliases and
    ``LIVE_*`` settings for real deployments. A host shell may export those,
    however, and silently change a test's default settings or credentials.
    Scrub all matching keys by prefix so every test starts completely hermetic
    unless it explicitly opts in via monkeypatch.setenv.
    """
    for key in list(os.environ):
        if key.startswith(("LIVE_", "BINANCE_", "UPBIT_")):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def _isolate_signal_step_sidecars(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    import src.cli.commands.live as live_cli

    monkeypatch.setattr(live_cli, "default_weights_path", lambda: tmp_path / "state" / "deployed_target_weights.parquet.enc")


@pytest.fixture(autouse=True)
def _isolate_alert_outbox(monkeypatch: pytest.MonkeyPatch, tmp_path, _isolate_exchange_credentials: None) -> None:
    """Route the durable alert outbox to a per-test file.

    ``LiveSettings`` defaults to ``DATA_DIR/state/alert_outbox.json``; without
    isolation every daemon/runner test would share one real outbox and dedupe
    against each other. The explicit dependency on the credential scrub keeps
    the env var alive regardless of fixture ordering.
    """
    monkeypatch.setenv("LIVE_ALERT_OUTBOX_PATH", str(tmp_path / "alert_outbox.json"))


@pytest.fixture(autouse=True)
def _isolate_live_state_paths(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Route every default ``data/state`` artifact of the live runner to per-test paths.

    The runner, scheduler and CLI fall back to ``DATA_DIR/state/...`` for the order journal, tax
    ledger, execution-quality, microstructure and portfolio-state histories and the daemon
    heartbeat when ``LiveSettings`` leaves them unset. Tests that do not set them explicitly would
    write synthetic rows (AAAUSDT, fixed 2026-08 dates) into the developer's real state directory.
    Audit logs (``logs/live/shadow_cycle``) are routed to a per-test root as well. The default-path helpers are replaced instead of the ``LIVE_*`` settings so tests that assert
    ``LiveSettings`` defaults keep observing the real defaults.
    """
    import src.live.execution_quality as execution_quality
    import src.live.ledger_resync as ledger_resync
    import src.live.microstructure as microstructure
    import src.live.portfolio_state as portfolio_state
    import src.live.runner as runner
    import src.live.scheduler as scheduler
    import src.live.tax_ledger as tax_ledger

    state = tmp_path / "isolated_state"
    replacements = {
        "default_order_journal_path": state / "order_journal.jsonl",
        "default_tax_ledger_dir": state / "tax_ledger",
        "default_execution_quality_dir": state / "execution_quality",
        "default_microstructure_dir": state / "microstructure",
        "default_portfolio_state_dir": state / "portfolio_state",
    }
    # order_journal keeps its real helper: nothing inside that module calls it, and its default is asserted directly.
    for module in (tax_ledger, execution_quality, microstructure, portfolio_state, runner, ledger_resync):
        for name, target in replacements.items():
            if hasattr(module, name):
                monkeypatch.setattr(module, name, lambda target=target: target)
    monkeypatch.setattr(scheduler, "DATA_DIR", tmp_path / "isolated_data")
    import src.live.audit as audit

    monkeypatch.setattr(audit, "AUDIT_LOG_ROOT", tmp_path / "isolated_logs")


@pytest.fixture(autouse=True)
def _isolate_live_process_logs(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """src.cli.commands.live 로그 디렉터리를 테스트 격리 경로로 돌린다."""
    import logging

    import src.cli.commands.live as live_cli

    monkeypatch.setattr(live_cli, "_LIVE_LOG_DIR", tmp_path / "live_logs")
    before = list(logging.getLogger().handlers)
    yield
    for handler in [h for h in logging.getLogger().handlers if h not in before]:
        logging.getLogger().removeHandler(handler)
        with contextlib.suppress(Exception):
            handler.close()


@pytest.fixture(autouse=True)
def _neutralize_execution_depth_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    """실행 윈도우 깊이 캡처는 단위 테스트에서 실제 스레드/소켓을 열지 않는다.

    ``run_shadow_cycle`` 은 기본적으로 WS 레코더를 기동하므로, 스텁하지 않으면
    실제 aiohttp 세션과 30분 post-window 대기로 테스트가 멈춘다. 깊이 캡처 자체를
    검증하는 테스트는 자체 스텁으로 덮어쓴다.
    """
    import src.live.runner as runner_mod
    from src.live.depth_capture import DepthCaptureSummary

    class _NullDepthRecorder:
        def __init__(self, *args: object, **kwargs: object) -> None:
            return None

        def start(self) -> None:
            return None

        def stop(self, *, post_window_s: float, shutdown: object = None) -> DepthCaptureSummary:
            return DepthCaptureSummary(
                rows_received=0, rows_persisted=0, symbols_requested=0, symbols_seen=0,
                symbols_missing=(), reconnects=0, parts=0, flush_failures=0,
            )

    monkeypatch.setattr(runner_mod, "ExecutionDepthRecorder", _NullDepthRecorder)
