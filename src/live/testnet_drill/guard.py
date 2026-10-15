"""Isolation gate for testnet drill runs."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

from src.common.paths import APP_ROOT, DATA_DIR
from src.live.errors import LiveTradingError
from src.live.settings import TESTNET_FAPI_URL, ExecutionMode, LiveSettings

DRILL_SCRATCH_ROOT = APP_ROOT / "scratch" / "testnet_drill"

_PRODUCTION_PATH_FIELDS = (
    "ledger_path",
    "order_journal_path",
    "fills_dir",
    "execution_quality_dir",
    "portfolio_state_dir",
    "microstructure_dir",
    "tax_ledger_dir",
    "heartbeat_path",
    "alert_outbox_path",
    "ledger_resync_backup_dir",
)


class DrillRefused(LiveTradingError):  # noqa: N818 - spec pins the name
    """Drill preconditions are not met; no venue or filesystem contact was made."""


def drill_scratch_root() -> Path:
    """Root under which every drill workspace lives (patch point for hermetic tests)."""
    return DRILL_SCRATCH_ROOT


def _host(url: str) -> str:
    return urlsplit(url if "://" in url else f"https://{url}").hostname or ""


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def assert_drill_allowed(settings: LiveSettings, workspace: Path) -> None:
    """Raise DrillRefused unless settings and workspace describe an isolated testnet drill."""
    violations: list[str] = []
    if settings.mode is not ExecutionMode.LIVE_TESTNET:
        violations.append("mode")
    venue_host = _host(TESTNET_FAPI_URL)
    if _host(settings.order_base_url) != venue_host:
        violations.append("order_base_url")
    if _host(settings.market_data_base_url) != venue_host:
        violations.append("market_data_base_url")
    order_key = settings.order_api_key.get_secret_value() if settings.order_api_key is not None else None
    order_secret = settings.order_api_secret.get_secret_value() if settings.order_api_secret is not None else None
    if not order_key or not order_secret:
        violations.append("order_api_credentials")
    elif settings.api_key is not None and order_key == settings.api_key.get_secret_value():
        violations.append("order_api_key_separation")
    resolved_workspace = _resolve(workspace)
    drill_root = _resolve(drill_scratch_root())
    state_root = _resolve(DATA_DIR / "state")
    if resolved_workspace != drill_root and not resolved_workspace.is_relative_to(drill_root):
        violations.append("workspace_root")
    if (
        resolved_workspace == state_root or resolved_workspace.is_relative_to(state_root)
    ) and "workspace_root" not in violations:
        violations.append("workspace_root")
    if settings.record_run_id is not None:
        violations.append("record_run_id")
    for field in _PRODUCTION_PATH_FIELDS:
        raw = getattr(settings, field, None)
        if raw is None:
            continue
        candidate = _resolve(raw)
        if not candidate.is_relative_to(resolved_workspace) or candidate.is_relative_to(state_root):
            violations.append("production_paths")
            break
    if violations:
        raise DrillRefused(f"drill refused: {', '.join(violations)}")
