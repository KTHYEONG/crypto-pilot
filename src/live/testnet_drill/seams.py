"""Bind production entries (runner cycle, ledger resync) to one drill session for the duration of a block.

The runner constructs its own venue clients and audit paths; inside ``production_bound`` both resolve to the drill
session so every request passes the testnet host guard, the wire log and the per-order notional cap, and every audit
line lands in the drill workspace instead of the production log tree.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from src.live.testnet_drill.session import DrillSession


class GuardedVenueClient:
    """Session client whose order posts go through ``guarded_new_order`` (namespace scope, notional cap)."""

    def __init__(self, session: DrillSession) -> None:
        self._session = session

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session.client, name)

    def new_order(self, params: Mapping[str, Any]) -> Any:
        return self._session.guarded_new_order(params, scope="namespace")


def _workspace_audit_path(session: DrillSession) -> Any:
    def resolve(name: str, *args: Any, **kwargs: Any) -> Path:
        path = session.workspace / "logs" / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    return resolve


@contextmanager
def production_bound(session: DrillSession) -> Iterator[GuardedVenueClient]:
    """Route runner/ledger-resync client construction and audit paths to the drill session."""
    import src.live.ledger_resync as resync_module
    import src.live.runner as runner_module

    client = GuardedVenueClient(session)
    audit_path = _workspace_audit_path(session)
    patches: list[tuple[Any, str, Any]] = [
        (runner_module, "_order_client", lambda settings, decision_time: client),
        (runner_module, "_market_client", lambda settings, decision_time: client),
        (runner_module, "default_audit_log_path", audit_path),
        (resync_module, "default_audit_log_path", audit_path),
    ]
    originals = [(module, name, getattr(module, name)) for module, name, _ in patches]
    try:
        for module, name, value in patches:
            setattr(module, name, value)
        yield client
    finally:
        for module, name, value in originals:
            setattr(module, name, value)
