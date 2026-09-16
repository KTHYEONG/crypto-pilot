"""Workstation-to-VPS runtime secret provisioning (INV-SECRET-SSOT).

The local workstation file (default ``~/.quant.env``) is the single source
of truth. Only keys declared in :data:`RUNTIME_ENV_SPEC` are extracted, in
allow-list fashion, into a ``TARGET=VALUE`` fragment that is installed on
the VPS over SSH stdin with 0600 atomic replacement.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from src.common.errors import ProvisioningError


@dataclass(frozen=True)
class RuntimeEnvKey:
    target: str
    sources: tuple[str, ...]


RUNTIME_ENV_SPEC: tuple[RuntimeEnvKey, ...] = (
    RuntimeEnvKey(target="BINANCE_API_KEY", sources=("BINANCE_API_KEY",)),
    RuntimeEnvKey(target="BINANCE_SECRET_KEY", sources=("BINANCE_SECRET_KEY", "BINANCE_SECRET")),
    RuntimeEnvKey(target="LIVE_ARTIFACT_KEY", sources=("LIVE_ARTIFACT_KEY",)),
    RuntimeEnvKey(target="LIVE_ALERT_GMAIL_USER", sources=("LIVE_ALERT_GMAIL_USER", "ALERT_GMAIL_USER")),
    RuntimeEnvKey(
        target="LIVE_ALERT_GMAIL_APP_PASSWORD",
        sources=("LIVE_ALERT_GMAIL_APP_PASSWORD", "ALERT_GMAIL_APP_PASSWORD"),
    ),
    RuntimeEnvKey(target="LIVE_ORDER_API_KEY", sources=("LIVE_ORDER_API_KEY",)),
    RuntimeEnvKey(target="LIVE_ORDER_API_SECRET", sources=("LIVE_ORDER_API_SECRET",)),
)

REMOTE_RUNTIME_ENV_PATH: str = "/home/ubuntu/quant-secrets/crypto-pilot.env"

REMOTE_RUNTIME_INSTALL_SCRIPT: str = (
    "set -euo pipefail\n"
    f'ENV_PATH="{REMOTE_RUNTIME_ENV_PATH}"\n'
    'DIR="$(dirname "$ENV_PATH")"\n'
    'mkdir -p "$DIR"\n'
    'chmod 0700 "$DIR"\n'
    'TMP="$(mktemp "$DIR/.env.XXXXXX")"\n'
    'cat > "$TMP"\n'
    'chmod 600 "$TMP"\n'
    'chown ubuntu:ubuntu "$TMP"\n'
    'mv -f "$TMP" "$ENV_PATH"\n'
    'chmod 600 "$ENV_PATH"\n'
    'chown ubuntu:ubuntu "$ENV_PATH"\n'
)


def parse_workstation_assignments(source_path: Path, accepted_keys: frozenset[str]) -> dict[str, str]:
    """Parse ``KEY=VALUE`` assignments, collecting only ``accepted_keys``."""
    assignments: dict[str, str] = {}
    text = source_path.read_text(encoding="utf-8")
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        body = stripped
        if body.startswith("export "):
            body = body[len("export ") :].strip()
        key, _, value = body.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if not value:
            continue
        if key not in accepted_keys:
            continue
        if key in assignments:
            raise ProvisioningError(f"duplicate workstation assignment for key: {key}")
        assignments[key] = value
    return assignments


def build_runtime_fragment(source_path: Path) -> str:
    """Build the canonical ``TARGET=VALUE`` fragment for the VPS runtime."""
    accepted: frozenset[str] = frozenset(source for key in RUNTIME_ENV_SPEC for source in key.sources)
    parsed = parse_workstation_assignments(source_path, accepted)
    lines: list[str] = []
    for key in RUNTIME_ENV_SPEC:
        value: str | None = None
        for source in key.sources:
            candidate = parsed.get(source)
            if candidate:
                value = candidate
                break
        if not value:
            raise ProvisioningError(f"missing runtime secret for key: {key.target}")
        lines.append(f"{key.target}={value}")
    return "\n".join(lines) + "\n"


def install_runtime_fragment(host: str, fragment: str) -> None:
    """Install ``fragment`` at :data:`REMOTE_RUNTIME_ENV_PATH` via SSH stdin.

    ssh concatenates argv[2:] with spaces before handing it to the remote
    login shell, so passing ("bash", "-c", SCRIPT) as separate elements lets
    the newline-bearing SCRIPT get re-split: -c only receives the first
    word ("set"), and the remaining lines execute in the outer login shell.
    That stray ``bash -c set`` dumps the whole shell environment to stdout
    (measured). ``shlex.quote`` collapses the script into one argv element
    so the remote shell parses it as exactly ``bash -c <SCRIPT>``.
    """
    subprocess.run(  # noqa: S603 - contract R7 pins ssh argv; fragment via stdin only
        ["ssh", host, f"bash -c {shlex.quote(REMOTE_RUNTIME_INSTALL_SCRIPT)}"],  # noqa: S607
        input=fragment,
        text=True,
        check=True,
    )
