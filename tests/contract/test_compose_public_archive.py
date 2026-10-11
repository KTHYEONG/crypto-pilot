"""Compose contract: public-archive mirror runs only via the archive profile."""

from __future__ import annotations

from pathlib import Path

COMPOSE = Path("docker-compose.yml")


def _service_block() -> list[str]:
    lines = COMPOSE.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line == "  public-archive:")
    block = []
    for line in lines[start + 1 :]:
        if len(line) > 2 and line.startswith("  ") and line[2] != " ":
            break
        block.append(line)
    return block


def test_archive_service_is_profile_gated() -> None:
    block = "\n".join(_service_block())
    assert 'profiles: ["archive"]' in block
    assert 'restart: "no"' in block
    assert "env_file" not in block
    assert "ARCHIVE_INCREMENTAL=1" in block
    assert '"archive", "collect", "--dataset", "all", "--execute", "--workers", "2"' in block


def test_archive_service_resource_budget() -> None:
    block = "\n".join(_service_block())
    assert "mem_limit: 3g" in block
    assert "ohlcv" not in block
