"""Ops CLI wiring contract (R8): provision-env builds before install, honors --dry-run."""

from __future__ import annotations


def test_ops_provision_env_builds_before_install_and_respects_dry_run(tmp_path, monkeypatch) -> None:
    import src.cli.commands.ops as ops
    from src.cli.main import build_root_parser

    events: list[str] = []

    monkeypatch.setattr(ops, "build_runtime_fragment", lambda path: events.append("build") or "BINANCE_API_KEY=key\n")
    monkeypatch.setattr(ops, "install_runtime_fragment", lambda host, fragment: events.append(f"install:{host}"))

    source = tmp_path / ".quant.env"
    source.write_text("BINANCE_API_KEY=key\n", encoding="utf-8")

    parser = build_root_parser()

    args = parser.parse_args(["ops", "provision-env", "--host", "or-vps", "--source", str(source)])
    args.handler(args)
    assert events == ["build", "install:or-vps"]

    events.clear()
    dry_args = parser.parse_args(
        ["ops", "provision-env", "--host", "or-vps", "--source", str(source), "--dry-run"]
    )
    dry_args.handler(dry_args)
    assert events == ["build"]
