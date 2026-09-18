"""SCENARIO_LIVE_DAEMON_11: 배포 자산(Dockerfile/compose/.dockerignore) 정적 계약.

실제 docker build는 CI와 로컬 검증 명령(`docker build -t crypto-pilot-live:local .`)에서
수행하며, 여기서는 시크릿 미포함과 상태 보존 마운트를 소스 레벨로 검증한다.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_SCENARIO_LIVE_DAEMON_11_DOCKERFILE_BUILDS() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert 'CMD ["uv", "run", "python", "-m", "src.cli.main", "live", "daemon"]' in dockerfile
    # 어떤 레이어도 .env나 키 파일을 굽지 않는다(I-NO-SECRET-IN-IMAGE).
    assert ".env" not in dockerfile
    assert "pem" not in dockerfile

    # .dockerignore의 제외 패턴이 시크릿을 계속 걸러낸다.
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert ".env" in dockerignore
    assert "*.pem" in dockerignore

    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "env_file: /home/ubuntu/quant-secrets/crypto-pilot.env" in compose  # 시크릿은 env_file 주입만 허용된다
    assert "./data/state:/app/data/state" in compose  # I-STATE-SURVIVES-REDEPLOY
    assert "./logs:/app/logs" in compose


def test_docker_compose_has_independent_liquidation_collector_service() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    # 청산 스트림은 live 데몬과 독립된 서비스로 24/7 가동된다.
    assert "stream-liquidations" in compose
    assert '"data"' in compose  # command runs the data CLI group
    assert "unless-stopped" in compose
    assert "./data/futures/liquidations:/app/data/futures/liquidations" in compose
    assert "env_file: /home/ubuntu/quant-secrets/crypto-pilot.env" in compose
    # 기존 live 데몬 서비스 계약이 깨지지 않는다.
    assert "./data/state:/app/data/state" in compose


#: 본 모듈이 검증하는 시나리오 ID(lean_check 추적용).
COVERED_SCENARIOS: tuple[str, ...] = (
    "SCENARIO_LIVE_DAEMON_11_DOCKERFILE_BUILDS",
    "test_docker_compose_has_independent_liquidation_collector_service",
    "test_dockerfile_keeps_uv_cache_out_of_image",
    "test_dockerignore_excludes_workspace_caches",
    "test_compose_uses_absolute_secret_path_and_declares_live_mode",
    "test_deploy_workflow_builds_native_arm64_and_tags_commit_sha",
)


def test_deploy_workflow_waits_for_daemon_idle_gate_before_recreate() -> None:
    workflow = (ROOT / ".github" / "workflows" / "deploy.yml").read_text(encoding="utf-8")

    gate = workflow.index("python3 -m src.application.ops.daemon_idle_gate")
    recreate = workflow.index("up -d --force-recreate")
    assert gate < recreate
    assert "cat ~/crypto-pilot/data/state/live_daemon_heartbeat.json" in workflow
    assert '--waited-s "$waited"' in workflow
    assert '"$rc" -ne 10' in workflow
    assert "sleep 60" in workflow


def test_docker_compose_memory_budget_fits_oci_a1_host() -> None:
    from pathlib import Path

    # Given
    root = Path(__file__).resolve().parents[2]
    compose = (root / "docker-compose.yml").read_text(encoding="utf-8")

    # When
    live_block, liquidation_block = compose.split("  liquidation-collector:\n", 1)

    # Then
    assert "container_name: mhs-live-daemon" in live_block
    assert "mem_limit: 3g" in live_block
    assert "mem_limit: 1200m" not in compose
    assert "container_name: liquidation-collector" in liquidation_block
    assert "mem_limit: 768m" in liquidation_block
    assert "HARDWARE_MAX_WORKERS" not in compose


def test_dockerfile_keeps_uv_cache_out_of_image() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")

    assert dockerfile.splitlines()[0].startswith("# syntax=docker/dockerfile:1")
    assert "UV_CACHE_DIR=/root/.cache/uv" in dockerfile
    assert dockerfile.count("--mount=type=cache,target=/root/.cache/uv,sharing=locked") == 2
    assert "UV_NO_SYNC=1" in dockerfile
    assert 'CMD ["uv", "run", "python", "-m", "src.cli.main", "live", "daemon"]' in dockerfile
    assert "scratch/uv-cache" not in dockerfile


def test_dockerignore_excludes_workspace_caches() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    dockerignore = (root / ".dockerignore").read_text(encoding="utf-8")

    for entry in ("scratch/", "tmp/", ".mypy_cache", ".ruff_cache", ".serena"):
        assert entry in dockerignore, entry
    assert ".env" in dockerignore
    assert "*.pem" in dockerignore


def test_compose_uses_absolute_secret_path_and_declares_live_mode() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    compose = (root / "docker-compose.yml").read_text(encoding="utf-8")

    assert compose.count("env_file: /home/ubuntu/quant-secrets/crypto-pilot.env") == 2
    assert "env_file: .env" not in compose
    assert "LIVE_MODE=paper" in compose
    assert "live_mainnet" not in compose
    assert "./data/state:/app/data/state" in compose
    assert "./logs:/app/logs" in compose


def test_deploy_workflow_builds_native_arm64_and_tags_commit_sha() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github" / "workflows" / "deploy.yml").read_text(encoding="utf-8")

    assert "runs-on: ubuntu-24.04-arm" in workflow
    assert "platforms: linux/arm64" in workflow
    assert "linux/amd64" not in workflow
    assert "setup-qemu-action" not in workflow
    assert "sha-${{ github.sha }}" in workflow
    assert "${{ env.IMAGE }}:latest" in workflow
    assert "python3 -m src.application.ops.daemon_idle_gate" in workflow


def test_gate_waits_on_fresh_busy_heartbeat() -> None:
    from datetime import datetime

    from src.application.ops.daemon_idle_gate import BUSY_STAGES, decide_deploy

    assert frozenset({"refresh", "signal", "execute"}) == BUSY_STAGES
    now = datetime.fromisoformat("2026-09-15T01:30:00+00:00")
    heartbeat = {"stage": "signal", "status": "RUNNING", "ts": "2026-09-15T01:29:00+00:00"}
    decision = decide_deploy(heartbeat, now=now, waited_s=0.0, max_wait_s=3600.0, stale_after_s=2700.0)
    assert decision.action == "wait"
    assert decision.reason == "busy:signal"


def test_gate_proceeds_on_idle_heartbeat() -> None:
    from datetime import datetime

    from src.application.ops.daemon_idle_gate import decide_deploy, main

    now = datetime.fromisoformat("2026-09-15T01:30:00+00:00")
    decision = decide_deploy(
        {"stage": "idle", "ts": "2026-09-15T01:29:00+00:00"},
        now=now, waited_s=0.0, max_wait_s=3600.0, stale_after_s=2700.0,
    )
    assert (decision.action, decision.reason) == ("proceed", "idle")
    import json
    import tempfile
    from pathlib import Path as _Path

    with tempfile.TemporaryDirectory() as tmp:
        hb = _Path(tmp) / "hb.json"
        hb.write_text(json.dumps({"stage": "idle", "ts": "2026-09-15T01:29:00+00:00"}), encoding="utf-8")
        assert main(["--heartbeat-file", str(hb), "--waited-s", "0", "--now", "2026-09-15T01:30:00+00:00"]) == 0


def test_gate_preserves_strict_stale_boundary() -> None:
    from datetime import datetime

    from src.application.ops.daemon_idle_gate import decide_deploy

    now = datetime.fromisoformat("2026-09-15T01:30:00+00:00")
    at_threshold = {"stage": "execute", "ts": "2026-09-15T00:45:00+00:00"}
    over_threshold = {"stage": "execute", "ts": "2026-09-15T00:44:59+00:00"}
    at_decision = decide_deploy(at_threshold, now=now, waited_s=0.0, max_wait_s=3600.0, stale_after_s=2700.0)
    assert at_decision.action == "wait"
    over_decision = decide_deploy(over_threshold, now=now, waited_s=0.0, max_wait_s=3600.0, stale_after_s=2700.0)
    assert over_decision.action == "proceed_stale"
    assert over_decision.reason.startswith("stale:execute age_s=")


def test_gate_proceeds_on_wait_timeout() -> None:
    from datetime import datetime

    from src.application.ops.daemon_idle_gate import decide_deploy, main

    now = datetime.fromisoformat("2026-09-15T01:30:00+00:00")
    heartbeat = {"stage": "signal", "status": "RUNNING", "ts": "2026-09-15T01:29:30+00:00"}
    decision = decide_deploy(heartbeat, now=now, waited_s=3600.0, max_wait_s=3600.0, stale_after_s=2700.0)
    assert decision.action == "proceed_timeout"
    assert decision.reason == "max_wait:signal waited_s=3600"
    import json
    import tempfile
    from pathlib import Path as _Path

    with tempfile.TemporaryDirectory() as tmp:
        hb = _Path(tmp) / "hb.json"
        hb.write_text(json.dumps(heartbeat), encoding="utf-8")
        assert main(["--heartbeat-file", str(hb), "--waited-s", "3600", "--now", "2026-09-15T01:30:00+00:00"]) == 0


def test_gate_keeps_malformed_heartbeat_behavior(tmp_path, capsys) -> None:
    import json

    from src.application.ops.daemon_idle_gate import decide_deploy, main

    from datetime import datetime

    now = datetime.fromisoformat("2026-09-15T01:30:00+00:00")
    for heartbeat in (None, {}, {"stage": "signal"}, {"stage": "signal", "ts": "not-a-time"}):
        decision = decide_deploy(heartbeat, now=now, waited_s=0.0, max_wait_s=3600.0, stale_after_s=2700.0)
        assert (decision.action, decision.reason) == ("proceed", "no_heartbeat")
    import pytest

    with pytest.raises(ValueError, match="tz-aware"):
        decide_deploy({"stage": "idle", "ts": "2026-09-15T01:29:00+00:00"}, now=datetime(2026, 9, 15, 1, 30), waited_s=0.0, max_wait_s=3600.0, stale_after_s=2700.0)
    busy = tmp_path / "busy.json"
    busy.write_text(json.dumps({"stage": "signal", "ts": "2026-09-15T01:29:00+00:00"}), encoding="utf-8")
    assert main(["--heartbeat-file", str(busy), "--waited-s", "0", "--now", "2026-09-15T01:30:00+00:00"]) == 10
    assert capsys.readouterr().out.strip() == "action=wait reason=busy:signal"
    for raw in ("", "<html>", "[1, 2]"):
        target = tmp_path / "case.json"
        target.write_text(raw, encoding="utf-8")
        assert main(["--heartbeat-file", str(target), "--waited-s", "0", "--now", "2026-09-15T01:30:00+00:00"]) == 0
        assert capsys.readouterr().out.strip() == "action=proceed reason=no_heartbeat"


def test_gate_runs_under_bare_python_and_matches_workflow(tmp_path) -> None:
    import json
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github" / "workflows" / "deploy.yml").read_text(encoding="utf-8")
    assert "python3 -m src.application.ops.daemon_idle_gate" in workflow
    hb = tmp_path / "hb.json"
    hb.write_text(json.dumps({"stage": "refresh", "ts": "2026-09-15T01:29:00+00:00"}), encoding="utf-8")
    result = subprocess.run(  # noqa: S603 - fixed argv: this interpreter + the repo gate module
        [sys.executable, "-I", "-m", "src.application.ops.daemon_idle_gate", "--heartbeat-file", str(hb), "--waited-s", "0", "--now", "2026-09-15T01:30:00+00:00"],
        capture_output=True, text=True, check=False, cwd=root,
    )
    assert result.returncode == 10
    assert result.stdout.strip() == "action=wait reason=busy:refresh"


def test_ops_cli_delegates_daemon_idle_gate(tmp_path, capsys) -> None:
    import json

    import pytest

    from src.cli.main import build_root_parser

    hb = tmp_path / "hb.json"
    hb.write_text(json.dumps({"stage": "signal", "ts": "2026-09-15T01:29:00+00:00"}), encoding="utf-8")
    parser = build_root_parser()
    args = parser.parse_args(["ops", "daemon-idle-gate", "--heartbeat-file", str(hb), "--waited-s", "0", "--now", "2026-09-15T01:30:00+00:00"])
    with pytest.raises(SystemExit) as exc:
        args.handler(args)
    assert exc.value.code == 10
    assert capsys.readouterr().out.strip() == "action=wait reason=busy:signal"
    idle = tmp_path / "idle.json"
    idle.write_text(json.dumps({"stage": "idle", "ts": "2026-09-15T01:29:00+00:00"}), encoding="utf-8")
    idle_args = parser.parse_args(["ops", "daemon-idle-gate", "--heartbeat-file", str(idle), "--waited-s", "0", "--now", "2026-09-15T01:30:00+00:00"])
    assert idle_args.handler(idle_args) is None

