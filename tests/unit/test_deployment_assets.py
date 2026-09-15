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
    assert "env_file: .env" in compose  # 시크릿은 env_file 주입만 허용된다
    assert "./data/state:/app/data/state" in compose  # I-STATE-SURVIVES-REDEPLOY
    assert "./logs:/app/logs" in compose


def test_docker_compose_has_independent_liquidation_collector_service() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    # 청산 스트림은 live 데몬과 독립된 서비스로 24/7 가동된다.
    assert "stream-liquidations" in compose
    assert '"data"' in compose  # command runs the data CLI group
    assert "unless-stopped" in compose
    assert "./data/futures/liquidations:/app/data/futures/liquidations" in compose
    assert "env_file: .env" in compose
    # 기존 live 데몬 서비스 계약이 깨지지 않는다.
    assert "./data/state:/app/data/state" in compose


#: 본 모듈이 검증하는 시나리오 ID(lean_check 추적용).
COVERED_SCENARIOS: tuple[str, ...] = (
    "SCENARIO_LIVE_DAEMON_11_DOCKERFILE_BUILDS",
    "test_docker_compose_has_independent_liquidation_collector_service",
)


def test_deploy_workflow_waits_for_daemon_idle_gate_before_recreate() -> None:
    workflow = (ROOT / ".github" / "workflows" / "deploy.yml").read_text(encoding="utf-8")

    gate = workflow.index("python3 tools/devops/daemon_idle_gate.py")
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

