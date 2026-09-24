#!/usr/bin/env bash
# Usage: compose_recreate.sh <image>
# Run from ~/crypto-pilot on the host after the daemon idle gate has passed.
#
# recorder는 fingerprint가 그대로면 재시작하지 않는다. 데몬(120s grace)과 함께
# 묶어 recreate하면 recorder의 청산 커버리지에 불필요한 공백이 생긴다.
set -euo pipefail

IMAGE="${1:?Usage: compose_recreate.sh <image>}"

if docker compose version >/dev/null 2>&1; then
  C="docker compose"
else
  C="docker-compose"
fi

$C pull

action="recreate"
reason="not_running"

running="$(docker inspect -f '{{.State.Running}}' market-recorder 2>/dev/null || true)"
if [ "$running" = "true" ]; then
  reason="running_fp_unreadable"
  running_fp="$(docker exec market-recorder cat /app/.recorder_fingerprint 2>/dev/null || true)"
  case "$running_fp" in
    sha256:?*)
      reason="image_fp_unreadable"
      image_fp="$(docker run --rm --entrypoint cat "$IMAGE" /app/.recorder_fingerprint 2>/dev/null || true)"
      case "$image_fp" in
        sha256:?*)
          if [ "$running_fp" = "$image_fp" ]; then
            reason="compose_config_changed"
            expected_hash="$($C config --hash market-recorder 2>/dev/null || true)"
            actual_hash="$(docker inspect -f '{{index .Config.Labels "com.docker.compose.config-hash"}}' market-recorder 2>/dev/null || true)"
            if [ -n "$expected_hash" ] && [ "$expected_hash" = "$actual_hash" ]; then
              action="keep"
              reason="unchanged"
            fi
          else
            reason="fingerprint_changed"
          fi
          ;;
      esac
      ;;
  esac
fi

echo "[SYS] stage=deploy_recreate recorder_action=$action reason=$reason"

# recorder 정지(수 초)가 데몬의 120s grace를 기다리지 않도록 recorder를 먼저 처리한다.
if [ "$action" = "recreate" ]; then
  $C up -d --no-deps --force-recreate market-recorder
fi
$C up -d --no-deps --force-recreate --remove-orphans mhs-live
docker image prune -f
