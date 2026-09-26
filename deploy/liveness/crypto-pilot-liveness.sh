#!/usr/bin/env bash
# crypto-pilot liveness checker: 호스트 systemd 타이머가 5분마다 실행한다.
# 데몬 컨테이너 상태를 읽고, 데몬 이미지와 같은 이미지의 일회성 컨테이너로
# `live liveness-check`를 실행해 하트비트 정체·단계 초과·컨테이너 중단을 감지한다.
# 설치: systemctl --user enable --now crypto-pilot-liveness.timer
# 왜 호스트 타이머인가: 죽은 데몬 스스로는 자신의 죽음을 알릴 수 없으므로,
# 데몬과 별개인 호스트 프로세스가 감시해야 한다.
set -uo pipefail

ROOT="${CRYPTO_PILOT_ROOT:-$HOME/crypto-pilot}"
LOG_DIR="${HOME}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/crypto-pilot-liveness-$(date -u +%F).log"
ENV_FILE="/home/ubuntu/quant-secrets/crypto-pilot.env"
CONTAINER="mhs-live-daemon"

log() {
  printf '%s [SYS] stage=liveness %s\n' "$(date -u +%FT%TZ)" "$*" >> "${LOG_FILE}"
}

RUNNING=0
RESTARTING=0
OOM=0
RESTART_COUNT=0
STARTED_AT=""
IMAGE=""

if docker inspect "${CONTAINER}" >/dev/null 2>&1; then
  RUNNING="$(docker inspect -f '{{.State.Running}}' "${CONTAINER}" 2>/dev/null || echo false)"
  RESTARTING="$(docker inspect -f '{{.State.Restarting}}' "${CONTAINER}" 2>/dev/null || echo false)"
  OOM="$(docker inspect -f '{{.State.OOMKilled}}' "${CONTAINER}" 2>/dev/null || echo false)"
  RESTART_COUNT="$(docker inspect -f '{{.RestartCount}}' "${CONTAINER}" 2>/dev/null || echo 0)"
  STARTED_AT="$(docker inspect -f '{{.State.StartedAt}}' "${CONTAINER}" 2>/dev/null || echo '')"
  IMAGE="$(docker inspect -f '{{.Config.Image}}' "${CONTAINER}" 2>/dev/null || echo '')"
  [ "${RUNNING}" = "true" ] && RUNNING=1 || RUNNING=0
  [ "${RESTARTING}" = "true" ] && RESTARTING=1 || RESTARTING=0
  [ "${OOM}" = "true" ] && OOM=1 || OOM=0
else
  log "status=CONTAINER_MISSING container=${CONTAINER}"
fi

if [ -z "${IMAGE}" ]; then
  IMAGE="ghcr.io/kthyeong/crypto-pilot-live:latest"
fi

to_flag() {
  if [ "$1" = "1" ]; then printf '1'; else printf '0'; fi
}

ARGS=(live liveness-check
  --container-running "$(to_flag "${RUNNING}")"
  --container-restarting "$(to_flag "${RESTARTING}")"
  --container-oom "$(to_flag "${OOM}")"
  --restart-count "${RESTART_COUNT}")
if [ -n "${STARTED_AT}" ] && [ "${STARTED_AT}" != "null" ]; then
  ARGS+=(--container-started-at "${STARTED_AT}")
fi

if docker run --pull never --rm --network host --memory 256m \
  --env-file "${ENV_FILE}" \
  -v "${ROOT}/data/state:/app/data/state" \
  "${IMAGE}" /app/.venv/bin/python -m src.cli.main "${ARGS[@]}"; then
  log "status=OK running=${RUNNING} restarting=${RESTARTING} oom=${OOM} restarts=${RESTART_COUNT}"
  exit 0
else
  code=$?
  log "status=CHECK_FAILED exit=${code} running=${RUNNING} restarting=${RESTARTING} oom=${OOM} restarts=${RESTART_COUNT}"
  exit "${code}"
fi
