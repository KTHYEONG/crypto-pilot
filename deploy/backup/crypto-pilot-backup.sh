#!/usr/bin/env bash
# crypto-pilot 전용 Drive 백업 단일 작성자.
# 왜 단일 스크립트인가: 기존에는 호스트 전용 스크립트 2개가 같은 Drive 경로에 잠금 없이 겹쳐 썼다.
# 왜 copy-only인가: rclone copy는 삭제 전파가 없어 로컬 정리가 Drive 원본을 지우지 않는다.
# 왜 _versions/<날짜>에 30일 보관인가: 덮어쓴 원본을 날짜 폴더로 피신시켜 오복구를 가능하게 한다.
# 왜 폴더명으로 prune하는가: rclone이 원본 mtime을 유지하므로 --min-age는 오늘 옮긴 옛 파일을 즉시 지운다.
# 왜 공용 락을 잡는가: 모든 Drive 작성자가 같은 flock에서 직렬화되어야 경합 복사가 사라진다.
set -uo pipefail

CRYPTO_PILOT_ROOT="${CRYPTO_PILOT_ROOT:-$HOME/crypto-pilot}"
RCLONE_BIN="${RCLONE_BIN:-$(command -v rclone 2>/dev/null || echo "$HOME/.local/bin/rclone")}"
REMOTE_ROOT="${REMOTE_ROOT:-gdrive:quant-lake/live/crypto-pilot}"
QUANT_GDRIVE_LOCK="${QUANT_GDRIVE_LOCK:-/run/user/$(id -u)/quant-gdrive.lock}"
LOCK_WAIT_SEC="${LOCK_WAIT_SEC:-7200}"
VERSION_RETENTION_DAYS="${VERSION_RETENTION_DAYS:-30}"
ARCHIVE_VERSION_RETENTION_DAYS="${ARCHIVE_VERSION_RETENTION_DAYS:-7}"
BACKUP_TODAY_UTC="${BACKUP_TODAY_UTC:-$(date -u +%F)}"
LOG_DIR="${LOG_DIR:-$HOME/logs}"
LOG_FILE="${LOG_DIR}/crypto-pilot-backup-${BACKUP_TODAY_UTC}.log"

mkdir -p "$LOG_DIR"
mkdir -p "$(dirname "$QUANT_GDRIVE_LOCK")"
touch "$QUANT_GDRIVE_LOCK"

log_line() {
  printf '%s\n' "$1" >> "$LOG_FILE"
}

exec {LOCK_FD}>"$QUANT_GDRIVE_LOCK"
if ! flock -w "$LOCK_WAIT_SEC" "$LOCK_FD"; then
  log_line "[SYS] stage=gdrive_backup step=lock status=failed"
  exit 75
fi

BACKUP_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

FAILED=0

# Step: data (copy-only, filtered, versioned).
DATA_RC=0
"$RCLONE_BIN" copy "$CRYPTO_PILOT_ROOT/data" "$REMOTE_ROOT/data" \
  --filter-from "$CRYPTO_PILOT_ROOT/deploy/crypto-pilot.rclone-filter" \
  --exclude ".env*" \
  --exclude "*.key" \
  --exclude "*_key.txt" \
  --backup-dir "$REMOTE_ROOT/_versions/$BACKUP_TODAY_UTC/data" \
  --fast-list --transfers 4 -v || DATA_RC=$?
if [ "$DATA_RC" -eq 0 ]; then
  log_line "[SYS] stage=gdrive_backup step=data status=ok rc=0"
else
  log_line "[SYS] stage=gdrive_backup step=data status=failed rc=$DATA_RC"
  FAILED=1
fi

# Step: orders (live-only order events).
ORDERS_SRC="$CRYPTO_PILOT_ROOT/logs/live/orders"
if [ -d "$ORDERS_SRC" ]; then
  ORDERS_RC=0
  "$RCLONE_BIN" copy "$ORDERS_SRC" "$REMOTE_ROOT/logs/live/orders" \
    --include "*.jsonl" \
    --backup-dir "$REMOTE_ROOT/_versions/$BACKUP_TODAY_UTC/logs/live/orders" \
    --fast-list -v || ORDERS_RC=$?
  if [ "$ORDERS_RC" -eq 0 ]; then
    log_line "[SYS] stage=gdrive_backup step=orders status=ok rc=0"
  else
    log_line "[SYS] stage=gdrive_backup step=orders status=failed rc=$ORDERS_RC"
    FAILED=1
  fi
else
  log_line "[SYS] stage=gdrive_backup step=orders status=skipped rc=0 reason=absent"
fi

# Step: public_archive (copy-only, dedicated short-retention versions, own marker).
# rclone exit code 10 is DurationExceeded (--max-duration reached): partial, not failure.
ARCHIVE_FAILED=0
ARCHIVE_SRC="$CRYPTO_PILOT_ROOT/data/public_archive"
if [ ! -d "$ARCHIVE_SRC" ]; then
  log_line "[SYS] stage=gdrive_backup step=public_archive status=skipped rc=0 reason=absent"
else
  ARCHIVE_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  ARCHIVE_RC=0
  "$RCLONE_BIN" copy "$ARCHIVE_SRC" "$REMOTE_ROOT/data/public_archive" \
    --exclude ".archive-*/**" \
    --exclude "*.tmp" \
    --exclude "*.tmp-*" \
    --exclude "*.partial" \
    --backup-dir "$REMOTE_ROOT/_versions_archive/$BACKUP_TODAY_UTC/public_archive" \
    --max-duration "${ARCHIVE_MAX_DURATION:-2h}" \
    --fast-list --transfers 4 -v || ARCHIVE_RC=$?
  if [ "$ARCHIVE_RC" -eq 0 ]; then
    log_line "[SYS] stage=gdrive_backup step=public_archive status=ok rc=0"
    ARCHIVE_STATUS_DIR="$CRYPTO_PILOT_ROOT/deploy/backup/status"
    ARCHIVE_FINISHED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    ARCHIVE_PARTIAL="$ARCHIVE_STATUS_DIR/last_archive_success.json.partial"
    if ! { mkdir -p "$ARCHIVE_STATUS_DIR" &&
      printf '{"started_at": "%s", "finished_at": "%s", "rc": 0}' "$ARCHIVE_STARTED_AT" "$ARCHIVE_FINISHED_AT" > "$ARCHIVE_PARTIAL" &&
      sync &&
      mv -f "$ARCHIVE_PARTIAL" "$ARCHIVE_STATUS_DIR/last_archive_success.json"; }; then
      log_line "[SYS] stage=gdrive_backup step=archive_status status=failed"
      ARCHIVE_FAILED=1
    fi
  elif [ "$ARCHIVE_RC" -eq 10 ]; then
    log_line "[SYS] stage=gdrive_backup step=public_archive status=partial rc=10"
  else
    log_line "[SYS] stage=gdrive_backup step=public_archive status=failed rc=$ARCHIVE_RC"
    ARCHIVE_FAILED=1
  fi
fi

prune_versions() {
  local root="$1" retention="$2" step="$3"
  local listing="" rc=0 cutoff entry name normalized purge_rc
  listing="$("$RCLONE_BIN" lsf --dirs-only "$root" 2>/dev/null)" || rc=$?
  if [ "$rc" -eq 3 ]; then
    log_line "[SYS] stage=gdrive_backup step=$step status=ok rc=0"
    return 0
  elif [ "$rc" -ne 0 ]; then
    log_line "[SYS] stage=gdrive_backup step=$step status=failed rc=$rc"
    return 1
  fi
  if ! [[ "$retention" =~ ^[0-9]+$ ]] ||
    ! cutoff="$(date -u -d "$BACKUP_TODAY_UTC - $retention days" +%F)"; then
    log_line "[SYS] stage=gdrive_backup step=$step status=failed rc=1 reason=invalid_retention"
    return 1
  fi
  while IFS= read -r entry; do
    [[ "$entry" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}/$ ]] || continue
    name="${entry%/}"
    normalized="$(date -u -d "$name" +%F 2>/dev/null)" || continue
    [ "$normalized" = "$name" ] || continue
    if [ "$name" \< "$cutoff" ]; then
      purge_rc=0
      "$RCLONE_BIN" purge "$root/$name" || purge_rc=$?
      [ "$purge_rc" -eq 0 ] || rc="$purge_rc"
    fi
  done <<< "$listing"
  if [ "$rc" -eq 0 ]; then
    log_line "[SYS] stage=gdrive_backup step=$step status=ok rc=0"
    return 0
  else
    log_line "[SYS] stage=gdrive_backup step=$step status=failed rc=$rc"
    return 1
  fi
}

# Step: prune (dated folders only, never by modtime).
prune_versions "$REMOTE_ROOT/_versions" "$VERSION_RETENTION_DAYS" prune || FAILED=1
ARCHIVE_PRUNE_FAILED=0
prune_versions "$REMOTE_ROOT/_versions_archive" "$ARCHIVE_VERSION_RETENTION_DAYS" prune_archive || ARCHIVE_PRUNE_FAILED=1

if [ "$FAILED" -eq 0 ]; then
  STATUS_DIR="$CRYPTO_PILOT_ROOT/deploy/backup/status"
  mkdir -p "$STATUS_DIR"
  FINISHED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  STATUS_PARTIAL="$STATUS_DIR/last_success.json.partial"
  printf '{"started_at": "%s", "finished_at": "%s", "rc": 0}' "$BACKUP_STARTED_AT" "$FINISHED_AT" > "$STATUS_PARTIAL"
  sync
  if ! mv -f "$STATUS_PARTIAL" "$STATUS_DIR/last_success.json"; then
    log_line "[SYS] stage=gdrive_backup step=status status=failed"
    exit 1
  fi
  log_line "[SYS] stage=gdrive_backup step=status status=ok"
  if [ "$ARCHIVE_FAILED" -ne 0 ] || [ "$ARCHIVE_PRUNE_FAILED" -ne 0 ]; then
    log_line "[SYS] stage=gdrive_backup status=failed reason=archive"
    exit 1
  fi
  log_line "[SYS] stage=gdrive_backup status=ok"
  exit 0
else
  log_line "[SYS] stage=gdrive_backup status=failed"
  exit 1
fi
