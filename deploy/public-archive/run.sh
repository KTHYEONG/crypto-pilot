#!/usr/bin/env bash
# Public-archive incremental mirror: collect, then backup-gated prune.
# Drive is never written here, so the shared gdrive lock is intentionally not taken.
set -uo pipefail

CRYPTO_PILOT_ROOT="${CRYPTO_PILOT_ROOT:-$HOME/crypto-pilot}"
LOG_DIR="${LOG_DIR:-$HOME/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/crypto-pilot-public-archive-$(date -u +%F).log}"

mkdir -p "$LOG_DIR" || exit 1
cd "$CRYPTO_PILOT_ROOT" || exit 1

log_rc() {
  printf '[SYS] stage=public_archive step=%s rc=%s\n' "$1" "$2" >> "$LOG_FILE"
}

RC1=0
RC2=0

docker compose --profile archive run --rm public-archive >> "$LOG_FILE" 2>&1 || RC1=$?
log_rc 1 "$RC1"

docker compose --profile archive run --rm public-archive /app/.venv/bin/python -m src.cli.main archive prune >> "$LOG_FILE" 2>&1 || RC2=$?
log_rc 2 "$RC2"

if [ "$RC1" -ne 0 ]; then
  exit "$RC1"
fi
exit 0
