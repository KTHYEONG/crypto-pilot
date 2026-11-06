#!/usr/bin/env bash
# Usage: crypto-ownership-fixup.sh <check|fix>
# Re-owns entries under data/, logs/, deploy/mhs that root containers created,
# so the non-root (host ubuntu) services can write them.
#   check: read-only; exit 0 when clean, 10 when foreign-owned entries exist.
#   fix:   re-owns foreign entries via a root helper container; exit 0 only if a re-check is clean.
# Config (env overrides):
#   CRYPTO_FIXUP_ROOT=$HOME/crypto-pilot       Repo root on the host.
#   CRYPTO_IMAGE=ghcr.io/kthyeong/crypto-pilot-live:latest  Image whose /usr/bin/find and chown run the re-own.
set -Eeuo pipefail

MODE="${1:-}"
case "$MODE" in
  check|fix) ;;
  *) echo "usage: crypto-ownership-fixup.sh <check|fix>" >&2; exit 2 ;;
esac

ROOT="${CRYPTO_FIXUP_ROOT:-$HOME/crypto-pilot}"
IMAGE="${CRYPTO_IMAGE:-ghcr.io/kthyeong/crypto-pilot-live:latest}"
UID_N="$(id -u)"
GID_N="$(id -g)"
OWNED_PATHS=(data logs deploy/mhs)

if [ "$UID_N" = "0" ]; then
  echo "crypto-ownership-fixup mode=${MODE} status=refused reason=root_caller" >&2
  exit 2
fi

existing_paths() {
  local rel
  for rel in "${OWNED_PATHS[@]}"; do
    if [ -e "$ROOT/$rel" ]; then printf '%s\n' "$ROOT/$rel"; fi
  done
}

has_foreign() {
  local out rc=0
  out="$(find "$1" \( ! -user "$UID_N" -o ! -group "$GID_N" \) -print -quit 2>/dev/null)" || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "crypto-ownership-fixup mode=${MODE} status=error reason=find_failed" >&2
    exit 2
  fi
  [ -n "$out" ]
}

foreign_paths() {
  local p
  while IFS= read -r p; do
    if has_foreign "$p"; then printf '%s\n' "$p"; fi
  done < <(existing_paths)
}

FOREIGN="$(foreign_paths)"

if [ "$MODE" = "check" ]; then
  if [ -z "$FOREIGN" ]; then
    echo "foreign=0"
    echo "crypto-ownership-fixup mode=check status=clean"
    exit 0
  fi
  echo "foreign=1"
  echo "crypto-ownership-fixup mode=check status=foreign"
  exit 10
fi

if [ -z "$FOREIGN" ]; then
  echo "foreign=0"
  echo "crypto-ownership-fixup mode=fix status=clean"
  exit 0
fi

while IFS= read -r p; do
  if ! docker run --rm --pull never --network none --user 0:0 --entrypoint /usr/bin/find \
    -v "$p:/fix" "$IMAGE" /fix "(" ! -user "$UID_N" -o ! -group "$GID_N" ")" \
    -exec chown -h "$UID_N:$GID_N" {} + >/dev/null; then
    echo "crypto-ownership-fixup mode=fix step=reown status=docker_failed" >&2
  fi
done <<< "$FOREIGN"

REMAINING="$(foreign_paths)"
if [ -z "$REMAINING" ]; then
  echo "foreign=0"
  echo "crypto-ownership-fixup mode=fix status=fixed"
  exit 0
fi
echo "foreign=1"
echo "crypto-ownership-fixup mode=fix status=residual"
exit 11
