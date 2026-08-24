"""Deployed-artifact seal CLI: keygen / seal / unseal subcommands.

키는 LIVE_ARTIFACT_KEY 환경변수로만 공급한다(argv 로 받으면 프로세스 목록에 유출된다).
"""

from __future__ import annotations

import argparse
import base64
import logging
import os
import secrets
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import SecretStr

from src.live.crypto import derive_key, open_bytes, seal_bytes
from src.live.errors import ArtifactSealError

logger = logging.getLogger("SealArtifact")

_KEY_ENV_VAR = "LIVE_ARTIFACT_KEY"


def _key_from_env() -> SecretStr:
    value = os.environ.get(_KEY_ENV_VAR)
    if not value:
        logger.error("[SYS] missing %s environment variable", _KEY_ENV_VAR)
        raise SystemExit(2)
    return SecretStr(value)


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        logger.error("[SYS] cannot read %s: %s", path, exc)
        raise SystemExit(2) from exc


def _write_bytes(path: Path, blob: bytes) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    except OSError as exc:
        logger.error("[SYS] cannot write %s: %s", path, exc)
        raise SystemExit(2) from exc


def main(argv: Sequence[str] | None = None) -> int:
    """서브커맨드 진입점. 성공 시 0, 봉투 오류 시 1 을 반환한다."""
    parser = argparse.ArgumentParser(prog="seal_artifact", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("keygen", help="emit a fresh base64 32-byte artifact key")

    seal_parser = subparsers.add_parser("seal", help="seal a plaintext file")
    seal_parser.add_argument("--in", dest="input_path", type=Path, required=True)
    seal_parser.add_argument("--out", dest="output_path", type=Path, required=True)

    unseal_parser = subparsers.add_parser("unseal", help="open a sealed file")
    unseal_parser.add_argument("--in", dest="input_path", type=Path, required=True)
    unseal_parser.add_argument("--out", dest="output_path", type=Path, required=True)

    args = parser.parse_args(argv)

    if args.command == "keygen":
        print(base64.b64encode(secrets.token_bytes(32)).decode("ascii"))
        return 0

    key = derive_key(_key_from_env())
    payload = _read_bytes(args.input_path)
    try:
        blob = seal_bytes(payload, key) if args.command == "seal" else open_bytes(payload, key)
    except ArtifactSealError as exc:
        logger.error("[SYS] seal operation failed: %s", exc)
        return 1
    _write_bytes(args.output_path, blob)
    return 0


if __name__ == "__main__":
    sys.exit(main())
