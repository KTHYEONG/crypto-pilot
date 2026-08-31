"""AES-256-GCM 아티팩트 봉투(seal). 도메인 지식 0, I/O 0의 순수 변환 계층.

I-SEAL-DETERMINISTIC: nonce = HMAC-SHA256(key, MAGIC || plaintext)[:12].
동일 평문·동일 키는 바이트 동일한 봉투를 낳아 git diff 노이즈가 없고, 서로 다른
평문은 압도적 확률로 다른 nonce 를 받아 GCM nonce 재사용이 발생하지 않는다.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import io
from pathlib import Path

import pandas as pd
from pydantic import SecretStr

from src.live.errors import ArtifactSealError

#: 포맷/버전 마법이자 AES-GCM AAD(헤더 변조 탐지).
MAGIC = b"CPSEAL01"

_NONCE_LEN = 12
_TAG_LEN = 16
_HEADER_LEN = len(MAGIC) + _NONCE_LEN
_KEY_LEN = 32

#: 오라클 방지: 실패 원인을 구분하지 않는 단일 메시지.
_GENERIC_SEAL_ERROR = "artifact seal envelope is invalid"


def derive_key(secret: SecretStr) -> bytes:
    """base64 디코드 후 정확히 32 바이트(AES-256 키)여야 한다."""
    raw = secret.get_secret_value().strip()
    # env_file / sops exec-env 파이프라인이 값을 감싼 따옴표를 그대로 넘기는 경우가 있다.
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
        raw = raw[1:-1].strip()
    try:
        key = base64.b64decode(raw.encode("ascii"), validate=True)
    except Exception as exc:  # noqa: BLE001 - 유형 불문 잘못된 키는 동일하게 실패한다
        raise ArtifactSealError(_GENERIC_SEAL_ERROR) from exc
    if len(key) != _KEY_LEN:
        raise ArtifactSealError(_GENERIC_SEAL_ERROR)
    return key


def seal_bytes(plaintext: bytes, key: bytes) -> bytes:
    """MAGIC + nonce(12) + AESGCM(key).encrypt(nonce, plaintext, MAGIC)."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce = _deterministic_nonce(key, plaintext)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, MAGIC)
    return MAGIC + nonce + ciphertext


def open_bytes(blob: bytes, key: bytes) -> bytes:
    """MAGIC 불일치 / 길이 부족 / GCM 태그 실패는 전부 동일 예외로 통일한다."""
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if len(blob) < _HEADER_LEN + _TAG_LEN or blob[: len(MAGIC)] != MAGIC:
        raise ArtifactSealError(_GENERIC_SEAL_ERROR)
    nonce = blob[len(MAGIC):_HEADER_LEN]
    ciphertext = blob[_HEADER_LEN:]
    try:
        return AESGCM(key).decrypt(nonce, ciphertext, MAGIC)
    except InvalidTag as exc:
        raise ArtifactSealError(_GENERIC_SEAL_ERROR) from exc


def read_sealed_parquet(path: Path, key: bytes) -> pd.DataFrame:
    """복호화 결과를 io.BytesIO 로 parquet 역직렬화한다. 평문은 디스크에 쓰지 않는다."""
    plaintext = open_bytes(path.read_bytes(), key)
    return pd.read_parquet(io.BytesIO(plaintext))


def _deterministic_nonce(key: bytes, plaintext: bytes) -> bytes:
    """I-SEAL-DETERMINISTIC: HMAC-SHA256(key, MAGIC || plaintext)[:12]."""
    return hmac.new(key, MAGIC + plaintext, hashlib.sha256).digest()[:_NONCE_LEN]
