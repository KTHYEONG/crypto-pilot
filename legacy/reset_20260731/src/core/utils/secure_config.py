import base64
import json
import logging
import os
from typing import Any, cast

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

_logger = logging.getLogger("SecureConfig")


def _get_fernet_key(passphrase: str) -> bytes:
    """패스프레이즈로부터 유효한 Fernet 키를 생성합니다."""
    salt = b"my_coin_traider_salt"  # Fixed salt for deterministic key from passphrase
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(passphrase.encode()))
    return key


def encrypt_config(config: dict[str, Any], passphrase: str) -> bytes:
    """설정 딕셔너리를 바이트로 암호화합니다."""
    f = Fernet(_get_fernet_key(passphrase))
    data = json.dumps(config).encode("utf-8")
    return f.encrypt(data)


def decrypt_config(encrypted_data: bytes, passphrase: str) -> dict[str, Any]:
    """암호화된 바이트를 설정 딕셔너리로 복호화합니다."""
    f = Fernet(_get_fernet_key(passphrase))
    decrypted = f.decrypt(encrypted_data)
    return cast(dict[str, Any], json.loads(decrypted.decode("utf-8")))


def get_strategy_secret() -> str | None:
    """환경 변수에서 전략 비밀 키를 조회합니다."""
    return os.getenv("STRATEGY_SECRET_KEY")
