import os
import json
import base64
import logging
from typing import Optional, Dict, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

_logger = logging.getLogger("SecureConfig")

def _get_fernet_key(passphrase: str) -> bytes:
    """Generate a valid Fernet key from a passphrase."""
    salt = b'my_coin_traider_salt' # Fixed salt for deterministic key from passphrase
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(passphrase.encode()))
    return key

def encrypt_config(config: Dict[str, Any], passphrase: str) -> bytes:
    """Encrypt config dict to bytes."""
    f = Fernet(_get_fernet_key(passphrase))
    data = json.dumps(config).encode('utf-8')
    return f.encrypt(data)

def decrypt_config(encrypted_data: bytes, passphrase: str) -> Dict[str, Any]:
    """Decrypt bytes to config dict."""
    f = Fernet(_get_fernet_key(passphrase))
    decrypted = f.decrypt(encrypted_data)
    return json.loads(decrypted.decode('utf-8'))

def get_strategy_secret() -> Optional[str]:
    """Retrieve strategy secret from environment."""
    return os.getenv("STRATEGY_SECRET_KEY")
