"""SCENARIO_LIVE_22: AES-256-GCM 아티팩트 봉투의 결정성/변조 탐지/parquet 왕복."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import pandas as pd
import pytest
from pydantic import SecretStr

from src.live.crypto import MAGIC, derive_key, open_bytes, read_sealed_parquet, seal_bytes
from src.live.errors import ArtifactSealError

KEY = bytes(range(32))
KEY_B64 = base64.b64encode(KEY).decode()
P1 = b"deployed-target-weights-payload"
P2 = b"different-plaintext"


def test_SCENARIO_LIVE_22_seal_roundtrip_deterministic_tamper_evident(
    tmp_path: Path,
) -> None:
    # 라운드트립과 결정성: 동일 평문·동일 키는 바이트 동일 봉투다(I-SEAL-DETERMINISTIC).
    sealed_once = seal_bytes(P1, KEY)
    assert open_bytes(sealed_once, KEY) == P1
    assert seal_bytes(P1, KEY) == sealed_once

    # 상이 평문은 상이 nonce(HMAC 유도)를 받는다.
    assert seal_bytes(P2, KEY)[8:20] != sealed_once[8:20]

    # 변조 탐지: 임의 1바이트를 뒤집으면 전부 ArtifactSealError 다(오라클 방지 통일).
    for flip_index in (0, 10, len(sealed_once) - 1):
        tampered = bytearray(sealed_once)
        tampered[flip_index] ^= 0xFF
        with pytest.raises(ArtifactSealError):
            open_bytes(bytes(tampered), KEY)

    # 잘못된 키 / 짧은 blob / MAGIC 불일치도 예외 타입을 구분하지 않는다.
    wrong_key = bytes(range(32, 64))
    with pytest.raises(ArtifactSealError):
        open_bytes(sealed_once, wrong_key)
    with pytest.raises(ArtifactSealError):
        open_bytes(b"short", KEY)
    bad_magic = bytearray(sealed_once)
    bad_magic[0:8] = b"XXXXXXXX"
    with pytest.raises(ArtifactSealError):
        open_bytes(bytes(bad_magic), KEY)

    # 포맷 헤더와 AAD 가 일치한다.
    assert sealed_once[:8] == MAGIC


def test_derive_key_validates_base64_32_bytes() -> None:
    assert derive_key(SecretStr(KEY_B64)) == KEY
    # env 파이프라인이 감싼 따옴표/공백은 허용한다.
    assert derive_key(SecretStr(f'  "{KEY_B64}"\n')) == KEY
    assert derive_key(SecretStr(f"'{KEY_B64}'")) == KEY
    with pytest.raises(ArtifactSealError):
        derive_key(SecretStr("not-base64!!!"))
    with pytest.raises(ArtifactSealError):
        derive_key(SecretStr(base64.b64encode(b"too-short").decode()))


def test_read_sealed_parquet_roundtrip(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {"AAAUSDT": [0.02], "BUSDT": [-0.02]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-08-24 00:00Z")]),
    )
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=True)
    path = tmp_path / "deployed_target_weights.parquet.enc"
    path.write_bytes(seal_bytes(buffer.getvalue(), KEY))

    restored = read_sealed_parquet(path, KEY)
    pd.testing.assert_frame_equal(restored, frame)


#: 본 모듈이 검증하는 시나리오 ID(lean_check 추적용).
COVERED_SCENARIOS: tuple[str, ...] = (
    "SCENARIO_LIVE_22",  # SEAL_ROUNDTRIP_DETERMINISTIC_TAMPER_EVIDENT
)
