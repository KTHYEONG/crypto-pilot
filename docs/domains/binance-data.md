---
title: Binance Futures Data Architecture
domain: futures-data
type: architecture
status: active
priority: critical
ai_read_policy: always
related_paths:
  - src/core/exchange/binance_client.py
  - src/domain/futures/backtest/data_loader.py
  - src/core/utils/binance_vision.py
change_triggers:
  - src/core/exchange/binance_*.py
  - src/core/utils/binance_vision.py
last_verified: 2026-05-25
---

# Binance Futures Data Architecture

## 1. Overview
Binance API 및 Vision 아카이브로부터 고해상도(1m) 및 의사결정용(1h/4h) 데이터를 수집, 정제, 캐싱하여 백테스트 및 라이브 엔진에 공급하는 데이터 레이어입니다. CCXT와 Vision을 병합한 Hybrid 전략을 통해 2019년 이후의 모든 시계열 데이터 가용성을 보장합니다.

---

## 2. Core Components

| Component | Responsibility |
|---|---|
| `binance_client.py` | Rest API/Websocket을 통한 원시 데이터(FAPI) 수집 |
| `binance_vision.py` | Vision S3 아카이브(.zip) 다운로드 및 체크섬 검증 |
| `data_loader.py` | 로컬 캐시 및 DB로부터 데이터 로딩 및 정렬 |
| `cache_manager.py` | 시계열 데이터의 직렬화 및 고속 캐싱 관리 |

---

## 3. Data Flow

```text
[Binance API / Vision S3] 
  -> [Hybrid Ingestion (CCXT < 2020 <= Vision)] 
  -> [Cleaning/Imputation & Checksum Verify] 
  -> [Local Storage/Cache (Parquet)] 
  -> [Domain Model Alignment (2D Numpy)]
```

---

## 4. Business Rules

### Must Follow
- **PIT Integrity:** 모든 데이터는 타임스탬프 기준으로 정렬되어야 하며 미래 데이터 참조 금지.
- **Hybrid Source Selection:** 2020년 이전은 CCXT, 이후는 Vision 아카이브를 우선 사용.
- **Checksum Verification:** Vision 다운로드 시마다 `.CHECKSUM` 파일의 SHA256 검증 필수.
- **Imputation Audit:** 누락 데이터 보간 시 `warm_mask=False` 등으로 보간 여부를 명시.

### Must Not Do
- **Roll Spread Usage (2020+):** 2020년 이후 구간에서는 과대추정 위험으로 Roll(1984) spread 모델 사용 금지 (대신 bookDepth 사용).
- **Redundant CCXT Calls:** 2019년 구간 데이터는 1회 수집 후 로컬 동결하여 재호출 방지.

### Invariants
- **Timestamp Uniqueness:** 동일 심볼/타임프레임 내 중복 타임스탬프 금지.
- **Deterministic Manifest:** `data_manifest_hash`를 통해 데이터 셋의 재현성 지문 유지.

---

## 5. Detailed Specifications

### 5.1 소스별 가용 데이터 및 범위

| 데이터 종류 | 소스 | 가용 범위 | 비고 |
|---|---|---|---|
| 심볼 상장일 | FAPI `exchangeInfo` | 2019-09-25~ | 누락분은 Vision 첫 파일로 보완 |
| OHLCV (Pre-2020) | CCXT `fetch_ohlcv` | 2019-09~2019-12 | Vision 부재 구간 전용 |
| OHLCV (Post-2020) | Vision `klines` | 2020-01~ | Daily/Monthly 아카이브 |
| 펀딩비 | FAPI / Vision | 2019-09~ | FAPI(전범위), Vision(Monthly 전용) |
| OI / LSR | Vision `metrics` | 2020-09-01~ | sum_open_interest, LSR 컬럼 포함 |
| 호가창 Spread/Depth | Vision `bookDepth` | 2020~ | 12 depth level, spread 계산용 |

### 5.2 Half-spread 계산 전략

- **date ≥ 2020-01-01:** Vision `bookDepth` 집계. `half_spread = median(ask_price - mid_price)`
- **date < 2020-01-01:** Roll spread fallback. `half_spread ≈ 0.5 × (High - Low) / Close` 또는 고정치(BTC: 0.02%) 적용.

### 5.3 Data Manifest Schema
```python
@dataclass
class ManifestRow:
    source: str           # "vision" | "ccxt" | "fapi"
    symbol: str
    tf: str
    period: str           # YYYY-MM-DD
    sha256: str           # 무결성 검증용 해시
    is_final: bool        # 확정 데이터 여부
```

---

## 6. Examples
- **Input:** 2019-11-01 데이터 요청 -> **Output:** CCXT 소스 선택 및 로컬 캐시 확인.
- **Input:** 2024-05-01 데이터 요청 -> **Output:** Vision `daily/klines` 다운로드 및 SHA256 검증.

---

## 7. Testing Expectations
- **Continuity Test:** 타임스탬프 간격의 일관성 및 누락 구간 보간 여부 확인.
- **Integrity Test:** 다운로드된 파일의 SHA256이 Manifest와 일치하는지 확인.
- **Deterministic Isolation:** 테스트는 실 Binance/FAPI/Vision 엔드포인트를 호출하지 않고, 네트워크 경계(`urllib/ccxt`)를 mock 하여 반복 가능하게 유지.
- **Boundary-Only Mocking:** 내부 변환 로직/계산 로직은 실제 구현을 검증하고 외부 I/O 경계만 대체.
