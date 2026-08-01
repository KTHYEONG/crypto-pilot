---
title: Binance Data Architecture
domain: market-data
type: architecture
status: active
priority: critical
ai_read_policy: always
related_paths:
  - src/market_data/binance/futures.py
  - src/market_data/binance/spot.py
  - src/market_data/binance/margin.py
  - src/market_data/binance/vision.py
  - src/market_data/services/futures_collection.py
  - src/market_data/services/spot_collection.py
  - src/market_data/services/borrow_collection.py
  - src/market_data/storage/manifest.py
change_triggers:
  - src/market_data/binance/*.py
  - src/market_data/services/*.py
last_verified: 2026-08-01
---

# Binance Data Ingestion & Storage Architecture

## 1. Overview
Binance REST API (Futures FAPI, Spot SAPI/V3, Margin SAPI) 및 Binance Vision S3 아카이브(`data.binance.vision`)로부터 Spot, Perpetual Futures, Margin Borrow 데이터를 수집, 정제, 무결성 검증 후 Parquet 데이터셋 및 JSON 매니페스트로 저장/관리하는 모듈화된 데이터 아키텍처입니다.

---

## 2. Core Components

| Component | Path | Responsibility |
|---|---|---|
| `BinanceClient` | `src/market_data/binance/futures.py` | Futures FAPI 원시 Klines (`/fapi/v1/klines`) 및 Funding Rate (`/fapi/v1/fundingRate`) 수집 |
| `BinanceSpotClient` | `src/market_data/binance/spot.py` | Spot REST API Klines (`/api/v3/klines`) 수집 및 타임프레임 변환 |
| `BinanceMarginClient` | `src/market_data/binance/margin.py` | SAPI 차입 이자율/이력 (`/sapi/v1/margin/interestRateHistory`) 수집 |
| `BinanceVisionDownloader` | `src/market_data/binance/vision.py` | Vision S3 (UM Futures metrics `.zip`) 아카이브 병렬 다운로드 및 SHA256 체크섬 검증 |
| `futures_collection.py` | `src/market_data/services/futures_collection.py` | Futures OHLCV (1m/1h), Funding Rate, Vision Metrics 수집 서비스 |
| `spot_collection.py` | `src/market_data/services/spot_collection.py` | Spot OHLCV (1h) 및 차입 이자율 병합 서비스 |
| `borrow_collection.py` | `src/market_data/services/borrow_collection.py` | 현물 마진 차입 이자율 수집/정규화/임포트 서비스 |
| `manifest.py` | `src/market_data/storage/manifest.py` | 데이터 무결성 지문(SHA256), 수집 메타데이터 및 품질(NaN Count 등) 매니페스트 관리 |

---

## 3. Data Flow

```text
[Binance FAPI / SAPI / Vision S3]
  -> [Ingestion & Rate Limit / Concurrency Control]
  -> [Normalization & Imputation Audit (Full Field Extraction)]
  -> [SHA256 Manifest Tracking & Local Storage (Parquet)]
  -> [Research / Evaluation Data Loaders]
```

---

## 4. Supported Data Types, Sources & Available Ranges

프로젝트에서 수집 및 활용 가능한 전체 데이터 종류, 수집 대상 소스 및 시간 가용 범위는 다음과 같습니다:

| 데이터 종류 (Data Type) | 수집 경로 / 소스 | 가용 범위 (Available Range) | 저장 형식 및 정규화 컬럼 스키마 | 비고 / 주요 활용 |
|---|---|---|---|---|
| **Futures OHLCV** (1m, 1h) | FAPI `/fapi/v1/klines` (실시간/최근) 및 Vision Monthly Klines (`monthly/klines/`) | 2019-09-25 ~ 현재 (상장일 이후) | Parquet (`timestamp`, `open`, `high`, `low`, `close`, `volume`, `datetime`, `quote_volume`, `taker_buy_base_volume`, `taker_buy_quote_volume`, `trades`) | Taker 볼륨 및 거래 건수 포함 전체 11개 raw 필드 보존 |
| **Spot OHLCV** (1h) | Spot `/api/v3/klines` | 2017-08-17 ~ 현재 (상장일 이후) | Parquet (`timestamp`, `open`, `high`, `low`, `close`, `volume`, `datetime`, `quote_volume`, `taker_buy_base_volume`, `taker_buy_quote_volume`, `trades`) | Cash & Carry 전용 현물 가격 및 거래량 데이터 |
| **Funding Rate** (8h / Event) | FAPI `/fapi/v1/fundingRate` 및 Vision Monthly Funding (`monthly/fundingRate/`) | 2019-09-25 ~ 현재 | Parquet (`timestamp`, `funding_rate`, `datetime`) | 선물 펀딩비 결제 이력 (00:00, 08:00, 16:00 UTC) |
| **Futures Metrics** (5m) | Binance Vision S3 (`daily/metrics/`) | 2020-09-01 ~ 현재 (Vision 아카이브) | Parquet (`timestamp`, `datetime`, `available_at`, `symbol`, `sum_open_interest`, `sum_open_interest_value`, `long_short_ratio`, `top_trader_long_short_ratio`, `sum_taker_long_short_vol_ratio`) | 미결제약정(OI), 롱숏비율(LSR), Taker 롱숏 볼륨 비율 (5분 릴리스 지연 `available_at` 반영) |
| **Indicator Klines** (Mark/Index/Premium) | Binance Vision S3 (`monthly/markPriceKlines`, `indexPriceKlines`, `premiumIndexKlines`) | 2020-01 ~ 현재 (Vision 아카이브) | Parquet (OHLCV 파생 스키마) | Mark Price, Index Price, Premium Index 시계열 데이터 |
| **Orderbook Depth (5 Level)** | Binance Vision S3 (`daily/bookDepth/`) | 2020-01 ~ 현재 (Vision 아카이브) | Raw / DataFrame (`timestamp`, `ask_price`, `bid_price` 등) | 호가창 스프레드(Half-spread) 및 슬리피지/심도 백테스트 분석용 |
| **Margin Borrow Rate** (Hourly/Daily) | Margin SAPI `/sapi/v1/margin/interestRateHistory` 및 CSV Import | 31일 이내 (SAPI) / 전체 과거 (CSV import) | Parquet (`timestamp`, `borrow_rate`, `accrual_seconds`) | 현물 차입 이자율 (SAPI 31일 경계 제한 시 CSV 수동 임포트 병합) |

---

## 5. Business Rules & Invariants

### Must Follow
- **PIT Integrity & Release Lag:** 모든 시계열 데이터는 타임스탬프 순으로 정렬되어야 하며 미래 참조를 금지합니다. 특히 Vision Metrics는 `available_at = timestamp + 5분` 지연을 적용하여 인과적 피드포워드 규칙을 보장합니다.
- **Full Field Extraction:** REST Klines의 모든 11개 필드(`quote_volume`, `taker_buy_base_volume` 등)를 완전히 정규화하여 저장합니다. 누락 시 유동성 게이트 등 파생 지표가 NaN으로 오염되는 것을 방지합니다.
- **Manifest Fingerprinting:** 데이터셋 생성/업데이트 시 파일 SHA256 해시, 레코드 수, NaN 개수, 수집 타임스탬프를 `manifest.json`에 기록하여 재현 가능한 지문(Provenance)을 보장합니다.
- **Cache Merge Precedence:** 기존 Parquet 캐시와 신규 수집 데이터를 병합할 때(`drop_duplicates(subset=["timestamp"], keep="last")`), 동일 timestamp에 대해 신규 fetch 데이터가 기존 캐시를 오버라이드하여 캐시 자가치유(Self-healing)를 보장합니다.

### Must Not Do
- **Zero-Filling Missing Funding/Borrow:** missing funding rate나 borrow rate를 절대로 `0.0`으로 임의 대체해서는 안 됩니다. 데이터 부재 시 `DataIntegrityError`를 발생시켜 평가를 PENDING으로 처리해야 합니다.
- **Unverified Vision Download:** Vision 아카이브 zip 파일 다운로드 시 제공되는 `.CHECKSUM` 파일과의 SHA256 검증을 생략해서는 안 됩니다.

---

## 6. Directory Layout & Storage Protocol

```text
data/
  ├── futures/
  │   ├── ohlcv/
  │   │   ├── 1m/ {SYMBOL}.parquet
  │   │   └── 1h/ {SYMBOL}.parquet
  │   ├── funding/ {SYMBOL}.parquet
  │   └── metrics/ {SYMBOL}.parquet
  ├── spot/
  │   ├── ohlcv/ 1h/ {SYMBOL}.parquet
  │   ├── borrow/ {SYMBOL}.parquet
  │   └── manifest.json
  └── manifest.json
```

---

## 7. Testing Expectations
- **Continuity Test:** 타임스탬프 간격의 일관성 및 누락 구간 보간 여부 확인.
- **Integrity Test:** 다운로드된 파일의 SHA256이 Manifest와 일치하는지 확인.
- **Deterministic Isolation:** 테스트는 실 Binance/FAPI/Vision 엔드포인트를 호출하지 않고, 네트워크 경계(`urllib/ccxt`)를 mock 하여 반복 가능하게 유지.
- **Boundary-Only Mocking:** 내부 변환 로직/계산 로직은 실제 구현을 검증하고 외부 I/O 경계만 대체.
