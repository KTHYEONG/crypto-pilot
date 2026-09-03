---
title: Binance Data Collection & Storage Architecture
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
  - src/market_data/services/mhs_execution.py
  - src/market_data/storage/manifest.py
change_triggers:
  - src/market_data/binance/*.py
  - src/market_data/services/*.py
  - src/market_data/storage/*.py
last_verified: 2026-09-03
---

# 바이낸스 데이터 수집 및 스토리지 아키텍처 (Binance Data Architecture)

## 1. 개요 (Overview)
본 문서는 Binance REST API (Futures FAPI, Spot SAPI/V3, Margin SAPI) 및 Binance Vision S3 아카이브(`data.binance.vision`)로부터 Perpetual Futures, Spot, Margin Borrow 및 Microstructure 데이터를 수집, 정규화, 무결성 검증 후 Parquet 데이터셋 및 JSON 매니페스트로 저장하고 관리하는 데이터 파이프라인 아키텍처를 정의합니다.

연구(Research) 및 실거래(Live) 환경 모두에서 Point-In-Time (PIT) 인과성을 엄격히 유지하며, 미래 데이터 누출(Look-ahead bias)을 원천 차단합니다.

---

## 2. 핵심 컴포넌트 및 책임 (Core Components)

| 컴포넌트 | 소스 경로 | 주요 책임 및 역할 |
|---|---|---|
| [`BinanceClient`](file:///home/kth/crypto-pilot/src/market_data/binance/futures.py) | [src/market_data/binance/futures.py](file:///home/kth/crypto-pilot/src/market_data/binance/futures.py) | Futures FAPI 원시 Klines (`/fapi/v1/klines`) 및 Funding Rate (`/fapi/v1/fundingRate`) 수집, Rate limit 및 재시도 제어 |
| [`BinanceSpotClient`](file:///home/kth/crypto-pilot/src/market_data/binance/spot.py) | [src/market_data/binance/spot.py](file:///home/kth/crypto-pilot/src/market_data/binance/spot.py) | Spot REST API Klines (`/api/v3/klines`) 수집 및 타임프레임 변환 |
| [`BinanceMarginClient`](file:///home/kth/crypto-pilot/src/market_data/binance/margin.py) | [src/market_data/binance/margin.py](file:///home/kth/crypto-pilot/src/market_data/binance/margin.py) | SAPI 차입 이자율 및 이력 (`/sapi/v1/margin/interestRateHistory`) 수집 |
| [`BinanceVisionDownloader`](file:///home/kth/crypto-pilot/src/market_data/binance/vision.py) | [src/market_data/binance/vision.py](file:///home/kth/crypto-pilot/src/market_data/binance/vision.py) | Vision S3 아카이브(Klines, Funding, Metrics, BookDepth 등 `.zip`) 병렬 다운로드 및 SHA256 체크섬 검증 |
| [`futures_collection.py`](file:///home/kth/crypto-pilot/src/market_data/services/futures_collection.py) | [src/market_data/services/futures_collection.py](file:///home/kth/crypto-pilot/src/market_data/services/futures_collection.py) | Futures OHLCV (1m/1h), Funding Rate, Vision Metrics 수집 및 캐시 오케스트레이션 서비스 |
| [`spot_collection.py`](file:///home/kth/crypto-pilot/src/market_data/services/spot_collection.py) | [src/market_data/services/spot_collection.py](file:///home/kth/crypto-pilot/src/market_data/services/spot_collection.py) | Spot OHLCV (1h) 및 차입 이자율 병합 서비스 |
| [`borrow_collection.py`](file:///home/kth/crypto-pilot/src/market_data/services/borrow_collection.py) | [src/market_data/services/borrow_collection.py](file:///home/kth/crypto-pilot/src/market_data/services/borrow_collection.py) | 현물 마진 차입 이자율 수집, 정규화 및 수동 CSV 임포트 서비스 |
| [`mhs_execution.py`](file:///home/kth/crypto-pilot/src/market_data/services/mhs_execution.py) | [src/market_data/services/mhs_execution.py](file:///home/kth/crypto-pilot/src/market_data/services/mhs_execution.py) | MHS 체결 시뮬레이션 전용 5m OHLCV 및 1h Mark Price 데이터셋 수집 서비스 |
| [`manifest.py`](file:///home/kth/crypto-pilot/src/market_data/storage/manifest.py) | [src/market_data/storage/manifest.py](file:///home/kth/crypto-pilot/src/market_data/storage/manifest.py) | 데이터 무결성 SHA256 지문, 레코드 수, 수집 메타데이터 및 품질(NaN 카운트 등) 매니페스트 관리 |

---

## 3. 데이터 파이프라인 흐름 (Data Flow)

```text
[Binance FAPI / SAPI / Vision S3]
        │
        ▼ (HTTP REST API 요청 / S3 병렬 다운로드 & Checksum 검증)
[Ingestion & Rate Limit / Concurrency Control]
        │
        ▼ (11개 raw 필드 정규화, 타임존 UTC 고정, 결손 감사)
[Normalization & Imputation Audit (Full Field Extraction)]
        │
        ▼ (기존 캐시 자가치유 병합 & SHA256 해시 등록)
[SHA256 Manifest Tracking & Local Storage (Parquet, zstd)]
        │
        ▼
[Research / MHS Execution / Live Replay Data Loaders]
```

---

## 4. 수집 가능한 전체 데이터 항목, 소스 및 가용 범위 (Supported Data Inventory)

프로젝트에서 바이낸스로부터 수집 및 활용 가능한 전체 데이터 종류, 수집 소스 경로, 시간 가용 범위 및 저장 스키마는 다음과 같습니다:

| 데이터 종류 (Data Type) | 수집 경로 / 소스 | 가용 범위 (Available Range) | 저장 형식 및 정규화 컬럼 스키마 | 비고 / 주요 활용 |
|---|---|---|---|---|
| **Futures OHLCV** (1m, 1h) | FAPI `/fapi/v1/klines` (실시간/최근) 및 Vision Monthly Klines (`monthly/klines/`) | 2019-09-25 ~ 현재 (심볼별 상장일 이후) | Parquet (`timestamp`, `open`, `high`, `low`, `close`, `volume`, `datetime`, `quote_volume`, `taker_buy_base_volume`, `taker_buy_quote_volume`, `trades`) | Taker 볼륨 및 거래 건수 포함 전체 11개 raw 필드 보존. 유동성 게이트 및 알파 신호 산출의 기본 데이터 |
| **Futures Execution OHLCV** (5m) | FAPI `/fapi/v1/klines` 및 Vision Klines | 2021-01-01 ~ 현재 | Parquet (`timestamp`, `open`, `high`, `low`, `close`, `volume`, `datetime`, `quote_volume`, `taker_buy_base_volume`, `taker_buy_quote_volume`, `trades`) | MHS 체결 리플레이([`simulated_inventory_ledger`](file:///home/kth/crypto-pilot/src/mhs/execution/ledger.py))의 5분봉 프록시 체결 및 슬리피지/수수료 시뮬레이션용 |
| **Spot OHLCV** (1h) | Spot `/api/v3/klines` | 2017-08-17 ~ 현재 (상장일 이후) | Parquet (`timestamp`, `open`, `high`, `low`, `close`, `volume`, `datetime`, `quote_volume`, `taker_buy_base_volume`, `taker_buy_quote_volume`, `trades`) | Cash & Carry 차익거래 및 현물-선물 베이시스 분석용 현물 가격/거래량 데이터 |
| **Funding Rate** (8h / Event) | FAPI `/fapi/v1/fundingRate` 및 Vision Monthly Funding (`monthly/fundingRate/`) | 2019-09-25 ~ 현재 | Parquet (`timestamp`, `funding_rate`, `datetime`) | 선물 펀딩비 결제 이력 (00:00, 08:00, 16:00 UTC). 원장의 `Accrued Funding Charge` 정산에 필수 |
| **Futures Metrics** (5m) | Binance Vision S3 (`daily/metrics/`) | 2020-09-01 ~ 현재 (Vision 아카이브) | Parquet (`timestamp`, `datetime`, `available_at`, `symbol`, `sum_open_interest`, `sum_open_interest_value`, `long_short_ratio`, `top_trader_long_short_ratio`, `sum_taker_long_short_vol_ratio`) | 미결제약정(OI), 롱숏비율(LSR), 상위 트레이더 포지션 비율, Taker 볼륨 비율 (5분 릴리스 지연 `available_at` 필수 반영) |
| **Indicator Klines** (Mark/Index/Premium) | Binance Vision S3 (`monthly/markPriceKlines`, `indexPriceKlines`, `premiumIndexKlines`) | 2020-01 ~ 현재 (Vision 아카이브) | Parquet (`timestamp`, `open`, `high`, `low`, `close`, `datetime`) | Mark Price(MTM 평가용 단일 진실원천), Index Price, Premium Index 시계열 데이터 |
| **Orderbook Depth (5 Level)** | Binance Vision S3 (`daily/bookDepth/`) | 2020-01 ~ 현재 (Vision 아카이브) | Raw / Parquet (`timestamp`, `ask_price_1~5`, `ask_qty_1~5`, `bid_price_1~5`, `bid_qty_1~5` 등) | 호가창 스프레드(Half-spread), 오더북 비대칭도 및 슬리피지/시장 충격(Market Impact) 정밀 모델링용 |
| **Margin Borrow Rate** (Hourly/Daily) | Margin SAPI `/sapi/v1/margin/interestRateHistory` 및 CSV Import | 최근 31일 (SAPI 제한) / 전체 과거 (CSV 수동 임포트) | Parquet (`timestamp`, `borrow_rate`, `accrual_seconds`) | 현물 마진 차입 이자율 (SAPI 31일 경계 제한 시 과거 CSV 수동 임포트와 무손실 병합) |

---

## 5. 실시간 데이터 스트림 스키마 (Live Data Streams)

실거래 및 섀도우 트레이딩 환경에서 기록 및 관리되는 데이터 스트림 및 파티션 정책은 다음과 같습니다:

| 스트림명 (Stream) | 저장 경로 (Path) | 파티션 정책 (Partition Scheme) | 이벤트 시간 컬럼 | 주요 스키마 컬럼 (Schema) | 보관 정책 (Retention) |
|---|---|---|---|---|---|
| `ohlcv/1h` | `data/futures/ohlcv/1h/*.parquet` | per-symbol parquet | `timestamp` (epoch ms, int64) | `timestamp, open, high, low, close, volume, quote_vol` | `data_retention_days` (기본 220일) |
| `markPriceKlines/1h` | `data/futures/markPriceKlines/1h/*.parquet` | per-symbol parquet | `timestamp` (epoch ms, int64) | `timestamp, open, high, low, close` | `data_retention_days` (기본 220일) |
| `funding` | `data/futures/funding/*.parquet` | per-symbol parquet | `timestamp` (epoch ms, int64) | `timestamp, funding_rate` | `data_retention_days` (기본 220일) |
| `live_fills` | `data/state/live_fills/*.parquet` | 월별 샤드 (Monthly) | `decision_time` (UTC), `timestamp` (fill) | `decision_time, timestamp, symbol, side, qty, price` | 영구 보관 (Kept) |
| `live_execution_quality`| `data/state/live_execution_quality/*.parquet` | 월별 샤드 (Monthly) | `decision_time` (UTC) | `decision_time, symbol, slippage_bps` | 영구 보관 (Kept) |
| `live_microstructure` | `data/state/live_microstructure/*.parquet` | 월별 샤드 (Monthly) | `decision_time` (UTC) | `decision_time, symbol, spread_bps` | 영구 보관 (Kept) |
| `live_portfolio_state` | `data/state/live_portfolio_state/*.parquet` | 순환 샤드 (Rotating) | `decision_time` (UTC) | `decision_time, equity, positions` | 영구 보관 (Kept) |
| `live_orderbook` | `data/state/live_orderbook/live_orderbook_YYYYMMDD.parquet` | 일별 파켓 (Daily) | `captured_at` (UTC) | `captured_at, decision_time, symbol, best_bid, best_ask` | `orderbook_retention_days` (기본 365일) |
| `live_tax_ledger` | `data/state/live_tax_ledger/*.jsonl` | 월별 JSONL (Monthly) | `decision_time` (UTC) | `decision_time, realized_pnl` | 영구 보관 (Kept forever) |

- 모든 Parquet 파일은 **zstd** 압축 알고리즘을 사용합니다.
- `decision_time`은 전략의 정규 이벤트 시각이며, `live_fills`는 실제 거래소 체결 시각인 `timestamp`를 함께 유지합니다.

---

## 6. 비즈니스 규칙 및 불변식 (Business Rules & Invariants)

### 반드시 준수해야 하는 규칙 (Must Follow)
1. **Point-In-Time (PIT) 무결성 및 릴리스 지연 (Release Lag):**
   - 모든 시계열 데이터는 타임스탬프 순으로 정렬되어야 하며 미래 데이터 참조를 원천 금지합니다.
   - 특히 Vision Metrics (OI, LSR 등)는 수집 시 `available_at = timestamp + 5분` 지연을 강제 적용하여 인과적 피드포워드(Causal Feedforward) 규칙을 엄격히 보장합니다.
2. **전체 필드 보존 (Full Field Extraction):**
   - Binance REST Klines의 모든 11개 필드(`quote_volume`, `taker_buy_base_volume`, `taker_buy_quote_volume`, `trades` 등)를 온전히 보존하여 저장합니다.
   - 거래대금이나 테이커 볼륨 누락 시 유동성 게이트, 플로우 불균형(`flow_imb`) 신호가 NaN으로 오염되는 것을 방지합니다.
3. **매니페스트 지문 등록 (Manifest Fingerprinting):**
   - 데이터셋 신규 생성 및 증분 업데이트 시 파일의 SHA256 해시, 레코드 수, 결손(NaN) 수, 수집 시각을 `manifest.json`에 기록하여 재현 가능한 데이터 출처(Data Provenance)를 보장합니다.
4. **캐시 자가치유 병합 (Self-healing Cache Merge Precedence):**
   - 기존 Parquet 캐시와 신규 수집 데이터를 병합할 때(`drop_duplicates(subset=["timestamp"], keep="last")`), 동일 timestamp에 대해서는 신규 fetch 데이터가 기존 캐시를 오버라이드하여 데이터 정정(Self-healing)을 보장합니다.

### 절대 금지해야 하는 행위 (Must Not Do)
1. **결손 펀딩비/이자율 임의 대체 금지 (No Zero-Filling):**
   - 누락된 펀딩비(funding rate)나 차입 이자율(borrow rate)을 절대로 `0.0`으로 임의 대체해서는 안 됩니다. 데이터 부재 시 [`DataIntegrityError`](file:///home/kth/crypto-pilot/src/common/errors.py)를 발생시키고 해당 구간 평가를 Fail-Closed 처리해야 합니다.
2. **체크섬 미검증 다운로드 금지 (No Unverified Download):**
   - Vision 아카이브 zip 파일 다운로드 시 함께 제공되는 `.CHECKSUM` 파일과의 SHA256 검증을 생략해서는 안 됩니다. 체크섬 불일치 시 즉시 폐기하고 재수집해야 합니다.

---

## 7. 로컬 스토리지 디렉토리 구조 (Directory Layout)

```text
data/
  ├── futures/
  │   ├── ohlcv/
  │   │   ├── 1m/ {SYMBOL}.parquet       # 1분봉 원시 캔들
  │   │   ├── 5m/ {SYMBOL}.parquet       # 5분봉 MHS 체결 리플레이용 캔들
  │   │   └── 1h/ {SYMBOL}.parquet       # 1시간봉 시그널 생성용 캔들
  │   ├── markPriceKlines/
  │   │   └── 1h/ {SYMBOL}.parquet       # 1시간봉 마크 가격 캔들
  │   ├── funding/ {SYMBOL}.parquet      # 선물 8시간 펀딩비
  │   ├── metrics/ {SYMBOL}.parquet      # Vision 5분 메트릭스 (OI, LSR 등)
  │   └── bookDepth/ {SYMBOL}.parquet    # 5단계 오더북 뎁스
  ├── spot/
  │   ├── ohlcv/ 1h/ {SYMBOL}.parquet    # 현물 1시간봉 캔들
  │   ├── borrow/ {SYMBOL}.parquet       # 마진 차입 이자율
  │   └── manifest.json
  ├── state/                             # 실거래/섀도우 런타임 영속 상태
  │   ├── live_fills/
  │   ├── live_execution_quality/
  │   ├── live_microstructure/
  │   ├── live_portfolio_state/
  │   ├── live_orderbook/
  │   └── live_tax_ledger/
  └── manifest.json                      # 전체 데이터셋 SHA256 무결성 매니페스트
```

---

## 8. 테스트 및 품질 검증 기준 (Testing Expectations)
- **Continuity Test:** 타임스탬프 간격(1m, 5m, 1h 등)의 누락 구간 및 중복 검증.
- **Integrity Test:** 저장된 Parquet 파일의 SHA256 해시가 `manifest.json` 지문과 100% 일치하는지 확인.
- **Deterministic Isolation:** 단위 테스트 시 외부 Binance/Vision 엔드포인트를 직접 호출하지 않고 네트워크 계층을 Mocking하여 결정론적 테스트 수행.
- **Boundary-Only Mocking:** 원시 바이트 수신 및 디코딩 경계만 Mocking하고, 내부 파싱/정규화/무결성 로직은 실제 구현 코드를 직접 구동하여 검증.
