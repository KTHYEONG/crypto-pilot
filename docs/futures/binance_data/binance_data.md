# Binance Futures 데이터 수집 스펙 (v1.3)

**작성일**: 2026-05-18 | **최종 검증**: 2026-05-18  
**상태**: 검증 완료 및 아키텍처 확정  
**목표**: 데이터 수집 레이어의 가용 범위 정의 및 재현성 검증 보장  

이 문서는 Binance Futures 유니버스 구축에 사용되는 가용 데이터 소스, 수집 전략, 그리고 품질 스펙을 정의합니다.

---

## 데이터 수집 레이어

> 2026-05-18 실 API/Vision 탐색 테스트로 확인된 내용만 기재.  
> 테스트 파일: `tests/integrations/test_data_availability.py`

### 소스별 가용 데이터 및 범위

| 데이터 종류 | 소스 | 가용 시작일 | 비고 |
|---|---|---|---|
| 심볼 상장일 | FAPI `exchangeInfo.onboardDate` | 2019-09-25~ | 645/731 커버, 누락분은 Vision 첫파일로 보완 |
| 현재 심볼 전체 메타 | FAPI `exchangeInfo` | 현재 | tick_size·step_size·status 포함 |
| SETTLING 심볼 (상폐진행중) | FAPI `exchangeInfo` | 현재 | 112개 확인, deliveryDate 신뢰 금지 |
| 전체 심볼 목록 (상폐 포함) | Vision S3 XML 목록 | — | **857개**, LUNA·DEFI·YFII 포함 |
| OHLCV (2019-09~2019-12) | CCXT `fetch_ohlcv` | BTC: 2019-09-08 | **Vision에 없는 구간**, CCXT 전용 |
| OHLCV (2020-01~) | Vision `daily/klines/` | 2020-01-29 | daily/monthly 모두 존재 |
| 상폐 심볼 OHLCV | Vision `daily/klines/` | 심볼별 상이 | LUNA(2021~2022) 등 kline 존재 확인 |
| 펀딩비 (전범위) | FAPI `/fapi/v1/fundingRate` | BTC: 2019-09-10 | 완전한 역사, 페이지네이션으로 전량 수집 |
| 펀딩비 (아카이브) | Vision `monthly/fundingRate/` | 2020-01 | daily 경로 없음, **monthly만 존재** |
| OI / LSR (최근) | FAPI `openInterestHist` | 최근 500봉 (~3주) | 딥히스토리 미지원 |
| OI / LSR (딥히스토리) | Vision `daily/metrics/` | **2020-09-01** | sum_open_interest, LSR 컬럼 확인, BinanceVisionDownloader 확장 필요 |
| 호가창 spread/depth | Vision `daily/bookDepth/` | 2020~ (추정) | **0.5 MB/일**, 12 depth level, ~2,600 스냅샷/일 (불규칙 간격) |
| premiumIndex / markPrice | Vision `daily/premiumIndexKlines/` | — | 가용 확인 |
| bookTicker | Vision `daily/bookTicker/` | — | 가용 확인 |

---

### OHLCV 수집 Hybrid 전략

Vision 아카이브가 2019년 데이터를 보존하지 않으므로 구간별 소스를 분리:

```
date < 2020-01-01  →  CCXT fetch_ohlcv  (API, 약 4개월 구간)
date ≥ 2020-01-01  →  Vision daily/klines  (아카이브 우선)
                       → CCXT fallback (Vision 파일 없을 경우)
```

---

### Half-spread 계산 전략 (cost_model.py)

Roll(1984) 모델 검증 결과: NaN 발생률 46.8%, 실제 대비 5.8배 과대추정 → **2020+ 구간에서 Roll spread 사용 금지**.

```
date ≥ 2020-01-01  →  Vision bookDepth 집계
                       일별 스냅샷 resample → 시간대별 평균 depth/spread
                       half_spread = median(ask_price - mid_price) over bar window

date < 2020-01-01  →  Roll spread fallback
                       half_spread ≈ 0.5 × (High - Low) / Close  (Corwin-Schultz 변형)
                       또는 고정 추정치 (BTC: 0.02%, 소형 알트: 0.05%)
```

---

### BinanceVisionDownloader 확장 필요 항목

기존 구현(`src/core/utils/binance_vision.py`)은 `daily/metrics`만 지원. Ledger 적재를 위해 추가 필요:

| 추가 메서드 | Vision 경로 | 용도 | 파이프라인 상태 |
|---|---|---|---|
| `fetch_klines_archive` | `daily/klines/{sym}/{tf}/` | OHLCV 수집 (상폐 심볼 포함) | 활성 |
| `fetch_funding_monthly` | `monthly/fundingRate/{sym}/` | 펀딩비 아카이브 | 활성 |
| `fetch_bookdepth_daily` | `daily/bookDepth/{sym}/` | spread/depth 집계용 | 활성 |
| `fetch_premiumindex_daily` | `daily/premiumIndexKlines/{sym}/` | basis 계산용 | 활성 (단, 파이프라인 연동 비활성) |
| `fetch_metrics_daily` | `daily/metrics/{sym}/` | OI / LSR 수집 | 활성 (단, 파이프라인 연동 비활성) |
| `list_all_symbols` | S3 XML 목록 (`?list-type=2`) | 전체 심볼 발견 (상폐 포함) | 활성 |
| `verify_checksum` | `{path}.CHECKSUM` (SHA256 동봉) | **다운로드마다 SHA256 검증 필수** | 활성 |

#### 비활성화된 유니버스 필터 (파이프라인 연동 로직 미비)

`fetch_premiumindex_daily`와 `fetch_metrics_daily` 다운로더는 구현이 완료되어 활성 상태이나, 실제 파이프라인(`pipeline.py`) 및 `filters.py` 내 연동 로직 미비로 인해 아래 두 Stage 5 필터가 현재 비활성 상태로 유지 중이다. 향후 파이프라인 및 필터 연동이 완료되고 Ledger에 실측값이 공급되면 필터를 복구해야 한다.

| 제거된 필터 | 감지 대상 리스크 | 현재 대안 방어막 | 공백 수준 |
|---|---|---|---|
| `basis_z_score <= 2.5` | Mark-Index basis 이상치 (선물 프리미엄 폭발, 시세 조작 의심) | `vol_30d` 상한(400%)으로 간접 커버; `funding_zscore` 이상치가 basis spike와 동반하는 경우 부분 커버; `listing_age_days >= 90`으로 신규 상장 초기 basis 이상 방어 | 중간 — 파이프라인 및 필터 연동 완료 시 복구 |
| `oi_usdt_median / adv <= 12.0` | OI/ADV 비율 과레버리지·crowding 감지 | `vol_30d` 상한으로 고레버리지 청산 vol spike 간접 감지; `adv_usdt_median >= 25M`으로 대량 청산 충격 완화; `funding_zscore`로 롱/숏 편중 간접 감지 | 낮음 — 파이프라인 및 필터 연동 완료 시 복구 |

---

### Data Manifest (v1.3 — 재현성 잠금)

```python
# data/futures/data_manifest.parquet
@dataclass
class ManifestRow:
    source: str           # "vision" | "ccxt" | "fapi"
    symbol: str
    tf: str
    period: str           # YYYY-MM-DD
    url: str
    sha256: str           # Vision: .CHECKSUM 파일 / CCXT: 자체 계산
    bytes: int
    fetched_at_utc: str
    is_final: bool        # monthly archive=True / 당월 daily=False (provisional)
```

- **Vision**: 각 `.zip`에 동봉된 `.CHECKSUM` SHA256 검증 → 불일치 시 재시도 후 실패 기록
- **CCXT(2019 구간)**: **1회만 수집 후 로컬 parquet 동결**, 자체 SHA256 고정. 이후 재호출 금지.
- **`UniverseSnapshot.data_manifest_hash`**: build에 투입된 (symbol, period, sha256) 집합의 해시. `config_hash` + `data_manifest_hash` = 완전한 재현성 지문.
- provisional(`is_final=False`) 행은 `knowledge_date` lag과 결합해 당월 미확정 데이터 식별.
