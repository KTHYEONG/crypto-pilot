---
title: MHS Architecture - 01. Data Panel & Universe Selection
domain: research-mhs
type: architecture
status: active
priority: critical
ai_read_policy: always
related_paths:
  - src/mhs/panel.py
  - src/mhs/marks.py
  - src/mhs/pipeline/stages/panel.py
  - src/mhs/pipeline/stages/selection.py
  - src/mhs/params.py
change_triggers:
  - src/mhs/panel.py
  - src/mhs/marks.py
  - src/mhs/pipeline/stages/selection.py
last_verified: 2026-09-03
---

# 01. 데이터 패널 구축 및 Point-In-Time (PIT) 유니버스 선정

## 1. 개요 (Overview)
백테스트 및 실시간 신호 생성에서 미래 데이터를 사전에 인지하고 종목을 고르는 생존 편향(Survivorship bias)과 미래 참조 편향(Look-ahead bias)은 전략의 실전 성과를 붕괴시키는 가장 치명적인 요인입니다.

MHS는 매 1시간 매매 결정 시각(Point-In-Time, PIT)마다 과거에만 관측 가능했던 데이터만을 사용하여 3단계 필터링을 통해 실행 유니버스를 동적으로 확정합니다.

---

## 2. 1시간 패널 데이터 인제스천 및 무결성 검증 (Panel Ingestion)

패널 데이터 적재 및 관리는 [`src/mhs/pipeline/stages/panel.py`](file:///home/kth/crypto-pilot/src/mhs/pipeline/stages/panel.py) 및 [`src/mhs/panel.py`](file:///home/kth/crypto-pilot/src/mhs/panel.py)의 [`build_mhs_data_panel`](file:///home/kth/crypto-pilot/src/mhs/panel.py)에 의해 수행됩니다.

### 1) 주요 적재 데이터셋
- **Futures OHLCV (1h)**: Close, Open, High, Low, Volume, Quote Volume, Taker Buy Quote Volume.
- **Funding Rates (8h)**: 선물 펀딩비 결제 시계열 (각 1시간봉에 Causal Forward-Fill).
- **Mark Price Klines (1h)**: 마크 가격 시계열 (MTM 평가 및 청산 판정용).

### 2) 시스템 자원 제약 (RAM Budget Guard)
대규모 심볼의 다년간 시계열을 메모리에 로드할 때 OOM(Out of Memory)으로 인한 비정상 종료를 방지하기 위해 시스템 메모리를 사전 점검합니다:
- `RAM_BUDGET_FRACTION = 0.85` (시스템 가용 메모리의 최대 85%만 소비 허용)
- `RAM_RESERVE_FLOOR_BYTES = 256MB` (최소 여유 버퍼 보장)
- 예산 초과 시 안전하게 조기 종료(Short-circuit)하여 시스템 패닉을 차단합니다.

---

## 3. 3단계 Point-In-Time (PIT) 심볼 선정 파이프라인

```text
전체 바이낸스 선물 심볼 (Perpetual Futures)
       │
       ▼ [1단계] 데이터 결손 & 상장 이력 필터 (Source Gap Guard)
결손 심볼(SLP, CTK 등) 제거 & 최소 720시간 데이터 보유 자산 선별
       │
       ▼ [2단계] 유동성 반분 필터 (Liquid-Half Eligibility)
최근 30일(720시간) 거래대금 중앙값(Median) 이상인 유동성 상위 50% 자격 부여
       │
       ▼ [3단계] 최종 실행 로스터 & 히스테리시스 (PIT Top-30 Roster + Schmitt-Trigger)
거래대금 상위 30개 종목 선정 (진입: 30위 이내, 탈락: 60위 밖으로 밀려날 때만)
```

### 1단계: 소스 갭 가드 (Source Gap Guard)
- **장기 결손 심볼 배제**: 바이낸스 API/Vision 아카이브 상에서 4시간 이상의 비정상 데이터 단절이나 갭이 실측된 심볼(`SLPUSDT`, `CTKUSDT`, `LITUSDT` 등 [`MHS_SOURCE_GAP_EXCLUDED_SYMBOLS`](file:///home/kth/crypto-pilot/src/mhs/params.py))은 초기 단계에서 완전히 제외됩니다.
- **신규 상장 자산 보호**: 각 결정 시점 기준으로 최소 720시간(30일)의 거래 이력이 누적되지 않은 신규 상장 심볼은 유니버스 진입 자격을 얻지 못합니다.

### 2단계: 유동성 반분 필터 (Liquid-Half Eligibility)
- **동적 유동성 임계값 산출**:
  - 매 결정 시각 $t$에서 각 심볼의 최근 720시간(30일) 누적 거래대금(Quote Volume)을 계산합니다.
  - 전체 유효 심볼들의 720시간 거래대금의 **단면 중앙값(Cross-Sectional Median)**을 계산합니다.
- **상위 50% 자격 부여 (`eligible_mask`)**:
  - 거래대금이 중앙값 이상인 상위 50%의 코인만 알파 신호 계산 및 랭킹 산출 대상(Eligible)으로 지정합니다.
  - 비유동성 잡코인의 비정상 가격 왜곡이나 슬리피지 폭탄을 원천 차단합니다.

### 3단계: 최종 실행 로스터 30개 선별 및 Schmitt-Trigger 히스테리시스
- **기본 로스터 크기**: 거래대금 기준 상위 30개 심볼 ([`execution_universe_size = 30`](file:///home/kth/crypto-pilot/src/mhs/params.py))을 실제 체결 리플레이 대상 유니버스로 채택합니다.
- **Schmitt-Trigger 히스테리시스 (2.0x Exit Multiplier)**:
  - 30위 경계 부근에서 심볼이 29위 $\leftrightarrow$ 31위를 잦게 왕복하며 불필요한 진입/퇴출 주문을 쏟아내는 진동 매매(Churning)를 막기 위해 이중 임계값을 적용합니다:
    - **신규 진입**: 거래대금 랭킹 **30위 이내**에 들어올 때만 신규 편입.
    - **퇴출 (탈락)**: 랭킹이 **60위 (`30 * 2.0`) 밖**으로 완전히 밀려날 때만 로스터에서 제외.
  - 이 메커니즘을 통해 포트폴리오 회전율(Turnover)과 불필요한 거래 수수료를 30% 이상 절감합니다.

---

## 4. Historical Mark Price 캐시 및 Causal Forward-Fill

마크 가격(Mark Price)은 신호 생성이나 유니버스 랭킹에는 일절 영향을 주지 않으며, 오직 **포트폴리오 평가(MTM), 미실현 손익 산출, 펀딩비 결제**에만 사용됩니다.

- **Causal Forward-Fill**:
  - 1시간 단위 마크 캔들(`markPriceKlines/1h`)을 사용하며, $t$ 시점의 마크 가격은 $t-1$봉의 Close 가격을 인과적으로 Forward-Fill하여 참조합니다.
- **Fail-Closed Gap 방어**:
  - 마크 가격 시계열에 누락이나 결손이 발견될 경우, 임의의 수치로 보간(Interpolation)하지 않고 [`DataIntegrityError`](file:///home/kth/crypto-pilot/src/common/errors.py)를 발생시켜 시스템을 Fail-Closed 상태로 보호합니다.
- **Mark Source 계약**:
  - `src/mhs/marks.py`의 [`_get_symbol_mark_frame`](file:///home/kth/crypto-pilot/src/mhs/marks.py)을 통해 단일화된 캐시 레이어에서 제공됩니다.

---

## 5. 핵심 코드 진입점 (Key Code Reference)

| 역할 | 소스 파일 | 핵심 함수 및 상수 |
|---|---|---|
| 패널 데이터 적재 | [src/mhs/pipeline/stages/panel.py](file:///home/kth/crypto-pilot/src/mhs/pipeline/stages/panel.py) | `load_panel` |
| 패널 빌더 | [src/mhs/panel.py](file:///home/kth/crypto-pilot/src/mhs/panel.py) | `build_mhs_data_panel` |
| 유니버스 선정 | [src/mhs/pipeline/stages/selection.py](file:///home/kth/crypto-pilot/src/mhs/pipeline/stages/selection.py) | `select_horizons` |
| 마크 가격 캐시 | [src/mhs/marks.py](file:///home/kth/crypto-pilot/src/mhs/marks.py) | `_get_symbol_mark_frame` |
| 결손 제외 심볼 목록 | [src/mhs/params.py](file:///home/kth/crypto-pilot/src/mhs/params.py) | `MHS_SOURCE_GAP_EXCLUDED_SYMBOLS` |
