# Binance Futures 유니버스 아키텍처 (v1.3)

**작성일**: 2026-05-18 | **최종 검증**: 2026-05-18  
**상태**: 아키텍처 확정 — 데이터 가용성 검증 완료, Phase 1 구현 준비됨  
**목표**: Point-in-time PIT 준수, 생존편향/상폐편향 제거, 거래가능 자산만 선별

---

## 목차

1. [핵심 설계 원칙](#핵심-설계-원칙)
2. [2개 프로세스 분리](#2개-프로세스-분리)
3. [디렉토리 구조](#디렉토리-구조)
4. [Ledger - 생존편향 해소](#ledger---생존편향-해소)
5. [데이터 수집 레이어 (Binance Data)](binance_data.md)
6. [7단계 Funnel](#7단계-funnel)
7. [Stage 6 - 분산 전략](#stage-6---분산-전략)
8. [데이터 계약](#데이터-계약)
9. [요구사항 충족 매핑](#요구사항-충족-매핑)
10. [미확정 사항](#미확정-사항)

---

## 핵심 설계 원칙

기존 코드의 치명적 결함: **스크리닝 시점에 라이브 API(`fetch_tickers`)를 호출** → PIT 원천적 불가능.

**재설계의 1원칙:**

```
유니버스 빌드는 외부 세계와 단절된 순수 함수다.
build_universe(as_of, tf, cfg) -> UniverseSnapshot
  - as_of 이후 데이터를 절대 읽지 않음
  - 동일 입력 → 동일 출력 (결정론적)
```

이를 통해:
- **PIT 순수성**: 백테스트(과거 `as_of`)와 라이브(`as_of=now`)가 동일 코드 경로 → train/live skew 제로
- **생존편향 해소**: Ledger에 상폐 코인의 과거 행 보존
- **재현성**: 동일 ledger + 동일 config → 동일 결과

---

## 2개 프로세스 분리

유일하게 거래소 API를 접하는 곳과 순수 스크리닝 로직을 물리적으로 분리:

```
┌─────────────────────────────────────────────────────┐
│ [프로세스 A: Ledger 적재]                            │
│ • 온라인 · append-only · 스케줄러로 주기 실행        │
│ • exchangeInfo + klines + funding + OI              │
│ • "날짜 태그" (knowledge_date)하여 ledger에 기록     │
│ • 유일 API 접점 ⚠️                                  │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
        [universe_ledger.parquet]
        (상폐 코인 포함 전체 역사)
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│ [프로세스 B: Universe 빌드]                          │
│ • 오프라인 · 순수함수 · 거래소 API 미접근            │
│ • ledger 읽기만 가능 (as_of 이전만)                │
│ • build_universe(as_of) → 7단계 funnel             │
│ • 백테스트/라이브 동일 호출                          │
└──────────────────────────────────────────────────────┘
                   │
                   ▼
        [UniverseSnapshot]
        (dated, reproducible)
```

**핵심 이점**:
- CRITICAL-1(룩어헤드) 구조적 해결
- 스냅샷 영속화 → 백테스트 재현성
- 레이턴시 감소 → 라이브는 오프라인 계산만 수행

---

## 디렉토리 구조

`src/domain/futures/universe/` — 각 파일 단일 책임, ≤500 줄 (CLAUDE.md §7):

```
src/domain/futures/universe/
├── __init__.py              # 공개 API
│                           # build_universe(as_of, tf, cfg) -> UniverseSnapshot
│                           # load_universe_snapshot(as_of, tf)
│                           # update_ledger() [프로세스 A]
│
├── config.py               # UniverseConfig (frozen dataclass)
│                          # 모든 임계값 중앙화, config_hash로 재현성
│
├── contracts.py            # 스키마 (dataclass)
│                          # SymbolMeta, LedgerRow, FilterReport,
│                          # UniverseSnapshot, RejectCode(Enum)
│
├── ledger.py               # Ledger 빌드/갱신/질의
│                          # - as_of 시점 수익 심볼 쿼리
│                          # - append-only 로직
│                          # - 상장/상폐일 복원
│
├── exchange_meta.py        # Binance exchangeInfo 수집
│                          # - 계약 메타 (tick_size, step_size, etc)
│                          # - 상장/상폐일 파싱
│                          # - 거래소 status 추적
│
├── data_quality.py         # Stage 2: 데이터 품질 검증
│                          # - bar 커버리지 / gap 검사
│                          # - NaN / Inf / frozen bar 감지
│                          # - 최근 60d 연속성 (last_60d_coverage ≥ 95%)
│                          # - 0 volume bar 수 (n_zero_volume_bars_60d ≤ 1)
│                          # - 타임스탬프 정합성
│
├── structure.py            # Stage 1: 자산 구조 필터
│                          # - PERP swap 확인
│                          # - USDT margin/quote 확인
│                          # - 레버리지토큰 배제 (UP/DOWN/BULL/BEAR)
│                          # - contract multiplier 검증
│
├── liquidity.py            # Stage 3: 유동성 & 체결성
│                          # - ADV (30d 중앙값, robust to spikes)
│                          # - Amihud illiquidity score
│                          # - 호가 깊이 (if available)
│                          # - 목표 클립 체결성 추정
│
├── cost_model.py           # Stage 4: 거래 비용 모델
│                          # - Maker/Taker 수수료 × 2 (round-trip)
│                          # - Half-spread × 2
│                          # - √ impact slippage: k·σ·√(Q/ADV)
│                          # - 펀딩 캐리 (부호 포함)
│                          # - Cost-to-edge 비율
│
├── risk_events.py          # Stage 5: 리스크·조작·이벤트
│                          # - 상장 연령 (≥90d)
│                          # - 상폐/정산 스케줄 임박 여부
│                          # - 펌프/wash-trading 휴리스틱
│                          # - 정지 이력 / 상태 이상
│                          # - 펀딩 이상치 (z-blowout)
│                          # - 변동성 밴드 게이트 (vol_30d_band)
│                          # - mark-index basis 이상치 (premiumIndexKlines)
│                          # - manual_event_ledger: risk_event_override 인터페이스
│
├── selection.py            # Stage 6: 선택 & 분산
│                          # - 생존자 상관 클러스터링
│                          # - ENB (effective # of bets) 극대화
│                          # - 히스테리시스 (entry/exit rank band)
│                          # - 앵커 강제 포함 (BTC/ETH)
│                          # - 시장 바스켓 기준축
│
├── pipeline.py             # 오케스트레이션
│                          # - Stage 0~6 순차 실행
│                          # - FilterReport 감사 추적
│                          # - UniverseSnapshot 생성
│
└── persistence.py          # 스냅샷 & ledger I/O
                           # - Parquet / JSON 저장
                           # - 스키마 버저닝
                           # - 타임스탬프 태깅
```

---

## Ledger - 생존편향 해소

### 구조

`universe_ledger.parquet` — 일별 append-only 패널:

```python
@dataclass
class LedgerRow:
    symbol: str              # e.g., "BTC/USDT"
    date: str                # YYYY-MM-DD
    knowledge_date: str      # 해당 데이터가 실제로 "알려진" 날짜
                            # (일반적으로 date + 1-2일, same-day 룩어헤드 방지)
    
    # 상장/상폐 상태
    is_listed: bool
    is_trading: bool
    status: str              # "TRADING", "HALT", "DELISTED", "PRE_TRADING"
    first_kline_date: str    # de-facto 상장일 (복원됨)
    delist_date: str | None  # de-facto 상폐일 (복원됨)
    delist_announcement: str | None
    
    # 유동성 & 활동성
    adv_usdt_median: float   # 추세 30d quote-volume 중앙값 (robust)
    adv_usdt_mean: float     # (참고용)
    has_kline: bool          # 4h/1h kline 보유
    has_funding: bool        # 펀딩비 데이터 보유
    n_bar_gaps: int          # 장시간 gap 개수
    max_gap_bars: int        # 최대 gap 크기
    frozen_bars: int         # 연속 동일 종가 봉 수
    
    # 데이터 품질 (Stage 2에서 쓰임)
    last_60d_coverage: float      # 최근 60d bar 커버리지 비율 (0~1)
    n_zero_volume_bars_60d: int   # 최근 60d 0-volume bar 수
    
    # 리스크 신호 (Stage 5에서 쓰임)
    funding_rate_8h: float        # 최근 8h 펀딩비 (원천 주기, 연환산 금지)
    open_interest_usdt: float
    oi_usdt_median: float         # 30d OI 중앙값 (깊이 proxy)
    oi_change_30d: float          # OI 변화율 — crowding/squeeze 신호
    listing_age_days: int         # 첫 kline으로부터 경과 일수
    vol_30d: float                # 30d 수익률 변동성 (Stage 5 vol_band 게이트)
    basis_z_score: float | None   # mark-index basis z-score (premiumIndexKlines, 게이트)
    basis_annualized_mean: float | None  # 구조적 carry 수준 (descriptive 피처)
    basis_vol: float | None       # peg 불안정성 (descriptive 피처)
    risk_event_override: str | None  # 수동 이벤트 태그 (ManualEventRow 참조)
    
    # 메타
    updated_at_utc: str      # ledger 행 생성 시각
```

### 상장/상폐일 복원 전략 (검증 확정)

**상장일 복원**:
1. `exchangeInfo.onboardDate` 필드 → **731개 중 645개 커버** (현재 거래 심볼 대부분)
2. onboardDate 없는 심볼 → Vision S3 klines 디렉토리의 첫 파일 날짜로 대체

**상폐일 복원**:
1. `exchangeInfo.status == "SETTLING"` → **즉시 is_trading=False 처리**
2. `deliveryDate` 필드는 **신뢰 금지** — 실제 거래 종료보다 수개월 앞선 결정일 기록 (검증됨: SXPUSDT deliveryDate=2025-12-05, Vision klines 2026-05-16까지 존재)
3. **실제 상폐일 = Vision klines 마지막 파일 날짜** (유일한 신뢰 소스)
4. Vision S3 klines 디렉토리에서 **857개 심볼** 발견 → 상폐 심볼(LUNA, DEFI, YFII) 포함 확인

**미래 (forward)**:
- 프로세스 A가 매일 `exchangeInfo` 스냅샷 적재 → SETTLING 전환 시점 정확히 포착
- `ledger_confidence`: `"official"` (onboardDate 있음) vs `"reconstructed"` (Vision 최초파일 기반)

### Availability Lag

핵심: `date` ≠ `knowledge_date`

```
실제 거래 발생: T일 (예: 5월 18일)
  ↓
데이터 수집: T+1 (5월 19일, daily settlement 후)
  ↓
knowledge_date = T+1 으로 태그
  ↓
Stage 스크리닝: knowledge_date ≤ as_of 인 행만 사용
  → same-day 룩어헤드 원천 차단
```

이는 "T일 데이터가 T일에 실제로 사용 가능했는가?"라는 현실적 질문에 답합니다.

---

## 데이터 수집 레이어

> [!NOTE]
> Binance Futures의 가용 데이터 소스, 수집 범위, 하이브리드 수집 전략 및 영속화 스키마(Data Manifest) 등 데이터 수집과 관련된 세부 스펙은 별도 문서인 [binance_data.md](binance_data.md)에서 자세히 확인할 수 있습니다.

해당 파트는 유니버스 구축 시 사용되는 CCXT API, Binance Vision 아카이브 하이브리드 전략, Half-spread 실측 계산 및 데이터 재현성 잠금(Manifest) 등을 다룹니다.

---

## 7단계 Funnel

`pipeline.py`가 다음 순서로 실행. 각 stage는 통과/탈락 + 메트릭을 `FilterReport`에 기록 → 완전한 감사 추적:

### Stage 0: 적격 모집단 (Eligibility Universe)

```
입력: as_of 날짜
출력: is_listed & is_trading @ as_of 인 전 심볼 (상폐예정도 포함, 미래상장은 제외)
```

ledger에서 `knowledge_date ≤ as_of and is_trading == True` 필터링.

---

### Stage 1: 자산 구조 (structure.py)

```
검사 항목                      설명
─────────────────────────────────────────────────────────────────
PERP swap 확인                 Quarterly/monthly 선물 제외
USDT margin/quote             USDT-margined 또는 USDT-quoted만
status == TRADING 만 허용      HALT / DELISTED / SETTLING 모두 즉시 배제
                               ※ SETTLING: 검증됨 — deliveryDate 이후에도
                                  수개월간 잔존, 거래불가 상태이므로 즉시 제외
레버리지 토큰 배제             UP·DOWN·BULL·BEAR 패턴 매칭
Contract multiplier           정상 범위 내 (1 또는 사전정의값)
Normalized symbol             1000XXX 같은 호환성 정규화
```

**목적**: 데이터 이상한 자산, 거래 불가능한 자산 즉시 제거.  
**주의**: `deliveryDate` 필드로 상폐 예측 금지 — SETTLING 상태 확인이 유일한 신뢰 방법.

---

### Stage 2: 데이터 품질 (data_quality.py)

```
검사 항목                      기준                비고
─────────────────────────────────────────────────────────────────
커버리지                       IS window 80% 이상   IS = 21개월 → @4h ≥ 3,834봉 중 3,067봉
단일 gap                       ≤ G봉 (예: 200)    장기 중단 배제
연속 동일 종가                  ≤ S봉 (예: 10)     frozen bar 감지
최근 60d 연속성                 last_60d_coverage  IS 평균 커버와 독립 — 최신 데이터 공백
                               ≥ 95%              탐지 (구간 평균을 통과해도 최근 공백 발생 가능)
0 volume bar                   ≤ 1봉 / 60일       체결 없는 봉 검출 (frozen bar는 가격만 보므로
                                                   거래량 0을 독립 신호로 추가 검사)
NaN / Inf                      없음                수치 정합성
funding 컬럼                   보유                FAPI로 2019-09~현재 전량 수집 가능
OI 컬럼 (2020-09 이후)         보유                Vision metrics에서 수집 (BinanceVisionDownloader)
OI 컬럼 (2020-09 이전)         결측 허용           FAPI 딥히스토리 미지원, NaN 처리 필요
타임스탐프                      단조증가, UTC       시간 역순 방지
```

**핵심**: 15개월 IS 윈도우에서 **커버리지 ≥ 80%** 의무화 → 신규상장 저데이터 심볼 배제.

기존 코드 `min_is_bars=500`(≈83일)은 약함. **v1.3 확정 기준**: `@4h ≥ 3,067봉` (IS 3,834봉 × 80%).

**Walk-Forward 평가 방식 (v1.3 확정)**:
```
mode            = rolling           # anchored 아님 — 크립토 concept drift 대응
is_months       = 21
oos_months      = 3
step_months     = 3
purge           = label_horizon     # IS/OOS 경계 누수 차단 (≥ triple-barrier vertical)
embargo_days    = 7
```
데이터 2020-01~현재 ≈ 77개월 기준 OOS fold **약 16~17개** (≈ 4년 연속 OOS equity curve).  
단일 IS/OOS split 금지 — OOS 1개는 통계적 신뢰성 불가(Deflated Sharpe Ratio 계산 불가).

`last_60d_coverage`와 `n_zero_volume_bars_60d`는 IS 평균과 독립 검사. IS 80% 통과 후에도 최근 60d 공백이 있으면 배제.

---

### Stage 3: 유동성 & 체결성 (liquidity.py)

```
검사 항목                      기준                    계산
─────────────────────────────────────────────────────────────────
ADV (Average Daily Volume)    ≥ 25M USDT            추세 30d 중앙값
                                                     (mean이 아님, spike robust)

Amihud illiquidity            상한 설정              |ret| / (volume·price)
                                                     tail 180봉

호가 깊이 (if available)       depth@Kbps ≥ X%       호가창 데이터 있을 시
                              of ADV

체결성 추정                    screening_clip / ADV  screening_clip_usdt로 impact 추정
                               ≤ 0.5% ADV            (자본 티어별 설정값 아래 참조)
```

**중앙값 사용 이유**: 한 번의 폭증(listing pump, 공지)이 ADV 기준값을 왜곡하지 않음.

**자본 티어별 Clip 설계 (v1.3 확정)**:
```python
# 스크리닝용 — universe gate에서 impact 계산 기준 (외생 고정, equity 비의존)
# 보수적 방향으로 설정 (impact 과대추정이 안전)
SCREENING_CLIP_BY_TIER = {
    "seed":   1_000,   # 초기 실전 시작 / 연구용 기본
    "small":  5_000,   # 소규모 실전
    "mid":   10_000,   # 자본 성장 후 1단계
    "large": 25_000,   # 자본 성장 후 2단계
    "xlarge":50_000,   # 개인 규모 상한
}

# capacity 평가 — 유니버스가 지탱 가능한 AUM ceiling 분석용
CAPACITY_CLIP_LIST = [50_000, 100_000]  # 성장 가능성 상한 시뮬레이션

# 라이브 실거래 — equity 비례, 유니버스 gate와 분리
live_clip_usdt = equity * position_size_pct  # (작게 시작, 유니버스 gate와 무관)
```
**원칙**: `screening_clip_usdt`는 equity 순환 방지를 위해 **반드시 외생 고정**, tier는 분기 재배포 시 수동 업그레이드.  
`capacity_clip_list`로 스냅샷마다 capacity ceiling 보고 → 자본이 ceiling에 접근 시 경고.

---

### Stage 4: 거래 비용 (cost_model.py)

**Round-trip Execution Cost 산식 (v1.3: funding 분리)**:

```
# ── 순수 실행 마찰 (라운드트립당 1회, turnover에 비례) ──
execution_cost_bps = 2·fee_bps
                   + 2·half_spread_bps
                   + k·σ·√(screening_clip_usdt / adv_usdt)·1e4
                   + tick_cost_bps        (반올림 마찰: tick_size/price × 0.5 × 1e4)

# ── Funding carry: 절대 합산 금지, 별도 피처로 분리 ──
funding_carry_8h   = per-8h 원천 주기로 저장 (Binance 정산 주기 일치)
                     부호 포함 — 숏 + funding > 0 → 수익
```

**각 항목 (execution_cost_bps)**:
- `2·fee`: Maker + Taker round-trip (Binance Futures 기본 ~0.04%×2)
- `2·half_spread`: **소스 분기 (검증 확정)**
  - `as_of ≥ 2020-01-01` → **Vision bookDepth 집계** (0.5 MB/일, ~2,600 스냅샷)
    - `half_spread = median(best_ask - mid) over 4h window`
  - `as_of < 2020-01-01` → **Corwin-Schultz 변형 fallback**
    - `half_spread ≈ (High - Low) / (2 × Close)` (Roll 모델 사용 금지)
    - Roll(1984) 검증 결과: NaN 46.8%, 실측 대비 5.8배 과대추정 → 폐기
- **√ impact model**: σ = 4h 수익률 변동성, k ≈ 0.1~0.2, clip = `screening_clip_usdt`
- `tick_cost_bps`: `(tick_size / price) × 0.5 × 1e4` — bookDepth half_spread가 이미 tick granularity 반영(2020+)이므로 **반올림 마찰만** 포착, 이중 가산 금지.

**Funding carry 분리 근거**: funding은 (a) holding time 비례 연속 수익/비용, (b) signed alpha 성분. execution_cost는 (a) 라운드트립당 1회, (b) 순수 마찰. 차원이 달라 합산 시 cost gate가 carry 우호 자산을 오배제하고, 복리 carry 효과가 비용으로 소거됨.

**게이트**: `execution_cost_bps ≤ max_cost_bps` (예: 50bps) — funding 미포함  
**스냅샷 피처**: `execution_cost_bps`, `funding_carry_8h` — 둘 다 운반, 절대 합산 경로 없음  
**알파/optimizer 레이어**: `funding_carry_8h`는 carry-harvesting sleeve 평가 입력으로만 사용

---

### Stage 5: 리스크·조작·이벤트 (risk_events.py)

```
검사 항목                      기준 / 휴리스틱          목적
─────────────────────────────────────────────────────────────────
상장 연령                       ≥ 90일                가격발견·언락·유동성 불안정 구간 배제
                                                     (v1.2: 30d→90d 상향 — 상장빔/락업 이슈)
상폐/정산 스케줄                snapshot 유효기간+    예정된 퇴출 방지
                               holding horizon 내
                               비예정 확인

펌프-덤프 검출                  return z-score > 3    극단값 + 반전 추적
                              (높음 → 낮음 within
                              수일)

비정상 거래 활동              volume/trade-count    abnormal activity heuristic
(Abnormal activity)           ratio inconsistent    (Amihud illiquidity가 부분 커버하나
                              with price-move       명시적 패턴 감지 보완)

거래 정지 이력                  status != TRADING     이력 상태 확인
                              어느 시점이든

펀딩비 이상치                   |funding_rate|       *단순 고펀딩은 통과*
(조작/스퀴즈)                  z-score > 2.5 OR      - 이상치(급등/급락)만 배제
                              부호 급반전 (< 1일)     - 구조적 고펀딩 → 피처로

OI/ADV 농도 비율               OI_usdt_median /      OI 집중도(crowding proxy) — 청산 리스크
                              adv_usdt              (과도한 레버리지 신호 아님, 개별 포지션 크기와 무관)

변동성 밴드                     vol_30d ∈              하한: ADV 통과 후에도 거래 소멸 코인
                               [min_vol, max_vol]     상한: 구조적 펌프앤덤프 배제
                               (config 파라미터화)     (min_vol / max_vol 예: 0.3%~25% / day)

mark-index basis 이상치        |basis_z| ≤ 2.5       마킹 조작·청산 연쇄 신호 (게이트)
(premiumIndexKlines)           AND 30분 내 부호        (펀딩 이상치와 독립 — basis는
                               급반전 없음             price-discovery 왜곡을 포착)
                               ── 아래는 게이트 아님, descriptive 피처로 스냅샷 운반 ──
                               basis_annualized_mean  구조적 carry 수준
                               basis_vol              peg 불안정성 → 체결 리스크
                               (예측 피처 금지 — alpha_factory 전담)

수동 리스크 이벤트              risk_event_override    token unlock·해킹·규제 이슈 등
(manual_event_ledger)          ≠ None → 즉시 배제     자동 수집 불가 데이터를 수동 태깅
                                                     인터페이스만 제공 (자동화 미포함)
```

**핵심 (피드백 반영)**:
- 펀딩비는 "**배제**가 아니라 **이상치만 배제**"
- 구조적 고펀딩(일관된 carry) 통과 → Stage 6에서 `funding_carry` 신호로 활용
- 이를 통해 선물 carry 알파(특히 숏북)를 놓치지 않음

**추가 (v1.2)**:
- 상장 연령 30d → **90d** 상향 (상장빔·락업 해소 최소 구간)
- `vol_30d_band`: 변동성 하한(죽은 코인)·상한(펌프앤덤프) 양방향 게이트
- `basis_anomaly`: `premiumIndexKlines` 기반 mark-index basis 이상치 — 펀딩 이상치와 독립 신호
- `risk_event_override`: 수동 이벤트 태그 인터페이스 (token unlock 등, 자동 수집 미포함)

**Manual event PIT 안전 규칙 (v1.3)**:
```python
@dataclass(frozen=True)
class ManualEventRow:
    symbol: str
    event_type: EventType     # SCHEDULED_UNLOCK | EXCHANGE_HALT |
                              # REGULATORY | SECURITY_INCIDENT
    event_date: str           # 실제 이벤트 발생일
    knowledge_date: str       # 필수 non-null — 정보가 공개적으로 알 수 있게 된 날
                              # SCHEDULED_UNLOCK: 스케줄 공시일 (PIT-safe ✓)
                              # SECURITY_INCIDENT: 거래소 disclosure 날짜
    severity: str
    action: str               # "exclude" | "flag"
    source_url: str
    recorded_at_utc: str      # 감사용 only (PIT 판정 미사용)
```
- **fail-closed**: `knowledge_date` null → build_universe가 해당 행 **무시** (추정 날짜 적용 금지)
- `knowledge_date ≤ as_of` 필터를 ledger와 **동일하게 적용** (룩어헤드 원천 차단)
- `SCHEDULED_UNLOCK`은 스케줄이 상장 시 공개되므로 PIT-safe — 적극 활용
- **Hindsight-selection 편향 정량화**: backtest를 `scheduled-only` vs `scheduled+discretionary` 두 버전으로 실행해 delta 보고 의무화

---

### Stage 6: 선택 (selection.py) — 멤버십 결정 전담

**v1.3 재설계**: Universe selection과 Portfolio selection 물리적 분리.

| | Universe (Stage 6) | Portfolio (optimizer) |
|---|---|---|
| **역할** | 어떤 자산이 *플레이어블 셋*인가 | 각 자산에 *얼마 배분*하는가 |
| **전략 의존성** | strategy-agnostic | 전략 목적함수에 의존 |
| **ENB** | 피처로 운반만 | 실제 ENB 타게팅 |
| **상관 클러스터링** | cluster_id 피처 운반 | 분산/HRP 결정 |

Stage 5 생존자 대상 실행:

#### 6.1 베타 reference 계산 (descriptive only)

```python
# reference index — 베타 계산용, 멤버 선발 기준 아님
market_basket = [
    ("BTC/USDT", 0.45), ("ETH/USDT", 0.25), ("SOL/USDT", 0.08), # ... cap-weighted
]
market_returns = weighted_sum(symbol_returns, weights)
# → beta_vs_market 피처로 스냅샷에 운반, 선발 제외 기준으로 사용 금지
```

#### 6.2 상관 클러스터링 (피처 운반 전용, 멤버 제거 금지)

```
Step 1: pairwise correlation 산출 (trailing 250 trading days, PIT-safe)
Step 2: distance = 1 - |corr| 변환
Step 3: hierarchical clustering (ward linkage)
Step 4: cluster_id 할당 → SymbolMeta 피처로 운반
  ※ 클러스터 기반 멤버 제거 금지 — optimizer가 cluster_id로 분산 결정
  ※ 예외: 동일 underlying 중복 instrument (1000X vs X 등) de-dup은 universe 권한
```

ENB / HRP / risk-parity → **optimizer 레이어 전담** (universe layer 밖).

#### 6.3 Tradeability Composite Rank (선발 기준)

```python
# strategy-agnostic tradeability 점수
tradeable_score = (
    w_liq   * normalize(adv_usdt_median)       # 유동성
  + w_cost  * normalize(1 / execution_cost_bps) # 비용 역수
  + w_qual  * normalize(last_60d_coverage)      # 데이터 품질
  + w_age   * normalize(listing_age_days)       # 안정성
)
# 상위 K_in 선발 (예: 15~25), hysteresis band로 churn 제어
```

#### 6.4 히스테리시스 (churn 제어)

```
진입 조건: tradeable_score rank ≤ K_in  (엄격, 예: top 20)
이탈 조건: rank > K_out                  (느슨, 예: top 35)
최소 dwell: 1분기
하드 게이트: Stage 1~5 fail → 즉시 퇴출 (hysteresis 무시)
```

#### 6.5 앵커 강제 포함

```python
if "BTC/USDT" not in final_list:
    final_list.insert(0, "BTC/USDT")  # role="anchor"
if "ETH/USDT" not in final_list:
    final_list.insert(1, "ETH/USDT")  # role="anchor"
# anchor role → HMM reference / 벤치마크 / 가중치 0 허용 (거래 멤버와 구분)
```

#### Stage 6 출력 (스냅샷 피처)

```python
# SymbolMeta 운반 피처 — optimizer 입력
adv_usdt, execution_cost_bps, funding_carry_8h,
beta_vs_market, cluster_id, tradeable_rank,
basis_annualized_mean, basis_vol,
oi_usdt_median,             # OI 절대값 (깊이)
oi_to_adv,                  # OI/ADV 비율 (crowding proxy)
oi_change_30d,              # OI 변화율 (포지션 유입/유출, squeeze 선행)
capacity_clip_usdt_list,    # capacity ceiling 분석용
data_manifest_hash          # 데이터 재현성 지문
```

---

## 데이터 계약

### UniverseSnapshot

```python
@dataclass(frozen=True, slots=True)
class UniverseSnapshot:
    # PIT & 재현성
    as_of: str                      # YYYY-MM-DD, 기준 시점
    tf: str                         # "4h", "1h"
    schema_version: int             # 스키마 버저닝
    config_hash: str                # UniverseConfig.hash → 재현성 지문
    data_manifest_hash: str         # 투입 데이터 (symbol, period, sha256) 집합의 해시
    
    # 시장 참고
    basket_ref: tuple[str, ...]     # 베타/마켓 reference 심볼
    basket_weights: tuple[float, ...] # 가중치
    
    # 선택된 자산
    selected: tuple[SymbolMeta, ...]  # 최종 선택, 메타 포함
    # SymbolMeta: symbol, role (regular/anchor),
    #   adv_usdt, execution_cost_bps, funding_carry_8h,  ← 절대 합산 금지
    #   beta_vs_market, cluster_id, tradeable_rank,
    #   basis_annualized_mean, basis_vol,
    #   oi_usdt_median, oi_to_adv, oi_change_30d,
    #   capacity_clip_usdt_list
    
    # 감사 추적
    rejected: dict[str, FilterReport]  # {symbol: FilterReport}
    # FilterReport: stage별 통과/탈락 + 메트릭 + RejectCode
    
    # 메타
    generated_at_utc: str           # 생성 시각
    ledger_confidence: str          # "reconstructed" or "official"
    n_stage0: int                   # Stage 0 진입 수
    n_stage1_pass: int              # ... (감시용)
    # ...
```

### FilterReport

```python
@dataclass(frozen=True)
class FilterReport:
    symbol: str
    stage0_pass: bool
    stage1_reason: RejectCode | None
    stage1_metrics: dict[str, float]
    # ... (stage 2~6)
    
    final_rank: int | None
    final_cluster_id: int | None
    audit_trail: list[str]          # 사람이 읽을 메시지
```

### 영속화 포맷

```
results/universe/
├── snapshot_4h_2026-05-18.parquet
│   (UniverseSnapshot 직렬화)
├── snapshot_4h_2026-05-18.json    (가독성)
├── snapshot_1h_2026-05-18.parquet
└── ...

data/futures/
├── universe_ledger.parquet        (LedgerRow 일별 append-only)
└── data_manifest.parquet          (ManifestRow — sha256·source·is_final 잠금)
```

백테스트는 과거 `as_of` 스냅샷을 재생, 라이브는 최신 스냅샷 사용.

---

## 요구사항 충족 매핑

| # | 요구사항 | 충족 메커니즘 | 모듈 |
|---|---|---|---|
| 1 | **PIT 거래가능성** | 프로세스 A/B 분리 + `knowledge_date` lag | ledger, pipeline |
| 2 | **생존/상폐편향 제거** | append-only ledger + 상장/상폐일 복원 + 상폐 심볼 히스토리 | ledger, exchange_meta |
| 3 | **유동성 & 체결성** | ADV 중앙값, Amihud, 호가깊이, 클립체결성 | liquidity |
| 4 | **거래비용 / 슬리피지 / 영향** | √-impact 모델, round-trip 수수료, 펀딩캐리 | cost_model |
| 5 | **데이터 품질** | 커버리지 80%, gap/NaN/frozen/타임스탬프 | data_quality |
| 6 | **자산 구조** | PERP/USDT/상태/레버리지토큰/multiplier | structure |
| 7 | **리스크·조작·이벤트** | 상장연령, 상폐임박, 펌프, 정지, 펀딩이상 | risk_events |
| **피드백** | **분산/breadth** | 상관 클러스터링, ENB 극대화 | selection |
| **피드백** | **펀딩 캐리** | 이상치만 배제, 구조적 고펀딩→피처 | risk_events, selection |
| **피드백** | **Churn 제어** | 히스테리시스 (entry/exit band) | selection |
| **검증 확정** | **실측 spread** | Vision bookDepth (2020+), Corwin-Schultz fallback (2019) | cost_model |
| **검증 확정** | **OI 딥히스토리** | Vision metrics 2020-09-01부터, BinanceVisionDownloader 확장 | data_quality |
| **검증 확정** | **상폐 심볼 복원** | Vision S3 857개 목록 + 마지막 kline 날짜 = 실제 상폐일 | ledger |
| **v1.2** | **상장 연령 90d** | Stage 5 ≥90d 게이트 (30d→90d 상향) + Stage 2 최근 60d 연속성 | risk_events, data_quality |
| **v1.2** | **변동성 밴드** | vol_30d_band 하한/상한 게이트 | risk_events |
| **v1.2** | **Tick 마찰 비용** | tick_cost_bps = tick_size/price × 0.5 × 1e4 → cost_bps 항목 추가 | cost_model |
| **v1.2** | **Mark-index basis** | premiumIndexKlines basis z-score 이상치 → Stage 5 추가 | risk_events |
| **v1.2** | **최근 60d 연속성** | last_60d_coverage ≥ 95% + n_zero_volume_bars_60d ≤ 1 | data_quality |
| **v1.2** | **Token unlock 인터페이스** | risk_event_override 수동 태깅 (자동화 미포함, Accepted Limitation) | risk_events |
| **v1.3** | **자본 티어 clip** | seed~xlarge 5단계 외생 고정 + capacity_clip_list + live_clip 분리 | config, liquidity |
| **v1.3** | **Walk-Forward 평가** | rolling 21/3 ~16 fold, purge=label_horizon, embargo=7d, DSR 적용 가능 | pipeline |
| **v1.3** | **Funding/cost 분리** | execution_cost_bps(게이트) + funding_carry_8h(피처) 완전 분리 | cost_model |
| **v1.3** | **Data manifest** | SHA256 lockfile, data_manifest_hash → config_hash와 결합한 완전 재현성 | persistence |
| **v1.3** | **Manual event PIT** | knowledge_date 필수·fail-closed·카테고리 분리·hindsight delta 보고 | risk_events |
| **v1.3** | **Basis/OI 피처 분리** | basis_z(게이트) / basis_mean·vol·oi_change(descriptive 피처, 예측 금지) | risk_events |
| **v1.3** | **Stage 6 관심사 분리** | 멤버십(tradeable_rank·hysteresis·anchor) vs 분산(optimizer). ENB/HRP 이전 | selection |

---

## 미확정 사항

검증으로 해소된 항목은 제거, 아직 결정이 필요한 항목만 남김.

### ✅ 해소됨

| 항목 | 결정 내용 |
|---|---|
| 호가창 데이터 소스 | Vision `bookDepth` 사용 (0.5 MB/일 실용적). Roll spread 폐기. |
| 과거 exchangeInfo | 보유 없음 확인. Vision S3 목록 + `onboardDate` + SETTLING 상태로 충분히 대체 가능. |
| OI/LSR 딥히스토리 | Vision `metrics/` 2020-09-01부터 가용 확인. BinanceVisionDownloader 확장으로 처리. |
| 펀딩비 Vision 경로 | `daily/` 없음, `monthly/` 확인. FAPI가 완전한 역사 보유로 primary 소스. |
| **클립 크기 (v1.3)** | 자본 티어 고정값 (seed 1k→small 5k→mid 10k→large 25k→xlarge 50k). 외생 고정 — equity 순환 차단. capacity_clip_list=[50k, 100k]. live_clip은 별도 equity 비례. [→ Stage 3](#stage-3-유동성--체결성-liquiditypy) |
| **IS 윈도우 (v1.3)** | IS=21개월, OOS=3개월, step=3개월, rolling walk-forward, ~16 fold. Stage 2 기준: 3,067봉(IS 3,834봉×80%). [→ Stage 2](#stage-2-데이터-품질-data_qualitypy) |

### 🚫 수용된 한계 (Accepted Limitations)

설계 결정에 의해 의도적으로 포함하지 않는 항목. "미확정"이 아니라 **명시적 포기**:

| 항목 | 미포함 이유 | 부분 대체 |
|---|---|---|
| **거래소간 가격 괴리** | Binance 단일 거래소 설계 원칙. 외부 API 접점 추가 불가. | 내부 funding/basis 이상치로 간접 감지 |
| **해킹·규제 뉴스** | 비정형 외부 뉴스 데이터. 자동 수집 파이프라인 범위 외. | risk_event_override 수동 태깅으로 대응 |
| **Token Unlock 자동화** | 베스팅 스케줄 API 외부 소스 의존. v1.1 범위 외. | risk_event_override 수동 태깅으로 대응 |
| **자유유통/FDV/공급구조** | Binance/Vision 소스에 미포함. CoinGecko 등 외부 의존 없음. | OI/ADV 비율로 간접 감지 |
| **청산(Liquidation)/ADL/Leverage bracket** | 거래소 내부 기제. 백테스트는 체결 기록만 관측 가능, 청산 이전 상태 불가시. 개별 포지션 크기 알 수 없음. | OI_to_ADV, basis_anomaly로 위험 신호만 감지 |

---

## 구현 로드맵 (추후)

설정된 후 다음 순서로 구현 예정:

1. **Phase 1**: `contracts.py` + `config.py` (데이터 계약 고정)
2. **Phase 2**: `ledger.py` + `exchange_meta.py` (데이터 기초층)
3. **Phase 3**: `data_quality.py` + `structure.py` + `liquidity.py` (Stage 1~3)
4. **Phase 4**: `cost_model.py` + `risk_events.py` (Stage 4~5)
5. **Phase 5**: `selection.py` (Stage 6, 클러스터링 핵심)
6. **Phase 6**: `pipeline.py` + `persistence.py` (오케스트레이션 & I/O)
7. **Phase 7**: `opt_main_futures.py` Step 1 통합 + 테스트

각 phase 후 단위 테스트 + 주요 모듈 정합성 검증.

---

## 참고: 기존 코드와의 단절

현재 `src/domain/futures/optimization/screener.py` 는 **완전히 제거**됩니다 (신 universe 모듈로 대체).

**마이그레이션 경로**:
- `orchestrate_universe_discovery()` → `build_universe(as_of=now)` 호출로 축소
- 백테스트는 과거 스냅샷 재생(`load_universe_snapshot(as_of)`) → 재현성 보장
- 라이브는 최신 스냅샷 사용

---

**문서 버전**: v1.3 (2026-05-18)  
**상태**: 아키텍처 확정 — 자본 티어·Walk-Forward·관심사 분리 완전 반영  
**다음**: Phase 1 (`contracts.py` + `config.py`) 구현 시작
