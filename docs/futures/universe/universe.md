# Binance Futures 유니버스 아키텍처 (v1.4 - AI Optimized)

**최종 검증/확정**: 2026-05-22  
**핵심 설계 목적**: Point-in-time(PIT) 준수, 생존/상폐 편향 제거, 결정론적 재현성 보장 및 거래 불가능 자산 원천 차단.

## 0. 2026-05-22 반영 사항 (코드 사실)

- `sync_utils.run_historical_sync()`는 `sync_mode="full_history_master"`일 때 전체 USDT perpetual 모집단 기준으로 동기화를 수행한다.
- 동기화 실행마다 coverage 리포트가 저장된다.
  - 경로: `logs/futures/universe/sync_coverage_report.parquet`
  - 단위: symbol row append
  - 핵심 컬럼: `run_ts_utc`, `sync_mode`, `start_date`, `end_date`, `symbols_total`, `sync_tasks_total`, `synced_symbols`, `task_coverage_ratio`, `symbol`, `synced_days`, `is_synced`
- 목적: full-history 수집 시점의 실행 커버리지/누락 심볼을 run 단위로 추적하기 위한 lightweight audit trail.
- **초기 단일 파이프라인 수집(Pre-fetch) 및 오프라인 백테스트 도입**:
  - 기존의 백테스팅 연산 중 발생하는 동적 API 호출과 이로 인한 포트 고갈 문제를 예방하기 위해, 백테스팅에 필요한 모든 시간대(`1h, 1d, 4h, 1m`) 및 `funding` 데이터를 안전한 멀티프로세스 커넥션 풀을 활용해 일괄 선 수집(Pre-fetch)하도록 데이터 단일 파이프라인 구축.
  - `storage.py`의 `run_historical_sync` 및 `sync_single_symbol_data`에 `sync_4h=True` 파라미터를 추가하여 4h 데이터의 명시적 선 수집을 완결함.
  - **1m 데이터의 Targeted Pre-fetch 제어**: 1m 데이터는 시계열 용량이 매우 크고 네트워크 수집 오버헤드가 극대화되므로, 유니버스 필터링 이전(1.5단계)에는 수집을 원천 생략(`sync_1m=False`)합니다. 이후 **유니버스 7단계 필터링을 완전히 통과하여 최종 백테스팅 대상으로 선정된 정예 심볼군(`load_symbols`)에 대해서만 3단계 데이터 로드 직전에 선별 수집(Targeted Pre-fetch)**하도록 최적화하여 불필요한 트래픽과 디스크 낭비를 차단합니다.
  - 멀티프로세스 워커 `_worker` 인자 구조를 7요소 튜플(`symbol, start_date, end_date, delist_date, sync_1m, sync_funding, sync_4h`)로 확장 및 리팩토링하여 병렬 처리 성능과 타입 안정성을 확보함.
  - CLI 인자(`--symbols`)가 주입될 경우, 명시된 대상 심볼 및 유니버스 필수 심볼(anchors, macros)로 타겟 수집 대상을 엄격하게 제한(Targeted Sync)하여 시스템 불필요 리소스를 차단함.

---

## 1. 핵심 아키텍처 및 데이터 흐름

유니버스 빌드는 외부 거래소 API와 완전히 격리된 **순수 함수(Pure Function)**로 동작한다.

```
build_universe(as_of: date, tf: str, cfg: UniverseConfig) -> UniverseSnapshot
```
* **결정론적 재현성**: 동일 `as_of` 일자 + 동일 `UniverseConfig` + 동일 `universe_ledger.parquet` ➡️ 항상 동일한 유니버스 스냅샷 반환.
* **룩어헤드 차단**: `as_of` 시점 이후에 알려진 정보(`knowledge_date > as_of`)는 데이터 쿼리 및 연산에서 원천 배제.

### 2개 프로세스 분리 구조

```
┌─────────────────────────────────────────────────────┐
│ [프로세스 A: Ledger 적재 및 갱신 (Pre-fetch 동기화)]  │
│ - 온라인 · append-only · Smart Sync 적용            │
│ - Smart Filter: 거래량 상위 40% 엘리트 심볼 선별      │
│ - Pre-fetch Pipeline: 1h, 1d, 4h, 1m, funding 일괄 수집│
│ - Parallel Worker: 7요소 튜플 기반 병렬 고속 동기화     │
│ - 데이터 소스: Binance Vision & CCXT API            │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼ [data/futures/universe_ledger.parquet] (상폐 자산 포함 역사 패널)
                   │
┌──────────────────┴──────────────────────────────────┐
│ [프로세스 B: Universe 빌드]                          │
│ - 오프라인 · 순수 함수 · 거래소 API 비접촉           │
│ - as_of 이전 데이터만 쿼리 (`knowledge_date <= as_of`) │
│ - pipeline.py 실행 ➡️ 7단계 Funnel 통과            │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼ [logs/futures/universe/snapshots/...] (Reproducible Snapshot)
```

---

## 2. 디렉토리 구조 및 모듈 매핑

`src/domain/futures/universe/` 디렉토리 내 각 파일의 단일 책임과 역할 정의:

| 파일명 | 주된 역할 및 책임 | 주요 외부 라이브러리 |
|---|---|---|
| `__init__.py` | 유니버스 패키지 공개 API 통합 노출 및 Export 관리 | `pandas` |
| `config.py` | 각 Stage 임계값 중앙화 및 결정론적 `config_hash` 생성 | `dataclasses`, `hashlib`, `json` |
| `models.py` | 기존 `contracts.py`, `ledger.py`, `structure.py` 통합. `LedgerRow`, `SymbolMeta`, `FilterReport`, `UniverseSnapshot` 데이터 스키마 정의 및 상장/상폐일 복원, 쿼리, `apply_structure_stage` 포함 | `dataclasses`, `enum`, `pandas` |
| `filters.py` | 기존 `liquidity.py`, `cost_model.py`, `risk_events.py` 등의 필터 연산 함수 통합 (`apply_liquidity_stage`, `apply_cost_model_stage`, `apply_risk_events_stage` 제공) | `pandas`, `numpy` |
| `data_quality.py` | **Stage 2**: 데이터 누락률, gap 크기, frozen bar 검사 및 연속성 평가 | `pandas`, `numpy` |
| `selection.py` | **Stage 6**: 종합 점수화(Rank), 히스테리시스, 상관성 클러스터링 및 앵커 결합 | `pandas`, `numpy` |
| `membership.py` | 유니버스 스냅샷 통계, Churn Control, Dwell time 등을 처리하는 로직 | `pandas`, `numpy` |
| `storage.py` | Parquet/JSON 영속화, S3 XML 파싱, Smart Sync(1h, 1d, 4h, 1m, funding 병렬 수집) 오케스트레이션 및 동기화 수행 (`sync_4h` 명시적 지원) | `pandas`, `pyarrow`, `multiprocessing` |
| `pipeline.py` | Stage 0~6 순차 실행, `FilterReport` 생성 및 스냅샷 오케스트레이션 | `pandas`, `datetime` |

---

## 3. 데이터 계약 및 영속화 스키마 (`models.py`)

### 3.1 LedgerRow (원시 데이터 패널 행)
`universe_ledger.parquet`에 일단위 append-only 형식으로 기록되는 구조.
* `date`: 실제 데이터 발생 일자 (YYYY-MM-DD)
* `knowledge_date`: 해당 행의 정보가 시스템에 실제로 가용해진 날짜 (T+1 적용으로 same-day 룩어헤드 방지)
* `is_listed` / `is_trading` / `status`: 상장 여부 및 활성 거래 상태 ("TRADING", "SETTLING", "DELISTED" 등)
* `first_kline_date` / `delist_date`: FAPI `exchangeInfo.onboardDate` 기준 실제 상장일(누락 시 첫 kline 관측일 fallback) 및 실제 최종 kline 관측일 기준 상폐일. `listing_age_days`는 본 값 기준으로 산출되어 sync 윈도우 시작점에 고정되지 않음.
* `adv_usdt_median`: 30일 거래대금 중앙값
* `amihud_30d`: 가격 영향 지수 (PIT 준수 리샘플링 기반)
* `mark_price`: 해당 시점의 기준 가격
* `last_60d_coverage`: 최근 60일 데이터 존재율 (0.0 ~ 1.0)
* `n_zero_volume_bars_60d`: 최근 60일 내 거래량 0인 봉 개수
* `risk_event_override`: 수동 리스크 배제 사유 태그

### 3.2 SymbolMeta (스냅샷 탑재 개별 자산 정보)
유니버스 빌드 통과 후 `UniverseSnapshot` 내 `selected` 튜플에 담겨 포트폴리오 최적화(Optimizer) 레이어로 전달되는 피처 데이터셋.
* `symbol`: 심볼명 (예: `"BTC/USDT"`)
* `role`: 역할군 (`"anchor"` 또는 `"regular"`)
* `adv_usdt` / `execution_cost_bps`: 30일 median ADV 및 예상 총 라운드트립 마찰 비용 (※ 절대 상호 합산 금지)
* `funding_carry_8h`: 최근 8시간 원천 펀딩비 (signed alpha 성분으로 별도 보존)
* `beta_vs_market`: market basket 대비 historical beta
* `cluster_id`: 상관성 거리 기준 클러스터링 군집 번호
* `tradeable_rank`: Stage 6 종합 스코어 랭킹
* `basis_annualized_mean` / `basis_vol`: Mark-index basis 기초 통계 피처 (alpha 신호용; `premiumIndexKlines` downloader 구현 후 실측값 공급 예정)

---

## 4. 7단계 유니버스 필터링 Funnel 상세

`pipeline.py` 오케스트레이터를 통해 실행되며, 각 Stage는 통과 데이터프레임과 감사용 `FilterReport`를 반환한다.

### Stage 0: 적격 모집단 (Eligibility)
* **목적**: `as_of` 기준 상장 및 거래가 활성화되어 있는 모집단 추출.
* **로직**: `knowledge_date <= as_of` 이고 `is_listed == True` 및 `is_trading == True`인 전 심볼 쿼리. (미래 상장 예정 자산 자동 제외)
* **구현 모듈**: `models.py` ➡️ `load_ledger_slice()`

---

### Stage 1: 자산 구조 필터 (Structure)
* **목적**: 거래 불가 상태, 레버리지 상품 및 규격 외 계약 자산 원천 차단.
* **구현 모듈**: `models.py` ➡️ `apply_structure_stage()`
* **검증 규칙**:
  1. `contract_type == "PERPETUAL"` (기한부 선물 배제)
  2. `quote_asset == "USDT"` 또는 `margin_asset == "USDT"` (USDT 마진 계약만 허용)
  3. `status == "TRADING"` (HALT, SETTLING 등 비정상 거래 상태 즉각 제외)
     * *주의*: `deliveryDate` 정보로 임의의 상폐 시점 예측 금지. 오직 `status == "SETTLING"` 인가 여부로만 판별.
  4. 레버리지 토큰 배제: 심볼 문자열 내 `UP`, `DOWN`, `BULL`, `BEAR` 키워드 정규식 패턴 매칭 차단.
  5. `contract_multiplier` 유효성 검사: 수치가 유한하고 `> 0.0` 인지 확인.
* **기술 스택**: `pandas` DataFrame 벡터화 마스킹 연산 (`is_perp & is_usdt_quote & is_trading & ...`)

---

### Stage 2: 데이터 품질 필터 (Data Quality)
* **목적**: 백테스트 및 라이브 시그널 연산 시 결측치/이상값으로 인한 연산 오류 예방.
* **구현 모듈**: `data_quality.py` ➡️ `apply_data_quality_stage()`
* **설정 파라미터 (`Stage2Config`)**:
  * `min_is_coverage = 0.80` (최소 데이터 충족률 80%)
  * `min_is_bars_4h = 1_296` (9개월 기준 4시간 봉 기준값: $9 \times 30 \times 6 \times 80\% = 1{,}296$. Stage 5의 `listing_age_days >= 90`이 단기 상장 종목을 별도로 걸러주므로, Stage 2는 롤링 지표 계산에 충분한 기간만 요구함. 백테스트 IS 윈도우 충분성은 optimizer fold에서 별도 검증.)
    * **PIT 보장**: `n_is_bars` / `expected_is_bars`는 `storage.py`에서 각 레저 행의 `date`까지 **누적된 4h 봉 수**로 산출된다. 전체 수집 길이를 모든 행에 상수로 기록하던 look-ahead 결함은 제거되었으며, 이로써 Stage 2 게이트가 `as_of` 시점별로 정상 동작한다.
  * `min_coverage_60d = 0.95` (최근 60일 연속성 점검: 단기 결측 감지)
  * `max_zero_volume_bars_60d = 1` (체결이 없는 동결 자산 차단)
  * `max_gap_bars = 200` (최대 허용 단일 데이터 Gap 크기; `max_gap_bars` 컬럼이 레저에 공급될 때 활성화)
  * `max_gap_count = 1` (최대 허용 gap 발생 횟수)
  * `max_frozen_bars_60d = 4` (연속 동일 종가 5회 이상 발생 시 시세 이상 자산으로 간주)
* **체크리스트**:
  * NaN / Inf 값 부존재 확인 (`~has_nan & ~has_inf`)
  * 타임스탬프 단조 증가 및 UTC 정합성 확인
  * 필수 데이터 컬럼 존재 여부 (`has_kline`, `has_funding`)
* **Walk-Forward 검증 메타 (rolling)**:
  * `is_months = 24`, `oos_months = 6`, `step_months = 3`, `purge = 1 decision bar`, `embargo_days = 0`
* **기술 스택**: `numpy.where`, `pandas.to_numeric` 벡터화 예외 판별.

---

### Stage 3: 유동성 & 체결성 필터 (Liquidity & Capacity)
* **목적**: 슬리피지 통제 불가능 및 AUM 수용 한계 자산 배제.
* **구현 모듈**: `filters.py` ➡️ `apply_liquidity_stage()`
* **설정 파라미터 (`Stage3Config`)**:
  * `min_adv_usdt_median = 25_000_000.0` (30일 Median ADV 하한선 25M USDT. Spike 왜곡 방지용 Median 필수 사용)
  * `max_amihud_30d = 1.63e-9` (일별 $|ret| / (volume \times price)$ 기준 30일 Amihud Illiquidity 상한선. ADV ≥ 25M 통과 종목 실측 분포: p90≈4.6e-10, p95≈6.3e-10, p99≈1.1e-9, max≈5.2e-8. 임계값 = p99 × 1.5 ≈ 1.63e-9 → ADV 게이트 통과 종목 중 명백 비유동 상위 ~1%만 탈락. 구 임계값 6e-8은 실측 max(5.2e-8)보다도 높아 영구 inert였으므로 실측 분포 기반으로 재산정함.)
  * `max_clip_to_adv = 0.005` (ADV 대비 1회 집행 Clip 비율 상한 0.5%)
* **자본 티어별 Clip 설계 (외생 고정 적용으로 순환 루프 제거)**:
  $$\text{screening\_clip\_usdt} = \begin{cases} 
  1,000 & \text{Tier: seed} \\
  5,000 & \text{Tier: small} \\
  10,000 & \text{Tier: mid (Default)} \\
  25,000 & \text{Tier: large} \\
  50,000 & \text{Tier: xlarge} 
  \end{cases}$$
  * AUM Ceiling 수용력 평가용 고정 Clip 리스트: `[50_000, 100_000]`
* **기술 스택**: `pandas` 연산 시 분모 0 치환 방지 `replace(0, np.nan)` 처리 및 벡터화 연산.

---

### Stage 4: 거래 비용 모델 필터 (Execution Cost)
* **목적**: 마찰 비용이 기대 에지(Alpha)를 갉아먹는 자산 사전 제외. (펀딩 캐리는 에지 성분이므로 본 단계 비용에서 완전 격리 보존)
* **구현 모듈**: `filters.py` ➡️ `apply_cost_model_stage()`
* **비용 함수 및 수식**:
  $$\text{execution\_cost\_bps} = 2 \cdot \text{taker\_fee\_bps} + 2 \cdot \text{half\_spread\_bps} + \text{impact\_bps} + \text{tick\_cost\_bps}$$
  * `taker_fee_bps`: Binance 기본 요율 적용.
  * `half_spread_bps` (실측 기반 분기 처리):
    * **2020-01-01 이후**: `bookDepth` 데이터 실측 중앙값 사용 (`median(best_ask - mid) over 4h window`).
    * **2020-01-01 이전 (Fallback)**: `Corwin-Schultz` OHLC 변형 모델 적용 (Roll 스프레드식 사용 금지).
      $$\text{spread} = \frac{2(e^\alpha - 1)}{1+e^\alpha}, \quad \text{half\_spread\_bps} = \text{spread} \times 5,000$$
      $$\alpha = \frac{\sqrt{2\beta} - \sqrt{\beta}}{3 - 2\sqrt{2}} - \sqrt{\frac{\gamma}{3 - 2\sqrt{2}}}$$
      $$\beta = \ln(H_t/L_t)^2, \quad \gamma = \ln\left(\frac{\max(H_t, H_{t-1})}{\min(L_t, L_{t-1})}\right)^2$$
  * `impact_bps` (Square-root Impact): $k \cdot \sigma \cdot \sqrt{\frac{\text{screening\_clip\_usdt}}{\text{adv\_usdt\_median}}} \cdot 10,000$ (여기서 $k \approx 18.0$, $\sigma$는 30일 변동성)
  * `tick_cost_bps`: $\frac{\text{tick\_size}}{\text{mark\_price}} \times 0.5 \times 10,000$ (이중 계산을 막기 위한 반올림 마찰 보정)
* **필터 게이트**: `execution_cost_bps <= max_execution_cost_bps` (기본 임계값: 50.0 bps)

---

### Stage 5: 리스크 & 이상치 필터 (Risk Events)
* **목적**: 펌프앤덤프, 극단적 유동성 고갈 및 인위적 시세 조작 자산 배제.
* **구현 모듈**: `filters.py` ➡️ `apply_risk_events_stage()`
* **검증 규칙**:
  1. **상장 연령 (Listing Age)**: 최소 90일 이상 경과 자산만 허용 (`listing_age_days >= 90`). (상장 빔 및 초기 락업 해제 충격 방지)
  2. **변동성 밴드 (Vol Band)**: 4h 바 기준 연율화 변동성 `vol_30d`가 `[0.05, 4.0]` 범위 내 존재할 것 (5%~400% 연율). 활동성 없는 고사 코인 및 극단적 meme/junk 코인 동시 배제.
  3. **펀딩비 이상치**: 8시간 원천 펀딩비의 **MAD 기반 robust z-score**(`funding_zscore`) 절대값이 2.5를 초과하거나, 1일 이내 급격한 부호 반전(`enable_funding_sign_flip`; 양쪽 펀딩비 모두 `|funding| > funding_sign_flip_min_abs`(0.001) 조건 충족 시에만 이상치 인정)이 일어나는 조작/스퀴즈성 자산 차단. (※ 단순 고펀딩 일관 유지 자산은 carry harvest 전략 활용을 위해 **정상 통과** 시킴)
     * **MAD robust z 산출**: `funding_zscore`는 `storage.py`에서 종목별 30일 롤링 윈도우로 PIT 산출된다. 표준편차 대신 MAD(중앙값 절대편차, breakdown point 50%, `scale='normal'`)를 사용해 fat-tail 극단치의 masking 효과를 차단하며, 펀딩비가 장기 안정 구간일 때 MAD 분모가 0에 수렴해 z가 발산하는 수치 불안정을 막기 위해 출력값을 `[-50, 50]`으로 클리핑한다.
  4. **수동 리스크 오버라이드 (Manual Override)**:
     * `ManualEventRow` 수동 입력 이벤트 발생 시 배제.
     * **Fail-Closed 원칙**: `knowledge_date`가 누락된 수동 배제 요청은 PIT 정합성을 해치므로 해당 레코드 자체를 무시(배제하지 않음)하여 휴리스틱 오염 차단.
* **기술 스택**: `numpy.where` 조건 결합 마스킹.

---

### Stage 6: 멤버십 선택 & 랭킹 (Selection)
* **목적**: 전략 독립적(Strategy-Agnostic) 관점에서 거래 가치가 가장 높은 최정예 유니버스 멤버 선발. (※ 분산 최적화 - ENB, HRP 등은 본 유니버스 레이어가 아닌 **Optimizer 단에서 처리**하므로 멤버 선별만 수행)
* **구현 모듈**: `selection.py` ➡️ `apply_selection_stage()`
* **핵심 알고리즘**:
  1. **Tradeability Composite Score 산출**:
     $$\text{tradeable\_score} = 0.40 \cdot \text{liq\_norm} + 0.30 \cdot \text{cost\_inv\_norm} + 0.20 \cdot \text{quality\_norm} + 0.10 \cdot \text{stability\_norm}$$
     * `liq_norm`: $normalize(adv\_usdt\_median)$
     * `cost_inv_norm`: $normalize(1 / execution\_cost\_bps)$
     * `quality_norm`: $normalize(last\_60d\_coverage)$
     * `stability_norm`: $normalize(listing\_age\_days)$
     * (여기서 $normalize$는 $[0.0, 1.0]$ 범위 내 선형 Min-Max Scaling)
  2. **상관성 클러스터링 피처 생성 (WARD Linkage)**:
     * 피처 운반용 correlation 산출 (250 거래일 윈도우) ➡️ Distance $D = 1 - |corr|$ 변환 ➡️ 계층적 클러스터링 ➡️ 자산 메타정보 `cluster_id` 피처로 snapshot에 운반. (본 단계에서는 제거 없이 메타 전달만 담당)
  3. **히스테리시스 필터 (Hysteresis Churn Control)**:
     * 유니버스 잦은 교체로 인한 거래비용 폭증(Churn) 제어.
     * 신규 진입 장벽: $\text{Rank} \le K_{in}$ (기본값: 20)
     * 이탈 장벽: $\text{Rank} > K_{out}$ (기본값: 35)
     * 최소 Dwell 일수: `90일` (Dwell 미달 시 Rank가 이탈 범위 내에 있어도 퇴출 유예)
     * 단, Stage 1~5에서 결격 사유 발생 시 히스테리시스 및 Dwell 조건을 무시하고 **즉시 퇴출**.
  4. **앵커 자산 강제 편입**:
     * `"BTCUSDT"`, `"ETHUSDT"`는 스코어 및 랭킹에 관계없이 강제 유니버스 최상단 고정 편입 (`role = "anchor"`).
     * 앵커 자산은 포트폴리오 최적화 시 가중치 0% 배분을 허용하여 가상 reference로 기능하게 함.
     * **주의**: 앵커 심볼 표기는 슬래시 없는 Binance 원형 심볼(`BTCUSDT`)을 사용한다. 슬래시 포함(`BTC/USDT`) 사용 시 Stage 1~5 자연 통과 심볼(`BTCUSDT`)과 phantom churn artifact가 발생한다.

---

## 5. 데이터 재현성 잠금 및 검증 감사 계약

### 5.1 Data Manifest
외부 환경 변화에 흔들리지 않는 재현성을 달성하기 위해 입력 파일들의 무결성을 SHA256 체크섬으로 기록하는 `data_manifest.parquet`을 유지한다.
* `data_manifest_hash`는 파이프라인 시동 시 투입되는 모든 Klines 파일의 SHA256 지문을 정렬한 문자열의 마스터 해시다.
* `UniverseSnapshot.config_hash` 와 결합하여 오프라인 백테스트 시 동일 입력 데이터를 사용했는지 물리적으로 완벽히 검증한다.

---

## 6. 유니버스 품질 검증 가드레일 (Quality Gates)

`opt_main_futures.py` 실행 시, 유니버스 빌드 직후 다음의 객관적 지표를 검증하여 통과하지 못할 경우 최적화 단계를 즉시 중단(Hard-Stop)한다. 이는 전략(Alpha)과 무관하게 유니버스 자체가 복리 자산 극대화에 적합한 '건전한 토대'를 제공하는지 측정하기 위함이다.

### 6.1 비용 및 수용력 지표 (Cost & Capacity)
전략의 실행력을 결정하는 물리적 한계치를 측정한다.

> **임계값 구분**: Stage 3/4의 per-symbol 게이트(개별 자산 최소 요건)와 Quality Gate의 hard-stop(포트폴리오 레벨 집계 게이트)은 역할이 다르므로 수치가 다를 수 있다.
> - Stage 3 per-symbol: `adv_usdt_median >= 25M` (개별 종목 최소 유동성)
> - Stage 4 per-symbol: `execution_cost_bps <= 50 bps` (개별 종목 최대 비용)
> - Quality Gate: 선발된 유니버스 전체의 중앙값 기준 집계 게이트
> - **Cost 비고**: `bookDepth` 미적재 시 impact 계산이 ~12 bps 수준으로 고정되어 Stage 4 게이트가 사실상 비활성 상태임. `bookDepth` 공급 이후 Stage 4 재기능 예정.

| 평가지표 | Excellent | Good (Pass) | Fail (Hard-Stop) |
| :--- | :---: | :---: | :---: |
| **중앙값 집행 비용 (Median Cost)** | < 18.0 bps | 18.0 ~ 50.0 bps | **> 50.0 bps** |
| **중앙값 유동성 (Median ADV)** | > 100M USDT | 25M ~ 100M USDT | **< 25M USDT** |

*   **Median Cost**: 슬리피지(Impact)와 수수료를 포함한 왕복 마찰 비용의 중앙값.
*   **Median ADV**: 유니버스 내 자산들의 30일 일평균 거래대금 중앙값.

### 6.2 예기치 않은 강제 퇴출률 (Forced Dropout Rate)
유니버스 로직이 리스크 자산을 사전에 얼마나 잘 식별하는지 측정한다.

*   **정의**: 90일 최소 유지 기간(Dwell Time)을 채우기 전, 비정상적 사유(상폐, 펀딩비 조작, 유동성 고갈 등 Stage 1~5 결격)로 긴급 퇴출된 자산의 비율.
*   **공식**: $\frac{\text{Count of dropouts where RejectCode} \neq \text{RANKED\_OUT}}{\text{Total Previous Universe Size}}$
*   **소표본 가드**: 이전 유니버스 크기 `prev_universe_size < 10`인 경우, 통계적으로 무의미한 비율 산출을 방지하기 위해 해당 지표 산출을 보류하고 `WARNING` 로그를 남긴다. (예: 유니버스 크기 3에서 1개 탈락 = 33%는 오탐)
*   **평가 구간** (`prev_universe_size >= 10` 조건 충족 시):
    *   **Excellent (0% ~ 3%)**: 리스크 사전 차단 완벽.
    *   **Good (3% ~ 10%)**: 일반적인 시장 변동성 내 수용 가능.
    *   **Fail (> 10%)**: 유니버스 필터링 엔진 결함으로 간주. (강제 청산 슬리피지 위험 과다)

### 6.3 종합 품질 스코어 (Snapshot Score)
파이프라인 모니터링을 위한 전략 독립적 **무차원** 지표.

$$Score_{universe} = fill\_rate \times \log_{10}\!\left(\frac{\text{median\_adv\_usdt}}{10^6}\right) \times \frac{1}{\text{mAEC\_bps}}$$

| 항목 | 정의 | 단위 |
|---|---|---|
| $fill\_rate$ | $n_{selected} / K_{in}$ — 목표 인원 대비 실제 충원율 | [0, 1] |
| $\log_{10}(\text{median\_adv} / 10^6)$ | ADV outlier 완화: 1M→0, 10M→1, 100M→2, 1B→3 | 무차원 |
| $1 / \text{mAEC\_bps}$ | 유니버스 중앙값 집행 비용 역수 (낮을수록 고점수) | $\text{bps}^{-1}$ |

**예시 수치**: 완전 충원(fill=1.0), Median ADV = 100M USDT, mAEC = 15 bps  
$$Score = 1.0 \times \log_{10}(100) \times \frac{1}{15} \approx 0.133$$

**설계 근거**: 구 공식 $\frac{Capacity \times Stability}{mAEC}$ 는 Capacity(USD) × Stability(무단위) / Cost(bps) 구조로 차원이 불일치하고 BTC ADV 같은 outlier가 지수를 지배하는 문제가 있었다. 신 공식은 ADV에 log 스케일을 적용하여 outlier를 완화하고, fill_rate 가중을 통해 충원 실패 패널티를 직접 반영하며, 완전 무차원이므로 시계열 비교가 가능하다.

이 지표는 백테스팅 진입 전 유니버스 빌드 결과의 '순수 품질'을 역사적으로 비교 및 로깅하는 용도로 활용된다.
