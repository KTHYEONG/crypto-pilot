# 데이터 흐름 및 시계열 불변식 (Data Flow & Temporal Invariants)

## 1. 단계별 데이터 변환 파이프라인 (Data Transformation Flow)

원시 거래소 데이터가 시스템을 거쳐 연구 파이프라인과 실거래 원장으로 흘러가는 구체적인 데이터 흐름은 다음과 같습니다:

| 단계 (Stage) | 입력 데이터 (Input) | 처리 내용 (Processing) | 출력 데이터 (Output) | 메인 모듈 (Main Module) |
| :--- | :--- | :--- | :--- | :--- |
| **1. 마켓 데이터 인제스천** | Binance FAPI, Spot v3, Margin SAPI, Vision S3 아카이브 | REST 폴링, 비동기 웹소켓 수집, 컬럼형 Parquet 저장, SHA-256 매니페스트 무결성 추적 | 정규화된 Parquet 시계열 (`ohlcv/1h`, `markPriceKlines/1h`, `funding`) | [`src/market_data/services/`](file:///home/kth/crypto-pilot/src/market_data/services/) |
| **2. Tail 증분 갱신 & 프루닝** | 기존 Parquet 파일, 최신 Binance REST API 델타 | 로컬 디스크 tail 타임스탬프 파악 후 `max(tail - 2h, now - lookback)` 구간 증분 패치, 보존 한도 초과 원자적 temp-replace 축소 | 최신 Parquet (약 20초 이내), 슬라이딩 윈도우 스토리지 | [`src/live/data_refresh.py`](file:///home/kth/crypto-pilot/src/live/data_refresh.py), [`src/market_data/retention.py`](file:///home/kth/crypto-pilot/src/market_data/retention.py) |
| **3. PIT 유니버스 필터링** | 1시간봉 거래대금(Quote Volume), 상장 이력 | 소스 갭 가드(결손 심볼 배제) $\rightarrow$ 직전 720시간 거래대금 중앙값 상위 50% $\rightarrow$ 상위 60위 진입/120위 탈락 히스테리시스 | `execution_mask` (불리언 매트릭스: 타임스탬프 x 심볼) | [`src/mhs/frozen_research_universe.py`](file:///home/kth/crypto-pilot/src/mhs/frozen_research_universe.py), [`src/quant/universe/pit_universe.py`](file:///home/kth/crypto-pilot/src/quant/universe/pit_universe.py) |
| **4. 피처 및 알파 신호 산출** | 1시간봉 종가 및 테이커 매수대금, 8시간 펀딩비 | 5개 직교 횡단면 피처(테이커 플로우 불균형 168h/720h, 모멘텀 336h, 잔차 모멘텀 336h, 왜도 168h) 산출 | 횡단면 알파 신호 행렬 | [`src/mhs/features.py`](file:///home/kth/crypto-pilot/src/mhs/features.py), [`src/quant/technical_experts/cross_sectional.py`](file:///home/kth/crypto-pilot/src/quant/technical_experts/cross_sectional.py) |
| **5. Frozen Top-20 결합** | 5개 피처 행렬, 상위 20개 활성 심볼 | 5개 직교 피처 랭크 평균 결합, 달러 중립화 및 종목별 5% 클립(Gross 복원) | `target_weights` (달러 중립, Gross=1.0) | [`src/mhs/frozen_research_candidate.py`](file:///home/kth/crypto-pilot/src/mhs/frozen_research_candidate.py), [`src/mhs/books.py`](file:///home/kth/crypto-pilot/src/mhs/books.py) |
| **6. 인과적 베이지안 Kelly 사이징** | 단위북 누적 일별 수익률, 계좌 자본, 거래소 브래킷 | 사전 $\mu_0=0$(730일 가중) 기반 전일 실적으로 사후 $\mu, \sigma$ 갱신, 기대수익 50% 할인 켈리 및 증거금 상한 산출 | 최적 동적 노출 배율 $L^*$ 및 명목 목표 포지션 | [`src/mhs/account_policy.py`](file:///home/kth/crypto-pilot/src/mhs/account_policy.py) |
| **7. 3분봉 체결 원장 & 실계좌 리플레이** | 목표 포지션, 3분봉 OHLCV, 1시간봉 마크 가격, 8시간 펀딩비 | 엄격한 회계 순서: MTM 평가 $\rightarrow$ 8시간 펀딩비 정산 $\rightarrow$ 테이커/메이커(30분 대기 후 폴백) 체결 $\rightarrow$ 증거금·수수료 차감 | [`SimulatedInventoryLedgerResult`](file:///home/kth/crypto-pilot/src/mhs/execution/ledger.py), [`replay_account`](file:///home/kth/crypto-pilot/src/mhs/execution/account_replay.py) |
| **8. 백테스트 레지스트리 및 아티팩트 봉인** | 완성된 모의 실행 원장 및 일별 단위 수익률 | SQLite3 단일 원장 및 `index.jsonl` 영속화, 부트스트랩 단위수익률 `LIVE_ARTIFACT_KEY` AES-256-GCM 암호화 봉인 | `data/backtests/registry.sqlite3`, `deploy/mhs/*.parquet.enc` | [`src/backtests/registry.py`](file:///home/kth/crypto-pilot/src/backtests/registry.py), [`src/live/crypto.py`](file:///home/kth/crypto-pilot/src/live/crypto.py) |
| **9. 라이브 런타임 인계** | 봉인 아티팩트, 1h 120일 패널 창 | 23:03 UTC 데몬: 증분 갱신 $\rightarrow$ 인프로세스 frozen 신호 스텝 $\rightarrow$ strict_passive 메이커/테이커 발주 $\rightarrow$ 잔고 대조 | 디스크 상태 파일 영속화 (`live_fills`, `live_portfolio_state`, `live_tax_ledger`) | [`src/live/scheduler.py`](file:///home/kth/crypto-pilot/src/live/scheduler.py), [`src/live/frozen_signal.py`](file:///home/kth/crypto-pilot/src/live/frozen_signal.py) |

---

## 2. 타임스탬프 규격 및 시계열 인과성 불변식 (Temporal Invariants)

```text
스냅샷 시점 t_snap (22:00 UTC)         결정/발주 시점 t_rel (23:03 UTC)         체결 윈도우 및 정산 (23:00 ~ 00:00 UTC)
───────┬─────────────────────────────────────┬────────────────────────────────────┬──────────►
       │ 22:00 1h 캔들 완료 (Close)          │ 1h 120일 패널 기반 Top-20 신호 산출│ 3분봉(3m) 10개(30분) 앵커 지정가 대기:
       │ 마크 가격 (Mark Price) 확정          │ 베이지안 Kelly 동적 노출 계산      │ 미체결분 테이커 폴백 체결 완료
       │ 관측 데이터셋 확정                  │ 주문 의도 (Order Intent) 제출      │ 00:00 UTC 8시간 펀딩비 결제 및 MTM 정산
```

### 1) 이벤트 타임(Event Time)과 처리 시점(Processing Time)
* **이벤트 타임**: 모든 캔들의 타임스탬프는 해당 캔들의 **시작 시각(Open Timestamp, 밀리초 UTC)**을 의미합니다. 즉, `2024-01-01 00:00:00` 1시간봉은 `01:00:00`에 완성(Close)됩니다.
* **관측 스냅샷 시점 ($t_{\text{snap}} = 22\text{:00 UTC}$)**: Frozen 전략은 22:00 UTC에 종료된 1시간봉까지를 닫힌 과거 데이터로 스냅샷합니다.
* **신호 공개 및 발주 시점 ($t_{\text{rel}} = 23\text{:03 UTC}$)**: 전략 신호는 23:00 UTC(`release_hour_utc = 23`)에 확정되며, 네트워크 지연 및 안전 버퍼(3분)를 반영하여 23:03 UTC에 라이브 데몬이 발주합니다.
* **체결 윈도우 ($23\text{:00} \sim 00\text{:00 UTC}$)**:
  * **Strict Passive 메이커 집행**: 발주 시점의 기준가(anchor)에 지정가를 배치하고 최대 30분(10개 3분봉) 동안 대기하며, 미체결 잔량은 테이커로 즉시 폴백 체결합니다.
  * **Taker Parity 집행**: 신호 확정 즉시 다음 3분봉에서 전량 테이커로 체결합니다.

### 2) 타임존 및 달력 기준
* **엄격한 UTC 강제**: 모든 타임스탬프, 데이터프레임 인덱스, 로그는 타임존이 명시된 UTC(`tz="UTC"`)여야 합니다. Naive 타임스탬프는 `DataIntegrityError`로 즉시 거부됩니다.
* **암호화폐 24/7/365 연속 그리드**: 시장 휴장일이나 주말 단절이 존재하지 않습니다. 결측치는 거래소 장애나 시세 누락을 의미합니다.

### 3) Point-in-Time (PIT) 인과성 규칙
* **생존 편향 배제**: 유니버스 선정은 매 시점 과거 720시간(30일) 동안 관측된 거래대금만을 기준으로 평가합니다. 미래의 상장 유지 여부나 누적 수익률 정보가 유입되지 않도록 통제합니다.
* **마크 가격 Causal Forward-Fill**: 마크 가격(`markPriceKlines/1h`)은 MTM 포트폴리오 평가와 청산 위험 판정에만 사용되며, 알파 신호 스코어링에는 개입하지 않습니다. 시점 $t$에서는 직전 완료된 $t-1$ 마크 가격을 인과적으로 전방 채움(Forward-Fill)하여 참조합니다.
* **8시간 펀딩비 정산 메커니즘**: 펀딩비는 매일 00:00, 08:00, 16:00 UTC에 결제됩니다. 신규 체결 이전의 직전 보유 수량(Pre-trade quantity)과 현재 마크 가격을 기준으로 정산액을 계산합니다:
  $$\text{펀딩비 정산액} = - (\text{펀딩 비율} \times \text{체결 전 보유 수량} \times \text{마크 가격})$$
* **인과적 베이지안 Kelly 갱신**: 미래 수익률을 알지 못하는 상태에서 사전 분포($\mu_0=0$, 가중치 730일)로부터 매일 직전 거래일까지 관측된 단위수익률로만 사후 분포를 갱신합니다.

---

## 3. 데이터 무결성 및 Fail-Closed 보호 장치

* **임의 보간 없는 Fail-Closed 원칙**: 시세 데이터에 결측치가 발생했을 때 임의의 수치로 선형 보간(Interpolation)하면 왜곡된 가상 수익률이 생성될 수 있으므로, 결손 발생 시 즉시 `DataIntegrityError`를 발생시키고 해당 사이클을 중단합니다.
* **Fill-Mark 괴리율 보호 밴드**: 프록시 체결 가격과 해당 시점의 마크 가격 간 로그 괴리가 5%(`FILL_MARK_PRICE_PROTECTION_BAND = 0.05`)를 초과할 경우 비정상 호가로 간주하고 이상 플래그를 작동합니다.
* **무손실 원자적 보존 관리(Pruning)**:
  * 재생성 가능한 시계열(`ohlcv/1h`, `markPriceKlines/1h`, `funding`)에 한해 최소 보존 기간을 강제합니다.
  * 프루닝 파일 갱신 시 임시 파일 생성 후 원자적으로 대체(`tmp.replace(dest)`)하여 프로세스 비정상 종료 시에도 파일 손상이 발생하지 않도록 보장합니다.

