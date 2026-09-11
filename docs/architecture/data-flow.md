# 데이터 흐름 및 시계열 불변식 (Data Flow & Temporal Invariants)

## 1. 단계별 데이터 변환 파이프라인 (Data Transformation Flow)

원시 거래소 데이터가 시스템을 거쳐 연구 파이프라인과 실거래 원장으로 흘러가는 구체적인 데이터 흐름은 다음과 같습니다:

| 단계 (Stage) | 입력 데이터 (Input) | 처리 내용 (Processing) | 출력 데이터 (Output) | 메인 모듈 (Main Module) |
| :--- | :--- | :--- | :--- | :--- |
| **1. 마켓 데이터 인제스천** | Binance FAPI, Spot v3, Margin SAPI, Vision S3 아카이브 | REST 폴링, 비동기 웹소켓 수집, 컬럼형 Parquet 저장, SHA256 매니페스트 무결성 추적 | 정규화된 Parquet 시계열 (`ohlcv/1h`, `markPriceKlines/1h`, `funding`) | [`src/market_data/services/`](file:///home/kth/crypto-pilot/src/market_data/services/) |
| **2. Tail 증분 갱신 & 프루닝** | 기존 Parquet 파일, 최신 Binance REST API 델타 | 로컬 디스크 tail 타임스탬프 파악 후 `max(tail - 2h, now - lookback)` 구간만 증분 패치, 220일 초과 데이터 원자적 temp-replace 축소 | 갱신된 최신 Parquet (20초 이내), 150MB 한도 내로 프루닝된 스토리지 | [`src/live/data_refresh.py`](file:///home/kth/crypto-pilot/src/live/data_refresh.py), [`src/market_data/retention.py`](file:///home/kth/crypto-pilot/src/market_data/retention.py) |
| **3. PIT 유니버스 필터링** | 1시간봉 거래대금(Quote Volume), 상장 이력 | 소스 갭 가드(결손 심볼 배제) $\rightarrow$ 직전 720시간 거래대금 중앙값 상위 50% $\rightarrow$ 상위 60위 진입/120위 탈락 히스테리시스 | `execution_mask` (불리언 매트릭스: 타임스탬프 x 심볼) | [`src/mhs/pipeline/stages/selection.py`](file:///home/kth/crypto-pilot/src/mhs/pipeline/stages/selection.py), [`src/quant/universe/pit_universe.py`](file:///home/kth/crypto-pilot/src/quant/universe/pit_universe.py) |
| **4. 피처 및 알파 신호 산출** | 1시간봉 종가 및 테이커 매수대금, 8시간 펀딩비 | 멀티 호라이즌 롤링 수익률(48h~504h), 테이커 플로우 불균형(720h/168h), 고유 모멘텀(BTC 베타 잔차), 수익률 왜도 | 호라이즌별 원시 알파 신호 행렬 | [`src/quant/technical_experts/cross_sectional.py`](file:///home/kth/crypto-pilot/src/quant/technical_experts/cross_sectional.py), [`src/mhs/books.py`](file:///home/kth/crypto-pilot/src/mhs/books.py) |
| **5. 위원회 및 캐리 결합** | 5개 피처 행렬, 펀딩 캐리 슬리브 (30% 비중) | 부호 안전 비용 분해, Train-only 증거 기반 가중치 부여, Causal Lag-1 자기상관 적응형 트랜치 평활 | `committee_target_weights` (달러 중립, Gross 정규화) | [`src/mhs/committee.py`](file:///home/kth/crypto-pilot/src/mhs/committee.py), [`src/mhs/pipeline/stages/committee.py`](file:///home/kth/crypto-pilot/src/mhs/pipeline/stages/committee.py) |
| **6. 포트폴리오 및 레짐 리스크** | 위원회 가중치, 롤링 BTC 베타 시계열, 전략 누적 수익률 | 720바 시장 베타 직교화, BTC 급락 크래시 틸트, 21일 전략 P&L 실현 변동성 타겟팅 켈리 사이징 | 레버리지 상한이 적용된 최종 목표 비중 `target_weights` | [`src/mhs/regime.py`](file:///home/kth/crypto-pilot/src/mhs/regime.py), [`src/mhs/scaling.py`](file:///home/kth/crypto-pilot/src/mhs/scaling.py) |
| **7. 3분봉 체결 원장 리플레이** | 최종 목표 비중, 3분봉 OHLCV, 1시간봉 마크 가격, 8시간 펀딩비 | 엄격한 회계 순서: MTM 평가 $\rightarrow$ 8시간 펀딩비 정산 $\rightarrow$ 3분봉 즉시 테이커 체결 $\rightarrow$ 3-tier 수수료 차감 $\rightarrow$ 현금/재고 갱신 | `SimulatedInventoryLedgerResult` (자산 곡선, 회전율, 체결 내역, 순현금) | [`src/mhs/execution/ledger.py`](file:///home/kth/crypto-pilot/src/mhs/execution/ledger.py), [`src/mhs/execution/strategy_replay.py`](file:///home/kth/crypto-pilot/src/mhs/execution/strategy_replay.py) |
| **8. 통계적 검증 및 게이트** | 완성된 모의 실행 원장 시계열 | 168시간 엠바고 16-Fold Anchored Purged Walk-Forward CV, 104개 탐색 경로 보정 Deflated Sharpe Ratio, 9대 스트레스 시나리오 | `MhsHorizonDiagnosticReport`, Research GO 적격 판정 | [`src/mhs/evidence.py`](file:///home/kth/crypto-pilot/src/mhs/evidence.py), [`src/mhs/pipeline/stages/fold.py`](file:///home/kth/crypto-pilot/src/mhs/pipeline/stages/fold.py) |
| **9. 라이브 런타임 인계** | 불변 봉인 파라미터, 부트스트랩 웜업 꼬리 | 00:00 UTC 크론: 인프로세스 tail 갱신 $\rightarrow$ 실시간 신호 스텝 $\rightarrow$ 페이퍼/실거래 주문 생성 $\rightarrow$ 거래소 잔고 대조 | 디스크 상태 파일 영속화 (`live_fills`, `live_portfolio_state`, `live_tax_ledger`) | [`src/live/scheduler.py`](file:///home/kth/crypto-pilot/src/live/scheduler.py), [`src/mhs/live_signal_step.py`](file:///home/kth/crypto-pilot/src/mhs/live_signal_step.py) |

---

## 2. 타임스탬프 규격 및 시계열 인과성 불변식 (Temporal Invariants)

```text
이벤트 시점 t-1                     결정 시점 t (00:00 UTC)                  체결 시점 t + 3m
───────┬─────────────────────────────────────┬────────────────────────────────────┬──────────►
       │ 1h 캔들 t-1 완료 (Close)             │ 신호 및 목표 비중(Target Weights) 산출 │ 3분봉(3m) 캔들 완료:
       │ 마크 가격 (Mark Price) 확정          │ 기존 보유 재고와 대조하여 주문 수량 결정│ 프록시 체결 (Proxy Fill) 집행
       │ 8시간 펀딩비 결제 (0/8/16 UTC)      │ 주문 의도 (Order Intent) 발령        │ 수수료 차감 및 MTM 원장 반영
```

### 1) 이벤트 타임(Event Time)과 처리 시점(Processing Time)
* **이벤트 타임**: 모든 캔들의 타임스탬프는 해당 캔들의 **시작 시각(Open Timestamp, 밀리초 UTC)**을 의미합니다. 즉, `2024-01-01 00:00:00` 1시간봉은 `01:00:00`에 완성(Close)됩니다.
* **결정 시점 $t$**: 전략의 매매 결정은 캔들이 완성된 직후에만 집행됩니다. 시점 $t$에서 생성된 신호는 엄격히 $t$ 시점 이전에 완료된 데이터만을 참조합니다.
* **체결 시점 $t + \Delta$**: 백테스트에서 주문은 결정 시점 바로 다음의 3분봉(`3m`)에서 체결됩니다. $t + 3\text{m}$에 체결된 포지션은 $t$부터 $t + 3\text{m}$ 사이의 가격 변동 손익을 취할 수 없습니다.

### 2) 타임존 및 달력 기준
* **엄격한 UTC 강제**: 모든 타임스탬프, 데이터프레임 인덱스, 로그는 타임존이 명시된 UTC(`tz="UTC"`)여야 합니다. Naive 타임스탬프는 [`DataIntegrityError`](file:///home/kth/crypto-pilot/src/common/errors.py)로 즉시 거부됩니다.
* **암호화폐 24/7/365 연속 그리드**: 시장 휴장일이나 주말 단절이 존재하지 않습니다. 결측치는 거래소 장애나 시세 누락을 의미합니다.

### 3) Point-in-Time (PIT) 인과성 규칙
* **생존 편향 배제**: 유니버스 선정은 매 시간 $t$ 시점에서 과거 720시간(30일) 동안 관측된 거래대금만을 기준으로 평가합니다. 미래의 상장 유지 여부나 누적 수익률 정보가 유입되지 않도록 통제합니다.
* **마크 가격 Causal Forward-Fill**: 마크 가격(`markPriceKlines/1h`)은 MTM 포트폴리오 평가와 청산 위험 판정에만 사용되며, 알파 신호 스코어링에는 개입하지 않습니다. 시점 $t$에서는 직전 완료된 $t-1$ 마크 가격을 인과적으로 전방 채움(Forward-Fill)하여 참조합니다.
* **8시간 펀딩비 정산 메커니즘**: 펀딩비는 매일 00:00, 08:00, 16:00 UTC에 결제됩니다. 신규 체결 이전의 직전 보유 수량(Pre-trade quantity)과 현재 마크 가격을 기준으로 정산액을 계산합니다:
  $$\text{펀딩비 손익} = - (\text{펀딩 비율} \times \text{체결 전 보유 수량} \times \text{마크 가격})$$
* **시계열 엠바고 및 퍼징 격리**: 교차 검증 시 Train 셋과 Test 셋 사이에 168시간(7일)의 엠바고 구간을 두어, 168시간 모멘텀 신호의 자기상관이 테스트 구간으로 누출(Leakage)되는 것을 방지합니다.

---

## 3. 데이터 무결성 및 Fail-Closed 보호 장치

* **임의 보간 없는 Fail-Closed 원칙**: 시세 데이터에 결측치가 발생했을 때 임의의 수치로 선형 보간(Interpolation)하면 왜곡된 가상 수익률이 생성될 수 있으므로, 결손 발생 시 즉시 [`DataIntegrityError`](file:///home/kth/crypto-pilot/src/common/errors.py)를 발생시키고 해당 사이클을 중단합니다.
* **Fill-Mark 괴리율 보호 밴드**: 프록시 체결 가격과 해당 시점의 마크 가격 간 로그 괴리가 5%(`FILL_MARK_PRICE_PROTECTION_BAND = 0.05`)를 초과할 경우 비정상 호가로 간주하고 이상 플래그를 작동합니다.
* **무손실 원자적 프루닝(Pruning)**:
  * 재생성 가능한 시계열(`ohlcv/1h`, `markPriceKlines/1h`, `funding`)에 한해 최소 보존 기간(`retention_days >= SIGNAL_PANEL_WINDOW_DAYS + 30`)을 강제합니다.
  * 프루닝 파일 갱신 시 임시 파일 생성 후 원자적으로 대체(`tmp.replace(dest)`)하여 프로세스 비정상 종료 시에도 파일 손상이 발생하지 않도록 보장합니다.
