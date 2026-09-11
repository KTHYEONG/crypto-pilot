# 서브시스템 컴포넌트 명세 (Subsystem Components)

## 1. 마켓 데이터 및 스토리지 (Market Data & Storage)

### MarketDataService & RetentionService
**책임 (Responsibility)**
거래소 엔드포인트 및 Vision 아카이브로부터 다중 해상도 시세(1h/3m OHLCV, 8h 펀딩비, 1h 마크 가격, 청산 스트림)를 수집·캐싱하고, 디스크 tail 기반 증분 갱신과 무손실 원자적 프루닝을 수행합니다. 저사양 클라우드 환경에서 디스크 용량을 150MB 이내로 유지합니다.

**입력 (Input)**
Binance REST API (`/fapi/v1`, `/api/v3`, `/sapi/v1`), Binance Vision S3 아카이브, Binance 실시간 청산 WebSocket.

**출력 (Output)**
`data/futures/{ohlcv,markPriceKlines,funding}/` 하위의 zstd 압축 컬럼형 Parquet 파일, `RefreshReport`.

**의존성 (Dependencies)**
`ccxt`, `pyarrow`, `tenacity`, `pydantic`.

**핵심 구현 (Key Implementation)**
- [`src/market_data/services/collection.py`](file:///home/kth/crypto-pilot/src/market_data/services/collection.py)
- [`src/market_data/services/futures_collection.py:DataCollector`](file:///home/kth/crypto-pilot/src/market_data/services/futures_collection.py)
- [`src/market_data/retention.py:prune_market_data`](file:///home/kth/crypto-pilot/src/market_data/retention.py)
- [`src/live/data_refresh.py:refresh_live_market_data`](file:///home/kth/crypto-pilot/src/live/data_refresh.py)

---

## 2. 유니버스 필터링 및 선정 (Universe Selection)

### PITUniverseSelector
**책임 (Responsibility)**
미래 데이터를 참조하지 않는 3단계 Point-In-Time (PIT) 인과적 필터를 적용하여 실행 유니버스를 동적으로 확정합니다. 생존 편향을 차단하고 Schmitt-Trigger 히스테리시스를 통해 불필요한 포지션 진동 및 턴오버를 억제합니다.

**입력 (Input)**
과거 1시간봉 거래대금(Quote Volume) 시계열, 심볼 상장 이력, 결손 심볼 목록(`MHS_SOURCE_GAP_EXCLUDED_SYMBOLS`).

**출력 (Output)**
`(타임스탬프, 심볼)` 차원의 유효 실행 불리언 매트릭스 (`execution_mask`).

**의존성 (Dependencies)**
`src.common.errors.UniverseIntegrityError`, `src.mhs.params`.

**핵심 구현 (Key Implementation)**
- [`src/mhs/pipeline/stages/selection.py:select_horizons`](file:///home/kth/crypto-pilot/src/mhs/pipeline/stages/selection.py)
- [`src/quant/universe/pit_universe.py:schmitt_trigger_universe`](file:///home/kth/crypto-pilot/src/quant/universe/pit_universe.py)

---

## 3. 신호 추출 및 위원회 조립 (Signal & Committee Assembly)

### MultiHorizonCommitteeEngine
**책임 (Responsibility)**
단기 반등(48h) 및 장기 모멘텀(72h~504h 앙상블) 북을 구축하고, $k=5$ `flow_momentum` 경제적 신호 위원회로 결합합니다. 30% 비중의 펀딩 캐리 슬리브를 혼합하며, Causal Trailing Lag-1 자기상관에 따라 3행 트랜치 평활 여부를 적응형으로 제어합니다.

**입력 (Input)**
1시간봉 종가 패널, 테이커 매수대금 패널, 8시간 펀딩비 패널, `execution_mask`.

**출력 (Output)**
위원회 원시 및 정규화 목표 가중치 행렬 (`committee_target_weights`).

**의존성 (Dependencies)**
`src.quant.technical_experts.cross_sectional`, `src.mhs.params`.

**핵심 구현 (Key Implementation)**
- [`src/mhs/committee.py:build_flow_momentum_committee`](file:///home/kth/crypto-pilot/src/mhs/committee.py)
- [`src/mhs/committee.py:train_evidence_weights`](file:///home/kth/crypto-pilot/src/mhs/committee.py)
- [`src/mhs/pipeline/stages/committee.py:build_committee`](file:///home/kth/crypto-pilot/src/mhs/pipeline/stages/committee.py)

---

## 4. 포트폴리오 최적화 및 리스크 제어 (Portfolio Risk Control)

### PortfolioRiskController
**책임 (Responsibility)**
모멘텀 크래시 및 시장 시스템 위기로부터 자본을 보존합니다. 20% 포트폴리오 추적 오차 리밸런스 게이트를 적용하고, 720바 OLS 시장 베타(BTC) 직교화, BTC 급락 틸트, 그리고 전략 자체 21일 실현 변동성 타겟팅 기반 켈리 사이징을 집행합니다.

**입력 (Input)**
미스케일 위원회 가중치, 롤링 720바 BTC 수익률, 전략 과거 누적 자산 곡선.

**출력 (Output)**
레버리지 상한이 적용된 최종 리스크 스케일링 목표 비중 (`target_weights`).

**의존성 (Dependencies)**
`src.mhs.regime`, `src.mhs.scaling`, `scipy.stats`.

**핵심 구현 (Key Implementation)**
- [`src/mhs/regime.py:beta_neutralize_weights`](file:///home/kth/crypto-pilot/src/mhs/regime.py)
- [`src/mhs/regime.py:crash_regime_tilt_weights`](file:///home/kth/crypto-pilot/src/mhs/regime.py)
- [`src/mhs/scaling.py:pnl_volatility_target_scale`](file:///home/kth/crypto-pilot/src/mhs/scaling.py)
- [`src/mhs/books.py:portfolio_rebalance_trigger`](file:///home/kth/crypto-pilot/src/mhs/books.py)

---

## 5. 체결 시뮬레이션 및 재고 원장 (Execution Simulation & Ledger)

### SimulatedInventoryLedger
**책임 (Responsibility)**
포트폴리오 손익(PnL)을 체결 기반으로 일원화하여 산출하는 기준 원장 역할을 수행합니다. 3분봉 단위 즉시 테이커 체결 프록시를 구동하고, 3-tier 수수료(2.64~6.07 bps) 차감, 8시간 단위 펀딩비 실정산, 마크 가격 MTM 평가를 거쳐 엄격한 현금 및 계약 수량 원장을 유지합니다.

**입력 (Input)**
타임스탬프 정렬 모의 체결 내역, 3분봉 및 1시간봉 마크 가격 패널, 펀딩비 시계열, 초기 자본금.

**출력 (Output)**
연속 자산 곡선(Equity), 연율화 회전율, 체결 내역, 순현금 잔고가 포함된 `SimulatedInventoryLedgerResult`.

**의존성 (Dependencies)**
`src.common.errors.DataIntegrityError`, `src.mhs.execution.accumulator`.

**핵심 구현 (Key Implementation)**
- [`src/mhs/execution/ledger.py:simulated_inventory_ledger`](file:///home/kth/crypto-pilot/src/mhs/execution/ledger.py)
- [`src/mhs/execution/strategy_replay.py:strategy_aware_execution_replay`](file:///home/kth/crypto-pilot/src/mhs/execution/strategy_replay.py)
- [`src/mhs/execution/accumulator.py:ExecutionAccumulator`](file:///home/kth/crypto-pilot/src/mhs/execution/accumulator.py)

---

## 6. 통계적 유의성 검증 및 리서치 게이트 (Validation & Gates)

### PurgedWalkForwardValidator
**책임 (Responsibility)**
시계열 데이터 누출을 방지하고 다중 가설 검정 과적합을 통계적으로 검증합니다. 168시간(1주) 엠바고가 강제된 16-Fold Anchored Purged Walk-Forward CV를 실행하고, 104개 탐색 시도 경로의 분산/첨도를 보정한 Deflated Sharpe Ratio (DSR) 및 9대 합성 스트레스 시나리오 내구성을 평가합니다.

**입력 (Input)**
리플레이된 포트폴리오 일별 손익 시계열, 시도된 파라미터 트라이얼 풀 히스토리.

**출력 (Output)**
`MhsHorizonDiagnosticReport`, 폴드별 OOS 통과 여부, DSR t-stat, 블록 부트스트랩 파산 확률.

**의존성 (Dependencies)**
`src.mhs.evidence`, `src.quant.evaluation.reliability`.

**핵심 구현 (Key Implementation)**
- [`src/mhs/evidence.py:phase_1_anchored_purged_folds`](file:///home/kth/crypto-pilot/src/mhs/evidence.py)
- [`src/mhs/evidence.py:deflated_sharpe_ratio`](file:///home/kth/crypto-pilot/src/mhs/evidence.py)
- [`src/mhs/pipeline/stages/fold.py:run_folds`](file:///home/kth/crypto-pilot/src/mhs/pipeline/stages/fold.py)

---

## 7. 24/7 라이브 자동매매 런타임 (Live Runtime & Daemon)

### LiveDaemonScheduler & SignalStep
**책임 (Responsibility)**
매시간 정각(`00:00 UTC`)에 무인으로 기동되어 시세 증분 갱신, 전략 파라미터 암호학적 봉인(Seal) 검증, 실시간 신호 산출, 섀도우/페이퍼 주문 집행, 거래소 지갑 잔고 대조, 그리고 영구 세무 장부(`live_tax_ledger`) 기록을 총괄합니다.

**입력 (Input)**
암호화된 봉인 파라미터 파일(`strategy_params.json.enc`, `strategy_bootstrap.parquet.enc`), 실시간 계좌 잔고, 최신 캔들 시세.

**출력 (Output)**
실거래 주문(또는 섀도우 모의 체결), `data/state/` 하위 JSONL 상태 파일, 데몬 하트비트.

**의존성 (Dependencies)**
`src.live.settings:LiveSettings`, `src.live.runner`, `src.live.executor`.

**핵심 구현 (Key Implementation)**
- [`src/live/scheduler.py:run_daemon`](file:///home/kth/crypto-pilot/src/live/scheduler.py)
- [`src/mhs/live_signal_step.py:compute_live_signal_step`](file:///home/kth/crypto-pilot/src/mhs/live_signal_step.py)
- [`src/mhs/live_strategy.py:load_strategy_params`](file:///home/kth/crypto-pilot/src/mhs/live_strategy.py)
- [`src/live/runner.py:run_shadow_cycle`](file:///home/kth/crypto-pilot/src/live/runner.py)

---

## 8. 경보 및 데브옵스 인프라 (Alerting & DevOps)

### AlertingEngine & SecurityPipeline
**책임 (Responsibility)**
제로 트러스트 보안 체계와 다채널 장애 통지를 담당합니다. 외부 공인 포트 노출 없이 Tailscale 사설망을 경유하여 배포를 수행하고, Mozilla SOPS + Age로 환경변수를 복호화하며, 프로세스 연속 중단이나 시세 지연 발생 시 이메일 및 웹훅으로 비상 알림을 분배합니다.

**입력 (Input)**
연속 사이클 HALT 이벤트, 예외 에러, 데이터 지연(Staleness) 경고.

**출력 (Output)**
Gmail SMTP STARTTLS 전송, HTTP Webhook POST 디스패치.

**의존성 (Dependencies)**
`smtplib`, `urllib.request`, `src.live.settings`.

**핵심 구현 (Key Implementation)**
- [`src/live/alerting.py:post_alert`](file:///home/kth/crypto-pilot/src/live/alerting.py)
- [`src/live/alerting.py:send_email_alert`](file:///home/kth/crypto-pilot/src/live/alerting.py)
- [`.github/workflows/deploy.yml`](file:///home/kth/crypto-pilot/.github/workflows/deploy.yml)
