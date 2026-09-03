# Crypto-Pilot 시스템 아키텍처 문서 (Architecture Documentation)

본 디렉토리는 **Multi-Horizon Market State (MHS)** 전략 연구 및 실행 파이프라인과 **바이낸스 데이터 인프라**에 대한 공식 기술 아키텍처 사양을 제공합니다.

---

## 📚 1. MHS 메인 로직 문서 시리즈 (MHS Architecture Series)

MHS의 전체 파이프라인은 신호 생성부터 체결 리플레이, 통계적 교차 검증, 실거래 배포까지의 흐름에 따라 순서대로 번호화되어 있습니다:

| 번호 | 문서명 | 주요 내용 | 관련 소스 코드 |
|:---:|---|---|---|
| **00** | [00. Overview](file:///home/kth/crypto-pilot/docs/architecture/00_overview.md) | • 목표 가중치 PnL 착시 배제 및 원장 단일 진실원천 철학<br>• Phase 1 Raw Alpha 검증 (1.0x Gross, Kelly 배제)<br>• 3단계 상용화 로드맵 및 비순환 계층 계약(Layering) | [`src/mhs/contracts.py`](file:///home/kth/crypto-pilot/src/mhs/contracts.py)<br>[`src/mhs/pipeline/`](file:///home/kth/crypto-pilot/src/mhs/pipeline/) |
| **01** | [01. Universe](file:///home/kth/crypto-pilot/docs/architecture/01_universe.md) | • 1시간 패널 데이터 인제스천 및 RAM 예산 가드<br>• 3단계 PIT 유니버스 선정 (Source Gap, Liquid-Half 50%, Top-30 Schmitt-Trigger)<br>• Causal Mark Price 캐시 및 Fail-closed 무결성 | [`src/mhs/panel.py`](file:///home/kth/crypto-pilot/src/mhs/panel.py)<br>[`src/mhs/pipeline/stages/selection.py`](file:///home/kth/crypto-pilot/src/mhs/pipeline/stages/selection.py) |
| **02** | [02. Signals](file:///home/kth/crypto-pilot/docs/architecture/02_signals.md) | • Fast Reversal (48h) vs Slow Momentum (72h~504h)<br>• 연도별 Worst-Year Discovery & Qualification 게이트<br>• 19개 호라이즌 동일가중 앙상블(RC-2) 및 보조 슬리브 | [`src/mhs/discovery.py`](file:///home/kth/crypto-pilot/src/mhs/discovery.py)<br>[`src/mhs/books.py`](file:///home/kth/crypto-pilot/src/mhs/books.py) |
| **03** | [03. Committee](file:///home/kth/crypto-pilot/docs/architecture/03_committee.md) | • 경제적 신호 위원회 (k=5 Committee, flow_momentum)<br>• 부호 안전 비용 분해 및 Train-only 증거 기반 가중치<br>• 자기상관 기반 레짐 적응형 트랜치 평활 (ADR-20260817)<br>• Growth Risk Envelope 체계 | [`src/mhs/committee.py`](file:///home/kth/crypto-pilot/src/mhs/committee.py)<br>[`src/mhs/pipeline/stages/committee.py`](file:///home/kth/crypto-pilot/src/mhs/pipeline/stages/committee.py) |
| **04** | [04. Portfolio](file:///home/kth/crypto-pilot/docs/architecture/04_portfolio.md) | • 포트폴리오 추적 오차 20% 리밸런스 트리거 (RC-1)<br>• 720바 롤링 OLS 시장 베타 직교화 (RC-4)<br>• BTC 참조 자산 크래시 틸트 및 전략 P&L 변동성 타겟팅<br>• 자본 스케일링 오버레이 계약 | [`src/mhs/regime.py`](file:///home/kth/crypto-pilot/src/mhs/regime.py)<br>[`src/mhs/scaling.py`](file:///home/kth/crypto-pilot/src/mhs/scaling.py) |
| **05** | [05. Execution](file:///home/kth/crypto-pilot/docs/architecture/05_execution.md) | • 5분봉 고해상도 체결 리플레이 (Immediate-Taker, Strict Limit, x3 Cost)<br>• Simulated Inventory Ledger 회계 처리 순서 및 단일 진실원천<br>• 17차 5분봉 연율화 버그 수정 및 CAGR 정상화 | [`src/mhs/execution/ledger.py`](file:///home/kth/crypto-pilot/src/mhs/execution/ledger.py)<br>[`src/mhs/execution/strategy_replay.py`](file:///home/kth/crypto-pilot/src/mhs/execution/strategy_replay.py) |
| **06** | [06. Validation](file:///home/kth/crypto-pilot/docs/architecture/06_validation.md) | • 3-Fold Anchored Purged Walk-Forward (168h Purge/Embargo)<br>• 9대 합성 스트레스 시나리오 (BTC 폭락, 상관계수 1, API 장애 등)<br>• 꼬리 민감도 및 2,000경로 블록 부트스트랩 배포 준비도<br>• 최신 실측 성능 (Sharpe 1.0792, 3/3 Fold 통과) | [`src/mhs/evidence.py`](file:///home/kth/crypto-pilot/src/mhs/evidence.py)<br>[`src/mhs/research_go.py`](file:///home/kth/crypto-pilot/src/mhs/research_go.py) |
| **07** | [07. Live](file:///home/kth/crypto-pilot/docs/architecture/07_live.md) | • 매 시간 정각 Live Signal Step 실행 라이프사이클<br>• 전략 파라미터 불변 봉인 (Seal) 및 암호학적 해시 검증<br>• 샤딩 상태 영속화 (Live Fills, Portfolio State, Tax Ledger)<br>• Shadow Cycle 및 Reconcile 리스크 감시 엔진 | [`src/mhs/live_strategy.py`](file:///home/kth/crypto-pilot/src/mhs/live_strategy.py)<br>[`src/mhs/live_signal_step.py`](file:///home/kth/crypto-pilot/src/mhs/live_signal_step.py)<br>[`src/mhs/live_runtime.py`](file:///home/kth/crypto-pilot/src/mhs/live_runtime.py) |

---

## 🌐 2. 데이터 인프라 분리 문서 (Data Infrastructure)

바이낸스 API 및 아카이브 데이터 수집과 실시간 데이터 스트림 관리에 관한 상세 규격은 다음 별도 문서에 분리되어 체계적으로 관리됩니다:

- **[바이낸스 데이터 아키텍처](file:///home/kth/crypto-pilot/docs/architecture/data/binance.md)**:
  - **수집 지원 데이터 항목**: Futures OHLCV (1m, 5m, 1h), Spot OHLCV (1h), Funding Rate (8h), Futures Metrics (5m), Indicator Mark/Index Klines (1h), Orderbook Depth (5-Level), Margin Borrow Rate (Hourly/Daily).
  - **데이터 소스**: FAPI REST, Spot API v3, Margin SAPI, Binance Vision S3 아카이브.
  - **실시간 데이터 스트림 (Live Data Streams)**: 1h OHLCV, 1h MarkPrice, Funding, Live Fills, Live Microstructure, Live Portfolio State, Live Orderbook, Live Tax Ledger의 파티션 및 보관 주기(Retention) 정책.
  - **데이터 무결성 불변식**: Point-In-Time (PIT) 인과성, Vision Metrics 5분 릴리스 지연 강제, 11개 필드 전체 추출(Full Field Extraction), 매니페스트 SHA256 추적 및 자가치유 캐시 병합.
