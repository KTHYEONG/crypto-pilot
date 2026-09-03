---
title: MHS Architecture - 00. Overview and Philosophy
domain: research-mhs
type: architecture
status: active
priority: critical
ai_read_policy: always
related_paths:
  - src/mhs/contracts.py
  - src/mhs/params.py
  - src/mhs/pipeline/orchestrator.py
  - src/mhs/pipeline/runner.py
  - src/mhs/pipeline/context.py
  - src/mhs/execution/ledger.py
change_triggers:
  - src/mhs/*.py
  - src/mhs/pipeline/*.py
last_verified: 2026-09-03
---

# 00. Multi-Horizon Market State (MHS) 시스템 개요 및 철학

## 1. 개요 및 주 목적 (Overview & Core Purpose)

**Multi-Horizon Market State (MHS)**는 다양한 거래 보유 기간(Horizon)의 횡단면(Cross-Sectional) 알파 신호를 독립된 북(Book)과 경제적 신호 위원회(Committee)로 구축하고, 5분봉 단위의 실제 계약수량 기반 **모의 실행 원장([`simulated_inventory_ledger`](file:///home/kth/crypto-pilot/src/mhs/execution/ledger.py))**을 통해 전략의 순수한 알파 및 리스크 내성을 검증하는 **Phase 1 알파 연구 및 실행 파이프라인**입니다.

### 🎯 MHS의 핵심 설계 철학

1. **Target Weight 근사 PnL의 착시 완전 배제**:
   - 기존의 많은 퀀트 백테스트가 목표 비중(Target Weight) 변경 시점의 가격 차이만으로 가상 수익률을 산출함으로써, 실제 체결 지연, 체결 슬리피지, 펀딩비 결제, 호가 스프레드에 따른 심각한 성능 왜곡(Backtest Over-optimism)을 초래합니다.
   - MHS는 실제 주문 체결(Proxy Fill), 1시간 단위 마크-투-마켓(MTM) 평가, 8시간 펀딩비 정산, 테이커/메이커 수수료 및 슬리피지가 완전 통합된 현금 및 계약 수량 원장([`simulated_inventory_ledger`](file:///home/kth/crypto-pilot/src/mhs/execution/ledger.py))을 **단일 진실 원천(Single Source of Truth)**으로 사용하여 순수 PnL을 정밀 측정합니다.

2. **미래 편향(Look-ahead bias) 원천 차단 (Point-In-Time Integrity)**:
   - 매 매매 결정 시각(Point-In-Time, PIT)에서 인과적(Causal)으로 관측 가능한 과거 유동성 및 마크 가격만을 사용합니다.
   - 심볼 선정, 호라이즌 디스커버리, 위원회 가중치 산출 등 모든 통계 파라미터는 엄격히 Train 구간에서만 적합(Fit)되며, 검증 구간(Validation/OOS)으로의 정보 누출을 차단합니다.

3. **Kelly 및 동적 사이징의 의도적 배제 (Raw Alpha 검증)**:
   - MHS Phase 1 파이프라인에는 Fractional Kelly, 가변 레버리지 배율, 포트폴리오 사이징 로직이 **의도적으로 배제**되어 있습니다.
   - 신호 자체의 순수한 우위(Edge)가 검증되지 않은 상태에서 복잡한 자금 관리 로직을 얹어 백테스트 곡선을 포장하는 과적합(Overfitting)을 막고, **1.0x Gross 자본(자기자본 100%) 조건 하에서 전략의 순수한 알파 생존력**을 가감 없이 평가합니다.

---

## 2. 단계별 파이프라인 및 MHS의 위치 (Three-Phase Progression)

MHS는 암호화폐 퀀트 시스템의 3단계 상용화 로드맵 중 **Phase 1 연구 게이트**를 담당합니다.

```text
[Phase 1: Research GO] (MHS 핵심 영역) ──► [Phase 2: Execution GO] ──► [Phase 3: Pilot / Scale GO]
  • 1.0x Gross 자본 기준                   • L1/L2 오더북 심도 기반 체결       • Fractional Kelly & Sizing
  • Taker 체결 & 3배 비용 스트레스           • 실시간 주문 대기/체결 레이턴시     • 동적 레버리지 배분
  • Sharpe ≥ 0.6 및 자본 보존 검증          • 실제 슬리피지/마켓 임팩트 측정     • 실전 자금 투입 및 스케일업
```

- **Phase 1: Research GO (현재 MHS 단계)**:
  - 1.0x Gross 자본, Taker 체결 및 스프레드/비용 3배 스트레스 조건에서 순수 알파 신호가 Autocorrelation-adjusted Sharpe ≥ 0.6 및 자본 보존을 달성하는지 통계적으로 검증합니다.
- **Phase 2: Execution GO**:
  - 실제 L1/L2 오더북 데이터, 실시간 Limit/Taker 체결 딜레이 및 시장 충격(Market Impact)을 검증합니다.
- **Phase 3: Pilot GO / Scale GO**:
  - Growth Risk Envelope 기반 Fractional Kelly, 동적 레버리지 배분 모델을 적용하여 실전 자금을 집행합니다.

---

## 3. 전체 시스템 아키텍처 및 파이프라인 흐름 (System Pipeline Flow)

MHS는 신호 생성(1시간봉 격자)과 체결 리플레이(5분봉 격자)의 시간 해상도를 분리하여 계산 효율성과 체결 정밀도를 동시에 달성합니다.

```text
 1시간(1h) OHLCV + Funding + PIT Lifecycle
        │
        ├─ [01] 3단계 심볼 유니버스 선정 (Source Gap Guard ──► Liquid-Half 50% ──► Top-30 PIT Roster & Schmitt-Trigger)
        ├─ [02] 멀티 호라이즌 신호 (Fast Reversal 48h / Slow Momentum 72h~504h 앙상블)
        ├─ [03] 위원회(Committee) 및 레짐 적응형 트랜치 평활 (k=5 Committee, Whipsaw 트랜치 vs Trend raw)
        ├─ [04] 포트폴리오 조립 & 2중 레짐 제어 (Beta 직교화 + BTC 크래시 틸트 + P&L Vol 타겟팅 + Rebalance Trigger)
        │
        ▼ 결정 시각별 Top-30 PIT Execution Target Weights
        │
        ├─ [05] 5분(5m) OHLCV High/Low/Close ──► 체결 리플레이 (Immediate-Taker / Strict Limit / x3 Cost)
        ├─ [05] Causal Mark Price Cache ──────► MTM 평가 및 8h 펀딩비 실시간 정산
        └─ [05] Timestamped Fill Events ──────► Simulated Inventory Ledger (단일 진실 원천 원장)
                                                │
                                                ▼
                                   [06] 3-Fold Level 2 Anchored Purged Walk-Forward Cross-Validation
                                                │
                                                ▼
                                   [06] 9대 합성 스트레스 시나리오 및 Research GO 게이트 판정
                                                │
                                                ▼
                                   [07] 실시간 라이브 런타임 및 상태 영속화 (Production Deployment)
```

---

## 4. 계층형 아키텍처 계약 (Layering & Acyclic Composition Contract)

시스템의 복잡성을 통제하고 순환 참조(Circular Import)를 원천 방지하기 위해 엄격한 단방향 계층 구조를 강제합니다 (계약 테스트로 검증됨).

```text
컴포지션 루트 (src/mhs/diagnostic_run.py, src/mhs/pipeline/orchestrator.py)
        │ imports
        ▼
파이프라인 레이어 (src/mhs/pipeline: runner, context, config)
        │ drives
        ▼
스테이지 레이어 (src/mhs/pipeline/stages/*)
        │ consume via module object (no private from-imports)
        ▼
평가 및 도메인 로직 (src/mhs/evaluation/*, evidence, scaling, statistics, execution)
        │ use
        ▼
퀀트 프리미티브 및 데이터 계층 (src/quant/*, src/market_data/*)
```

- **하향 단방향 의존성**: 상위 계층은 하위 계층을 참조할 수 있지만, 하위 계층(예: `src/mhs/evaluation/`, `src/mhs/execution/`)은 절대로 상위 파이프라인 계층(`src/mhs/pipeline/`)을 import하지 않습니다.
- **컨텍스트 기반 통신**: 모든 파이프라인 스테이지는 공유 상태 객체인 [`PipelineContext`](file:///home/kth/crypto-pilot/src/mhs/pipeline/context.py)를 통해서만 입출력을 교환합니다.

---

## 5. MHS 아키텍처 문서 시리즈 맵 (Document Series Index)

MHS의 메인 로직은 다음의 8개 연속 문서로 체계화되어 있습니다:

- **[00. Overview](file:///home/kth/crypto-pilot/docs/architecture/00_overview.md)**: 시스템 개요, 3단계 로드맵, 설계 철학, 아키텍처 계층
- **[01. Universe](file:///home/kth/crypto-pilot/docs/architecture/01_universe.md)**: 1시간 패널 데이터, 3단계 PIT 유니버스 선정(Source Gap, Liquid-Half, Top-30 Schmitt-Trigger), 마크 가격 캐시
- **[02. Signals](file:///home/kth/crypto-pilot/docs/architecture/02_signals.md)**: Reversal vs Momentum, Discovery/Qualification 게이트, 호라이즌 동일가중 앙상블(RC-2), 트렌드 및 펀딩 슬리브
- **[03. Committee](file:///home/kth/crypto-pilot/docs/architecture/03_committee.md)**: k=5 경제적 위원회, 증거 기반 가중치, 자기상관 기반 레짐 적응형 트랜치 평활, Growth Risk Envelope
- **[04. Portfolio](file:///home/kth/crypto-pilot/docs/architecture/04_portfolio.md)**: 북 블렌드, RC-1 Rebalance Trigger, 베타 직교화(RC-4), BTC 크래시 틸트, 전략 P&L 변동성 타겟팅
- **[05. Execution](file:///home/kth/crypto-pilot/docs/architecture/05_execution.md)**: 5분봉 체결 프록시, `simulated_inventory_ledger` 원장 구조, 17차 연율화 버그 수정
- **[06. Validation](file:///home/kth/crypto-pilot/docs/architecture/06_validation.md)**: 3-Fold Anchored Purged Validation, 9대 스트레스 시나리오, 꼬리위험 분석, Research GO 게이트 기준
- **[07. Live](file:///home/kth/crypto-pilot/docs/architecture/07_live.md)**: 실시간 신호 스텝, 샤딩 상태 영속화, Shadow Cycle 및 Reconcile 리스크 감시

데이터 수집 파이프라인에 관한 독립 문서는 다음을 참고하십시오:
- **[Binance Data Architecture](file:///home/kth/crypto-pilot/docs/architecture/data/binance.md)**: 바이낸스 API 및 Vision S3 수집, 전체 지원 데이터 항목 및 스키마, 라이브 데이터 스트림 규격
