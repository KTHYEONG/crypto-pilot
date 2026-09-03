---
title: MHS Architecture - 07. Live Runtime & Production
domain: live-execution
type: architecture
status: active
priority: critical
ai_read_policy: always
related_paths:
  - src/mhs/live_runtime.py
  - src/mhs/live_strategy.py
  - src/mhs/live_signal_step.py
  - src/live/
change_triggers:
  - src/mhs/live_*.py
  - src/live/*.py
last_verified: 2026-09-03
---

# 07. 실시간 운영 런타임 및 프로덕션 배포 아키텍처

## 1. 개요 (Overview)
백테스트 및 오프라인 연구를 통과한 MHS 전략은 실거래(Live) 및 섀도우 트레이딩(Shadow Trading) 환경에서 무결성을 유지하며 동작해야 합니다.

MHS의 라이브 아키텍처는 오프라인에서 검증된 수학적 모델과 100% 동일한 로직을 실시간 이벤트 루프 상에서 실행하도록 보장하는 **불변 전략 파라미터 봉인(Strategy Parameter Seal)**과 **실시간 상태 원장 관리 체계**를 갖추고 있습니다.

---

## 2. 라이브 신호 생성 스텝 (Live Signal Step)

매 1시간 정각(00분 00초 UTC)에 스케줄러가 기동되면 [`src/mhs/live_signal_step.py`](file:///home/kth/crypto-pilot/src/mhs/live_signal_step.py)의 [`compute_live_signal_step`](file:///home/kth/crypto-pilot/src/mhs/live_signal_step.py)이 순차적으로 실행됩니다:

```text
[정각 00:00 UTC 스케줄러 트리거]
       │
       ▼ 1. 실시간 캔들 및 마크 가격 수집
최근 완료된 1시간 캔들 및 최신 Mark Price 캐시 동기화
       │
       ▼ 2. PIT 3단계 유니버스 확정
결손 심볼 배제 ──► 720h 유동성 중앙값 50% ──► Schmitt-Trigger Top-30 로스터 확정
       │
       ▼ 3. k=5 위원회 신호 및 트랜치 적응형 계산
Causal Trailing Autocorr 측정 ──► 휩소 시 3행 평활, 추세 시 Raw 신호 확정
       │
       ▼ 4. 포트폴리오 조립 및 레짐 제어
베타 직교화(Market Beta Neutral) + BTC 하락 크래시 틸트 + 전략 P&L 변동성 스케일 적용
       │
       ▼ 5. 포트폴리오 리밸런스 트리거 (RC-1)
추적 오차가 20% 이상일 때만 신규 목표 수량(Target Units) 발령
       │
       ▼ 6. 주문 엔진(Execution Router) 또는 섀도우 원장 전달
실주문 생성 (Phase 2/3) 또는 Shadow Ledger 이벤트 기록 (Phase 1)
```

---

## 3. 전략 파라미터 봉인 및 무결성 검증 (Strategy Seal & Artifacts)

라이브 런타임 시작 시 과거 연구 단계에서 승인된 파라미터가 오염되지 않았는지 암호학적 해시를 검증합니다 ([`src/mhs/live_strategy.py`](file:///home/kth/crypto-pilot/src/mhs/live_strategy.py)):

- **`strategy_params.json`**: 위원회 구성, 윈도우 크기, 리스크 포락선, 트리거 임계값 등 모든 파라미터의 JSON 스냅샷.
- **`strategy_bootstrap.parquet`**: 전략 초기 가동을 위한 사전 웜업(Warmup) 시계열 데이터.
- **봉인 검증 (Seal Verification)**:
  - 런타임 구동 시 `BOUND_FLAGS` 22개 핵심 설정의 SHA256 해시를 대조하여, 파라미터가 단 1바이트라도 변경된 경우 [`ArtifactSealError`](file:///home/kth/crypto-pilot/src/live/errors.py)를 발생시키고 실행을 중단합니다.

---

## 4. 실시간 상태 영속화 및 샤딩 스토리지 (State Persistence)

실거래 환경에서 전원 차단, 프로세스 재시작 등 돌발 장애가 발생하더라도 전략의 연속성을 보장하기 위해 모든 원장과 메트릭은 디스크에 실시간 분할 저장됩니다:

- **`live_fills/` (월별 샤드)**: 거래소 실제 체결 이력 (Timestamp, Symbol, Side, Qty, Price, Fee).
- **`live_portfolio_state/` (순환 샤드)**: 매 시간별 현금 잔고, 종목별 보유 수량, 총 자산(Equity).
- **`live_microstructure/` (월별 샤드)**: 실시간 호가 스프레드, 심도 데이터.
- **`live_execution_quality/` (월별 샤드)**: 주문 대비 실제 체결 슬리피지(bps).
- **`live_tax_ledger/` (월별 JSONL)**: 회계 정산 및 세무 증빙을 위한 실현 손익 영구 기록.

---

## 5. Shadow Trading Cycle 및 Reconcile 리스크 감시

실제 자금을 투입하기 전 가상 주문을 체결하고 실시간 데이터와의 정합성을 감시하는 섀도우 사이클이 상시 가동됩니다:

1. **상태 대조 (Reconciliation Engine)**:
   - 매 주기마다 거래소의 실제 지갑 잔고/미체결 포지션과 시스템 내부 원장의 `simulated_units`를 비교합니다.
   - 허용 오차 이상의 괴리 발생 시 즉시 경보(Alert)를 발송하고 주문 집행을 동결합니다.
2. **비상 킬스위치 (Kill-Switch)**:
   - 일일 손실 한도 초과, API 레이턴시 비정상 지연, 마크 가격과의 5% 이상 괴리 감지 시 모든 활성 주문을 즉시 취소하고 포지션을 축소하는 비상 차단 로직이 내장되어 있습니다.

---

## 6. 핵심 코드 진입점 (Key Code Reference)

| 역할 | 소스 파일 | 핵심 함수 및 클래스 |
|---|---|---|
| 라이브 전략 봉인 및 파라미터 | [src/mhs/live_strategy.py](file:///home/kth/crypto-pilot/src/mhs/live_strategy.py) | `LiveStrategy`, `BOUND_FLAGS` |
| 실시간 신호 산출 스텝 | [src/mhs/live_signal_step.py](file:///home/kth/crypto-pilot/src/mhs/live_signal_step.py) | `compute_live_signal_step` |
| 라이브 런타임 오케스트레이터 | [src/mhs/live_runtime.py](file:///home/kth/crypto-pilot/src/mhs/live_runtime.py) | `LiveMhsRuntime` |
| 실시간 체결 엔진 | [src/live/](file:///home/kth/crypto-pilot/src/live/) | 실시간 거래소 통신, 리스크 감시 및 주문 라우터 |
