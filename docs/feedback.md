# 발견된 문제 기록

## 2026-08-12 — MHS `OHLCV_IMMEDIATE_TAKER` 체결가 미검증 (미해결)

- **증상**: `research run portfolio mhs-horizon-diagnostic`에서 fold0(2023 검증) 및 상위 books(fast_reversal/slow_momentum) 리플레이가 `CAPITAL_INVARIANT_BREACH`로 완주하지 못함. 예외 메시지: `symbol='FILUSDT' ts=2022-02-26 01:05:00 decision_price=18.90(유한) fill_price=nan` — `decision_price`는 유한한데 `fill_price`만 NaN.
- **오진(誤診)**: 직전 조사에서 `laddered_fill_schedule`(`OHLCV_LADDERED_PROXY` 실행 경로)의 미검증 종가 경계를 원인으로 지목해 수정(`docs/specs/mhs_laddered_fill_closes_gap_fix.md`, `ADR_20260812_MHS_LADDERED_FILL_CLOSES_GAP_FIX`)했으나, 재실행 결과 **동일 메시지가 글자 하나까지 동일하게 재현**됨 — 이 수정은 실제로 실행되지 않는 경로(`OHLCV_LADDERED_PROXY`는 `--ladder-diagnostic` opt-in 전용)를 고친 것이었음이 드러남. 코드 자체는 유효한 방어(테스트로 검증됨)지만 이번 크래시의 원인은 아니었다.
- **실제 원인(확인, 미수정)**: 프로덕션 경로(`_run_anchored_fold`, 상위 books 리플레이)가 실제로 쓰는 execution bound는 `OHLCV_IMMEDIATE_TAKER`. `src/mhs/execution.py`의 이 분기(`strategy_aware_execution_replay` L710 부근, `_BoundExecutionReplayAccumulator.consume` L1348 부근) 둘 다:
  ```python
  if execution_bound == "OHLCV_IMMEDIATE_TAKER":
      fill_pos = submit_pos
      fill_price = float(closes_values[fill_pos, col])   # 유한성 검증 없음
      ...
  ```
  STRICT/TOUCH/LADDERED 분기는 전부 `timeout_close`/`adverse`/`closes`를 체결 전 검증하지만, IMMEDIATE_TAKER 분기만 `closes_values[submit_pos, col]`를 검증 없이 곧바로 체결가로 사용한다. `submit_pos`가 결손 구간(예: FILUSDT의 2022-02-26 Binance Vision 아카이브 결손) 안에 들어가면 정확히 이 증상이 재현된다.
- **다음 조치**: `OHLCV_IMMEDIATE_TAKER` 분기에 기존 `MISSING_ACTIVE_ORDER_OHLCV` 스킵 패턴(다른 분기들과 동일한 방식)을 적용하는 후속 `/spec` 필요 — 아직 미착수.

### 교훈 (재발 방지)

- **원인 수정 전 반드시 실제 실행 경로를 먼저 확인할 것**: 코드에 유사한 취약점이 여러 실행 경로(bound)에 중복 존재할 때, CLI 기본 플래그로 실제 프로덕션이 어느 경로를 타는지 추적하지 않고 "그럴듯한" 경로부터 고치면 시간을 낭비하고 오진을 초래한다. 이번 건은 `--ladder-diagnostic`이 opt-in이라는 사실을 사전에 확인했다면 피할 수 있었다.
- **수정 후 실제 데이터로 재현·검증할 것**: 유닛 테스트 통과와 `/check` PASS는 "그 코드가 옳다"는 것만 증명하며 "그 코드가 실제 실패를 해결한다"는 것은 증명하지 않는다 — 반드시 원 실패를 재현했던 것과 동일한 실행으로 재확인해야 한다.
