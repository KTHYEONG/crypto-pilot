# MHS Horizon Diagnostic — Latest Result

- **Document Date**: 2026-08-12 (13차, `OHLCV_IMMEDIATE_TAKER` 결손 가드 수정 후 재검증)
- **Domain**: Research / MHS (Multi-Horizon Market State)
- **Run Metadata**: `start=2021-01-01`, `end=2025-12-31`, `execution_timeframe=5m`, `execution_universe_size=30`, `eligible_symbols=446`, `run_elapsed_seconds=581.7`
- **CLI**: `research run portfolio mhs-horizon-diagnostic` (기본 플래그, `--fold-safe-horizon`/`--discovery-gate` 미사용)
- **Source**: [`mhs_horizon_diagnostic.json`](file:///home/kth/crypto-pilot/docs/results/mhs_horizon_diagnostic.json)
- **Research GO 판정 기준**: `daily autocorr-adjusted Sharpe >= 0.6` (primary) AND `stress Sharpe > 0`, 3-fold anchored 전부 통과
- **직전 결과와 비교 목적 상이**: 12차는 `--fold-safe-horizon --discovery-gate` 부가 진단 플래그로 실행됐고, `slow_momentum`/`blend`의 `fold0(2023)/fold1(2024)` 전체가 `OHLCV_IMMEDIATE_TAKER` 체결가 미검증 결손(`docs/feedback.md` 2026-08-12 기록) 오진 때문에 `CAPITAL_INVARIANT_BREACH`로 조기 중단된 채 기록된 수치다. 이번 13차는 [`mhs_immediate_taker_fill_guard`](file:///home/kth/crypto-pilot/docs/specs/mhs_immediate_taker_fill_guard.md) 수정 이후 동일 커맨드를 완주 재실행한 결과이며, 두 실행의 primary 숫자를 직접 비교하지 말 것 — 12차는 애초에 오염된 조기 중단 수치였다.

## 1. Primary metrics (`slow_momentum` == `blend`, `fast_reversal` blend 자본 0%)

| metric | value |
| :--- | ---: |
| `primary_autocorr_sharpe` | 0.1819 |
| `primary_naive_sharpe` | 0.0399 |
| `primary_max_drawdown` | -0.3922 |
| `primary_net_ann` | 0.0029 |
| `primary_geometric_cagr` | 0.0004 |
| `primary_annualized_turnover` | 5.1248 |
| `stress_naive_sharpe` (x3 cost) | -0.0844 |
| `pre_vol_target_reference_naive_sharpe` | 0.0306 |
| `blend.failure` | null |

## 2. Research GO gate

| field | value |
| :--- | :--- |
| `eligible` | `false` |
| `evaluated_folds` | 3 |
| `folds_passed` | 1 |
| `reason_codes` | `CAPITAL_INVARIANT_BREACH`, `PRIMARY_AUTOCORR_SHARPE_BELOW_0_6`, `RELEVANT_EXECUTION_DATA_GAP`, `STRESS_SHARPE_NOT_POSITIVE`, `UNSPECIFIED_POLICY` |

`CAPITAL_INVARIANT_BREACH`는 이번에도 `fast_reversal` 북 기인(blend 자본 0%, primary 판정과 무관)이지만 **원인이 12차와 다르다**: `pre-trade equity must be positive and finite (ts=2025-07-14 13:05 pre_trade_equity=-38.78)` — 음의 자기자본으로 인한 별개의 기존 이슈이며, `docs/feedback.md`가 다룬 `OHLCV_IMMEDIATE_TAKER` NaN 체결가 이슈와는 무관하다(미해결, 범위 밖).

`RELEVANT_EXECUTION_DATA_GAP`은 fold0에서 신규 관측됨 — `termination_counts.MISSING_DATA=76`. 이는 이번 수정이 의도한 정확한 동작: 과거엔 결손 체결가가 크래시로 이어졌으나, 이제는 해당 주문만 스킵되고 `MISSING_ACTIVE_ORDER_OHLCV` 데이터 갭으로 기록되어 fold가 완주된다.

## 3. Fold detail

| fold | validation_start | validation_end | `primary_autocorr_sharpe` | `primary_max_drawdown` | `stress_naive_sharpe` | `failures` | `slow_horizon_hours` | `slow_horizon_source` |
| ---: | :--- | :--- | ---: | ---: | ---: | :--- | ---: | :--- |
| 0 | 2023-01-08 | 2023-12-31 | -0.1461 | -0.2009 | -0.2093 | `PRIMARY_AUTOCORR_SHARPE_BELOW_0_6`, `RELEVANT_EXECUTION_DATA_GAP`, `STRESS_SHARPE_NOT_POSITIVE` | 168 | `frozen_default` |
| 1 | 2024-01-08 | 2024-12-31 | -0.5044 | -0.1825 | -0.3448 | `PRIMARY_AUTOCORR_SHARPE_BELOW_0_6`, `STRESS_SHARPE_NOT_POSITIVE` | 168 | `frozen_default` |
| 2 | 2025-01-08 | 2025-12-31 | 1.6054 | -0.4093 | 0.2646 | (none) | 168 | `frozen_default` |

3개 fold 전부 완주(`OHLCV_IMMEDIATE_TAKER` 결손 가드 수정 전에는 fold0/fold1이 `CAPITAL_INVARIANT_BREACH`로 완주 자체가 불가능했다). `--fold-safe-horizon` 미사용 실행이라 전부 `frozen_default`.

## 4. `mhs_immediate_taker_fill_guard` 수정 검증 (이번 차수의 핵심 목적)

- **수정 전**: `research run portfolio mhs-horizon-diagnostic` 자체가 `CAPITAL_INVARIANT_BREACH`로 미완주 (`docs/feedback.md` 2026-08-12).
- **수정**: `src/mhs/execution.py`의 `strategy_aware_execution_replay`/`_BoundExecutionReplayAccumulator.consume` 양쪽 `OHLCV_IMMEDIATE_TAKER` 분기에 STRICT/TOUCH/LADDERED와 동일한 `MISSING_ACTIVE_ORDER_OHLCV` 스킵 가드 이식(`docs/specs/mhs_immediate_taker_fill_guard.md`).
- **수정 후**: 동일 커맨드 완주, `status=COMPLETE`, 로그 전체에서 `CAPITAL_INVARIANT_BREACH`/`DataIntegrityError`/`Traceback` 관련 미포착 예외 0건. `termination_counts.MISSING_DATA=76`으로 결손이 크래시 대신 진단 데이터로 정상 흡수됨.
- **잔여 이슈**: `fast_reversal` 북의 별개 음의 자기자본 `CAPITAL_INVARIANT_BREACH`(§2)는 미해결 상태로 남음 — blend 자본 0%라 Research GO 판정에는 영향 없음.

## 5. 관찰

- Primary Sharpe(0.18)는 여전히 GO 임계값(0.6) 대비 크게 미달, MDD -0.39, stress Sharpe 음수 — 신호 자체의 근본적 개선 없이는 GO 불가능한 상태는 12차와 방향성이 동일.
- fold0/fold1이 이번에 처음으로 실제 Sharpe 수치를 산출함(과거엔 크래시로 도달 불가) — fold0(-0.15)/fold1(-0.50) 모두 음수, fold2(1.61)만 양수로 fold 간 변동성이 매우 큼(`fold_concentration` 계열 실패 가능성, §2의 `UNSPECIFIED_POLICY`와 함께 확인 필요).
- fold-safe-horizon/discovery-gate를 이번엔 켜지 않았으므로 12차 §4~§6의 discovery 미채택 결론은 이번 실행으로 재확인되지 않음 — 필요 시 별도 재실행 필요.

## 6. 다음 스텝 후보

| 후보 | 상태 |
| :--- | :--- |
| `fast_reversal`의 독립적 `CAPITAL_INVARIANT_BREACH`(음의 자기자본, 2025-07-14) 근본 원인 조사 | 미착수 (blend 자본 0%라 Research GO엔 무관하지만 북 자체는 여전히 미완주) |
| fold0/fold1/fold2 간 Sharpe 부호 반전 원인(신호 vs 유니버스/기간별 레짐) 분리 검증 | 미착수 |
| `--fold-safe-horizon --discovery-gate` 포함 재실행으로 12차 결론(§4~6, no-op 확인) 재검증 | 미착수 |
| 2023/2024 방향성 손실을 고치는 신호 재설계 | 미탐색 |
