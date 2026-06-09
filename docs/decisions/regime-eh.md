---
title: Regime Audit & Hardening — Decision Records
domain: futures/strategy
type: adr
status: active
priority: high
ai_read_policy: when_related
related_paths:
  - src/domain/futures/strategy/regime_evaluation.py
  - src/domain/futures/strategy_runtime/bridge.py
  - tests/unit/domain/futures/strategy_runtime/test_bridge.py
  - tests/integration/execution/test_opt_main_futures_bypass.py
last_verified: 2026-06-09
---

## [2026-06-09] Regime Hardening & Runtime Test Alignment
- **Delta:** 
  1. `ml` -> `alo` phase rename 반영: `test_run_config.py` 및 `test_opt_main_futures_strategy_mode.py` 수정.
  2. `test_ml_alignment.py`에서 `downside_penalty` scaling 식을 `cfg.downside_penalty *`로 정합성 일치.
  3. `test_bridge.py` 내 dynamic path mock 대상을 `candidate_workflow`에서 `candidate_gate/edge`로 수정, ensemble 컬럼(`archetype`, `entry_regime_code`) 및 diagnostics null fields 보정.
  4. `test_opt_main_futures_bypass.py` 내 `_run_regime_evaluation_stage` dynamic mock 처리 추가 및 strict-type linting 에러 해소.
  5. `regime_evaluation.py`의 `newey_west_tstat` 내 std/variance가 0일 때 발생하는 `rho` NaN 현상에 대한 defensive guard `rho = 0.0 if math.isnan(rho_raw) else ...` 적용.
- **Rationale:** 
  최근 regime hardening 리팩토링 커밋(`bf78f76`, `9e46115` 등)으로 발생한 application/domain/integration 10개 테스트 오류와 lint 경고를 모두 해결하고, runtime-bridge/optimization bypass 파이프라인의 mock 구조를 실제 모듈 분할에 맞추어 일치시킴.
- **Edge Cases/Trade-offs:** 
  `newey_west_tstat`에서 sample variance가 0인 극단적인 경우(모든 리턴이 동일 상수일 때 등) naive NaN이 전파되어 metric logic이 마비되는 현상을 방지함.
