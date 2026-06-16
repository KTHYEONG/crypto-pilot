---
title: Layer 3 Holdout Engineering History
domain: futures.strategy.tiered_workflow
type: adr
status: active
priority: critical
ai_read_policy: when_related
---

## 2026-06-16 L3 빈 holdout 구조적 수정 — IS+OOS 데이터 병합 (PART4)
- **Delta:** `pick_strategy_data_maps`가 `oos_data_maps`를 버리고 IS-only를 반환하던 동작을 IS+OOS `concat+sort+dedup` 병합으로 교체. `full_strategy_maps`를 쓰는 모든 호출부(bridge, END-coverage 필터, `align_data_maps`)가 자동으로 holdout_end까지 데이터를 보게 됨.
- **Rationale:** `aligned.datetimes`가 구조적으로 `holdout_start`에서 끝나, `_resolve_holdout_span`이 항상 `empty_holdout_window`를 raise — "intersection tail truncation(상장폐지 심볼)"이라는 기존 진단은 오진이었고, 실제 원인은 데이터 소스 자체가 IS-only였던 것.
- **Edge Cases:** `keep="first"`로 IS 우선 — 경계 timestamp 중복 시 미래(OOS) 행이 과거를 덮어쓰지 않음. 부작용은 `layer2-eh.md`의 "L2 AWF fold anchoring 복원" 항목 참조(같은 작업에서 발견된 L2 fold 붕괴 regression).

## 2026-06-16 L3 평가체계 lean 보강 (PART2) — Phase D silent fallback 제거 (PART3)
- **Delta:** L3 게이트를 `cagr<0` 단일조건에서 5단계 순차 게이트(`insufficient_trades`→`negative_return`→`sharpe_rel`→`mdd_rel`→`mdd_abs`)로 교체. `total_return`, `equity_multiple`, `sortino`, `n_trades`, `cvar95`, `avg_gross_exposure`를 `Layer3Result`에 추가(L2 헬퍼 재사용, 신규 수학 없음). `except Exception` 발생 시 legacy Phase D fallback으로 조용히 넘어가던 동작을 제거 — 즉시 `RunnerResult(exit_code=1, reason="tiered_pipeline_error:...")`로 실패.
- **Rationale:** L3는 "1회 백테스팅으로 실제 복리자산증식 성과 판단"이 목적이므로 L2(Optuna 검증)와 동일한 수준의 풍부한 진단 지표는 불필요하나, CAGR/MDD/Sharpe/MAR만으론 빈약 — 단일패스 복리(`equity_multiple`)와 거래량 하한이 누락되어 있었음. Phase D fallback은 legacy 경로로, holdout 실패를 가려 "조용한 오류"를 만드는 위험이 있어 제거.
- **Edge Cases:** `max_mdd_abs`(기본 0.35)는 baseline 자체가 붕괴한 경우를 방어하는 절대 캡. `min_trades`(기본 10)는 L3 자체 기준으로 L2의 30보다 완화(단일 holdout 윈도우 특성 고려).
