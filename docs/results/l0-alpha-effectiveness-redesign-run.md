# L0 Alpha Effectiveness Redesign — Real Run Verification

- Run date: 2026-07-07
- 관련 spec: `docs/specs/l0-alpha-effectiveness-redesign.md` (sync 시 제거됨, 이 문서가 실측 기록 SSOT)
- 비교 대상: `4h_1783384093`(수정 전) vs `4h_1783387872`(수정 후), 동일 커맨드/시드

## 실행 명령
```
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. timeout 900 uv run python src/execution/opt_main_futures.py \
  --phase l1 --sync skip --timeframe 4h --trials 1 --seed 42 \
  --alpha-foundry gate --alpha-foundry-total-l1-budget 30 --alpha-foundry-min-conviction-lcb-bps 5.0
```

## 발견 1 — `weak_rank_ic`가 유일한 "candidate" 등급을 "seed"로 강등

| | 수정 전 | 수정 후 |
|---|---|---|
| discovery_tier=candidate | 1건 (`mtf_breakout_retest:mtf_bor_20_4h`) | **0건** |
| discovery_tier=seed | 2건 | 3건 |
| discovery_tier=blocked | 33건 | 33건(불변) |
| weak_rank_ic 부여 건수 | (필드 없음) | 9/36 |

`mtf_bor_20_4h`(rank_ic=0.0078)가 `_rank_ic_soft_floor(n_events)` 미만이라 `weak_rank_ic` 획득 → soft_flags 비어있지 않으므로 `candidate`가 아닌 `seed`로 재분류. **현재 27개 family 전체에서 "진짜 예측력이 통계적으로 확인된" 등급은 0건**임이 시스템 자체에서 처음으로 명시적으로 드러남. L1 예산 배분은 유지(우선순위만 0.70배 감쇠), 승격 3건 자체는 불변.

## 발견 2 — gate 판정 완전 불변 확인(회귀 없음)

`gate_passed`/`discovery_tier`의 blocked 카운트(33건), L1 승격 3건(lsr_oi_regime_filter, mtf_breakout_retest, trend_pullback_continuation) 모두 수정 전후 동일 — Rule 1(gross/cost 로깅)과 Rule 2(soft flag)가 게이트 로직 자체를 전혀 건드리지 않았음을 실측으로 재확인.

## 발견 3 — `total_cost_bps` 단위 설계 결함 (spec 오류, 코드 정상)

실측 중 발견: `total_cost_bps`(예: 457568.27)가 `mean_gross_bps`(예: 67.26)와 자릿수가 맞지 않음. 원인 규명:
- `mean_gross_bps = total_gross / n_events` (건당 평균)
- `total_cost_bps = total_cost` (그대로, **합계**) — spec에서 단위를 맞추지 않은 설계 실수

검증: `total_cost_bps / n_events = mean_gross_bps - mean_net_bps`가 부동소수점 오차 내로 정확히 성립(예: 457568.27/15159=30.18, 67.26-30.18=37.08=mean_net_bps ✓) — **계산 로직은 정확**, 필드명/단위만 `mean_gross_bps`와 불일치.

**✅ 해결됨(2026-07-07 후속 수정)**: `total_cost_bps` → `mean_cost_bps`(= total_cost/n_events)로 교체. 재실측(`4h_1783391061`) 확인 — `mean_gross_bps`(예: 67.26) - `mean_cost_bps`(예: 30.18) = `mean_net_bps`(37.08)가 오차 2.8e-14 수준으로 정확히 성립, 세 필드 전부 건당 bps로 직접 비교 가능해짐. 214개 회귀 테스트 전체 통과, lint/mypy 클린.

## 결론

Rule 2(`weak_rank_ic`)가 의도한 목적(저품질 통계적 우연을 가시화)을 정확히 달성했고, 그 결과 "통과 후보조차 진짜 예측력 미검증"이라는 사실을 discovery_tier 레벨에서 최초로 공식 확인. Rule 1(gross/cost 로깅)의 단위 결함도 후속 수정으로 해소 완료.
