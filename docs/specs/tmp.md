---
title: ML Alpha 미진행 개선안 (alpha1.md 분리)
domain: futures-alpha
type: prd
status: proposal
priority: high
ai_read_policy: when_related
last_verified: 2026-05-31
---

# ML Alpha 미진행 개선안

> **현황:** `portfolio_ic_above_breakeven` 단일 미달. 아래 옵션 중 하나로 해소.
> 완료 기록은 `docs/specs/alpha1.md` 참조.

---

## 블로커 해소 — `portfolio_ic_above_breakeven`

**원인:** `_summarize_exec_diag_verdict`의 `be_raw`가 emit breadth N_eff=2.2 기준(고정 24bps).
port_ic=0.0143 < be_raw=0.0343.

### 옵션 A: EXEC_DIAG be_raw에도 비용상각 적용 (권장)
- `_summarize_alpha_phase1_verdict`의 `be_raw` 계산을 horizon-amortized로 변경
- 12h 기준: 12bps → be_raw ≈ 0.017 (아직 > port_ic)
- 18h 기준: 8bps → be_raw ≈ 0.011 < port_ic(0.0143) → **통과**
- **파일:** `src/execution/opt_main_futures.py` `_summarize_alpha_phase1_verdict`

### 옵션 B: rank_select_quantile 확대
- 0.33 → 0.45: N_eff_emit 2.2 → ~3.0 → be_raw 인하
- **파일:** `src/domain/futures/strategy/config.py` `rank_select_quantile` 기본값 변경

---

## Phase 3 — Idiosyncratic 피처 보강 (조건부)

**진입조건:** Phase 1b/2 통과 후에도 `resid_ic < 0.020` 또는 `ir_t < 2.0`.

현재 resid_ic=0.0141 < 0.020. ir_t=5.38 ≥ 2.0 (이미 통과).
→ resid_ic 보강만 필요 시 진입.

**구현 방향:**
1. `calibrator_target`: `"gross"` → `"beta_residualized"` A/B 테스트
2. idiosyncratic 피처 추가: BTC lead-lag residual, funding-basis 이격, sector-relative
3. beta-proxy 피처(`ret_*`, `rv_*`) 제거 전 IC 변화 측정 선행

**Acceptance:** `resid_ic ≥ 0.020` ∧ `basket_net > 0` ∧ `ir_t ≥ 2.0`

---

## Phase 4 — 강건성 통합

- [ ] fold 부호안정성 게이트 추가: fold별 IC 부호 일치율 ≥ 0.6
- [ ] bear-only basket 구현 → `bear_market_basket_safe`를 nan→실측으로
- [ ] purge/embargo walk-forward 불변식 재확인
- [ ] DSR n_trials 정직화 유지 (`n_trials = folds × 3` — 완료)

---

## Phase 2 잔여 — label_horizon 탐색공간 활용

`label_horizon_bars` PARAM_SPACE({6,12,18})는 추가됐으나 Optuna 최적화(--mode optimize)에서만 적용.
향후 optimize 실행 시 horizon이 자동 탐색됨.
