---
title: ML Alpha 실전투입 견고화 — Phase 0~2 완료 기록
domain: futures-alpha
type: prd
status: active
priority: critical
ai_read_policy: when_related
related_paths:
  - src/domain/futures/strategy/ml_builder.py
  - src/domain/futures/strategy/alpha_evaluation.py
  - src/execution/opt_main_futures.py
  - src/domain/futures/optimization/opt_config.py
last_verified: 2026-05-31
---

# ML Alpha 실전투입 견고화 — Phase 0~2 완료 기록

> **목표:** `ALPHA_PASS=FALSE` → `TRUE`. 게이트 정의 동결(FROZEN). **잔여 블로커: `portfolio_ic_above_breakeven` 단일 미달.**

---

## 1. 현재 게이트 상태

| 게이트 코드 | 의미 | 현재값 | 임계 | 상태 |
|------------|------|--------|------|------|
| `signal_skill_passes` | resid_ic > be_eff | 0.0141 > 0.0131 | gap=+0.001 | ✅ |
| `signal_t_stat_too_low` | t_stat_nw ≥ 3.0 | 3.62 | OK | ✅ |
| `bear_regime_ic_negative` | bear_ic ≥ 0 | 0.0240 | OK | ✅ |
| **`portfolio_ic_above_breakeven`** | port_ic > be_raw | 0.0143 < 0.0343 | FAIL | ❌ |
| `basket_net_positive` | basket_net > 0 ∧ ir_t ≥ 2 | +8.43bps / +5.38 | OK | ✅ |
| `signal_preserved_after_selection` | presv ≥ 0.5 | +1.01 | OK | ✅ |
| `multi_horizon_sweep_passes` | sweep ≥ 1/3 | 2/3 | OK | ✅ |
| `bear_market_basket_safe` | bear_basket ≥ 0 | nan(미구현) | OK | ✅ |

**잔여 블로커:** `portfolio_ic_above_breakeven` — port_ic(0.0143) < be_raw(0.0343). emit N_eff=2.2 기준 be_raw가 높아서 발생. `signal_skill_passes`(be_eff=0.0131 기준)는 이미 통과.

---

## 2. 완료된 작업 (2026-05-31)

### Phase 0′ — Selection 메커니즘 진단 ✅
- `diagnose_selection_monotonicity` (alpha_evaluation.py:994) + opt_main 로그 + 단위테스트 3건
- 실측: `mono_rho=1.00`, `beta_tilt=-0.293`(long=저베타/short=고베타), `top-bot=+35.5bps`
- **결론:** dense signal 단조성 완전. basket 손실 = rank_cs_neutral 별도 L/S 선택이 NET +35.5bps 파괴

### Phase 1 — Rank-Based Emission ✅ (판정 무영향 확인)
- `_emit_rank_sized_alpha` (ml_builder.py:226), StrategyMLConfig 3필드, opt_config 탐색공간 추가
- clip_lim=1.0 버그픽스, quality gate → OOS IC 기반(`_test_rank_ic`)으로 교체
- **교훈:** emit 패널은 G1·G2 판정에 미사용. basket은 scoreboard rank_cs_neutral 재구성 사용

### Phase 1b — NET 신호 Basket 선택 ✅
- opt_main_futures.py:845 rank_cs_neutral 블록: 별도 L/S → `rank_score_long - rank_score_short` 기준 단일 선택
- **결과:** basket ew_bps: -12.94 → +32.43, presv: -0.21 → +1.01, ir_t: -2.61 → +5.38

### Phase 2 — 비용-호라이즌 공동최적화 ✅
- `sweep_horizon_breakeven`에 `cost_map: dict[int, float] | None` 파라미터 추가
- opt_main sweep 호출: `_cost_map = {h: 24.0 / max(1, h // 6) for h in [6,12,18]}`
- `label_horizon_bars` PARAM_SPACE 추가 ({6,12,18} categorical)
- **결과:** SWEEP 0/3 → 2/3 (12h: breakeven 0.0234→0.0117 ✅, 18h: 0.0192→0.0064 ✅)

### 부수 개선 ✅
- 게이트 이름 전면 가독화 (`raw_ic_non_positive` → `portfolio_ic_below_raw_breakeven` 등 14종)
- 테스트: 607 passed (이전 574)

---

## 3. 잔여 블로커 — `portfolio_ic_above_breakeven`

```
port_ic = 0.0143  (rank_cs_neutral NET 신호 basket IC)
be_raw  = 0.0343  (24bps / (sigma_r × √N_eff_emit), N_eff_emit=2.2)
gap_raw = -0.0200  → FAIL
```

`signal_skill_passes`(be_eff=0.0131, corr-N_eff=15)는 통과. `portfolio_ic_above_breakeven`은 `_summarize_exec_diag_verdict`의 emit breadth 기반 별도 체크.

해소 경로 → `docs/specs/tmp.md`

---

## 4. 검증

```bash
uv run pytest tests/unit/domain/futures/strategy/test_alpha_evaluation.py \
  tests/unit/domain/futures/strategy/test_ml_builder.py \
  tests/unit/execution/test_opt_main_futures_strategy_mode.py --tb=short

uv run python -m src.execution.opt_main_futures --mode alpha
```

**현재 출력:**
```
signal_skill_passes=OK  portfolio_ic_above_breakeven=FAIL  basket_net_positive=OK
signal_preserved_after_selection=OK  multi_horizon_sweep_passes=OK  bear_market_basket_safe=OK
SWEEP: [6h: ❌] [12h: ✅] [18h: ✅]
>> ALPHA_PASS: FALSE  fail=['portfolio_ic_below_raw_breakeven']
```
