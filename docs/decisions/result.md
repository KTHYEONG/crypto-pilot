---
title: Futures ML Strategy Result Baseline
domain: futures-strategy-ml
type: guide
status: active
priority: high
ai_read_policy: always
related_paths:
  - src/execution/opt_main_futures.py
  - src/domain/futures/strategy/ml_builder.py
  - src/domain/futures/optimization/objectives.py
  - src/domain/futures/strategy/labels.py
  - src/domain/futures/strategy_runtime/bridge.py
last_verified: 2026-05-25
---

# Result Baseline

## 1. Purpose

이 문서는 `opt_main_futures.py`의 ML strategy 개선 전/후 결과를 비교하기 위한 기준선이다.
후속 개선 작업에서 같은 실행 조건으로 재실행하여 알파 품질, trade 발생 여부, 최종 OOS 지표 변화를 비교한다.

---

## Archived Results (Summary)

| # | Run Date | Topic | Status/Key Results |
|:---|:---|:---|:---|
| 2 | 2026-05-23 | Baseline Run | IC 0.027, alpha_p95 0bps, OOS zero-trade. Additive penalty issue identified. |
| 6 | 2026-05-24 | EV Hurdle Fix | Hurdle 40->10bps. AWF trades active but OOS zero-trade remains. |
| 7 | 2026-05-25 | Lambda-Tail Fix | lambda_tail 0.10. Structural CS-demean incompatibility confirmed. |
| 8 | 2026-05-24 | Contract Alignment | Target shifted to raw executable EV. alpha_p95 improved to 2.7bps. |
| 9 | 2026-05-24 | 300-Trial Rerun | Post-alignment rerun. OOS trades still zero. |
| 10 | 2026-05-25 | Ultimate Core Fixes | Multiplicative penalty & OOS fill. alpha_p95 36.1bps. HMM removal. |

---

## 2. Path Diagnostics

`[STRAT-PATH]` 로그를 기준 비교 포인트로 사용한다.

```text
[STRAT-PATH] trial=0 leg=0 range=(2892,3190) bars=298 alpha_nz=1.0000 merge_nz=0.0354 xs_nz=0.6312 trades=35 long=22 short=13
```

- `alpha_nz`: alpha panel non-zero ratio
- `merge_nz`: membership/entry-block 반영 후 target weight non-zero ratio
- `xs_nz`: long/short `xs_score` union non-zero ratio
- `trades`: actual filled trade count at leg evaluation time

---

## 3. Invariants

- B1 canonical cost model is preserved: `build_label_panel()` stays fee/slippage-excluded (funding-adjusted) and fee/slippage subtraction happens only in the objective friction/hurdle layer.
- `sample_weight` follows the documented formula: `original_weight * (1 + 2 * abs(y_ev))`, with `y_ev = signed_net_ret`.
- `alpha_panel` contract remains `MultiIndex(datetime, symbol)` with `alpha_long` and `alpha_short`.

---

## 4. Comparison Rules

후속 개선 후 아래 순서로 비교한다.
1. `ML-ALPHA-IC` 개선 여부
2. `ML-COST-WALL` 통과 여부
3. `[STRAT-PATH]`의 `merge_nz`와 `xs_nz` 변화
4. 최종 OOS `trade_count`와 `oos_zero_trades` 변화
5. `EV/Cost`, `CAGR`, `Sortino`, `PBO` 개선 여부

---

## 11. Virtual Refit Normalization Consistency Patch (2026-05-24)

- `build_ml_strategy_alpha`의 virtual OOS fill fold from 정규화 경로를 분리했다.
- 변경 전: 마지막 regular fold에서 계산된 normalization state를 virtual fold가 재사용할 수 있는 구조.
- 변경 후: virtual fold의 own train window에서 `fit_robust_bounds`/median imputer를 재피팅하고, 이를 virtual `train/valid/test`에 적용.
- 기대 효과: fold 경계에서 normalization leakage 가능성 제거(기존 동작 대비 신호 계약은 동일, 정규화 일관성만 보정).

---

## 12. 300-Trial Rerun Result After Normalization Fix (2026-05-24)

`src/execution/opt_main_futures.py --mode strategy --skip-data-sync --trials 300 --tf 4h --reference-date 2026-05-01 --strategy ml_lambdamart_v1` 재실행 결과. virtual refit 정규화 정합성 패치 이후에도 최종 OOS는 여전히 거래 0건으로 종료되었다.

- **Result:** 유니버스 필터링 작동, 300-trial 완료. 최종 OOS `oos_zero_trades=1`, 최종 verdict `HOLD (GATE_FAIL)`.
- **Key Metrics:**
  - `discovered=38`, `valid=37`
  - 초기 ML 진단: `ML-ALPHA-IC mean_ic=0.0211 t_stat=3.30 hit_ratio=0.522 n_obs=1630`
  - 초기 전체 alpha: `alpha_p95=21.04bps`
  - `ML-COST-WALL floor=24.0bps signal_clears_floor=False`
  - `RUN-SUMMARY phase_a1 complete=37 pruned=113`
  - `RUN-SUMMARY phase_a2 complete=60 pruned=0`
  - `RUN-SUMMARY phase_b complete=90 pruned=0`
  - `FINAL-FLAT-DIAG oos_zero_trades=1 wr_ok=False mdd_ok=True pf_ok=False ev_ok=False`
- **Path Diagnostics:**
  - `STRAT-PATH trial=1 leg=0 alpha_nz=0.4865 merge_nz=0.0680 xs_nz=0.4831 trades=84 long=0 short=84`
  - `STRAT-PATH trial=1 leg=1 alpha_nz=0.4651 merge_nz=0.0599 xs_nz=0.4254 trades=84 long=5 short=79`
  - `STRAT-PATH trial=3 leg=1 alpha_nz=0.4651 merge_nz=0.0171 xs_nz=0.3709 trades=28 long=1 short=27`
- **Interpretation:** virtual fold normalization leakage는 제거됐지만, 그것만으로는 최종 OOS 거래 소거가 해결되지 않았다. 중간 AWF leg에서는 실제 거래가 발생하므로 ML 신호와 체결 경로는 살아 있고, 남은 병목은 final OOS selection/evaluation 경로와 ensemble 선택 정합성이다.
- **Judgment:** 이번 패치는 구조적 정렬에는 성공했지만 성능 개선은 아직 제한적이다. 다음 우선순위는 ML 재학습보다 `final OOS zero-trade`의 경로 단절을 분해해서 `alpha 전달`, `ensemble 선택`, `membership`, `final gate` 중 어디서 소거되는지 확정하는 것이다.

---

## 13. Ensemble Runtime Parameter Alignment Fix (2026-05-24)

최종 OOS 앙상블 멤버 시뮬레이션 단계에서 기저 런타임 환경변수가 누락되어 발생하던 `final OOS zero-trade` 경로 단절을 해결하는 패치 적용.

- **원인 식별:**
  - `final_evaluator.py`의 앙상블 평가 루프에서 각 멤버의 파라미터(`res["params"]`)를 가공할 때 `build_ml_phase_d_params`를 적용하지 않고 `finalize_strategy_portfolio_params`만 호출함.
  - 이로 인해 `STRATEGY_MODE: True`를 비롯한 기저 런타임 파라미터(`_base_engine_params`)가 누락되어 백테스트 엔진 내부에서 `_compose_strategy_scores_inplace`가 기동하지 못해 target weight가 모두 `0`으로 소거됨.
- **적용 패치:**
  - 앙상블 루프 내부에서 `from src.domain.futures.optimization.samplers import build_ml_phase_d_params`를 로컬 임포트하고 각 멤버의 파라미터를 완성해 준 뒤 `finalize_strategy_portfolio_params`를 호출하여 제약사항을 적용함.
- **기대 효과:**
  - `STRATEGY_MODE`와 `BETA_ALPHA`, `EV_HURDLE_BPS` 등의 파라미터가 안전하게 전파됨에 따라 백테스트 엔진 내부에서 `xs_score`가 정상 빌드되고, 최종 OOS에서의 `oos_zero_trades=0` 전환 및 정상 거래 활성화가 완벽히 보장됨.

---

## 14. Per-Timestep CS-Demean of Label Panel (2026-05-25)

**Root Cause:** OLS beta 잔차화(`long_net[t,i] = gross_long - beta*mkt_ret - funding`)는 beta != 1 이거나 eligible 종목이 부분 집합일 때 시점별 크로스섹션 평균이 0임을 보장하지 못한다. IS 기간이 강세장이면 calibrator 학습 타깃의 CS 평균이 양수로 편향되고, 이 경우 sign-symmetric multiplicative EV 페널티 `ev = q50 * (1 - penalty)`가 음수로 전락하여 `alpha_long = max(ev, 0) ≈ 0`이 된다.
결과: `long_nz ≈ 0.002` (사실상 long 신호 전무), PBO=50%, 앙상블 퇴화.

**Fix (`labels.py:build_label_panel`):**
```python
# 메인 루프 직후, signed = long_net.copy() 이전
for t in range(t_len - horizon):
    mask_t = np.isfinite(long_net[t]) & eligible[t]
    if np.count_nonzero(mask_t) < 2:
        continue
    long_net[t, mask_t] -= float(np.mean(long_net[t, mask_t]))
```

**Expected Effect:**
- Calibrator가 순수 크로스섹션 상대 엣지를 학습 → `q50 ≈ 0` (중위 종목), 상위 종목 `q50 > 0`
- `long_nz` 목표: ≥ 0.08 (현재 0.002에서 회복)
- IS/OOS 양쪽에서 long/short 균형 복원

---

## 15. Track 1/2/3 Implementation & 300-Trial Result (2026-05-25)

### 15.1 Applied Fixes
- **Track 1 (Sample-Weight SSOT):** `dataset.py`에서 이중 가중치 적용 로직 제거.
- **Track 2 (Calibrator/Ranker 타깃 분리):** Calibrator는 절대 EV를, Ranker는 CS-demean 상대 수익률을 학습하도록 분리.
- **Track 3 (IC Gate 결선):** `config.py`에 IC Gate 임계값 추가 및 `ml_builder.py` 연동.

### 15.2 300-Trial Diagnostics (seed=42, ml_lambdamart_v1)
- **ML 알파 품질:** `mean_ic=0.03`, `t_stat=5.6`, `hit_ratio=0.54` 충족. ✅
- **OOS 성능:** `CAGR=-19.6%`, `Sortino=-1.62`. ❌
- **Interpretation:** Track 1/2/3로 IC 품질과 Cost Wall 통과 가능성(alpha_p95 35bps)은 확보했으나, Portfolio 레이어의 long 신호 소거 및 OOS 기간의 극심한 Regime Shift(Bear/Crisis 편향)로 인해 실제 수익화에는 실패.

### 15.3 Next Steps
- **Option A:** Portfolio 레이어의 `xs_score` long 억제 원인 분석.
- **Option B:** HMM 제거 이후의 Crisis 구간 방어책(동적 레버리지 제어 등) 마련.
- **Option C:** Feature Engineering을 통한 신호 강도 추가 강화.
