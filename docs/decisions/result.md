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
last_verified: 2026-05-24
---

# Result Baseline

## 1. Purpose

이 문서는 `opt_main_futures.py`의 ML strategy 개선 전/후 결과를 비교하기 위한 기준선이다.
후속 개선 작업에서 같은 실행 조건으로 재실행하여 알파 품질, trade 발생 여부, 최종 OOS 지표 변화를 비교한다.

**기준 실행 명령 (모든 결과는 아래 조건으로만 비교한다):**
```bash
timeout 3600 uv run python src/execution/opt_main_futures.py \
  --mode strategy \
  --skip-data-sync \
  --trials 300 \
  --tf 4h \
  --reference-date 2026-05-01 \
  --strategy ml_lambdamart_v1
```
- 유니버스 필터링 ON (전체 심볼 자동 선택)
- `--trials 300` 고정
- `--skip-universe` / 심볼 수 축소 결과는 비교 기준으로 사용하지 않는다

---

## 2. Baseline Run (2026-05-23, pre-fix)

유니버스 필터링 및 전체 최적화 파이프라인 첫 실행 결과. 이후 모든 개선의 비교 기준.

- **Result:** 유니버스 필터링 작동, 300-trial 완료. 최종 OOS `oos_zero_trades=1`.
- **Key Metrics:**
  - `discovered=38`, `valid=37`
  - `ML-ALPHA-IC mean_ic=0.0270 t_stat=3.39 hit_ratio=0.546`
  - `alpha_p95=0.00bps`
  - `RUN-SUMMARY phase_a1 complete=40 pruned=110`
  - `RUN-SUMMARY phase_a2 complete=60 pruned=0`
  - `RUN-SUMMARY phase_b complete=90 pruned=0`
- **Interpretation:** IC 품질은 건전하나 calibrator의 additive penalty가 alpha_p95를 0으로 수축시켜 cost wall을 통과하지 못함.

---

## 3. Path Diagnostics

`[STRAT-PATH]` 로그를 기준 비교 포인트로 사용한다.

```text
[STRAT-PATH] trial=0 leg=0 range=(2892,3190) bars=298 alpha_nz=1.0000 merge_nz=0.0354 xs_nz=0.6312 trades=35 long=22 short=13
```

- `alpha_nz`: alpha panel non-zero ratio
- `merge_nz`: membership/entry-block 반영 후 target weight non-zero ratio
- `xs_nz`: long/short `xs_score` union non-zero ratio
- `trades`: actual filled trade count at leg evaluation time

---

## 4. Invariants

- B1 canonical cost model is preserved: `build_label_panel()` stays fee/slippage-excluded (funding-adjusted) and fee/slippage subtraction happens only in the objective friction/hurdle layer.
- `sample_weight` follows the documented formula: `original_weight * (1 + 2 * abs(y_ev))`, with `y_ev = signed_net_ret`.
- `alpha_panel` contract remains `MultiIndex(datetime, symbol)` with `alpha_long` and `alpha_short`.

---

## 5. Comparison Rules

후속 개선 후 아래 순서로 비교한다.
1. `ML-ALPHA-IC` 개선 여부
2. `ML-COST-WALL` 통과 여부
3. `[STRAT-PATH]`의 `merge_nz`와 `xs_nz` 변화
4. 최종 OOS `trade_count`와 `oos_zero_trades` 변화
5. `EV/Cost`, `CAGR`, `Sortino`, `PBO` 개선 여부

---

## 6. EV Hurdle Fix (2026-05-24)

`EV_HURDLE_BPS` 탐색 범위 `[5.0, 100.0] → [3.0, 20.0]` 하향 및 기본값 `40.0 → 10.0bps` 완화.

- **Result:** 유니버스 필터링 작동, 300-trial 완료. 최종 OOS `oos_zero_trades=1` 유지.
- **Key Metrics:**
  - `discovered=38`, `valid=37`
  - `ML-ALPHA-IC mean_ic=0.0232 t_stat=2.94 hit_ratio=0.541`
  - `alpha_p95=0.00bps` (calibrator penalty로 인해 수축 유지)
  - `RUN-SUMMARY phase_a1 complete=60 pruned=90`
  - `RUN-SUMMARY phase_a2 complete=60 pruned=0`
  - `RUN-SUMMARY phase_b complete=90 pruned=0`
- **Interpretation:** AWF 중간 leg에서는 정상 거래 발생 확인. 그러나 calibrator의 additive penalty 구조가 final OOS alpha를 0으로 만드는 문제가 미해결 상태.

---

## 7. Lambda-Tail Fix (2026-05-25)

### 7.1 Applied Fix
- `config.py`: `lambda_tail` 기본값 `0.25 → 0.10`
- `calibrator.py`: `lam_dynamic` 상한 캡 `clip(lam * unc/med_unc, 0, lam * 2.0)` 추가
- Optuna per-trial 연동은 `MLPhaseDContext.strategy_cfg` 공유 구조로 인해 별도 이슈로 분리

### 7.2 300-Trial Result (2026-05-25)

- **Result:** 유니버스 필터링 작동, 300-trial 완료. 최종 OOS `oos_zero_trades=1` 유지.
- **Key Metrics:**
  - `discovered=38`, `valid=37`
  - `ML-ALPHA-IC mean_ic=0.0245 t_stat=3.06 hit_ratio=0.547` ✅
  - `alpha_p95=0.00bps` ❌ (37심볼 full run에서 재현)
  - AWF leg trades: `38/leg` (§6 대비 유지)
  - `oos_zero_trades=1` ❌
  - `RUN-SUMMARY phase_a1 complete=46 pruned=104`
  - `RUN-SUMMARY phase_a2 complete=60 pruned=0`
  - `RUN-SUMMARY phase_b complete=90 pruned=0`

### 7.3 구조적 근본 원인 확정

lambda_tail 수치 하향과 lam_dynamic 캡 적용만으로는 37심볼 전체 유니버스에서 `alpha_p95=0.00bps`가 재현됨.

**원인: additive penalty의 구조적 CS-demean 비호환성**
```
CS-demean 학습 → q50 ≈ 0 (37심볼 크로스섹션에서 완전 수렴)
ev_long = q50 - lam_dynamic * downside
       ≈ 0   - 0.10 * |q10|   →  항상 음수
```
- 수치 조정(lambda_tail 값)만으로는 해결 불가 — 수식 구조 자체를 변경해야 함

**다음 fix 방향: additive → sign-symmetric multiplicative penalty 전환**
```python
# 현재 (additive):
#   ev = q50 - lam * downside                        # q50 ≥ 0 (long)
#   ev = q50 + lam * upside                          # q50 < 0 (short)
#   → q50 ≈ 0이면 penalty가 항상 ev를 음수로 만듦

# 제안 (multiplicative, sign-symmetric):
#   ev = q50 * (1 - lam * downside / uncertainty)    # q50 ≥ 0 (long)
#   ev = q50 * (1 - lam * upside   / uncertainty)    # q50 < 0 (short)
#   - q50 ≈ 0 → ev ≈ 0 (음수 불가)
#   - downside/uncertainty, upside/uncertainty ∈ [0, 1]
#   - long은 하방 위험, short은 상방 위험으로 각자의 꼬리를 페널티화
#   - penalty가 q50 크기에 비례 → CS-demean 환경 중립적
```

### 7.4 Next Comparison Criteria
1. `alpha_p95 > 0bps` (37심볼 full run) — multiplicative penalty 적용 후 확인
2. 300-trial 재실행 후 `oos_zero_trades=0` 전환 여부
3. `EV/Cost`, `CAGR`, `Sortino` 개선 여부

---

## 8. Contract Realignment Patch (2026-05-24)

코드 레벨 정렬 패치 적용. 300-trial full rerun은 아직 미실행.

- **Applied:**
  - calibrator target을 `CS-demean y_ev`에서 raw executable EV로 변경
  - ML builder fold 단계의 `ev_test` group-centering 제거
  - runtime `[ML-COST-WALL]` 로그를 `objectives.py`에서 trial `EV_HURDLE_BPS` 기준으로 출력
  - ML builder의 default cost-wall 로그 기준을 `FUTURES_DEFAULT_EV_HURDLE_BPS`와 동기화
- **Expected Effect:**
  - `rank quality`와 `absolute EV tradeability` 충돌 완화
  - `alpha_p95`의 구조적 0 수축 완화
  - `STRAT-PATH`에서 `xs_nz`/`trades` 개선 여지 확보

---

## 9. 300-Trial Rerun Result (2026-05-24)

`src/execution/opt_main_futures.py`를 동일 기준으로 300-trial 재실행한 결과. 계약 정렬 패치 이후 중간 ML 신호는 개선됐지만, 최종 OOS는 아직 `oos_zero_trades=1`로 남았다.

- **Result:** 유니버스 필터링 작동, 300-trial 완료. 최종 OOS `oos_zero_trades=1`, 최종 verdict `HOLD (GATE_FAIL)`.
- **Key Metrics:**
  - `discovered=38`, `valid=37`
  - 초기 ML 진단: `ML-ALPHA-IC mean_ic=0.0296 t_stat=3.79 hit_ratio=0.538`
  - 초기 전체 alpha: `alpha_p95=2.70bps`
  - `RUN-SUMMARY phase_a1 complete=47 pruned=103`
  - `RUN-SUMMARY phase_a2 complete=60 pruned=0`
  - `RUN-SUMMARY phase_b complete=90 pruned=0`
  - `FINAL-FLAT-DIAG oos_zero_trades=1 wr_ok=False mdd_ok=True pf_ok=False ev_ok=False`
- **Interpretation:** `rank quality`와 `cost-wall passability`는 개선됐다. 다만 final ensemble/OOS 선택 경로가 여전히 trade를 소거해서, 실거래 관점의 최종 지표는 0-trade 상태로 종료됐다.
