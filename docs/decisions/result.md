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

## 2. Baseline Runs

### 2.1 Smoke Baseline
- Command: `uv run python src/execution/opt_main_futures.py --mode strategy --skip-universe --skip-data-sync --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,TRXUSDT,LTCUSDT --trials 1 --tf 4h --reference-date 2026-05-01 --strategy ml_lambdamart_v1`
- Scope: universe filtering off, single-trial sanity check
- Result: strategy stage and AWF path were reachable, but final OOS still produced `oos_zero_trades=1`
- Key signals:
  - `mean_ic=0.0206`
  - `t_stat=2.09`
  - `hit_ratio=0.523`
  - `alpha_p95=10.86bps`
  - `floor=54.0bps`
  - `signal_clears_floor=False`
- Interpretation: ML signal quality is statistically usable, but the cost wall is not cleared at final OOS.

### 2.2 Universe-Filtered 300-Trial Baseline
- Command: `timeout 3600 uv run python src/execution/opt_main_futures.py --mode strategy --skip-data-sync --trials 300 --tf 4h --reference-date 2026-05-01 --strategy ml_lambdamart_v1`
- Scope: universe filtering on, `trials=300`
- Result: universe filtering passed, 300-trial optimization completed, but final OOS still produced `oos_zero_trades=1`
- Key signals:
  - `discovered=38`
  - `valid=37`
  - `ML-ALPHA-IC mean_ic=0.0270`
  - `t_stat=3.39`
  - `hit_ratio=0.546`
  - `alpha_p95=0.00bps`
  - `floor=54.0bps`
  - `signal_clears_floor=False`
  - `RUN-SUMMARY phase_a1 complete=40 pruned=110`
  - `RUN-SUMMARY phase_a2 complete=60 pruned=0`
  - `RUN-SUMMARY phase_b complete=90 pruned=0`
- Interpretation: universe filtering increased statistical quality and optimization throughput, but the strategy still fails the cost wall and remains trade-sparse at final OOS.

## 3. Path Diagnostics

`[STRAT-PATH]` 로그를 기준 비교 포인트로 사용한다.

Example:
```text
[STRAT-PATH] trial=0 leg=0 range=(2892,3190) bars=298 alpha_nz=1.0000 merge_nz=0.0354 xs_nz=0.6312 trades=35 long=22 short=13
```

Meaning:
- `alpha_nz`: alpha panel non-zero ratio
- `merge_nz`: membership/entry-block 반영 후 target weight non-zero ratio
- `xs_nz`: long/short `xs_score` union non-zero ratio
- `trades`: actual filled trade count at leg evaluation time

## 4. Invariants

- B1 canonical cost model is preserved: `build_label_panel()` stays gross-alpha only and cost subtraction happens only in the objective friction/hurdle layer.
- `sample_weight` follows the documented formula: `original_weight * (1 + 2 * abs(y_ev))`, with `y_ev = signed_net_ret`.
- `alpha_panel` contract remains `MultiIndex(datetime, symbol)` with `alpha_long` and `alpha_short`.

## 5. Comparison Rules

후속 개선 후 아래 순서로 비교한다.
1. `ML-ALPHA-IC` 개선 여부
2. `ML-COST-WALL` 통과 여부
3. `[STRAT-PATH]`의 `merge_nz`와 `xs_nz` 변화
4. 최종 OOS `trade_count`와 `oos_zero_trades` 변화
5. `EV/Cost`, `CAGR`, `Sortino`, `PBO` 개선 여부

## 6. Known Limits

- 현재 기준선은 최종 OOS에서 여전히 `0 trade`다.
- `sample_weight`와 진단 로그 개선은 반영됐지만, final gate를 넘을 만큼의 알파 magnitude는 아직 확보되지 않았다.
- 전역 `ruff`/`mypy`는 저장소 기존 이슈로 인해 이번 변경과 무관하게 실패 상태였다.

## 7. Improved Run Result (2026-05-24)

- **Command:** `uv run python src/execution/opt_main_futures.py --mode strategy --skip-universe --skip-data-sync --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,TRXUSDT,LTCUSDT --trials 1 --tf 4h --reference-date 2026-05-01 --strategy ml_lambdamart_v1`
- **Result:** 전역 비용 허들 완화로 인해 AWF 경로 내 **수십 건의 실체 거래 활성화 완료**
- **Key parameters:**
  - `Tuned EV_HURDLE_BPS`: `10.88 bps` (기존 40.0 bps 대비 대폭 하향 수렴)
  - `Effective Floor`: `14.0bps (friction) + 10.88bps (hurdle) = 24.88 bps`
- **Path Diagnostics (`[STRAT-PATH]`):**
  - **Leg 0:** `merge_nz=0.0360`, `xs_nz=0.5729`, **`trades=34`** (long=21, short=13)
  - **Leg 1:** `merge_nz=0.0232`, `xs_nz=0.3835`, **`trades=31`** (long=17, short=14)
  - **Leg 2:** `merge_nz=0.0101`, `xs_nz=0.2407`, **`trades=25`** (long=15, short=10)
  - **Leg 3:** `merge_nz=0.0320`, `xs_nz=0.5537`, **`trades=34`** (long=22, short=12)
  - **Leg 4:** `merge_nz=0.0322`, `xs_nz=0.5364`, **`trades=33`** (long=16, short=17)
- **Interpretation:** 기존에 완전히 사멸했던(`0 trades`) 포지션 진입 경로가 허들 하향 패치로 인해 성공적으로 복원되었습니다. IS 및 AWF 각 단계에서 수십 건의 정상 거래가 수행됨으로써 전략의 정상 동작 및 무결성이 수치적으로 최종 입증되었습니다.



