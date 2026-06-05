# Mode Alpha — 최신 검증 결과

**실행 일시:** 2026-06-05 (2차 — WF gate 배선 완료 후 재실행)  
**실행 명령어:** `UV_CACHE_DIR=/tmp/uv-cache FUTURES_STRATEGY_NAME=candidate_ml PYTHONPATH=. uv run python src/execution/opt_main_futures.py --phase alpha --sync skip --timeframe 4h --trials 1 --date 2026-05-01`  
**상태:** `Active Signals: 248`, `Status: PROMOTED` ✅ *(이전: `0 / BLOCKED`)*

---

## 실행 요약

```text
[WINDOW] 2022-10-01 ~ 2026-03-31 | IS: 2023-10-01 | OOS: 2025-10-01
[UNIVERSE] 94개 심볼 발견 | Stage6: 20개 선택
[DATA] 91/94 로드 (96.8%) | 준비 완료: 63개
[BRIDGE] Active Signals: 248 (sel=61) | Status: PROMOTED | Execution Time: 50.41s
[STRATEGY] Total Stage Time: 121.57s
WF: n_folds=4  wf_scheme=anchored  fit=152132  cal=153167  oos=42683
```

---

## 핵심 변화 요약 (vs 2026-06-03)

| 지표 | 이전 (2026-06-03) | 이번 (2026-06-05) | 변화 원인 |
|---|---|---|---|
| Active Signals | **0 (BLOCKED)** | **248 (PROMOTED)** ✅ | 비용모델 현실화 + L-2 라벨 수정 |
| breakeven_floor | 12.0 bps | **3.8 bps** | cost_floor 24→7.5bps (ExecutionCostModel) |
| mu_p50 (edge) | 9.6 bps | **38.6 bps** | L-2: exit_px = 장벽가 (close→barrier price) |
| eligible | 448 | **1,115** | 비용 게이트 완화 |
| selected | 45 | **61** | — |
| gate calibration | collapse (False) | **일부 accepted** (2/4 폴드) | WF 폴드별 재학습 |
| OOS events | ~11,194 | **42,683** | WF 4-fold concat |

### BLOCKED 해소 근본 원인

1. **L-2 (exit_px = barrier price)**: TP 도달 시 실현가를 종가 대신 장벽가로 산출 → mu_p50 9.6→38.6bps (+303%). 이전 음수에 가깝던 edge가 실제 TP 수준으로 상향 보정.
2. **비용 현실화 (24→7.5bps RT)**: `ExecutionCostModel(maker_ratio=0.75)` 기본값 적용 → breakeven_floor 12.0→3.8bps. 동일 signal이 더 많이 게이트 통과.
3. **WF 4-fold**: 단일 OOS split 대비 3.8× 더 많은 OOS 관측(~11k→42k) → eligible 448→1,115.

---

## Pipeline Diagnostics (최신)

```text
[BRIDGE][WF] fold_cost_survival=[True, True, True, True] pass_ratio=1.00 min_required=0.60
  → 교차폴드 일관성 게이트 통과 (모든 폴드 mean_net > cost_floor)

[DIAG][PIPELINE] raw=248365 labeled=104552 promoted=104552
  fit=152132 cal=153167 oos=42683  n_folds=4  wf_scheme=anchored

[DIAG][PIPELINE_GATE]
  calibrated=True  reason=calibration_accepted  (폴드 2,3 / 총 4)
  mean=0.4393  median=0.4370  p90=0.5167  max=0.8511
  pct_ge40=0.774  pct_ge45=0.399  pct_ge50=0.132

[DIAG][PIPELINE_EDGE]
  cost_bps=7.5  floor_bps=3.8
  mu_mean=39.1  mu_p50=38.6  mu_p90=55.6  mu_max=71.7
  q10_p10=-917.2  q10_median=-609.3

[DIAG][PIPELINE_SELECT]
  policy=utility_topk  zero_reason=selected_nonzero
  eligible=1115  selected=61  breakeven_floor=3.8bps
```

---

## Candidate Top Strategies

| Rank | Strategy Name | Sample (OOS) | Profit(bps) | Win Rate | P/L | Score | Action | Δ vs 이전 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | funding_zscore_carry:fzs_168 | 3692 (1306) | **58.4** | 47.1% | 1.33 | 0.081 | KEEP | +24.4bps ↑ |
| 2 | rsi_reversion:rsi_14 | 4409 (1838) | **56.1** | 51.4% | 1.36 | -0.057 | KEEP | +27.8bps ↑ |
| 3 | vol_regime_reversion:vrr_40 | 4297 (1556) | **52.6** | 42.4% | 1.16 | -0.106 | DROP | **역전** (-6.2→+52.6) |
| 4 | cross_sectional_momentum:cs_mom_20 | 15486 (6480) | **43.6** | 44.4% | 1.19 | -0.079 | DROP | **역전** (-1.8→+43.6) |
| 5 | vol_breakout:bb_compress_20 | 375 (174) | 43.3 | 41.4% | 1.41 | -0.288 | KEEP | **신규 진입** |
| 6 | btc_regime_pullback:btc_pullback_50 | 1962 (667) | **41.3** | 44.4% | 1.21 | -0.184 | KEEP | +21.3bps ↑ |
| 7 | cross_sectional_momentum:cs_mom_10 | 10612 (4488) | **33.1** | 46.1% | 1.21 | -0.049 | KEEP | +29.5bps ↑ |
| 8 | funding_carry:funding_24 | 3763 (1560) | 27.4 | 44.6% | 1.31 | -0.017 | KEEP | +11.5bps ↑ |
| 9 | funding_acceleration_carry:fac_48 | 15100 (6743) | 24.8 | 48.6% | 1.16 | -0.003 | DROP | **신규 진입** |
| 10 | btc_corr_regime:bcr_96 | 24336 (9896) | **22.7** | 41.9% | 1.25 | 0.017 | KEEP | **역전** (-4.9→+22.7) |

> **관찰:** 전략 수익 분포가 전반적으로 상향 이동. L-2 수정(exit_px = barrier price)으로 TP 도달 시 realized gain이 현실화됨. 이전에 음수였던 vrr_40, cs_mom_20, bcr_96이 양수로 역전.

---

## Ablation Study

| Model Alias | CAGR | MaxDD | MAR | Equity | Trades | Deploy | Pass | Δ vs 이전 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Equal Size | -26.9% | 61.9% | -0.43 | 390,820 | 8 | 0.66 | **N** | — |
| Rule Promo NL | -32.3% | 22.2% | -1.45 | 793,903 | 37 | 1.00 | **N** | — |
| Rule Promo Oracle | -35.1% | 28.4% | -1.24 | 774,059 | 83 | 1.00 | **N** | — |
| Kelly (No ML) | -0.6% | 2.5% | -0.24 | 982,300 | 2153 | 0.66 | **N** | — |
| ML Gate | 0.0% | 0.0% | 0.87 | 1,000,411 | 67 | 0.04 | **N** | — |
| ML Gate+Edge | -2.1% | 10.3% | -0.21 | 937,168 | 49 | 0.01 | **N** | — |
| ML Full (Capped) | 0.0% | 0.0% | 0.67 | 1,000,451 | 69 | 0.04 | **N** | — |
| Cand. ML | **0.1%** | 0.0% | **2.95** | 1,000,357 | 69 | 0.19 | **Y** | MAR 1.47→2.95 ↑ |
| Direct Edge | **0.0%** | 0.0% | **2.78** | 1,000,274 | 80 | 0.22 | **Y** | **신규 Pass** |
| Variant Prior | 0.0% | 0.0% | **2.73** | 1,000,224 | 66 | 0.20 | **Y** | MAR 1.20→2.73 ↑ |
| Promo Filter | 0.0% | 0.1% | 0.30 | 1,000,242 | 100 | 0.29 | **N** | Y→N (MAR 하락) |
| Val. Selection | 0.0% | 0.0% | 0.00 | 1,000,000 | 0 | 0.00 | **N** | — |
| Identity Feat | 0.0% | 0.1% | 0.15 | 1,000,061 | 61 | 0.24 | **N** | Y→N |
| Market Feat | 0.1% | 0.0% | 1.04 | 1,000,307 | 40 | 0.18 | **N** | Y→N (fold DSR/PBO 재계산) |

---

## 해석

### 해소된 사항 ✅
- **Active Signals: 0 → 248**: 비용모델 현실화(24→7.5bps) + L-2 라벨 수정이 주 원인. 이전 BLOCKED의 직접 원인은 `breakeven_floor=12bps`가 현실보다 지나치게 높았기 때문이다.
- **전략 수익 현실화**: 모든 KEEP 전략의 Profit(bps)이 대폭 상승. 이전에 음수였던 전략 중 일부가 실제로는 수익성이 있음이 드러남.
- **WF 4-fold**: OOS 관측이 4× 증가하여 selection pool 확대(eligible 448→1,115).

### 2차 실행 변경사항 (2026-06-05 WF gate 배선 후)

| 변경 | 이전 (1차) | 이번 (2차) | 원인 |
|---|---|---|---|
| WF fold gate 로그 | 미출력 | `[True,True,True,True] pass=1.00` | `min_wf_fold_pass_ratio` 배선 완료 |
| Market Feat ablation | **Y** (MAR 1.04) | **N** (MAR 1.04) | `fold_oos_boundaries` 배선 → DSR/PBO 4-fold 실제 경계 사용, block_pass_ratio 재계산 |

Market Feat: MAR=1.04로 동일하나 WF 4-fold 경계 기준 `block_pass_ratio < 0.70` 판정 → FAIL. 6개월 인위 블록보다 **fold-aligned 평가가 더 엄격**하게 동작한 것 — 의도된 결과.

### 잔존 주의사항 ⚠️
- **q10 패턴 지속**: `q10_p10=-917bps`, `q10_min=-2083bps`. `pct_q10_ge_cat=0.029` (≥-300 기준 2.9%). catastrophic shortfall 필터가 대부분을 차단 중.
- **gate calibration collapse**: 4폴드 중 앞 2개(raw_std≈0.024~0.027)만 collapse. 뒤 2개 폴드(더 많은 데이터)는 accepted. 초기 폴드 fit_obs 부족이 원인.
- **Promo Filter / Identity Feat 계속 N**: WF fold-aligned DSR/PBO 적용 후 block_pass_ratio 기준 미달. fold 단위 성과 불안정성이 드러남.

### 다음 작업 우선순위
1. **q10 극단값 원인 분리**: `q10_min=-2083bps` — 특정 심볼/시점 outlier 여부 event-level 분포 분석.
2. **gate calibration 안정화**: 초기 폴드 raw_std < 0.03 collapse 패턴 → dynamic threshold 또는 min_fit_obs 상향 조정 검토.
3. **Promo/Identity Feat block_pass_ratio**: 어느 fold에서 음수 return인지 확인 후 promotion 조건 재점검.
