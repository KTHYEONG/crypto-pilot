# Mode ML — 최신 검증 결과

**실행 일시:** 2026-06-05 (4차 — audit 주의사항 3건 수정 후 재실행)  
**실행 명령어:** `UV_CACHE_DIR=/tmp/uv-cache FUTURES_STRATEGY_NAME=candidate_ml PYTHONPATH=. uv run python src/execution/opt_main_futures.py --phase ml --sync skip --timeframe 4h --trials 1 --date 2026-05-01`  
**상태:** `Active Signals: 985 (sel=428)`, `Status: PROMOTED` ✅

---

## 실행 요약

```text
[WINDOW] 2022-10-01 ~ 2026-03-31 | IS: 2023-10-01 | OOS: 2025-10-01
[UNIVERSE] 94개 심볼 발견 | Stage6: 20개 선택
[DATA] 91/94 로드 (96.8%) | 준비 완료: 63개
[BRIDGE] Active Signals: 985 (sel=428) | Status: PROMOTED | Execution Time: 51.61s
[STRATEGY] Total Stage Time: 124.51s
WF: n_folds=4  wf_scheme=anchored  fold_cost_survival=[True, True, True, True] pass_ratio=1.00
```

---

## 핵심 변화 요약 (vs 2026-06-05 2차)

| 지표 | 2차 (이전) | 3차 (현재) | 변화 원인 |
|---|---|---|---|
| Active Signals | 248 (sel=61) | **985 (sel=428)** ✅ | RC3: q10 stop-clipping → eligible 4.2× |
| eligible | 1,115 | **4,733** | q10_bound_to_stop → pct_q10_ge_cat 2.9%→12.1% |
| pct_q10_ge_cat | 2.9% | **12.1%** | RC3 |
| q10_median | −609bps | **−498bps** | RC3 stop-clip |
| Ablation PASS | **3개 (Y)** | **0개** | RC2: 무행동 노이즈 PASS 제거 |
| MAR (Cand. ML) | 2.95 (허위) | **0.00** | RC2: max_dd < 0.01 → MAR=0 |
| 전략 phase | `strategy` (full 동의어) | 제거 완료 | `full`/`ml`/`signal` 3개로 정리 |

---

## 3차 실행: RC1/RC2/RC3 수정 내역

### RC1 — Ablation 엔진에 TP/SL 장벽 배선 (Phase 0/1)
- `ablation._run_backtest_and_evaluate` → `aligned_data`에 `candidate_stop_atr_mult` / `candidate_take_profit_atr_mult` 배열 주입 (`cfg.eval_apply_candidate_barriers=True`)
- `_compute_realized_edge`: price-based formula `(exit/entry−1)×side` — LONG/SHORT 문자열 변환 처리
- **진단 결과**: `candidate_ml_full` capture = 0.975 (이전), 배선 후 실현 엣지가 예측 엣지와 정합

### RC2 — 게이트가 무행동을 보상하던 모순 제거 (Phase 2)
- `evaluate_compound_backtest`: `deployed_bar_fraction`, `trade_count` 신규 인수
- MAR guard: `max_dd < mar_min_drawdown_floor (0.01)` → MAR=0 (이전엔 0.0001%/0.00005% → MAR=2.95)
- `min_cagr_for_promotion=0.02` 절대 CAGR 하한 추가
- `enforce_deployment_in_compound_gate=True`: 배치량이 적은 변형은 PASS 불가
- **결과**: 이전 3개 spurious PASS → **0개**, 모든 ablation 변형이 정직한 N

### RC3 — Q10 하방 타겟을 실현 가능한 손절가로 클립 (Phase 3)
- `candidate_labels.py`: `sl_thr_bps = stop_atr_mult × ATR / entry_px × 10000` 컬럼 추가
- `candidate_dataset.py`: `y_q10 = max(mae_bps, −sl_thr_bps)` — 종가 기반 페이퍼 드로다운 → 손절 한도로 클립
- **결과**: `pct_q10_ge_cat` 2.9% → **12.1%** (4.2×), eligible 1,115 → **4,733**

---

## Pipeline Diagnostics (최신)

```text
[BRIDGE][WF] fold_cost_survival=[True, True, True, True] pass_ratio=1.00 min_required=0.60

[DIAG][PIPELINE_GATE]
  calibrated=True  reason=calibration_accepted
  mean=0.4393  median=0.4370  p90=0.5167  max=0.8511
  pct_ge40=0.774  pct_ge45=0.399  pct_ge50=0.132

[DIAG][PIPELINE_EDGE]
  cost_bps=7.5  floor_bps=3.8
  mu_mean=39.1  mu_median=38.6  mu_p90=55.6  mu_max=71.7
  q10_mean=-524.5  q10_p10=-805.5  q10_median=-498.7  q10_min=-1405.5

[DIAG][PIPELINE_SELECT]
  policy=utility_topk  zero_reason=selected_nonzero
  eligible=4733  selected_pre_group=474  selected=428  breakeven_floor=3.8bps
```

---

## Edge Attribution Diagnostics (RC1 기준)

| Variant | pred_p50 | real_p50 | capture | pass_deploy |
|---|---|---|---|---|
| rule_only_equal_size | nan | −44bps | nan | N (trades=8) |
| rule_promo_no_leak | nan | −45.5bps | nan | Y |
| rule_promo_oos_oracle | nan | −46bps | nan | Y |
| candidate_ml_full | 34.6bps | −4.2bps | −0.12 | Y |
| candidate_ml_direct_edge | 45.3bps | −2.6bps | −0.06 | Y |
| candidate_ml_variant_prior | 42.9bps | **+3.1bps** | 0.07 | Y |

> **해석:** Rule-only 변형의 `real_p50≈−45bps` = 장벽 없이 전기간 flat-hold 시 손실 (RC1 확증). ML 변형들은 이제 배리어 적용되어 실현 엣지가 대폭 개선됐으나 여전히 pred_p50 대비 capture <1. 이는 signal 자체 엣지의 박약함을 시사.

---

## Ablation Study (4차 — 장벽 공정 비교 적용)

> **4차 변경:** audit 주의사항 수정 — `evaluate_compound_backtest` sentinel fix, `_build_barrier_arrays` 방어적 정렬, **rule-only 변형에도 장벽 배선** (barrier_events 파라미터). rule-vs-ML 비교가 이제 동일한 실행 의미론으로 측정됨.

| Model Alias | CAGR | MaxDD | MAR | Equity | Trades | Deploy | Pass | Δ vs 3차 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Equal Size | -24.6% | 58.0% | -0.42 | 428,501 | 8 | 0.66 | **N** | 장벽 배선 후 소폭 개선 |
| Rule Promo NL | -31.0% | 21.0% | -1.48 | 802,521 | 37 | 1.00 | **N** | 동등 수준 |
| Rule Promo Oracle | -34.8% | 27.0% | -1.29 | 775,797 | 83 | 1.00 | **N** | — |
| Kelly (No ML) | -0.5% | 2.0% | -0.23 | 986,381 | 2153 | 0.66 | **N** | — |
| ML Gate | -0.0% | 0.0% | 0.00 | 999,846 | 328 | 0.12 | **N** | 장벽 배선 포함 |
| ML Gate+Edge | -12.1% | 34.8% | -0.35 | 677,874 | 223 | 0.06 | **N** | — |
| ML Full (Capped) | -0.0% | 0.0% | 0.00 | 999,702 | 319 | 0.12 | **N** | — |
| Cand. ML | 0.0% | 0.0% | 0.00 | 1,000,175 | 319 | 0.59 | **N** | — |
| Direct Edge | 0.0% | 0.0% | 0.00 | 1,000,167 | 323 | 0.60 | **N** | — |
| Variant Prior | 0.0% | 0.0% | 0.00 | 1,000,252 | 322 | 0.60 | **N** | — |
| Promo Filter | 0.4% | 0.1% | 0.00 | 1,002,183 | 326 | 0.65 | **N** | — |
| Val. Selection | 0.0% | 0.0% | 0.00 | 1,000,000 | 0 | 0.00 | **N** | — |
| Identity Feat | 0.0% | 0.0% | 0.00 | 1,000,257 | 304 | 0.56 | **N** | — |
| Market Feat | -0.0% | 0.0% | 0.00 | 999,882 | 262 | 0.52 | **N** | — |

---

## 해석

### 해소된 사항 ✅
- **RC2 — 무행동 PASS 제거**: MAR 노이즈 폭발 방지 + CAGR 2% 하한 + 배치 의무화. 이전 3개 spurious PASS(`Cand. ML MAR=2.95`, `Direct Edge MAR=2.78`, `Variant Prior MAR=2.73`)가 모두 N으로 전환.
- **RC3 — q10 현실화**: `pct_q10_ge_cat` 2.9% → 12.1%. eligible 1,115 → 4,733. Active Signals 248 → 985.
- **RC1 — 측정 정합**: 배리어 배선으로 라벨 ↔ 평가 의미론 일치. `real_edge` 어트리뷰션 가시화.

### 새로운 핵심 관찰
- **Ablation PASS = 0인 것은 정상**: RC2 이후 게이트가 실제 복리 성장이 있는 변형만 통과시킴. 현재 0.0% CAGR은 신호 엣지 자체가 비용 대비 박약함을 솔직하게 반영.
- **ML Gate+Edge 급락 (−12.1% CAGR)**: uncapped Kelly + RC3으로 더 많은 후보 선택 → Kelly 분모(variance) 대비 mu 비율로 인한 과대 sizing이 원인. cap projection 없이 동작하는 Variant 4의 설계상 한계.
- **Promo Filter = CAGR 0.4%이나 MAR=0.00**: max_dd=0.1%이므로 `0.4%/0.1% = MAR 4.0`이 나와야 하는데 0.00인 이유 = `min_cagr_for_promotion=0.02 (2%)` 미달로 fail. `0.4% < 2%` → 정상 동작.

### 잔존 과제 (다음 우선순위)
1. **Signal edge 강화**: rule_promo_oracle이 −35%인 한 ML은 noise amplifier에 불과. 유효한 signal family 발굴이 최우선.
2. **q10_median 여전히 −498bps**: RC3으로 2.9%→12.1%로 개선됐으나 catastrophic_shortfall_bps=300 기준 88%가 여전히 차단. stop-clip이 완전히 적용되려면 labeling 단계의 stop 파라미터 범위 검토 필요.
3. **ML Gate+Edge variance 폭발**: Variant 4 uncapped Kelly에서 관측. 실제 배포 path인 Variant 5+6에는 cap이 적용돼 영향 없으나, q10 극단값이 많을 경우 sizing 왜곡 발생 가능.
4. **gate calibration**: 초기 WF 폴드 `raw_std < 0.03` collapse 패턴 — `min_fit_obs` 상향 또는 dynamic threshold 검토.
