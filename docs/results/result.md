# Mode Full (ML) — 최신 검증 결과

**실행 일시:** 2026-06-05 (Phase 2 + Universe-Signal-ML Coupling 적용 후)  
**실행 명령어:** `PYTHONPATH=. uv run python src/execution/opt_main_futures.py --phase ml --sync skip --timeframe 4h --trials 1 --date 2026-05-01`  
**상태:** `Active Signals: 681 (sel=97)`, `Status: PROMOTED (ML Phase 2 + metadata coupling 적용 버전)`

---

## 실행 요약

```text
[WINDOW] 2022-10-01 ~ 2026-03-31 | IS: 2023-10-01 | OOS: 2025-10-01
[UNIVERSE] 94개 심볼 발견
[PIPELINE] raw=272819 labeled=6350 promoted=6350 fit=9869 cal=9165 oos=2355 n_folds=4 wf_scheme=anchored
[BRIDGE][WF] fold_cost_survival=[True, True, True, True] pass_ratio=1.00 min_required=0.60
[PIPELINE_GATE] calibrated=True reason=calibration_accepted mean=0.4250 median=0.4006 p90=0.4936 max=1.0000
[PIPELINE_EDGE] mu_mean=21.4 mu_median=29.2 mu_p90=50.4 mu_max=65.4 q10_mean=-284.8 q10_p10=-448.9 q10_median=-259.7 q10_min=-668.8
[PIPELINE_SELECT] policy=utility_topk zero_reason=selected_nonzero eligible=1000 selected_pre_group=100 selected=97 n_keep=100
```

---

## 백테스트 성과 (OOS)

| Model | CAGR | MaxDD | MAR | Equity | Trades | Deploy | Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| Equal Size | -17.6% | 47.0% | -0.37 | 559,693 | 8 | 0.66 | N |
| Rule Promo NL | -14.1% | 16.0% | -0.88 | 913,690 | 402 | 0.64 | N |
| Rule Promo Oracle | 2.6% | 3.6% | 0.71 | 1,015,037 | 182 | 0.17 | N |
| Kelly (No ML) | -0.1% | 0.4% | 0.00 | 997,350 | 2149 | 0.66 | N |
| ML Gate | 0.0% | 0.0% | 0.00 | 1,000,149 | 99 | 0.08 | N |
| ML Gate+Edge | -3.6% | 11.3% | -0.32 | 895,411 | 66 | 0.01 | N |
| ML Full (Capped) | 0.0% | 0.0% | 0.00 | 1,000,059 | 98 | 0.08 | N |
| Cand. ML | 0.0% | 0.0% | 0.00 | 1,000,143 | 98 | 0.39 | N |
| Direct Edge | -0.0% | 0.0% | 0.00 | 999,892 | 131 | 0.26 | N |
| Variant Prior | -0.1% | 0.0% | 0.00 | 999,662 | 128 | 0.26 | N |
| Promo Filter | -0.1% | 0.1% | 0.00 | 999,472 | 545 | 0.88 | N |
| Val. Selection | -0.5% | 0.5% | 0.00 | 997,080 | 80 | 0.28 | N |
| Identity Feat | 0.0% | 0.0% | 0.00 | 1,000,163 | 95 | 0.37 | N |
| Market Feat | 0.0% | 0.0% | 0.00 | 1,000,107 | 93 | 0.38 | N |

---

## ML Layer 개선 결과 (Phase 2 진단)

### 1. q10 Catastrophic Pessimism 완화
* `selection_shortfall_mode="penalty_only"`와 `max_expected_shortfall_bps=300.0` 조합은 유지되었고, `utility_topk` 선택이 `eligible=1000`, `selected=97`까지 살아났습니다.
* 최신 실행에서 `q10_mean=-284.8bps`, `q10_p10=-448.9bps`, `selection` 단계의 `selected_nonzero`가 유지되어 hard zero-exposure로 되돌아가지 않았습니다.

### 2. Isotonic Calibration을 통한 확률 보정 붕괴 해결
* `gate_calibration_method: isotonic`이 유지되었고, 최신 실행에서 `calibration_accepted`와 `cal_std 0.0387`을 확인했습니다.
* `PIPELINE_GATE`의 `mean=0.4250`, `median=0.4006`, `p90=0.4936`으로 gate 확률이 극단적으로 붕괴하지 않았습니다.

### 3. 피처 누수(Feature Leakage) 억제
* `sym_ret_1` 및 `mkt_ret_1`과 같은 즉각적 리턴 피처를 완전히 제거하여, IS/OOS 성능 괴리를 억제하는 기초를 마련하였습니다.
* Stage6 메타데이터(`vol_30d`, `tradeable_score`)가 candidate dataset과 selection diagnostics까지 전달되도록 확장했습니다.

---

## 다음 단계
1. **변동성 스케일에 연동된 리스크 타겟 설계**: 고변동성 알파 코인 포진으로 인한 q10 손실의 근본적 오차를 교정하기 위해, 숏폴 리스크 한도를 절대값(bps)이 아닌 자산별 ATR의 N배 형식으로 표준화(Normalization)하는 3차 스펙 작성이 필요합니다.
2. **ML 앙상블 및 하이퍼파라미터 튜닝**: 현재 일반성 강화 중심의 LGBM 하이퍼파라미터(`max_depth=2`, `reg_lambda=100.0`) 외에 추가적인 트리 앙상블 기법을 검토하여 CAGR 개선 극대화를 꾀합니다.
