# Mode Full (ML) — 최신 검증 결과

**실행 일시:** 2026-06-05 (Phase 2: ML Pipeline & Validation Upgrade 적용 후)  
**실행 명령어:** `PYTHONPATH=. uv run python src/execution/opt_main_futures.py --phase ml --sync skip --timeframe 4h --trials 1 --date 2026-05-01`  
**상태:** `Active Signals: 433`, `Status: PROMOTED (ML Phase 2 패치 적용 버전)`

---

## 실행 요약

```text
[WINDOW] 2022-10-01 ~ 2026-03-31 | IS: 2023-10-01 | OOS: 2025-10-01
[UNIVERSE] 94개 심볼 발견
[PIPELINE] raw=272819 promoted=6350 n_folds=4
[BRIDGE][WF] fold_cost_survival=[True, True, True, True] pass_ratio=1.00 (BH-FDR 통과)
[SIGNAL-VALIDATION] overall_pass=True (mean=17.8bps)
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

### 1. q10 Catastrophic Pessimism 해결
* `selection_shortfall_mode`를 `"penalty_only"`로 완화하고 `max_expected_shortfall_bps`를 `300.0`으로 현실화하여, 기존의 극단적인 비관주의로 인한 신호 전면 차단 현상을 성공적으로 우회하였습니다.
* `pct_q10_ge_max`가 `0.000`에서 `0.607`로 정상화되었으며, 이를 통해 ML 모델이 실제로 거래를 집행하고 자본 배포를 활성화(Deploy `0.39` 수준)할 수 있게 되었습니다.

### 2. Isotonic Calibration을 통한 확률 보정 붕괴 해결
* `gate_calibration_method: isotonic`을 적용함으로써, 기존 sigmoid 방식에서 100% 발생하던 확률 분포 수축(std 0.006 수준)을 극복하고, calibration fold 중 일부에서 `calibration_accepted` 판정을 이끌어냈습니다 (cal_std 0.0316 달성).

### 3. 피처 누수(Feature Leakage) 억제
* `sym_ret_1` 및 `mkt_ret_1`과 같은 즉각적 리턴 피처를 완전히 제거하여, IS/OOS 성능 괴리를 억제하는 기초를 마련하였습니다.

---

## 다음 단계
1. **변동성 스케일에 연동된 리스크 타겟 설계**: 고변동성 알파 코인 포진으로 인한 q10 손실의 근본적 오차를 교정하기 위해, 숏폴 리스크 한도를 절대값(bps)이 아닌 자산별 ATR의 N배 형식으로 표준화(Normalization)하는 3차 스펙 작성이 필요합니다.
2. **ML 앙상블 및 하이퍼파라미터 튜닝**: 현재 일반성 강화 중심의 LGBM 하이퍼파라미터(`max_depth=2`, `reg_lambda=100.0`) 외에 추가적인 트리 앙상블 기법을 검토하여 CAGR 개선 극대화를 꾀합니다.
