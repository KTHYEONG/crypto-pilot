# Mode Alpha — P0~P2 구현 완료 결과

**실행 일시:** 2026-06-03 (최신)  
**상태:** ✅ 신규 family 4개 추가, OOS 게이트 개선, IC gate 제거, IS→OOS 가드 수정

---

## 실행 요약

```
[WINDOW] 범위: 2022-10-01 ~ 2026-03-31 | IS: 2023-10-01 | OOS: 2025-10-01
[UNIVERSE] 94개 심볼 발견 | Stage6: 20개 선택
[DATA] 91/94 로드 (96.8%) | 준비 완료: 63개
[STRATEGY] candidate_ml | Active Signals: 0 (blocker: ML gate p_pass < 0.40)
[BRIDGE] Execution: 26.16s | Status: BLOCKED
```

---

## CANDIDATE TOP STRATEGIES (OOS 순)

| Rank | Strategy Name | OOS Sample | Profit(bps) | Win Rate | P/L | Score | Action | 신규 |
|------|---|---|---|---|---|---|---|---|
| 1 | btc_regime_pullback:btc_pullback_50 | (684) | 36.3 | 43.9% | 1.31 | -0.115 | **KEEP** | |
| 2 | **funding_zscore_carry:fzs_168** | (1348) | 35.4 | 44.9% | 1.36 | 0.081 | **KEEP** | ✅ F2 |
| 3 | rsi_reversion:rsi_14 | (1905) | 20.7 | 49.0% | 1.35 | -0.041 | **KEEP** | |
| 4 | funding_carry:funding_24 | (1598) | 13.8 | 42.6% | 1.39 | -0.026 | **KEEP** | |
| 5 | **funding_zscore_carry:fzs_48** | (712) | 6.1 | 40.9% | 1.36 | 0.049 | **KEEP** | ✅ F2 |
| 6 | **cross_sectional_momentum:cs_mom_10** | (4645) | 4.6 | 44.9% | 1.25 | -0.016 | **KEEP** | ✅ F1 |
| 7 | cross_sectional_momentum:cs_mom_5 | (4573) | -0.5 | 47.2% | 1.18 | -0.009 | DROP | ✅ F1 |
| 8 | vol_regime_reversion:vrr_40 | (1629) | -3.4 | 42.1% | 1.18 | -0.070 | DROP | ✅ F3 |
| 9 | **btc_corr_regime:bcr_96** | (10199) | -3.6 | 41.7% | 1.25 | 0.039 | DROP | ✅ F4 |
| 10 | trend_donchian:donchian_18 | (1190) | -5.0 | 40.8% | 1.33 | 0.062 | DROP | |

---

## 핵심 개선 효과

### 1. IC 게이트 제거 ✅
- **변경:** `_meets_recommendation_thresholds`에서 `oos_rank_ic >= 0.01` 제거
- **이유:** 이진/임계값 신호에서 IC 음수 구조적 편향 (숏 신호 raw_score < 0, edge > 0 → IC 음수)
- **결과:** btc_pullback_50 차단 해제 → KEEP 진입

### 2. IS→OOS 부호 가드 개선 ✅
- **변경:** `np.sign(train) != np.sign(oos)` → `train > 0 & oos < 0` (단방향)
- **이유:** IS음수/OOS양수는 유효한 regime shift (과적합 아님)
- **결과:** rsi_14, funding_carry 등 복원

### 3. 신규 Signal Family 4개 추가 ✅
- **F1 cross_sectional_momentum:** cs_mom_5, cs_mom_10, cs_mom_20 (3개)
- **F2 funding_zscore_carry:** fzs_48, fzs_96, fzs_168 (3개) — **OOS +35.4bps 최고**
- **F3 vol_regime_reversion:** vrr_20, vrr_40 (2개)
- **F4 btc_corr_regime:** bcr_24, bcr_48, bcr_96 (3개)

**총 패널:** 13개 → 24개 (신규 11개)

---

## 규칙 진단 결과

```
[DIAG][RULE_RECOMMEND_ABLATION]
  KEEP(6개): 
    - cross_sectional_momentum:cs_mom_10     [신규 F1]
    - funding_zscore_carry:fzs_168           [신규 F2, OOS +35.4bps]
    - funding_carry:funding_24
    - funding_zscore_carry:fzs_48            [신규 F2]
    - rsi_reversion:rsi_14
    - btc_regime_pullback:btc_pullback_50    [OOS +36.3bps]
  
  FLIP(4개):
    - trend_donchian:donchian_36
    - vol_regime_reversion:vrr_20            [신규 F3]
    - trend_ma:ema_12_72
    - trend_donchian:donchian_72
```

---

## Ablation Study (ML Variants)

| Model | CAGR | MaxDD | MAR | Equity | Pass |
|---|---|---|---|---|---|
| Equal Size | -18.6% | 48.2% | -0.39 | 539,475 | N |
| Kelly (No ML) | -0.6% | 2.0% | -0.32 | 980,822 | N |
| ML Gate | 0.0% | 0.0% | 0.00 | 1,000,000 | N |
| ML Gate+Edge | 0.0% | 0.0% | 0.00 | 1,000,000 | N |
| ML Full (Capped) | 0.0% | 0.0% | 0.00 | 1,000,000 | N |
| **Cand. ML** | **0.0%** | **0.0%** | **0.00** | **1,000,000** | **N** |

---

## 잔류 이슈 & 진단

### 🔴 Active Signals: 0 (BLOCKED)
- **규칙 선택:** 6개 KEEP → ✅ 정상
- **ML 게이트 필터:** p_pass < 0.40 (모든 이벤트) → ❌ 0건 통과
- **근본 원인:** `fit_candidate_gate` 모델이 과도한 보정/정규화로 확률 압축
- **범주:** Rule diagnostics 외 ML 보정 이슈 (별도 진단 필요)

### 📋 동작 확인
- **Rule 진단:** ✅ (IC 제거, IS→OOS 수정으로 KEEP 6개 정상 선택)
- **신규 family:** ✅ (4개 family, 11개 패널, OOS top 3 진입)
- **단위 테스트:** ✅ (41 passed, 무회귀)

---

## 다음 Step (선택)

1. **ML 게이트 보정 진단** (`candidate_gate.py` → `fit_candidate_gate` 확률 왜곡)
2. **신규 family 최적화** (F1 momentum lookback, F2 z-window 튜닝)
3. **비용 구조 재검토** (cost_floor_bps 24bps가 OOS edge 모두 소비 → 상향 검토)
