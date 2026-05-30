# 선물 ML Alpha 분석 및 진단 결과 보고서

## 1. 개요 및 최종 판정
- **ALPHA_PASS**: **TRUE** ✅ (Phase 1 레지듀얼 IC 및 필터 통과)
  - `resid_ic=0.0141` (합격)
  - `be_eff=0.0131` (gap=+0.0010, 합격)
  - `t_stat=3.62` (>= 2.0, 합격)
  - `bear_ic=0.0240` (>= 0.0, 합격)
  - `dsr=1.000` (>= 0.95, 합격)
- **EXEC_DIAG**: **FAIL** ❌ (포트폴리오 집행 게이트 미통과)
  - 실패 사유: `['port_ic_non_positive', 'port_ic_below_raw_breakeven', 'basket_net_bps_non_positive']`
  - `port_ic=-0.0030`, `be_raw=0.0386`, `gap_raw=-0.0416`, `basket_net_bps=-36.94`

---

## 2. 세부 진단 데이터

### 2.1 원본 및 레지듀얼 시그널 통계 (Raw vs. Resid)
- `raw_cs_std`: 0.0369
- `resid_cs_std`: 0.0347
- `var_retention`: 0.941
- `n_ts`: 3704
- `raw_nz`: 0.979
- `resid_nz`: 1.000

### 2.2 데이터 무결성 게이트 (Data Integrity)
- `zero_price`: 0.000000
- `ohlc_violation`: 0.000000
- `bar_gap`: 0
- `nan_decomposition`:
  - `universe_inactive`: 99.81%
  - `warmup`: 0.19%
  - `price_missing`: 0.00%
  - `kill`: 0.00%

### 2.3 피처 무결성 및 선택 (Feature Integrity & Selection)
- **전체 요약**: 56개 중 26개 선택됨 (kept=26/56)
- **감지된 이슈**:
  - `constant` (9개): `['basis_1', 'basis_mean_6', 'execution_cost_rank', 'oi_ret_1', 'oi_z_18', 'basis_missing_ind', 'oi_missing_ind', 'adv_missing_ind', 'execution_cost_missing_ind']`
  - `drifted` (5개)
  - `redundant` (8개): `[('ret_3', 'rev_3', -1.0), ('ret_6', 'rev_6', -1.0), ('cs_rank_ret_18', 'cs_sharpe_18', 0.961), ('dollar_volume_rank', 'adv_rank', 1.0), ...]`
  - `leakage` (0개)
- **Kept Features**: `['ret_1', 'ret_3', 'ret_6', 'ret_12', 'ret_18', 'ret_36', 'momentum_autocorr', 'cs_residual_momentum', 'vwap_deviation', 'rv_6', 'rv_18', 'rv_36', 'downside_rv_18', 'atr_pct_14', 'vol_of_vol_36', 'funding_z_30d', 'funding_sign_persistence_6', 'funding_rate_momentum', 'volume_z_18', 'btc_ret_6', 'market_median_ret_6', 'market_dispersion_6', 'positive_breadth_6', 'micro_hl_spread_1', 'micro_close_to_hl_1', 'funding_missing_ind']`

---

## 3. 학습 및 OOS 성과 (ML Training & OOS Performance)
- **ML 학습 소요 시간**: 3개 fold 병렬 학습 완료 (소요시간: 22,937.49 ms)
- **OOS 평가 (Out-Of-Sample)**:
  - `dense_ranker ic`: 0.0454 (t-stat=5.10, hit_ratio=0.536, breadth=3.7)
  - `N_raw`: 17.0, `N_eff`: 1.5
  - `sigma_r`: 666.3 bps
  - `be_raw`: 0.0116, `be_eff`: 0.0174
  - `gap_resid_eff`: +0.0273 (초과 성과 달성)

### 3.1 레지메별 성과 (Regime IC)
- **Bull**: 0.011
- **Bear**: 0.024
- **Chop**: -0.006

---

## 4. 최종 스코어보드 (Alpha Scoreboard)

| Metric | RESID_IC | T-STAT | N_EFF | DSR | BE_EFF(12h) | BEAR_IC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Value** | 0.0141 | 3.62 | 15.0 | 1.0000 | 0.0131 | 0.0240 |
| **Result** | ✅ Pass | ✅ Pass | 15.0 (N_eff) | ✅ Pass | ✅ (gap=+10.3bps) | ✅ Pass |

---

## 5. 집행 포트폴리오 진단 결과 (EXEC_DIAG)
- **Basket PnL**: `ew_bps=-12.94`, `net_bps=-36.94` (IR t-stat=-2.61, hit=0.494)
- **Horizon Sweep**:
  - `6h`: ic=-0.0043 (breakeven=0.0369) ❌ Fail
  - `12h`: ic=-0.0065 (breakeven=0.0263) ❌ Fail
  - `18h`: ic=-0.0080 (breakeven=0.0216) ❌ Fail
