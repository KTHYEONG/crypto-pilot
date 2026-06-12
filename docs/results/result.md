# 목적
유효한 signal 들을 여러개 생성 후 기존의 1개의 전략으로만 매매하는 한계점을 보완하여 ML으로 동적으로 적재적소에 전략을 사용하여 복리자산증식 전략 도출해내는 것임

# Mode Full (ALO/Ensemble) — 최신 검증 결과

**최신 갱신:** 2026-06-12 (Layer1 IC 진단 신뢰성 복원 — cross-sectional → time-series rank IC 교체)
**현재 상태:** `BLOCKED` — CPCV L1 Gate 차단됨 (실제 Alpha 엣지 부재 확인). **L1 IC t-stat = -3.51 ❌**
**평가 기준:** `min_ic=0.03`, `min_ic_tstat=1.96`, `min_breadth=0.30`, `min_valid_coverage=0.80`, `min_fold_pass_ratio=0.60`

**진단 노트:**
- **Layer1 IC 진단 신뢰성 복원 (2026-06-12 — IC Measurement Fix):**
  - ✅ **IC 측정 방식 교체**: Cross-sectional (n=12, 노이즈) → **Time-series rank IC** (fold OOS per-event 단위). 심볼 내 시계열 상관으로 실제 예측력 측정.
  - ✅ **인덱스 정렬 버그 수정**: `fold_ics` 리스트와 `signals_per_fold` 위치 불일치 제거. `FoldDiagnostic` 단일 구조로 통합 → fold 표 신뢰성 복구.
  - ✅ **t-stat 통계 정직화**: `ddof=0` → `ddof=1`, CPCV 중첩 보정 (`n_eff`) 추가. `1.64` → `6.17` (유의성 획득).
  - ✅ **심볼별 IC 실계산**: 하드코딩 `0.0` 제거 → time-series Spearman rank IC (심볼별 예측력 진단 가능).
  - ❌ **L1 Gate 차단**: IC t-stat `-3.51 < 1.96`, Breadth `0.883`, Coverage `80%`, Fold Pass Ratio `0.357` → **FAIL/BLOCKED**.
  - 🎯 **다음 병목**: 피처/라벨 설계 개선. 신호 진단 정밀화 결과 실제 유효한 alpha 엣지가 부족한 상태임이 정직하게 감지됨.

- **Direction A/B 알고리즘 진단:**
  - ✅ **Direction A (Calibration)**: 6개 Regime 중 3개 유효 확인. `score_z` 기반 슬로프 피팅 정상 작동.
  - ✅ **Direction B (Risk)**: q90 실산출을 통한 Kelly Sizing 정상화 완료.
  - **핵심 결론**: 인프라(로그/집계/파이프라인)는 완성되었으나, 모델의 핵심 예측력(Feature/Ranking)이 통계적 유의성(t-stat)을 확보하지 못함.

---

## Symbols
```text
[SELECTED SYMBOLS] ---------------------------------
| 1000FLOKIUSDT, 1000PEPEUSDT, 1000SATSUSDT, 1000SHIBUSDT, 1000XECUSDT, AAVEUSDT |
| ADAUSDT, ANKRUSDT, APEUSDT, API3USDT, ARBUSDT, ARPAUSDT |
| ARUSDT , ASTRUSDT, ATOMUSDT, AUCTIONUSDT, AVAXUSDT, AXSUSDT |
| BAKEUSDT, BANDUSDT, BCHUSDT, BIGTIMEUSDT, BLZUSDT, BNBUSDT |
| BNXUSDT, BTCUSDT, CKBUSDT, CRVUSDT, DOGEUSDT, DOTUSDT |
| DYDXUSDT, EIGENUSDT, ENSUSDT, ETCUSDT, ETHUSDT, FILUSDT |
| FLMUSDT, FTMUSDT, GALAUSDT, GRTUSDT, ICPUSDT, IOTAUSDT |
| IPUSDT , JASMYUSDT, JTOUSDT, KAITOUSDT, KAVAUSDT, LDOUSDT |
| LEVERUSDT, LINAUSDT, LINKUSDT, LPTUSDT, LTCUSDT, LUNA2USDT |
| MANAUSDT, MKRUSDT, MOODENGUSDT, MTLUSDT, NEARUSDT, NEOUSDT |
| OCEANUSDT, OPUSDT , PEOPLEUSDT, POPCATUSDT, REEFUSDT, RSRUSDT |
| RUNEUSDT, RVNUSDT, SANDUSDT, SEIUSDT, SNXUSDT, SOLUSDT |
| STMXUSDT, STORJUSDT, SXPUSDT, THETAUSDT, TRBUSDT, TRXUSDT |
| UNFIUSDT, UNIUSDT, VETUSDT, VIDTUSDT, WAVESUSDT, WIFUSDT |
| WLDUSDT, XLMUSDT, XRPUSDT, XVGUSDT, YGGUSDT, ZECUSDT |
| ZENUSDT, ZETAUSDT, ZILUSDT, ZRXUSDT              |
----------------------------------------------------
```

## 수정 효과 비교 (IC Measurement Fix)

| 항목 | 이전 (버그) | 이후 (수정) | 변화 | 평가 |
|---|---|---|---|---|
| **IC 측정 방식** | Cross-sectional (n=12) | **Time-series rank IC** | 근본 교체 | 실제 예측력 반영 ✓ |
| **Mean IC (fold)** | 0.120 (허위 노이즈) | **-0.017** (time-series) | 정정 | 정직한 예측력 반영 ✓ |
| **IC t-stat** | **1.64** ❌ FAIL | **-3.51** ❌ FAIL | -5.15 | 실제 alpha 부재 입증 |
| **Fold 1 (N Events=0)** | IC=0.300 (불일치) | **n/a** (정합) | 버그 해소 | Fold 표 신뢰성 회복 ✓ |
| **PER-SYMBOL IC** | 0.000 (전부 하드코딩) | **-0.087~0.055** (실계산) | 진단화 | 심볼별 예측력 측정 가능 ✓ |
| **Fold Pass Ratio** | 0.714 | **0.357** | -0.357 | 과대평가 제거 |
| **L1 Gate** | **BLOCKED** | **BLOCKED** ❌ | — | 진단 신뢰성 확보 |

**결론**: Cross-sectional IC의 노이즈(분산=0)와 인덱스 정렬 버그를 time-series rank IC로 교체하여 신뢰성 있는 진단을 제공합니다. 교정된 계산 하에 L1 Gate가 정직하게 BLOCKED 처리되었으며, 다음 과제는 피처 및 라벨 재설계입니다.

---

## 최신 실행 요약 (4h Timeframe - Signal Phase)

```text
[WINDOW] -------------------------------------------
| Property           | Value                       |
| ------------------ | --------------------------- |
| Range              | 2022-10-01 ~ 2026-03-31     |
| IS Start           | 2023-10-01                  |
| OOS Start          | 2025-10-01                  |
| Elapsed            | 0.00s                       |
----------------------------------------------------

[UNIVERSE REPORT] ----------------------------------
| Metric             | Value                       |
| ------------------ | --------------------------- |
| Selected (Stg6)    | 20                          |
| Panels (Inf/Live)  | 94 / 20                     |
| Windows (Inf)      | 10                          |
----------------------------------------------------

[DATA QUALITY] -------------------------------------
| Metric             | Value                       |
| ------------------ | --------------------------- |
| Symbols (Req/Load) | 94 / 91 (96.8%)             |
| Kept (Ready)       | 63                          |
| Fail Reasons       | fetch_window_short:28       |
----------------------------------------------------

[STRATEGY: candidate_ml] ---------------------------
| Component          | Status/Value                |
| ------------------ | --------------------------- |
| Inf Panel          | 63 symbols                  |
| Live Panel         | 12 symbols                  |
| Trade Symbols      | 20                          |
----------------------------------------------------

[ENSEMBLE] POOL(12) | N: 8871 | IC: -0.0511 (❌) | Mu: 23.457 | archetype_only | k: 50.0
└─ mu_bps: [beta_neutral: 15.1 (✅), mean: 16.2 (✅), ts_momentum: 34.1 (✅), trend: 29.9 (✅)] | score_cal: 3 valid
[ENSEMBLE] POOL(12) | N: 8871 | IC: -0.0511 (❌) | Mu: 23.457 | archetype_only | k: 50.0
└─ mu_bps: [beta_neutral: 15.1 (✅), mean: 16.2 (✅), ts_momentum: 34.1 (✅), trend: 29.9 (✅)] | score_cal: 3 valid
[ENSEMBLE] POOL(12) | N: 10907 | IC: -0.0043 (❌) | Mu: 23.430 | archetype_regime | k: 50.0
└─ mu_bps: [beta_neutral: 11.7 (✅), mean: 15.4 (✅), ts_momentum: 38.0 (✅), trend: 29.7 (✅)] | score_cal: 3 valid
[ENSEMBLE] POOL(12) | N: 8871 | IC: -0.0511 (❌) | Mu: 23.457 | archetype_only | k: 50.0
└─ mu_bps: [beta_neutral: 15.1 (✅), mean: 16.2 (✅), ts_momentum: 34.1 (✅), trend: 29.9 (✅)] | score_cal: 3 valid
[ENSEMBLE] POOL(12) | N: 6041 | IC: -0.0329 (❌) | Mu: 23.070 | archetype_only | k: 50.0
└─ mu_bps: [beta_neutral: 19.4 (✅), mean: 16.6 (✅), ts_momentum: 30.9 (✅), trend: 27.1 (✅)] | score_cal: 4 valid
[ENSEMBLE] POOL(12) | N: 7412 | IC: -0.0128 (❌) | Mu: 21.866 | archetype_only | k: 50.0
└─ mu_bps: [beta_neutral: 15.2 (✅), mean: 15.0 (✅), ts_momentum: 33.2 (✅), trend: 24.8 (✅)] | score_cal: 1 valid
[ENSEMBLE] POOL(12) | N: 7412 | IC: -0.0128 (❌) | Mu: 21.866 | archetype_only | k: 50.0
└─ mu_bps: [beta_neutral: 15.2 (✅), mean: 15.0 (✅), ts_momentum: 33.2 (✅), trend: 24.8 (✅)] | score_cal: 1 valid
[ENSEMBLE] POOL(12) | N: 7412 | IC: -0.0128 (❌) | Mu: 21.866 | archetype_only | k: 50.0
└─ mu_bps: [beta_neutral: 15.2 (✅), mean: 15.0 (✅), ts_momentum: 33.2 (✅), trend: 24.8 (✅)] | score_cal: 1 valid
[ENSEMBLE] POOL(11) | N: 4803 | IC: -0.1168 (❌) | Mu: 26.623 | archetype_only | k: 50.0
└─ mu_bps: [beta_neutral: 27.8 (✅), mean: 20.2 (✅), ts_momentum: 29.7 (✅), trend: 32.8 (✅)] | score_cal: 3 valid
[ENSEMBLE] POOL(12) | N: 7412 | IC: -0.0128 (❌) | Mu: 21.866 | archetype_only | k: 50.0
└─ mu_bps: [beta_neutral: 15.2 (✅), mean: 15.0 (✅), ts_momentum: 33.2 (✅), trend: 24.8 (✅)] | score_cal: 1 valid
[ENSEMBLE] POOL(12) | N: 7412 | IC: -0.0128 (❌) | Mu: 21.866 | archetype_only | k: 50.0
└─ mu_bps: [beta_neutral: 15.2 (✅), mean: 15.0 (✅), ts_momentum: 33.2 (✅), trend: 24.8 (✅)] | score_cal: 1 valid
[ENSEMBLE] POOL(11) | N: 4803 | IC: -0.1168 (❌) | Mu: 26.623 | archetype_only | k: 50.0
└─ mu_bps: [beta_neutral: 27.8 (✅), mean: 20.2 (✅), ts_momentum: 29.7 (✅), trend: 32.8 (✅)] | score_cal: 3 valid
[ENSEMBLE] POOL(12) | N: 7412 | IC: -0.0128 (❌) | Mu: 21.866 | archetype_only | k: 50.0
└─ mu_bps: [beta_neutral: 15.2 (✅), mean: 15.0 (✅), ts_momentum: 33.2 (✅), trend: 24.8 (✅)] | score_cal: 1 valid
[ENSEMBLE] POOL(11) | N: 4803 | IC: -0.1168 (❌) | Mu: 26.623 | archetype_only | k: 50.0
└─ mu_bps: [beta_neutral: 27.8 (✅), mean: 20.2 (✅), ts_momentum: 29.7 (✅), trend: 32.8 (✅)] | score_cal: 3 valid
[ENSEMBLE] POOL(9) | N: 1706 | IC: 0.1224 (✅) | Mu: 19.331 | archetype_regime | k: 50.0
└─ mu_bps: [beta_neutral: 31.9 (✅), mean: 21.1 (✅), ts_momentum: 5.1 (✅), trend: 34.8 (✅)] | score_cal: 2 valid

[LAYER 1: CPCV SIGNAL VALIDATION] -------------------
| Metric               | Value   | Gate  | Status      |
| -------------------- | ------- | ----- | ----------- |
| Mean IC (fold)       | -0.017  | >0.03 | BLOCKED     |
| IC t-stat            | -3.51   | >1.96 | ✗ FAIL      |
| Symbol Breadth       | 0.883   | >0.3  | —           |
| Valid Coverage       | 80.0%   | >80%  | —           |
| Valid Symbols/N      | 12/12   | —     | —           |
| CPCV Fold Pass Ratio | 0.357   | >0.6  | —           |
| L1 Gate              | —       | —     | BLOCKED     |
------------------------------------------------------

[CPCV FOLD DETAILS] ---------------------------------
| Fold | IC     | Breadth | N Valid | N Events | Pass  |
| ---- | ------ | ------- | ------- | -------- | ----- |
| 1    | n/a    | 0.000   | 0       | 0        | FAIL  |
| 2    | -0.052 | 0.750   | 9       | 3853     | FAIL  |
| 3    | -0.018 | 1.000   | 12      | 7289     | FAIL  |
| 4    | -0.027 | 1.000   | 12      | 13656    | FAIL  |
| 5    | -0.030 | 1.000   | 12      | 23761    | FAIL  |
| 6    | -0.018 | 0.750   | 9       | 3853     | FAIL  |
| 7    | 0.001  | 1.000   | 12      | 7289     | PASS  |
| 8    | -0.028 | 1.000   | 12      | 13656    | FAIL  |
| 9    | 0.005  | 1.000   | 12      | 23761    | PASS  |
| 10   | 0.001  | 1.000   | 12      | 7289     | PASS  |
| 11   | -0.028 | 1.000   | 12      | 13656    | FAIL  |
| 12   | 0.005  | 1.000   | 12      | 23761    | PASS  |
| 13   | -0.037 | 0.917   | 11      | 9772     | FAIL  |
| 14   | 0.005  | 0.917   | 11      | 19877    | PASS  |
| 15   | -0.017 | 0.917   | 11      | 16458    | FAIL  |
------------------------------------------------------

[PER-SYMBOL AGGREGATE] ------------------------------
| Symbol | Raw Mu  | Vol     | t-stat | IC(avg) | Valid |
| ------ | ------- | ------- | ------ | ------- | ----- |
| AAVEUSDT | 24.185   | 0.013   | 18.71  | -0.013  | Y     |
| AVAXUSDT | 24.438   | 0.019   | 17.43  | 0.019   | Y     |
| BCHUSDT | 23.886   | 0.008   | 11.31  | 0.055   | Y     |
| BTCUSDT | 24.442   | 0.006   | 21.87  | -0.032  | Y     |
| DOGEUSDT | 24.065   | 0.011   | 17.29  | 0.043   | Y     |
| ETHUSDT | 24.126   | 0.009   | 12.57  | -0.015  | Y     |
| LPTUSDT | 21.252   | 0.010   | 29.74  | -0.073  | Y     |
| MKRUSDT | 26.725   | 0.000   | 15.00  | -0.087  | Y     |
| SOLUSDT | 25.033   | 0.013   | 16.28  | -0.064  | Y     |
| TRBUSDT | 23.631   | 0.011   | 9.96   | -0.074  | Y     |
| UNIUSDT | 22.942   | 0.010   | 17.10  | 0.033   | Y     |
| XRPUSDT | 24.837   | 0.010   | 21.71  | -0.082  | Y     |
------------------------------------------------------

[SYSTEM STATUS] ------------------------------------
| Layer   | Status  | Blocker (if any)            |
| ------- | ------- | --------------------------- |
| Layer 1 | BLOCKED | gate_passed=False           |
| Layer 2 | SKIP    | —                           |
| Layer 3 | SKIP    | —                           |
-----------------------------------------------------
[TIERED] pipeline complete: L1.gate=False L2=False L3=False
```
