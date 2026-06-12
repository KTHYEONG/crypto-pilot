# 목적
유효한 signal 들을 여러개 생성 후 기존의 1개의 전략으로만 매매하는 한계점을 보완하여 ML으로 동적으로 적재적소에 전략을 사용하여 복리자산증식 전략 도출해내는 것임

# Mode Full (ALO/Ensemble) — 최신 검증 결과

**최신 갱신:** 2026-06-12 (동적 유니버스 L1 Breadth Scope 고정 — Tiered aligned 범위 교정, Method B)
**현재 상태:** `blocked` — CPCV L1 Gate 통과 실패. **Active Signals = 0**
**평가 기준:** `min_ic=0.03`, `min_ic_tstat=1.96`, `min_breadth=0.30`, `min_valid_coverage=0.80`, `min_fold_pass_ratio=0.60`

**진단 노트:**
- **Scope Mismatch 근본 해소 (2026-06-12 — Method B):**
  - ✅ **Breadth 분모 교정**: `aligned_tiered` scope를 `data_stage.valid_symbols(63)` → `Stage6 OOS ∩ data-valid(12)`로 축소. `Valid Symbols/N: 12/63` → `12/12`.
  - ✅ **Symbol Breadth 정상화**: `0.168` → `0.883` (0.30 게이트 통과 ✓).
  - ✅ **Valid Coverage 급상승**: `0.0%` → `93.3%` (0.80 게이트 통과 ✓).
  - ❌ **L1 Gate 여전히 BLOCKED**: `IC t-stat = 1.64 < 1.96` (alpha 신호 품질 문제). 범위 수정이 breadth/coverage 교정하나, 통계적 유의성 향상 불가능.
  - 🎯 **다음 병목**: Feature 예측력 강화 (t-stat ↑) 또는 훈련 데이터 기간 확장 필요. Scope 문제 아님.

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

## 수정 효과 비교 (Scope Mismatch Fix)

| 항목 | 이전 (버그 상태) | 이후 (수정) | 변화 | 게이트 평가 |
|---|---|---|---|---|
| **Valid Symbols/N** | 12/**63** | 12/**12** | 분모 교정 ✓ | — |
| **Symbol Breadth** | **0.168** | **0.883** | +0.715 | 0.30 **통과** ✓ |
| **Valid Coverage** | **0.0%** | **93.3%** | +93.3p | 80% **통과** ✓ |
| **Mean IC** | 0.120 | 0.120 | — | >0.03 **통과** ✓ |
| **IC t-stat** | 1.64 | 1.64 | — | >1.96 **FAIL** ✗ |
| **Fold Pass Ratio** | 0.714 | 0.714 | — | >0.6 **통과** ✓ |
| **실행 시간** | 53.28s | 30.24s | -42.1% | — |

**결론**: Breadth/Coverage 두 게이트를 교정했으나, IC t-stat(1.64 < 1.96)이 여전히 L1을 BLOCK. 이는 피처 예측력 부재로 인한 근본 문제 (scope 이슈 아님).

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

[ENSEMBLE] ETHUSDT | N: 8871 | IC: -0.0511 (❌) | Mu: 23.457 | archetype_only | k: 50.0
└─ mu_bps: [beta_neutral: 15.1 (✅), mean: 16.2 (✅), ts_momentum: 34.1 (✅), trend: 29.9 (✅)] | score_cal: 3 valid
[ENSEMBLE] ETHUSDT | N: 8871 | IC: -0.0511 (❌) | Mu: 23.457 | archetype_only | k: 50.0
└─ mu_bps: [beta_neutral: 15.1 (✅), mean: 16.2 (✅), ts_momentum: 34.1 (✅), trend: 29.9 (✅)] | score_cal: 3 valid
[ENSEMBLE] ETHUSDT | N: 8871 | IC: -0.0511 (❌) | Mu: 23.457 | archetype_only | k: 50.0
└─ mu_bps: [beta_neutral: 15.1 (✅), mean: 16.2 (✅), ts_momentum: 34.1 (✅), trend: 29.9 (✅)] | score_cal: 3 valid
[ENSEMBLE] ETHUSDT | N: 10907 | IC: -0.0043 (❌) | Mu: 23.430 | archetype_regime | k: 50.0
└─ mu_bps: [beta_neutral: 11.7 (✅), mean: 15.4 (✅), ts_momentum: 38.0 (✅), trend: 29.7 (✅)] | score_cal: 3 valid
[ENSEMBLE] ETHUSDT | N: 6041 | IC: -0.0329 (❌) | Mu: 23.070 | archetype_only | k: 50.0
└─ mu_bps: [beta_neutral: 19.4 (✅), mean: 16.6 (✅), ts_momentum: 30.9 (✅), trend: 27.1 (✅)] | score_cal: 4 valid
[ENSEMBLE] ETHUSDT | N: 7412 | IC: -0.0128 (❌) | Mu: 21.866 | archetype_only | k: 50.0
└─ mu_bps: [beta_neutral: 15.2 (✅), mean: 15.0 (✅), ts_momentum: 33.2 (✅), trend: 24.8 (✅)] | score_cal: 1 valid
[ENSEMBLE] ETHUSDT | N: 7412 | IC: -0.0128 (❌) | Mu: 21.866 | archetype_only | k: 50.0
└─ mu_bps: [beta_neutral: 15.2 (✅), mean: 15.0 (✅), ts_momentum: 33.2 (✅), trend: 24.8 (✅)] | score_cal: 1 valid
[ENSEMBLE] ETHUSDT | N: 7412 | IC: -0.0128 (❌) | Mu: 21.866 | archetype_only | k: 50.0
└─ mu_bps: [beta_neutral: 15.2 (✅), mean: 15.0 (✅), ts_momentum: 33.2 (✅), trend: 24.8 (✅)] | score_cal: 1 valid
[ENSEMBLE] ETHUSDT | N: 4803 | IC: -0.1168 (❌) | Mu: 26.623 | archetype_only | k: 50.0
└─ mu_bps: [beta_neutral: 27.8 (✅), mean: 20.2 (✅), ts_momentum: 29.7 (✅), trend: 32.8 (✅)] | score_cal: 3 valid
[ENSEMBLE] ETHUSDT | N: 7412 | IC: -0.0128 (❌) | Mu: 21.866 | archetype_only | k: 50.0
└─ mu_bps: [beta_neutral: 15.2 (✅), mean: 15.0 (✅), ts_momentum: 33.2 (✅), trend: 24.8 (✅)] | score_cal: 1 valid
[ENSEMBLE] ETHUSDT | N: 7412 | IC: -0.0128 (❌) | Mu: 21.866 | archetype_only | k: 50.0
└─ mu_bps: [beta_neutral: 15.2 (✅), mean: 15.0 (✅), ts_momentum: 33.2 (✅), trend: 24.8 (✅)] | score_cal: 1 valid
[ENSEMBLE] ETHUSDT | N: 4803 | IC: -0.1168 (❌) | Mu: 26.623 | archetype_only | k: 50.0
└─ mu_bps: [beta_neutral: 27.8 (✅), mean: 20.2 (✅), ts_momentum: 29.7 (✅), trend: 32.8 (✅)] | score_cal: 3 valid
[ENSEMBLE] ETHUSDT | N: 7412 | IC: -0.0128 (❌) | Mu: 21.866 | archetype_only | k: 50.0
└─ mu_bps: [beta_neutral: 15.2 (✅), mean: 15.0 (✅), ts_momentum: 33.2 (✅), trend: 24.8 (✅)] | score_cal: 1 valid
[ENSEMBLE] ETHUSDT | N: 4803 | IC: -0.1168 (❌) | Mu: 26.623 | archetype_only | k: 50.0
└─ mu_bps: [beta_neutral: 27.8 (✅), mean: 20.2 (✅), ts_momentum: 29.7 (✅), trend: 32.8 (✅)] | score_cal: 3 valid
[ENSEMBLE] ETHUSDT | N: 1706 | IC: 0.1224 (✅) | Mu: 19.331 | archetype_regime | k: 50.0
└─ mu_bps: [beta_neutral: 31.9 (✅), mean: 21.1 (✅), ts_momentum: 5.1 (✅), trend: 34.8 (✅)] | score_cal: 2 valid

[LAYER 1: CPCV SIGNAL VALIDATION] -------------------
| Metric               | Value   | Gate  | Status      |
| -------------------- | ------- | ----- | ----------- |
| Mean IC (HAC)        | 0.120   | >0.03 | ✓ PASS      |
| IC t-stat            | 1.64    | >1.96 | ✗ FAIL      |
| Symbol Breadth       | 0.883   | >0.3  | ✓ PASS      |
| Valid Coverage       | 93.3%   | >80%  | ✓ PASS      |
| Valid Symbols/N      | 12/12   | —     | —           |
| CPCV Fold Pass Ratio | 0.714   | >0.6  | ✓ PASS      |
| L1 Gate              | —       | —     | BLOCKED     |
------------------------------------------------------

[CPCV FOLD DETAILS] ---------------------------------
| Fold | IC     | Breadth | N Valid | N Events | Pass  |
| ---- | ------ | ------- | ------- | -------- | ----- |
| 1    | 0.300  | 0.000   | 0       | 0        | PASS  |
| 2    | 0.077  | 0.750   | 9       | 3853     | PASS  |
| 3    | 0.000  | 1.000   | 12      | 7289     | FAIL  |
| 4    | -0.448 | 1.000   | 12      | 13656    | FAIL  |
| 5    | -0.100 | 1.000   | 12      | 23761    | FAIL  |
| 6    | 0.133  | 0.750   | 9       | 3853     | PASS  |
| 7    | 0.329  | 1.000   | 12      | 7289     | PASS  |
| 8    | 0.413  | 1.000   | 12      | 13656    | PASS  |
| 9    | -0.399 | 1.000   | 12      | 23761    | FAIL  |
| 10   | 0.413  | 1.000   | 12      | 7289     | PASS  |
| 11   | 0.329  | 1.000   | 12      | 13656    | PASS  |
| 12   | 0.127  | 1.000   | 12      | 23761    | PASS  |
| 13   | 0.073  | 0.917   | 11      | 9772     | PASS  |
| 14   | 0.436  | 0.917   | 11      | 19877    | PASS  |
------------------------------------------------------

[PER-SYMBOL AGGREGATE] ------------------------------
| Symbol | Raw Mu  | Vol     | t-stat | IC(avg) | Valid |
| ------ | ------- | ------- | ------ | ------- | ----- |
| AAVEUSDT | 24.185 | 0.013 | 18.71 | 0.000 | Y |
| AVAXUSDT | 24.438 | 0.019 | 17.43 | 0.000 | Y |
| BCHUSDT  | 23.886 | 0.008 | 11.31 | 0.000 | Y |
| BTCUSDT  | 24.442 | 0.006 | 21.87 | 0.000 | Y |
| DOGEUSDT | 24.065 | 0.011 | 17.29 | 0.000 | Y |
| ETHUSDT  | 24.126 | 0.009 | 12.57 | 0.000 | Y |
| LPTUSDT  | 21.252 | 0.010 | 29.74 | 0.000 | Y |
| MKRUSDT  | 26.725 | 0.000 | 15.00 | 0.000 | Y |
| SOLUSDT  | 25.033 | 0.013 | 16.28 | 0.000 | Y |
| TRBUSDT  | 23.631 | 0.011 | 9.96  | 0.000 | Y |
| UNIUSDT  | 22.942 | 0.010 | 17.10 | 0.000 | Y |
| XRPUSDT  | 24.837 | 0.010 | 21.71 | 0.000 | Y |
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
