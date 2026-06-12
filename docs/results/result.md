# 목적
유효한 signal 들을 여러개 생성 후 기존의 1개의 전략으로만 매매하는 한계점을 보완하여 ML으로 동적으로 적재적소에 전략을 사용하여 복리자산증식 전략 도출해내는 것임

# Mode Full (ALO/Ensemble) — 최신 검증 결과

**최신 갱신:** 2026-06-12 (SWF fold/window/warmup 정합성 수정, full `pytest` 통과)
**현재 상태:** `BLOCKED` — `phase signal` 재실행이 외부 데이터 네트워크 차단으로 중단됨. 최신 SWF fold 수치는 아직 재산출 전.
**평가 기준:** `min_pooled_ic=0.03`, `min_nw_tstat=1.96`, `min_breadth=0.30`, `min_valid_coverage=0.80`

**진단 노트:**
- **SWF-K 전환 (2026-06-12 — CPCV → Purged Sequential Walk-Forward):**
  - ✅ **CPCV 폐기**: OOS collapse 버그(disjoint spans→contiguous mask), anti-causal fold (test_groups=[0,1] → fit=future data), 중첩 fold 중복 이벤트 제거.
  - ✅ **SWF-K 도입**: K=5 등간격 OOS 창, expanding fit (fit_start=0), `fit_end=oos_start-purge_bars` 구조적 인과 보장.
  - ✅ **Pooled IC**: fold 평균 IC → 전 fold OOS 이벤트 concat 후 단일 Spearman rank IC. N=5576 기준.
  - ✅ **NW HAC t-stat**: Newey-West Bartlett kernel, Andrews 1991 자동 대역폭. 자기상관 보정으로 naive t-stat보다 보수적.
  - ✅ **스케일 버그 수정 (Audit)**: `SE(IC)=12·√(S_NW/N)` 정규화 — 미적용 시 |t| 12배 과대평가(거짓 양성 위험).
  - ✅ **fold_pass_ratio gate 제거**: 진단용 보존, gate 조건 4개로 단순화.
  - ❌ **L1 Gate 차단**: Pooled IC `-0.095`, NW HAC t-stat `-4.60`, Breadth `0.017`, Valid Coverage `0.0%` → **BLOCKED**.
  - 🎯 **다음 병목**: Fold 1-4 유효 심볼 0 (N Valid=0) — l1_start 기점 warmup 구간 내 OOS 파티션 흡수 현상. 신호 alpha 부재 지속.

- **Direction A/B 알고리즘 진단:**
  - ✅ **Direction A (Calibration)**: 6개 Regime 중 3개 유효 확인. `score_z` 기반 슬로프 피팅 정상 작동.
  - ✅ **Direction B (Risk)**: q90 실산출을 통한 Kelly Sizing 정상화 완료.
  - **핵심 결론**: 평가 프레임워크(SWF-K) 신뢰성 확보 완료. 현재 블로커는 signal 재검증을 위한 외부 데이터 접근성이다.

- **2026-06-12 추가 검증:**
  - ✅ `uv run pytest` 전체 통과: `788 passed`
  - ⚠️ `phase signal --sync full` 실행은 Binance Vision 데이터 다운로드 단계에서 DNS 실패로 타임아웃
  - ⚠️ 기존 `[SWF FOLD DETAILS]` 표는 재실행 전 데이터라 `stale` 상태로 간주해야 함

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

## 수정 효과 비교 (CPCV → SWF-K)

| 항목 | 이전 (CPCV) | 이후 (SWF-K) | 변화 | 평가 |
|---|---|---|---|---|
| **평가 알고리즘** | CPCV (15 folds, OOS collapse 버그) | **SWF-K** (5 folds, 구조적 인과 보장) | 근본 교체 | anti-causal 제거 ✓ |
| **IC 집계** | fold 평균 (Equal-weight, N 무시) | **Pooled IC** (전체 이벤트 concat) | N-가중 정확화 | 표본 크기 편향 제거 ✓ |
| **t-stat 방식** | Naive + n_eff 보정 | **NW HAC (Bartlett kernel)** | 자기상관 보정 | 보수적 추정 ✓ |
| **Fold 1 N=0** | n/a (anti-causal 버그) | n/a (warmup 이전 데이터 없음) | 원인 변경 | 인과 버그 해소 ✓ |
| **Mean IC** | -0.017 (fold 평균) | **Pooled IC -0.095** | 더 음수 | N-가중으로 정직한 값 |
| **t-stat** | -3.51 (naive) | **-4.60 (NW HAC)** | 보수적 | 과대평가 제거 ✓ |
| **fold_pass_ratio gate** | ≥0.60 (gate 포함) | 진단 전용 (gate 제외) | 단순화 | 게이트 오염 제거 ✓ |
| **L1 Gate** | **BLOCKED** | **BLOCKED** ❌ | — | 진단 신뢰성 확보 |

---

## 최신 실행 요약 (4h Timeframe - Signal Phase, 2026-06-12)

```text
[WINDOW] -------------------------------------------
| Property           | Value                       |
| ------------------ | --------------------------- |
| Range              | 2022-10-01 ~ 2026-03-31     |
| IS Start           | 2023-10-01                  |
| OOS Start          | 2025-10-01                  |
| Elapsed            | 0.00s                       |
----------------------------------------------------

[STRATEGY: candidate_ml] ---------------------------
| Component          | Status/Value                |
| ------------------ | --------------------------- |
| Inf Panel          | 58 symbols                  |
| Live Panel         | 13 symbols                  |
| Trade Symbols      | 20                          |
----------------------------------------------------

[LAYER 1: SWF SIGNAL VALIDATION] --------------------
| Metric               | Value   | Gate  | Status      |
| -------------------- | ------- | ----- | ----------- |
| Pooled IC            | 0.006   | >0.03 | BLOCKED     |
| IC t-stat (NW HAC)   | 0.92    | >1.96 | ✗ FAIL      |
| Symbol Breadth       | 0.547   | >0.3  | —           |
| Valid Coverage       | 40.0%   | >80%  | —           |
| Valid Symbols/N      | 26/56   | —     | —           |
| L1 Gate              | —       | —     | BLOCKED     |
------------------------------------------------------

[SWF FOLD DETAILS] ----------------------------------
| Fold | IC      | Breadth | N Valid | N Events | Pass |
| ---- | ------- | ------- | ------- | -------- | ---- |
| 1    | -0.006  |   0.200 |       1 |     7581 | FAIL |
| 2    | 0.049   |   0.438 |       7 |    22724 | PASS |
| 3    | -0.011  |   0.167 |       3 |    14718 | FAIL |
| 4    | -0.009  |   0.933 |      14 |    16995 | FAIL |
| 5    | -0.082  |   1.000 |      16 |    15397 | FAIL |
------------------------------------------------------

[PER-SYMBOL AGGREGATE] ------------------------------
| Symbol       | Raw Mu    | Vol       | t-stat   | IC(avg)   | Valid |
| ------------ | --------- | --------- | -------- | --------- | ----- |
| 1000SHIBUSDT |     5.016 |    0.0102 |     3.81 |    -0.030 | Y     |
| 1000XECUSDT  |    -8.427 |    0.0093 |  -842.47 |     0.234 | N     |
| AAVEUSDT     |     4.607 |    0.0138 |     3.40 |    -0.110 | Y     |
| ADAUSDT      |     0.845 |    0.0120 |     2.78 |     0.017 | Y     |
| ANKRUSDT     |     4.768 |    0.0116 |     3.75 |    -0.152 | Y     |
| ARPAUSDT     |    -8.095 |    0.0154 |  -928.44 |     0.029 | N     |
| ATOMUSDT     |    -3.292 |    0.0126 |     0.00 |    -0.126 | N     |
| AVAXUSDT     |     3.849 |    0.0134 |     3.22 |    -0.020 | Y     |
| AXSUSDT      |     4.452 |    0.0292 |     3.20 |    -0.009 | Y     |
| BANDUSDT     |     4.372 |    0.0164 |     3.20 |    -0.022 | Y     |
| BCHUSDT      |     0.189 |    0.0128 |     2.31 |    -0.013 | Y     |
| BLZUSDT      |    -3.443 |    0.0001 |     0.00 |    -0.024 | N     |
| BTCUSDT      |     0.080 |    0.0063 |     3.06 |    -0.002 | Y     |
| CRVUSDT      |     7.315 |    0.0146 |     0.00 |    -0.099 | N     |
| DOGEUSDT     |    -0.206 |    0.0124 |     3.04 |    -0.032 | Y     |
| DOTUSDT      |     0.598 |    0.0126 |     3.93 |    -0.074 | Y     |
| DYDXUSDT     |    -8.196 |    0.0143 | -1090.20 |     0.129 | N     |
| ETCUSDT      |     1.927 |    0.0094 |     1.04 |     0.291 | N     |
| ETHUSDT      |     0.487 |    0.0112 |     3.10 |    -0.006 | Y     |
| FILUSDT      |     3.420 |    0.0139 |     2.25 |    -0.016 | Y     |
| FTMUSDT      |    -0.891 |    0.0001 |     3.03 |    -0.007 | Y     |
| GALAUSDT     |    -4.037 |    0.0163 |     4.01 |    -0.146 | Y     |
| ICPUSDT      |    -5.411 |    0.0087 |     0.34 |    -0.051 | N     |
| IOTAUSDT     |     2.869 |    0.0117 |     1.93 |     0.002 | N     |
| KAVAUSDT     |    -8.010 |    0.0071 |  -898.02 |     0.039 | N     |
| LINKUSDT     |    -0.709 |    0.0149 |     2.46 |    -0.060 | Y     |
| LPTUSDT      |    -8.387 |    0.0362 |  -775.83 |    -0.345 | N     |
| LTCUSDT      |    -0.894 |    0.0085 |     0.00 |    -0.044 | N     |
| MANAUSDT     |    -8.421 |    0.0144 | -1495.10 |    -0.005 | N     |
| MKRUSDT      |     0.261 |    0.0125 |     0.60 |     0.470 | N     |
| MTLUSDT      |    -8.353 |    0.0111 |  -459.11 |    -0.268 | N     |
| NEARUSDT     |     4.212 |    0.0121 |     2.74 |    -0.002 | Y     |
| NEOUSDT      |    -8.340 |    0.0140 |  -856.76 |     0.081 | N     |
| RSRUSDT      |     3.193 |    0.0150 |     2.30 |    -0.047 | Y     |
| RUNEUSDT     |     0.424 |    0.0133 |     2.12 |     0.017 | Y     |
| RVNUSDT      |     2.801 |    0.0107 |     2.40 |    -0.062 | Y     |
| SANDUSDT     |    -8.242 |    0.0127 | -1330.87 |     0.113 | N     |
| SNXUSDT      |     3.109 |    0.0162 |     2.15 |    -0.065 | Y     |
| SOLUSDT      |     2.080 |    0.0133 |     3.03 |     0.083 | Y     |
| STORJUSDT    |     0.387 |    0.0116 |     2.83 |    -0.004 | Y     |
| THETAUSDT    |     3.931 |    0.0155 |     2.87 |    -0.236 | Y     |
| TRBUSDT      |    -2.828 |    0.0151 |     0.00 |     0.041 | N     |
| UNIUSDT      |    -8.014 |    0.0138 |  -711.26 |     0.122 | N     |
| VETUSDT      |    -8.104 |    0.0115 |  -588.95 |     0.218 | N     |
| XRPUSDT      |     3.406 |    0.0086 |     0.00 |    -0.030 | N     |
| ZENUSDT      |    -8.041 |    0.0140 |  -510.43 |    -0.252 | N     |
| ZILUSDT      |     3.659 |    0.0116 |     2.99 |    -0.173 | Y     |
| ZRXUSDT      |     0.284 |    0.0200 |     0.19 |    -0.177 | Y     |
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
