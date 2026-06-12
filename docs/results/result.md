# 목적
유효한 signal 들을 여러개 생성 후 기존의 1개의 전략으로만 매매하는 한계점을 보완하여 ML으로 동적으로 적재적소에 전략을 사용하여 복리자산증식 전략 도출해내는 것임

## Symbols
```text
[SELECTED SYMBOLS] ---------------------------------
| 1000FLOKIUSDT, 1000LUNCUSDT, 1000PEPEUSDT, 1000SATSUSDT, 1000SHIBUSDT, 1000XECUSDT |
| AAVEUSDT, ADAUSDT, ANKRUSDT, API3USDT, ARBUSDT, ARPAUSDT |
| ARUSDT , ASTRUSDT, ATOMUSDT, AUCTIONUSDT, AVAXUSDT, AXSUSDT |
| BAKEUSDT, BANDUSDT, BCHUSDT, BIGTIMEUSDT, BIOUSDT, BLZUSDT |
| BNBUSDT, BNXUSDT, BTCUSDT, CKBUSDT, CRVUSDT, DOGEUSDT |
| DOTUSDT, DYDXUSDT, EIGENUSDT, ENSUSDT, ETCUSDT, ETHUSDT |
| FILUSDT, FLMUSDT, FTMUSDT, GALAUSDT, ICPUSDT, IOTAUSDT |
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

[DATA-INTEGRITY] Starting market data integrity check for 56 symbols...
[DATA-INTEGRITY] PASS: 56/56 symbols passed. (Bars: 7518, NaN: 0.0%, Zero/Neg: 0.0%, Hi>=Lo: PASS)

[ENSEMBLE] POOL(2) | N: 1954 | IC: 0.0194 (✅) | Mu: -6.654 | archetype_only | k: 50.0
└─ mu_bps: [beta_neutral: 6.5 (✅), mean: -7.3 (❌), position_unwind: -8.1 (❌), ts_momentum: 1.3 (✅), trend: -13.4 (❌)] | score_cal: 1 valid
[ENSEMBLE] POOL(18) | N: 18311 | IC: 0.0402 (✅) | Mu: 0.079 | archetype_regime | k: 50.0
└─ mu_bps: [beta_neutral: 33.4 (✅), forced_flow_reversal: -14.4 (❌), mean: -4.4 (❌), position_unwind: 17.1 (✅), ts_momentum: 8.9 (✅), trend: 16.4 (✅)] | score_cal: 3 valid
[ENSEMBLE] POOL(5) | N: 6707 | IC: -0.0620 (❌) | Mu: -4.978 | archetype_only | k: 50.0
└─ mu_bps: [beta_neutral: 11.0 (✅), mean: -7.4 (❌), position_unwind: 16.4 (✅), ts_momentum: 2.0 (✅), trend: 1.9 (✅)] | score_cal: 0 valid
[ENSEMBLE] POOL(33) | N: 34706 | IC: 0.0681 (✅) | Mu: 1.255 | archetype_only | k: 50.0
└─ mu_bps: [beta_neutral: 37.4 (✅), forced_flow_reversal: -7.8 (❌), mean: -3.2 (❌), position_unwind: 14.9 (✅), ts_momentum: 6.1 (✅), trend: 12.6 (✅)] | score_cal: 2 valid
[ENSEMBLE] POOL(34) | N: 49306 | IC: 0.0005 (✅) | Mu: 3.175 | archetype_only | k: 50.0
└─ mu_bps: [beta_neutral: 40.9 (✅), forced_flow_reversal: -5.0 (❌), mean: -1.3 (❌), position_unwind: 5.4 (✅), ts_momentum: 15.0 (✅), trend: 7.5 (✅)] | score_cal: 1 valid

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
| FILUSDT      |     3.420 |    0.0139 |     2.25 |    -0.017 | Y     |
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
| THETAUSDT    |     3.931 |    0.0155 |     2.87 |    -0.237 | Y     |
| TRBUSDT      |    -2.828 |    0.0151 |     0.00 |     0.041 | N     |
| UNIUSDT      |    -8.014 |    0.0138 |  -711.26 |     0.122 | N     |
| VETUSDT      |    -8.104 |    0.0115 |  -588.95 |     0.218 | N     |
| XRPUSDT      |     3.406 |    0.0086 |     0.00 |    -0.030 | N     |
| ZENUSDT      |    -8.041 |    0.0140 |  -510.43 |    -0.252 | N     |
| ZILUSDT      |     3.659 |    0.0116 |     2.99 |    -0.173 | Y     |
| ZRXUSDT      |    -1.195 |    0.0200 |     2.43 |    -0.081 | Y     |
------------------------------------------------------

[SYSTEM STATUS] ------------------------------------
| Layer   | Status  | Blocker (if any)            |
| ------- | ------- | --------------------------- |
| Layer 1 | BLOCKED | gate_passed=False           |
| Layer 2 | SKIP    | —                           |
| Layer 3 | SKIP    | —                           |
-----------------------------------------------------
[TIERED] pipeline complete: L1.gate=False L2=False L3=False
