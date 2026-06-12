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

[DATA QUALITY] -------------------------------------
| Metric             | Value                       |
| ------------------ | --------------------------- |
| Symbols (Req/Load) | 94 / 90 (95.7%)             |
| Kept (Ready)       | 58                          |
| Fail Reasons       | fetch_window_short:32       |
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
[BRIDGE-PROF] total=15.7654s align=0.0836s rules=2.7175s events=1.4689s label=6.4970s diagnostics=2.2620s promotions=0.0451s walk_forward=0.0000s post_wf=0.0000s selection=0.0000s weights=0.0000s alpha_panel=2.6223s accounted=15.6964s unaccounted=0.0689s
[TIERED] USE_CS_RANK_ENGINE=True — entering Tiered pipeline
[TIERED] aligned scope: 56 symbols (historical union ∩ data-valid)
[SCORE-CAL-DIAG] valid=1/4 obs_too_low=0 neg_slope_or_oos_fail=3 (min_obs=60)
[ENSEMBLE] POOL(2) | N: 1954 | IC: 0.0194 (✅) | Mu: -6.654 | archetype_only | k: 50.0
└─ mu_bps: [beta_neutral: 6.5 (✅), mean: -7.3 (❌), position_unwind: -8.1 (❌), ts_momentum: 1.3 (✅), trend: -13.4 (❌)] | score_cal: 1 valid
[SCORE-CAL-DIAG] valid=0/6 obs_too_low=0 neg_slope_or_oos_fail=6 (min_obs=60)
[ENSEMBLE] POOL(5) | N: 6707 | IC: -0.0620 (❌) | Mu: -4.978 | archetype_only | k: 50.0
└─ mu_bps: [beta_neutral: 11.0 (✅), mean: -7.4 (❌), position_unwind: 16.4 (✅), ts_momentum: 2.0 (✅), trend: 1.9 (✅)] | score_cal: 0 valid
[SCORE-CAL-DIAG] valid=3/6 obs_too_low=0 neg_slope_or_oos_fail=3 (min_obs=60)
[ENSEMBLE] POOL(18) | N: 18311 | IC: 0.0402 (✅) | Mu: 0.079 | archetype_regime | k: 50.0
└─ mu_bps: [beta_neutral: 33.4 (✅), forced_flow_reversal: -14.4 (❌), mean: -4.4 (❌), position_unwind: 17.1 (✅), ts_momentum: 8.9 (✅), trend: 16.4 (✅)] | score_cal: 3 valid
[SCORE-CAL-DIAG] valid=2/6 obs_too_low=0 neg_slope_or_oos_fail=4 (min_obs=60)
[ENSEMBLE] POOL(33) | N: 34706 | IC: 0.0681 (✅) | Mu: 1.255 | archetype_only | k: 50.0
└─ mu_bps: [beta_neutral: 37.4 (✅), forced_flow_reversal: -7.8 (❌), mean: -3.2 (❌), position_unwind: 14.9 (✅), ts_momentum: 6.1 (✅), trend: 12.6 (✅)] | score_cal: 2 valid
[SCORE-CAL-DIAG] valid=1/6 obs_too_low=0 neg_slope_or_oos_fail=5 (min_obs=60)
[ENSEMBLE] POOL(34) | N: 49306 | IC: 0.0005 (✅) | Mu: 3.175 | archetype_only | k: 50.0
└─ mu_bps: [beta_neutral: 40.9 (✅), forced_flow_reversal: -5.0 (❌), mean: -1.3 (❌), position_unwind: 5.4 (✅), ts_momentum: 15.0 (✅), trend: 7.5 (✅)] | score_cal: 1 valid
[LAYER 1: SWF SIGNAL VALIDATION] --------------------
| Metric               | Value   | Gate  | Status      |
| -------------------- | ------- | ----- | ----------- |
| CS IC Mean           | -0.012  | >0.03 | BLOCKED     |
| CS IC t-stat         | -0.56   | >1.96 | ✗ FAIL      |
| CS Fold Pass%        | 20.0%   | >60%  | —           |
| Strategy Panel       | 11/31   | >5    | —           |
| Panel Diversity      | 0.564   | >0.50 | —           |
| Decile Lift          | 21.20bps | >0.00 | —           |
| Symbol Breadth       | 0.182   | >0.3  | —           |
| Valid Symbols/N      | 2/56    | —     | —           |
| L1 Gate              | —       | —     | BLOCKED     |
----------------------------------------------------
[SWF FOLD DETAILS] ----------------------------------
| Fold | IC      | Breadth | N Valid | N Events | Pass |
| ---- | ------- | ------- | ------- | -------- | ---- |
| 1    | -0.006  |   0.400 |       2 |     7581 | FAIL |
| 2    | 0.049   |   0.000 |       0 |    22724 | PASS |
| 3    | -0.011  |   0.056 |       1 |    14718 | FAIL |
| 4    | -0.009  |   0.267 |       4 |    16995 | FAIL |
| 5    | -0.082  |   0.188 |       3 |    15397 | FAIL |
----------------------------------------------------
[STRATEGY-PANEL] ------------------------------------
| Strategy                               | Edge  | T-stat | Consistency | Valid |
| -------------------------------------- | ----- | ------ | ----------- | ----- |
| trend_pullback_continuation:tpc_50_200 |  70.4 |   2.93 |        1.00 | Y     |
| dual_momentum:dm_24_96                 |  64.4 |   2.94 |        1.00 | Y     |
| mtf_breakout_retest:mtf_bor_20         |  55.9 |   1.68 |        0.80 | Y     |
| funding_zscore_carry:fzs_168           |  37.9 |   3.60 |        1.00 | Y     |
| funding_zscore_carry:fzs_96            |  37.7 |   3.08 |        0.80 | Y     |
| dual_momentum:dm_12_48                 |  36.5 |   1.99 |        0.60 | Y     |
| residual_reversion:rr_24               |  29.7 |   2.94 |        0.80 | Y     |
| trend_pullback_continuation:tpc_20_100 |  26.1 |   1.99 |        0.80 | Y     |
| residual_reversion:rr_48               |  24.3 |   1.78 |        0.80 | Y     |
| bollinger_reversion:bollinger_20       |  13.3 |   1.67 |        0.80 | Y     |
----------------------------------------------------
[SWF-LEGACY-IC] pooled_ic=-0.0175 pooled_tstat=-0.76 breadth=0.182 valid_coverage=0.000
[SWF-DIAG] static_share=0.559 dynamic_share=0.441 score_cal_ratio=0.240 decile_lift=21.20bps
[PER-SYMBOL AGGREGATE] ------------------------------
| Symbol       | Raw Mu    | Vol       | t-stat   | IC(avg)   | Valid |
| ------------ | --------- | --------- | -------- | --------- | ----- |
| 1000SHIBUSDT |     5.016 |    0.0102 |     2.02 |    -0.030 | N     |
| 1000XECUSDT  |    -8.427 |    0.0093 |    -0.77 |     0.234 | N     |
| AAVEUSDT     |     4.607 |    0.0138 |     1.20 |    -0.110 | N     |
| ADAUSDT      |     0.845 |    0.0120 |     1.18 |     0.017 | N     |
| ANKRUSDT     |     4.768 |    0.0116 |     0.77 |    -0.152 | N     |
| ARPAUSDT     |    -8.095 |    0.0154 |     0.01 |     0.029 | N     |
| ATOMUSDT     |    -3.292 |    0.0126 |     0.76 |    -0.126 | N     |
| AVAXUSDT     |     3.849 |    0.0134 |     1.52 |    -0.020 | N     |
| AXSUSDT      |     4.452 |    0.0292 |     1.23 |    -0.009 | N     |
| BANDUSDT     |     4.372 |    0.0164 |     0.60 |    -0.022 | N     |
| BCHUSDT      |     0.189 |    0.0128 |     2.29 |    -0.013 | N     |
| BLZUSDT      |    -3.443 |    0.0001 |     1.73 |    -0.024 | N     |
| BTCUSDT      |     0.080 |    0.0063 |    -0.40 |    -0.002 | N     |
| CRVUSDT      |     7.315 |    0.0146 |     0.86 |    -0.099 | N     |
| DOGEUSDT     |    -0.206 |    0.0124 |     1.99 |    -0.032 | N     |
| DOTUSDT      |     0.598 |    0.0126 |    -0.45 |    -0.074 | N     |
| DYDXUSDT     |    -8.196 |    0.0143 |    -0.70 |     0.129 | N     |
| ETCUSDT      |     1.927 |    0.0094 |    -0.21 |     0.291 | N     |
| ETHUSDT      |     0.487 |    0.0112 |     0.25 |    -0.006 | N     |
| FILUSDT      |     3.420 |    0.0139 |     1.95 |    -0.017 | N     |
| FTMUSDT      |    -0.891 |    0.0001 |     3.32 |    -0.007 | N     |
| GALAUSDT     |    -4.037 |    0.0163 |     1.66 |    -0.146 | N     |
| ICPUSDT      |    -5.411 |    0.0087 |     2.39 |    -0.051 | N     |
| IOTAUSDT     |     2.869 |    0.0117 |     2.40 |     0.002 | Y     |
| KAVAUSDT     |    -8.010 |    0.0071 |     0.34 |     0.039 | N     |
| LINKUSDT     |    -0.709 |    0.0149 |     1.52 |    -0.060 | N     |
| LPTUSDT      |    -8.387 |    0.0362 |     0.80 |    -0.345 | N     |
| LTCUSDT      |    -0.894 |    0.0085 |    -0.36 |    -0.044 | N     |
| MANAUSDT     |    -8.421 |    0.0144 |    -1.50 |    -0.005 | N     |
| MKRUSDT      |     0.261 |    0.0125 |    -1.41 |     0.470 | N     |
| MTLUSDT      |    -8.353 |    0.0111 |    -0.42 |    -0.268 | N     |
| NEARUSDT     |     4.212 |    0.0121 |     1.84 |    -0.001 | N     |
| NEOUSDT      |    -8.340 |    0.0140 |     0.64 |     0.081 | N     |
| RSRUSDT      |     3.193 |    0.0150 |     1.75 |    -0.047 | N     |
| RUNEUSDT     |     0.424 |    0.0133 |     2.16 |     0.017 | Y     |
| RVNUSDT      |     2.801 |    0.0107 |     0.93 |    -0.062 | N     |
| SANDUSDT     |    -8.242 |    0.0127 |     1.18 |     0.113 | N     |
| SNXUSDT      |     3.109 |    0.0162 |     3.27 |    -0.065 | N     |
| SOLUSDT      |     2.080 |    0.0133 |     1.44 |     0.083 | N     |
| STORJUSDT    |     0.387 |    0.0116 |     1.41 |    -0.004 | N     |
| THETAUSDT    |     3.931 |    0.0155 |     1.03 |    -0.236 | N     |
| TRBUSDT      |    -2.828 |    0.0151 |    -0.33 |     0.041 | N     |
| UNIUSDT      |    -8.014 |    0.0138 |     0.49 |     0.122 | N     |
| VETUSDT      |    -8.104 |    0.0115 |     0.74 |     0.218 | N     |
| XRPUSDT      |     3.406 |    0.0086 |     0.69 |    -0.030 | N     |
| ZENUSDT      |    -8.041 |    0.0140 |    -0.11 |    -0.252 | N     |
| ZILUSDT      |     3.659 |    0.0116 |     1.77 |    -0.173 | N     |
| ZRXUSDT      |    -1.195 |    0.0200 |     0.03 |    -0.081 | N     |
------------------------------------------------------
[SYSTEM STATUS] ------------------------------------
| Layer   | Status  | Blocker (if any)            |
| ------- | ------- | --------------------------- |
| Layer 1 | BLOCKED | gate_passed=False           |
| Layer 2 | SKIP    | —                           |
| Layer 3 | SKIP    | —                           |
-----------------------------------------------------
[TIERED] pipeline complete: L1.gate=False L2=False L3=False
