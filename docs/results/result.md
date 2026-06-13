[WINDOW] -------------------------------------------
| Property           | Value                       |
| ------------------ | --------------------------- |
| Range              | 2022-10-01 ~ 2026-03-31     |
| IS Start           | 2023-10-01                  |
| OOS Start          | 2025-10-01                  |
| Elapsed            | 0.00s                       |
----------------------------------------------------
discover_universe_timeline: l2_start(2024-10-01) < oos_start(2025-10-01), ignoring l2_start

[UNIVERSE REPORT] ----------------------------------
| Metric             | Value                       |
| ------------------ | --------------------------- |
| Selected (Stg6)    | 20                          |
| Panels (Inf/Live)  | 94 / 20                     |
| Windows (Inf)      | 12                          |
----------------------------------------------------
[UNIVERSE] Discovery complete: 94 symbols (2.64s)

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
[CACHE] Backfill: 2022-04-01 ~ 2026-03-31 | Symbols: 94 | Last: 2026-04-01
Sync mode=full targeted_symbols=94
Loaded symbol sync profiles from cache: /home/kth/my_coin_traider/data/futures/symbol_sync_profiles.json
[SYNC-COVERAGE] rows=36 file=/home/kth/my_coin_traider/logs/futures/universe/sync_coverage_report.parquet
Ledger update complete.

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
[BRIDGE-PROF] total=16.0236s align=0.0822s rules=2.6428s events=1.5452s label=6.6442s diagnostics=2.3066s promotions=0.0474s walk_forward=0.0000s post_wf=0.0000s selection=0.0000s weights=0.0000s alpha_panel=2.6833s accounted=15.9518s unaccounted=0.0719s
[TIERED] USE_CS_RANK_ENGINE=True — entering Tiered pipeline
[TIERED] aligned scope: 56 symbols (historical union ∩ data-valid)
[WORKFLOW] Fold 0 skipped Ensemble (fit=0 < 2)
[SCORE-CAL-DIAG] valid=0/3 obs_too_low=0 neg_slope_or_oos_fail=3 (min_obs=60)
[ENSEMBLE] POOL(2) | N: 824 | IC: 0.1359 (✅) | Mu: 10.217 | archetype_regime | k: 50.0
└─ mu_bps: [beta_neutral: 11.7 (✅), mean: 6.8 (✅), position_unwind: 10.6 (✅), ts_momentum: 31.8 (✅), trend: 13.9 (✅)] | score_cal: 0 valid
[SCORE-CAL-DIAG] valid=0/4 obs_too_low=0 neg_slope_or_oos_fail=4 (min_obs=60)
[ENSEMBLE] POOL(2) | N: 1685 | IC: 0.2194 (✅) | Mu: 8.046 | archetype_regime | k: 50.0
└─ mu_bps: [beta_neutral: 17.3 (✅), mean: 8.3 (✅), position_unwind: 5.9 (✅), ts_momentum: 5.8 (✅), trend: 4.2 (✅)] | score_cal: 0 valid
[SCORE-CAL-DIAG] valid=0/4 obs_too_low=0 neg_slope_or_oos_fail=4 (min_obs=60)
[ENSEMBLE] POOL(2) | N: 1493 | IC: 0.3053 (✅) | Mu: 7.902 | archetype_regime | k: 50.0
└─ mu_bps: [beta_neutral: 10.8 (✅), mean: 8.2 (✅), position_unwind: 5.8 (✅), ts_momentum: 11.9 (✅), trend: 0.1 (✅)] | score_cal: 0 valid
[SCORE-CAL-DIAG] valid=0/3 obs_too_low=0 neg_slope_or_oos_fail=3 (min_obs=60)
[ENSEMBLE] POOL(2) | N: 828 | IC: 0.1369 (✅) | Mu: 9.729 | archetype_regime | k: 50.0
└─ mu_bps: [beta_neutral: 11.4 (✅), mean: 6.5 (✅), position_unwind: 10.2 (✅), ts_momentum: 31.5 (✅), trend: 11.6 (✅)] | score_cal: 0 valid
[SCORE-CAL-DIAG] valid=0/5 obs_too_low=0 neg_slope_or_oos_fail=5 (min_obs=60)
[ENSEMBLE] POOL(2) | N: 2494 | IC: 0.0948 (✅) | Mu: 4.101 | archetype_regime | k: 50.0
└─ mu_bps: [beta_neutral: 13.4 (✅), mean: 4.0 (✅), position_unwind: 11.2 (✅), ts_momentum: 8.0 (✅), trend: -5.6 (❌)] | score_cal: 0 valid
[SCORE-CAL-DIAG] valid=0/6 obs_too_low=0 neg_slope_or_oos_fail=6 (min_obs=60)
[ENSEMBLE] POOL(5) | N: 5440 | IC: -0.0527 (❌) | Mu: 2.182 | archetype_regime | k: 50.0
└─ mu_bps: [beta_neutral: 34.3 (✅), mean: 1.7 (✅), position_unwind: 31.5 (✅), ts_momentum: -0.1 (❌), trend: -9.5 (❌)] | score_cal: 0 valid
[SCORE-CAL-DIAG] valid=0/6 obs_too_low=0 neg_slope_or_oos_fail=6 (min_obs=60)
[ENSEMBLE] POOL(5) | N: 5381 | IC: -0.0461 (❌) | Mu: 1.739 | archetype_regime | k: 50.0
└─ mu_bps: [beta_neutral: 35.7 (✅), mean: 1.2 (✅), position_unwind: 31.9 (✅), ts_momentum: -0.4 (❌), trend: -10.8 (❌)] | score_cal: 0 valid
[SCORE-CAL-DIAG] valid=0/4 obs_too_low=0 neg_slope_or_oos_fail=4 (min_obs=60)
[ENSEMBLE] POOL(2) | N: 1685 | IC: 0.2194 (✅) | Mu: 8.046 | archetype_regime | k: 50.0
└─ mu_bps: [beta_neutral: 17.3 (✅), mean: 8.3 (✅), position_unwind: 5.9 (✅), ts_momentum: 5.8 (✅), trend: 4.2 (✅)] | score_cal: 0 valid
[SCORE-CAL-DIAG] valid=0/6 obs_too_low=0 neg_slope_or_oos_fail=6 (min_obs=60)
[ENSEMBLE] POOL(5) | N: 5431 | IC: -0.0502 (❌) | Mu: 2.091 | archetype_regime | k: 50.0
└─ mu_bps: [beta_neutral: 34.3 (✅), mean: 1.5 (✅), position_unwind: 31.4 (✅), ts_momentum: 0.2 (✅), trend: -9.2 (❌)] | score_cal: 0 valid
[SCORE-CAL-DIAG] valid=2/6 obs_too_low=0 neg_slope_or_oos_fail=4 (min_obs=60)
[ENSEMBLE] POOL(18) | N: 11897 | IC: 0.0584 (✅) | Mu: 4.434 | archetype_regime | k: 50.0
└─ mu_bps: [beta_neutral: 20.2 (✅), mean: 2.9 (✅), position_unwind: 14.3 (✅), ts_momentum: 5.8 (✅), trend: 11.5 (✅)] | score_cal: 2 valid
[SCORE-CAL-DIAG] valid=2/6 obs_too_low=0 neg_slope_or_oos_fail=4 (min_obs=60)
[ENSEMBLE] POOL(18) | N: 13161 | IC: -0.0065 (❌) | Mu: 2.981 | archetype_only | k: 50.0
└─ mu_bps: [beta_neutral: 15.0 (✅), mean: 1.4 (✅), position_unwind: 13.2 (✅), ts_momentum: 9.2 (✅), trend: 7.3 (✅)] | score_cal: 2 valid
[SCORE-CAL-DIAG] valid=0/5 obs_too_low=0 neg_slope_or_oos_fail=5 (min_obs=60)
[ENSEMBLE] POOL(2) | N: 2494 | IC: 0.0948 (✅) | Mu: 4.101 | archetype_regime | k: 50.0
└─ mu_bps: [beta_neutral: 13.4 (✅), mean: 4.0 (✅), position_unwind: 11.2 (✅), ts_momentum: 8.0 (✅), trend: -5.6 (❌)] | score_cal: 0 valid
[SCORE-CAL-DIAG] valid=2/6 obs_too_low=0 neg_slope_or_oos_fail=4 (min_obs=60)
[ENSEMBLE] POOL(5) | N: 9455 | IC: 0.0041 (✅) | Mu: 8.657 | archetype_regime | k: 50.0
└─ mu_bps: [beta_neutral: 20.9 (✅), mean: 7.0 (✅), position_unwind: 20.8 (✅), ts_momentum: 10.1 (✅), trend: 17.4 (✅)] | score_cal: 2 valid
[SCORE-CAL-DIAG] valid=3/6 obs_too_low=0 neg_slope_or_oos_fail=3 (min_obs=60)
[ENSEMBLE] POOL(18) | N: 29690 | IC: 0.0684 (✅) | Mu: 14.746 | archetype_regime | k: 50.0
└─ mu_bps: [beta_neutral: 52.7 (✅), forced_flow_reversal: 4.8 (✅), mean: 9.3 (✅), position_unwind: 13.1 (✅), ts_momentum: 24.3 (✅), trend: 38.5 (✅)] | score_cal: 3 valid
[SCORE-CAL-DIAG] valid=3/6 obs_too_low=0 neg_slope_or_oos_fail=3 (min_obs=60)
[ENSEMBLE] POOL(18) | N: 33368 | IC: 0.0746 (✅) | Mu: 13.564 | archetype_only | k: 50.0
└─ mu_bps: [beta_neutral: 56.8 (✅), forced_flow_reversal: 3.9 (✅), mean: 8.3 (✅), position_unwind: 22.3 (✅), ts_momentum: 22.7 (✅), trend: 29.4 (✅)] | score_cal: 3 valid
[LAYER 1 HARD GATE] --------------------------------
| Gate                        | Value   | Threshold | Status  | Blocker |
| --------------------------- | ------- | --------- | ------- | ------- |
| trained_outer_fold_coverage |   1.000 | >=0.800   | PASS    | -       |
| stable_ready_symbol_count   |   1.000 | >=6.000   | FAIL    | 1.000   |
| stable_ready_symbol_ratio   |   0.018 | >=0.300   | FAIL    | 0.018   |
| ready_outer_fold_ratio      |   0.000 | >=0.600   | FAIL    | 0.000   |
| opportunity_ic_mean         |   0.000 | >=0.020   | FAIL    | 0.000   |
| opportunity_ic_tstat        |   0.000 | >=1.960   | FAIL    | 0.000   |
| probe_gross_edge_bps        |   0.000 | >0.000    | FAIL    | 0.000   |
| probe_gross_edge_tstat      |   0.000 | >=1.960   | FAIL    | 0.000   |
| Layer1 Gate                 | -       | ALL       | BLOCKED | stable_ready_symbol_count:1.000; stable_ready_symbol_ratio:0.018; ready_outer_fold_ratio:0.000; opportunity_ic_mean:0.000; opportunity_ic_tstat:0.000; probe_gross_edge_bps:0.000; probe_gross_edge_tstat:0.000 |
------------------------------------------------------
[LAYER 1 OUTER FOLDS] ------------------------------
| Fold | Registry Source End | Outer Start | Ready Symbols | Times | IC     | Probe  | Status |
| ---- | ------------------- | ----------- | ------------- | ----- | ------ | ------ | ------ |
| 0    | 2602                | 2848        | 0             | 0     | 0.000  | 0.00   | FAIL   |
| 0    | 3129                | 3506        | 0             | 0     | 0.000  | 0.00   | FAIL   |
| 4164 | 3655                | 4164        | 1             | 0     | 0.000  | 0.00   | FAIL   |
| 4822 | 4182                | 4822        | 1             | 0     | 0.000  | 0.00   | FAIL   |
------------------------------------------------------
[SYSTEM STATUS] ------------------------------------
| Layer   | Status  | Blocker (if any)            |
| ------- | ------- | --------------------------- |
| Layer 1 | BLOCKED | gate_passed=False           |
| Layer 2 | SKIP    | —                           |
| Layer 3 | SKIP    | —                           |
-----------------------------------------------------
[TIERED] pipeline complete: L1.gate=False L2=False L3=False