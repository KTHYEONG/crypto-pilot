================================================================================
LOCAL DATA STORAGE (LEDGER & CACHE STATUS)
================================================================================

  Sync Mode: FULL (Pre-loaded from cache)
  [SKIPPED] All records in 'universe_ledger.db' are up-to-date. (No sync required)

--------------------------------------------------------------------------------

[TF-PROBE AUDIT] SOURCE READINESS
-----------------------------------------------------------------------------------------
| TF       | Ready          | Median Bars  | Source Mix                                 |
-----------------------------------------------------------------------------------------
| 1h       | 253/253        | 22524        | 1h:253                                     |
| 2h       | 253/253        | 22524        | 1h:253                                     |
| 4h       | 253/253        | 5613         | 4h:253                                     |
| 6h       | 253/253        | 22524        | 1h:253                                     |
| 8h       | 253/253        | 22524        | 1h:253                                     |
| 12h      | 253/253        | 22524        | 1h:253                                     |
-----------------------------------------------------------------------------------------
================================================================================
SYSTEM CONTEXT | DATA PIPELINE PREPARATION
================================================================================

TIME PROFILE
  Test Horizon  : 2022-10-01 ~ 2026-03-31
  IS / OOS Split: 2025-10-01 (In-Sample Cutoff)

UNIVERSE FUNNEL
  [1] Market Pool     : 377 symbols discovered (Binance USDT-M)
  [2] Capacity Limit  : 150 symbols selected (Top-N Liquidity)
  [3] Integrity Pass  : 57 symbols loaded (Passed Gaps & Frozen checks)

STRATEGY ENGINE
  Active Engine : Alpha-Ensemble Engine
  Target Scope  : 57 symbols ready for Layer 1 execution

--------------------------------------------------------------------------------

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
● [LAYER 1: SIGNAL ROBUSTNESS & ENSEMBLE VERIFICATION]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[TIERED] Base scope: 57/57 loaded symbols
[TIERED] Sub-window admission: 52/57 symbols admitted (min_bars=1500, oos_cov>=90%)
[TIERED] Sub-window drops: {'missing_map': 0, 'empty_frame': 0, 'late_start': 5, 'min_bars': 0, 'no_holdout': 0, 'holdout_coverage': 0}
Launching tf probe: 6 tf tasks, 6 workers
tf=12h: 1995 cells computed
tf=8h: 1995 cells computed
tf=6h: 1995 cells computed
tf=4h: 1995 cells computed
tf=2h: 1995 cells computed
tf=1h: 1995 cells computed
[TF-PROBE] 32 winning cells across 6 tf: ['12h', '1h', '2h', '4h', '6h', '8h']

[TF-PROBE AUDIT] TIMEFRAME SELECTION
------------------------------------------------------------------------------------------------------
| TF       | Winning    | Families                 | Variants                           | Decision   |
------------------------------------------------------------------------------------------------------
| 1h       | 4          | rsi_reversion:2, btc_... | btc_regime_pullback:btc_pullbac... | SELECT     |
| 2h       | 7          | btc_regime_pullback:2... | btc_regime_pullback:btc_pullbac... | SELECT     |
| 4h       | 5          | btc_regime_pullback:2... | btc_regime_pullback:btc_pullbac... | SELECT     |
| 6h       | 6          | btc_regime_pullback:3... | btc_regime_pullback:btc_pullbac... | SELECT     |
| 8h       | 4          | btc_regime_pullback:2... | btc_regime_pullback:btc_pullbac... | SELECT     |
| 12h      | 6          | btc_regime_pullback:2... | btc_regime_pullback:btc_pullbac... | SELECT     |
------------------------------------------------------------------------------------------------------

[TF-PROBE AUDIT] GATE SURVIVORSHIP
-------------------------------------------------------------------------------------------------------
| TF       | Cells    | Pass t   | Pass FDR   | Pass Edge  | Pass Fold  | Winning  | Top Fail         |
-------------------------------------------------------------------------------------------------------
| 1h       | 1995     | 27       | 4          | 4          | 4          | 4        | tstat            |
| 2h       | 1995     | 16       | 7          | 7          | 7          | 7        | tstat            |
| 4h       | 1995     | 26       | 5          | 5          | 5          | 5        | tstat            |
| 6h       | 1995     | 17       | 7          | 6          | 6          | 6        | tstat            |
| 8h       | 1995     | 16       | 4          | 4          | 4          | 4        | tstat            |
| 12h      | 1995     | 20       | 6          | 6          | 6          | 6        | tstat            |
-------------------------------------------------------------------------------------------------------

[TF-PROBE AUDIT] BRIDGE INJECTION
------------------------------------------------------------------------------------------------
| TF       | Symbols    | Winning Keys   | Projected    | Source Mix                           |
------------------------------------------------------------------------------------------------
| 1h       | 52         | 4              | 4            | 1h:52                                |
| 2h       | 52         | 6              | 6            | 1h:52                                |
| 12h      | 52         | 3              | 3            | 1h:52                                |
| 8h       | 52         | 3              | 3            | 1h:52                                |
| 6h       | 52         | 4              | 4            | 1h:52                                |
------------------------------------------------------------------------------------------------
[TF-PROBE] Injected 20 extra panels from probe

● [DATA-INTEGRITY AUDIT]
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ✅ ALL 52 SYMBOLS PASSED
  METRICS : Total Bars: 8,761
  DETAIL  : [NaN: 0.0%] [Zero/Neg: 0.0%] [Range: PASS]
──────────────────────────────────────────────────────────────────────────────

[ALIGN-CUBE] post-join active_mask mean=0.6818 entry_block_mean=0.2559 (was 1.0 / 0.0 before cube injection)
[ENS] Arch-Only   | SYM: 52 | EVT:   5,245 | TOTAL: +19.3 bps | [TRD: -16.4❌ TMO: +33.1✅ MRV: +38.9✅ CRY: +11.4✅ UNW: +60.7✅ BTN: +31.7✅]
[ENS] Arch-Only   | SYM: 52 | EVT:  55,364 | TOTAL: +20.1 bps | [TRD: +38.6✅ TMO: +24.5✅ MRV: +13.9✅ CRY: -33.4❌ FLO: +15.4✅ UNW:  -8.8❌ BTN: +15.3✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 104,558 | TOTAL: +26.7 bps | [TRD: +43.7✅ TMO: +21.6✅ MRV: +16.5✅ CRY:  +2.3✅ FLO: +16.8✅ UNW:  +5.5✅ BTN: +16.7✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 158,345 | TOTAL: +22.7 bps | [TRD: +39.9✅ TMO: +26.2✅ MRV:  +9.7✅ CRY:  +7.0✅ FLO:  +8.5✅ UNW:  +2.3✅ BTN: +15.2✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 205,890 | TOTAL: +33.5 bps | [TRD: +52.3✅ TMO: +30.3✅ MRV: +18.3✅ CRY: +18.1✅ FLO: +23.0✅ UNW: +12.3✅ BTN: +25.4✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 256,255 | TOTAL: +36.5 bps | [TRD: +54.2✅ TMO: +31.4✅ MRV: +21.5✅ CRY: +25.9✅ FLO: +22.7✅ UNW: +18.0✅ BTN: +33.4✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 295,301 | TOTAL: +39.6 bps | [TRD: +56.9✅ TMO: +34.2✅ MRV: +25.0✅ CRY: +25.5✅ FLO: +23.8✅ UNW: +19.1✅ BTN: +32.9✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 344,691 | TOTAL: +44.5 bps | [TRD: +70.6✅ TMO: +38.9✅ MRV: +26.3✅ CRY:  +3.4✅ FLO: +19.2✅ UNW: +11.6✅ BTN: +40.8✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 377,549 | TOTAL: +42.2 bps | [TRD: +64.8✅ TMO: +44.4✅ MRV: +23.2✅ CRY:  +6.8✅ FLO: +15.7✅ UNW: +10.3✅ BTN: +39.8✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 114,745 | TOTAL: +25.0 bps | [TRD: +41.8✅ TMO: +19.3✅ MRV: +14.7✅ CRY:  +2.5✅ FLO: +15.4✅ UNW:  +3.9✅ BTN: +15.0✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 414,672 | TOTAL: +42.6 bps | [TRD: +63.6✅ TMO: +44.8✅ MRV: +23.7✅ CRY: +12.1✅ FLO: +23.3✅ UNW: +12.5✅ BTN: +38.9✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 248,451 | TOTAL: +37.1 bps | [TRD: +56.3✅ TMO: +32.1✅ MRV: +22.5✅ CRY: +19.4✅ FLO: +23.0✅ UNW: +18.7✅ BTN: +29.6✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 356,972 | TOTAL: +42.8 bps | [TRD: +67.0✅ TMO: +39.8✅ MRV: +25.1✅ CRY:  +2.0✅ FLO: +14.1✅ UNW: +11.1✅ BTN: +42.1✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 467,296 | TOTAL: +45.1 bps | [TRD: +68.3✅ TMO: +45.4✅ MRV: +24.4✅ CRY: +15.1✅ FLO: +31.6✅ UNW: +14.6✅ BTN: +38.8✅]

● [LAYER 1 OUTER FOLD READINESS]
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ✅ READY (4/4 Folds Ready)

  [✅] Fold #0 (FitEnd: 2023-09-30 -> OOS: 2023-10-01 ~ 2023-12-31)
       ├─ Symbols : 7 symbols loaded [ANKRUSDT, BANDUSDT, BCHUSDT, MTLUSDT, NEARUSDT, NEOUSDT, ZILUSDT]
       ├─ Events  : 284 unique events
       └─ Quality : Edge: 64.86 bps

  [✅] Fold #1 (FitEnd: 2023-12-31 -> OOS: 2023-12-31 ~ 2024-03-31)
       ├─ Symbols : 7 symbols loaded [AVAXUSDT, DOGEUSDT, DOTUSDT, LTCUSDT, MTLUSDT, NEARUSDT, SANDUSDT]
       ├─ Events  : 2431 unique events
       └─ Quality : Edge: 23.33 bps

  [✅] Fold #2 (FitEnd: 2024-03-31 -> OOS: 2024-04-01 ~ 2024-07-01)
       ├─ Symbols : 18 symbols loaded [ANKRUSDT, ATOMUSDT, BANDUSDT, DOGEUSDT, DOTUSDT, FILUSDT, GALAUSDT, KAVAUSDT, +10 more]
       ├─ Events  : 6920 unique events
       └─ Quality : Edge: 154.56 bps

  [✅] Fold #3 (FitEnd: 2024-07-01 -> OOS: 2024-07-01 ~ 2024-09-30)
       ├─ Symbols : 20 symbols loaded [ADAUSDT, ANKRUSDT, ARUSDT, ATOMUSDT, DOGEUSDT, DOTUSDT, FILUSDT, GALAUSDT, +12 more]
       ├─ Events  : 7206 unique events
       └─ Quality : Edge: 186.60 bps

──────────────────────────────────────────────────────────────────────────────

● [LAYER 1 HARD GATE CHECKS]
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ✅ PASSED (5/5 Passed)

  ✅ [Time-Coverage  ] :    1.000 (Target >=0.800 )
  ✅ [Signal-Quality ] :    1.000 (Target >=0.900 )
  ✅ [Symbol-Breadth ] :   21.125 (Target >=3.000 )
  ✅ [Stable-Folds   ] :    1.000 (Target >=0.500 )
  ✅ [Min-Profit     ] :   49.400 (Target >0.000  )
──────────────────────────────────────────────────────────────────────────────

[ENS-FINAL] Arch-Only   | SYM: 52 | EVT: 574,135 | TOTAL: +45.4 bps | [TRD: +65.8✅ TMO: +48.6✅ MRV: +23.1✅ CRY: +26.6✅ FLO: +24.6✅ UNW: +19.6✅ BTN: +37.6✅]

[L1 FINAL PROMOTION SUMMARY] 🚀
  RANK  SYMBOL       STRATEGY (Family)                EDGE(bps)  LCB(bps)    CONV    FOLDS   t(blk)
  ────  ──────────   ───────────────────────────────  ─────────  ────────  ──────  ───────  ───────
  #1    RUNEUSDT     trend_donchian (donchian_72)        +303.8    +207.3    1.00      4/4     3.51
  #2    STORJUSDT    trend_pullback_continuation ...     +293.8    +190.6    1.00      4/4     3.38
  #3    SNXUSDT      trend_pullback_continuation ...     +280.4    +153.4    1.00      4/4     4.18
  #4    NEARUSDT     trend_donchian (donchian_72)        +273.6    +118.5    1.00      4/4     2.43
  #5    ZRXUSDT      trend_pullback_continuation ...     +269.5    +106.5    1.00      3/4     2.61
  #6    ZECUSDT      trend_donchian (donchian_72)        +199.9     +80.8    0.99      3/4     2.55
  #7    1000XECUSDT  dual_momentum (dm_12_48)            +122.2     +80.6    1.00      4/4     3.31
  #8    AAVEUSDT     residual_reversion (rr_16_6h)       +104.5     +52.0    1.00      4/4     3.02
  #9    TRXUSDT      trend_donchian (donchian_72)         +72.4     +39.2    1.00      3/4     3.48
  #10   LTCUSDT      funding_zscore_carry (fzs_48)        +85.4     +37.5    0.99      3/4     2.25
  #11   GALAUSDT     trend_ma (ema_18_108)                +67.7     +35.3    0.99      4/4     2.15
  #12   LTCUSDT      funding_carry (funding_24)           +75.7     +30.7    0.99      4/4     2.61
  #13   TRBUSDT      trend_ma (ema_18_108)               +103.2     +29.5    0.99      2/4     2.72
  #14   LTCUSDT      funding_zscore_carry (fzs_168)       +77.2     +22.3    0.99      4/4     2.51
  #15   LTCUSDT      funding_zscore_carry (fzs_96)        +84.3     +21.1    1.00      4/4     2.55
  #16   ZILUSDT      trend_ma (ema_18_108)                +52.0     +17.9    0.99      2/4     2.00
  #17   ANKRUSDT     trend_ma (ema_18_108)                +36.1     +11.5    0.97      3/4     1.37
  #18   ZRXUSDT      trend_ma (ema_18_108)                +30.1      +9.1    0.95      3/4     0.97
  ────  ──────────   ───────────────────────────────  ─────────  ────────  ──────  ───────  ───────
  [NOT PROMOTED] 2497 pairs | top: no_incremental_edgex1257, quality_weight_zerox1201, negative_gross_edgex534


● [LAYER UNIVERSE AUDIT]
──────────────────────────────────────────────────────────────────────────────
  LAYER  WINDOW RANGE                    SYMS   ACTIVE (min/med/max)       ENTRY    KILL  WARNINGS
  ─────  ──────────────────────────────  ────   ────────────────────  ──────────  ──────  ────────
  L1     2023-04-01 ~ 2024-09-30           52     0 /  49.0 / 52               0       6  —
──────────────────────────────────────────────────────────────────────────────


>> LAYER 1: PASS -> Proceeding to Layer 2.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
● [LAYER 2: OPTUNA TUNING]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ● [STUDY] l2_study_4h_4b58837f09a5 | trials=200 | events=5512 | symbols=12
  ────────────────────────────────────────────────────────────────────────────
[L2-OPT]: 100%|█████████████████████████████████████████████████████| 200/200 [01:13<00:00,  2.73it/s, Best CAGR: 367.18% | Current: -30.26%]
[L2-SELECTION] 1 gate-pass 후보 수집 → champion Trial #131 Sortino=3.5281 CAGR=0.3591
[L2-SELECTION] Champion selected. Trial #131, Objective=3.0547, DSR=0.9263 (n_eff=4.31)
[L2-DEPLOY-C4] L*=1.806 (binding=mdd) | realized_mode=return_scaling | kelly=0.250(불변) | tf=4h
  ● [FINAL SIMULATION]
[L2-DEPLOY] L*=1.8057 binding=champion | CAGR=0.3591 MDD=0.0608 CVaR95=0.0064 RiskUtil=0.203
● [LAYER 2 PORTFOLIO SCORECARD] (2024-12-22 ~ 2025-09-30)
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ✅ PASS

  ✅ [Growth    ] CAGR: +35.9% (>=30.0%) | PnL: +14.2% | Equity x1.14
  ✅ [Efficiency] Sharpe: 2.225 (>=1.000) | Sortino: 3.528 (>=1.500) | Calmar: 5.906 (>=1.000)
  ✅ [Risk      ] MDD: 6.1% (<=30.0%) | CVaR95: 0.6% (<=6.0%) | RiskUtil: 20.3%
  ✅ [Robust    ] Fold: 100.0% (>=60.0%) | Trades: 90 (>=30) | Friction: 100.0%
  ✅ [Uplift    ] Sharpe Uplift: +0.38 (>=+0.20)
  ✅ [Integrity ] DSR: 0.926 (>=0.60) | PSR: 0.977 (diag)
  [Diag     ] RelMDD: 1.80x | Turnover: 0.035
──────────────────────────────────────────────────────────────────────────────

  [ FOLD DETAIL BREAKDOWN ]
  ──────────────────────────────────────────────────────────────────────────
  ├─ Fold #1 : ✅ Sharpe:  2.446 | CAGR:   +58.2% | MDD:   6.1% | Status: PASS | Period: 2024-12-22 ~ 2025-03-26
       Symbols: 6 [AAVEUSDT, GALAUSDT, LTCUSDT, NEARUSDT, TRBUSDT, TRXUSDT]
  ├─ Fold #2 : ✅ Sharpe:  3.411 | CAGR:   +47.6% | MDD:   2.2% | Status: PASS | Period: 2025-03-26 ~ 2025-06-28
       Symbols: 6 [AAVEUSDT, GALAUSDT, LTCUSDT, TRBUSDT, TRXUSDT, ZECUSDT]
  └─ Fold #3 : ✅ Sharpe:  0.808 | CAGR:    +7.6% | MDD:   5.3% | Status: PASS | Period: 2025-06-28 ~ 2025-09-30
       Symbols: 7 [AAVEUSDT, GALAUSDT, LTCUSDT, NEARUSDT, RUNEUSDT, TRBUSDT, TRXUSDT]

● [LAYER UNIVERSE AUDIT]
──────────────────────────────────────────────────────────────────────────────
  LAYER  WINDOW RANGE                    SYMS   ACTIVE (min/med/max)       ENTRY    KILL  WARNINGS
  ─────  ──────────────────────────────  ────   ────────────────────  ──────────  ──────  ────────
  L2     2024-10-01 ~ 2025-09-30           52    35 /  38.0 / 43             552      19  entry_block_spike
──────────────────────────────────────────────────────────────────────────────

>> LAYER 2: PASS -> Proceeding to Final Holdout.
>> TARGET PHASE l2 REACHED -> Stopping pipeline.
[PHASE] phase=l2 completed strategy/candidate evaluation only; optimization/training skipped