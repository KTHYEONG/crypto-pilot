================================================================================
LOCAL DATA STORAGE (LEDGER & CACHE STATUS)
================================================================================

  Sync Mode: FULL (Pre-loaded from cache)
  [SKIPPED] All records in 'universe_ledger.db' are up-to-date. (No sync required)

--------------------------------------------------------------------------------
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
[PROMO_FILTER] no variants recommended; advisory-only pass-through

● [DATA-INTEGRITY AUDIT]
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ✅ ALL 52 SYMBOLS PASSED
  METRICS : Total Bars: 8,761
  DETAIL  : [NaN: 0.0%] [Zero/Neg: 0.0%] [Range: PASS]
──────────────────────────────────────────────────────────────────────────────

[ALIGN-CUBE] post-join active_mask mean=0.6818 entry_block_mean=0.2559 (was 1.0 / 0.0 before cube injection)
[ENS] Arch-Only   | SYM: 52 | EVT:   3,432 | TOTAL:  +9.3 bps | [TRD: -18.4❌ TMO: +31.1✅ MRV: +43.3✅ CRY:  +9.4✅ UNW: +57.1✅ BTN: +35.6✅]
[ENS] Arch-Only   | SYM: 52 | EVT:  38,314 | TOTAL: +24.3 bps | [TRD: +39.6✅ TMO: +25.4✅ MRV: +16.2✅ CRY: -33.0❌ FLO: +19.2✅ UNW:  -8.2❌ BTN: +27.1✅]
[ENS] Arch-Only   | SYM: 52 | EVT:  74,035 | TOTAL: +32.5 bps | [TRD: +44.9✅ TMO: +22.8✅ MRV: +19.5✅ CRY:  +3.4✅ FLO: +21.5✅ UNW:  +6.6✅ BTN: +21.8✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 110,019 | TOTAL: +29.7 bps | [TRD: +41.3✅ TMO: +27.6✅ MRV: +10.7✅ CRY:  +8.4✅ FLO: +13.7✅ UNW:  +3.7✅ BTN: +19.3✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 144,934 | TOTAL: +40.8 bps | [TRD: +53.8✅ TMO: +31.8✅ MRV: +16.5✅ CRY: +19.6✅ FLO: +27.7✅ UNW: +13.8✅ BTN: +32.0✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 182,817 | TOTAL: +44.2 bps | [TRD: +55.7✅ TMO: +32.9✅ MRV: +24.3✅ CRY: +27.5✅ FLO: +27.5✅ UNW: +19.5✅ BTN: +42.6✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 212,168 | TOTAL: +46.8 bps | [TRD: +58.3✅ TMO: +35.6✅ MRV: +27.2✅ CRY: +26.9✅ FLO: +28.1✅ UNW: +20.5✅ BTN: +41.6✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 255,079 | TOTAL: +52.3 bps | [TRD: +72.2✅ TMO: +40.5✅ MRV: +28.5✅ CRY:  +5.0✅ FLO: +23.7✅ UNW: +13.1✅ BTN: +49.4✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 284,255 | TOTAL: +49.5 bps | [TRD: +66.3✅ TMO: +45.9✅ MRV: +25.5✅ CRY:  +8.3✅ FLO: +19.5✅ UNW: +11.8✅ BTN: +43.1✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 313,006 | TOTAL: +50.2 bps | [TRD: +65.1✅ TMO: +46.3✅ MRV: +27.9✅ CRY: +13.6✅ FLO: +27.0✅ UNW: +14.0✅ BTN: +44.1✅]
[ENS] Arch-Only   | SYM: 52 | EVT:  80,760 | TOTAL: +31.0 bps | [TRD: +43.0✅ TMO: +20.5✅ MRV: +18.2✅ CRY:  +3.7✅ FLO: +20.3✅ UNW:  +5.1✅ BTN: +19.0✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 177,204 | TOTAL: +44.3 bps | [TRD: +57.7✅ TMO: +33.5✅ MRV: +23.5✅ CRY: +20.8✅ FLO: +27.5✅ UNW: +20.1✅ BTN: +37.1✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 266,303 | TOTAL: +49.7 bps | [TRD: +68.4✅ TMO: +41.2✅ MRV: +25.9✅ CRY:  +3.4✅ FLO: +17.9✅ UNW: +12.5✅ BTN: +48.6✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 348,245 | TOTAL: +53.9 bps | [TRD: +70.0✅ TMO: +47.2✅ MRV: +28.7✅ CRY: +16.9✅ FLO: +35.7✅ UNW: +16.3✅ BTN: +43.5✅]

● [LAYER 1 OUTER FOLD READINESS]
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ✅ READY (4/4 Folds Ready)

  [✅] Fold #0 (FitEnd: 2023-09-30 -> OOS: 2023-10-01 ~ 2023-12-31)
       ├─ Symbols : 7 symbols loaded [ANKRUSDT, ARPAUSDT, ARUSDT, BANDUSDT, BCHUSDT, MTLUSDT, NEOUSDT]
       ├─ Events  : 349 unique events
       └─ Quality : Edge: 46.12 bps

  [✅] Fold #1 (FitEnd: 2023-12-31 -> OOS: 2023-12-31 ~ 2024-03-31)
       ├─ Symbols : 7 symbols loaded [AVAXUSDT, BANDUSDT, DOGEUSDT, DOTUSDT, NEARUSDT, RUNEUSDT, SANDUSDT]
       ├─ Events  : 2855 unique events
       └─ Quality : Edge: 60.90 bps

  [✅] Fold #2 (FitEnd: 2024-03-31 -> OOS: 2024-04-01 ~ 2024-07-01)
       ├─ Symbols : 18 symbols loaded [ANKRUSDT, ATOMUSDT, BANDUSDT, DOGEUSDT, DOTUSDT, FILUSDT, GALAUSDT, KAVAUSDT, +10 more]
       ├─ Events  : 6912 unique events
       └─ Quality : Edge: 149.85 bps

  [✅] Fold #3 (FitEnd: 2024-07-01 -> OOS: 2024-07-01 ~ 2024-09-30)
       ├─ Symbols : 19 symbols loaded [ADAUSDT, ANKRUSDT, ARUSDT, ATOMUSDT, DOGEUSDT, DOTUSDT, FILUSDT, GALAUSDT, +11 more]
       ├─ Events  : 7219 unique events
       └─ Quality : Edge: 188.07 bps

──────────────────────────────────────────────────────────────────────────────

● [LAYER 1 HARD GATE CHECKS]
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ✅ PASSED (5/5 Passed)

  ✅ [Time-Coverage  ] :    1.000 (Target >=0.800 )
  ✅ [Signal-Quality ] :    1.000 (Target >=0.900 )
  ✅ [Symbol-Breadth ] :   22.617 (Target >=3.000 )
  ✅ [Stable-Folds   ] :    1.000 (Target >=0.500 )
  ✅ [Min-Profit     ] :   54.089 (Target >0.000  )
──────────────────────────────────────────────────────────────────────────────

[ENS-FINAL] Arch-Only   | SYM: 52 | EVT: 432,167 | TOTAL: +54.3 bps | [TRD: +67.5✅ TMO: +50.3✅ MRV: +26.8✅ CRY: +28.4✅ FLO: +28.5✅ UNW: +21.4✅ BTN: +40.9✅]

[L1 FINAL PROMOTION SUMMARY] 🚀
  RANK  SYMBOL       STRATEGY (Family)                EDGE(bps)  LCB(bps)    CONV    FOLDS   t(blk)
  ────  ──────────   ───────────────────────────────  ─────────  ────────  ──────  ───────  ───────
  #1    RUNEUSDT     trend_donchian (donchian_72)        +301.0    +207.3    1.00      4/4     3.47
  #2    STORJUSDT    trend_pullback_continuation ...     +292.4    +190.6    1.00      4/4     3.36
  #3    SNXUSDT      trend_pullback_continuation ...     +278.5    +153.4    1.00      4/4     4.15
  #4    NEARUSDT     trend_donchian (donchian_72)        +267.6    +118.5    1.00      4/4     2.38
  #5    ARUSDT       trend_donchian (donchian_72)        +216.8    +106.6    1.00      4/4     2.52
  #6    ZRXUSDT      trend_pullback_continuation ...     +269.5    +106.5    1.00      3/4     2.61
  #7    KAVAUSDT     trend_pullback_continuation ...     +200.2     +95.6    1.00      4/4     3.06
  #8    ZECUSDT      trend_donchian (donchian_72)        +198.8     +80.8    0.99      3/4     2.54
  #9    1000XECUSDT  dual_momentum (dm_12_48)            +121.6     +80.6    1.00      4/4     3.29
  #10   ANKRUSDT     residual_reversion (rr_24)          +127.2     +65.9    1.00      4/4     4.97
  #11   TRXUSDT      trend_donchian (donchian_72)         +70.3     +39.2    1.00      3/4     3.37
  #12   LTCUSDT      funding_zscore_carry (fzs_48)        +82.6     +37.5    0.99      3/4     2.18
  #13   GALAUSDT     trend_ma (ema_18_108)                +69.1     +35.3    0.99      4/4     2.19
  #14   RUNEUSDT     rsi_reversion (rsi_6)                +56.2     +32.7    1.00      4/4     2.28
  #15   LTCUSDT      funding_carry (funding_24)           +73.2     +30.7    0.99      4/4     2.53
  #16   TRBUSDT      trend_ma (ema_18_108)               +101.6     +29.5    0.99      2/4     2.68
  #17   LTCUSDT      funding_zscore_carry (fzs_168)       +75.6     +22.3    0.99      4/4     2.46
  #18   LTCUSDT      funding_zscore_carry (fzs_96)        +82.0     +21.1    1.00      4/4     2.48
  #19   ZILUSDT      trend_ma (ema_18_108)                +53.3     +17.9    0.99      2/4     2.05
  #20   ANKRUSDT     trend_ma (ema_18_108)                +38.0     +11.5    0.97      3/4     1.44
  #21   ZRXUSDT      trend_ma (ema_18_108)                +29.0      +9.1    0.95      3/4     0.94
  #22   NEARUSDT     trend_ma (ema_18_108)                +36.4      +7.6    0.97      3/4     1.26
  ────  ──────────   ───────────────────────────────  ─────────  ────────  ──────  ───────  ───────
  [NOT PROMOTED] 1455 pairs | top: no_incremental_edgex777, quality_weight_zerox662, negative_gross_edgex291


● [LAYER UNIVERSE AUDIT]
──────────────────────────────────────────────────────────────────────────────
  LAYER  WINDOW RANGE                    SYMS   ACTIVE (min/med/max)       ENTRY    KILL  WARNINGS
  ─────  ──────────────────────────────  ────   ────────────────────  ──────────  ──────  ────────
  L1     2023-04-01 ~ 2024-09-30           52     0 /  49.0 / 52               0       6  —
──────────────────────────────────────────────────────────────────────────────


>> LAYER 1: PASS -> Proceeding to Layer 2.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
● [LAYER 2: PORTFOLIO ALLOCATION & RISK OPTIMIZATION]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[L2-SIGNAL-PRE] predict_layer1_signals: registry_symbols=15 activation_floor=0.0 start_idx=5484 end_idx=7674
  ● [HYPERPARAMETER OPTIMIZATION]
    - Study Name : l2_study_4h_4d406ac7aa9d
    - Config     : 200 trials
    - Provenance : events=7679 unique_symbols=ANKRUSDT,ARUSDT,GALAUSDT,KAVAUSDT,LTCUSDT,NEARUSDT,RUNEUSDT,SNXUSDT,TRBUSDT,TRXUSDT,ZECUSDT,ZILUSDT,ZRXUSDT fp=601e53f2e1e8
  ────────────────────────────────────────────────────────────────────────────
[L2-OPT]: 100%|█████████████████████████████████████████████████████| 200/200 [01:41<00:00,  1.97it/s, Best CAGR: 516.93% | Current: 435.05%]
[L2-SELECTION] No feasible candidate found within fallback window (reason=uplift)
[L2-DEPLOY-C4] L*=1.615 (binding=mdd) | realized_mode=return_scaling | kelly=0.250(불변) | tf=4h

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
● [LAYER 2: PORTFOLIO ALLOCATION & RISK OPTIMIZATION]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[L2-SIGNAL-PRE] predict_layer1_signals: registry_symbols=15 activation_floor=0.0 start_idx=5484 end_idx=7674
[L2-DEPLOY] L*=1.6146 binding=champion | CAGR=2.0898 MDD=0.1659 CVaR95=0.0216 RiskUtil=0.553
● [LAYER 2 PORTFOLIO SCORECARD] (2024-12-22 ~ 2025-09-30)
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ❌ BLOCKED (uplift)

  ✅ [Growth    ] CAGR: +209.0% (>=30.0%) | PnL: +75.0% | Equity x1.75
  ✅ [Efficiency] Sharpe: 2.647 (>=1.000) | Sortino: 4.121 (>=1.500) | Calmar: 12.596 (>=1.000)
  ✅ [Risk      ] MDD: 16.6% (<=30.0%) | CVaR95: 2.2% (<=6.0%) | RiskUtil: 55.3%
  ✅ [Robust    ] Fold: 100.0% (>=60.0%) | Trades: 117 (>=30) | Friction: 100.0%
  ❌ [Uplift    ] Sharpe Uplift: +0.01 (>=+0.20)
  ✅ [Integrity ] DSR: 0.888 (>=0.60) | PSR: 0.991 (diag)
  [Diag     ] RelMDD: 1.70x | Turnover: 0.077
──────────────────────────────────────────────────────────────────────────────

  [ FOLD DETAIL BREAKDOWN ]
  ──────────────────────────────────────────────────────────────────────────
  ├─ Fold #1 : ✅ Sharpe:  4.107 | CAGR:  +778.6% | MDD:  15.0% | Status: PASS | Period: 2024-12-22 ~ 2025-03-26
       Symbols: 6 [GALAUSDT, LTCUSDT, NEARUSDT, RUNEUSDT, TRBUSDT, TRXUSDT]
  ├─ Fold #2 : ✅ Sharpe:  1.634 | CAGR:   +97.7% | MDD:  15.6% | Status: PASS | Period: 2025-03-26 ~ 2025-06-28
       Symbols: 7 [GALAUSDT, LTCUSDT, NEARUSDT, RUNEUSDT, TRBUSDT, TRXUSDT, ZECUSDT]
  └─ Fold #3 : ✅ Sharpe:  1.907 | CAGR:   +70.4% | MDD:  11.5% | Status: PASS | Period: 2025-06-28 ~ 2025-09-30
       Symbols: 8 [ARUSDT, GALAUSDT, KAVAUSDT, LTCUSDT, NEARUSDT, RUNEUSDT, TRBUSDT, TRXUSDT]

● [LAYER UNIVERSE AUDIT]
──────────────────────────────────────────────────────────────────────────────
  LAYER  WINDOW RANGE                    SYMS   ACTIVE (min/med/max)       ENTRY    KILL  WARNINGS
  ─────  ──────────────────────────────  ────   ────────────────────  ──────────  ──────  ────────
  L2     2024-10-01 ~ 2025-09-30           52    35 /  38.0 / 43             552      19  entry_block_spike
──────────────────────────────────────────────────────────────────────────────

>> LAYER 2: BLOCKED -> gate_passed=False
!! FAIL: exit_code=1 reason=layer2_blocked:uplift