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
[ENS] Arch-Only   | SYM: 52 | EVT:  80,760 | TOTAL: +31.0 bps | [TRD: +43.0✅ TMO: +20.5✅ MRV: +18.2✅ CRY:  +3.7✅ FLO: +20.3✅ UNW:  +5.1✅ BTN: +19.0✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 284,255 | TOTAL: +49.5 bps | [TRD: +66.3✅ TMO: +45.9✅ MRV: +25.5✅ CRY:  +8.3✅ FLO: +19.5✅ UNW: +11.8✅ BTN: +43.1✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 313,006 | TOTAL: +50.2 bps | [TRD: +65.1✅ TMO: +46.3✅ MRV: +27.9✅ CRY: +13.6✅ FLO: +27.0✅ UNW: +14.0✅ BTN: +44.1✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 177,204 | TOTAL: +44.3 bps | [TRD: +57.7✅ TMO: +33.5✅ MRV: +23.5✅ CRY: +20.8✅ FLO: +27.5✅ UNW: +20.1✅ BTN: +37.1✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 266,303 | TOTAL: +49.7 bps | [TRD: +68.4✅ TMO: +41.2✅ MRV: +25.9✅ CRY:  +3.4✅ FLO: +17.9✅ UNW: +12.5✅ BTN: +48.6✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 348,245 | TOTAL: +53.9 bps | [TRD: +70.0✅ TMO: +47.2✅ MRV: +28.7✅ CRY: +16.9✅ FLO: +35.7✅ UNW: +16.3✅ BTN: +43.5✅]

● [LAYER 1 OUTER FOLD READINESS]
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ✅ READY (4/4 Folds Ready)

  [✅] Fold #0 (FitEnd: 2023-09-30 -> OOS: 2023-10-01 ~ 2023-12-31)
       ├─ Symbols : 21 symbols loaded [AAVEUSDT, ANKRUSDT, ARPAUSDT, ARUSDT, BANDUSDT, BCHUSDT, DOTUSDT, ENSUSDT, +13 more]
       ├─ Events  : 2905 unique events
       └─ Quality : Edge: 126.38 bps

  [✅] Fold #1 (FitEnd: 2023-12-31 -> OOS: 2023-12-31 ~ 2024-03-31)
       ├─ Symbols : 25 symbols loaded [1000XECUSDT, ADAUSDT, ARUSDT, ATOMUSDT, AVAXUSDT, BANDUSDT, BTCUSDT, CRVUSDT, +17 more]
       ├─ Events  : 8357 unique events
       └─ Quality : Edge: 77.09 bps

  [✅] Fold #2 (FitEnd: 2024-03-31 -> OOS: 2024-04-01 ~ 2024-07-01)
       ├─ Symbols : 38 symbols loaded [1000SHIBUSDT, 1000XECUSDT, ADAUSDT, ANKRUSDT, ARUSDT, ATOMUSDT, AVAXUSDT, AXSUSDT, +30 more]
       ├─ Events  : 13420 unique events
       └─ Quality : Edge: 71.59 bps

  [✅] Fold #3 (FitEnd: 2024-07-01 -> OOS: 2024-07-01 ~ 2024-09-30)
       ├─ Symbols : 40 symbols loaded [1000SHIBUSDT, 1000XECUSDT, ADAUSDT, ANKRUSDT, ARUSDT, ATOMUSDT, AVAXUSDT, AXSUSDT, +32 more]
       ├─ Events  : 15150 unique events
       └─ Quality : Edge: 125.22 bps

──────────────────────────────────────────────────────────────────────────────

● [LAYER 1 HARD GATE CHECKS]
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ✅ PASSED (5/5 Passed)

  ✅ [Time-Coverage  ] :    1.000 (Target >=0.800 )
  ✅ [Signal-Quality ] :    1.000 (Target >=0.900 )
  ✅ [Symbol-Breadth ] :   41.557 (Target >=3.000 )
  ✅ [Stable-Folds   ] :    1.000 (Target >=0.500 )
  ✅ [Min-Profit     ] :   64.587 (Target >0.000  )
──────────────────────────────────────────────────────────────────────────────

[ENS-FINAL] Arch-Only   | SYM: 52 | EVT: 432,167 | TOTAL: +54.3 bps | [TRD: +67.5✅ TMO: +50.3✅ MRV: +26.8✅ CRY: +28.4✅ FLO: +28.5✅ UNW: +21.4✅ BTN: +40.9✅]

[L1 FINAL PROMOTION SUMMARY] 🚀
  RANK  SYMBOL       STRATEGY (Family)                EDGE(bps)  SIG(t-stat)     STATUS
  ────  ──────────   ───────────────────────────────  ─────────  ──────────────  ──────────────────
  #1    ANKRUSDT     residual_reversion (rr_24)          +127.2  [5/5] 4.97      [L2-PASS] Q:hi
  #2    SNXUSDT      trend_pullback_continuation ...     +278.5  [4/5] 4.15      [L2-PASS] Q:hi
  #3    ANKRUSDT     residual_reversion (rr_48)          +101.1  [4/5] 4.05      [L2-PASS] Q:mid
  #4    AXSUSDT      residual_reversion (rr_24)           +70.0  [4/5] 3.61      [L2-PASS] Q:hi
  #5    RUNEUSDT     trend_donchian (donchian_72)        +301.0  [3/5] 3.47      [L2-PASS] Q:hi
  #6    TRXUSDT      trend_donchian (donchian_72)         +70.3  [3/5] 3.37      [L2-PASS] Q:hi
  #7    STORJUSDT    trend_pullback_continuation ...     +292.4  [3/5] 3.36      [L2-PASS] Q:hi
  #8    1000XECUSDT  dual_momentum (dm_12_48)            +121.6  [3/5] 3.29      [L2-PASS] Q:hi
  #9    GALAUSDT     trend_donchian (donchian_72)        +258.7  [3/5] 3.23      [L2-PASS] Q:mid
  #10   ZILUSDT      trend_pullback_continuation ...     +179.2  [3/5] 3.14      [L2-PASS] Q:hi
  #11   LTCUSDT      dual_momentum (dm_12_48)             +67.2  [3/5] 3.13      [L2-PASS] Q:mid
  #12   VETUSDT      trend_donchian (donchian_72)        +177.1  [3/5] 3.11      [L2-PASS] Q:hi
  #13   KAVAUSDT     trend_pullback_continuation ...     +200.2  [3/5] 3.06      [L2-PASS] Q:hi
  #14   XRPUSDT      residual_reversion (rr_24)           +38.5  [3/5] 2.99      [L2-PASS] Q:lo
  #15   GALAUSDT     residual_reversion (rr_24)           +76.3  [3/5] 2.79      [L2-PASS] Q:lo
  #16   ATOMUSDT     trend_pullback_continuation ...     +158.9  [3/5] 2.79      [L2-PASS] Q:mid
  #17   AAVEUSDT     dual_momentum (dm_12_48)            +100.7  [3/5] 2.79      [L2-PASS] Q:mid
  #18   ETHUSDT      rsi_reversion (rsi_14)               +31.6  [3/5] 2.75      [L2-PASS] Q:hi
  #19   ETHUSDT      trend_donchian (donchian_72)         +96.7  [3/5] 2.68      [L2-PASS] Q:hi
  #20   TRBUSDT      trend_ma (ema_18_108)               +101.6  [3/5] 2.68      [L2-PASS] Q:hi
  #21   TRBUSDT      trend_pullback_continuation ...     +324.8  [3/5] 2.67      [L2-PASS] Q:hi
  #22   ZRXUSDT      trend_pullback_continuation ...     +269.5  [3/5] 2.61      [L2-PASS] Q:hi
  #23   FILUSDT      residual_reversion (rr_24)           +55.3  [3/5] 2.55      [L2-PASS] Q:lo
  #24   ZECUSDT      trend_donchian (donchian_72)        +198.8  [3/5] 2.54      [L2-PASS] Q:hi
  #25   LTCUSDT      funding_carry (funding_24)           +73.2  [3/5] 2.53      [L2-PASS] Q:hi
  #26   ARUSDT       trend_donchian (donchian_72)        +216.8  [3/5] 2.52      [L2-PASS] Q:hi
  #27   ZILUSDT      trend_pullback_continuation ...      +91.5  [3/5] 2.51      [L2-PASS] Q:hi
  #28   LTCUSDT      funding_zscore_carry (fzs_96)        +82.0  [2/5] 2.48      [L2-PASS] Q:hi
  #29   AVAXUSDT     funding_extreme_reversal (fe...      +45.8  [2/5] 2.48      [L2-PASS] Q:mid
  #30   LTCUSDT      funding_zscore_carry (fzs_168)       +75.6  [2/5] 2.46      [L2-PASS] Q:hi
  #31   IOTAUSDT     trend_pullback_continuation ...     +172.1  [2/5] 2.45      [L2-PASS] Q:hi
  #32   UNIUSDT      trend_pullback_continuation ...     +143.1  [2/5] 2.43      [L2-PASS] Q:mid
  #33   NEARUSDT     trend_donchian (donchian_72)        +267.6  [2/5] 2.38      [L2-PASS] Q:hi
  #34   THETAUSDT    trend_pullback_continuation ...     +204.2  [2/5] 2.37      [L2-PASS] Q:hi
  #35   SOLUSDT      trend_donchian (donchian_72)        +165.7  [2/5] 2.35      [L2-PASS] Q:hi
  #36   RVNUSDT      trend_pullback_continuation ...     +178.4  [2/5] 2.34      [L2-PASS] Q:hi
  #37   THETAUSDT    trend_donchian (donchian_72)        +179.4  [2/5] 2.34      [L2-PASS] Q:hi
  #38   RSRUSDT      trend_donchian (donchian_72)        +199.5  [2/5] 2.30      [L2-PASS] Q:hi
  #39   ANKRUSDT     trend_donchian (donchian_72)        +161.7  [2/5] 2.30      [L2-PASS] Q:lo
  #40   RUNEUSDT     rsi_reversion (rsi_6)                +56.2  [2/5] 2.28      [L2-PASS] Q:hi
  #41   ZILUSDT      trend_donchian (donchian_72)        +133.3  [2/5] 2.21      [L2-PASS] Q:lo
  #42   GALAUSDT     trend_ma (ema_18_108)                +69.1  [2/5] 2.19      [L2-PASS] Q:hi
  #43   SANDUSDT     trend_pullback_continuation ...     +259.1  [2/5] 2.19      [L2-PASS] Q:hi
  #44   LTCUSDT      funding_zscore_carry (fzs_48)        +82.6  [2/5] 2.18      [L2-PASS] Q:hi
  #45   LPTUSDT      funding_zscore_carry (fzs_48)       +115.0  [2/5] 2.16      [L2-PASS] Q:mid
  #46   FILUSDT      trend_pullback_continuation ...     +182.4  [2/5] 2.15      [L2-PASS] Q:lo
  #47   ZECUSDT      dual_momentum (dm_12_48)             +83.4  [2/5] 2.11      [L2-PASS] Q:lo
  #48   ZILUSDT      trend_ma (ema_18_108)                +53.3  [2/5] 2.05      [L2-PASS] Q:hi
  #49   NEOUSDT      funding_zscore_carry (fzs_48)        +92.1  [2/5] 2.03      [L2-PASS] Q:hi
  #50   GALAUSDT     trend_pullback_continuation ...     +209.9  [2/5] 2.03      [L2-PASS] Q:mid
  #51   BANDUSDT     trend_ma (ema_18_108)                +52.5  [2/5] 1.92      [L2-PASS] Q:hi
  #52   1000SHIBUSDT trend_pullback_continuation ...     +132.6  [2/5] 1.89      [L2-PASS] Q:lo
  #53   ETHUSDT      funding_zscore_carry (fzs_168)       +44.6  [2/5] 1.67      [L2-PASS] Q:mid
  #54   CRVUSDT      trend_pullback_continuation ...     +168.6  [2/5] 1.63      [L2-PASS] Q:mid
  #55   KAVAUSDT     trend_ma (ema_18_108)                +33.6  [2/5] 1.63      [L2-PASS] Q:hi
  #56   ZRXUSDT      trend_donchian (donchian_72)        +200.8  [2/5] 1.63      [L2-PASS] Q:lo
  #57   BNBUSDT      mtf_breakout_retest (mtf_bor...     +225.5  [2/5] 1.57      [L2-PASS] Q:mid
  #58   BTCUSDT      trend_donchian (donchian_72)         +60.6  [2/5] 1.56      [L2-PASS] Q:lo
  #59   AVAXUSDT     trend_ma (ema_12_72)                 +35.5  [1/5] 1.50      [L2-PASS] Q:hi
  #60   TRBUSDT      trend_ma (ema_12_72)                 +53.9  [1/5] 1.48      [L2-PASS] Q:lo
  #61   ADAUSDT      trend_ma (ema_18_108)                +28.2  [1/5] 1.44      [L2-PASS] Q:lo
  #62   ANKRUSDT     trend_ma (ema_18_108)                +38.0  [1/5] 1.44      [L2-PASS] Q:hi
  #63   CRVUSDT      trend_ma (ema_12_72)                 +35.9  [1/5] 1.43      [L2-PASS] Q:hi
  #64   RVNUSDT      trend_ma (ema_12_72)                 +35.1  [1/5] 1.40      [L2-PASS] Q:hi
  #65   DOTUSDT      trend_ma (ema_18_108)                +24.8  [1/5] 1.34      [L2-PASS] Q:hi
  #66   BNBUSDT      trend_donchian (donchian_72)         +90.6  [1/5] 1.33      [L2-PASS] Q:mid
  #67   BANDUSDT     residual_reversion (rr_24)           +67.6  [1/5] 1.29      [L2-PASS] Q:lo
  #68   FILUSDT      trend_ma (ema_12_72)                 +32.3  [1/5] 1.28      [L2-PASS] Q:hi
  #69   NEARUSDT     trend_ma (ema_18_108)                +36.4  [1/5] 1.26      [L2-PASS] Q:hi
  #70   IOTAUSDT     trend_ma (ema_18_108)                +39.1  [1/5] 1.25      [L2-PASS] Q:hi
  #71   ARUSDT       trend_ma (ema_18_108)                +34.4  [1/5] 1.23      [L2-PASS] Q:hi
  #72   THETAUSDT    trend_ma (ema_18_108)                +29.6  [1/5] 1.17      [L2-PASS] Q:hi
  #73   AVAXUSDT     trend_ma (ema_18_108)                +32.0  [1/5] 1.17      [L2-PASS] Q:mid
  #74   SANDUSDT     trend_ma (ema_18_108)                +30.9  [1/5] 1.16      [L2-PASS] Q:hi
  #75   FILUSDT      trend_ma (ema_18_108)                +27.4  [1/5] 1.15      [L2-PASS] Q:hi
  #76   SNXUSDT      trend_ma (ema_18_108)                +26.4  [1/5] 1.06      [L2-PASS] Q:lo
  #77   ATOMUSDT     trend_ma (ema_18_108)                +19.7  [1/5] 1.06      [L2-PASS] Q:hi
  #78   XRPUSDT      trend_ma (ema_18_108)                +18.0  [1/5] 1.05      [L2-PASS] Q:mid
  #79   RUNEUSDT     trend_ma (ema_18_108)                +28.9  [1/5] 1.00      [L2-PASS] Q:lo
  #80   1000SHIBUSDT trend_donchian (donchian_72)        +193.2  [1/5] 0.99      [L2-PASS] Q:lo
  #81   ZRXUSDT      trend_ma (ema_18_108)                +29.0  [1/5] 0.94      [L2-PASS] Q:hi
  #82   XLMUSDT      trend_ma (ema_18_108)                +12.9  [1/5] 0.92      [L2-PASS] Q:hi
  #83   CRVUSDT      trend_ma (ema_18_108)                +22.5  [1/5] 0.92      [L2-PASS] Q:lo
  #84   AXSUSDT      trend_ma (ema_12_72)                 +16.7  [1/5] 0.79      [L2-PASS] Q:mid
  #85   DOGEUSDT     trend_ma (ema_18_108)                +17.0  [1/5] 0.68      [L2-PASS] Q:mid
  #86   1000XECUSDT  trend_ma (ema_12_72)                 +13.3  [1/5] 0.65      [L2-PASS] Q:lo
  ────  ──────────   ───────────────────────────────  ─────────  ──────────────  ──────────────────
  [NOT PROMOTED] 1391 pairs | top: no_incremental_edgex777, quality_weight_zerox598, negative_gross_edgex291


● [LAYER UNIVERSE AUDIT]
──────────────────────────────────────────────────────────────────────────────
  LAYER  WINDOW RANGE                    SYMS   ACTIVE (min/med/max)       ENTRY    KILL  WARNINGS
  ─────  ──────────────────────────────  ────   ────────────────────  ──────────  ──────  ────────
  L1     2023-04-01 ~ 2024-09-30           52     0 /  49.0 / 52               0       6  —
──────────────────────────────────────────────────────────────────────────────

[TIERED] Phase=l1 — stopping after L1 (not a multilayer phase)