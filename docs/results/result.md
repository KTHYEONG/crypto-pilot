================================================================================
LOCAL DATA STORAGE (LEDGER & CACHE STATUS)
================================================================================

  Sync Mode: FULL (Pre-loaded from cache)
  [SKIPPED] All records in 'universe_ledger.db' are up-to-date. (No sync required)

--------------------------------------------------------------------------------
🔍 [TF-PROBE AUDIT] SOURCE READINESS Dashboard
  ├── 1h   : Ready 253/253 | Median Bars: 22524  | Mix: 1h:253
  ├── 2h   : Ready 253/253 | Median Bars: 22524  | Mix: 1h:253
  ├── 4h   : Ready 253/253 | Median Bars: 5613   | Mix: 4h:253
  ├── 6h   : Ready 253/253 | Median Bars: 22524  | Mix: 1h:253
  ├── 8h   : Ready 253/253 | Median Bars: 22524  | Mix: 1h:253
  ├── 12h  : Ready 253/253 | Median Bars: 22524  | Mix: 1h:253
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
📊 [L1: SWF SCOPE & ADMISSION]
  ├─ Symbols : 52/57 Admitted
  └─ Details : Base 57 | Dropped 5 (late_start: 5)
🧬 [L1: MULTI-TF PANEL INJECTION]
  └─ Active : [8h] Proj=13 Syms=52 | [12h] Proj=14 Syms=52 | [6h] Proj=13 Syms=52

● [DATA-INTEGRITY AUDIT]
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ✅ ALL 52 SYMBOLS PASSED
  METRICS : Total Bars: 8,761
  DETAIL  : [NaN: 0.0%] [Zero/Neg: 0.0%] [Range: PASS]
──────────────────────────────────────────────────────────────────────────────


● [LAYER 1 OUTER FOLD READINESS]
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ✅ READY (3/4 Folds Ready)

  [✅] Fold #0 (FitEnd: 2023-09-30 -> OOS: 2023-10-01 ~ 2023-12-31)
       ├─ Symbols : 15 symbols loaded [ANKRUSDT, ARPAUSDT, ARUSDT, BANDUSDT, BCHUSDT, BNBUSDT, ENSUSDT, ETHUSDT, +7 more]
       ├─ Events  : 3314 unique events
       └─ Quality : Edge: 83.68 bps

  [❌] Fold #1 (FitEnd: 2023-12-31 -> OOS: 2023-12-31 ~ 2024-03-31)
       ├─ Symbols : 13 symbols loaded [ATOMUSDT, AVAXUSDT, BANDUSDT, BTCUSDT, DOGEUSDT, DOTUSDT, KAVAUSDT, LPTUSDT, +5 more]
       ├─ Events  : 5330 unique events
       └─ Quality : Edge: -23.97 bps
       └─ BLOCKERS: non_positive_gross_edge

  [✅] Fold #2 (FitEnd: 2024-03-31 -> OOS: 2024-04-01 ~ 2024-07-01)
       ├─ Symbols : 17 symbols loaded [1000XECUSDT, ANKRUSDT, ATOMUSDT, BANDUSDT, DOGEUSDT, DOTUSDT, GALAUSDT, KAVAUSDT, +9 more]
       ├─ Events  : 6465 unique events
       └─ Quality : Edge: 122.42 bps

  [✅] Fold #3 (FitEnd: 2024-07-01 -> OOS: 2024-07-01 ~ 2024-09-30)
       ├─ Symbols : 17 symbols loaded [1000XECUSDT, ADAUSDT, ANKRUSDT, BNBUSDT, DOTUSDT, GALAUSDT, KAVAUSDT, NEARUSDT, +9 more]
       ├─ Events  : 6045 unique events
       └─ Quality : Edge: 148.50 bps

──────────────────────────────────────────────────────────────────────────────
🏁 STATUS : ✅ PASSED (5/5 Passed)
  👉 Cov:1.000(>=0.80) | Qual:1.000(>=0.90) | Brd:25.973(>=3.00) | Fld:0.750(>=0.50) | Prf:52.000(>0.00)
📈 [ENS-FINAL] Arch-Only | SYM:52 | EVT:681,037 | TOTAL:+62.7 bps | [TRD:+72.3✅ TMO:+52.0✅ MRV:+29.1✅ CRY:+30.1✅ FLO:+32.2✅ UNW:+23.1✅ BTN:+42.6✅]

🏆 [L1 FINAL PROMOTION SUMMARY] 🚀 (Top 5 / 28 Promoted)
  RANK  SYMBOL       STRATEGY (Family)                EDGE(bps)  LCB(bps)    CONV    FOLDS   t(blk)
  ────  ──────────   ───────────────────────────────  ─────────  ────────  ──────  ───────  ───────
  #1    RUNEUSDT     trend_donchian (donchian_72_4h)     +309.6    +186.0    1.00      4/4     3.45
  #2    STORJUSDT    trend_pullback_continuation ...     +294.2    +184.8    1.00      4/4     3.42
  #3    TRBUSDT      trend_pullback_continuation ...     +330.1    +177.9    1.00      4/4     2.97
  #4    SNXUSDT      trend_pullback_continuation ...     +273.5    +177.5    1.00      4/4     3.80
  #5    NEARUSDT     trend_donchian (donchian_72_4h)     +274.9    +129.4    0.99      4/4     2.39
  └─ 🚀 And 23 more pairs promoted (e.g. ZRXUSDT, SANDUSDT, KAVAUSDT, ARUSDT, ZECUSDT, ANKRUSDT, +17 more)
  ────  ──────────   ───────────────────────────────  ─────────  ────────  ──────  ───────  ───────
  [NOT PROMOTED] 1708 pairs | top: no_incremental_edgex929, quality_weight_zerox737, negative_gross_edgex306


● [LAYER 1 OUTER FOLD READINESS]
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ✅ READY (4/4 Folds Ready)

  [✅] Fold #0 (FitEnd: 2023-09-30 -> OOS: 2023-10-01 ~ 2023-12-31)
       ├─ Symbols : 17 symbols loaded [ANKRUSDT, ARUSDT, ATOMUSDT, BANDUSDT, ENSUSDT, ETCUSDT, KAVAUSDT, LPTUSDT, +9 more]
       ├─ Events  : 1354 unique events
       └─ Quality : Edge: 129.22 bps

  [✅] Fold #1 (FitEnd: 2023-12-31 -> OOS: 2023-12-31 ~ 2024-03-31)
       ├─ Symbols : 20 symbols loaded [1000XECUSDT, AAVEUSDT, ADAUSDT, ANKRUSDT, ARUSDT, ATOMUSDT, AVAXUSDT, BANDUSDT, +12 more]
       ├─ Events  : 2992 unique events
       └─ Quality : Edge: 15.60 bps

  [✅] Fold #2 (FitEnd: 2024-03-31 -> OOS: 2024-04-01 ~ 2024-07-01)
       ├─ Symbols : 29 symbols loaded [1000SHIBUSDT, 1000XECUSDT, AAVEUSDT, ADAUSDT, ANKRUSDT, ARUSDT, ATOMUSDT, AXSUSDT, +21 more]
       ├─ Events  : 6848 unique events
       └─ Quality : Edge: 61.17 bps

  [✅] Fold #3 (FitEnd: 2024-07-01 -> OOS: 2024-07-01 ~ 2024-09-30)
       ├─ Symbols : 25 symbols loaded [AAVEUSDT, ADAUSDT, ANKRUSDT, ARUSDT, AVAXUSDT, AXSUSDT, BCHUSDT, BNBUSDT, +17 more]
       ├─ Events  : 6790 unique events
       └─ Quality : Edge: 131.87 bps

──────────────────────────────────────────────────────────────────────────────
🏁 STATUS : ✅ PASSED (5/5 Passed)
  👉 Cov:1.000(>=0.80) | Qual:1.000(>=0.90) | Brd:34.361(>=3.00) | Fld:1.000(>=0.50) | Prf:29.513(>0.00)
📈 [ENS-FINAL] Arch-Only | SYM:52 | EVT:456,109 | TOTAL:+60.9 bps | [TRD:+67.6✅ TMO:+57.5✅ MRV:+28.7✅ CRY:+22.1✅]

🏆 [L1 FINAL PROMOTION SUMMARY] 🚀 (Top 5 / 37 Promoted)
  RANK  SYMBOL       STRATEGY (Family)                EDGE(bps)  LCB(bps)    CONV    FOLDS   t(blk)
  ────  ──────────   ───────────────────────────────  ─────────  ────────  ──────  ───────  ───────
  #1    GALAUSDT     trend_donchian (donchian_72_6h)     +303.5    +176.0    1.00      4/4     3.67
  #2    ANKRUSDT     trend_donchian (donchian_72_6h)     +218.7    +154.6    1.00      4/4     3.56
  #3    NEARUSDT     trend_donchian (donchian_72_6h)     +261.5    +147.3    1.00      4/4     2.84
  #4    ZRXUSDT      trend_donchian (donchian_72_6h)     +253.5    +133.0    1.00      4/4     1.96
  #5    RUNEUSDT     trend_donchian (donchian_72_6h)     +234.5    +128.8    1.00      4/4     2.69
  └─ 🚀 And 32 more pairs promoted (e.g. ARUSDT, ZILUSDT, BNBUSDT, SOLUSDT, LINKUSDT, DOTUSDT, +26 more)
  ────  ──────────   ───────────────────────────────  ─────────  ────────  ──────  ───────  ───────
  [NOT PROMOTED] 639 pairs | top: no_incremental_edgex335, quality_weight_zerox304, negative_gross_edgex143


● [LAYER 1 OUTER FOLD READINESS]
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ✅ READY (4/4 Folds Ready)

  [✅] Fold #0 (FitEnd: 2023-09-30 -> OOS: 2023-10-01 ~ 2023-12-31)
       ├─ Symbols : 14 symbols loaded [ARPAUSDT, ARUSDT, ATOMUSDT, AVAXUSDT, ENSUSDT, ETHUSDT, IOTAUSDT, MTLUSDT, +6 more]
       ├─ Events  : 918 unique events
       └─ Quality : Edge: 107.05 bps

  [✅] Fold #1 (FitEnd: 2023-12-31 -> OOS: 2023-12-31 ~ 2024-03-31)
       ├─ Symbols : 13 symbols loaded [AAVEUSDT, ADAUSDT, ANKRUSDT, ARUSDT, AVAXUSDT, BTCUSDT, NEARUSDT, RSRUSDT, +5 more]
       ├─ Events  : 2685 unique events
       └─ Quality : Edge: 38.94 bps

  [✅] Fold #2 (FitEnd: 2024-03-31 -> OOS: 2024-04-01 ~ 2024-07-01)
       ├─ Symbols : 21 symbols loaded [AAVEUSDT, ADAUSDT, ARUSDT, ATOMUSDT, BANDUSDT, BNBUSDT, CRVUSDT, DOGEUSDT, +13 more]
       ├─ Events  : 6115 unique events
       └─ Quality : Edge: 72.35 bps

  [✅] Fold #3 (FitEnd: 2024-07-01 -> OOS: 2024-07-01 ~ 2024-09-30)
       ├─ Symbols : 21 symbols loaded [AAVEUSDT, ADAUSDT, ANKRUSDT, ARUSDT, ATOMUSDT, BNBUSDT, DOTUSDT, ETCUSDT, +13 more]
       ├─ Events  : 5426 unique events
       └─ Quality : Edge: 228.39 bps

──────────────────────────────────────────────────────────────────────────────
🏁 STATUS : ✅ PASSED (5/5 Passed)
  👉 Cov:1.000(>=0.80) | Qual:1.000(>=0.90) | Brd:28.509(>=3.00) | Fld:1.000(>=0.50) | Prf:54.980(>0.00)
📈 [ENS-FINAL] Arch-Only | SYM:52 | EVT:452,492 | TOTAL:+59.5 bps | [TRD:+65.1✅ TMO:+65.5✅ MRV:+19.0✅ CRY:+24.8✅]

🏆 [L1 FINAL PROMOTION SUMMARY] 🚀 (Top 5 / 37 Promoted)
  RANK  SYMBOL       STRATEGY (Family)                EDGE(bps)  LCB(bps)    CONV    FOLDS   t(blk)
  ────  ──────────   ───────────────────────────────  ─────────  ────────  ──────  ───────  ───────
  #1    NEARUSDT     trend_donchian (donchian_72_8h)     +336.7    +191.9    1.00      4/4     3.35
  #2    GALAUSDT     trend_donchian (donchian_72_8h)     +300.7    +179.0    1.00      4/4     3.71
  #3    KAVAUSDT     trend_pullback_continuation ...     +235.3    +135.7    1.00      3/4     3.54
  #4    RUNEUSDT     trend_donchian (donchian_72_8h)     +244.6    +119.0    1.00      4/4     2.45
  #5    ZILUSDT      trend_donchian (donchian_72_8h)     +182.4    +105.6    1.00      4/4     2.61
  └─ 🚀 And 32 more pairs promoted (e.g. THETAUSDT, ANKRUSDT, ZECUSDT, AAVEUSDT, DOTUSDT, ARUSDT, +26 more)
  ────  ──────────   ───────────────────────────────  ─────────  ────────  ──────  ───────  ───────
  [NOT PROMOTED] 639 pairs | top: no_incremental_edgex354, quality_weight_zerox285, negative_gross_edgex127


● [LAYER 1 OUTER FOLD READINESS]
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ✅ READY (4/4 Folds Ready)

  [✅] Fold #0 (FitEnd: 2023-09-30 -> OOS: 2023-10-01 ~ 2023-12-31)
       ├─ Symbols : 27 symbols loaded [1000XECUSDT, AAVEUSDT, ANKRUSDT, ATOMUSDT, AVAXUSDT, AXSUSDT, BNBUSDT, DYDXUSDT, +19 more]
       ├─ Events  : 2867 unique events
       └─ Quality : Edge: 94.84 bps

  [✅] Fold #1 (FitEnd: 2023-12-31 -> OOS: 2023-12-31 ~ 2024-03-31)
       ├─ Symbols : 28 symbols loaded [1000XECUSDT, AAVEUSDT, ADAUSDT, ANKRUSDT, API3USDT, ATOMUSDT, AVAXUSDT, BANDUSDT, +20 more]
       ├─ Events  : 3369 unique events
       └─ Quality : Edge: 7.33 bps

  [✅] Fold #2 (FitEnd: 2024-03-31 -> OOS: 2024-04-01 ~ 2024-07-01)
       ├─ Symbols : 32 symbols loaded [1000XECUSDT, ADAUSDT, ANKRUSDT, API3USDT, ARUSDT, ATOMUSDT, AVAXUSDT, AXSUSDT, +24 more]
       ├─ Events  : 6474 unique events
       └─ Quality : Edge: 110.57 bps

  [✅] Fold #3 (FitEnd: 2024-07-01 -> OOS: 2024-07-01 ~ 2024-09-30)
       ├─ Symbols : 38 symbols loaded [1000SHIBUSDT, 1000XECUSDT, ADAUSDT, ANKRUSDT, ARPAUSDT, ARUSDT, ATOMUSDT, AVAXUSDT, +30 more]
       ├─ Events  : 8695 unique events
       └─ Quality : Edge: 149.05 bps

──────────────────────────────────────────────────────────────────────────────
🏁 STATUS : ✅ PASSED (5/5 Passed)
  👉 Cov:1.000(>=0.80) | Qual:1.000(>=0.90) | Brd:40.584(>=3.00) | Fld:1.000(>=0.55) | Prf:49.137(>0.00)
📈 [ENS-FINAL] Arch-Only | SYM:52 | EVT:539,813 | TOTAL:+66.6 bps | [TRD:+72.2✅ TMO:+67.9✅ MRV:+33.7✅ CRY:+27.1✅]

🏆 [L1 FINAL PROMOTION SUMMARY] 🚀 (Top 5 / 60 Promoted)
  RANK  SYMBOL       STRATEGY (Family)                EDGE(bps)  LCB(bps)    CONV    FOLDS   t(blk)
  ────  ──────────   ───────────────────────────────  ─────────  ────────  ──────  ───────  ───────
  #1    RUNEUSDT     trend_donchian (donchian_72_...     +253.5    +176.2    1.00      4/4     3.12
  #2    KAVAUSDT     trend_pullback_continuation ...     +193.9    +145.3    1.00      4/4     3.49
  #3    ZILUSDT      trend_donchian (donchian_72_...     +226.4    +135.9    1.00      4/4     3.63
  #4    ANKRUSDT     trend_donchian (donchian_72_...     +219.9    +131.4    1.00      4/4     3.90
  #5    NEARUSDT     trend_donchian (donchian_72_...     +227.8    +123.6    1.00      4/4     2.59
  └─ 🚀 And 55 more pairs promoted (e.g. GALAUSDT, BCHUSDT, ZECUSDT, ARUSDT, DOTUSDT, ENSUSDT, +49 more)
  ────  ──────────   ───────────────────────────────  ─────────  ────────  ──────  ───────  ───────
  [NOT PROMOTED] 668 pairs | top: no_incremental_edgex383, quality_weight_zerox285, negative_gross_edgex125


● [LAYER UNIVERSE AUDIT]
──────────────────────────────────────────────────────────────────────────────
  LAYER  WINDOW RANGE                    SYMS   ACTIVE (min/med/max)       ENTRY    KILL  WARNINGS
  ─────  ──────────────────────────────  ────   ────────────────────  ──────────  ──────  ────────
  L1     2023-04-01 ~ 2024-09-30           52     0 /  49.0 / 52               0       6  —
──────────────────────────────────────────────────────────────────────────────

[TIERED] Phase=l1 — stopping after L1 (not a multilayer phase)