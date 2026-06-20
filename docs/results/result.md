Updated symbol sync profiles cache: /home/kth/my_coin_traider/data/futures/symbol_sync_profiles.json
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
[PROMO_FILTER] no variants recommended by diagnostics; blocking all candidates (fail-closed)
[ALIGN-CUBE] post-join active_mask mean=0.6818 entry_block_mean=0.2559 (was 1.0 / 0.0 before cube injection)
[WORKFLOW] Fold 0 skipped Ensemble (fit=0 < 2)
[WORKFLOW] Fold 1 skipped Ensemble (fit=0 < 2)
[ENS] Arch-Only   | SYM: 52 | EVT:   4,226 | TOTAL: +16.4 bps | [BRK: +37.1✅ MOM: +13.0✅ TRD: +13.2✅ MRV: +31.7✅ UNI: +59.6✅]
[ENS] Arch-Only   | SYM: 52 | EVT:  39,106 | TOTAL: +11.8 bps | [BRK: +26.3✅ MOM:  +7.8✅ TRD: +43.3✅ MRV: +31.7✅ UNI: -12.7❌ F:18.0✅]
[ENS] Arch-Only   | SYM: 52 | EVT:  71,108 | TOTAL: +12.2 bps | [BRK: +17.7✅ MOM:  +9.2✅ TRD: +39.4✅ MRV: +22.1✅ UNI:  +2.9✅ F:11.9✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 104,066 | TOTAL: +12.1 bps | [BRK: +15.8✅ MOM:  +8.3✅ TRD: +43.2✅ MRV: +29.6✅ UNI:  +0.6✅ F:8.9✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 132,896 | TOTAL: +16.0 bps | [BRK: +27.1✅ MOM: +12.3✅ TRD: +45.2✅ MRV: +23.5✅ UNI:  +9.3✅ F:12.4✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 162,245 | TOTAL: +18.2 bps | [BRK: +37.4✅ MOM: +13.9✅ TRD: +43.5✅ MRV: +29.1✅ UNI: +14.3✅ F:14.4✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 180,861 | TOTAL: +19.5 bps | [BRK: +36.2✅ MOM: +15.4✅ TRD: +40.3✅ MRV: +32.1✅ UNI: +15.1✅ F:15.6✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 200,255 | TOTAL: +21.1 bps | [BRK: +43.2✅ MOM: +15.8✅ TRD: +49.4✅ MRV: +35.3✅ UNI:  +7.0✅ F:17.0✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 214,349 | TOTAL: +22.3 bps | [BRK: +37.7✅ MOM: +16.3✅ TRD: +51.1✅ MRV: +44.7✅ UNI:  +6.5✅ F:18.1✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 233,383 | TOTAL: +23.6 bps | [BRK: +38.8✅ MOM: +17.5✅ TRD: +51.7✅ MRV: +43.8✅ UNI:  +8.8✅ F:19.2✅]
[ENS] Arch-Only   | SYM: 52 | EVT:  77,875 | TOTAL: +11.8 bps | [BRK: +15.2✅ MOM:  +9.0✅ TRD: +38.8✅ MRV: +20.3✅ UNI:  +1.6✅ F:11.5✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 159,105 | TOTAL: +17.8 bps | [BRK: +31.8✅ MOM: +13.6✅ TRD: +47.3✅ MRV: +26.6✅ UNI: +14.9✅ F:14.0✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 206,273 | TOTAL: +21.3 bps | [BRK: +42.9✅ MOM: +16.2✅ TRD: +45.6✅ MRV: +37.4✅ UNI:  +7.0✅ F:17.1✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 264,007 | TOTAL: +23.7 bps | [BRK: +37.4✅ MOM: +17.8✅ TRD: +50.8✅ MRV: +45.0✅ UNI: +10.4✅ F:19.3✅]

● [LAYER 1 OUTER FOLD READINESS]
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ✅ READY (4/4 Folds Ready)

  [✅] Fold #0 (FitEnd: 2023-09-30 -> OOS: 2023-10-01 ~ 2023-12-31)
       ├─ Symbols : 16 symbols loaded [1000XECUSDT, AAVEUSDT, ANKRUSDT, ARPAUSDT, BCHUSDT, ETHUSDT, FILUSDT, GALAUSDT, +8 more]
       ├─ Events  : 712 unique events
       └─ Quality : Edge: 103.20 bps

  [✅] Fold #1 (FitEnd: 2023-12-31 -> OOS: 2023-12-31 ~ 2024-03-31)
       ├─ Symbols : 17 symbols loaded [1000XECUSDT, ANKRUSDT, API3USDT, ARUSDT, BANDUSDT, BTCUSDT, ETHUSDT, GALAUSDT, +9 more]
       ├─ Events  : 420 unique events
       └─ Quality : Edge: 15.14 bps

  [✅] Fold #2 (FitEnd: 2024-03-31 -> OOS: 2024-04-01 ~ 2024-07-01)
       ├─ Symbols : 29 symbols loaded [1000XECUSDT, ANKRUSDT, BANDUSDT, BCHUSDT, BNBUSDT, BTCUSDT, CRVUSDT, DOGEUSDT, +21 more]
       ├─ Events  : 859 unique events
       └─ Quality : Edge: 85.59 bps

  [✅] Fold #3 (FitEnd: 2024-07-01 -> OOS: 2024-07-01 ~ 2024-09-30)
       ├─ Symbols : 33 symbols loaded [1000SHIBUSDT, 1000XECUSDT, ANKRUSDT, ARPAUSDT, ARUSDT, BNBUSDT, BTCUSDT, CRVUSDT, +25 more]
       ├─ Events  : 937 unique events
       └─ Quality : Edge: 68.96 bps

──────────────────────────────────────────────────────────────────────────────

● [LAYER 1 HARD GATE CHECKS]
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ✅ PASSED (5/5 Passed)

  ✅ [Time-Coverage  ] :    1.000 (Target >=0.800 )
  ✅ [Signal-Quality ] :    1.000 (Target >=0.900 )
  ✅ [Symbol-Breadth ] :   36.538 (Target >=3.000 )
  ✅ [Stable-Folds   ] :    1.000 (Target >=0.500 )
  ✅ [Min-Profit     ] :   26.873 (Target >0.000  )
──────────────────────────────────────────────────────────────────────────────

[ENS-FINAL] Arch-Only   | SYM: 52 | EVT: 314,823 | TOTAL: +25.3 bps | [BRK: +35.1✅ MOM: +18.3✅ TRD: +55.4✅ MRV: +49.1✅ UNI: +15.8✅ F:20.7✅]

[L1 FINAL PROMOTION SUMMARY] 🚀
  RANK  SYMBOL       STRATEGY (Family)                EDGE(bps)  SIG(t-stat)     STATUS
  ────  ──────────   ───────────────────────────────  ─────────  ──────────────  ──────────────────
  #1    ANKRUSDT     residual_reversion (rr_24)          +130.6  [5/5] 5.08      [L2-PASS] Q:hi
  #2    ANKRUSDT     residual_reversion (rr_48)          +120.2  [5/5] 4.83      [L2-PASS] Q:mid
  #3    CRVUSDT      dual_momentum (dm_24_96)            +290.6  [5/5] 4.55      [L2-PASS] Q:hi
  #4    ZENUSDT      dual_momentum (dm_24_96)            +213.4  [4/5] 4.23      [L2-PASS] Q:hi
  #5    STORJUSDT    trend_pullback_continuation ...     +357.3  [4/5] 4.11      [L2-PASS] Q:hi
  #6    LTCUSDT      funding_carry (funding_24)           +50.8  [4/5] 3.95      [L2-PASS] Q:hi
  #7    SNXUSDT      trend_pullback_continuation ...     +258.6  [4/5] 3.86      [L2-PASS] Q:hi
  #8    LTCUSDT      funding_zscore_carry (fzs_48)        +66.9  [4/5] 3.83      [L2-PASS] Q:hi
  #9    DOGEUSDT     funding_carry (funding_24)           +86.1  [4/5] 3.70      [L2-PASS] Q:hi
  #10   LTCUSDT      funding_zscore_carry (fzs_96)        +58.9  [4/5] 3.69      [L2-PASS] Q:hi
  #11   IOTAUSDT     trend_pullback_continuation ...     +237.2  [3/5] 3.37      [L2-PASS] Q:mid
  #12   AAVEUSDT     dual_momentum (dm_12_48)            +116.7  [3/5] 3.23      [L2-PASS] Q:mid
  #13   SOLUSDT      vol_breakout (bb_compress_20)       +127.1  [3/5] 3.11      [L2-PASS] Q:hi
  #14   LTCUSDT      dual_momentum (dm_12_48)             +65.3  [3/5] 3.04      [L2-PASS] Q:mid
  #15   ZILUSDT      trend_pullback_continuation ...     +110.8  [3/5] 3.03      [L2-PASS] Q:mid
  #16   UNIUSDT      trend_pullback_continuation ...     +177.9  [3/5] 3.02      [L2-PASS] Q:mid
  #17   SNXUSDT      vol_breakout (bb_compress_20)       +125.9  [3/5] 3.01      [L2-PASS] Q:mid
  #18   RSRUSDT      vol_term_structure_gate (vts...     +155.8  [3/5] 2.93      [L2-PASS] Q:mid
  #19   1000XECUSDT  dual_momentum (dm_12_48)            +107.8  [3/5] 2.93      [L2-PASS] Q:hi
  #20   XRPUSDT      rsi_reversion (rsi_6)                +43.4  [3/5] 2.92      [L2-PASS] Q:hi
  #21   ZILUSDT      trend_pullback_continuation ...     +164.4  [3/5] 2.88      [L2-PASS] Q:mid
  #22   BTCUSDT      funding_zscore_carry (fzs_48)        +31.4  [3/5] 2.86      [L2-PASS] Q:mid
  #23   ATOMUSDT     vol_breakout (bb_compress_20)        +81.9  [3/5] 2.85      [L2-PASS] Q:mid
  #24   RUNEUSDT     trend_donchian (donchian_72)        +134.4  [3/5] 2.79      [L2-PASS] Q:mid
  #25   BTCUSDT      funding_zscore_carry (fzs_96)        +28.7  [3/5] 2.69      [L2-PASS] Q:mid
  #26   MTLUSDT      funding_carry (funding_24)          +106.0  [3/5] 2.69      [L2-PASS] Q:hi
  #27   AXSUSDT      residual_reversion (rr_24)           +51.9  [3/5] 2.68      [L2-PASS] Q:hi
  #28   KAVAUSDT     trend_pullback_continuation ...     +165.3  [3/5] 2.53      [L2-PASS] Q:hi
  #29   ZRXUSDT      trend_pullback_continuation ...     +253.5  [2/5] 2.45      [L2-PASS] Q:hi
  #30   RVNUSDT      trend_pullback_continuation ...     +185.7  [2/5] 2.43      [L2-PASS] Q:mid
  #31   TRBUSDT      trend_pullback_continuation ...     +294.1  [2/5] 2.42      [L2-PASS] Q:hi
  #32   LTCUSDT      funding_zscore_carry (fzs_168)       +39.5  [2/5] 2.42      [L2-PASS] Q:hi
  #33   MTLUSDT      vol_term_structure_gate (vts...     +153.1  [2/5] 2.38      [L2-PASS] Q:hi
  #34   MTLUSDT      funding_zscore_carry (fzs_96)       +104.6  [2/5] 2.37      [L2-PASS] Q:mid
  #35   1000SHIBUSDT funding_carry (funding_24)           +49.2  [2/5] 2.36      [L2-PASS] Q:mid
  #36   ATOMUSDT     trend_pullback_continuation ...     +134.3  [2/5] 2.36      [L2-PASS] Q:mid
  #37   1000SHIBUSDT funding_zscore_carry (fzs_48)        +64.3  [2/5] 2.31      [L2-PASS] Q:mid
  #38   BNBUSDT      mtf_breakout_retest (mtf_bor...     +327.5  [2/5] 2.28      [L2-PASS] Q:lo
  #39   SANDUSDT     trend_pullback_continuation ...     +256.7  [2/5] 2.17      [L2-PASS] Q:hi
  #40   ENSUSDT      vol_term_structure_gate (vts...     +178.7  [2/5] 2.16      [L2-PASS] Q:mid
  #41   ZILUSDT      funding_carry (funding_24)           +60.6  [2/5] 2.14      [L2-PASS] Q:hi
  #42   RUNEUSDT     trend_donchian (donchian_18)         +67.6  [2/5] 2.11      [L2-PASS] Q:lo
  #43   GALAUSDT     trend_pullback_continuation ...     +217.3  [2/5] 2.10      [L2-PASS] Q:lo
  #44   ADAUSDT      funding_zscore_carry (fzs_168)       +56.2  [2/5] 2.08      [L2-PASS] Q:mid
  #45   CRVUSDT      trend_pullback_continuation ...     +213.5  [2/5] 2.07      [L2-PASS] Q:mid
  #46   LINKUSDT     funding_zscore_carry (fzs_168)       +56.5  [2/5] 2.03      [L2-PASS] Q:mid
  #47   TRXUSDT      vol_term_structure_gate (vts...      +23.3  [2/5] 2.00      [L2-PASS] Q:hi
  #48   RUNEUSDT     trend_donchian (donchian_36)         +87.5  [2/5] 1.96      [L2-PASS] Q:lo
  #49   LINKUSDT     funding_zscore_carry (fzs_96)        +60.1  [2/5] 1.91      [L2-PASS] Q:mid
  #50   RUNEUSDT     vol_term_structure_gate (vts...      +85.9  [2/5] 1.68      [L2-PASS] Q:hi
  #51   VETUSDT      funding_zscore_carry (fzs_96)        +50.1  [2/5] 1.66      [L2-PASS] Q:hi
  #52   THETAUSDT    funding_zscore_carry (fzs_96)        +71.8  [1/5] 1.39      [L2-PASS] Q:mid
  #53   BANDUSDT     residual_reversion (rr_24)           +67.8  [1/5] 1.29      [L2-PASS] Q:lo
  #54   THETAUSDT    trend_pullback_continuation ...      +86.9  [1/5] 1.01      [L2-PASS] Q:mid
  #55   ZENUSDT      funding_zscore_carry (fzs_168)       +54.1  [1/5] 0.93      [L2-PASS] Q:mid
  ────  ──────────   ───────────────────────────────  ─────────  ──────────────  ──────────────────
  [NOT PROMOTED] 1430 pairs | top: no_incremental_edgex760, quality_weight_zerox668, negative_gross_edgex346


● [LAYER UNIVERSE AUDIT]
──────────────────────────────────────────────────────────────────────────────
  LAYER  WINDOW RANGE                    SYMS   ACTIVE (min/med/max)       ENTRY    KILL  WARNINGS
  ─────  ──────────────────────────────  ────   ────────────────────  ──────────  ──────  ────────
  L1     2023-04-01 ~ 2024-09-30           52     0 /  49.0 / 52               0       6  —
──────────────────────────────────────────────────────────────────────────────

[TIERED] Phase=l1 — stopping after L1 (not a multilayer phase)