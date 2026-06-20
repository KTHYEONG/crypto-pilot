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
  [1] Market Pool     : 426 symbols discovered (Binance USDT-M)
  [2] Capacity Limit  : 150 symbols selected (Top-N Liquidity)
  [3] Integrity Pass  : 61 symbols loaded (Passed Gaps & Frozen checks)

STRATEGY ENGINE
  Active Engine : Alpha-Ensemble Engine
  Target Scope  : 61 symbols ready for Layer 1 execution

--------------------------------------------------------------------------------

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
● [LAYER 1: SIGNAL ROBUSTNESS & ENSEMBLE VERIFICATION]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[TIERED] Sub-window admission: 56/61 symbols admitted (min_bars=1500, oos_cov>=90%)
[PROMO_FILTER] no variants recommended by diagnostics; blocking all candidates (fail-closed)
[TIERED] 💠 Scope: 56 symbols (Historical Union ∩ Data-Valid)
[ALIGN-CUBE] post-join active_mask mean=0.9107 entry_block_mean=0.0893 (was 1.0 / 0.0 before cube injection)
[WORKFLOW] Fold 0 skipped Ensemble (fit=0 < 2)
[WORKFLOW] Fold 1 skipped Ensemble (fit=0 < 2)
[WORKFLOW] Fold 2 skipped Ensemble (fit=9 < 2)
[ENS] Arch-Only   | SYM: 55 | EVT:   5,145 | TOTAL:  +9.3 bps | [BRK: +37.8✅ MOM:  +9.6✅ TRD:  -7.8❌ MRV: -11.8❌ UNI: +52.5✅]
[ENS] Arch-Only   | SYM: 55 | EVT:  36,341 | TOTAL:  +3.7 bps | [BRK: +28.2✅ MOM:  +2.5✅ TRD:  +6.6✅ MRV:  +4.3✅ UNI:  +4.9✅ F:11.0✅]
[ENS] Arch-Only   | SYM: 55 | EVT:  66,073 | TOTAL: +15.0 bps | [BRK: +18.5✅ MOM: +12.3✅ TRD: +49.1✅ MRV: +20.2✅ UNI:  -1.0❌ F:32.2✅]
[ENS] Arch-Only   | SYM: 56 | EVT:  96,345 | TOTAL: +11.5 bps | [BRK: +12.1✅ MOM:  +8.6✅ TRD: +40.6✅ MRV: +24.8✅ UNI:  -2.0❌ F:11.3✅]
[ENS] Arch-Only   | SYM: 56 | EVT: 124,607 | TOTAL: +17.4 bps | [BRK: +23.9✅ MOM: +13.6✅ TRD: +55.1✅ MRV: +24.6✅ UNI:  +3.6✅ F:13.6✅]
[ENS] Arch-Only   | SYM: 56 | EVT: 148,054 | TOTAL: +18.3 bps | [BRK: +25.5✅ MOM: +14.1✅ TRD: +52.2✅ MRV: +28.6✅ UNI:  +8.3✅ F:14.5✅]
[ENS] Arch-Only   | SYM: 56 | EVT: 173,050 | TOTAL: +20.0 bps | [BRK: +32.3✅ MOM: +15.9✅ TRD: +46.7✅ MRV: +32.9✅ UNI: +14.4✅ F:16.0✅]
[ENS] Arch-Only   | SYM: 55 | EVT:   7,495 | TOTAL: -12.0 bps | [BRK: +39.8✅ MOM: -12.2❌ TRD: -28.1❌ MRV: -37.5❌ UNI: +16.3✅]
[ENS] Arch-Only   | SYM: 56 | EVT: 188,265 | TOTAL: +20.3 bps | [BRK: +33.3✅ MOM: +16.1✅ TRD: +44.5✅ MRV: +33.3✅ UNI: +14.3✅ F:16.3✅]
[ENS] Arch-Only   | SYM: 56 | EVT: 206,656 | TOTAL: +21.2 bps | [BRK: +40.6✅ MOM: +16.4✅ TRD: +53.9✅ MRV: +30.6✅ UNI:  +6.2✅ F:17.1✅]
[ENS] Arch-Only   | SYM: 55 | EVT:  74,526 | TOTAL: +14.2 bps | [BRK: +16.4✅ MOM: +11.5✅ TRD: +41.9✅ MRV: +22.2✅ UNI:  +4.0✅ F:13.7✅]
[ENS] Arch-Only   | SYM: 56 | EVT: 135,260 | TOTAL: +17.5 bps | [BRK: +24.5✅ MOM: +14.0✅ TRD: +48.0✅ MRV: +25.0✅ UNI:  +9.9✅ F:13.8✅]
[ENS] Arch-Only   | SYM: 56 | EVT: 180,639 | TOTAL: +20.5 bps | [BRK: +32.5✅ MOM: +16.4✅ TRD: +44.3✅ MRV: +33.5✅ UNI: +14.1✅ F:16.4✅]

● [LAYER 1 OUTER FOLD READINESS]
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ✅ READY (4/4 Folds Ready)

  [✅] Fold #0 (FitEnd: 2023-08-14 -> OOS: 2023-10-01 ~ 2023-12-31)
       ├─ Symbols : 7 symbols loaded [ARPAUSDT, LINKUSDT, MANAUSDT, MKRUSDT, NEOUSDT, THETAUSDT, XRPUSDT]
       ├─ Events  : 350 unique events
       └─ Quality : Edge: 59.93 bps

  [✅] Fold #1 (FitEnd: 2023-10-31 -> OOS: 2023-12-31 ~ 2024-03-31)
       ├─ Symbols : 24 symbols loaded [ANKRUSDT, ARPAUSDT, BANDUSDT, BCHUSDT, BLZUSDT, BNBUSDT, BTCUSDT, CRVUSDT, +16 more]
       ├─ Events  : 866 unique events
       └─ Quality : Edge: 82.00 bps

  [✅] Fold #2 (FitEnd: 2024-01-17 -> OOS: 2024-04-01 ~ 2024-07-01)
       ├─ Symbols : 28 symbols loaded [ADAUSDT, ANKRUSDT, BANDUSDT, BCHUSDT, BNBUSDT, BTCUSDT, CRVUSDT, DOGEUSDT, +20 more]
       ├─ Events  : 874 unique events
       └─ Quality : Edge: 98.44 bps

  [✅] Fold #3 (FitEnd: 2024-04-04 -> OOS: 2024-07-01 ~ 2024-09-30)
       ├─ Symbols : 34 symbols loaded [1000SHIBUSDT, 1000XECUSDT, ADAUSDT, ANKRUSDT, ARPAUSDT, ARUSDT, ATOMUSDT, BLZUSDT, +26 more]
       ├─ Events  : 1148 unique events
       └─ Quality : Edge: 74.06 bps

──────────────────────────────────────────────────────────────────────────────

● [LAYER 1 HARD GATE CHECKS]
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ✅ PASSED (5/5 Passed)

  ✅ [Time-Coverage  ] :    1.000 (Target >=0.800 )
  ✅ [Signal-Quality ] :    1.000 (Target >=0.900 )
  ✅ [Symbol-Breadth ] :   36.804 (Target >=3.000 )
  ✅ [Stable-Folds   ] :    1.000 (Target >=0.500 )
  ✅ [Min-Profit     ] :   56.670 (Target >0.000  )
──────────────────────────────────────────────────────────────────────────────

[ENS-FINAL] Arch-Only   | SYM: 56 | EVT: 324,791 | TOTAL: +25.3 bps | [BRK: +33.2✅ MOM: +18.8✅ TRD: +54.1✅ MRV: +49.9✅ UNI: +15.0✅ F:20.7✅]

[L1 FINAL PROMOTION SUMMARY] 🚀
  RANK  SYMBOL       STRATEGY (Family)                EDGE(bps)  SIG(t-stat)     STATUS
  ────  ──────────   ───────────────────────────────  ─────────  ──────────────  ──────────────────
  #1    ANKRUSDT     residual_reversion (rr_24)          +128.8  [5/5] 5.01      [L2-PASS] Q:hi
  #2    LPTUSDT      residual_reversion (rr_48)          +198.8  [5/5] 4.82      [L2-PASS] Q:lo
  #3    ANKRUSDT     residual_reversion (rr_48)          +118.6  [5/5] 4.77      [L2-PASS] Q:mid
  #4    MTLUSDT      vol_breakout (bb_compress_20)       +115.6  [5/5] 4.75      [L2-PASS] Q:mid
  #5    CRVUSDT      dual_momentum (dm_24_96)            +292.5  [5/5] 4.58      [L2-PASS] Q:hi
  #6    STORJUSDT    trend_pullback_continuation ...     +359.0  [4/5] 4.12      [L2-PASS] Q:hi
  #7    LTCUSDT      funding_carry (funding_24)           +51.0  [4/5] 3.96      [L2-PASS] Q:hi
  #8    SNXUSDT      trend_pullback_continuation ...     +260.4  [4/5] 3.88      [L2-PASS] Q:hi
  #9    LTCUSDT      funding_zscore_carry (fzs_48)        +67.1  [4/5] 3.85      [L2-PASS] Q:mid
  #10   LTCUSDT      funding_zscore_carry (fzs_96)        +59.9  [4/5] 3.75      [L2-PASS] Q:hi
  #11   DOGEUSDT     funding_carry (funding_24)           +85.9  [4/5] 3.69      [L2-PASS] Q:hi
  #12   AAVEUSDT     dual_momentum (dm_12_48)            +115.8  [3/5] 3.20      [L2-PASS] Q:mid
  #13   IOTAUSDT     trend_pullback_continuation ...     +226.6  [3/5] 3.12      [L2-PASS] Q:mid
  #14   ZILUSDT      trend_pullback_continuation ...     +113.1  [3/5] 3.10      [L2-PASS] Q:mid
  #15   MTLUSDT      funding_carry (funding_24)          +136.5  [3/5] 3.09      [L2-PASS] Q:hi
  #16   UNIUSDT      trend_pullback_continuation ...     +181.7  [3/5] 3.09      [L2-PASS] Q:mid
  #17   SOLUSDT      vol_breakout (bb_compress_20)       +125.5  [3/5] 3.07      [L2-PASS] Q:hi
  #18   SNXUSDT      vol_breakout (bb_compress_20)       +125.3  [3/5] 2.99      [L2-PASS] Q:mid
  #19   LTCUSDT      dual_momentum (dm_12_48)             +64.2  [3/5] 2.99      [L2-PASS] Q:mid
  #20   ZENUSDT      dual_momentum (dm_24_96)            +169.1  [3/5] 2.99      [L2-PASS] Q:lo
  #21   BLZUSDT      residual_reversion (rr_24)          +132.4  [3/5] 2.99      [L2-PASS] Q:mid
  #22   XRPUSDT      rsi_reversion (rsi_6)                +43.5  [3/5] 2.93      [L2-PASS] Q:mid
  #23   RSRUSDT      vol_term_structure_gate (vts...     +155.5  [3/5] 2.93      [L2-PASS] Q:mid
  #24   BTCUSDT      funding_zscore_carry (fzs_48)        +32.1  [3/5] 2.92      [L2-PASS] Q:mid
  #25   ZILUSDT      trend_pullback_continuation ...     +164.6  [3/5] 2.89      [L2-PASS] Q:mid
  #26   ATOMUSDT     vol_breakout (bb_compress_20)        +80.5  [3/5] 2.80      [L2-PASS] Q:mid
  #27   RUNEUSDT     trend_donchian (donchian_72)        +133.3  [3/5] 2.77      [L2-PASS] Q:mid
  #28   BTCUSDT      funding_zscore_carry (fzs_96)        +29.1  [3/5] 2.72      [L2-PASS] Q:mid
  #29   AXSUSDT      residual_reversion (rr_24)           +52.0  [3/5] 2.68      [L2-PASS] Q:mid
  #30   ICPUSDT      trend_pullback_continuation ...     +210.6  [3/5] 2.66      [L2-PASS] Q:mid
  #31   KAVAUSDT     trend_pullback_continuation ...     +165.5  [3/5] 2.53      [L2-PASS] Q:hi
  #32   ZECUSDT      dual_momentum (dm_24_96)            +151.0  [3/5] 2.52      [L2-PASS] Q:mid
  #33   ZRXUSDT      trend_pullback_continuation ...     +254.6  [2/5] 2.46      [L2-PASS] Q:hi
  #34   LTCUSDT      funding_zscore_carry (fzs_168)       +39.8  [2/5] 2.43      [L2-PASS] Q:mid
  #35   TRBUSDT      trend_pullback_continuation ...     +291.3  [2/5] 2.40      [L2-PASS] Q:hi
  #36   BLZUSDT      dual_momentum (dm_12_48)            +135.4  [2/5] 2.40      [L2-PASS] Q:hi
  #37   1000XECUSDT  rsi_reversion (rsi_6)                +62.8  [2/5] 2.39      [L2-PASS] Q:hi
  #38   ATOMUSDT     trend_pullback_continuation ...     +134.6  [2/5] 2.36      [L2-PASS] Q:mid
  #39   1000SHIBUSDT funding_carry (funding_24)           +49.0  [2/5] 2.35      [L2-PASS] Q:mid
  #40   1000SHIBUSDT funding_zscore_carry (fzs_48)        +64.6  [2/5] 2.32      [L2-PASS] Q:mid
  #41   BNBUSDT      mtf_breakout_retest (mtf_bor...     +331.4  [2/5] 2.31      [L2-PASS] Q:lo
  #42   SANDUSDT     trend_pullback_continuation ...     +258.0  [2/5] 2.18      [L2-PASS] Q:hi
  #43   ENSUSDT      vol_term_structure_gate (vts...     +180.2  [2/5] 2.18      [L2-PASS] Q:mid
  #44   MTLUSDT      funding_zscore_carry (fzs_96)        +99.9  [2/5] 2.17      [L2-PASS] Q:lo
  #45   ZILUSDT      funding_carry (funding_24)           +61.0  [2/5] 2.16      [L2-PASS] Q:hi
  #46   GALAUSDT     trend_pullback_continuation ...     +217.3  [2/5] 2.10      [L2-PASS] Q:lo
  #47   RUNEUSDT     trend_donchian (donchian_18)         +67.3  [2/5] 2.10      [L2-PASS] Q:lo
  #48   CRVUSDT      trend_pullback_continuation ...     +215.0  [2/5] 2.08      [L2-PASS] Q:mid
  #49   ADAUSDT      funding_zscore_carry (fzs_168)       +55.7  [2/5] 2.06      [L2-PASS] Q:mid
  #50   LINKUSDT     funding_zscore_carry (fzs_168)       +56.8  [2/5] 2.04      [L2-PASS] Q:mid
  #51   TRXUSDT      vol_term_structure_gate (vts...      +22.8  [2/5] 1.97      [L2-PASS] Q:hi
  #52   RUNEUSDT     trend_donchian (donchian_36)         +87.0  [2/5] 1.95      [L2-PASS] Q:lo
  #53   LINKUSDT     funding_zscore_carry (fzs_96)        +60.7  [2/5] 1.93      [L2-PASS] Q:mid
  #54   FTMUSDT      trend_pullback_continuation ...     +124.2  [2/5] 1.88      [L2-PASS] Q:lo
  #55   RUNEUSDT     vol_term_structure_gate (vts...      +87.7  [2/5] 1.72      [L2-PASS] Q:mid
  #56   VETUSDT      funding_zscore_carry (fzs_96)        +50.3  [2/5] 1.67      [L2-PASS] Q:mid
  #57   MKRUSDT      btc_regime_pullback (btc_pul...      +41.6  [2/5] 1.58      [L2-PASS] Q:mid
  #58   THETAUSDT    funding_zscore_carry (fzs_96)        +71.2  [1/5] 1.38      [L2-PASS] Q:mid
  #59   BANDUSDT     residual_reversion (rr_24)           +69.1  [1/5] 1.32      [L2-PASS] Q:lo
  #60   THETAUSDT    trend_pullback_continuation ...      +88.2  [1/5] 1.03      [L2-PASS] Q:mid
  ────  ──────────   ───────────────────────────────  ─────────  ──────────────  ──────────────────
  [NOT PROMOTED] 1537 pairs | top: no_incremental_edgex818, quality_weight_zerox713, negative_gross_edgex388


● [LAYER UNIVERSE AUDIT]
──────────────────────────────────────────────────────────────────────────────
  LAYER  WINDOW RANGE                    SYMS   ACTIVE (min/med/max)       ENTRY    KILL  WARNINGS
  ─────  ──────────────────────────────  ────   ────────────────────  ──────────  ──────  ────────
  L1     2023-04-01 ~ 2024-09-30           56     0 /  47.0 / 51          16,470      15  —
──────────────────────────────────────────────────────────────────────────────

[TIERED] Phase=l1 — stopping after L1 (not a multilayer phase)