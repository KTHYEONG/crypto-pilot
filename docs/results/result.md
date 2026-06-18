[SYNC] Ledger up-to-date (Last: 2026-03-31)
discover_universe_timeline: l2_start(2024-10-01) < oos_start(2025-10-01), current universe timeline is still 2-way and does not treat l2_start as a separate boundary
[UNIVERSE] 🌐 2022-04-01 ~ 2026-03-31 | Target: 94 symbols
[SQL-DB]   💾 Loaded start dates from universe_ledger.db
Sync mode=full targeted_symbols=94
Loaded symbol sync profiles from cache: /home/kth/my_coin_traider/data/futures/symbol_sync_profiles.json
[SQL-DB]   ⚡ [SKIPPED] All symbols are up-to-date. No sync or disk scans required.
┌────────────────────────────────────────────────────────────────────────────────┐
│ ● [SYSTEM CONTEXT: INFRASTRUCTURE & DATA PREPARATION]                          │
├────────────────────────────────────────────────────────────────────────────────┤
│ Window:   Range: 2022-10-01 ~ 2026-03-31 (IS:2023-10-01, OOS:2025-10-01)       │
│ Universe: Discovered: 94 symbols | Selected: 20 | Live Panel: 20               │
│ Quality:  Loaded: 59.6% (56/94) | Ready: 56 | Dropped: None                    │
│ Strategy: Engine: Alpha-Ensemble Engine | Inf Panel: 94 | Trade Scope: 56      │
└────────────────────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
● [LAYER 1: SIGNAL ROBUSTNESS & ENSEMBLE VERIFICATION]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

● [DATA-INTEGRITY AUDIT]
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ✅ ALL 54 SYMBOLS PASSED
  METRICS : Total Bars: 8,605
  DETAIL  : [NaN: 0.0%] [Zero/Neg: 0.0%] [Range: PASS]
──────────────────────────────────────────────────────────────────────────────

[TIERED] 💠 Scope: 54 symbols (Historical Union ∩ Data-Valid)
[ENS] Arch-Only   | SYM:  3 | EVT:     353 | TOTAL: +40.8 bps | IC:-0.04❌ | [BRK: -13.4❌ MOM: +26.2✅ TRD:+214.7✅ MRV: +32.4✅ UNI:  +8.3✅]
[ENS] Arch-Only   | SYM:  3 | EVT:   2,101 | TOTAL: +12.9 bps | IC:-0.03❌ | [BRK:  -4.0❌ MOM:  +9.9✅ TRD: +25.8✅ MRV: +40.3✅ UNI:  +9.8✅]
[ENS] Arch-Only   | SYM:  3 | EVT:   3,755 | TOTAL: +12.6 bps | IC:-0.03❌ | [BRK:  -0.1❌ MOM: +10.8✅ TRD: +19.4✅ MRV: +25.1✅ UNI: +16.4✅]
[ENS] Arch-Only   | SYM: 18 | EVT:   6,587 | TOTAL: +15.2 bps | IC:-0.01❌ | [BRK: +30.7✅ MOM: +13.0✅ TRD: +19.0✅ MRV: +16.8✅ UNI: +52.2✅]
[ENS] Arch-Only   | SYM: 18 | EVT:  16,309 | TOTAL:  +5.1 bps | IC:-0.03❌ | [BRK: +21.2✅ MOM:  +3.7✅ TRD: +11.3✅ MRV:  +6.0✅ UNI: +11.6✅]
[ENS] Arch-Only   | SYM: 18 | EVT:  26,073 | TOTAL: +13.4 bps | IC:-0.02❌ | [BRK: +13.7✅ MOM: +10.5✅ TRD: +48.3✅ MRV: +17.8✅ UNI:  +3.3✅]
[ENS] Arch-Only   | SYM: 26 | EVT:  34,059 | TOTAL:  +9.5 bps | IC:-0.02❌ | [BRK: +10.1✅ MOM:  +6.6✅ TRD: +37.2✅ MRV: +19.6✅ UNI:  -0.0❌]
[ENS] Arch-Only   | SYM: 26 | EVT:  42,600 | TOTAL: +15.0 bps | IC:-0.05❌ | [BRK: +21.2✅ MOM: +11.1✅ TRD: +52.7✅ MRV: +20.5✅ UNI:  +6.6✅ F:2.7✅]
[ENS] Arch-Only   | SYM: 26 | EVT:  49,357 | TOTAL: +15.8 bps | IC:-0.09❌ | [BRK: +24.0✅ MOM: +12.1✅ TRD: +44.1✅ MRV: +23.3✅ UNI: +13.5✅ F:4.7✅]
[ENS] Arch-Only   | SYM: 30 | EVT:  56,130 | TOTAL: +16.2 bps | IC:-0.10❌ | [BRK: +24.3✅ MOM: +12.4✅ TRD: +44.5✅ MRV: +21.7✅ UNI: +17.8✅ F:6.0✅]
[ENS] Arch-Only   | SYM: 34 | EVT:  61,011 | TOTAL: +16.5 bps | IC:-0.05❌ | [BRK: +24.6✅ MOM: +13.0✅ TRD: +40.6✅ MRV: +21.2✅ UNI: +18.5✅ F:6.3✅]
[ENS] Arch-Only   | SYM: 34 | EVT:  67,028 | TOTAL: +16.6 bps | IC:-0.05❌ | [BRK: +32.7✅ MOM: +12.7✅ TRD: +46.3✅ MRV: +19.9✅ UNI:  +7.9✅ F:6.9✅]
[ENS] Arch-Only   | SYM: 18 | EVT:   7,371 | TOTAL:  +8.2 bps | IC:+0.05✅ | [BRK: +29.3✅ MOM:  +5.8✅ TRD: +15.4✅ MRV:  +9.6✅ UNI: +38.9✅]
[ENS] Arch-Only   | SYM: 18 | EVT:  28,683 | TOTAL: +10.0 bps | IC:+0.00✅ | [BRK: +11.8✅ MOM:  +7.3✅ TRD: +37.9✅ MRV: +15.9✅ UNI:  +1.3✅]
[ENS] Arch-Only   | SYM: 26 | EVT:  45,651 | TOTAL: +14.7 bps | IC:-0.08❌ | [BRK: +25.4✅ MOM: +11.5✅ TRD: +39.5✅ MRV: +19.9✅ UNI: +13.8✅ F:3.9✅]
[ENS] Arch-Only   | SYM: 34 | EVT:  58,632 | TOTAL: +16.3 bps | IC:-0.06❌ | [BRK: +23.7✅ MOM: +13.0✅ TRD: +40.7✅ MRV: +20.4✅ UNI: +17.7✅ F:6.3✅]

● [LAYER 1 OUTER FOLD READINESS]
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ✅ READY (3/4 Folds Ready)

  [✅] Fold #0 (FitEnd: 2023-08-14 -> OOS: 2023-10-01 ~ 2023-12-31)
       ├─ Symbols : 10 symbols loaded [BTCUSDT, CRVUSDT, ETCUSDT, ETHUSDT, FTMUSDT, NEOUSDT, SNXUSDT, SOLUSDT, +2 more]
       ├─ Events  : 1014 unique events
       └─ Quality : Edge: 80.92 bps

  [❌] Fold #1 (FitEnd: 2023-10-31 -> OOS: 2023-12-31 ~ 2024-03-31)
       ├─ Symbols : 13 symbols loaded [AAVEUSDT, ADAUSDT, ARPAUSDT, BTCUSDT, CRVUSDT, ETCUSDT, ETHUSDT, LTCUSDT, +5 more]
       ├─ Events  : 880 unique events
       └─ Quality : Edge: -5.81 bps
       └─ BLOCKERS: non_positive_gross_edge

  [✅] Fold #2 (FitEnd: 2024-01-17 -> OOS: 2024-04-01 ~ 2024-07-01)
       ├─ Symbols : 13 symbols loaded [ADAUSDT, BCHUSDT, BTCUSDT, CRVUSDT, ETHUSDT, FILUSDT, LINKUSDT, MKRUSDT, +5 more]
       ├─ Events  : 937 unique events
       └─ Quality : Edge: 187.96 bps

  [✅] Fold #3 (FitEnd: 2024-04-04 -> OOS: 2024-07-01 ~ 2024-09-30)
       ├─ Symbols : 9 symbols loaded [ANKRUSDT, BTCUSDT, DOGEUSDT, ETHUSDT, FTMUSDT, GALAUSDT, NEARUSDT, SOLUSDT, +1 more]
       ├─ Events  : 535 unique events
       └─ Quality : Edge: 18.07 bps

──────────────────────────────────────────────────────────────────────────────

● [LAYER 1 HARD GATE CHECKS]
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ✅ PASSED (5/5 Passed)

  ✅ [Time-Coverage  ] :    1.000 (Target >=0.800 )
  ✅ [Signal-Quality ] :    1.000 (Target >=0.900 )
  ✅ [Symbol-Breadth ] :   18.925 (Target >=3.000 )
  ✅ [Stable-Folds   ] :    0.750 (Target >=0.500 )
  ✅ [Min-Profit     ] :   51.689 (Target >0.000  )
──────────────────────────────────────────────────────────────────────────────

[ENS-FINAL] Arch-Only   | SYM: 45 | EVT: 100,404 | TOTAL: +22.3 bps | IC:-0.05❌ | [BRK: +27.2✅ MOM: +16.3✅ TRD: +52.7✅ MRV: +41.7✅ UNI: +15.2✅ F:12.3✅]

[L1 FINAL PROMOTION SUMMARY] 🚀
  RANK  SYMBOL       STRATEGY (Family)                EDGE(bps)  SIG(t-stat)     STATUS
  ────  ──────────   ───────────────────────────────  ─────────  ──────────────  ──────────────────
  #1    STORJUSDT    trend_pullback_continuation ...     +383.7  [4/5] 4.25      PROMOTED (Best Q)
  #2    SOLUSDT      vol_breakout (bb_compress_20)       +152.3  [4/5] 4.17      PROMOTED (Best Q)
  #3    ETHUSDT      vol_breakout (bb_compress_20)        +44.0  [4/5] 3.67      WATCH
  #4    STORJUSDT    trend_pullback_continuation ...     +142.5  [4/5] 3.57      PROMOTED (Best Q)
  #5    ICPUSDT      dual_momentum (dm_24_96)            +242.5  [3/5] 3.50      PROMOTED (Best Q)
  #6    ZECUSDT      dual_momentum (dm_12_48)            +178.5  [3/5] 3.44      PROMOTED (Best Q)
  #7    BTCUSDT      trend_donchian (donchian_18)         +37.1  [3/5] 3.40      PROMOTED
  #8    SNXUSDT      trend_pullback_continuation ...     +382.9  [3/5] 3.24      PROMOTED (Best Q)
  #9    SNXUSDT      vol_breakout (bb_compress_20)       +129.3  [3/5] 3.22      PROMOTED (Best Q)
  #10   LTCUSDT      funding_extreme_reversal (fe...      +53.5  [3/5] 3.08      PROMOTED (Best Q)
  #11   BTCUSDT      funding_zscore_carry (fzs_48)        +33.3  [3/5] 3.03      PROMOTED (Best Q)
  #12   BTCUSDT      funding_zscore_carry (fzs_96)        +30.3  [3/5] 2.83      PROMOTED (Best Q)
  #13   ICPUSDT      vol_regime_reversion (vrr_40)       +117.5  [3/5] 2.82      REJECTED
  #14   SNXUSDT      mtf_trend_pullback (mtf_tpb_...     +216.3  [3/5] 2.71      PROMOTED (Best Q)
  #15   LINKUSDT     trend_pullback_continuation ...     +272.2  [3/5] 2.68      PROMOTED
  #16   ETCUSDT      dual_momentum (dm_12_48)            +129.1  [3/5] 2.63      PROMOTED (Best Q)
  #17   NEOUSDT      mtf_trend_pullback (mtf_tpb_...     +358.4  [3/5] 2.62      WATCH
  #18   MKRUSDT      rsi_reversion (rsi_14)              +108.6  [3/5] 2.57      PROMOTED (Best Q)
  #19   ETCUSDT      vol_term_structure_gate (vts...     +179.3  [3/5] 2.57      PROMOTED
  #20   LTCUSDT      funding_zscore_carry (fzs_168)       +61.4  [3/5] 2.50      PROMOTED (Best Q)
  #21   CRVUSDT      trend_pullback_continuation ...     +268.3  [2/5] 2.49      PROMOTED (Best Q)
  #22   ADAUSDT      bollinger_reversion (bolling...      +55.3  [2/5] 2.47      PROMOTED
  #23   ARPAUSDT     funding_zscore_carry (fzs_168)      +102.7  [2/5] 2.44      PROMOTED (Best Q)
  #24   SNXUSDT      trend_pullback_continuation ...     +169.7  [2/5] 2.44      PROMOTED (Best Q)
  #25   ICPUSDT      residual_reversion (rr_24)           +86.9  [2/5] 2.42      PROMOTED (Best Q)
  #26   FILUSDT      rsi_reversion (rsi_14)               +84.2  [2/5] 2.37      PROMOTED (Best Q)
  #27   LTCUSDT      dual_momentum (dm_12_48)             +76.9  [2/5] 2.28      WATCH
  #28   STORJUSDT    trend_donchian (donchian_72)        +128.1  [2/5] 2.22      REJECTED
  #29   ADAUSDT      mtf_trend_pullback (mtf_tpb_...     +188.9  [2/5] 2.22      PROMOTED
  #30   ETHUSDT      trend_pullback_continuation ...      +73.6  [2/5] 2.20      REJECTED
  #31   BTCUSDT      vol_breakout (bb_compress_20)        +21.4  [2/5] 2.19      WATCH
  #32   ANKRUSDT     trend_ma (ema_12_72)                 +46.5  [2/5] 2.18      PROMOTED (Best Q)
  #33   LTCUSDT      funding_carry (funding_24)           +56.2  [2/5] 2.13      PROMOTED (Best Q)
  #34   NEARUSDT     mtf_breakout_retest (mtf_bor...     +205.2  [2/5] 2.11      WATCH
  #35   DOGEUSDT     funding_carry (funding_24)           +93.4  [2/5] 2.11      PROMOTED (Best Q)
  #36   CRVUSDT      trend_ma (ema_12_72)                 +69.2  [2/5] 2.10      PROMOTED (Best Q)
  #37   ADAUSDT      funding_zscore_carry (fzs_168)       +60.8  [2/5] 2.06      PROMOTED (Best Q)
  #38   BTCUSDT      dual_momentum (dm_12_48)             +48.2  [2/5] 2.06      REJECTED
  #39   FTMUSDT      funding_carry (funding_24)           +68.5  [2/5] 2.02      WATCH
  #40   LTCUSDT      funding_zscore_carry (fzs_48)        +49.8  [2/5] 1.98      REJECTED
  #41   ADAUSDT      funding_zscore_carry (fzs_96)        +69.3  [2/5] 1.94      PROMOTED (Best Q)
  #42   LINKUSDT     rsi_reversion (rsi_6)                +41.6  [2/5] 1.91      PROMOTED
  #43   ADAUSDT      trend_pullback_continuation ...      +63.7  [2/5] 1.82      WATCH
  #44   BCHUSDT      funding_zscore_carry (fzs_96)        +89.5  [2/5] 1.81      WATCH
  #45   BCHUSDT      funding_zscore_carry (fzs_168)       +44.9  [2/5] 1.78      REJECTED
  #46   LTCUSDT      funding_zscore_carry (fzs_96)        +37.0  [2/5] 1.72      PROMOTED (Best Q)
  #47   RUNEUSDT     rsi_reversion (rsi_6)                +58.6  [2/5] 1.67      REJECTED
  #48   MKRUSDT      rsi_reversion (rsi_6)                +74.8  [2/5] 1.63      PROMOTED
  #49   NEOUSDT      funding_zscore_carry (fzs_168)       +59.1  [2/5] 1.62      PROMOTED (Best Q)
  #50   ICPUSDT      trend_donchian (donchian_36)         +93.0  [2/5] 1.61      WATCH
  #51   FTMUSDT      trend_pullback_continuation ...     +215.5  [2/5] 1.60      PROMOTED (Best Q)
  #52   LINKUSDT     funding_carry (funding_24)           +50.4  [2/5] 1.60      PROMOTED
  #53   GALAUSDT     vol_regime_reversion (vrr_40)       +104.8  [2/5] 1.60      WATCH
  #54   ETHUSDT      dual_momentum (dm_12_48)             +41.5  [2/5] 1.59      WATCH
  #55   SNXUSDT      funding_zscore_carry (fzs_168)       +57.3  [2/5] 1.58      WATCH
  #56   ARPAUSDT     taker_imbalance_momentum (ti...      +75.3  [2/5] 1.58      PROMOTED
  #57   RUNEUSDT     trend_donchian (donchian_72)         +58.0  [2/5] 1.56      WATCH
  #58   NEOUSDT      bollinger_reversion (bolling...      +43.7  [1/5] 1.45      WATCH
  #59   ICPUSDT      mtf_breakout_retest (mtf_bor...     +198.4  [1/5] 1.45      REJECTED
  #60   RUNEUSDT     trend_ma (ema_12_72)                 +24.8  [1/5] 1.42      PROMOTED
  #61   LINKUSDT     funding_zscore_carry (fzs_96)        +58.2  [1/5] 1.41      WATCH
  #62   ETHUSDT      funding_zscore_carry (fzs_168)       +15.3  [1/5] 1.34      WATCH
  #63   ETHUSDT      rsi_reversion (rsi_14)               +15.2  [1/5] 1.31      WATCH
  #64   ETHUSDT      funding_zscore_carry (fzs_96)        +16.8  [1/5] 1.29      WATCH
  #65   NEARUSDT     bollinger_reversion (bolling...      +50.6  [1/5] 1.28      REJECTED
  #66   AAVEUSDT     dual_momentum (dm_24_96)            +164.7  [1/5] 1.26      PROMOTED
  #67   DOGEUSDT     trend_pullback_continuation ...     +158.9  [1/5] 1.23      WATCH
  #68   LINKUSDT     funding_zscore_carry (fzs_168)       +44.9  [1/5] 1.21      PROMOTED
  #69   ETHUSDT      trend_ma (ema_18_108)                 +8.0  [1/5] 1.12      WATCH
  #70   NEARUSDT     vol_term_structure_gate (vts...      +76.1  [1/5] 1.11      PROMOTED (Best Q)
  #71   ADAUSDT      funding_zscore_carry (fzs_48)        +41.0  [1/5] 1.10      WATCH
  #72   BCHUSDT      trend_ma (ema_12_72)                 +15.7  [1/5] 1.09      REJECTED
  #73   NEARUSDT     funding_zscore_carry (fzs_168)       +58.9  [1/5] 1.07      PROMOTED (Best Q)
  #74   ETHUSDT      funding_extreme_reversal (fe...      +11.1  [1/5] 1.03      PROMOTED
  #75   FTMUSDT      trend_ma (ema_12_72)                 +18.5  [1/5] 1.02      WATCH
  #76   ADAUSDT      rsi_reversion (rsi_6)                +18.0  [1/5] 1.00      WATCH
  #77   BCHUSDT      dual_momentum (dm_24_96)            +117.2  [1/5] 0.94      PROMOTED
  #78   RUNEUSDT     trend_donchian (donchian_18)         +27.3  [1/5] 0.92      REJECTED
  #79   DOGEUSDT     trend_ma (ema_12_72)                 +17.1  [1/5] 0.86      WATCH
  #80   LINKUSDT     funding_zscore_carry (fzs_48)        +47.6  [1/5] 0.86      WATCH
  #81   NEOUSDT      rsi_reversion (rsi_6)                +22.0  [1/5] 0.83      PROMOTED
  #82   LINKUSDT     residual_reversion (rr_24)           +31.9  [1/5] 0.82      REJECTED
  #83   SOLUSDT      funding_zscore_carry (fzs_168)       +37.5  [1/5] 0.81      WATCH
  #84   NEOUSDT      funding_zscore_carry (fzs_96)        +39.2  [1/5] 0.78      REJECTED
  #85   ETCUSDT      btc_regime_pullback (btc_pul...      +37.4  [1/5] 0.77      REJECTED
  #86   NEARUSDT     funding_zscore_carry (fzs_96)        +66.1  [1/5] 0.75      PROMOTED (Best Q)
  #87   RUNEUSDT     dual_momentum (dm_12_48)             +46.2  [1/5] 0.74      PROMOTED
  #88   ADAUSDT      rsi_reversion (rsi_14)               +18.7  [1/5] 0.72      REJECTED
  #89   FTMUSDT      trend_pullback_continuation ...      +81.8  [1/5] 0.67      REJECTED
  #90   ICPUSDT      dual_momentum (dm_12_48)             +32.0  [1/5] 0.39      REJECTED
  #91   ICPUSDT      trend_donchian (donchian_18)         +32.2  [1/5] 0.32      WATCH
  ────  ──────────   ───────────────────────────────  ─────────  ──────────────  ──────────────────


● [LAYER UNIVERSE AUDIT]
──────────────────────────────────────────────────────────────────────────────
  LAYER  WINDOW RANGE                    SYMS   ACTIVE (min/med/max)       ENTRY    KILL  WARNINGS
  ─────  ──────────────────────────────  ────   ────────────────────  ──────────  ──────  ────────
  L1     2023-04-01 ~ 2024-09-30           54     0 /  15.0 / 18         135,022      41  —
──────────────────────────────────────────────────────────────────────────────


>> LAYER 1: PASS -> Proceeding to Layer 2.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
● [LAYER 2: PORTFOLIO ALLOCATION & RISK OPTIMIZATION]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ● [HYPERPARAMETER OPTIMIZATION]
    - Study Name : l2_study_4h_00c9dc842b63
    - Config     : 200 trials
  ────────────────────────────────────────────────────────────────────────────
 [OPT] Deleted existing study 'l2_study_4h_00c9dc842b63' for a fresh start.
[L2-OPT]: 100%|█████████████████████████████████████████████████████| 200/200 [01:35<00:00,  2.10it/s, Best CAGR: 257.29% | Current: -37.84%]
[L2-SELECTION] No feasible candidate found within fallback window (reason=cagr)
[L2-DEPLOY-C4] L*=1.313 (binding=mdd) | realized_mode=return_scaling | kelly=0.250(불변) | tf=4h

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
● [LAYER 2: PORTFOLIO ALLOCATION & RISK OPTIMIZATION]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[L2-DEPLOY] L*=1.3134 binding=champion | CAGR=0.2453 MDD=0.1312 CVaR95=0.0095 RiskUtil=0.437
● [LAYER 2 PORTFOLIO SCORECARD] (2024-12-22 ~ 2025-09-30)
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ❌ BLOCKED (cagr)

  ❌ [Growth    ] CAGR: +24.5% (>=30.0%) | PnL: +14.1% | Equity x1.14
  ✅ [Efficiency] Sharpe: 1.230 (>=1.000) | Sortino: 1.934 (>=1.500) | Calmar: 1.869 (>=1.000)
  ✅ [Risk      ] MDD: 13.1% (<=30.0%) | CVaR95: 0.9% (<=6.0%) | RiskUtil: 43.7%
  ✅ [Robust    ] Fold: 100.0% (>=60.0%) | Trades: 300 (>=30) | Friction: 100.0%
  ❌ [Uplift    ] Sharpe Uplift: +0.16 (>=+0.20)
  ✅ [Integrity ] DSR: 0.834 (>=0.60) | PSR: 0.864 (diag)
  [Diag     ] RelMDD: 1.42x | Turnover: 0.142
──────────────────────────────────────────────────────────────────────────────

  [ FOLD DETAIL BREAKDOWN ]
  ──────────────────────────────────────────────────────────────────────────
  ├─ Fold #1 : ✅ Sharpe:  1.453 | CAGR:   +31.6% | MDD:  13.1% | Status: PASS | Period: 2024-12-22 ~ 2025-03-26
       Symbols: 10 [AAVEUSDT, BTCUSDT, CRVUSDT, DOGEUSDT, ETHUSDT, FTMUSDT, LINKUSDT, NEARUSDT, +2 more]
  ├─ Fold #2 : ✅ Sharpe:  2.962 | CAGR:   +27.0% | MDD:   3.6% | Status: PASS | Period: 2025-03-26 ~ 2025-06-28
       Symbols: 9 [AAVEUSDT, ADAUSDT, BTCUSDT, CRVUSDT, DOGEUSDT, ETHUSDT, FTMUSDT, LINKUSDT, +1 more]
  └─ Fold #3 : ✅ Sharpe:  0.698 | CAGR:   +15.6% | MDD:   9.7% | Status: PASS | Period: 2025-06-28 ~ 2025-09-30
       Symbols: 12 [AAVEUSDT, ADAUSDT, BCHUSDT, BTCUSDT, CRVUSDT, DOGEUSDT, ETHUSDT, LINKUSDT, +4 more]

● [LAYER UNIVERSE AUDIT]
──────────────────────────────────────────────────────────────────────────────
  LAYER  WINDOW RANGE                    SYMS   ACTIVE (min/med/max)       ENTRY    KILL  WARNINGS
  ─────  ──────────────────────────────  ────   ────────────────────  ──────────  ──────  ────────
  L2     2024-10-01 ~ 2025-09-30           54     4 /  15.0 / 18          86,271      31  —
──────────────────────────────────────────────────────────────────────────────

>> LAYER 2: BLOCKED -> gate_passed=False
!! FAIL: exit_code=1 reason=layer2_blocked:cagr