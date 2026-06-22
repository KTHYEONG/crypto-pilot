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
[TF-PROBE] 17 winning cells across 5 tf: ['12h', '2h', '4h', '6h', '8h']

[TF-PROBE AUDIT] TIMEFRAME SELECTION
------------------------------------------------------------------------------------------------------
| TF       | Winning    | Families                 | Variants                           | Decision   |
------------------------------------------------------------------------------------------------------
| 1h       | 0          | -                        | -                                  | REJECT     |
| 2h       | 1          | btc_regime_pullback:1    | btc_regime_pullback:btc_pullbac... | SELECT     |
| 4h       | 1          | btc_regime_pullback:1    | btc_regime_pullback:btc_pullbac... | SELECT     |
| 6h       | 5          | btc_regime_pullback:3... | btc_regime_pullback:btc_pullbac... | SELECT     |
| 8h       | 4          | btc_regime_pullback:2... | btc_regime_pullback:btc_pullbac... | SELECT     |
| 12h      | 6          | btc_regime_pullback:2... | btc_regime_pullback:btc_pullbac... | SELECT     |
------------------------------------------------------------------------------------------------------

[TF-PROBE AUDIT] GATE SURVIVORSHIP
-------------------------------------------------------------------------------------------------------
| TF       | Cells    | Pass t   | Pass FDR   | Pass Edge  | Pass Fold  | Winning  | Top Fail         |
-------------------------------------------------------------------------------------------------------
| 1h       | 1995     | 27       | 4          | 0          | 0          | 0        | tstat            |
| 2h       | 1995     | 16       | 7          | 1          | 1          | 1        | tstat            |
| 4h       | 1995     | 26       | 5          | 1          | 1          | 1        | tstat            |
| 6h       | 1995     | 17       | 7          | 5          | 5          | 5        | tstat            |
| 8h       | 1995     | 16       | 4          | 4          | 4          | 4        | tstat            |
| 12h      | 1995     | 20       | 6          | 6          | 6          | 6        | tstat            |
-------------------------------------------------------------------------------------------------------

[TF-PROBE AUDIT] BRIDGE INJECTION
------------------------------------------------------------------------------------------------
| TF       | Symbols    | Winning Keys   | Projected    | Source Mix                           |
------------------------------------------------------------------------------------------------
| 12h      | 52         | 3              | 3            | 1h:52                                |
| 6h       | 52         | 3              | 3            | 1h:52                                |
| 8h       | 52         | 3              | 3            | 1h:52                                |
| 2h       | 52         | 1              | 1            | 1h:52                                |
------------------------------------------------------------------------------------------------
[TF-PROBE] Injected 10 extra panels from probe
[PROMO_FILTER] no variants recommended; advisory-only pass-through

● [DATA-INTEGRITY AUDIT]
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ✅ ALL 52 SYMBOLS PASSED
  METRICS : Total Bars: 8,761
  DETAIL  : [NaN: 0.0%] [Zero/Neg: 0.0%] [Range: PASS]
──────────────────────────────────────────────────────────────────────────────

[ALIGN-CUBE] post-join active_mask mean=0.6818 entry_block_mean=0.2559 (was 1.0 / 0.0 before cube injection)
[ENS] Arch-Only   | SYM: 52 | EVT:   4,675 | TOTAL: +11.8 bps | [TRD: -17.9❌ TMO: +31.6✅ MRV: +28.4✅ CRY:  +9.9✅ UNW: +58.0✅ BTN: +33.2✅]
[ENS] Arch-Only   | SYM: 52 | EVT:  49,996 | TOTAL: +21.5 bps | [TRD: +38.9✅ TMO: +24.8✅ MRV: +15.5✅ CRY: -33.2❌ FLO: +16.7✅ UNW:  -8.5❌ BTN: +15.2✅]
[ENS] Arch-Only   | SYM: 52 | EVT:  95,257 | TOTAL: +28.1 bps | [TRD: +44.0✅ TMO: +21.9✅ MRV: +17.1✅ CRY:  +2.6✅ FLO: +18.0✅ UNW:  +5.8✅ BTN: +18.5✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 144,021 | TOTAL: +24.3 bps | [TRD: +40.2✅ TMO: +26.5✅ MRV: +10.1✅ CRY:  +7.3✅ FLO:  +9.6✅ UNW:  +2.6✅ BTN: +15.1✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 188,009 | TOTAL: +35.2 bps | [TRD: +52.6✅ TMO: +30.7✅ MRV: +18.7✅ CRY: +18.5✅ FLO: +24.1✅ UNW: +12.7✅ BTN: +24.8✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 234,511 | TOTAL: +38.3 bps | [TRD: +54.6✅ TMO: +31.7✅ MRV: +22.1✅ CRY: +26.3✅ FLO: +23.8✅ UNW: +18.3✅ BTN: +35.2✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 271,316 | TOTAL: +41.2 bps | [TRD: +57.2✅ TMO: +34.5✅ MRV: +25.5✅ CRY: +25.8✅ FLO: +24.7✅ UNW: +19.4✅ BTN: +33.3✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 318,063 | TOTAL: +46.3 bps | [TRD: +71.0✅ TMO: +39.3✅ MRV: +27.0✅ CRY:  +3.8✅ FLO: +20.3✅ UNW: +11.9✅ BTN: +42.3✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 349,303 | TOTAL: +43.8 bps | [TRD: +65.1✅ TMO: +44.7✅ MRV: +23.6✅ CRY:  +7.2✅ FLO: +16.5✅ UNW: +10.6✅ BTN: +40.2✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 104,522 | TOTAL: +26.4 bps | [TRD: +42.1✅ TMO: +19.5✅ MRV: +15.2✅ CRY:  +2.8✅ FLO: +16.5✅ UNW:  +4.2✅ BTN: +16.0✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 383,258 | TOTAL: +44.4 bps | [TRD: +63.9✅ TMO: +45.1✅ MRV: +24.1✅ CRY: +12.4✅ FLO: +24.1✅ UNW: +12.8✅ BTN: +40.3✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 227,346 | TOTAL: +38.8 bps | [TRD: +56.6✅ TMO: +32.4✅ MRV: +23.2✅ CRY: +19.7✅ FLO: +24.1✅ UNW: +19.0✅ BTN: +30.9✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 329,740 | TOTAL: +44.5 bps | [TRD: +67.4✅ TMO: +40.1✅ MRV: +25.7✅ CRY:  +2.3✅ FLO: +15.0✅ UNW: +11.5✅ BTN: +43.6✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 431,475 | TOTAL: +46.8 bps | [TRD: +68.6✅ TMO: +45.8✅ MRV: +24.1✅ CRY: +15.5✅ FLO: +32.4✅ UNW: +14.9✅ BTN: +39.9✅]

● [LAYER 1 OUTER FOLD READINESS]
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ✅ READY (4/4 Folds Ready)

  [✅] Fold #0 (FitEnd: 2023-09-30 -> OOS: 2023-10-01 ~ 2023-12-31)
       ├─ Symbols : 9 symbols loaded [ANKRUSDT, ARPAUSDT, ARUSDT, BANDUSDT, BCHUSDT, MTLUSDT, NEARUSDT, NEOUSDT, +1 more]
       ├─ Events  : 397 unique events
       └─ Quality : Edge: 48.58 bps

  [✅] Fold #1 (FitEnd: 2023-12-31 -> OOS: 2023-12-31 ~ 2024-03-31)
       ├─ Symbols : 9 symbols loaded [AVAXUSDT, BANDUSDT, DOGEUSDT, DOTUSDT, LTCUSDT, MTLUSDT, NEARUSDT, RUNEUSDT, +1 more]
       ├─ Events  : 2931 unique events
       └─ Quality : Edge: 75.23 bps

  [✅] Fold #2 (FitEnd: 2024-03-31 -> OOS: 2024-04-01 ~ 2024-07-01)
       ├─ Symbols : 18 symbols loaded [ANKRUSDT, ATOMUSDT, BANDUSDT, DOGEUSDT, DOTUSDT, FILUSDT, GALAUSDT, KAVAUSDT, +10 more]
       ├─ Events  : 6929 unique events
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
  ✅ [Symbol-Breadth ] :   22.085 (Target >=3.000 )
  ✅ [Stable-Folds   ] :    1.000 (Target >=0.500 )
  ✅ [Min-Profit     ] :   55.644 (Target >0.000  )
──────────────────────────────────────────────────────────────────────────────

[ENS-FINAL] Arch-Only   | SYM: 52 | EVT: 530,801 | TOTAL: +47.3 bps | [TRD: +66.1✅ TMO: +48.9✅ MRV: +23.0✅ CRY: +27.0✅ FLO: +25.5✅ UNW:+20.0✅ BTN: +38.4✅]

[L1 FINAL PROMOTION SUMMARY] 🚀
  RANK  SYMBOL       STRATEGY (Family)                EDGE(bps)  LCB(bps)    CONV    FOLDS   t(blk)
  ────  ──────────   ───────────────────────────────  ─────────  ────────  ──────  ───────  ───────
  #1    RUNEUSDT     trend_donchian (donchian_72)        +302.6    +207.3    1.00      4/4     3.49
  #2    STORJUSDT    trend_pullback_continuation ...     +293.5    +190.6    1.00      4/4     3.37
  #3    SNXUSDT      trend_pullback_continuation ...     +281.0    +153.4    1.00      4/4     4.19
  #4    NEARUSDT     trend_donchian (donchian_72)        +270.8    +118.5    1.00      4/4     2.41
  #5    ZRXUSDT      trend_pullback_continuation ...     +269.4    +106.5    1.00      3/4     2.61
  #6    ZECUSDT      trend_donchian (donchian_72)        +200.9     +80.8    0.99      3/4     2.56
  #7    1000XECUSDT  dual_momentum (dm_12_48)            +122.9     +80.6    1.00      4/4     3.33
  #8    AAVEUSDT     residual_reversion (rr_16_6h)       +104.1     +52.0    1.00      4/4     3.01
  #9    TRXUSDT      trend_donchian (donchian_72)         +72.4     +39.2    1.00      3/4     3.48
  #10   LTCUSDT      funding_zscore_carry (fzs_48)        +85.4     +37.5    0.99      3/4     2.25
  #11   GALAUSDT     trend_ma (ema_18_108)                +67.2     +35.3    0.99      4/4     2.13
  #12   LTCUSDT      funding_carry (funding_24)           +75.2     +30.7    0.99      4/4     2.59
  #13   TRBUSDT      trend_ma (ema_18_108)               +101.9     +29.5    0.99      2/4     2.68
  #14   LTCUSDT      funding_zscore_carry (fzs_168)       +76.5     +22.3    0.99      4/4     2.49
  #15   LTCUSDT      funding_zscore_carry (fzs_96)        +84.1     +21.1    1.00      4/4     2.55
  #16   ZILUSDT      trend_ma (ema_18_108)                +52.5     +17.9    0.99      2/4     2.02
  #17   ANKRUSDT     trend_ma (ema_18_108)                +36.9     +11.5    0.97      3/4     1.40
  #18   ZRXUSDT      trend_ma (ema_18_108)                +30.3      +9.1    0.95      3/4     0.98
  ────  ──────────   ───────────────────────────────  ─────────  ────────  ──────  ───────  ───────
  [NOT PROMOTED] 1978 pairs | top: no_incremental_edgex1008, quality_weight_zerox954, negative_gross_edgex413


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
  ● [STUDY] l2_study_4h_642115c4fe29 | trials=200 | events=5512 | symbols=12
  ────────────────────────────────────────────────────────────────────────────
[L2-OPT]: 100%|████████████████████████████████████████████████████| 200/200 [01:03<00:00,  3.17it/s, Best CAGR: 344.21% | Current: -112.73%]
[L2-SELECTION] 1 gate-pass 후보 수집 → champion Trial #100 Sortino=3.2170 CAGR=0.3304
[L2-SELECTION] Champion selected. Trial #100, Objective=2.9463, DSR=0.9275 (n_eff=4.59)
  ● [FINAL SIMULATION]
[L2-DEPLOY] L*=1.0000 binding=mdd | CAGR=0.3304 MDD=0.0627 CVaR95=0.0066 RiskUtil=0.209
[L2-DEPLOY] realization gap: risk_util=0.209 expected≈0.700 (결함 #1/#2 재발 의심 — vol-targeting 또는 gross 제약 확인 요망)
● [LAYER 2 PORTFOLIO SCORECARD] (2024-12-22 ~ 2025-09-30)
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ✅ PASS

  ✅ [Growth    ] CAGR: +33.0% (>=30.0%) | PnL: +24.7% | Equity x1.25
  ✅ [Efficiency] Sharpe: 2.028 (>=1.000) | Sortino: 3.217 (>=1.500) | Calmar: 5.274 (>=1.000)
  ✅ [Risk      ] MDD: 6.3% (<=30.0%) | CVaR95: 0.7% (<=6.0%) | RiskUtil: 20.9%
  ✅ [Robust    ] Fold: 100.0% (>=60.0%) | Trades: 89 (>=30) | Friction: 100.0%
  ❌ [Uplift    ] Sharpe Uplift: +0.16 (>=+0.20)
  ✅ [Integrity ] DSR: 0.928 (>=0.60) | PSR: 0.966 (diag)
  [Diag     ] RelMDD: 0.99x | Turnover: 0.057
──────────────────────────────────────────────────────────────────────────────

  [ FOLD DETAIL BREAKDOWN ]
  ──────────────────────────────────────────────────────────────────────────
  ├─ Fold #1 : ✅ Sharpe:  2.666 | CAGR:   +66.4% | MDD:   6.3% | Status: PASS | Period: 2024-12-22 ~ 2025-03-26
       Symbols: 6 [AAVEUSDT, GALAUSDT, LTCUSDT, NEARUSDT, TRBUSDT, TRXUSDT]
  ├─ Fold #2 : ✅ Sharpe:  2.719 | CAGR:   +40.6% | MDD:   3.2% | Status: PASS | Period: 2025-03-26 ~ 2025-06-28
       Symbols: 6 [AAVEUSDT, GALAUSDT, LTCUSDT, TRBUSDT, TRXUSDT, ZECUSDT]
  └─ Fold #3 : ✅ Sharpe:  0.138 | CAGR:    +0.8% | MDD:   5.2% | Status: PASS | Period: 2025-06-28 ~ 2025-09-30
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