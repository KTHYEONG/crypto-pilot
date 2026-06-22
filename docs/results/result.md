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
tf=12h: 2337 cells computed
tf=8h: 2337 cells computed
tf=6h: 2337 cells computed
tf=4h: 2337 cells computed
tf=2h: 2337 cells computed
tf=1h: 2337 cells computed
[TF-PROBE] 13 winning cells across 5 tf: ['12h', '2h', '4h', '6h', '8h']

[TF-PROBE AUDIT] TIMEFRAME SELECTION
------------------------------------------------------------------------------------------------------
| TF       | Winning    | Families                 | Variants                           | Decision   |
------------------------------------------------------------------------------------------------------
| 1h       | 0          | -                        | -                                  | REJECT     |
| 2h       | 1          | btc_regime_pullback:1    | btc_regime_pullback:btc_pullbac... | SELECT     |
| 4h       | 1          | btc_regime_pullback:1    | btc_regime_pullback:btc_pullbac... | SELECT     |
| 6h       | 3          | btc_regime_pullback:2... | btc_regime_pullback:btc_pullbac... | SELECT     |
| 8h       | 4          | btc_regime_pullback:2... | btc_regime_pullback:btc_pullbac... | SELECT     |
| 12h      | 4          | btc_regime_pullback:2... | btc_regime_pullback:btc_pullbac... | SELECT     |
------------------------------------------------------------------------------------------------------

[TF-PROBE AUDIT] GATE SURVIVORSHIP
-------------------------------------------------------------------------------------------------------
| TF       | Cells    | Pass t   | Pass FDR   | Pass Edge  | Pass Fold  | Winning  | Top Fail         |
-------------------------------------------------------------------------------------------------------
| 1h       | 2337     | 30       | 5          | 0          | 0          | 0        | tstat            |
| 2h       | 2337     | 20       | 7          | 1          | 1          | 1        | tstat            |
| 4h       | 2337     | 28       | 3          | 1          | 1          | 1        | tstat            |
| 6h       | 2337     | 17       | 4          | 3          | 3          | 3        | tstat            |
| 8h       | 2337     | 16       | 4          | 4          | 4          | 4        | tstat            |
| 12h      | 2337     | 20       | 4          | 4          | 4          | 4        | tstat            |
-------------------------------------------------------------------------------------------------------

● [DATA-INTEGRITY AUDIT]
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ✅ ALL 52 SYMBOLS PASSED
  METRICS : Total Bars: 8,761
  DETAIL  : [NaN: 0.0%] [Zero/Neg: 0.0%] [Range: PASS]
──────────────────────────────────────────────────────────────────────────────

[ALIGN-CUBE] post-join active_mask mean=0.6818 entry_block_mean=0.2559 (was 1.0 / 0.0 before cube injection)
[ENS] Arch-Only   | SYM: 52 | EVT:   3,109 | TOTAL:  +7.6 bps | [TRD: -19.0❌ TMO: +20.2✅ MRV: +42.9✅ CRY:  +9.1✅ BTN: +35.3✅]
[ENS] Arch-Only   | SYM: 52 | EVT:  36,032 | TOTAL: +24.0 bps | [TRD: +38.6✅ TMO: +27.9✅ MRV: +15.7✅ CRY: -36.4❌ FLO: +17.0✅ UNW:  -4.2❌ BTN: +27.3✅]
[ENS] Arch-Only   | SYM: 52 | EVT:  69,829 | TOTAL: +33.2 bps | [TRD: +44.5✅ TMO: +26.4✅ MRV: +19.7✅ CRY:  +3.6✅ FLO: +22.2✅ UNW: +10.4✅ BTN: +21.9✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 103,483 | TOTAL: +30.0 bps | [TRD: +40.7✅ TMO: +30.1✅ MRV: +10.7✅ CRY:  +8.4✅ FLO: +13.9✅ UNW:  -1.0❌ BTN: +19.3✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 136,288 | TOTAL: +42.0 bps | [TRD: +54.6✅ TMO: +33.5✅ MRV: +16.7✅ CRY: +19.8✅ FLO: +28.5✅ UNW:  +1.9✅ BTN: +32.3✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 170,679 | TOTAL: +46.0 bps | [TRD: +57.1✅ TMO: +35.3✅ MRV: +24.7✅ CRY: +27.8✅ FLO: +28.6✅ UNW: +28.1✅ BTN: +42.9✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 198,641 | TOTAL: +48.9 bps | [TRD: +60.0✅ TMO: +38.5✅ MRV: +27.6✅ CRY: +27.3✅ FLO: +29.3✅ UNW: +29.6✅ BTN: +42.0✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 236,913 | TOTAL: +54.8 bps | [TRD: +73.7✅ TMO: +42.7✅ MRV: +29.0✅ CRY:  +5.5✅ FLO: +25.1✅ UNW: +12.3✅ BTN: +49.9✅]
[ENS] Arch-Only   | SYM: 52 | EVT:  76,260 | TOTAL: +31.6 bps | [TRD: +42.6✅ TMO: +24.0✅ MRV: +18.3✅ CRY:  +3.8✅ FLO: +20.8✅ UNW:  +7.2✅ BTN: +19.2✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 264,661 | TOTAL: +51.3 bps | [TRD: +67.0✅ TMO: +48.3✅ MRV: +25.9✅ CRY:  +8.7✅ FLO: +20.5✅ UNW:  +7.9✅ BTN: +43.4✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 291,400 | TOTAL: +52.2 bps | [TRD: +66.3✅ TMO: +48.1✅ MRV: +28.3✅ CRY: +14.0✅ FLO: +28.0✅ UNW:  +6.5✅ BTN: +44.5✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 165,509 | TOTAL: +45.7 bps | [TRD: +58.6✅ TMO: +35.6✅ MRV: +23.8✅ CRY: +21.1✅ FLO: +28.4✅ UNW: +21.8✅ BTN: +37.3✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 247,644 | TOTAL: +52.1 bps | [TRD: +70.1✅ TMO: +43.5✅ MRV: +26.4✅ CRY:  +3.9✅ FLO: +19.3✅ UNW: +11.2✅ BTN: +49.1✅]
[ENS] Arch-Only   | SYM: 52 | EVT: 324,529 | TOTAL: +55.9 bps | [TRD: +71.4✅ TMO: +49.5✅ MRV: +29.1✅ CRY: +17.3✅ FLO: +36.7✅ UNW:  +9.9✅ BTN: +43.9✅]

● [LAYER 1 OUTER FOLD READINESS]
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ✅ READY (4/4 Folds Ready)

  [✅] Fold #0 (FitEnd: 2023-09-30 -> OOS: 2023-10-01 ~ 2023-12-31)
       ├─ Symbols : 9 symbols loaded [ARPAUSDT, ARUSDT, BANDUSDT, BCHUSDT, KAVAUSDT, MTLUSDT, NEOUSDT, RUNEUSDT, +1 more]
       ├─ Events  : 359 unique events
       └─ Quality : Edge: 84.51 bps

  [✅] Fold #1 (FitEnd: 2023-12-31 -> OOS: 2023-12-31 ~ 2024-03-31)
       ├─ Symbols : 18 symbols loaded [1000XECUSDT, ADAUSDT, ANKRUSDT, ARUSDT, AVAXUSDT, BANDUSDT, DOGEUSDT, DOTUSDT, +10 more]
       ├─ Events  : 3575 unique events
       └─ Quality : Edge: 84.26 bps

  [✅] Fold #2 (FitEnd: 2024-03-31 -> OOS: 2024-04-01 ~ 2024-07-01)
       ├─ Symbols : 23 symbols loaded [ANKRUSDT, ARUSDT, ATOMUSDT, BANDUSDT, CRVUSDT, DOGEUSDT, DOTUSDT, FILUSDT, +15 more]
       ├─ Events  : 7531 unique events
       └─ Quality : Edge: 147.76 bps

  [✅] Fold #3 (FitEnd: 2024-07-01 -> OOS: 2024-07-01 ~ 2024-09-30)
       ├─ Symbols : 22 symbols loaded [ADAUSDT, ANKRUSDT, ARUSDT, ATOMUSDT, BNBUSDT, DOGEUSDT, DOTUSDT, FILUSDT, +14 more]
       ├─ Events  : 7298 unique events
       └─ Quality : Edge: 168.72 bps

──────────────────────────────────────────────────────────────────────────────

● [LAYER 1 HARD GATE CHECKS]
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ✅ PASSED (5/5 Passed)

  ✅ [Time-Coverage  ] :    1.000 (Target >=0.800 )
  ✅ [Signal-Quality ] :    1.000 (Target >=0.900 )
  ✅ [Symbol-Breadth ] :   27.000 (Target >=3.000 )
  ✅ [Stable-Folds   ] :    1.000 (Target >=0.500 )
  ✅ [Min-Profit     ] :   56.443 (Target >0.000  )
──────────────────────────────────────────────────────────────────────────────

[ENS-FINAL] Arch-Only   | SYM: 52 | EVT: 402,343 | TOTAL: +55.4 bps | [TRD: +67.6✅ TMO: +52.8✅ MRV: +27.0✅ CRY: +28.6✅ FLO: +29.0✅ UNW: +10.1✅ BTN: +41.1✅]

[L1 FINAL PROMOTION SUMMARY] 🚀
  RANK  SYMBOL       STRATEGY (Family)                EDGE(bps)  LCB(bps)    CONV    FOLDS   t(blk)
  ────  ──────────   ───────────────────────────────  ─────────  ────────  ──────  ───────  ───────
  #1    GALAUSDT     residual_reversion (rr_24)          +394.7    +345.6    1.00      4/4    14.38
  #2    MTLUSDT      residual_reversion (rr_24)          +319.7    +277.3    1.00      3/3     9.57
  #3    RUNEUSDT     trend_donchian (donchian_72)        +303.4    +207.3    1.00      4/4     3.50
  #4    STORJUSDT    trend_pullback_continuation ...     +284.6    +190.6    1.00      4/4     3.27
  #5    SNXUSDT      trend_pullback_continuation ...     +277.6    +153.4    1.00      4/4     4.14
  #6    TRBUSDT      trend_pullback_continuation ...     +326.4    +146.6    1.00      4/4     2.69
  #7    NEARUSDT     trend_donchian (donchian_72)        +266.6    +118.5    1.00      4/4     2.37
  #8    SANDUSDT     trend_pullback_continuation ...     +258.1    +117.2    1.00      4/4     2.18
  #9    ARUSDT       trend_donchian (donchian_72)        +219.6    +106.6    1.00      4/4     2.56
  #10   ZRXUSDT      trend_pullback_continuation ...     +270.5    +106.5    1.00      3/4     2.62
  #11   IOTAUSDT     trend_pullback_continuation ...     +171.2    +105.3    0.99      3/3     2.43
  #12   VETUSDT      trend_donchian (donchian_72)        +177.0     +96.5    1.00      4/4     3.10
  #13   KAVAUSDT     trend_pullback_continuation ...     +202.2     +95.6    1.00      4/4     3.09
  #14   NEOUSDT      residual_reversion (rr_24)          +143.3     +84.8    1.00      4/4     4.27
  #15   DYDXUSDT     residual_reversion (rr_24)          +124.0     +84.4    1.00      4/4     4.35
  #16   ZECUSDT      trend_donchian (donchian_72)        +202.2     +80.8    0.99      3/4     2.58
  #17   THETAUSDT    trend_donchian (donchian_72)        +182.2     +78.0    1.00      4/4     2.37
  #18   1000XECUSDT  dual_momentum (dm_12_48)            +108.7     +69.2    1.00      4/4     2.94
  #19   API3USDT     residual_reversion (rr_24)          +186.1     +69.0    1.00      3/3     4.00
  #20   1000SHIBUSDT residual_reversion (rr_24)           +88.3     +62.3    1.00      4/4     3.81
  #21   RSRUSDT      trend_donchian (donchian_72)        +201.6     +56.5    0.99      4/4     2.33
  #22   MANAUSDT     residual_reversion (rr_24)           +82.8     +54.7    1.00      4/4     4.91
  #23   FILUSDT      residual_reversion (rr_24)           +79.1     +52.8    1.00      4/4     3.65
  #24   TRXUSDT      trend_donchian (donchian_72)         +71.0     +39.2    1.00      3/4     3.41
  #25   LTCUSDT      funding_zscore_carry (fzs_48)        +81.4     +37.3    0.99      3/4     2.15
  #26   SOLUSDT      trend_donchian (donchian_72)        +167.4     +36.8    0.99      3/4     2.38
  #27   GALAUSDT     trend_ma (ema_18_108)                +66.1     +34.5    0.99      4/4     2.10
  #28   RUNEUSDT     rsi_reversion (rsi_6)                +55.5     +32.7    1.00      4/4     2.25
  #29   LTCUSDT      funding_carry (funding_24)           +71.9     +30.6    0.99      4/4     2.48
  #30   ETHUSDT      dual_momentum (dm_12_48)             +86.5     +29.3    1.00      4/4     3.35
  #31   NEOUSDT      funding_zscore_carry (fzs_48)        +92.2     +26.1    0.99      3/4     2.04
  #32   TRBUSDT      trend_ma (ema_18_108)               +101.2     +26.0    0.99      2/4     2.67
  #33   LTCUSDT      funding_zscore_carry (fzs_168)       +75.8     +22.8    0.99      4/4     2.47
  #34   LTCUSDT      funding_zscore_carry (fzs_96)        +81.4     +21.6    1.00      4/4     2.47
  #35   ANKRUSDT     trend_ma (ema_18_108)                +41.0     +15.7    0.98      3/4     1.56
  #36   ZILUSDT      trend_ma (ema_18_108)                +49.3     +15.1    0.99      2/4     1.89
  #37   ZRXUSDT      trend_ma (ema_18_108)                +32.8     +14.4    0.96      3/4     1.06
  #38   ARUSDT       trend_ma (ema_18_108)                +41.1     +13.7    0.97      3/4     1.46
  #39   THETAUSDT    trend_ma (ema_18_108)                +31.6     +10.7    0.98      2/4     1.25
  #40   NEARUSDT     trend_ma (ema_18_108)                +35.4      +8.3    0.97      3/4     1.23
  ────  ──────────   ───────────────────────────────  ─────────  ────────  ──────  ───────  ───────
  [NOT PROMOTED] 1104 pairs | top: no_incremental_edgex562, quality_weight_zerox527, negative_gross_edgex205


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
  ● [STUDY] l2_study_4h_73f4fdd5b0a4 | trials=200 | events=12936 | symbols=27
  ────────────────────────────────────────────────────────────────────────────
[L2-OPT]: 100%|██████████████████████████████████████████████████████| 200/200 [01:32<00:00,  2.16it/s, Best CAGR: 248.12% | Current: 69.46%]
[L2-SELECTION] 2 gate-pass 후보 수집 → champion Trial #174 Sortino=2.3034 CAGR=0.5091
[L2-SELECTION] Champion selected. Trial #174, Objective=2.4812, DSR=0.7465 (n_eff=6.55)
  ● [FINAL SIMULATION]
[L2-DEPLOY] L*=1.0000 binding=mdd | CAGR=0.0183 MDD=0.1660 CVaR95=0.0146 RiskUtil=0.553
● [LAYER 2 PORTFOLIO SCORECARD] (2024-12-22 ~ 2025-09-30)
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ❌ BLOCKED (cagr)

  ❌ [Growth    ] CAGR: +1.8% (>=30.0%) | PnL: +4.3% | Equity x1.04
  ❌ [Efficiency] Sharpe: 0.193 (>=1.000) | Sortino: 0.266 (>=1.500) | Calmar: 0.111 (>=1.000)
  ✅ [Risk      ] MDD: 16.6% (<=30.0%) | CVaR95: 1.5% (<=6.0%) | RiskUtil: 55.3%
  ✅ [Robust    ] Fold: 66.7% (>=60.0%) | Trades: 308 (>=30) | Friction: 100.0%
  ❌ [Uplift    ] Sharpe Uplift: -0.07 (>=+0.20)
  ✅ [Integrity ] DSR: 0.746 (>=0.60) | PSR: 0.615 (diag)
  [Diag     ] RelMDD: 1.29x | Turnover: 0.375
──────────────────────────────────────────────────────────────────────────────

  [ FOLD DETAIL BREAKDOWN ]
  ──────────────────────────────────────────────────────────────────────────
  ├─ Fold #1 : ✅ Sharpe:  1.191 | CAGR:   +12.0% | MDD:  13.5% | Status: PASS | Period: 2024-12-22 ~ 2025-03-26
       Symbols: 18 [1000SHIBUSDT, ARUSDT, DYDXUSDT, ETHUSDT, FILUSDT, GALAUSDT, LTCUSDT, MANAUSDT, +10 more]
  ├─ Fold #2 : ✅ Sharpe:  0.249 | CAGR:    +1.0% | MDD:   9.1% | Status: PASS | Period: 2025-03-26 ~ 2025-06-28
       Symbols: 17 [1000SHIBUSDT, ARUSDT, DYDXUSDT, ETHUSDT, FILUSDT, GALAUSDT, IOTAUSDT, LTCUSDT, +9 more]
  └─ Fold #3 : ❌ Sharpe: -0.770 | CAGR:    -6.7% | MDD:  10.1% | Status: FAIL | Period: 2025-06-28 ~ 2025-09-30
       Symbols: 20 [1000SHIBUSDT, API3USDT, ARUSDT, DYDXUSDT, ETHUSDT, FILUSDT, GALAUSDT, KAVAUSDT, +12 more]

● [LAYER UNIVERSE AUDIT]
──────────────────────────────────────────────────────────────────────────────
  LAYER  WINDOW RANGE                    SYMS   ACTIVE (min/med/max)       ENTRY    KILL  WARNINGS
  ─────  ──────────────────────────────  ────   ────────────────────  ──────────  ──────  ────────
  L2     2024-10-01 ~ 2025-09-30           52    35 /  38.0 / 43             552      19  entry_block_spike
──────────────────────────────────────────────────────────────────────────────

>> LAYER 2: BLOCKED -> gate_passed=False
!! FAIL: exit_code=1 reason=layer2_blocked:cagr