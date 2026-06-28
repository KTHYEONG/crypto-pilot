================================================================================
LOCAL DATA STORAGE (LEDGER & CACHE STATUS)
================================================================================

  Sync Mode: FULL (Pre-loaded from cache)
  [SKIPPED] All records in 'universe_ledger.db' are up-to-date. (No sync required)

--------------------------------------------------------------------------------
🔍 [TF-PROBE AUDIT] SOURCE READINESS Dashboard
  ├── 4h   : Ready 334/334 | Median Bars: 5189   | Mix: 4h:334
  ├── 6h   : Ready 334/334 | Median Bars: 20790  | Mix: 1h:334
  ├── 8h   : Ready 334/334 | Median Bars: 20790  | Mix: 1h:334
  ├── 12h  : Ready 334/334 | Median Bars: 20790  | Mix: 1h:334
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
  └─ Active : [12h] Proj=14 Syms=52 | [8h] Proj=13 Syms=52 | [6h] Proj=13 Syms=52

● [DATA-INTEGRITY AUDIT]
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ✅ ALL 52 SYMBOLS PASSED
  METRICS : Total Bars: 8,761
  DETAIL  : [NaN: 0.0%] [Zero/Neg: 0.0%] [Range: PASS]
──────────────────────────────────────────────────────────────────────────────

📈 [ENS-FINAL] Arch-Only | SYM:52 | EVT:681,037 | TOTAL:+62.7 bps | [TRD:+72.3✅ TMO:+52.0✅ MRV:+29.1✅ CRY:+30.1✅ FLO:+32.2✅ UNW:+23.1✅ BTN:+42.6✅]

● [LAYER 1 OUTER FOLD READINESS]
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ✅ READY (4/4 Folds Ready)

  [✅] Fold #0 (FitEnd: 2023-09-30 -> OOS: 2023-10-01 ~ 2023-12-31)
       ├─ Symbols : 11 symbols loaded [ANKRUSDT, ARPAUSDT, AXSUSDT, BCHUSDT, DOTUSDT, ENSUSDT, FILUSDT, NEOUSDT, +3 more]
       ├─ Events  : 1380 unique events
       └─ Quality : Edge: 45.47 bps

  [✅] Fold #1 (FitEnd: 2023-12-31 -> OOS: 2023-12-31 ~ 2024-03-31)
       ├─ Symbols : 9 symbols loaded [ARUSDT, ATOMUSDT, AVAXUSDT, BTCUSDT, DOTUSDT, NEARUSDT, SANDUSDT, THETAUSDT, +1 more]
       ├─ Events  : 3772 unique events
       └─ Quality : Edge: 13.31 bps

  [✅] Fold #2 (FitEnd: 2024-03-31 -> OOS: 2024-04-01 ~ 2024-07-01)
       ├─ Symbols : 17 symbols loaded [1000XECUSDT, ANKRUSDT, API3USDT, ARUSDT, ATOMUSDT, BANDUSDT, DOGEUSDT, DOTUSDT, +9 more]
       ├─ Events  : 6831 unique events
       └─ Quality : Edge: 56.45 bps

  [✅] Fold #3 (FitEnd: 2024-07-01 -> OOS: 2024-07-01 ~ 2024-09-30)
       ├─ Symbols : 18 symbols loaded [1000XECUSDT, ADAUSDT, ANKRUSDT, ARUSDT, BCHUSDT, BNBUSDT, DOGEUSDT, DOTUSDT, +10 more]
       ├─ Events  : 7817 unique events
       └─ Quality : Edge: 183.20 bps

──────────────────────────────────────────────────────────────────────────────
🏁 STATUS : ✅ PASSED (5/5 Passed)
  👉 Cov:1.000(>=0.80) | Qual:1.000(>=0.90) | Brd:26.304(>=3.00) | Fld:1.000(>=0.50) | Prf:27.496(>0.00)

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

📈 [ENS-FINAL] Arch-Only | SYM:52 | EVT:456,109 | TOTAL:+60.9 bps | [TRD:+67.6✅ TMO:+57.5✅ MRV:+28.7✅ CRY:+22.1✅]

● [LAYER 1 OUTER FOLD READINESS]
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ✅ READY (3/4 Folds Ready)

  [✅] Fold #0 (FitEnd: 2023-09-30 -> OOS: 2023-10-01 ~ 2023-12-31)
       ├─ Symbols : 10 symbols loaded [1000XECUSDT, ADAUSDT, ARUSDT, ATOMUSDT, AVAXUSDT, BANDUSDT, ENSUSDT, PEOPLEUSDT, +2 more]
       ├─ Events  : 981 unique events
       └─ Quality : Edge: 138.41 bps

  [❌] Fold #1 (FitEnd: 2023-12-31 -> OOS: 2023-12-31 ~ 2024-03-31)
       ├─ Symbols : 9 symbols loaded [AAVEUSDT, ARUSDT, BTCUSDT, DOTUSDT, NEARUSDT, RSRUSDT, RVNUSDT, SANDUSDT, +1 more]
       ├─ Events  : 2522 unique events
       └─ Quality : Edge: -30.17 bps
       └─ BLOCKERS: non_positive_gross_edge

  [✅] Fold #2 (FitEnd: 2024-03-31 -> OOS: 2024-04-01 ~ 2024-07-01)
       ├─ Symbols : 16 symbols loaded [1000SHIBUSDT, AAVEUSDT, ARUSDT, BANDUSDT, BNBUSDT, DOGEUSDT, DOTUSDT, ETCUSDT, +8 more]
       ├─ Events  : 6268 unique events
       └─ Quality : Edge: 49.97 bps

  [✅] Fold #3 (FitEnd: 2024-07-01 -> OOS: 2024-07-01 ~ 2024-09-30)
       ├─ Symbols : 16 symbols loaded [AAVEUSDT, ADAUSDT, ANKRUSDT, BNBUSDT, DOTUSDT, ETCUSDT, FILUSDT, GALAUSDT, +8 more]
       ├─ Events  : 6457 unique events
       └─ Quality : Edge: 163.17 bps

──────────────────────────────────────────────────────────────────────────────
🏁 STATUS : ✅ PASSED (5/5 Passed)
  👉 Cov:1.000(>=0.80) | Qual:1.000(>=0.90) | Brd:25.252(>=3.00) | Fld:0.750(>=0.50) | Prf:42.086(>0.00)

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

📈 [ENS-FINAL] Arch-Only | SYM:52 | EVT:452,492 | TOTAL:+59.5 bps | [TRD:+65.1✅ TMO:+65.5✅ MRV:+19.0✅ CRY:+24.8✅]

● [LAYER 1 OUTER FOLD READINESS]
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ✅ READY (4/4 Folds Ready)

  [✅] Fold #0 (FitEnd: 2023-09-30 -> OOS: 2023-10-01 ~ 2023-12-31)
       ├─ Symbols : 8 symbols loaded [ARUSDT, ATOMUSDT, AVAXUSDT, ENSUSDT, ETHUSDT, NEARUSDT, RUNEUSDT, SOLUSDT]
       ├─ Events  : 531 unique events
       └─ Quality : Edge: 27.03 bps

  [✅] Fold #1 (FitEnd: 2023-12-31 -> OOS: 2023-12-31 ~ 2024-03-31)
       ├─ Symbols : 9 symbols loaded [AAVEUSDT, ARUSDT, AVAXUSDT, BTCUSDT, NEARUSDT, RSRUSDT, RVNUSDT, SANDUSDT, +1 more]
       ├─ Events  : 2459 unique events
       └─ Quality : Edge: 79.96 bps

  [✅] Fold #2 (FitEnd: 2024-03-31 -> OOS: 2024-04-01 ~ 2024-07-01)
       ├─ Symbols : 12 symbols loaded [BANDUSDT, DOGEUSDT, DOTUSDT, ETCUSDT, GALAUSDT, KAVAUSDT, NEARUSDT, SANDUSDT, +4 more]
       ├─ Events  : 5324 unique events
       └─ Quality : Edge: 100.18 bps

  [✅] Fold #3 (FitEnd: 2024-07-01 -> OOS: 2024-07-01 ~ 2024-09-30)
       ├─ Symbols : 13 symbols loaded [AAVEUSDT, ADAUSDT, ANKRUSDT, DOGEUSDT, DOTUSDT, GALAUSDT, KAVAUSDT, NEARUSDT, +5 more]
       ├─ Events  : 4515 unique events
       └─ Quality : Edge: 191.43 bps

──────────────────────────────────────────────────────────────────────────────
🏁 STATUS : ✅ PASSED (5/5 Passed)
  👉 Cov:1.000(>=0.80) | Qual:1.000(>=0.90) | Brd:21.000(>=3.00) | Fld:1.000(>=0.50) | Prf:54.308(>0.00)

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

📈 [ENS-FINAL] Arch-Only | SYM:52 | EVT:539,813 | TOTAL:+66.6 bps | [TRD:+72.2✅ TMO:+67.9✅ MRV:+33.7✅ CRY:+27.1✅]

● [LAYER 1 OUTER FOLD READINESS]
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ✅ READY (4/4 Folds Ready)

  [✅] Fold #0 (FitEnd: 2023-09-30 -> OOS: 2023-10-01 ~ 2023-12-31)
       ├─ Symbols : 13 symbols loaded [AAVEUSDT, ANKRUSDT, AVAXUSDT, BNBUSDT, ENSUSDT, GALAUSDT, NEARUSDT, RVNUSDT, +5 more]
       ├─ Events  : 1576 unique events
       └─ Quality : Edge: 77.98 bps

  [✅] Fold #1 (FitEnd: 2023-12-31 -> OOS: 2023-12-31 ~ 2024-03-31)
       ├─ Symbols : 19 symbols loaded [1000XECUSDT, AAVEUSDT, ADAUSDT, API3USDT, ARUSDT, AVAXUSDT, BTCUSDT, KAVAUSDT, +11 more]
       ├─ Events  : 2963 unique events
       └─ Quality : Edge: 11.33 bps

  [✅] Fold #2 (FitEnd: 2024-03-31 -> OOS: 2024-04-01 ~ 2024-07-01)
       ├─ Symbols : 17 symbols loaded [1000XECUSDT, API3USDT, ARUSDT, BNBUSDT, DOTUSDT, ENSUSDT, ETCUSDT, GALAUSDT, +9 more]
       ├─ Events  : 6159 unique events
       └─ Quality : Edge: 90.00 bps

  [✅] Fold #3 (FitEnd: 2024-07-01 -> OOS: 2024-07-01 ~ 2024-09-30)
       ├─ Symbols : 20 symbols loaded [1000SHIBUSDT, 1000XECUSDT, ANKRUSDT, BCHUSDT, CRVUSDT, DOTUSDT, ETCUSDT, ETHUSDT, +12 more]
       ├─ Events  : 6524 unique events
       └─ Quality : Edge: 157.24 bps

──────────────────────────────────────────────────────────────────────────────
🏁 STATUS : ✅ PASSED (5/5 Passed)
  👉 Cov:1.000(>=0.80) | Qual:1.000(>=0.90) | Brd:31.530(>=3.00) | Fld:1.000(>=0.55) | Prf:30.337(>0.00)

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


● [LAYER UNIVERSE AUDIT]
──────────────────────────────────────────────────────────────────────────────
  LAYER  WINDOW RANGE                    SYMS   ACTIVE (min/med/max)       ENTRY    KILL  WARNINGS
  ─────  ──────────────────────────────  ────   ────────────────────  ──────────  ──────  ────────
  L1     2023-04-01 ~ 2024-09-30           52     0 /  49.0 / 52               0       6  —
──────────────────────────────────────────────────────────────────────────────


>> LAYER 1: PASS -> Proceeding to Layer 2.
[REGIME]
metric        | value
compression   | on
states        | 3
status        | 🟢 stable
distribution  | bull=34.7% bear=28.1% crisis=37.3%
policy_mode   | soft
hard_block    | off
risk_cap      | on
policy_source | fit/cal
oos_debug     | evaluation only
note          | L2 verdict is reported separately in [REGIME-L2]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
● [LAYER 2: OPTUNA TUNING]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[REGIME-L2] active_states=3 compression=True path=regime_conditioned proof=True lift=54.42 t=15.24 fold_pass=1.00
[REGIME-L2] policy_mode=soft policy_source=fit/cal global_reliable=True allow=5 downweight=10 block=0 pooled=243 unstable=15 hard_block_eligible=0 sign_consistency=0.50 hard_block_enabled=False mean_cal_lift=-23.04 mean_conf=1.00
  ● [STUDY] l2_study_4h_95de9c21278a | trials=200 | events=67063 | symbols=42
  ────────────────────────────────────────────────────────────────────────────
 [OPT] Deleted existing study 'l2_study_4h_95de9c21278a' for a fresh start.
[L2-OPT]:   0%|                                                                                                   | 0/200 [00:00<?, ?it/s][L2-OPT] ProcessPool workers=5 (mem=4.1GB, mem_safe=5, cpu=8, batch=6)
[L2-OPT]: 100%|█████████████████████████████████████████████████| 200/200 [02:54<00:00,  1.15it/s, Best CAGR: 144.81% | Current: -181.53%]
[L2-SELECTION] No gate-passed trials found. Reducing diagnostic replay size to 3.
[L2-SELECTION] No feasible candidate found within fallback window (reason=cagr)
[L2-DEPLOY-C4] L*=1.431 (binding=mdd) | realized_mode=return_scaling | kelly=0.250(불변) | tf=4h
  ● [FINAL SIMULATION]
[L2-DEPLOY] L*=1.4315 binding=champion | CAGR=0.0375 MDD=0.2053 CVaR95=0.0174 RiskUtil=0.684
● [LAYER 2 PORTFOLIO SCORECARD] (2024-12-22 ~ 2025-09-30)
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ❌ BLOCKED (cagr)

  ❌ [Growth    ] CAGR: +3.8% (>=30.0%) | PnL: +5.0% | Equity x1.05
  ❌ [Efficiency] Sharpe: 0.275 (>=1.000) | Sortino: 0.382 (>=1.500) | Calmar: 0.183 (>=1.000)
  ✅ [Risk      ] MDD: 20.5% (<=30.0%) | CVaR95: 1.7% (<=6.0%) | RiskUtil: 68.4%
  ✅ [Robust    ] Fold: 66.7% (>=60.0%) | Trades: 177 (>=30) | Friction: 95.3%
  ❌ [Uplift    ] Sharpe Uplift: +0.20 (>=+0.20)
  ✅ [Integrity ] DSR: 0.692 (>=0.60) | PSR: 0.634 (diag)
  [Diag     ] RelMDD: 1.40x | Turnover: 0.065
──────────────────────────────────────────────────────────────────────────────

  [ FOLD DETAIL BREAKDOWN ]
  ──────────────────────────────────────────────────────────────────────────
  ├─ Fold #1 : ❌ Sharpe: -0.442 | CAGR:   -13.3% | MDD:  20.5% | Status: FAIL | Period: 2024-12-22 ~ 2025-03-26
       Symbols: 20 [1000SHIBUSDT, AAVEUSDT, ARPAUSDT, AXSUSDT, BCHUSDT, BNBUSDT, BTCUSDT, ETCUSDT, +12 more]
  ├─ Fold #2 : ✅ Sharpe:  1.857 | CAGR:   +24.8% | MDD:   6.9% | Status: PASS | Period: 2025-03-26 ~ 2025-06-28
       Symbols: 14 [AXSUSDT, BCHUSDT, BNBUSDT, ETCUSDT, ETHUSDT, LINKUSDT, LTCUSDT, NEARUSDT, +6 more]
  └─ Fold #3 : ✅ Sharpe:  0.379 | CAGR:    +3.3% | MDD:  11.8% | Status: PASS | Period: 2025-06-28 ~ 2025-09-30
       Symbols: 15 [1000SHIBUSDT, AAVEUSDT, BCHUSDT, BNBUSDT, BTCUSDT, DOTUSDT, ETCUSDT, KAVAUSDT, +7 more]

● [LAYER UNIVERSE AUDIT]
──────────────────────────────────────────────────────────────────────────────
  LAYER  WINDOW RANGE                    SYMS   ACTIVE (min/med/max)       ENTRY    KILL  WARNINGS
  ─────  ──────────────────────────────  ────   ────────────────────  ──────────  ──────  ────────
  L2     2024-10-01 ~ 2025-09-30           52    35 /  38.0 / 43             552      19  entry_block_spike
──────────────────────────────────────────────────────────────────────────────

>> LAYER 2: BLOCKED -> gate_passed=False
!! FAIL: exit_code=1 reason=layer2_blocked:cagr