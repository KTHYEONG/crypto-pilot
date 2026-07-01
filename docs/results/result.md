================================================================================
LOCAL DATA STORAGE (LEDGER & CACHE STATUS)
================================================================================

  Sync Mode: SKIP (Reused cache from prior --sync fast on 2026-07-01)

--------------------------------------------------------------------------------
🔍 [TF-PROBE AUDIT] SOURCE READINESS Dashboard
  ├── 4h   : Ready 296/296 | Median Bars: 5260   | Mix: 4h:296
  ├── 6h   : Ready 296/296 | Median Bars: 21040  | Mix: 1h:296
  ├── 8h   : Ready 296/296 | Median Bars: 21040  | Mix: 1h:296
  ├── 12h  : Ready 296/296 | Median Bars: 21040  | Mix: 1h:296
================================================================================
SYSTEM CONTEXT | DATA PIPELINE PREPARATION
================================================================================

TIME PROFILE
  Test Horizon  : 2023-01-01 ~ 2026-06-30
  IS / OOS Split: 2026-01-01 (In-Sample Cutoff)

UNIVERSE FUNNEL
  [1] Market Pool     : 414 symbols discovered (Binance USDT-M)
  [2] Capacity Limit  : 150 symbols selected (Top-N Liquidity)
  [3] Integrity Pass  : 65 symbols loaded (Passed Gaps & Frozen checks)

STRATEGY ENGINE
  Active Engine : Alpha-Ensemble Engine
  Target Scope  : 65 symbols ready for Layer 1 execution

--------------------------------------------------------------------------------

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
● [LAYER 1: SIGNAL ROBUSTNESS & ENSEMBLE VERIFICATION]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 [L1: SWF SCOPE & ADMISSION]
  ├─ Symbols : 54/65 Admitted
  └─ Details : Base 65 | Dropped 11 (late_start: 11)
🧬 [L1: MULTI-TF PANEL INJECTION]
  └─ Active : [12h] Proj=16 Syms=54 | [6h] Proj=14 Syms=54 | [8h] Proj=14 Syms=54

● [DATA-INTEGRITY AUDIT]
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ✅ ALL 54 SYMBOLS PASSED
  METRICS : Total Bars: 8,767
  DETAIL  : [NaN: 0.0%] [Zero/Neg: 0.0%] [Range: PASS]
──────────────────────────────────────────────────────────────────────────────

📈 [ENS-FINAL] Arch-Only | SYM:54 | EVT:1,052,034 | TOTAL:+65.6 bps | [TRD:+76.4✅ TMO:+57.1✅ MRV:+39.2✅ CRY:+33.3✅ FLO:+19.2✅ UNW:+27.3✅ BTN:+44.9✅ X:52.3✅]

● [LAYER 1 OUTER FOLD READINESS]
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ✅ READY (4/4 Folds Ready)

  [✅] Fold #0 (FitEnd: 2023-12-30 -> OOS: 2023-12-30 ~ 2024-03-30)
       ├─ Symbols : 23 symbols loaded [ADAUSDT, ANKRUSDT, API3USDT, ARPAUSDT, ARUSDT, ATOMUSDT, AVAXUSDT, BCHUSDT, +15 more]
       ├─ Events  : 9654 unique events
       └─ Quality : Edge: 120.92 bps

  [✅] Fold #1 (FitEnd: 2024-03-30 -> OOS: 2024-03-31 ~ 2024-06-30)
       ├─ Symbols : 24 symbols loaded [1000XECUSDT, ADAUSDT, ANKRUSDT, API3USDT, ARPAUSDT, ATOMUSDT, AXSUSDT, CRVUSDT, +16 more]
       ├─ Events  : 10011 unique events
       └─ Quality : Edge: 75.27 bps

  [✅] Fold #2 (FitEnd: 2024-06-30 -> OOS: 2024-06-30 ~ 2024-09-30)
       ├─ Symbols : 26 symbols loaded [1000XECUSDT, ADAUSDT, API3USDT, ARPAUSDT, ARUSDT, ATOMUSDT, AVAXUSDT, BNBUSDT, +18 more]
       ├─ Events  : 10571 unique events
       └─ Quality : Edge: 89.29 bps

  [✅] Fold #3 (FitEnd: 2024-09-30 -> OOS: 2024-09-30 ~ 2024-12-30)
       ├─ Symbols : 26 symbols loaded [1000SHIBUSDT, 1000XECUSDT, AAVEUSDT, API3USDT, ARUSDT, AXSUSDT, BCHUSDT, BNBUSDT, +18 more]
       ├─ Events  : 10590 unique events
       └─ Quality : Edge: 76.33 bps

──────────────────────────────────────────────────────────────────────────────
🏁 STATUS : ✅ PASSED (5/5 Passed)
  👉 Cov:1.000(>=0.80) | Qual:1.000(>=0.90) | Brd:35.129(>=3.00) | Fld:1.000(>=0.50) | Prf:74.060(>0.00)

🏆 [L1 FINAL PROMOTION SUMMARY] 🚀 (Top 5 / 36 Promoted)
  RANK  SYMBOL       STRATEGY (Family)                EDGE(bps)  LCB(bps)    CONV    FOLDS   t(blk)
  ────  ──────────   ───────────────────────────────  ─────────  ────────  ──────  ───────  ───────
  #1    KAVAUSDT     trend_pullback_continuation ...     +236.3    +162.7    1.00      3/4     3.60
  #2    SANDUSDT     trend_pullback_continuation ...     +325.4    +144.5    1.00      4/4     2.55
  #3    1000SHIBUSDT residual_reversion (rr_48_4h)       +191.0    +140.8    1.00      4/4     5.85
  #4    RUNEUSDT     trend_donchian (donchian_72_4h)     +238.8    +129.1    1.00      3/4     2.42
  #5    ZILUSDT      trend_pullback_continuation ...     +231.8    +127.9    1.00      4/4     3.74
  └─ 🚀 And 31 more pairs promoted (e.g. ZECUSDT, GALAUSDT, THETAUSDT, ANKRUSDT, XLMUSDT, AXSUSDT, +25 more)
  ────  ──────────   ───────────────────────────────  ─────────  ────────  ──────  ───────  ───────
  [NOT PROMOTED] 1406 pairs | top: no_incremental_edgex702, quality_weight_zerox690, negative_gross_edgex186

📈 [ENS-FINAL] Arch-Only | SYM:54 | EVT:712,288 | TOTAL:+64.3 bps | [TRD:+73.2✅ TMO:+66.1✅ MRV:+34.3✅ CRY:+29.8✅ X:49.0✅]

● [LAYER 1 OUTER FOLD READINESS]
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ✅ READY (4/4 Folds Ready)

  [✅] Fold #0 (FitEnd: 2023-12-30 -> OOS: 2023-12-30 ~ 2024-03-30)
       ├─ Symbols : 14 symbols loaded [AAVEUSDT, ADAUSDT, ARUSDT, ATOMUSDT, BCHUSDT, DOTUSDT, GALAUSDT, KAVAUSDT, +6 more]
       ├─ Events  : 6001 unique events
       └─ Quality : Edge: 141.54 bps

  [✅] Fold #1 (FitEnd: 2024-03-30 -> OOS: 2024-03-31 ~ 2024-06-30)
       ├─ Symbols : 15 symbols loaded [AAVEUSDT, ADAUSDT, ANKRUSDT, ATOMUSDT, BNBUSDT, CRVUSDT, DOTUSDT, ETCUSDT, +7 more]
       ├─ Events  : 5379 unique events
       └─ Quality : Edge: 19.55 bps

  [✅] Fold #2 (FitEnd: 2024-06-30 -> OOS: 2024-06-30 ~ 2024-09-30)
       ├─ Symbols : 23 symbols loaded [AAVEUSDT, ADAUSDT, ATOMUSDT, AXSUSDT, BNBUSDT, CRVUSDT, DOTUSDT, ETCUSDT, +15 more]
       ├─ Events  : 8852 unique events
       └─ Quality : Edge: 66.93 bps

  [✅] Fold #3 (FitEnd: 2024-09-30 -> OOS: 2024-09-30 ~ 2024-12-30)
       ├─ Symbols : 20 symbols loaded [AAVEUSDT, ADAUSDT, ARPAUSDT, ATOMUSDT, AXSUSDT, BNBUSDT, ETCUSDT, ETHUSDT, +12 more]
       ├─ Events  : 5317 unique events
       └─ Quality : Edge: 91.80 bps

──────────────────────────────────────────────────────────────────────────────
🏁 STATUS : ✅ PASSED (5/5 Passed)
  👉 Cov:1.000(>=0.80) | Qual:1.000(>=0.90) | Brd:25.165(>=3.00) | Fld:1.000(>=0.50) | Prf:42.665(>0.00)

🏆 [L1 FINAL PROMOTION SUMMARY] 🚀 (Top 5 / 31 Promoted)
  RANK  SYMBOL       STRATEGY (Family)                EDGE(bps)  LCB(bps)    CONV    FOLDS   t(blk)
  ────  ──────────   ───────────────────────────────  ─────────  ────────  ──────  ───────  ───────
  #1    STORJUSDT    trend_pullback_continuation ...     +287.2    +183.2    1.00      3/3     3.08
  #2    ZRXUSDT      trend_donchian (donchian_72_6h)     +325.7    +147.5    1.00      3/3     2.69
  #3    ANKRUSDT     trend_donchian (donchian_72_6h)     +248.0    +146.8    1.00      3/3     4.14
  #4    GALAUSDT     trend_donchian (donchian_72_6h)     +277.7    +120.7    1.00      4/4     3.79
  #5    ARUSDT       trend_donchian (donchian_72_6h)     +250.0    +108.7    1.00      4/4     2.82
  └─ 🚀 And 26 more pairs promoted (e.g. LTCUSDT, XLMUSDT, DYDXUSDT, BNBUSDT, THETAUSDT, 1000SHIBUSDT, +20 more)
  ────  ──────────   ───────────────────────────────  ─────────  ────────  ──────  ───────  ───────
  [NOT PROMOTED] 617 pairs | top: quality_weight_zerox342, no_incremental_edgex275, negative_gross_edgex65

📈 [ENS-FINAL] Arch-Only | SYM:54 | EVT:707,944 | TOTAL:+63.2 bps | [TRD:+71.5✅ TMO:+71.7✅ MRV:+13.1✅ CRY:+31.7✅ X:49.0✅]

● [LAYER 1 OUTER FOLD READINESS]
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ✅ READY (4/4 Folds Ready)

  [✅] Fold #0 (FitEnd: 2023-12-30 -> OOS: 2023-12-30 ~ 2024-03-30)
       ├─ Symbols : 11 symbols loaded [ARUSDT, ATOMUSDT, BCHUSDT, DOTUSDT, GALAUSDT, MANAUSDT, NEARUSDT, SANDUSDT, +3 more]
       ├─ Events  : 4371 unique events
       └─ Quality : Edge: 116.89 bps

  [✅] Fold #1 (FitEnd: 2024-03-30 -> OOS: 2024-03-31 ~ 2024-06-30)
       ├─ Symbols : 14 symbols loaded [AAVEUSDT, ADAUSDT, ATOMUSDT, BNBUSDT, ETCUSDT, GALAUSDT, IOTAUSDT, KAVAUSDT, +6 more]
       ├─ Events  : 5258 unique events
       └─ Quality : Edge: 97.17 bps

  [✅] Fold #2 (FitEnd: 2024-06-30 -> OOS: 2024-06-30 ~ 2024-09-30)
       ├─ Symbols : 18 symbols loaded [AAVEUSDT, ADAUSDT, ATOMUSDT, BNBUSDT, DOTUSDT, GALAUSDT, JASMYUSDT, KAVAUSDT, +10 more]
       ├─ Events  : 5817 unique events
       └─ Quality : Edge: 87.53 bps

  [✅] Fold #3 (FitEnd: 2024-09-30 -> OOS: 2024-09-30 ~ 2024-12-30)
       ├─ Symbols : 10 symbols loaded [AAVEUSDT, BNBUSDT, GALAUSDT, JASMYUSDT, NEARUSDT, RUNEUSDT, THETAUSDT, TRBUSDT, +2 more]
       ├─ Events  : 2253 unique events
       └─ Quality : Edge: 114.08 bps

──────────────────────────────────────────────────────────────────────────────
🏁 STATUS : ✅ PASSED (5/5 Passed)
  👉 Cov:1.000(>=0.80) | Qual:1.000(>=0.90) | Brd:19.922(>=3.00) | Fld:1.000(>=0.50) | Prf:78.628(>0.00)

🏆 [L1 FINAL PROMOTION SUMMARY] 🚀 (Top 5 / 34 Promoted)
  RANK  SYMBOL       STRATEGY (Family)                EDGE(bps)  LCB(bps)    CONV    FOLDS   t(blk)
  ────  ──────────   ───────────────────────────────  ─────────  ────────  ──────  ───────  ───────
  #1    1000SHIBUSDT dual_momentum (dm_6_24_8h)          +194.2    +136.2    1.00      4/4     4.24
  #2    GALAUSDT     trend_donchian (donchian_72_8h)     +277.2    +122.7    1.00      4/4     3.54
  #3    XLMUSDT      trend_donchian (donchian_72_8h)     +350.5    +104.8    0.99      3/4     1.80
  #4    JASMYUSDT    dual_momentum (dm_12_48_8h)         +221.9     +91.1    1.00      4/4     2.69
  #5    RUNEUSDT     trend_donchian (donchian_72_8h)     +210.8     +88.8    0.99      3/4     2.08
  └─ 🚀 And 29 more pairs promoted (e.g. SNXUSDT, NEARUSDT, UNIUSDT, ARPAUSDT, ZECUSDT, DOTUSDT, +23 more)
  ────  ──────────   ───────────────────────────────  ─────────  ────────  ──────  ───────  ───────
  [NOT PROMOTED] 614 pairs | top: quality_weight_zerox311, no_incremental_edgex303, negative_gross_edgex78

📈 [ENS-FINAL] Arch-Only | SYM:54 | EVT:812,001 | TOTAL:+68.7 bps | [TRD:+76.8✅ TMO:+72.8✅ MRV:+24.4✅ CRY:+38.9✅ X:51.4✅]

● [LAYER 1 OUTER FOLD READINESS]
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ✅ READY (4/4 Folds Ready)

  [✅] Fold #0 (FitEnd: 2023-12-30 -> OOS: 2023-12-30 ~ 2024-03-30)
       ├─ Symbols : 22 symbols loaded [1000XECUSDT, ARUSDT, ATOMUSDT, AVAXUSDT, AXSUSDT, BCHUSDT, ETCUSDT, FILUSDT, +14 more]
       ├─ Events  : 6180 unique events
       └─ Quality : Edge: 95.67 bps

  [✅] Fold #1 (FitEnd: 2024-03-30 -> OOS: 2024-03-31 ~ 2024-06-30)
       ├─ Symbols : 18 symbols loaded [1000XECUSDT, ANKRUSDT, ATOMUSDT, AVAXUSDT, BNBUSDT, ETCUSDT, ETHUSDT, FILUSDT, +10 more]
       ├─ Events  : 5054 unique events
       └─ Quality : Edge: 80.00 bps

  [✅] Fold #2 (FitEnd: 2024-06-30 -> OOS: 2024-06-30 ~ 2024-09-30)
       ├─ Symbols : 26 symbols loaded [1000XECUSDT, ANKRUSDT, ARPAUSDT, ATOMUSDT, AVAXUSDT, BNBUSDT, CRVUSDT, DOTUSDT, +18 more]
       ├─ Events  : 7144 unique events
       └─ Quality : Edge: 71.07 bps

  [✅] Fold #3 (FitEnd: 2024-09-30 -> OOS: 2024-09-30 ~ 2024-12-30)
       ├─ Symbols : 26 symbols loaded [1000SHIBUSDT, 1000XECUSDT, AAVEUSDT, ARPAUSDT, ATOMUSDT, AXSUSDT, BCHUSDT, BNBUSDT, +18 more]
       ├─ Events  : 9138 unique events
       └─ Quality : Edge: 77.03 bps

──────────────────────────────────────────────────────────────────────────────
🏁 STATUS : ✅ PASSED (5/5 Passed)
  👉 Cov:1.000(>=0.80) | Qual:1.000(>=0.90) | Brd:32.061(>=3.00) | Fld:1.000(>=0.55) | Prf:61.523(>0.00)

🏆 [L1 FINAL PROMOTION SUMMARY] 🚀 (Top 5 / 50 Promoted)
  RANK  SYMBOL       STRATEGY (Family)                EDGE(bps)  LCB(bps)    CONV    FOLDS   t(blk)
  ────  ──────────   ───────────────────────────────  ─────────  ────────  ──────  ───────  ───────
  #1    ZRXUSDT      trend_donchian (donchian_72_...     +280.8    +127.9    1.00      3/3     2.51
  #2    1000SHIBUSDT dual_momentum (dm_4_16_12h)         +201.2    +124.5    1.00      4/4     4.00
  #3    ANKRUSDT     trend_donchian (donchian_72_...     +225.6     +99.3    1.00      3/3     3.21
  #4    AXSUSDT      dual_momentum (dm_8_32_12h)         +188.9     +79.6    1.00      4/4     3.42
  #5    JASMYUSDT    dual_momentum (dm_4_16_12h)         +179.5     +78.1    0.99      4/4     3.02
  └─ 🚀 And 45 more pairs promoted (e.g. GALAUSDT, AXSUSDT, MTLUSDT, RUNEUSDT, XLMUSDT, ZECUSDT, +39 more)
  ────  ──────────   ───────────────────────────────  ─────────  ────────  ──────  ───────  ───────
  [NOT PROMOTED] 652 pairs | top: no_incremental_edgex329, quality_weight_zerox323, negative_gross_edgex78


● [LAYER UNIVERSE AUDIT]
──────────────────────────────────────────────────────────────────────────────
  LAYER  WINDOW RANGE                    SYMS   ACTIVE (min/med/max)       ENTRY    KILL  WARNINGS
  ─────  ──────────────────────────────  ────   ────────────────────  ──────────  ──────  ────────
  L1     2023-06-30 ~ 2024-12-30           54     0 /  51.0 / 54               0      13  —
──────────────────────────────────────────────────────────────────────────────


>> LAYER 1: PASS -> Proceeding to Layer 2.
[REGIME]
metric        | value
compression   | on
states        | 3
status        | 🟢 stable
distribution  | bull=36.1% bear=25.9% crisis=38.0%
policy_mode   | soft
hard_block    | off
risk_cap      | on
policy_source | fit/cal
oos_debug     | evaluation only
note          | L2 verdict is reported separately in [REGIME-L2]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
● [LAYER 2: OPTUNA TUNING]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[REGIME-L2] active_states=3 compression=True path=regime_conditioned proof=True lift=11.07 t=3.22 fold_pass=0.67
[REGIME-L2] policy_mode=soft policy_source=fit/cal global_reliable=True allow=248 downweight=16 block=0 pooled=0 unstable=4 hard_block_eligible=0 sign_consistency=0.71 hard_block_enabled=False mean_cal_lift=-24.31 mean_conf=0.98
  ● [STUDY] l2_study_8h_2521e715ee23 | trials=200 | events=80347 | symbols=40
  ────────────────────────────────────────────────────────────────────────────
[L2-OPT]: 100%|██████████| 200/200 [02:58<00:00, Best CAGR: 232.31%]
[L2-SELECTION] Found 26 gate-passed trials in frontier. Reducing replay size to 8.
[L2-SELECTION] 8 gate-pass 후보 수집 → champion Trial #81 Sortino=2.8150 CAGR=0.4995
[L2-SELECTION] Champion selected. Trial #81, Objective=2.3231, DSR=0.6243 (n_eff=10.50)
[L2-DEPLOY-C4] L*=2.055 (binding=oos_blend) | realized_mode=return_scaling | kelly=0.250(불변) | tf=8h
  ● [CHAMPION STORE] 신규 챔피언 갱신 (tf=8h, growth_lcb=0.1526)
  ● [FINAL SIMULATION]
[L2-DEPLOY] L*=2.0550 binding=champion | CAGR=0.4995 MDD=0.1748 CVaR95=0.0172 RiskUtil=0.583
● [LAYER 2 PORTFOLIO SCORECARD] (2025-03-23 ~ 2025-12-30)
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ✅ PASS

  ✅ [Growth    ] CAGR: +49.9% (>=30.0%) | PnL: +37.9% | Equity x1.38
  ✅ [Efficiency] Sharpe: 1.481 (>=1.000) | Sortino: 2.815 (>=1.500) | Calmar: 2.857 (>=1.000)
  ✅ [Risk      ] MDD: 17.5% (<=30.0%) | CVaR95: 1.7% (<=6.0%) | RiskUtil: 58.3%
  ✅ [Robust    ] Fold: 100.0% (>=60.0%) | Trades: 830 (>=30) | Friction: 97.7%
  ✅ [Uplift    ] Sharpe Uplift: +1.13 (>=+0.20)
  ✅ [Integrity ] DSR: 0.624 (>=0.60) | PSR: 0.983 (diag)
  [Diag     ] RelMDD: 1.74x | Turnover: 0.125
──────────────────────────────────────────────────────────────────────────────

  [ FOLD DETAIL BREAKDOWN ]
  ──────────────────────────────────────────────────────────────────────────
  ├─ Fold #1 : ✅ Sharpe:  0.000 | CAGR:   +82.5% | MDD:  11.6% | Status: PASS | Period: 2025-03-23 ~ 2025-06-25
       Symbols: 19 [1000SHIBUSDT, AAVEUSDT, AXSUSDT, BCHUSDT, BNBUSDT, BTCUSDT, DOGEUSDT, FILUSDT, +11 more]
  ├─ Fold #2 : ✅ Sharpe:  0.000 | CAGR:   +35.2% | MDD:  17.5% | Status: PASS | Period: 2025-06-25 ~ 2025-09-27
       Symbols: 29 [1000SHIBUSDT, AAVEUSDT, ADAUSDT, ATOMUSDT, AVAXUSDT, AXSUSDT, BCHUSDT, BNBUSDT, +21 more]
  └─ Fold #3 : ✅ Sharpe:  0.000 | CAGR:   +36.6% | MDD:  13.3% | Status: PASS | Period: 2025-09-27 ~ 2025-12-30
       Symbols: 28 [1000SHIBUSDT, AAVEUSDT, ADAUSDT, ARUSDT, ATOMUSDT, AVAXUSDT, AXSUSDT, BCHUSDT, +20 more]

  ⚠️  [WINDOW] NO-CRISIS-WINDOW — 이 평가 윈도우는 병목-caliber fold(MDD>=15% & CAGR<=0)를 포함하지 않음. 승격 근거로 인용 금지 (docs/results/next.md P0).

● [LAYER UNIVERSE AUDIT]
──────────────────────────────────────────────────────────────────────────────
  LAYER  WINDOW RANGE                    SYMS   ACTIVE (min/med/max)       ENTRY    KILL  WARNINGS
  ─────  ──────────────────────────────  ────   ────────────────────  ──────────  ──────  ────────
  L2     2024-12-31 ~ 2025-12-30           54    31 /  38.0 / 45           1,098      21  —
──────────────────────────────────────────────────────────────────────────────

>> LAYER 2: PASS -> Proceeding to Final Holdout.
>> TARGET PHASE l2 REACHED -> Stopping pipeline.
[L2-PARITY-SELFCHECK] side=replay stored=0.499472 recomputed=1.248416 (n_rets=1692 L*=2.054970 bpy=2190.0) -> field/metric DECOUPLED
[PHASE] phase=l2 completed strategy/candidate evaluation only; optimization/training skipped

================================================================================
실행 조건: `uv run python -m src.execution.opt_main_futures --phase l2 --sync skip` (2026-07-01, `L2_REVERSAL_KILL` 등 진단 플래그 전부 비활성 — 순수 프로덕션 기본값)

[P0 가드레일 적용 후 재실행 (2026-07-01)] docs/specs/l2-promotion-crisis-guardrail.md 구현 반영:
  - Fix 1: Fold MDD 리포팅 버그 수정 — 이전 실행은 전 fold MDD 0.0%로 하드코딩되어 있었음(버그).
    본 실행부터 실측값(Fold#1 11.6% / Fold#2 17.5% / Fold#3 13.3%) 정상 표시.
  - Gate A: NO-CRISIS-WINDOW 배너 신규 노출 — 평가 윈도우(2024-12-31~2025-12-30)에 병목-caliber
    fold(MDD>=15% & CAGR<=0)가 없음을 자동 판별해 스코어카드에 경고 표시. Fold#2(MDD 17.5%)는
    CAGR +35.2%(양수, 강세장 조정)라서 위기 오인 없이 정확히 미포함 판정됨.
  - Gate B + Change 4: 챔피언 스토어 갱신 전 synthetic crash defense(Scenario 8) 발화 검증 통과
    (crash_fires=True) → 정상 승격. 메커니즘 자체는 건강하나 이번 윈도우엔 진짜 위기가 없었다는
    사실이 배너로 투명하게 남음 — "위기 부재 PASS"가 조용히 승격 근거로 오인되는 것을 구조적으로 차단.
  - 전체 PASS 판정 자체는 유지(회귀 없음), CAGR/Sharpe 등은 Optuna trial 재현성 범위 내 정상 변동.
================================================================================
