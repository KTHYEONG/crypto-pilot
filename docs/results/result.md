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
  └─ Active : [12h] Proj=16 Syms=54 | [8h] Proj=14 Syms=54 | [6h] Proj=14 Syms=54

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
● [MARKET REGIME STATUS]
──────────────────────────────────────────────────────────────────────────────
  Compression  : on                 | Policy Mode : soft
  States       : 3                  | Status      : 🟢 stable
  Hard Block   : off                | Risk Cap    : on
  Source       : fit/cal            | OOS Debug   : evaluation only
  Distribution : bull=36.1% bear=25.9% crisis=38.0%
  Note         : L2 verdict is reported separately in [REGIME-L2]
──────────────────────────────────────────────────────────────────────────────

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
● [LAYER 2: OPTUNA TUNING]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ● [STUDY] l2_study_8h_8fd9d58f88b8
  ────────────────────────────────────────────────────────────────────────────
    Optuna DB : SQLite (/home/kth/my_coin_traider/logs/futures/optimization/optuna.db)
    Trials    : 200             | Events  : 80347          
    Symbols   : 40             
  ────────────────────────────────────────────────────────────────────────────
[L2-OPT]: 100%|██████████████████████████████████████████████████| 200/200 [02:09<00:00,  1.55it/s, Best CAGR: 231.23% | Current: 214.74%]
  ● [FINAL SIMULATION]
  ● [FINAL SIMULATION RESULT]
  ────────────────────────────────────────────────────────────────────────────
    Leverage (L*) : 2.0646 (binding: champion)
    CAGR / MDD    : +50.6% / 17.4%
    CVaR95 / Util : 1.7% / 58.1%
  ────────────────────────────────────────────────────────────────────────────
● [LAYER 2 PORTFOLIO SCORECARD] (2025-03-23 ~ 2025-12-30)
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ✅ PASS

  ✅ [Growth    ] CAGR: +50.6% (>=30.0%) | PnL: +38.2% | Equity x1.38
  ✅ [Efficiency] Sharpe: 1.483 (>=1.000) | Sortino: 2.822 (>=1.500) | Calmar: 2.903 (>=1.000)
  ✅ [Risk      ] MDD: 17.4% (<=30.0%) | CVaR95: 1.7% (<=6.0%) | RiskUtil: 58.1%
  ✅ [Robust    ] Fold: 100.0% (>=60.0%) | Trades: 830 (>=30) | Friction: 97.7%
  ✅ [Uplift    ] Sharpe Uplift: +1.13 (>=+0.20)
  ✅ [Integrity ] DSR: 0.720 (>=0.60) | PSR: 0.983 (diag)
  [Diag     ] RelMDD: 1.74x | Turnover: 0.125
──────────────────────────────────────────────────────────────────────────────

  [ FOLD DETAIL BREAKDOWN ]
  ──────────────────────────────────────────────────────────────────────────
  ├─ Fold #1 : ✅ Sharpe:  2.019 | CAGR:   +81.5% | MDD:  11.7% | Status: PASS | Period: 2025-03-23 ~ 2025-06-25
       Symbols: 19 [1000SHIBUSDT, AAVEUSDT, AXSUSDT, BCHUSDT, BNBUSDT, BTCUSDT, DOGEUSDT, FILUSDT, +11 more]
  ├─ Fold #2 : ✅ Sharpe:  1.277 | CAGR:   +36.9% | MDD:  17.4% | Status: PASS | Period: 2025-06-25 ~ 2025-09-27
       Symbols: 29 [1000SHIBUSDT, AAVEUSDT, ADAUSDT, ATOMUSDT, AVAXUSDT, AXSUSDT, BCHUSDT, BNBUSDT, +21 more]
  └─ Fold #3 : ✅ Sharpe:  1.137 | CAGR:   +37.6% | MDD:  13.2% | Status: PASS | Period: 2025-09-27 ~ 2025-12-30
       Symbols: 28 [1000SHIBUSDT, AAVEUSDT, ADAUSDT, ARUSDT, ATOMUSDT, AVAXUSDT, AXSUSDT, BCHUSDT, +20 more]

  ⚠️  [WINDOW] NO-CRISIS-WINDOW — 이 평가 윈도우는 병목-caliber fold(MDD>=15% & CAGR<=0)를 포함하지 않음. 승격 근거로 인용 금지 (docs/results/next.md P0).

● [LAYER UNIVERSE AUDIT]
──────────────────────────────────────────────────────────────────────────────
  LAYER  WINDOW RANGE                    SYMS   ACTIVE (min/med/max)       ENTRY    KILL  WARNINGS
  ─────  ──────────────────────────────  ────   ────────────────────  ──────────  ──────  ────────
  L2     2024-12-31 ~ 2025-12-30           54    31 /  38.0 / 45           1,098      21  —
──────────────────────────────────────────────────────────────────────────────

>> LAYER 2: PASS -> Proceeding to Final Holdout.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
● [LAYER 3: FINAL HOLDOUT & DEPLOYMENT READINESS]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

● [LAYER UNIVERSE AUDIT]
──────────────────────────────────────────────────────────────────────────────
  LAYER  WINDOW RANGE                    SYMS   ACTIVE (min/med/max)       ENTRY    KILL  WARNINGS
  ─────  ──────────────────────────────  ────   ────────────────────  ──────────  ──────  ────────
  L3     2025-12-31 ~ 2026-06-30           54    27 /  31.0 / 34           4,331      10  —
──────────────────────────────────────────────────────────────────────────────

● [LAYER 3: HOLDOUT VALIDATION SCORECARD] (2025-12-31 ~ 2026-06-30)
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ❌ BLOCKED (Reason: negative_return)

  ❌ [GROWTH    ] CAGR: -13.3% | Total Return: -13.2% (> 0.0%) | Equity x0.87
  ❌ [EFFICIENCY] Sharpe: -0.933 (>=0.000) | Sortino: -1.242 (>=0.000) | Baseline Sharpe: -0.768
  ✅ [RISK      ] MDD: 24.7% (<= 35.0%) | CVaR95: 1.1% (<= 6.0%) | Exposure: 1.0x
  ✅ [DEPLOY-READY] Trades: 241 (>= 10)
──────────────────────────────────────────────────────────────────────────────

  >> FINAL RESULT : ❌ BLOCKED (Reason: negative_return)

================================================================================