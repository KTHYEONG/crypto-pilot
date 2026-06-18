## 실행 결과

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
  STATUS  : ✅ READY (4/4 Folds Ready)

  [✅] Fold #0 (FitEnd: 2023-08-14 -> OOS: 2023-10-01 ~ 2023-12-31)
       ├─ Symbols : 7 symbols loaded [BTCUSDT, CRVUSDT, ETHUSDT, FTMUSDT, NEOUSDT, SOLUSDT, ZECUSDT]
       ├─ Events  : 390 unique events
       └─ Quality : Edge: 97.21 bps

  [✅] Fold #1 (FitEnd: 2023-10-31 -> OOS: 2023-12-31 ~ 2024-03-31)
       ├─ Symbols : 7 symbols loaded [BTCUSDT, CRVUSDT, LTCUSDT, SNXUSDT, STORJUSDT, XRPUSDT, ZECUSDT]
       ├─ Events  : 259 unique events
       └─ Quality : Edge: 50.81 bps

  [✅] Fold #2 (FitEnd: 2024-01-17 -> OOS: 2024-04-01 ~ 2024-07-01)
       ├─ Symbols : 7 symbols loaded [BTCUSDT, CRVUSDT, ETHUSDT, FILUSDT, MKRUSDT, NEOUSDT, STORJUSDT]
       ├─ Events  : 141 unique events
       └─ Quality : Edge: 247.03 bps

  [✅] Fold #3 (FitEnd: 2024-04-04 -> OOS: 2024-07-01 ~ 2024-09-30)
       ├─ Symbols : 7 symbols loaded [ANKRUSDT, BTCUSDT, FTMUSDT, GALAUSDT, NEARUSDT, SOLUSDT, ZRXUSDT]
       ├─ Events  : 115 unique events
       └─ Quality : Edge: 6.30 bps

──────────────────────────────────────────────────────────────────────────────

● [LAYER 1 HARD GATE CHECKS]
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ✅ PASSED (5/5 Passed)

  ✅ [Time-Coverage  ] :    1.000 (Target >=0.800 )
  ✅ [Signal-Quality ] :    1.000 (Target >=0.900 )
  ✅ [Symbol-Breadth ] :   13.517 (Target >=3.000 )
  ✅ [Stable-Folds   ] :    1.000 (Target >=0.500 )
  ✅ [Min-Profit     ] :   47.342 (Target >0.000  )
──────────────────────────────────────────────────────────────────────────────

[ENS-FINAL] Arch-Only   | SYM: 45 | EVT: 100,404 | TOTAL: +22.3 bps | IC:-0.05❌ | [BRK: +27.2✅ MOM: +16.3✅ TRD: +52.7✅ MRV: +41.7✅ UNI:+15.2✅ F:12.3✅]

[L1 FINAL PROMOTION SUMMARY] 🚀
  RANK  SYMBOL       STRATEGY (Family)                EDGE(bps)  SIG(t-stat)     STATUS
  ────  ──────────   ───────────────────────────────  ─────────  ──────────────  ──────────────────
  #1    STORJUSDT    trend_pullback_continuation ...     +383.7  [4/5] 4.25      PROMOTED (Best Q)
  #2    SOLUSDT      vol_breakout (bb_compress_20)       +152.3  [4/5] 4.17      PROMOTED
  #3    ZECUSDT      dual_momentum (dm_12_48)            +178.5  [3/5] 3.44      WATCH
  #4    SNXUSDT      trend_pullback_continuation ...     +382.9  [3/5] 3.24      PROMOTED
  #5    SNXUSDT      vol_breakout (bb_compress_20)       +129.3  [3/5] 3.22      WATCH
  #6    LTCUSDT      funding_extreme_reversal (fe...      +53.5  [3/5] 3.08      WATCH
  #7    BTCUSDT      funding_zscore_carry (fzs_48)        +33.3  [3/5] 3.03      REJECTED
  #8    BTCUSDT      funding_zscore_carry (fzs_96)        +30.3  [3/5] 2.83      WATCH
  #9    MKRUSDT      rsi_reversion (rsi_14)              +108.6  [3/5] 2.57      PROMOTED (Best Q)
  #10   LTCUSDT      funding_zscore_carry (fzs_168)       +61.4  [3/5] 2.50      PROMOTED
  #11   CRVUSDT      trend_pullback_continuation ...     +268.3  [2/5] 2.49      PROMOTED (Best Q)
  #12   ARPAUSDT     funding_zscore_carry (fzs_168)      +102.7  [2/5] 2.44      WATCH
  #13   SNXUSDT      trend_pullback_continuation ...     +169.7  [2/5] 2.44      WATCH
  #14   FILUSDT      rsi_reversion (rsi_14)               +84.2  [2/5] 2.37      WATCH
  #15   ANKRUSDT     trend_ma (ema_12_72)                 +46.5  [2/5] 2.18      WATCH
  #16   CRVUSDT      trend_ma (ema_12_72)                 +69.2  [2/5] 2.10      REJECTED
  ────  ──────────   ───────────────────────────────  ─────────  ──────────────  ──────────────────


● [LAYER UNIVERSE AUDIT]
──────────────────────────────────────────────────────────────────────────────
  LAYER  WINDOW RANGE                    SYMS   ACTIVE (min/med/max)       ENTRY    KILL  WARNINGS
  ─────  ──────────────────────────────  ────   ────────────────────  ──────────  ──────  ────────
  L1     2023-04-01 ~ 2024-09-30           54     0 /  15.0 / 18         135,022      41  —
──────────────────────────────────────────────────────────────────────────────


>> LAYER 1: PASS -> Target phase L1 reached. Stopping pipeline.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
● [LAYER 2: PORTFOLIO ALLOCATION & RISK OPTIMIZATION]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ● [HYPERPARAMETER OPTIMIZATION]
    - Study Name : l2_study_4h_03972e5343cd
    - Config     : 200 trials
  ────────────────────────────────────────────────────────────────────────────
 [OPT] Deleted existing study 'l2_study_4h_03972e5343cd' for a fresh start.
[L2-OPT]: 100%|██████████████████████████████████████████████████████| 200/200 [01:02<00:00,  3.18it/s, Best CAGR: 415.10% | Current: 78.36%]
[L2-SELECTION] No feasible candidate found within fallback window (reason=sortino_floor)
[L2-DEPLOY-C4] L*=2.750 (binding=mdd) | realized_mode=return_scaling | kelly=0.250(불변) | tf=4h

>> LAYER 1: PASS -> Proceeding to Layer 2.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
● [LAYER 2: PORTFOLIO ALLOCATION & RISK OPTIMIZATION]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[L2-DEPLOY] L*=2.7504 binding=champion | CAGR=0.4556 MDD=0.2942 CVaR95=0.0290 RiskUtil=0.981
● [LAYER 2 PORTFOLIO SCORECARD] (2024-12-22 ~ 2025-09-30)
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ❌ BLOCKED (sortino_floor)

  ✅ [Growth    ] CAGR: +45.6% (>=30.0%) | PnL: +14.4% | Equity x1.14
  ❌ [Efficiency] Sharpe: 0.941 (>=1.000) | Sortino: 1.406 (>=1.500) | Calmar: 1.549 (>=1.000)
  ✅ [Risk      ] MDD: 29.4% (<=30.0%) | CVaR95: 2.9% (<=6.0%) | RiskUtil: 98.1%
  ✅ [Robust    ] Fold: 100.0% (>=60.0%) | Trades: 111 (>=30) | Friction: 100.0%
  ✅ [Uplift    ] Sharpe Uplift: +0.80 (>=+0.20)
  ✅ [Integrity ] DSR: 0.777 (>=0.60) | PSR: 0.797 (diag)
  [Diag     ] RelMDD: 2.74x | Turnover: 0.072
──────────────────────────────────────────────────────────────────────────────

  [ FOLD DETAIL BREAKDOWN ]
  ──────────────────────────────────────────────────────────────────────────
  ├─ Fold #1 : ✅ Sharpe:  0.793 | CAGR:   +31.1% | MDD:  20.8% | Status: PASS | Period: 2024-12-22 ~ 2025-03-26
       Symbols: 3 [BTCUSDT, CRVUSDT, SOLUSDT]
  ├─ Fold #2 : ✅ Sharpe:  1.435 | CAGR:   +99.6% | MDD:  29.4% | Status: PASS | Period: 2025-03-26 ~ 2025-06-28
       Symbols: 3 [BTCUSDT, CRVUSDT, LTCUSDT]
  └─ Fold #3 : ✅ Sharpe:  0.574 | CAGR:   +18.0% | MDD:  24.6% | Status: PASS | Period: 2025-06-28 ~ 2025-09-30
       Symbols: 5 [BTCUSDT, CRVUSDT, LTCUSDT, MKRUSDT, SOLUSDT]

● [LAYER UNIVERSE AUDIT]
──────────────────────────────────────────────────────────────────────────────
  LAYER  WINDOW RANGE                    SYMS   ACTIVE (min/med/max)       ENTRY    KILL  WARNINGS
  ─────  ──────────────────────────────  ────   ────────────────────  ──────────  ──────  ────────
  L2     2024-10-01 ~ 2025-09-30           54     4 /  15.0 / 18          86,271      31  —
──────────────────────────────────────────────────────────────────────────────

>> LAYER 2: BLOCKED -> gate_passed=False
!! FAIL: exit_code=1 reason=layer2_blocked:sortino_floor