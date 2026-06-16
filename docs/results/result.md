# Layer 1 Signal Robustness & Ensemble Verification Result

## 1. 개요 및 실행 과정
Layer 1(L1)은 전략 후보군의 통계적 유의성과 견고함(Robustness)을 검증하는 단계입니다. 본 시스템은 다음의 엄격한 프로세스를 거쳐 심볼과 전략을 평가합니다.

1.  **Data-Integrity Audit:** 데이터의 결측치(NaN), 0 또는 음수 값, 고가-저가 역전 현상 등을 전수 조사하여 오염된 데이터를 사전에 차단합니다.
2.  **Ensemble Fitting (Arch-Only):** 개별 전략의 노이즈를 줄이기 위해 아키타입(Archetype)별로 데이터를 묶어 앙상블 학습을 수행하고, 전략의 기초 체력(Edge)을 추정합니다.
3.  **Outer Fold Readiness (Nested SWF):** 전체 데이터를 여러 구간(Fold)으로 나누어, 각 구간에서 전략이 일관되게 수익을 내는지 검증합니다.
4.  **Hard Gate Validation:** 모든 구간과 심볼에 걸쳐 집계된 지표가 사전에 정의된 5가지 통계적 허들(Hard Gate)을 모두 통과해야 최종 승격됩니다.

---

## 2. 최신 실행 결과 분석 (2026-06-15)

[SYNC] Ledger up-to-date (Last: 2026-03-31)
discover_universe_timeline: l2_start(2024-10-01) < oos_start(2025-10-01), ignoring l2_start
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
  METRICS : Total Bars: 7,518
  DETAIL  : [NaN: 0.0%] [Zero/Neg: 0.0%] [Range: PASS]
──────────────────────────────────────────────────────────────────────────────

[TIERED] 💠 Scope: 54 symbols (Historical Union ∩ Data-Valid)
[ENS] Sym:3   N:353  Mu:40.8  Mode:Arch-Only  k:50.0
└─Mu: B:-13.4❌ M:26.2✅ T:214.7✅ m:32.4✅ U:8.3✅
[ENS] Sym:3   N:2,101  Mu:12.9  Mode:Arch-Only  k:50.0
└─Mu: B:-4.0❌ M:9.9✅ T:25.8✅ m:40.3✅ U:9.8✅
[ENS] Sym:3   N:3,755  Mu:12.6  Mode:Arch-Only  k:50.0
└─Mu: B:-0.1❌ M:10.8✅ T:19.4✅ m:25.1✅ U:16.4✅
[ENS] Sym:18  N:6,587  Mu:15.2  Mode:Arch-Only  k:50.0
└─Mu: B:30.7✅ M:13.0✅ T:19.0✅ m:16.8✅ U:52.2✅
[ENS] Sym:18  N:16,309  Mu:5.1  Mode:Arch-Only  k:50.0
└─Mu: B:21.2✅ M:3.7✅ T:11.3✅ m:6.0✅ U:11.6✅
[ENS] Sym:18  N:26,073  Mu:13.4  Mode:Arch-Only  k:50.0
└─Mu: B:13.7✅ M:10.5✅ T:48.3✅ m:17.8✅ U:3.3✅
[ENS] Sym:26  N:34,059  Mu:9.5  Mode:Arch-Only  k:50.0
└─Mu: B:10.1✅ M:6.6✅ T:37.2✅ m:19.6✅ U:-0.0❌
[ENS] Sym:26  N:42,600  Mu:15.0  Mode:Arch-Only  k:50.0
└─Mu: B:21.2✅ F:2.7✅ M:11.1✅ T:52.7✅ m:20.5✅ U:6.6✅
[ENS] Sym:26  N:49,357  Mu:15.8  Mode:Arch-Only  k:50.0
└─Mu: B:24.0✅ F:4.7✅ M:12.1✅ T:44.1✅ m:23.3✅ U:13.5✅
[ENS] Sym:30  N:56,130  Mu:16.2  Mode:Arch-Only  k:50.0
└─Mu: B:24.3✅ F:6.0✅ M:12.4✅ T:44.5✅ m:21.7✅ U:17.8✅
[ENS] Sym:34  N:61,011  Mu:16.5  Mode:Arch-Only  k:50.0
└─Mu: B:24.6✅ F:6.3✅ M:13.0✅ T:40.6✅ m:21.2✅ U:18.5✅
[ENS] Sym:34  N:67,028  Mu:16.6  Mode:Arch-Only  k:50.0
└─Mu: B:32.7✅ F:6.9✅ M:12.7✅ T:46.3✅ m:19.9✅ U:7.9✅
[ENS] Sym:3   N:4,989  Mu:11.0  Mode:Arch-Only  k:50.0
└─Mu: B:11.2✅ M:9.3✅ T:12.1✅ m:20.6✅ U:29.3✅
[ENS] Sym:18  N:24,739  Mu:13.6  Mode:Arch-Only  k:50.0
└─Mu: B:16.3✅ M:10.8✅ T:46.5✅ m:18.7✅ U:2.0✅
[ENS] Sym:26  N:42,534  Mu:15.1  Mode:Arch-Only  k:50.0
└─Mu: B:21.1✅ F:2.8✅ M:11.3✅ T:52.5✅ m:20.5✅ U:6.9✅
[ENS] Sym:34  N:56,216  Mu:16.0  Mode:Arch-Only  k:50.0
└─Mu: B:24.5✅ F:6.0✅ M:12.4✅ T:43.4✅ m:20.5✅ U:17.7✅

● [LAYER 1 OUTER FOLD READINESS]
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ✅ READY (4/4 Folds Ready)

  [✅] Fold #0 (Fit: 2954 → OOS: 3288)
       ├─ Symbols : 7 symbols loaded
       ├─ Events  : 390 unique events
       └─ Quality : Edge: 124.81 bps

  [✅] Fold #1 (Fit: 3394 → OOS: 3837)
       ├─ Symbols : 7 symbols loaded
       ├─ Events  : 259 unique events
       └─ Quality : Edge: 50.81 bps

  [✅] Fold #2 (Fit: 3833 → OOS: 4386)
       ├─ Symbols : 7 symbols loaded
       ├─ Events  : 141 unique events
       └─ Quality : Edge: 245.64 bps

  [✅] Fold #3 (Fit: 4272 → OOS: 4935)
       ├─ Symbols : 7 symbols loaded
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
  ✅ [Min-Profit     ] :   49.227 (Target >0.000  )
──────────────────────────────────────────────────────────────────────────────

[ENS-FINAL] Sym:45  N:100,404  Mu:22.3  Mode:Arch-Only  k:50.0
└─Mu: B:27.2✅ F:12.3✅ M:16.3✅ T:52.7✅ m:41.7✅ U:15.2✅

[L1 FINAL PROMOTION SUMMARY] 🚀
--------------------------------------------------------------------------------------------
 RANK | SYMBOL   | STRATEGY (Family)              | EDGE(bps) | SIG(t-stat) | CONF(q) | STATUS
--------------------------------------------------------------------------------------------
  #1  | STORJUSDT | trend_pullback_continuation    |    +383.7 | ★★★★★ 4.25 |  0.130  | PROMOTED (Best Q)
  #2  | SOLUSDT  | vol_breakout (bb_compress_20   |    +152.3 | ★★★★★ 4.17 |  0.202  | PROMOTED
  #3  | ZECUSDT  | dual_momentum (dm_12_48)       |    +178.5 | ★★★★☆ 3.44 |  0.655  | WATCH
  #4  | SNXUSDT  | trend_pullback_continuation    |    +382.9 | ★★★★☆ 3.24 |  0.281  | PROMOTED
  #5  | SNXUSDT  | vol_breakout (bb_compress_20   |    +129.3 | ★★★★☆ 3.22 |  0.601  | WATCH
  #6  | LTCUSDT  | funding_extreme_reversal (fe   |     +53.5 | ★★★★☆ 3.08 |  0.416  | WATCH
  #7  | BTCUSDT  | funding_zscore_carry (fzs_48   |     +33.3 | ★★★★☆ 3.03 |  0.765  | REJECTED
  #8  | BTCUSDT  | funding_zscore_carry (fzs_96   |     +30.3 | ★★★☆☆ 2.83 |  0.416  | WATCH
  #9  | MKRUSDT  | rsi_reversion (rsi_14)         |    +108.6 | ★★★☆☆ 2.57 |  0.130  | PROMOTED (Best Q)
  #10 | LTCUSDT  | funding_zscore_carry (fzs_16   |     +61.4 | ★★★☆☆ 2.50 |  0.221  | PROMOTED
  #11 | CRVUSDT  | trend_pullback_continuation    |    +268.3 | ★★★☆☆ 2.49 |  0.130  | PROMOTED (Best Q)
  #12 | ARPAUSDT | funding_zscore_carry (fzs_16   |    +102.7 | ★★★☆☆ 2.44 |  0.416  | WATCH
  #13 | SNXUSDT  | trend_pullback_continuation    |    +169.7 | ★★★☆☆ 2.44 |  0.320  | WATCH
  #14 | FILUSDT  | rsi_reversion (rsi_14)         |     +84.2 | ★★★☆☆ 2.37 |  0.437  | WATCH
  #15 | ANKRUSDT | trend_ma (ema_12_72)           |     +46.5 | ★★★☆☆ 2.18 |  0.437  | WATCH
  #16 | CRVUSDT  | trend_ma (ema_12_72)           |     +69.2 | ★★★☆☆ 2.10 |  0.939  | REJECTED
--------------------------------------------------------------------------------------------

>> LAYER 1 RESULT: [PASS] -> Target phase L1 reached. Stopping pipeline.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
● [LAYER 2: PORTFOLIO ALLOCATION & RISK OPTIMIZATION]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ● [HYPERPARAMETER OPTIMIZATION]
    - Study Name : l2_study_4h_09eb97d7f27f
    - Config     : 120 trials
  ────────────────────────────────────────────────────────────────────────────
[L2-SELECTION] Replay mismatch on trial #92: stored_cagr=0.328186 replayed_cagr=0.473765
[L2-SELECTION] Champion selected. Trial #92, Objective=0.3097, DSR=0.3297 (n_eff=5.27)
  ● [CHAMPION STORE] 신규 챔피언 갱신 (tf=4h, growth_lcb=0.4170)
● [LAYER 2 PORTFOLIO SCORECARD]
──────────────────────────────────────────────────────────────────────────────
  STATUS  : ✅ PASS

  ✅ [Growth    ] CAGR: +32.8% (>=30.0%) | PnL: +15.2% | Equity x1.15
  ✅ [Efficiency] Sharpe: 1.959 (>=1.000) | Sortino: 2.142 (>=1.500) | Calmar: 5.213 (>=1.000)
  ✅ [Risk      ] MDD: 6.3% (<=30.0%) | CVaR95: 0.8% (<=6.0%) | RiskUtil: 21.0%
  ✅ [Robust    ] Fold: 100.0% (>=60.0%) | Trades: 58 (>=30) | Friction: 100.0%
  ✅ [Uplift    ] Sharpe Uplift: +2.07 (>=+0.20)
  [Diag     ] DSR: 0.330 (diag) | PSR: 0.924 (diag) | RelMDD: 1.22x | Turnover: 0.055
──────────────────────────────────────────────────────────────────────────────

  [ FOLD DETAIL BREAKDOWN ]
  ──────────────────────────────────────────────────────────────────────────
  ├─ Fold #1 : ✅ Sharpe:  1.874 | MDD:    3.4% | Status: PASS
  ├─ Fold #2 : ✅ Sharpe:  1.232 | MDD:    4.4% | Status: PASS
  └─ Fold #3 : ✅ Sharpe:  2.721 | MDD:    6.3% | Status: PASS
>> LAYER 2 RESULT: [PASS] -> Proceeding to Final Holdout.
>> TARGET PHASE l2 REACHED -> Stopping pipeline.
[PHASE] phase=l2 completed strategy/candidate evaluation only; optimization/training skipped
