[SYNC] Ledger up-to-date (Last: 2026-03-31)
discover_universe_timeline: l2_start(2024-10-01) < oos_start(2025-10-01), ignoring l2_start
[CACHE] Skip backfill as requested
┌──────────────────────────────────────────────────────────────────────────────┐
│ [SYSTEM CONTEXT: INFRASTRUCTURE & DATA PREPARATION]                          │
│ ────────────────────────────────────────────────────────────────────────────── │
│ Window:   Range: 2022-10-01 ~ 2026-03-31 (IS:2023-10-01, OOS:2025-10-01)     │
│ Universe: Discovered: 94 symbols | Selected: 20 | Live Panel: 20             │
│ Quality:  Loaded: 62.8% (59/94) | Ready: 59 | Dropped: None                  │
│ Strategy: Engine: Alpha-Ensemble Engine | Inf Panel: 94 | Trade Scope: 59    │
└──────────────────────────────────────────────────────────────────────────────┘

================================================================================
[LAYER 1: SIGNAL ROBUSTNESS & ENSEMBLE VERIFICATION]
================================================================================
[DATA-INTEGRITY] 💠 Starting audit for 57 symbols...
[DATA-INTEGRITY] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STATUS: ✅ ALL 57 SYMBOLS PASSED (Bars: 7,518)
  AUDIT:  [NaN: 0.0%] [Zero/Neg: 0.0%] [Hi>=Lo: PASS]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[TIERED] 💠 Scope: 57 symbols (Historical Union ∩ Data-Valid)
[WORKFLOW] Fold 0 skipped Ensemble (fit=185 < 2)
[ENS] Sym:2   N:1,183  IC:-0.065❌  Mu:10.1  Mode:Arch-Only  k:50.0
└─Mu: B:12.4✅ M:8.6✅ T:2.9✅ m:29.4✅ U:10.5✅
[ENS] Sym:2   N:3,153  IC:0.042✅  Mu:5.2  Mode:Arch-Only  k:50.0
└─Mu: B:16.2✅ M:4.7✅ T:0.1✅ m:7.3✅ U:17.9✅
[ENS] Sym:2   N:2,189  IC:0.172✅  Mu:6.5  Mode:Arch-Only  k:50.0
└─Mu: B:14.4✅ M:6.2✅ T:-2.9❌ m:15.5✅ U:5.4✅
[ENS] Sym:5   N:5,764  IC:0.056✅  Mu:3.0  Mode:Arch-Only  k:50.0
└─Mu: B:32.8✅ M:2.0✅ T:-6.0❌ m:5.0✅ U:33.1✅
[ENS] Sym:5   N:7,983  IC:-0.078❌  Mu:9.0  Mode:Arch-Only  k:50.0
└─Mu: B:22.8✅ M:6.9✅ T:20.3✅ m:12.7✅ U:21.1✅
[ENS] Sym:5   N:10,588  IC:0.016✅  Mu:7.9  Mode:Arch-Only  k:50.0
└─Mu: B:20.7✅ M:6.4✅ T:16.6✅ m:7.3✅ U:19.9✅
[ENS] Sym:18  N:17,164  IC:-0.027❌  Mu:10.5  Mode:Arch-Only  k:50.0
└─Mu: B:43.1✅ F:-5.3❌ M:6.4✅ T:23.4✅ m:24.2✅ U:18.0✅
[ENS] Sym:18  N:24,298  IC:-0.051❌  Mu:10.9  Mode:Arch-Only  k:50.0
└─Mu: B:54.5✅ F:1.0✅ M:7.4✅ T:23.5✅ m:10.7✅ U:17.4✅
[ENS] Sym:18  N:32,613  IC:0.034✅  Mu:14.2  Mode:Arch-Only  k:50.0
└─Mu: B:56.6✅ F:4.5✅ M:8.9✅ T:32.7✅ m:23.7✅ U:18.5✅
[ENS] Sym:33  N:34,977  IC:-0.020❌  Mu:14.2  Mode:Arch-Only  k:50.0
└─Mu: B:49.1✅ F:4.9✅ M:9.6✅ T:29.6✅ m:18.8✅ U:22.3✅
[ENS] Sym:33  N:42,719  IC:-0.004❌  Mu:13.7  Mode:Arch-Only  k:50.0
└─Mu: B:44.1✅ F:4.1✅ M:10.1✅ T:24.0✅ m:18.5✅ U:16.8✅
[ENS] Sym:2   N:1,493  IC:-0.111❌  Mu:7.9  Mode:Arch-Only  k:50.0
└─Mu: B:10.8✅ M:8.2✅ T:0.1✅ m:11.9✅ U:5.8✅
[ENS] Sym:5   N:5,381  IC:0.123✅  Mu:1.7  Mode:Arch-Only  k:50.0
└─Mu: B:35.7✅ M:1.2✅ T:-10.8❌ m:-0.4❌ U:31.9✅
[ENS] Sym:18  N:13,161  IC:0.019✅  Mu:3.0  Mode:Arch-Only  k:50.0
└─Mu: B:15.0✅ M:1.4✅ T:7.3✅ m:9.2✅ U:13.2✅
[ENS] Sym:18  N:33,368  IC:0.055✅  Mu:13.6  Mode:Arch-Only  k:50.0
└─Mu: B:56.8✅ F:3.9✅ M:8.3✅ T:29.4✅ m:22.7✅ U:22.3✅
[L1-NESTED] Outer fold 0: registry empty — prequential evidence produced 126 pairs, 0 qualified. Check l1_pair_* thresholds.
[LAYER 1 OUTER FOLDS] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STATUS: ❌ BLOCKED (0/4 Folds Ready)

  [❌] Fold #0 (Fit:2602 → OOS:2848)
       ReadySyms: 0 | Times: 0 | IC:   n/a | Probe: 0.000
       └─ Blockers: empty_opportunities

  [❌] Fold #1 (Fit:3129 → OOS:3506)
       ReadySyms: 1 | Times: 38 | IC:   n/a | Probe: -40.602
       └─ Blockers: insufficient_ready_symbols, non_positive_probe

  [❌] Fold #2 (Fit:3655 → OOS:4164)
       ReadySyms: 8 | Times: 203 | IC: 0.562 | Probe: -74.339
       └─ Blockers: non_positive_probe

  [❌] Fold #3 (Fit:4182 → OOS:4822)
       ReadySyms: 9 | Times: 263 | IC: 0.315 | Probe: 6.827
       └─ Blockers: non_positive_probe
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[LAYER 1 HARD GATE] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STATUS: ❌ BLOCKED (1/5 Passed)

  [✅] Fold-Cov        :   1.000 (>=0.800 )
  [❌] Match-Ratio     :   0.750 (>=1.000 )  ← BLOCKER
  [❌] Sym-Count       :   9.000 (>=17.100)  ← BLOCKER
  [❌] Fold-Ratio      :   0.000 (>=0.600 )  ← BLOCKER
  [❌] Probe-LCB       : -47.261 (>0.000  )  ← BLOCKER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

>> LAYER 1 RESULT: [BLOCKED] -> gate_passed=False
