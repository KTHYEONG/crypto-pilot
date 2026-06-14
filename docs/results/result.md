[SYNC] Ledger up-to-date (Last: 2026-03-31)
discover_universe_timeline: l2_start(2024-10-01) < oos_start(2025-10-01), ignoring l2_start
[UNIVERSE] 🌐 2022-04-01 ~ 2026-03-31 | Target: 94 symbols
[SQL-DB]   💾 Loaded start dates from universe_ledger.db
Sync mode=full targeted_symbols=94
Loaded symbol sync profiles from cache: /home/kth/my_coin_traider/data/futures/symbol_sync_profiles.json
[SQL-DB]   ⚡ [SKIPPED] All symbols are up-to-date. No sync or disk scans required.
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
[ENS] Sym:2   N:2,494  IC:0.013✅  Mu:4.1  Mode:Arch-Only  k:50.0
└─Mu: B:13.4✅ M:4.0✅ T:-5.6❌ m:8.0✅ U:11.2✅
[ENS] Sym:5   N:9,455  IC:0.007✅  Mu:8.7  Mode:Arch-Only  k:50.0
└─Mu: B:20.9✅ M:7.0✅ T:17.4✅ m:10.1✅ U:20.8✅
[ENS] Sym:18  N:29,690  IC:0.007✅  Mu:14.7  Mode:Arch-Only  k:50.0
└─Mu: B:52.7✅ F:4.8✅ M:9.3✅ T:38.5✅ m:24.3✅ U:13.1✅
[ENS] Sym:2   N:1,493  IC:-0.111❌  Mu:7.9  Mode:Arch-Only  k:50.0
└─Mu: B:10.8✅ M:8.2✅ T:0.1✅ m:11.9✅ U:5.8✅
[ENS] Sym:5   N:5,381  IC:0.123✅  Mu:1.7  Mode:Arch-Only  k:50.0
└─Mu: B:35.7✅ M:1.2✅ T:-10.8❌ m:-0.4❌ U:31.9✅
[ENS] Sym:18  N:13,161  IC:0.019✅  Mu:3.0  Mode:Arch-Only  k:50.0
└─Mu: B:15.0✅ M:1.4✅ T:7.3✅ m:9.2✅ U:13.2✅
[ENS] Sym:18  N:33,368  IC:0.055✅  Mu:13.6  Mode:Arch-Only  k:50.0
└─Mu: B:56.8✅ F:3.9✅ M:8.3✅ T:29.4✅ m:22.7✅ U:22.3✅
[L1-NESTED] Outer fold 0: registry empty — prequential evidence produced 0 pairs, 0 qualified. Check l1_pair_* thresholds.
[L1-NESTED] Outer fold 1: registry empty — prequential evidence produced 513 pairs, 0 qualified. Check l1_pair_* thresholds.
[LAYER 1 OUTER FOLDS] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STATUS: ❌ BLOCKED (0/4 Folds Ready)

  [❌] Fold #0 (Fit:2602 → OOS:2848)
       ReadySyms: 0 | Times: 0 | IC: 0.000 | Probe: 0.000
       └─ Blockers: empty_opportunities

  [❌] Fold #1 (Fit:3129 → OOS:3506)
       ReadySyms: 0 | Times: 0 | IC: 0.000 | Probe: 0.000
       └─ Blockers: empty_opportunities

  [❌] Fold #2 (Fit:3655 → OOS:4164)
       ReadySyms: 3 | Times: 78 | IC: 0.000 | Probe: -20.233
       └─ Blockers: non_positive_probe

  [❌] Fold #3 (Fit:4182 → OOS:4822)
       ReadySyms: 4 | Times: 91 | IC: 0.000 | Probe: 5.362
       └─ Blockers: non_positive_probe
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[LAYER 1 HARD GATE] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STATUS: ❌ BLOCKED (1/5 Passed)

  [✅] Fold-Cov        :   1.000 (>=0.800 )
  [❌] Match-Ratio     :   0.500 (>=1.000 )  ← BLOCKER
  [❌] Sym-Count       :   4.000 (>=17.100)  ← BLOCKER
  [❌] Fold-Ratio      :   0.000 (>=0.600 )  ← BLOCKER
  [❌] Probe-LCB       : -34.032 (>0.000  )  ← BLOCKER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

>> LAYER 1 RESULT: [BLOCKED] -> gate_passed=False