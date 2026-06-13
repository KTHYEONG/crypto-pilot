discover_universe_timeline: l2_start(2024-10-01) < oos_start(2025-10-01), ignoring l2_start
[CACHE] Backfill: 2022-04-01 ~ 2026-03-31 | Symbols: 94 | Last: 2026-04-01
Sync mode=full targeted_symbols=94
Loaded symbol sync profiles from cache: /home/kth/my_coin_traider/data/futures/symbol_sync_profiles.json
[SYNC-COVERAGE] rows=36 file=/home/kth/my_coin_traider/logs/futures/universe/sync_coverage_report.parquet
Ledger update complete.
┌──────────────────────────────────────────────────────────────────────────────┐
│ [SYSTEM CONTEXT: INFRASTRUCTURE & DATA PREPARATION]                          │
│ ────────────────────────────────────────────────────────────────────────────── │
│ Window:   Range: 2022-10-01 ~ 2026-03-31 (IS:2023-10-01, OOS:2025-10-01)     │
│ Universe: Discovered: 94 symbols | Selected: 20 | Live Panel: 20             │
│ Quality:  Loaded: 61.7% (58/94) | Ready: 58 | Dropped: None                  │
│ Strategy: Engine: Alpha-Ensemble Engine | Inf Panel: 94 | Trade Scope: 58    │
└──────────────────────────────────────────────────────────────────────────────┘

================================================================================
[LAYER 1: SIGNAL ROBUSTNESS & ENSEMBLE VERIFICATION]
================================================================================
[DATA-INTEGRITY] 💠 Starting audit for 56 symbols...
[DATA-INTEGRITY] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STATUS: ✅ ALL 56 SYMBOLS PASSED (Bars: 7,518)
  AUDIT:  [NaN: 0.0%] [Zero/Neg: 0.0%] [Hi>=Lo: PASS]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[TIERED] 💠 Scope: 56 symbols (Historical Union ∩ Data-Valid)
[WORKFLOW] Fold 0 skipped Ensemble (fit=0 < 2)
[ENS] Sym:2   N:824  IC:-0.357❌  Mu:10.2  Mode:Arch-Only  k:50.0
└─Mu: B:11.7✅ M:6.8✅ T:13.9✅ m:31.8✅ U:10.6✅
[ENS] Sym:2   N:1,685  IC:-0.066❌  Mu:8.0  Mode:Arch-Only  k:50.0
└─Mu: B:17.3✅ M:8.3✅ T:4.2✅ m:5.8✅ U:5.9✅
[ENS] Sym:2   N:1,493  IC:-0.111❌  Mu:7.9  Mode:Arch-Only  k:50.0
└─Mu: B:10.8✅ M:8.2✅ T:0.1✅ m:11.9✅ U:5.8✅
[ENS] Sym:2   N:828  IC:-0.372❌  Mu:9.7  Mode:Arch-Only  k:50.0
└─Mu: B:11.4✅ M:6.5✅ T:11.6✅ m:31.5✅ U:10.2✅
[ENS] Sym:2   N:2,494  IC:0.013✅  Mu:4.1  Mode:Arch-Only  k:50.0
└─Mu: B:13.4✅ M:4.0✅ T:-5.6❌ m:8.0✅ U:11.2✅
[ENS] Sym:5   N:5,440  IC:0.085✅  Mu:2.2  Mode:Arch-Only  k:50.0
└─Mu: B:34.3✅ M:1.7✅ T:-9.5❌ m:-0.1❌ U:31.5✅
[ENS] Sym:5   N:5,381  IC:0.123✅  Mu:1.7  Mode:Arch-Only  k:50.0
└─Mu: B:35.7✅ M:1.2✅ T:-10.8❌ m:-0.4❌ U:31.9✅
[ENS] Sym:2   N:1,685  IC:-0.066❌  Mu:8.0  Mode:Arch-Only  k:50.0
└─Mu: B:17.3✅ M:8.3✅ T:4.2✅ m:5.8✅ U:5.9✅
[ENS] Sym:5   N:5,431  IC:0.090✅  Mu:2.1  Mode:Arch-Only  k:50.0
└─Mu: B:34.3✅ M:1.5✅ T:-9.2❌ m:0.2✅ U:31.4✅
[ENS] Sym:18  N:11,897  IC:-0.008❌  Mu:4.4  Mode:Arch-Only  k:50.0
└─Mu: B:20.2✅ M:2.9✅ T:11.5✅ m:5.8✅ U:14.3✅
[ENS] Sym:18  N:13,161  IC:0.019✅  Mu:3.0  Mode:Arch-Only  k:50.0
└─Mu: B:15.0✅ M:1.4✅ T:7.3✅ m:9.2✅ U:13.2✅
[ENS] Sym:2   N:2,494  IC:0.013✅  Mu:4.1  Mode:Arch-Only  k:50.0
└─Mu: B:13.4✅ M:4.0✅ T:-5.6❌ m:8.0✅ U:11.2✅
[ENS] Sym:5   N:9,455  IC:0.007✅  Mu:8.7  Mode:Arch-Only  k:50.0
└─Mu: B:20.9✅ M:7.0✅ T:17.4✅ m:10.1✅ U:20.8✅
[ENS] Sym:18  N:29,690  IC:0.007✅  Mu:14.7  Mode:Arch-Only  k:50.0
└─Mu: B:52.7✅ F:4.8✅ M:9.3✅ T:38.5✅ m:24.3✅ U:13.1✅
[ENS] Sym:18  N:33,368  IC:0.055✅  Mu:13.6  Mode:Arch-Only  k:50.0
└─Mu: B:56.8✅ F:3.9✅ M:8.3✅ T:29.4✅ m:22.7✅ U:22.3✅
[LAYER 1 OUTER FOLDS] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STATUS: ❌ BLOCKED (0/4 Folds Ready)

  [❌] Fold #0 (Fit:2602 → OOS:2848)
       ReadySyms: 2 | Times: 0 | IC: 0.000 | Probe: 0.000
       └─ Blockers: insufficient_opportunity_timestamps, non_positive_probe

  [❌] Fold #1 (Fit:3129 → OOS:3506)
       ReadySyms: 2 | Times: 0 | IC: 0.000 | Probe: 0.000
       └─ Blockers: insufficient_opportunity_timestamps, non_positive_probe

  [❌] Fold #2 (Fit:3655 → OOS:4164)
       ReadySyms: 3 | Times: 0 | IC: 0.000 | Probe: 0.000
       └─ Blockers: insufficient_opportunity_timestamps, non_positive_probe

  [❌] Fold #3 (Fit:4182 → OOS:4822)
       ReadySyms: 11 | Times: 0 | IC: 0.000 | Probe: 0.000
       └─ Blockers: insufficient_opportunity_timestamps, non_positive_probe
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[LAYER 1 HARD GATE] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STATUS: ❌ BLOCKED (1/8 Passed)

  [✅] Fold-Cov        :   1.000 (>=0.800 )
  [❌] Sym-Count       :   3.000 (>=6.000 )  ← BLOCKER
  [❌] Sym-Ratio       :   0.054 (>=0.300 )  ← BLOCKER
  [❌] Fold-Ratio      :   0.000 (>=0.600 )  ← BLOCKER
  [❌] Opp-IC          :   0.000 (>=0.020 )  ← BLOCKER
  [❌] Opp-Tstat       :   0.000 (>=1.960 )  ← BLOCKER
  [❌] Probe-bps       :   0.000 (>0.000  )  ← BLOCKER
  [❌] Probe-Tstat     :   0.000 (>=1.960 )  ← BLOCKER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

>> LAYER 1 RESULT: [BLOCKED] -> gate_passed=False
