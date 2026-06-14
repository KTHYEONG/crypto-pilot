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
[ENS] Sym:2   N:233  IC:-0.047❌  Mu:36.6  Mode:Arch-Only  k:50.0
└─Mu: B:28.0✅ M:30.8✅ T:118.5✅ m:10.9✅ U:8.4✅
[ENS] Sym:2   N:2,355  IC:0.084✅  Mu:6.3  Mode:Arch-Only  k:50.0
└─Mu: B:14.3✅ M:6.3✅ T:-3.5❌ m:10.7✅ U:10.8✅
[ENS] Sym:2   N:1,290  IC:-0.061❌  Mu:6.8  Mode:Arch-Only  k:50.0
└─Mu: B:10.1✅ M:5.8✅ T:-1.7❌ m:23.2✅ U:6.7✅
[ENS] Sym:5   N:3,586  IC:0.093✅  Mu:11.8  Mode:Arch-Only  k:50.0
└─Mu: B:41.7✅ M:10.1✅ T:7.9✅ m:11.3✅ U:48.4✅
[ENS] Sym:5   N:6,226  IC:0.012✅  Mu:2.2  Mode:Arch-Only  k:50.0
└─Mu: B:32.1✅ M:1.2✅ T:-6.9❌ m:3.9✅ U:30.9✅
[ENS] Sym:5   N:8,792  IC:-0.004❌  Mu:9.3  Mode:Arch-Only  k:50.0
└─Mu: B:18.9✅ M:7.7✅ T:21.0✅ m:8.2✅ U:20.2✅
[ENS] Sym:18  N:11,760  IC:-0.030❌  Mu:5.0  Mode:Arch-Only  k:50.0
└─Mu: B:20.1✅ M:3.4✅ T:13.1✅ m:6.4✅ U:14.2✅
[ENS] Sym:18  N:20,242  IC:-0.049❌  Mu:13.9  Mode:Arch-Only  k:50.0
└─Mu: B:61.6✅ F:1.1✅ M:9.3✅ T:45.6✅ m:9.3✅ U:10.7✅
[ENS] Sym:18  N:27,380  IC:0.013✅  Mu:14.1  Mode:Arch-Only  k:50.0
└─Mu: B:52.3✅ F:3.9✅ M:8.9✅ T:39.3✅ m:21.6✅ U:12.9✅
[ENS] Sym:27  N:33,800  IC:0.042✅  Mu:13.5  Mode:Arch-Only  k:50.0
└─Mu: B:51.3✅ F:4.0✅ M:8.6✅ T:30.2✅ m:20.4✅ U:22.0✅
[ENS] Sym:33  N:38,947  IC:0.001✅  Mu:14.7  Mode:Arch-Only  k:50.0
└─Mu: B:47.6✅ F:5.0✅ M:10.4✅ T:28.2✅ m:20.2✅ U:22.9✅
[ENS] Sym:33  N:45,345  IC:-0.004❌  Mu:15.3  Mode:Arch-Only  k:50.0
└─Mu: B:55.7✅ F:6.2✅ M:10.2✅ T:38.9✅ m:19.4✅ U:7.5✅
[ENS] Sym:2   N:3,153  IC:0.042✅  Mu:5.2  Mode:Arch-Only  k:50.0
└─Mu: B:16.2✅ M:4.7✅ T:0.1✅ m:7.3✅ U:17.9✅
[ENS] Sym:5   N:8,420  IC:-0.102❌  Mu:9.5  Mode:Arch-Only  k:50.0
└─Mu: B:22.7✅ M:7.8✅ T:19.8✅ m:10.2✅ U:19.7✅
[ENS] Sym:18  N:20,170  IC:-0.041❌  Mu:14.0  Mode:Arch-Only  k:50.0
└─Mu: B:61.5✅ F:1.2✅ M:9.4✅ T:45.8✅ m:9.9✅ U:10.5✅
[ENS] Sym:33  N:33,888  IC:0.042✅  Mu:13.2  Mode:Arch-Only  k:50.0
└─Mu: B:51.3✅ F:3.9✅ M:8.6✅ T:28.5✅ m:18.4✅ U:22.0✅
[LAYER 1 OUTER FOLDS] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STATUS: ✅ READY (2/4 Folds Ready)

  [✅] Fold #0 (Fit:2954 → OOS:3288)
       ReadySyms: 3 | Times: 149 | IC:   n/a | Probe: 52.041

  [❌] Fold #1 (Fit:3394 → OOS:3837)
       ReadySyms: 4 | Times: 117 | IC:   n/a | Probe: 6.730
       └─ Blockers: non_positive_probe

  [✅] Fold #2 (Fit:3833 → OOS:4386)
       ReadySyms: 4 | Times: 85 | IC: 0.463 | Probe: 203.830

  [❌] Fold #3 (Fit:4272 → OOS:4935)
       ReadySyms: 5 | Times: 145 | IC:   n/a | Probe: 20.335
       └─ Blockers: non_positive_probe
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[LAYER 1 HARD GATE] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STATUS: ❌ BLOCKED (2/5 Passed)

  [✅] Fold-Cov        :   1.000 (>=0.800 )
  [✅] Match-Ratio     :   1.000 (>=1.000 )
  [❌] Sym-Count       :   5.000 (>=17.100)  ← BLOCKER
  [❌] Fold-Ratio      :   0.500 (>=0.600 )  ← BLOCKER
  [❌] Probe-LCB       :  -1.681 (>0.000  )  ← BLOCKER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

>> LAYER 1 RESULT: [BLOCKED] -> gate_passed=False
