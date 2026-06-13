discover_universe_timeline: l2_start(2024-10-01) < oos_start(2025-10-01), ignoring l2_start
[CACHE] Backfill: 2022-04-01 ~ 2026-03-31 | Symbols: 94 | Last: 2026-04-01
Sync mode=full targeted_symbols=94
Updated symbol sync profiles cache: /home/kth/my_coin_traider/data/futures/symbol_sync_profiles.json
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
[DATA-INTEGRITY] Starting market data integrity check for 56 symbols...
[DATA-INTEGRITY] PASS: 56/56 symbols passed. (Bars: 7518, NaN: 0.0%, Zero/Neg: 0.0%, Hi>=Lo: PASS)
[BRIDGE-PROF] total=17.6884s align=0.1055s rules=2.8839s events=1.5969s label=7.0333s diagnostics=2.5190s promotions=0.1549s walk_forward=0.0000s post_wf=0.0000s selection=0.0000s weights=0.0000s alpha_panel=3.2954s accounted=17.5891s unaccounted=0.0993s
[TIERED] aligned scope: 56 symbols (historical union ∩ data-valid)
[WORKFLOW] Fold 0 skipped Ensemble (fit=0 < 2)
[ENSEMBLE] ActiveSyms(2) | N: 824 | IC: 0.1359 ✅ | Mu: 10.22 | Mode: Archetype-Regime | k: 50.0
└─ Mu(bps): [Beta:11.7✅, Mean:6.8✅, Trnd:13.9✅, Mom:31.8✅, Unwnd:10.6✅] | ScoreCal:0/3 valid (LowObs:0, Fail:3)
[ENSEMBLE] ActiveSyms(2) | N: 1,685 | IC: 0.2194 ✅ | Mu: 8.05 | Mode: Archetype-Regime | k: 50.0
└─ Mu(bps): [Beta:17.3✅, Mean:8.3✅, Trnd:4.2✅, Mom:5.8✅, Unwnd:5.9✅] | ScoreCal:0/4 valid (LowObs:0, Fail:4)
[ENSEMBLE] ActiveSyms(2) | N: 1,493 | IC: 0.3053 ✅ | Mu: 7.90 | Mode: Archetype-Regime | k: 50.0
└─ Mu(bps): [Beta:10.8✅, Mean:8.2✅, Trnd:0.1✅, Mom:11.9✅, Unwnd:5.8✅] | ScoreCal:0/4 valid (LowObs:0, Fail:4)
[ENSEMBLE] ActiveSyms(2) | N: 828 | IC: 0.1369 ✅ | Mu: 9.73 | Mode: Archetype-Regime | k: 50.0
└─ Mu(bps): [Beta:11.4✅, Mean:6.5✅, Trnd:11.6✅, Mom:31.5✅, Unwnd:10.2✅] | ScoreCal:0/3 valid (LowObs:0, Fail:3)
[ENSEMBLE] ActiveSyms(2) | N: 2,494 | IC: 0.0948 ✅ | Mu: 4.10 | Mode: Archetype-Regime | k: 50.0
└─ Mu(bps): [Beta:13.4✅, Mean:4.0✅, Trnd:-5.6❌, Mom:8.0✅, Unwnd:11.2✅] | ScoreCal:0/5 valid (LowObs:0, Fail:5)
[ENSEMBLE] ActiveSyms(5) | N: 5,440 | IC: -0.0527 ❌ | Mu: 2.18 | Mode: Archetype-Regime | k: 50.0
└─ Mu(bps): [Beta:34.3✅, Mean:1.7✅, Trnd:-9.5❌, Mom:-0.1❌, Unwnd:31.5✅] | ScoreCal:0/6 valid (LowObs:0, Fail:6)
[ENSEMBLE] ActiveSyms(5) | N: 5,381 | IC: -0.0461 ❌ | Mu: 1.74 | Mode: Archetype-Regime | k: 50.0
└─ Mu(bps): [Beta:35.7✅, Mean:1.2✅, Trnd:-10.8❌, Mom:-0.4❌, Unwnd:31.9✅] | ScoreCal:0/6 valid (LowObs:0, Fail:6)
[ENSEMBLE] ActiveSyms(2) | N: 1,685 | IC: 0.2194 ✅ | Mu: 8.05 | Mode: Archetype-Regime | k: 50.0
└─ Mu(bps): [Beta:17.3✅, Mean:8.3✅, Trnd:4.2✅, Mom:5.8✅, Unwnd:5.9✅] | ScoreCal:0/4 valid (LowObs:0, Fail:4)
[ENSEMBLE] ActiveSyms(5) | N: 5,431 | IC: -0.0502 ❌ | Mu: 2.09 | Mode: Archetype-Regime | k: 50.0
└─ Mu(bps): [Beta:34.3✅, Mean:1.5✅, Trnd:-9.2❌, Mom:0.2✅, Unwnd:31.4✅] | ScoreCal:0/6 valid (LowObs:0, Fail:6)
[ENSEMBLE] ActiveSyms(18) | N: 11,897 | IC: 0.0584 ✅ | Mu: 4.43 | Mode: Archetype-Regime | k: 50.0
└─ Mu(bps): [Beta:20.2✅, Mean:2.9✅, Trnd:11.5✅, Mom:5.8✅, Unwnd:14.3✅] | ScoreCal:2/6 valid (LowObs:0, Fail:4)
[ENSEMBLE] ActiveSyms(18) | N: 13,161 | IC: -0.0065 ❌ | Mu: 2.98 | Mode: Archetype-Only | k: 50.0
└─ Mu(bps): [Beta:15.0✅, Mean:1.4✅, Trnd:7.3✅, Mom:9.2✅, Unwnd:13.2✅] | ScoreCal:2/6 valid (LowObs:0, Fail:4)
[ENSEMBLE] ActiveSyms(2) | N: 2,494 | IC: 0.0948 ✅ | Mu: 4.10 | Mode: Archetype-Regime | k: 50.0
└─ Mu(bps): [Beta:13.4✅, Mean:4.0✅, Trnd:-5.6❌, Mom:8.0✅, Unwnd:11.2✅] | ScoreCal:0/5 valid (LowObs:0, Fail:5)
[ENSEMBLE] ActiveSyms(5) | N: 9,455 | IC: 0.0041 ✅ | Mu: 8.66 | Mode: Archetype-Regime | k: 50.0
└─ Mu(bps): [Beta:20.9✅, Mean:7.0✅, Trnd:17.4✅, Mom:10.1✅, Unwnd:20.8✅] | ScoreCal:2/6 valid (LowObs:0, Fail:4)
[ENSEMBLE] ActiveSyms(18) | N: 29,690 | IC: 0.0684 ✅ | Mu: 14.75 | Mode: Archetype-Regime | k: 50.0
└─ Mu(bps): [Beta:52.7✅, Flow:4.8✅, Mean:9.3✅, Trnd:38.5✅, Mom:24.3✅, Unwnd:13.1✅] | ScoreCal:3/6 valid (LowObs:0, Fail:3)
[ENSEMBLE] ActiveSyms(18) | N: 33,368 | IC: 0.0746 ✅ | Mu: 13.56 | Mode: Archetype-Only | k: 50.0
└─ Mu(bps): [Beta:56.8✅, Flow:3.9✅, Mean:8.3✅, Trnd:29.4✅, Mom:22.7✅, Unwnd:22.3✅] | ScoreCal:3/6 valid (LowObs:0, Fail:3)
[LAYER 1 OUTER FOLDS] ------------------------------
| Fold | Registry Source End | Outer Start | Ready Symbols | Times | IC     | Probe  | Status |
| ---- | ------------------- | ----------- | ------------- | ----- | ------ | ------ | ------ |
| 0    | 2602                | 2848        | 0             | 0     | 0.000  | 0.000  | FAIL   |
| 0    | 3129                | 3506        | 0             | 0     | 0.000  | 0.000  | FAIL   |
| 4164 | 3655                | 4164        | 1             | 0     | 0.000  | 0.000  | FAIL   |
| 4822 | 4182                | 4822        | 1             | 0     | 0.000  | 0.000  | FAIL   |
------------------------------------------------------
[LAYER 1 HARD GATE] --------------------------------
| Gate                        | Value   | Threshold | Status  | Blocker |
| --------------------------- | ------- | --------- | ------- | ------- |
| Fold-Cov                    |   1.000 | >=0.800   | PASS    | -       |
| Sym-Count                   |   1.000 | >=6.000   | FAIL    | 1.000   |
| Sym-Ratio                   |   0.018 | >=0.300   | FAIL    | 0.018   |
| Fold-Ratio                  |   0.000 | >=0.600   | FAIL    | 0.000   |
| Opp-IC                      |   0.000 | >=0.020   | FAIL    | 0.000   |
| Opp-Tstat                   |   0.000 | >=1.960   | FAIL    | 0.000   |
| Probe-bps                   |   0.000 | >0.000    | FAIL    | 0.000   |
| Probe-Tstat                 |   0.000 | >=1.960   | FAIL    | 0.000   |
| Layer1 Gate                 | -       | ALL       | BLOCKED | Sym-Count:1.000; Sym-Ratio:0.018; Fold-Ratio:0.000; Opp-IC:0.000; Opp-Tstat:0.000; Probe-bps:0.000; Probe-Tstat:0.000 |
------------------------------------------------------

>> LAYER 1 RESULT: [BLOCKED] -> gate_passed=False