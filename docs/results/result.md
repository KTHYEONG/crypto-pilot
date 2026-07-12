# L0/L1 Discovery Snapshot

- Date: `2026-07-11`
- Run id: `4h_1783753822` (post-fix measurement run, `l0_cost_diagnostics_enabled=True`)
- Command:
  `UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. LOG_LEVEL=DEBUG timeout 1500 uv run python src/execution/opt_main_futures.py --phase l1 --sync skip --timeframe 4h --date 2026-07-11 --trials 1 --seed 42`
- Context: This snapshot **supersedes all prior L0/L1 snapshots** in this file's history. Three root-cause fixes landed this session, in dependency order:
  1. `[ADR_20260711_L0_HTF_RESAMPLE_ALIGNMENT_FIX]` — 2h/6h/8h/12h synthetic candles used the wrong resample convention (`closed="right"` instead of open-time `closed="left"`), verified against a live Binance 6h fetch (byte-identical after fix).
  2. `[ADR_20260711_L1_POOLED_ALPHA_ADMISSION_GENERALIZATION]` — L1's peer-exclusive incremental-edge test structurally zeroed out correlated systematic (trend/ts_mom) signals; generalized the existing `xs_alpha`-only pooled-admission bypass to cover these archetypes.
  3. **`[ADR_20260711_L0_NAN_COST_HTF_BLIND_REJECTION]` — the decisive fix.** `AlignedMarketData.execution_cost_bps_2d` defaults to an all-NaN array (not `None`) when a symbol/TF lacks liquidity-cost columns. `has_cost_2d = ... is not None` treated this NaN array as valid data, poisoning `edge_bps` (net) to NaN for every event on non-4h (and some 4h) panels, while `gross_bps` stayed clean. The gate then read `net_lcb_bps=0.0`/`nw_tstat=0.0` (the NaN→0.0 fallback) and **auto-rejected every candidate on every affected TF regardless of true underlying alpha** — a mathematical certainty (`0.0 <= min_lcb_net_bps(0.0)`, `0.0 < min_nw_tstat(1.25)`), not a statistical judgment.

## Why This Snapshot Invalidates Prior Conclusions

Every previous entry in this file's history concluded some variant of "1h/2h/6h/8h/12h show `gross alpha 부재`" (genuine alpha absence). **That conclusion was a false negative.** Those timeframes were never actually evaluated on real net-of-cost economics — they were structurally auto-rejected by the NaN-cost bug before any real signal could be observed. This was proven at the raw evidence-row level (742 diagnostic log lines, `edge_finite=1.000` for 100% of recipes post-fix), not inferred.

## L0 Gate Summary (real per-TF evidence, all three fixes applied)

| TF | Families Evaluated | Families Passed | Evidence Rows | Gate Passed | Selected for L1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1h | 3 | 2 | 8 | 5 | 5 |
| 2h | 4 | 4 | 10 | 10 | 10 |
| 4h | 25 | 12 | 46 | 24 | 18 |
| 6h | 7 | 6 | 19 | 13 | 13 |
| 8h | 7 | 6 | 19 | 13 | 13 |
| 12h | 8 | 7 | 18 | 13 | 13 |
| **Total** | - | - | **120** | **78** | **72** |

**L1 final gate: `PASSED` for the first time in this project's documented history** (8h `n_ready=53`, 12h `n_ready=98`, 2h `n_ready=19`; 4h/6h/1h remain blocked at the L1-nested stage — separate, unresolved mechanism, see below).

## Selected L1 Candidates by Family (net_lcb_bps range per TF)

| TF | Passed families | Representative net_lcb_bps range |
| --- | --- | --- |
| 1h | `trend_ma`, `trend_pullback_continuation` | 36.5 ~ 62.9 |
| 2h | `btc_regime_pullback`, `trend_pullback_continuation`, `trend_ma`, `residual_reversion` | 13.8 ~ 98.9 |
| 4h | 12 distinct families (fullest breadth: momentum, breakout, xs, carry, unwind archetypes) | -18.3 ~ 107.6 |
| 6h | `trend_pullback_continuation`, `btc_regime_pullback`, `trend_ma`, `trend_donchian`, `dual_momentum`, `mtf_breakout_retest` | 7.0 ~ 125.7 |
| 8h | same 6 families as 6h | 17.9 ~ 132.2 |
| 12h | 6h/8h set + `vol_term_structure_gate` | 4.7 ~ 144.3 |

## My Assessment: The Mechanism Works, The Diversity Goal Is Only Half-Met

The L0→L1 funnel *architecture* is sound and now verifiably functional: L0 spends ~285s (of a ~735s total run) reducing the combinatorial search space to 78 candidates before L1's more expensive walk-forward simulation runs. This is the intended "cheap pre-filter, expensive verification only on survivors" design, and it now works end-to-end without crashing or silently zeroing out real signal.

But three structural gaps prevent the *stated goal* — genuinely diverse signals discovered independently across multiple timeframes — from being fully realized:

1. **1h/2h have a structurally narrow search space, not a narrow result.** `_DEFAULT_PER_TF_FAMILIES` gives 1h only 3 families and 2h only 4, versus 25 at 4h. Families that pass strongly at 4h (`taker_imbalance_momentum`, `macd_4h`, `mtf_trend_pullback`) are never even attempted at 1h/2h. This is a config choice, not a gate failure — but it means "L0 found little diversity at 1h/2h" is not yet a fair test of whether diversity exists there.

2. **6h/8h/12h converge on the same 6 trend-following families every time.** This is the same underlying concern already validated empirically in the L1 pooled-admission investigation: these archetypes are highly correlated across time-aggregation levels. 78 "candidates" likely represent far fewer *independent* alpha discoveries than the count suggests — much of it may be one or two trend-following theses re-measured at different bar granularities.

3. **Diversity deduplication (BH-FDR + cross-bucket correlation clustering) only visibly acted on 4h** (`selected_for_l1(18) < gate_passed(24)`). Every other TF shows `selected_for_l1 == gate_passed` exactly — zero redundancy removed. Root cause not yet confirmed: could be legitimately small per-bucket candidate counts falling under the `top_k` threshold, or a wiring gap where HTF virtual-panel evidence bypasses the dedup stage that native-4h evidence goes through. Unresolved.

## Known Remaining Issues (unresolved, explicitly out of scope for the fixes above)

- **L1-nested pairwise stage still blocks 4h/6h/1h** at the outer-fold level (`empty_opportunities`, Symbols=0/Events=0 in most outer folds) — a separate, upstream-of-L1-admission mechanism discovered during the pooled-admission investigation, never root-caused. `[ADR_L0_STRATEGY_DELIVERY_HARDENING]` added locus-tagged blockers (`empty_opportunities:registry_empty` vs `empty_opportunities:prediction_unmatched`, `signal_selection.py`) so the next run can distinguish the two failure loci from logs alone — not yet re-measured.
- **`align_data_maps()`'s cost-diagnostics logging parameter is unwired** in the real pipeline call chain (only `evaluate_panel_gate`'s is wired) — a minor observability gap, not a correctness issue (the underlying fix is unconditional).
- Whether the 72 selected_for_l1 candidates survive realistic walk-forward OOS / cost-stress testing in L1 is **not yet verified** — gate-passed is a necessary, not sufficient, condition for deployability.
- 1h/2h family-pool widening ([LIMIT-05] `l1_ltf_family_pool_widened` config knob) is implemented and unit-tested but **not yet A/B-measured** against the narrow default — still an open experiment, not a decision.

## Cross-TF Independence Audit (2026-07-11, run `4h_1783775628`) — resolves Next-Priority item #3 below

- Command: same as above, with `L0_CROSS_TF_DIVERSITY_AUDIT=1` env flag (new opt-in gate, `enable_cross_tf_diversity_audit` on `AlphaFoundryRuntimeConfig`, wired into `bridge.py`'s multi-TF L0 path).
- Result (`[EVAL] stage=l0_cross_tf_independence_audit`):

  | Metric | Value |
  | --- | ---: |
  | `n_selected_total` | 72 (matches `Selected for L1` total above exactly — wiring integrity confirmed) |
  | `n_distinct_thesis_ids` | 13 (cheap proxy: family→thesis-group mapping) |
  | `n_independent_clusters` | **38** (measured proxy: cross-TF pairwise correlation clustering on the union of all 72 candidates, projected onto the 4h canonical grid) |
  | `n_demoted` | 34 |

- **Answer to the long-standing open question: of the 72 `selected_for_l1` candidates, only 38 (53%) are genuinely independent bets.** The other 34 (47%) are re-measurements of the same underlying thesis at a different bar granularity. This is a **measured confirmation**, not inference, of "My Assessment" point #2 above.
- The demoted set is dominated by `btc_regime_pullback` variants recurring across 2h/4h/6h/8h/12h (e.g. `btc_pullback_50`, `btc_pullback_100_slow`, `btc_pullback_50_rsi` each collapse into a single cross-TF cluster) — the "one or two trend-following theses re-measured at different bar granularities" hypothesis in point #2 is now the confirmed dominant driver, not a hypothesis.
- Implication for L2: portfolio risk-budget diversification assumptions should use 38 as the effective independent-strategy count, not 72.
- This required 4 debugging iterations to get a clean measurement (see decisions log): a logger-visibility gap (module logger not propagating DEBUG in this pipeline), a pre-existing gap where `panels_for_l1` never carried `metadata["recipe_id"]`, and a wrong canonical-TF choice ("finest TF present" instead of the run's actual anchor TF). All three were genuine bugs independent of this measurement's intent, now fixed and regression-tested.

## Cross-TF Pruning Admission (2026-07-11/12, run `4h_1783781808`) — attempted, blocked by a new canonical-TF conflict

- Command: same as above, with `L0_CROSS_TF_DIVERSITY_AUDIT=1 L0_CROSS_TF_PRUNING=1`. Goal: promote the read-only audit above into actual admission — narrow the 72 candidates handed to L1 down toward the measured 38 independent clusters, saving L1 walk-forward compute on the 34 known-redundant ones.
- **Result: pruning did not activate.** `compute_cross_tf_redundancy()`'s own guard (canonical TF must be at least as fine as every input TF, since the forward-fill projection cannot safely downsample fine→coarse) rejected `canonical_tf=4h` because 1h candidates (5 `selected_for_l1` in the L0 Gate Summary above) were present in the union. This is a genuine conflict between two independently-correct prior fixes: (a) canonical TF was pinned to the run's base TF (4h) to avoid a PIT-calendar-window `ValueError` when using an unrelated "finest TF present" choice (see Cross-TF Independence Audit entry above), vs. (b) the redundancy computation's own requirement that canonical be no coarser than any input TF. Neither fix is wrong in isolation; they simply weren't designed against each other.
- **Fail-open worked exactly as designed**: the `ValueError` was caught, logged, and the pipeline fell back to unpruned delivery — L1 results are **byte-identical to the `4h_1783753822` baseline** (`8h n_ready=53`, `12h n_ready=98`, `2h n_ready=19`, `gate_passed=True`), wall-clock 741.22s vs baseline's ~735s (no regression). The read-only audit re-confirmed `n_selected_total=72`, `n_independent_clusters=38` (reproducible).
- **Not yet resolved**: canonical-TF selection needs a redesign that satisfies both constraints simultaneously (e.g. dynamically picking the finest TF actually present in the pruning input, contingent on that TF's PIT calendar window covering all others — needs re-verification against the original date-range bug) — deferred to a follow-up spec, not patched ad hoc.

## L0/L1 Pipeline Latency Profiling (2026-07-12) — 26% wall-clock reduction, one hypothesis disproven

- Command: same as above, with `L0_PARALLEL_MAX_WORKERS=4` (new opt-in gate, `l0_parallel_max_workers` on `AlphaFoundryRuntimeConfig`, Phase-3 cross-TF gate parallelized via fork `ProcessPoolExecutor` + prefork COW cache).
- **Baseline breakdown** (`4h_1783781808`, 741.22s total): universe 2.0s, data load 60.1s, **L0 gate 272.87s (6 TFs fully sequential, zero internal parallelism)**, L1 nested walk-forward 110.97s (6 TFs sequential, but each TF already saturates 8 cores via its own internal `ProcessPoolExecutor`), and a **~283s (38%) completely untimed gap** between data load and the first L0 gate report.
- **Result — speed**: total wall-clock **741.22s → 547.87s (193.35s / 26% reduction)**.
- **Result — correctness**: L1 outputs byte-identical to baseline (`8h n_ready=53`, `12h n_ready=98`, `2h n_ready=19`, `gate_passed=True`) — parallelizing L0 Phase 3 did not change any downstream result, confirming the fixed-seed bootstrap determinism design held under real execution.
- **Result — memory**: peak RSS **16,717MB → 16,396MB** (slightly *lower*, not higher) — the fork-COW prefork cache design avoided the naive per-worker duplication risk that motivated the `[1,4]` worker-count hard cap.
- **New instrumentation added** (`stage=panel_construction`, `stage=tf_probe_scoped`) closed part of the observability gap but **disproved the leading hypothesis from the prior spec**: `tf_probe_scoped` measured 5.75s and `panel_construction` measured 34.04s — combined only ~40s, not the ~283s the panel/indicator-construction hypothesis predicted. **A substantial untimed gap still remains unexplained** even with this run's new instrumentation. Next candidate (unverified): per-TF artifact/evidence logging overhead (`maybe_write_alpha_foundry_report`'s CSV/JSON debug-log formatting, called once per TF for 46+ rows × dozens of columns) — not yet measured.
- **Follow-up** (2026-07-12): `stage=report_write`/`stage=l0_gate_multi_tf_wall` instrumentation added. `report_write` hypothesis **disproved** (0.002-0.004s per TF, negligible). `l0_gate_multi_tf_wall` (direct wall-clock of the entire L0 gate) measured **236.13-236.63s** — split further into `stage=l0_phase1_cheap_evidence` (135.94s, fully sequential) and `stage=l0_phase3_canonical_gate` (100.69s, 4-worker parallel), which sum to the total almost exactly (no hidden third contributor *inside the L0 gate*). Code trace confirmed Phase 1 and Phase 3 called `evaluate_alpha_cheap_gate_batch()` twice per TF with byte-identical inputs — genuine redundant computation, not just unparallelized work. A ~76-80s gap **outside** the L0 gate (universe/data-load/L1/misc bookkeeping) remains reproducible but unattributed — still open.

## L0 Phase-1/Phase-3 Cheap-Gate Deduplication (2026-07-12) — Phase 3 -44%, total -4.6% further

- Command: same as above, `L0_PARALLEL_MAX_WORKERS=4`, with `precomputed_cheap_evidences` threaded from Phase 1 into Phase 3 (skips Phase 3's redundant recomputation).
- **Result**: `stage=l0_phase3_canonical_gate` **100.69s → 56.65s (-44%)**; `stage=l0_gate_multi_tf_wall` **236.63s → 197.93s (-16.4%)**; total wall-clock **523.11s → 498.91s (-4.6%)**. Cumulative vs. this session's original baseline: **741.22s → 498.91s (-32.7%)**. `n_ready`/`gate_passed` byte-identical to every prior baseline.
- **Honest recalibration of the spec's prediction**: the spec projected ~387-410s total, assuming Phase 3's *entire* 100.69s was redundant cheap-gate recomputation. Measured: only 44.04s (44%) of Phase 3's cost was the redundant piece — the remaining 56.65s is genuine canonical-gate/diversity/budget-allocation work that cannot be eliminated this way. The spec's number was a stated upper bound, not a guaranteed outcome, and the gap between prediction and measurement is now on record rather than silently absorbed.
- **User's ~50% total-reduction target (~260-300s) not yet reached** — currently at 498.91s (-32.7% cumulative). Remaining candidate levers, both still unverified: the data-load stage (~48-54s, plausibly I/O-bound/threadable) and the persistent ~76-80s unattributed gap outside the L0 stage.

## Next Priority Candidates

1. **Data-load threading hypothesis** (unverified) — `load_futures_data_maps_for_symbols` (~48-54s) is plausibly I/O-bound (GIL releases during I/O, making `ThreadPoolExecutor` a reasonable candidate) but has no per-symbol timing yet to confirm it isn't CPU-bound (parsing/validation). Measure before implementing, per this session's established discipline.
2. **Find the remaining ~76-80s untimed gap outside the L0 stage** — `panel_construction`/`tf_probe_scoped`/`report_write` hypotheses all disproved; the gap sits in universe/data-load/L1/misc bookkeeping, not yet attributed to a specific function.
3. **Resolve the canonical-TF conflict blocking cross-TF pruning** (2026-07-11/12 finding, below) — needs a spec-level redesign, not a quick patch, since it sits at the intersection of two previously-independent bugfixes.
4. Root-cause why *intra-TF* diversity dedup only fires for 4h (`selected_for_l1(18) < gate_passed(24)` at 4h, `==` everywhere else) — this is a **different, still-open question** from the cross-TF audit above (intra-TF dedup operates per-`(family, timeframe)` bucket within a single TF's own gate call; it never saw other TFs' candidates in the first place, which the new cross-TF audit now also covers going forward). `[LIMIT-04]` diagnostic (`audit_full_family_correlation`, already wired via `enable_correlation_audit`) is the designated next step — determine whether HTF synthetic panels have depressed `valid_mask_2d` overlap vs. native 4h panels before assuming a fix is needed.
5. Decide (not yet A/B-measured) whether to widen 1h/2h's family search pool — `l1_ltf_family_pool_widened=True` vs `False`, same date/seed, compare `docs/results/result.md` deltas.
6. ~~Cross-TF correlation audit on the 78 selected_for_l1 candidates~~ — **done**, 2026-07-11.
7. Resolve the L1-nested `empty_opportunities` outer-fold blocker (4h/6h/1h) — locus-tagged blockers are now wired (see Known Remaining Issues); next run should reveal whether `registry_empty` (upstream nested-pairwise gate) or `prediction_unmatched` (downstream matching) dominates.

## Cross-TF Pruning Admission (2026-07-11/12, run `4h_1783781808`) — attempted, blocked by a new canonical-TF conflict

- Command: same as above, with `L0_CROSS_TF_DIVERSITY_AUDIT=1 L0_CROSS_TF_PRUNING=1`. Goal: promote the read-only audit above into actual admission — narrow the 72 candidates handed to L1 down toward the measured 38 independent clusters, saving L1 walk-forward compute on the 34 known-redundant ones.
- **Result: pruning did not activate.** `compute_cross_tf_redundancy()`'s own guard (canonical TF must be at least as fine as every input TF, since the forward-fill projection cannot safely downsample fine→coarse) rejected `canonical_tf=4h` because 1h candidates (5 `selected_for_l1` in the L0 Gate Summary above) were present in the union. This is a genuine conflict between two independently-correct prior fixes: (a) canonical TF was pinned to the run's base TF (4h) to avoid a PIT-calendar-window `ValueError` when using an unrelated "finest TF present" choice (see Cross-TF Independence Audit entry above), vs. (b) the redundancy computation's own requirement that canonical be no coarser than any input TF. Neither fix is wrong in isolation; they simply weren't designed against each other.
- **Fail-open worked exactly as designed**: the `ValueError` was caught, logged, and the pipeline fell back to unpruned delivery — L1 results are **byte-identical to the `4h_1783753822` baseline** (`8h n_ready=53`, `12h n_ready=98`, `2h n_ready=19`, `gate_passed=True`), wall-clock 741.22s vs baseline's ~735s (no regression). The read-only audit re-confirmed `n_selected_total=72`, `n_independent_clusters=38` (reproducible).
- **Not yet resolved**: canonical-TF selection needs a redesign that satisfies both constraints simultaneously (e.g. dynamically picking the finest TF actually present in the pruning input, contingent on that TF's PIT calendar window covering all others — needs re-verification against the original date-range bug) — deferred to a follow-up spec, not patched ad hoc.
