# Layer 1 Performance & Memory Profile Report

## # 🎯 Overview
This report documents the performance and memory usage metrics captured during the Layer 1 execution phase (`--phase l1 --sync skip`) with `LOG_LEVEL=DEBUG` on **2026-06-24**. All performance logs have been standardized to use the `[PERF]` prefix under standard `DEBUG` level for structured parsing and profiling.

---

## # 📊 Key Execution Timing Metrics

### 1. Data Processing & Universe Discovery
- **Universe Timeline Discovery**:
  - `[PERF] step=discover_universe_timeline elapsed=1.4140s`
- **Universe Quality Validation**:
  - `[PERF] step=validate_universe_quality elapsed=0.1779s`
- **Futures Data Loading**:
  - `[PERF] step=load_futures_data_maps_for_symbols elapsed=10.3367s`
- **Membership Mask Injection**:
  - `[PERF] step=inject_membership_masks_into_maps elapsed=1.3137s`

### 2. Strategy Calculation (Bridge & Alignment)
- **Candidate Strategy Execution**:
  - `[PERF] bridge_run_candidate_strategy n_syms=52 took=46.7828s`
- **Data Map Alignment**:
  - `[PERF] tiered_align_data_maps n_syms=52 tf=4h took=0.1263s`

### 3. Worker Calculation Specs (Multiprocessing & Resource Limit)
- **Evidence Task Multiprocessing Scheduling**:
  - `[PERF] worker_calc stage=evidence n_tasks=12 requested_workers=12 physical_cores=8 cpu_limit=6 max_workers=3 available_gb=6.74 frame_gb=0.55 estimated_proc_gb=0.87 compact=True workers=3`
- **Outer Task Multiprocessing Scheduling**:
  - `[PERF] worker_calc stage=outer n_tasks=4 requested_workers=4 physical_cores=8 cpu_limit=6 max_workers=3 available_gb=6.74 frame_gb=0.55 estimated_proc_gb=0.87 compact=True workers=2`

### 4. Step-by-Step Layer 1 Diagnostics
- **Nested Volatility 2D Calculation**:
  - `[PERF] l1_nested_volatility_2d took=0.0115s`
- **Nested Feature Cache Prime**:
  - `[PERF] l1_nested_feature_cache_prime took=0.0008s`
- **Nested Events Preparation**:
  - `[PERF] l1_nested_prepare_events took=0.6785s`
- **Nested Multiprocessing Preparation**:
  - `[PERF] l1_nested_mp_prep took=0.0854s`

### 5. Validation Folds Timing Summary
- **Evidence IPC Collection (12 tasks)**:
  - `[PERF] l1_evidence_ipc_collect n=12 took=9.3217s`
- **Outer IPC Collection (4 tasks)**:
  - `[PERF] l1_outer_ipc_collect n=4 took=6.6952s`
- **Avg Fold Processing Breakdown (n=12)**:
  - `[PERF] l1_evidence_fold_avg_profile schema=0.132s ds_fit=1.226s ds_es=0.000s ds_oos=0.100s edge_fit=0.171s inference=0.038s selection=0.113s`
- **Walk-Forward Avg Profile (n=16 folds)**:
  - `[PERF] l1_wf_summary wall: ev=24.2s out=6.8s total=31.0s avg: selection=0.223s ds_fit=1.330s schema=0.123s edge_fit=0.196s inference=0.049s`

---

## # 💾 Memory usage & Total Overhead

- **Total Execution Pipeline Time (L1 Total)**:
  - `[PERF] run_tiered_pipeline_l1_total took=154.6907s`
- **Timeframe Processing Summary (tf=12h)**:
  - `[PERF] per_tf_l1 tf=12h aligned=0.0000s folds=0.0003s run_l1=35.3987s total=35.3990s rss=5828MB`
- **Timeframe Processing Summary (tf=4h)**:
  - `[PERF] per_tf_l1 tf=4h aligned=0.0000s folds=0.0003s run_l1=48.5829s total=48.5832s rss=5721MB`
- **End Stage RAM Footprint**:
  - `[SYS] [MEM] stage=aggregate_l1 rss=5828MB`
  - `[SYS] [MEM] stage=l1_gate_complete rss=5828MB`
  - `[SYS] [MEM] stage=tiered_pipeline rss=5828MB delta=+324MB peak=7681MB`

---

## # 🚨 Top 5 Bottlenecks

Based on the execution time of individual operations, the top 5 performance bottlenecks are:

1. **`bridge_run_candidate_strategy`** (46.7828s)
   - *Description*: Execution of backtests for 52 candidate strategies. This is the single heaviest logic stage and has the highest optimization priority (e.g., caching or vectorization of evaluations).
2. **`l1_fit_inference_artifact`** (19.6176s)
   - *Description*: Fitting models and generating inference serialized artifacts for the promoted signals.
3. **`load_futures_data_maps_for_symbols`** (10.3367s)
   - *Description*: Database loading and parsing of OHLCV/orderbook historical tables for the active symbols.
4. **`l1_evidence_ipc_collect`** (9.3217s)
   - *Description*: Inter-process communication overhead and serialization/deserialization when collecting results from 12 parallel task processes.
5. **`l1_prequential_evidence_snapshots`** (7.8279s)
   - *Description*: Accumulating sequential snapshots and saving evidence registry metadata to disk.

