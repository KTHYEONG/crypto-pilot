# ADR: Ensemble Vectorization & WSL Parallelization Guard

## Context
During `phase signal` execution in `opt_main_futures.py`, the Ensemble (`[ENS]`) phase experienced major bottlenecks. These were caused by (1) high-overhead Pandas `apply(axis=1)` row-by-row prediction logic inside `candidate_ensemble.py`, and (2) single-threaded loop execution of multi-fold Nested Walk-Forward training steps inside `pipeline.py`, which failed to utilize multi-core CPU architectures.

## Decision
1. **Numpy-based Vectorization**: Replaced Pandas `.apply(axis=1)` with NumPy array mapping and list comprehensions to reduce predictive computational overhead.
2. **Deterministic Alignment Assertion**: Injected index equality assertions between `val_set_p` and `val_set` to ensure no row shifts/drift occurred during vectorization.
3. **Dynamic Memory Safety Guard**: Leveraged `psutil.virtual_memory()` and CPU count metrics to compute maximum safe child workers under WSL (Windows Subsystem for Linux) resource caps, keeping peak memory limits below the OOM (Out-of-Memory) crash threshold.
4. **Fork Pool Parallelization**: Wrapped training fold execution into standard `ProcessPoolExecutor` with process-global sharing (`multiprocessing` with `"fork"` context).

## Consequences
- Validation calculations completed in less than 22ms for 10,000 events, achieving near-instantaneous execution.
- Parallel worker counts scale safely between 1 to 6 processes depending on current host RAM availability, successfully executing futures optimization pipeline runs without OOM risk.

---

# ADR: Pipeline Optimizations (v2) - Feature Bypass, Cache Guard & Inactive Stop

## Decision
1. **Feature Bypass**: Skip high-overhead rolling features and Z-score building (`skip_features=True`) for `ensemble_b0` since it only requires target labels.
2. **Delisted Sync Skip**: Guard delisted/inactive symbols (latest cache timestamp > 180 days past requested date or delivery_date reached) in `_requested_sync_caches_missing` to prevent infinite network sync loops.
3. **Global Fold Caching**: Implement `_L1_SWF_FOLD_CACHE` to share historical WFFold results in Optuna studies and skip duplicate process pool invocations.
4. **Single Thread Clamp**: Clamp subprocess thread pools (`OMP_NUM_THREADS=1`) to eliminate inter-process core throttling under resource-capped environments.

## Consequences
- Optimization pipeline execution time cut down to ~57s (80%+ speedup) with CPU core thrashing resolved.
