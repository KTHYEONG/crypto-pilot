---
trigger: glob
priority: 10
---

# Performance & Optimization Directives (Measurement-Driven)

This document defines performance optimization guidelines focused on empirical benchmarks rather than static hardware mandates.

---

## 1. Measurement & Bottleneck Philosophy
- **Correctness First:** Prioritize code correctness and algorithmic soundness; optimize measured bottlenecks only.
- **Benchmark Driven:** Establish a benchmark before and after optimization to prove gains.
- **Hardware Config Reference:** Respect system hardware limits configured in `src/core/settings.py` (`HARDWARE_MAX_WORKERS`, `HARDWARE_MAX_RSS_RAM_GB`, `HARDWARE_MAX_VRAM_GB`).
- **Dynamic Resource Scaling:** Determine worker count, process pools, and batch sizes dynamically based on workload, memory footprint, and measured scaling.

---

## 2. Memory & Precision Optimization
- **Precision Validation:** Use `float32` for large arrays or intermediate feature matrices only after numerical-error validation. Retain `float64` for sensitive matrix inversions or compounding returns.
- **Memory Footprint Management:** Prefer in-place operations or view slices for large arrays. Avoid unnecessary deep copying.
- **Targeted Memory Releases:** Invoke explicit garbage collection (`gc.collect()`) or CUDA cache clearing (`torch.cuda.empty_cache()`) only when profiling indicates retained-memory pressure.

---

## 3. High-Performance Execution & Parallelism
- **Vectorization vs Loops:** Prefer vectorization (NumPy/Polars) for large hot-path computations. Allow standard Python loops for control-flow or lightweight tasks where vectorization overhead exceeds benefits.
- **Numba JIT Strategy:** Pass memory-contiguous arrays to JIT functions (`np.ascontiguousarray`). Use JIT caching when functions are repeatedly compiled across runs.
- **Measured Parallelization:** Restrict process pool execution to heavy tasks where task computation time significantly outweighs inter-process communication (IPC) overhead.

---

## 4. Performance Regression & Stability
- **Variance Tolerance:** Evaluate performance regressions using stable, repeatable benchmarks with realistic variance tolerances.
