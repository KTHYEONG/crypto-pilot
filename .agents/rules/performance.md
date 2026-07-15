---
trigger: glob
priority: 10
---

# HPC & Memory Management Directives (WSL & GPU Optimized Quant)

This document defines the physical hardware constraints and performance guidelines to maximize calculation speed and memory efficiency under the WSL2 and GPU virtualized environment.

---

## 1. Physical Resource Budget & Constraints (WSL & GPU Budget)

- **Available CPU**: 8 Processors (WSL Allocated)
  - **Directive**: Limit the `max_workers` configuration for parallel execution (`ProcessPoolExecutor`, `multiprocessing`) to **4~6**. Utilizing all 8 cores causes high context-switching overhead and can interrupt WSL virtual network/device communications.
- **Available RAM**: 18GB Physical Memory (20GB Swap)
  - **Directive**: Entering the Swap memory region drastically drops backtesting performance due to disk I/O bottlenecks. The maximum Resident Set Size (RSS) memory of any executing process must not exceed **12GB**.
- **Available GPU**: NVIDIA GeForce RTX 4070Ti (12GB VRAM)
  - **Directive**: WSL2 CUDA acceleration is fully supported. To prevent host OS memory paging and OOM crashes, limit active VRAM allocations to **8~9GB**. Manually clear GPU cache (e.g., `torch.cuda.empty_cache()`) or delete model variables immediately after heavy training/inference loops.

---

## 2. Memory Safety Guardrails

- **float32 Downcasting**:
  - Cast raw feature arrays, indicators, and large matrices from `float64` to `float32` during intermediate data steps to save 50% memory.
  - Keep `float64` strictly for compounding returns, covariance matrix inversions, and high-precision statistical metrics.
- **No Unnecessary Panel Deepcopy**:
  - Do not use `.copy(deep=True)` on Pandas DataFrames or NumPy ndarrays unless absolutely necessary. Use inplace modification or view slices.
- **Manual Garbage Collection (GC)**:
  - Delete large temporary panel instances (`del obj`) and manually invoke `gc.collect()` immediately after transition steps (e.g., TF transitions, fold completion) to return memory to the WSL kernel.
- **Prohibit Concurrency on Heavy Optimization Engines**:
  - The execution of [opt_main_futures.py](file:///home/kth/my_coin_traider/src/execution/opt_main_futures.py) itself consumes extreme memory. To prevent Out-Of-Memory (OOM) crashes, this script MUST NEVER be executed in parallel or concurrently (must run as a single process).

---

## 3. Execution Speed & HPC Optimization

- **No Pandas Loop (.iloc, .iterrows)**:
  - Row-by-row iteration on Pandas DataFrames is strictly prohibited. Use vectorized NumPy operations or Polars expressions to run calculations in C-level performance.
- **Numba JIT & Array Contiguity**:
  - Always unpack DataFrames or complex Python objects into raw NumPy ndarrays before passing them to Numba-jitted functions.
  - Ensure all arrays passed to `@njit` are memory contiguous. Call `np.ascontiguousarray(arr)` on sliced or resampled views to avoid memory copying overhead and L1/L2 cache misses.
  - **Enable Compilation Caching**: Always specify `@njit(cache=True)` for Numba-jitted functions to prevent startup overhead and compilation redundancy on every execution.
- **Optimized Parallelism**:
  - Restrict the use of `ProcessPoolExecutor` or `parallel=True` only to heavy, independent tasks (e.g., large parameter grid sweeps). Prohibit multiprocessing for lightweight operations where IPC (Inter-Process Communication) overhead exceeds math execution gains.
- **Early-Exit Architecture**:
  - Instantly reject failed hypotheses in the cheap gate (e.g., L0 Cheap Gate) to skip expensive downstream math operations (e.g., Bootstrap, Triple-barrier labels).

---

## 4. Testing & Verification Rules

- **Performance Regression Guard**:
  - When running `lean_check.py`, if the overall execution time increases by **15% or more** compared to the baseline, or RSS memory crosses the **12GB threshold**, the AI must halt development and optimize the time/space complexity (Big-O) of the code.
