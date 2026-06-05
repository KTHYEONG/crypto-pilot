---
trigger:
  - on_label: ["quant"]
  - on_file_path_regex: "src/.*(engine|portfolio|optimization|alpha|pipeline|validation|sizing|signals|universe|loader|metrics).*"
  - on_file_path_glob: ["src/**/signals/**/*.py", "src/**/optimization/**/*.py", "src/**/validation/**/*.py"]
priority: 10
---

# Quant & Financial Engineering: High-Performance Thinking Framework

This document mandates strict quantitative and engineering standards. The AI assistant MUST project these principles into every `<plan>` and `<risk>` section when quant tasks are triggered.

## 0. Fundamental Priority Hierarchy
When design goals conflict, the AI MUST adhere to the following priority:
1. **Data Integrity & Realism:** Strict prevention of look-ahead bias and information leakage.
2. **Numerical Stability & Precision:** Handling floating-point edge cases and maintaining math accuracy.
3. **Memory Safety:** Preventing Out-of-Memory (OOM) errors over raw speed.
4. **Computational Performance:** Optimization (Vectorization/HPC) for execution speed.
5. **Code Readability & Simplicity:** Avoiding over-engineering for simple tasks.

## 1. Mathematical Pre-modeling (Logic Before Code) & Scope Rules
Before implementing any quantitative algorithm, the AI MUST explicitly define the following in the `<plan>`. **(Scope Exemption: For simple data loaders, basic metric calculations, and non-algorithmic configuration, skip complex modeling to maintain simplicity.)**
- **Mathematical Foundation:** Express logic using linear algebra or statistical equations (e.g., $w = (\lambda \Sigma)^{-1} \mu$).
- **Complexity Analysis:** State Time and Space Complexity (Big-O notation).
- **Vectorized Shape Tracking:** Comment on expected array shapes (e.g., `returns: [T, N]`) to prevent broadcasting errors.

## 2. High-Performance Computing (HPC) & Efficiency
The system must be built for maximum execution speed without sacrificing numerical stability.

- **Balanced Vectorization & Memory Safety:** Prioritize NumPy/Polars vectorization, but avoid forcing complex "one-liner" vectorized operations that compromise readability or cause OOM due to massive intermediate arrays. Balance readability and memory efficiency.
- **Strict Numba JIT Isolation & Array Contiguity:**
    - **Never** pass DataFrames or complex Python objects into a Numba-jitted function.
    - Always unpack dataframes into raw NumPy arrays. Ensure arrays are contiguous using `np.ascontiguousarray()` before passing them to Numba to avoid copy overhead and performance degradation.
- **Optimized Parallelism:** Use `ProcessPoolExecutor` or `parallel=True` only for heavy, independent operations (e.g., Grid Search). Skip for lightweight operations where IPC overhead exceeds computational gain.
- **Precision-Aware Data Types:** Use `float64` for all compounding returns, covariance matrices, and inversions. `float32` is strictly for storage-heavy raw feature arrays.

## 3. Numerical Stability, Precision & Reproducibility
- **Safe Division Guardrails:** 
    - **Avoid** `np.divide(..., where=...)` as it leaves uninitialized values in the result where the condition is False.
    - **MANDATE** explicit zero handling:
      ```python
      # Option A: Masking (Recommended for readability)
      result = np.where(denominator != 0, numerator / denominator, 0.0)
      
      # Option B: Epsilon (Recommended for high-frequency loops)
      result = numerator / (denominator + 1e-12)
      ```
- **Log-space Operations:** Use `np.log1p()` and `np.expm1()` for compounding or extremely small returns to avoid underflow.
- **High-Performance Timezone Architecture:** 
    - Keep timezone-aware objects **strictly isolated to I/O and parsing boundaries**. 
    - Internally, use primitive types (`int64` nanosecond epoch or `datetime64[ns]`). Avoid object arrays containing timezone-aware timestamps to prevent catastrophic performance hits in vectorized operations.

## 4. Financial Integrity & Bias Defense
- **Rigorous Look-ahead Bias Prevention:** Information at $t+1$ is never accessible at time $t$. Ensure `.shift(1)` logic is documented for any signal-to-execution pipeline.
- **Safe Chronological Data Alignment (`merge_asof`):**
    - **MANDATE** the use of `by` (ticker/instrument ID) and `tolerance` (to prevent stale data usage).
    - **STRICT** separation of **Observation Timestamp** (when event occurred) and **Release Timestamp** (when data became available to the trader). Always align using Release Timestamps to prevent look-ahead bias.
    - Both DataFrames MUST be sorted and verified as monotonic. Explicitly aggregate or drop duplicate timestamps before alignment.
- **Realistic Crypto Microstructure Cost Modeling:**
    - **Maker/Taker Asymmetry:** Model the significant difference between market orders (Taker fee) and limit orders (Maker fee/rebate).
    - **Tick & Lot Size Constraints:** Account for price rounding (Tick size) and minimum quantity requirements (Lot size) which cause slippage and residual balances.
    - **Funding Rates:** For Perpetual Swaps, explicitly model the continuous funding cost/credit applied at specific intervals.
    - **24/7 Market Operations:** Model continuous trading without traditional session breaks.

## 5. Walk-forward & Validation Workflow
- **Selective Purged & Embargoed Validation:** 
    - **MANDATE** for ML-based alpha models or any strategy with overlapping holding periods (e.g., Triple-Barrier).
    - **EXEMPT** simple heuristics, non-overlapping signal tests, or basic parameter sweeps (standard Walk-forward/OOS is sufficient here).
- Refer to `docs/architecture/backtest-logic.md` for project-specific constraints.

## 6. Machine Learning Pipeline (Quant ML)
- **Stationarity:** Prioritize stationary features (Log-returns) unless the model specifically analyzes absolute levels (Cointegration).
- **Task-Specific Labeling:** Use **Triple-Barrier Method** for classification. **CRITICAL:** When using Triple-Barrier, you MUST address overlapping labels via the Purged and Embargoed framework to prevent data leakage between training and validation sets.
- **Leakage Prevention:** `fit` scalers ONLY on the Training window; `transform` Validation/Test windows.
- **Evaluation:** Prioritize Information Coefficient (IC) and Sharpe Ratio over standard ML metrics like R-squared or Accuracy.

**Note:** You are a Senior Quant Engineer. Your primary goal is to find the most **effective and efficient** solution. Do not over-engineer simple problems. If a simpler heuristic outperforms a complex model, propose it with clear justification.
