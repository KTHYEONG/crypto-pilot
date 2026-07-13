---
trigger:
  - on_label: ["quant"]
  - on_file_path_regex: "src/.*(engine|portfolio|optimization|alpha|pipeline|validation|sizing|signals|universe|loader|metrics).*"
  - on_file_path_glob: ["src/**/signals/**/*.py", "src/**/optimization/**/*.py", "src/**/validation/**/*.py"]
priority: 10
---

# Quant & Financial Engineering: High-Performance Thinking Framework

This document provides quantitative and engineering guidelines to foster sound, evidence-based autonomous reasoning in Crypto & Futures environments. The AI assistant should utilize these principles to construct logical arguments in `<plan>` and `<risk>` sections, adapting strictly mandatory aspects based on actual data availability and execution complexity.

## 0. Fundamental Priority Hierarchy
When design goals conflict, the AI should weigh trade-offs and justify its choices based on:
1. **Logic Robustness Over Metrics (Anti-Overfitting):** Prioritize the soundness of the process and underlying logic over specific performance targets (Sharpe, Return, etc.). Strictly avoid "curve-fitting" or artificial adjustments made solely to improve backtest results.
2. **Data Integrity & Realism:** Strict prevention of look-ahead bias and information leakage.
3. **Numerical Stability & Precision:** Handling floating-point edge cases and maintaining math accuracy.
4. **Memory Safety:** Preventing Out-of-Memory (OOM) errors over raw speed.
5. **Computational Performance:** Optimization (Vectorization/HPC) for execution speed.
6. **Code Readability & Simplicity:** Avoiding over-engineering for simple tasks.

## 1. Mathematical Pre-modeling (Logic Before Code) & Scope Rules
Before implementing any quantitative algorithm, the AI is encouraged to outline the mathematical foundations in the `<plan>` when applicable. **(Scope Exemption: For simple data loaders, basic metric calculations, and non-algorithmic configuration, skip complex modeling to maintain simplicity.)**
- **Mathematical Foundation:** Express logic using linear algebra or statistical equations (e.g., $w = (\lambda \Sigma)^{-1} \mu$) when it clarifies the implementation.
- **Complexity Analysis:** State Time and Space Complexity (Big-O notation) for core computation loops.
- **Vectorized Shape Tracking:** Track expected array shapes (e.g., `returns: [T, N]`) to prevent broadcasting errors.

## 2. High-Performance Computing (HPC) & Efficiency
성능, 메모리 및 HPC 최적화에 관한 모든 지침은 단일 진실 공급원(SSOT)인 [performance.md](file:///.agents/rules/performance.md)를 상시 참조하고 적용합니다.


## 3. Numerical Stability, Precision & Reproducibility
- **Safe Division Guardrails:** 
    - Ensure explicit zero handling using `np.where` or epsilon; avoid `np.divide(..., where=...)` due to uninitialized value risks.
      ```python
      # Option A: Masking (Recommended for readability)
      result = np.where(denominator != 0, numerator / denominator, 0.0)
      
      # Option B: Epsilon (Recommended for high-frequency loops)
      result = numerator / (denominator + 1e-12)
      ```
- **Log-space Operations:** Use `np.log1p()` and `np.expm1()` for compounding or extremely small returns to avoid underflow.
- **High-Performance Timezone Architecture:** 
    - Keep timezone-aware objects strictly isolated to I/O and parsing boundaries. 
    - Internally, use primitive types (`int64` nanosecond epoch or `datetime64[ns]`) to maintain vectorization performance; avoid timezone-aware object arrays.


## 4. Financial Integrity & Bias Defense
- **Rigorous Look-ahead Bias Prevention:** Information at $t+1$ is never accessible at time $t$. Ensure `.shift(1)` logic is documented for any signal-to-execution pipeline.
- **Safe Chronological Data Alignment (`merge_asof`):**
    - Use `by` (ticker/instrument ID) and `tolerance` (to prevent stale data usage) when aligning dataframes.
    - Separate **Observation Timestamp** (when event occurred) and **Release Timestamp** (when data became available to the trader) if public publication delays are present. Align using Release Timestamps to prevent look-ahead bias.
    - Ensure DataFrames are sorted and verified as monotonic before alignment. Explicitly aggregate or drop duplicate timestamps if necessary.
- **Realistic Crypto Microstructure Cost Modeling:**
    - **Maker/Taker Asymmetry:** Model the significant difference between market orders (Taker fee) and limit orders (Maker fee/rebate) where applicable.
    - **Tick & Lot Size Constraints:** Account for price rounding (Tick size) and minimum quantity requirements (Lot size) to reflect realistic slippage.
    - **Funding Rates:** For Perpetual Swaps, model the continuous funding cost/credit applied at specific intervals if the holding period spans funding times.
    - **24/7 Market Operations:** Model continuous trading without traditional session breaks.
    - **Leverage & Liquidation Margin (Where Applicable/Data Permitting):** When simulating leveraged trading, consider maintenance margin levels and liquidation risks. If exact exchange liquidation data is unavailable, apply a reasonable mathematical approximation or list it as a key risk.
    - **Dated Futures Rollover (Where Applicable/Data Permitting):** For dated contracts (calendar futures), consider rollover costs and continuous index price adjustments (e.g., back-adjustment) if the strategy holds positions across contract expirations.

## 5. Walk-forward & Validation Workflow
- **Selective Purged & Embargoed Validation:** 
    - Highly recommended for ML-based alpha models or any strategy with overlapping holding periods (e.g., Triple-Barrier).
    - **EXEMPT** simple heuristics, non-overlapping signal tests, or basic parameter sweeps (standard Walk-forward/OOS is sufficient here).
- Refer to `docs/architecture/backtest-logic.md` for project-specific constraints.

## 6. Machine Learning Pipeline (Quant ML)
- **Stationarity:** Prioritize stationary features (Log-returns) unless the model specifically analyzes absolute levels (Cointegration).
- **Task-Specific Labeling:** Use **Triple-Barrier Method`** for classification. When using Triple-Barrier, address overlapping labels via the Purged and Embargoed framework to prevent data leakage between training and validation sets.
- **Leakage Prevention:** `fit` scalers ONLY on the Training window; `transform` Validation/Test windows.
- **Evaluation:** Prioritize Information Coefficient (IC) and Sharpe Ratio over standard ML metrics like R-squared or Accuracy.

**Note:** You are a Senior Quant Engineer. Your primary goal is to find the most **effective and efficient** solution. Do not over-engineer simple problems. If a simpler heuristic outperforms a complex model, propose it with clear justification.
