---
trigger:
  - "src/**/signals/**/*.py"
  - "src/**/sizing/**/*.py"
  - "src/**/regimes/**/*.py"
  - "src/**/opt_*_utils/**/*.py"
  - "src/core/(indicators|optimization)/**/*.py"
  - "src/execution/(opt_main|trader)_*.py"
  - on_file_path_regex: "src/.*(engine|portfolio|metrics|data_collector|backtest).*"
  - on_label: ["quant"]
---
# Quant & Financial Engineering Directives
Priority: 1.Correctness > 2.No look-ahead > 3.Stability > 4.Reproducibility > 5.Trading Realism > 6.Efficiency

## 1. Hard Stop (Fail-Fast)
If critical parameters for 'Correctness' or 'Time-Series Safety' are missing: DO NOT generate code. Output "Task Classification" and explicitly list missing params under "Needs confirmation". Ask for clarification.

## 2. Core Constraints
- Data: Validate shape, dtype, time order. Explicitly handle NaNs/missing. No silent dropping/filling.
- Time-Series (No Leakage): Strict chronological order. Align features/labels/signals explicitly. Default backtest: Signal at `t` executes at `t+1`.
- Math Stability: Block `/0`. Prevent uncontrolled NaN/Inf. Justify `epsilon`.
- Realism: Parameterize commission, slippage, spread, funding fee, sizing.
- Reproducibility: Fix seeds. Explicit params. No hidden global state.

## 3. Performance & Code Quality
- Vectorization: Prefer NumPy/Pandas/Polars.
- Loops: Allowed for orchestration, asset/fold iteration, test cases, real-time events.
- Numba: 
  - NEVER pass DataFrames/Series to Numba. Use `.to_numpy()` explicitly.
  - Use `@njit(cache=True)` only for recursive, path-dependent logic, or proven bottlenecks. 
  - `fastmath=True` requires explicit justification.
- Quality: Explicit type hints/signatures. Validate inputs. Separate calculation from execution.

## 4. Anti-Patterns (Do NOT use unless explicitly justified)
- CPCV / Purged CV (unless overlapping labels)
- Triple-Barrier (unless path-dependent labeling)
- PCA / Cross-sectional normalization (unless large feature set / multi-asset)
- Long math derivations or architecture-level answers for isolated tasks.

## 5. Context-Specific Checks (Apply only if relevant)
- Backtest: Signal shifted? Costs applied on position change? Bounded exposure? Realized returns used?
- ML: Train/test split keeps time order? Scalers fit ONLY on train data?
- Crypto: 24/7 market, UTC, funding fees, missing candles, depeg risk, rate limits.

## 6. Output Modes & Templates
Determine task scope and output STRICTLY using the matching template.

[Mode: Micro] (Indicators, Utils, Snippets)
Task: [Type]
Assumptions: [Minimal]
Code: [Implementation]
Checks: [Edge cases]

[Mode: Standard] (Backtests, Signals, Features)
Task: [Type]
Assumptions: [Key assumptions]
Risks: [Relevant only]
Code: [Implementation]
Verification: [Main checks]

[Mode: Full] (ML, Portfolio, Execution)
Task: [Type, Data, Objective]
Assumptions: [Confirmed / Assumed / Needs confirmation]
Method Choice: [Method & Justification]
Code: [Implementation]
Verification: [Leakage, Stability, Friction, Performance]