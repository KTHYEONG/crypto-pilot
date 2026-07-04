---
trigger:
  - on_file_path_regex: "src/.*\\.py"
  - on_file_path_regex: "tests/.*\\.py"
priority: 9
---

# Unified Logging & Tagging Directives (AI-Optimized)

This document defines the strict logging rules and tag-based format requirements to optimize debugging efficiency, ensure token economy, and enable systematic grep-based log parsing.

---

## 1. Core Principles: AI-Reading Optimization

- **No Verbose Prose**: Log messages must omit conversational sentences (e.g., "Successfully started calculation for symbol"). Instead, output key-value structures.
- **Strictly Flat & Parsable**: Every `DEBUG`/`TRACE` level log must be optimized for direct programmatic extraction or regex parsing.
- **Categorized Isolation (Preferred over Unified Log)**:
  - High-frequency data (e.g., raw signal outputs, optimization trials) MUST be routed to dedicated, isolated files (e.g., `logs/optuna.jsonl`, `logs/memory.log`) instead of clogging the main system log.
  - This allows the AI to target precise file paths, saving context window space and token usage.

---

## 2. Standard Logging Levels & Outputs

### 2.1 INFO (Terminal-Clean Output)
- **Purpose**: Minimal progress reporting for humans.
- **Constraints**: 
  - Keep logs under 1 line per major phase transition.
  - Avoid outputting massive collections, lists, or matrix arrays.

### 2.2 DEBUG & TRACE (AI-Data Harvesting Output)
- **Purpose**: Targeted data capture for AI diagnostics and automated audits.
- **Constraint**: Must NEVER contain conversational descriptions.
- **Format**: `[TAG] key1=value1 key2=value2 ...`

---

## 3. Minimal Tag Schema (Max 4 Standard Tags)

To prevent tag proliferation, the AI MUST strictly categorize all debug/trace logs into one of these 4 tags:

| Standard Tag | Target Category | Required Payload Keys | Example |
| :--- | :--- | :--- | :--- |
| `[SYS]` | Memory, OS, runtime environment, execution speed | `stage`, `rss`, `delta`, `elapsed_ms` | `[SYS] stage=l2_sim_cache rss=397MB delta=+297MB` |
| `[DATA]` | Data integrity, inputs, dataset boundaries, shapes | `symbol`, `nan_pct`, `shape`, `status` | `[DATA] symbol=BTCUSDT nan_pct=0.0 status=PASS` |
| `[ALGO]` | Signals, weights, alpha allocation, active parameters | `symbol`, `sleeve`, `raw_mu`, `weight` | `[ALGO] symbol=BTCUSDT sleeve=trend raw_mu=0.9` |
| `[EVAL]` | Performance, metrics, optimization trials, backtest results | `trial`, `cagr`, `sharpe`, `mdd`, `er` | `[EVAL] trial=14 cagr=0.15 sharpe=1.2 mdd=0.08` |

---

## 4. Token & Parsability Optimizations

- **Avoid Redundant Prefixing**: Do not duplicate timestamps or filenames in the message body if the logger formatter already prepends them.
- **Float Formatting**: Limit float numbers to a maximum of 3 decimal places (e.g., use `%.3f` or `:.3f`) to save tokens.
- **Conditional Array Truncation**: When logging symbol lists or arrays, truncate after 5 items and suffix with `_truncated={count}`.
  - **Bad**: `[ALGO] symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', ... 50 more symbols]`
  - **Good**: `[ALGO] symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT'] truncated=45`
