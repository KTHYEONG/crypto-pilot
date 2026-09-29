# Operating Manual: Crypto-Pilot

> **Domain Identity:** 24/7 crypto perpetual futures quantitative research & systematic trading system.

## 1. Domain Ground Truth & Hard Boundaries
- **Strict Temporal Causality & Valuation Integrity:** Zero look-ahead bias. Mark-to-market valuation, signal calculation, and execution timing must strictly respect observable time series.
- **Margin & Ruin Prevention:** Sizing and leverage must strictly operate within exchange maintenance margin requirements (MMR) and capital preservation limits. Never risk catastrophic margin liquidation.
- **Derivative Friction Accounting:** Factor all real market friction into returns: funding rate transfers, maker/taker commission tiers, spread, and liquidity slippage.
- **Overfitting & Multi-Regime Validation:** Enforce strict temporal out-of-sample isolation (purged/embargoed splits). Strategy performance must withstand non-Gaussian fat tails and liquidity shocks.
- **Workspace Hygiene:** Keep exploratory experiments and temporary files strictly isolated under `scratch/`. Never commit raw cache or scratch files.

## 2. Autonomy & Execution Contract
- **Bias Toward Action & Diagnostic Autonomy:** For data queries, exploratory scratch diagnostics, and empirical root-cause isolation, execute immediately without asking for permission. When asked open-ended questions about bugs or data anomalies, proactively run scratch experiments under `scratch/` to discover truth. Never modify production code or commit in response to open-ended diagnostic queries.
- **Skills as On-Demand Tools:** Skills (`probe`, `spec`, `implement`, `check`, `refactor`, `commit`) are modular, independent utilities—NOT a mandatory sequential pipeline. When explicitly invoked via slash commands (`/probe`, `/spec`, etc.), execute only that targeted skill and halt for user review. Specs live under `docs/specs/` (gitignored for model/tool handoffs without repo bloat).

## 3. Project Toolchain & Verification
Verify code changes against the project's native toolchains before concluding tasks:
- **Quality Gate:** `uv run python tools/agent_skills/lean_check.py`
- **Test Runner:** `uv run pytest`
- **Git Commits:** Run the project's `commit` skill.

## 4. Communication & Language
- **Natural Korean:** Converse, explain rationales, and report findings in Korean (한국어). Inside structured output cards, retain English keys/badges while writing descriptions in Korean.
- **Technical English:** System instructions, rules, specifications (`docs/specs/`), code, and docstrings are written in English.

## 5. Domain Rule Routing
- **Domain Invariants:** [domain.md](.agents/rules/domain.md) — *Financial invariants, temporal causality, market frictions, and conservation laws.*
- **Testing & Quality:** [testing.md](.agents/rules/testing.md) — *Invariant-driven testing, boundary conditions, failure isolation, and diff-coverage.*
- **Architecture & Standards:** [code-style.md](.agents/rules/code-style.md) — *Module boundaries, strong static typing contracts, and toolchain alignment.*
- **Documentation & Comments:** [documentation.md](.agents/rules/documentation.md) — *Production docstrings, architecture specs, and non-obvious rationale.*
- **Performance & Optimization:** [performance.md](.agents/rules/performance.md) — *Vectorized panel builds, hot-loop profiling, and resource budgets.*
- **Logging & Diagnostics:** [logging.md](.agents/rules/logging.md) — *Operational logging, 6 fixed category taxonomy, and credential redaction.*
