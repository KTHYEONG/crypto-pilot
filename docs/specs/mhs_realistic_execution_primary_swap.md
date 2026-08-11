# MHS Research-GO Primary: Swap to Realistic Small-Scale Execution

- **Registered as**: `ADR_20260811_MHS_REALISTIC_EXECUTION_PRIMARY_SWAP` (pending)
- **Domain**: Research / MHS execution semantics, Research-GO gate
- **Supersedes (for Research-GO primary anchoring only)**: the "STRICT proxy is
  primary" framing in `docs/architecture/multi-horizon-market-state.md` §3/§5
- **Prior work**: `ADR_20260810_MHS_TOUCH_PROXY_FILL_MEASUREMENT` (first
  measured the 30-minute-timeout cost explosion), `ADR_20260810_MHS_EXECUTION_
  LADDER_AND_DISCOVERY_GATE` (tried price-splitting the same mechanic, made
  Sharpe worse)

## 0. Root cause (measured this session, not assumed)

The user asked why the same signal/weights produce a catastrophic full-replay
result (`primary_autocorr_sharpe=-1.837`, MDD=-99.24%, final equity 1.0→0.008
over 5 years) when the underlying prescreen `net_t` was only mildly positive
(1.63-1.86). The real ledger artifacts
(`docs/results/mhs_horizon_diagnostic_artifacts/_full/ledger.parquet`,
`daily_ledger.parquet`) were inspected directly (not re-derived from theory):

- The 5-year equity decay is not a slow cost bleed; it is dominated by a
  handful of single-5-minute-bar catastrophic losses (e.g. 2025-10-12 01:35
  UTC: -41.6% in one bar, `fill_turnover=1.49`). None of `mark_to_market_pnl`,
  `funding_charge`, or `fee_charge` account for these bars (residual ≈ the
  entire loss) — they are pure realized fill-price slippage.
- Every such bar sits **exactly 30 minutes after** a decision-time order entry
  (e.g. 01:05 → 01:35). `slow_momentum` rebalances all ~30 roster symbols
  simultaneously every 24h (`step_hours=24`), so on trend days many symbols'
  passive limit orders fail to be touched within `ExecutionSpec.passive_
  timeout_minutes=30` and **all** fall back to taker at the same bar —
  a synchronized, roster-wide forced-market-order event, not idiosyncratic
  noise. This matches and confirms the 4th-iteration ADR's "30분 타임아웃발
  비용 폭증" finding; the 5th-iteration price-ladder (K=4 tranches) could not
  fix it because the failure mode is timing-driven ("가격이 안 돌아온다"), not
  price-driven, exactly as that report concluded.
- **This mechanism does not reflect this account's realistic execution
  behavior.** The already-computed `deployment_readiness.participation_
  warnings` in every run show `fill_notional_to_1m_quote_volume≈1.19e-9`,
  `fill_notional_to_30m_quote_volume≈1.74e-10`,
  `daily_trade_notional_to_daily_quote_volume≈2.59e-12` — order sizes are
  9-12 orders of magnitude below the market's own volume. A patient,
  footprint-minimizing passive-order strategy (the entire rationale for
  `OHLCV_STRICT_PROXY`'s 30-minute touch-and-wait) is institutional-scale
  behavior; at this negligible footprint there is no market-impact reason to
  avoid simply crossing the spread. The value being protected by waiting
  (maker 2.0bps vs taker 5.0bps fee, `ExecutionSpec` — a 3bps saving, ~6bps
  round-trip) is small next to the tail risk the 30-minute wait actually adds.
- **Direct A/B confirmation, same signal/weights/prices, only the fill rule
  differs** (`replay_execution_window_pair` already computes both bounds on
  every run — this is not a new measurement, just a previously-unused one):

  | Metric | `OHLCV_STRICT_PROXY` (patient, current primary) | `OHLCV_IMMEDIATE_TAKER` (current stress) |
  | :--- | :--- | :--- |
  | Daily autocorr-adjusted Sharpe | **-1.837** | **+0.406** |
  | Naive Sharpe | -0.698 | +0.093 |
  | Final equity (5y, from 1.0) | 0.008 | **1.255** |
  | Max drawdown | -99.24% | **-44.6%** |
  | Worst single day | -48.1% | -3.7% |

  (Sharpe values recomputed this session via `autocorrelation_adjusted_
  sharpe` directly against `daily_ledger.parquet`'s `slow_momentum_stress`
  daily returns; all other figures from the artifacts or the existing
  top-level JSON.)

**Honest scope of this fix**: +0.406 is a large, real improvement over
-1.837, but it is still below the frozen `MHS_GO_PRIMARY_SHARPE_FLOOR=0.6`.
This spec corrects a genuine execution-modeling mismatch (not a data or
signal bug) and removes ~2.2 Sharpe points of pure execution-timing artifact,
but it does **not** by itself flip Research GO to eligible. It is a
prerequisite fix, not a complete solution — say so plainly when reporting
results, do not imply GO is achieved.

## 1. Design: swap primary, replace the stress gate, keep the old bound as a labeled reference

Both bounds are always computed today via `replay_execution_window_pair`
(`src/mhs/execution.py:1927`) — this spec does not add a new fill mechanic,
it changes which bound each `primary`/`stress` slot is fed by and adds one
new cost-stress variant of the *same* realistic bound.

1. **Primary → `OHLCV_IMMEDIATE_TAKER`.** At both call sites that currently
   do `primary, stress = replay_execution_window_pair(...)`
   (`src/application/research/mhs/evaluation.py:1515` top-level,
   `:2124` fold-level), replace the paired call with an explicit
   `replay_execution_windows(..., "OHLCV_IMMEDIATE_TAKER", ExecutionSpec(), ...)`
   feeding the `primary`/`strict` field slots unchanged (no dataclass field
   renamed — `MhsBookReport.primary`/`MhsFoldReport.strict` keep their
   existing names and JSON keys; only what populates them changes, per
   `python.md` §3 Scope Control and to avoid churning every downstream
   consumer/test of the report schema).
2. **Stress → real `SPREAD_AND_COST_X3`, not the old patient proxy.** The
   9 `synthetic_stress_scenarios()` (`src/mhs/evaluation.py:356`) are
   currently pure labels (`{"description": ...}`, never computed) because
   most need synthetic price generation the OHLCV proxy cannot produce
   (spec governance: "no scenario may tune Phase 1 parameters" / "reported
   rather than fabricated"). `SPREAD_AND_COST_X3` ("spread and costs
   triple") is the one exception: it needs no synthetic price, only a cost
   multiplier on the same real fills. Add
   `MHS_STRESS_COST_MULTIPLIER = 3.0` and a helper `_stress_cost_execution_
   spec() -> ExecutionSpec` (evaluation.py, next to `MHS_DISCOVERY_START`)
   returning `ExecutionSpec(maker_fee_bps=6.0, taker_fee_bps=15.0,
   taker_slippage_bps=9.0)` (each default field × 3.0). Feed `stress` from
   `replay_execution_windows(..., "OHLCV_IMMEDIATE_TAKER",
   _stress_cost_execution_spec(), ...)` — same realistic fill mechanic,
   3x the cost assumption. `MHS_GO_REASON_STRESS_SHARPE`'s check
   (`stress_sharpe > 0`) is unchanged in formula; it now tests cost-shock
   robustness of the realistic primary instead of re-testing the retired
   patient-chase mechanic (which would otherwise trivially and permanently
   fail this gate for the reason this spec just proved is a modeling
   artifact, not real risk — leaving it wired would silently reintroduce
   the bug under a different name).
3. **Keep the old patient/passive-chase bound, demoted to informational.**
   Do not delete `OHLCV_STRICT_PROXY` or its 30-minute-timeout mechanic —
   still exercised by other callers (`OHLCV_TOUCH_PROXY` diagnostics, unit
   tests). Add it as a new optional diagnostic field on `MhsBookReport`
   only (top-level report; not `MhsFoldReport` — Research-GO folds only need
   the two gating bounds), following the existing `touch`/`touch_naive_
   sharpe` opt-in-diagnostic field pattern exactly:
   `patient_reference: StrategyExecutionReplayResult | None = None`,
   `patient_reference_naive_sharpe: float | None = None`. Always computed
   (not gated behind a request flag, since it is cheap — already computed
   today as part of the pair) via one more `replay_execution_windows(...,
   "OHLCV_STRICT_PROXY", ExecutionSpec(), ...)` call, reported for reference
   ("what would 30-minute patient chasing have produced") but never gates
   Research GO.
4. **`fill_source` metadata** (`evaluation.py:2550`, currently the hardcoded
   literal `"OHLCV_STRICT_PROXY"`) becomes `"OHLCV_IMMEDIATE_TAKER"`.

## 2. Documentation updates (surgical, per `documentation.md` §4)

`docs/architecture/multi-horizon-market-state.md` §3 ("실행 proxy는 다음 두
경계를 별도 계산한다") and §5 (criterion 2/3 wording): swap which proxy is
named primary vs the labeled stress/reference bound, and add one sentence
citing the participation-ratio justification (§0 above) inline — no history,
no ADR tags, edit the existing table/prose in place per the file's 300-line
surgical-update constraint.

## 3. Verification plan

1. Unit tests covering `_stress_cost_execution_spec()` (asserts the exact
   3x-multiplied `ExecutionSpec` fields) and the two call-site swaps (a
   fixture-level test asserting `MhsBookReport.primary`/`MhsFoldReport.strict`
   carry `"OHLCV_IMMEDIATE_TAKER"` fill-source provenance, not
   `"OHLCV_STRICT_PROXY"`).
2. Existing execution/replay unit and integration tests must keep passing
   unchanged — `OHLCV_STRICT_PROXY`/`OHLCV_IMMEDIATE_TAKER` fill mechanics
   themselves are untouched; only which callers select which bound changes.
3. Post-implementation measurement (not fabricated here): rerun the real
   `research run portfolio mhs-horizon-diagnostic` full 2021-2025 replay and
   the anchored-fold Research-GO evaluation, and update `docs/results/
   mhs-res.md` with the actual measured `primary_autocorr_sharpe`,
   `stress_naive_sharpe` (now the ×3-cost check), `patient_reference_naive_
   sharpe`, MDD, and `research_go.eligible`/`reason_codes` — report the real
   numbers, including if Research GO is still `False` (expected per §0's
   honest-scope note; the fold-level floor is 0.6 and the top-level measured
   value this session was 0.406).
