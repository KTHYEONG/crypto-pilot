# XS Alpha Blend Joint Search: Turnover Diagnostics, Not a Turnover Gate (rev. 3)

## 1. Where this started

`tools/research/xs_alpha_blend_joint_search.py` runs a discovery-only Optuna
search over `(xs_alpha_weight, leverage_scale)`, scored by
`discovery_reliability_score` (LCB90 CAGR). Its one run found the best LCB90
this project has ever measured (13.21%), but the qualification-window replay
was rejected by `evaluate_xs_admission`'s `turnover_max=150.0x/yr` gate
(realized turnover 175.4x/yr).

Revisions 1 and 2 both tried to make the *search* avoid crossing that cap —
first via a frozen historical ratio (rejected: overfit magic number), then
via a self-adapting worst-fold statistic (empirically validated, but still
hard-baked the 150.0 cap into the objective's rejection logic).

## 2. The question that reopened this: is `turnover_max=150.0` itself justified?

Investigated before writing any more code:

- **No derivation found.** `git log -S "turnover_max: float = 150.0"` finds
  only the introducing commit (`a7460725`). Its docstring cites "spec section
  3.1" — that spec file does not exist anywhere in the current repo. No ADR
  explains how 150 was chosen.
- **The cost model gives it no structural basis.** `CostModel`
  (`src/research/contracts.py`) is flat: `fee_rate=0.0005`,
  `slippage_rate=0.0003`, identical per fill regardless of trade frequency —
  no market-impact or liquidity-depth term. The ledger already deducts cost
  proportional to realized turnover, and the *separate* `cost_breakeven_min`
  gate already verifies the realized edge clears that flat cost assumption
  with margin. Nothing about the cost model's accuracy degrades as turnover
  rises, so `turnover_max` cannot be protecting against unpriced/mispriced
  cost — that's already handled elsewhere.
- **This project's own data doesn't support it either.** A pull across all 10
  `docs/results/*.json` profiles (`annualized_turnover`/`sharpe`/`cagr` per
  window):

  | profile | window | turnover | sharpe | cagr | admitted |
  |---|---|---|---|---|---|
  | v8_joint | qualification | **175.42** | 1.462 | **0.411** | False (turnover_max) |
  | v3 (contextual) | qualification | 293.59 | -1.515 | -0.178 | False |
  | v8 | qualification | 46.69 | 1.374 | 0.081 | True |
  | v8_sized | qualification | 32.24 | 1.439 | 0.171 | True |
  | v6 | qualification | 22.66 | 1.333 | 0.261 | True |
  | v7 (positioning) | qualification | 15.19 | 1.269 | 0.246 | True |

  The exact point blocked by `turnover_max` has the **best CAGR of any row in
  the table**. The one genuinely catastrophic-turnover case (v3, 293.6x) is
  independently attributed by this project's own ADR history to an
  execution-delay routing bug — not a turnover/cost effect. Small N,
  confounded by different weight/leverage choices per profile, not a
  rigorous study — but there is no visible negative relationship here.

**Conclusion:** I cannot find evidence that 150.0 is anything but an
unexamined heuristic. Hard-baking a rejection on it into the search (what
revisions 1–2 did) risks silently discarding genuinely good candidates on the
authority of a number nobody can currently justify. But loosening or removing
a risk gate is a financial-correctness call — this project's own
`quant.md`/safety policy requires exactly that kind of decision go to the
user, not get resolved unilaterally by an agent mid-spec.

## 3. Design (revision 3, adopted)

Stop presupposing the threshold anywhere in the search. Report the trade-off
instead of enforcing an unverified assumption of it either way:

- **`discovery_reliability_score` reverts to its original form** — no
  `turnover_max` parameter, no rejection logic. Pure LCB90, exactly as before
  any of this feature's revisions. `src/research/technical_experts/xs_alpha_baseline_blend.py`
  and `src/application/research/technical/xs_alpha_baseline_blend.py` end up
  with **zero net changes** relative to before this whole feature started.
- **`compute_turnover_fold_upper_bound`** (the worst-6-month-calendar-fold
  statistic from revision 2) is kept in `reliability.py` — it's a genuinely
  useful, self-adapting, non-magic-number statistic — but is *repurposed*
  from a gate into a diagnostic. It's called only from the dev-tool search
  script, after the search completes, never inside the objective.
- **`tools/research/xs_alpha_blend_joint_search.py`'s `run()`** additionally
  reproduces the winning point's scaled realized-weight ledger and prints
  `discovery_worst_fold_turnover=<value>` alongside a comparison to
  `XsAdmissionConfig().turnover_max`, explicitly labeled as informational —
  something for the researcher to weigh, not something the code decides for
  them.
- **`evaluate_xs_admission` and `XsAdmissionConfig.turnover_max` are
  completely untouched.** The gate stays exactly as strict as it is today;
  this spec does not weaken or bypass it anywhere. The CLI's honest post-hoc
  admission check on real qualification data remains the sole authority on
  whether a candidate is actually admitted.

This is deliberately the smallest, lowest-risk design of the three
revisions: it removes a questionable assumption from code that would have
silently acted on it, without substituting a different unverified assumption
in its place, and without touching the shared risk-gate infrastructure other
strategies also depend on.

## 4. What this does *not* resolve, on purpose

- Whether `turnover_max=150.0` should actually change. That's flagged as an
  explicit open follow-up (see contract `open_follow_up_not_in_scope`), not
  decided here. A real answer needs a proper study: realized OOS
  Sharpe/CAGR/LCB90 vs. realized turnover across this project's full
  historical profile set, controlling for the fact that different profiles
  differ in more than just turnover.
- If that study finds the cap isn't predictive, the structurally correct fix
  is *not* just raising the number — it's making `CostModel` genuinely
  convex in trade frequency (a market-impact/liquidity-depth term), so
  turnover is organically priced through the P&L it actually causes instead
  of gated externally. `CostModel` is shared infrastructure used by every
  strategy in this codebase; that change needs its own dedicated spec and
  explicit sign-off, never bundled into a feature-scoped change like this
  one.
- Re-running the search under this design will very likely rediscover the
  *same* point as before (the objective didn't change), still showing high
  turnover, still blocked by the unchanged admission gate. That is the
  correct, honest outcome for this revision — the value add is that the
  researcher now sees the turnover number *before* deciding whether to spend
  a CLI run confirming it, and the decision about what to do next is
  explicitly theirs.

## 5. Contract

See `docs/specs/xs_alpha_blend_joint_turnover_aware_contract.json` (revision
3). Three real code changes, all confined to `reliability.py` (kept from rev
2: fold-label extraction refactor + the new diagnostic statistic) and the
dev-tool script (rev 3's new diagnostic printing). Explicit non-changes are
listed for the three files revisions 1–2 touched and this one reverts.

`--spec-only` re-validated against the current codebase after this rewrite:
all target paths resolve; only "not implemented yet" failures remain for the
genuinely new pieces (`compute_turnover_fold_upper_bound`, the diagnostic
block in `run`).

## 6. Post-implementation plan

After `/implement` + `/check`: re-run
`tools/research/xs_alpha_blend_joint_search.py` (same `max_trials=30, seed=0`
— expect the same `best_params` as before, since the objective is unchanged),
read off the printed turnover diagnostic, and bring both that number and this
spec's open follow-up question back to the user before deciding what (if
anything) to do about the `v8_joint` candidate or `turnover_max` itself.
Delete `scratch/test_turnover_ucb.py` once
`compute_turnover_fold_upper_bound` is implemented for real.
