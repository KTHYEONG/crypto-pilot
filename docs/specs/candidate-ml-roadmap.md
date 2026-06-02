# Candidate ML Roadmap

> spec_type: `prd` + `refactor`
> source: `docs/results/result.md`
> date: 2026-06-02
> status: active
> verdict: gate calibration은 해결됐고, 다음 병목은 ML threshold가 아니라 rule alpha 품질과 후보군 정제다.

## 1. Current Status

Observed in the latest diagnostics:

- Gate calibration is recovered.
- `gate_label1_rate=0.415`, `mean_p=0.4172`.
- Edge model is still negative across the full candidate pool.
- `mu_net mean=-43.3 bps`, `mu_net max=-34.1 bps`, `pct_ge1=0.000`.
- Rule-only ablations remain negative.
- Rule diagnostics show:
  - `keep=0`
  - `flip=7`
  - `best_group=variant=trend_donchian:donchian_18`
  - `best_mean_edge=6.2`

Interpretation:

1. The selector is not broken.
2. The gate is no longer the main bottleneck.
3. The rule family set still does not yield a sufficiently strong positive subset for production compounding.
4. Side flipping is a real signal on several families, but not enough yet to make the current pool compound safely.

## 2. Consolidated Decision

Do not do these next:

- do not lower `min_expected_net_bps` to force trades
- do not add family identity features before proving a positive rule subset exists
- do not treat more generic ML logging as the primary fix

Do these next:

1. prune or flip rule candidates based on `rule_diagnostics.py`
2. only then decide whether candidate identity features are worth adding
3. re-run OOS compounding backtests on the pruned subset

## 3. Remaining Phases

### Phase A: Candidate Pruning

Use `rule_diagnostics.py` as the source of truth for candidate selection.

Accepted candidate actions:

- `KEEP_CANDIDATE`
- `SIDE_FLIP_CANDIDATE`

Rejected actions:

- `DROP_OR_REWORK`
- `INSUFFICIENT_OBS`

Implementation intent:

- expose a simple config allowlist such as `enabled_candidate_variants`
- filter raw rule events before dataset construction
- preserve `entry_idx` execution semantics

Target effect:

- reduce the candidate pool to only the rule families/variants that have a defensible net edge after cost and hurdle

### Phase B: Dataset Identity Features

Only if Phase A produces at least one strong positive subset:

- add stable one-hot identity features for `family` and `variant`
- build the mapping from split-local categories only
- keep the feature set deterministic and leak-free

Target effect:

- let gate/edge models condition on candidate family identity instead of inferring it indirectly

### Phase C: OOS Compounding Validation

Only after Phase A/B:

- rerun the alpha phase
- confirm non-zero `passed` count in `candidate_portfolio.py`
- confirm candidate variants contribute to `rule_plus_ml_gate` or `candidate_ml_full`
- confirm compounding metrics are non-zero and survive friction

## 4. Target Files

Pending files for the next implementation pass:

- `src/domain/futures/strategy/rule_signals.py`
- `src/domain/futures/strategy/candidate_dataset.py`
- `src/domain/futures/strategy/candidate_portfolio.py`
- `src/domain/futures/strategy/config.py`
- `src/domain/futures/strategy/ablation.py`
- `tests/unit/domain/futures/strategy/test_rule_diagnostics.py`
- `tests/unit/domain/futures/strategy/test_candidate_dataset.py`
- `tests/unit/domain/futures/strategy/test_candidate_portfolio.py`

Already in place and treated as support code:

- `src/domain/futures/strategy/rule_diagnostics.py`
- `src/domain/futures/strategy/candidate_labels.py`
- `src/domain/futures/strategy/candidate_gate.py`
- `src/domain/futures/strategy/candidate_edge.py`

## 5. Acceptance Criteria

Phase A is accepted when:

- at least one candidate is classified as `KEEP_CANDIDATE`
- at least one `SIDE_FLIP_CANDIDATE` is validated on real labeled data
- the pruned candidate set yields a non-zero `passed` count

Phase B is accepted when:

- identity features are deterministic
- no future label or outcome leakage is introduced
- dataset tests prove fallback behavior remains intact

Phase C is accepted when:

- `rule_plus_ml_gate` or `candidate_ml_full` produces non-zero trades
- the backtest remains drawdown-aware and friction-aware
- compounding metrics are materially better than the current zero-trade baseline

## 6. Verification

After the next implementation pass:

```bash
uv run ruff check --fix src/domain/futures/strategy/*.py tests/unit/domain/futures/strategy/*.py
uv run mypy src/domain/futures/strategy/*.py
uv run pytest tests/unit/domain/futures/strategy/ --tb=short
```

If pruning is wired into the runtime path, rerun:

```bash
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. uv run python src/execution/opt_main_futures.py --phase alpha --timeframe 4h --sync skip
```

Expected outcome:

- rule diagnostics stay visible
- the candidate pool becomes smaller and more selective
- the ML path only evaluates rule subsets that already show positive signal quality

## 7. Summary

This file replaces the older audit, remediation, and rule-next-step docs.

The working thesis is now simple:

1. gate is fixed
2. rule alpha is still the bottleneck
3. ML should be used to amplify a proven positive subset, not to rescue a negative candidate pool
