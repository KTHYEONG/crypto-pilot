# MHS Deploy Gate Result

- run_id: `25df82961d6c4c51a5a2c63b3628a5dc`
- window: 2021-01-01 ~ 2025-12-31
- flags: `MhsRunConfig()` defaults, `name_drift_trim=False` (live parity)
- input_manifest: sealed, digest `706eaae4...` (2076 files)

## Gate verdict

| axis | code | metric | value | threshold | result |
|---|---|---|---|---|---|
| integrity | I0-I4 | - | - | - | PASS |
| edge | E1 | oos_ann_log_growth_lcb | 0.181674 | > 0 | PASS |
| edge | E2 | stress_ann_log_growth_lcb | -0.163568 | > 0 | FAIL |
| edge | E3 | profitable_folds (primary) | 11 | >= 12 | FAIL |
| edge | E3 | profitable_folds (stress) | 7 | >= 12 | FAIL |
| survival | S1-S3 | - | - | - | not evaluated (short-circuited at E-axis) |

`go = false`, `reason_codes = [E2_STRESS_GROWTH_LCB_NOT_POSITIVE, E3_EDGE_BREADTH_BELOW_BINOMIAL_CRITICAL]`

## Deploy gate raw metrics

```json
{
  "n_folds": 16.0,
  "breadth_critical_count": 12.0,
  "oos_ann_log_growth": 0.623968,
  "oos_ann_log_growth_lcb": 0.181674,
  "stress_ann_log_growth": 0.278725,
  "stress_ann_log_growth_lcb": -0.163568,
  "profitable_folds": 11.0,
  "profitable_folds_stress": 7.0
}
```

## Legacy metrics (old gate, reference only)

| metric | value |
|---|---|
| geometric_cagr | 1.895277 |
| max_drawdown | -0.382183 |
| calmar | 4.959084 |
| expected_shortfall | -0.010437 |
| worst_1d | -0.203557 |
| worst_7d | -0.159819 |
| research_go.eligible (old) | true (reason: SELECTION_WINDOW_OVERLAP) |
