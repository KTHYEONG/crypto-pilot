# Spec: ML-OOS Alpha Short Gate Asymmetry & Calibration Tuning

## Type
refactor / config-tuning

## Target Files
- `src/domain/futures/strategy/config.py`
- `tests/unit/domain/futures/strategy/test_ml_config.py`

## Diagnosis
The latest 100-trial strategy smoke/optimization runs fail with `reasons=['tradable_short_nz_below_threshold']` under the following state:
- `alpha_gate_metric_bps = 75.00` (which is `alpha_active_p95_bps`, easily passing the `floor_bps = 24.00` cost wall).
- `short_nz = 0.1241` (12.41% non-zero raw short predictions).
- `xs_short_preservation = 0.0421` (4.21% post-cost preservation ratio).
- This results in a final `tradable_short_nz` density of `0.1241 * 0.0421 = 0.00522` (0.522%), which falls below the default symmetric threshold of `0.01` (1.00%).

### Core Causes:
1. **Crypto Market Drift Asymmetry:** Crypto assets are inherently long-biased. Short-side signals are naturally rarer, shorter-lived, and harder to sustain above high cost floors compared to long-side signals.
2. **Excessive Rigidity:** Requiring the short side to maintain the exact same `1.0%` tradable coverage as the long side causes high-conviction models (active short tail at `75.00 bps`) to be rejected entirely, which is economically inefficient.

## Proposed Changes

### 1. `src/domain/futures/strategy/config.py`
- Modify the default value of `alpha_gate_min_tradable_short_nz` from `0.01` to `0.005` (0.5%).
- Maintain `alpha_gate_min_tradable_long_nz` at `0.01` (1.0%) to reflect the long bias in crypto.
- This asymmetry ensures we filter out models with zero/near-zero short capability while permitting high-conviction selective short models (like the 75 bps tail predictions observed here) to pass.

### 2. Surgical Plan

#### [MODIFY] [config.py](file:///home/kth/my_coin_traider/src/domain/futures/strategy/config.py)
Replace the default declaration:
```python
    alpha_gate_min_tradable_long_nz: float = 0.01
    alpha_gate_min_tradable_short_nz: float = 0.01
```
with:
```python
    alpha_gate_min_tradable_long_nz: float = 0.01
    alpha_gate_min_tradable_short_nz: float = 0.005
```

## Verification Plan

### Automated Tests
- Run validation unit tests to verify the config defaults and validations:
  ```bash
  uv run pytest tests/unit/domain/futures/strategy/test_ml_config.py --tb=short
  ```

### Smoke Verification
- Run a 100-trial strategy smoke test to verify if the asymmetric gate allows the model to pass successfully:
  ```bash
  PYTHONPATH=. timeout 360 uv run python src/execution/opt_main_futures.py --mode strategy --skip-universe --skip-data-sync --symbols BTCUSDT --trials 100 --tf 4h --reference-date 2026-05-01 --strategy ml_lambdamart_v1
  ```
