# Re-Alpha Execution Results - 2026-05-31 (Phase alpha4 latest)

## 현재 상태
- `ALPHA_PASS`: `FALSE`
- 실행 모드: `--mode alpha --sync-mode skip --trials 1 --tf 4h --reference-date 2026-05-01`
- 목적: validation-only rank selection policy 보정 + gate 진단 정합성 복구
- 최신 상태: validation 실패 시 `no-trade` fallback이 적용되어 `alpha_panel empty`로 종료됨

## 최신 실행 로그 (alpha4 최신 스모크)

```text
[RANK-POLICY] fold=0 polarity=1 q=0.20 floor=0.00 hold=12 val_lcb=-1.00 val_ir=0.00 mono=0.00
[RANK-POLICY] fold=1 polarity=1 q=0.20 floor=0.00 hold=12 val_lcb=-1.00 val_ir=0.00 mono=0.00
🔬 [OOS-DIAG] cause=sufficient_cofinite_check_ic oos_bars=1417 ge5_bars=1417
[RANK-POLICY] fold=0 polarity=1 q=0.20 floor=0.00 hold=12 val_lcb=-1.00 val_ir=0.00 mono=0.00
[RANK-POLICY] fold=1 polarity=1 q=0.20 floor=0.00 hold=12 val_lcb=-1.00 val_ir=0.00 mono=0.00
🔬 [OOS-DIAG] cause=sufficient_cofinite_check_ic oos_bars=1417 ge5_bars=1417
🔬 [EV-PRECLIP] fold=0 neg=0.0% p50=16.0bps p90=61.7bps p95=78.2bps n=8559
🔬 [EV-PRECLIP] fold=1 neg=0.0% p50=16.5bps p90=76.4bps p95=90.2bps n=8440
🔬 [EV-PRECLIP] vrefit neg=0.0% p50=10.3bps p90=61.6bps p95=90.2bps n=6624
🔬 [SCORE-IC] dense_ranker ic=0.0347 t=4.51 hit=0.570 breadth=3.7 (cf. emit_breadth≈1, target_breadth≥8)
⚠️ ALPHA: gate failed (evaluation continues) -> strategy ml alpha gate failed: reasons=['alpha_p95_below_cost_wall', 'tradable_long_nz_below_threshold', 'tradable_short_nz_below_threshold'] alpha_gate_metric_bps=0.00 alpha_full_matrix_p95_bps=0.00 floor_bps=24.00 long_nz=0.0000 short_nz=0.0000 xs_long_preservation=0.0000 xs_short_preservation=0.0000
⚠️ ALPHA: gate-bypass rerun also failed (generated alpha_long is all zero) -> evaluation with empty ml_out
!! FAIL: alpha_panel is empty — no OOS fold
```

## 판정

```text
핵심 결과:
- validation-only fallback은 정직하게 동작하지만, 현재 smoke는 `alpha_panel empty`로 끝나서 pass-oriented thresholds를 평가할 수 없음
- 이전 alpha3/alpha4에서 개선되었던 post-selection 성능은 latest smoke에서 재확인되지 않음
- 아직 `ALPHA_PASS=FALSE`인 이유:
  - validation 실패 시 no-trade fallback이 발동되어 OOS panel이 비어 있음
  - 따라서 `RANK-IC C3`, `L3-BASKET`, `C3-EXEC` pass 기준을 아직 검증하지 못함
```

## 테스트

```text
uv run pytest tests/unit/domain/futures/strategy/test_alpha_evaluation.py tests/unit/domain/futures/strategy/test_ml_builder.py tests/unit/domain/futures/forecast/test_compose.py tests/unit/execution/test_opt_main_futures_strategy_mode.py tests/e2e/test_cli_modes.py tests/integration/execution/test_opt_main_futures_bypass.py --tb=short
105 passed in 1.77s

UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. timeout 360 uv run python src/execution/opt_main_futures.py --mode alpha --sync-mode skip --trials 1 --tf 4h --reference-date 2026-05-01
latest smoke: `alpha_panel empty` / `FAIL`
```
