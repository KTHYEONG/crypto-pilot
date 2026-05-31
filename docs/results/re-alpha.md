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

---

# Re-Alpha Execution Results - 2026-05-31 (Phase alpha2 latest)

## 현재 상태
- `ALPHA_PASS`: `FALSE`
- 실행 모드: `--mode alpha --sync-mode skip --trials 1 --tf 4h --reference-date 2026-05-01`
- 목적: alpha gate taxonomy / cost-wall responsibility repair 확인
- 최신 상태: validation-only rank policy가 no-trade로 판정되었지만, 이제는 `alpha_panel empty`가 아니라 정책 경제성 실패와 rank IC 진단이 함께 출력됨

## 최신 실행 로그 (alpha2 최신 스모크)

```text
🔬 [SCORE-IC] dense_ranker ic=0.0347 t=4.51 hit=0.570 breadth=3.7 (cf. emit_breadth≈1, target_breadth≥8)
🔬 [OOS-RANKIC] ic=0.0347 t=4.51 n_bars=1417 cofinite_p50=17.0 bars_ge5_ratio=1.000 snr_oos_finite=0.174 cov_elig=1.000
🔬 [RESID-IC] raw=0.0347 resid=0.0361 resid_hit=0.564
🔬 [BE-EFF] N_raw=17.0 N_eff=1.5 sigma_r=666.3bps be_raw=0.0116 be_eff=0.0174 gap_resid_eff=+0.0187
📊 [PASS=❌] fail=['signal_below_effective_breakeven', 'signal_t_stat_too_low', 'basket_net_lcb_non_positive', 'policy_economics.validation_net_lcb_non_positive'] blockers={'rank_skill': ['signal_below_effective_breakeven'], 'post_selection': ['policy_economics.validation_net_lcb_non_positive', 'portfolio_ic_below_raw_breakeven', 'signal_lost_after_selection', 'no_profitable_horizon_found'], 'cost_turnover': ['basket_net_lcb_non_positive', 'basket_net_not_profitable'], 'statistical_robustness': ['signal_t_stat_too_low'], 'regime_stability': []}
>> ALPHA_PASS: FALSE [signal_skill_passes=FAIL portfolio_ic_above_breakeven=FAIL basket_net_positive=FAIL signal_preserved_after_selection=FAIL multi_horizon_sweep_passes=FAIL bear_market_basket_safe=OK]
```

## 판정

```text
핵심 결과:
- 24bps cost wall은 basket/economic 레이어에 남아 있고, rank-weight alpha magnitude에 직접 적용되지 않음
- validation-only no-trade policy가 `alpha_panel empty`로 숨지 않고 `policy_economics.validation_net_lcb_non_positive`로 드러남
- dense rank IC는 존재하지만 post-selection economics와 signal skill가 아직 기준 미달
- 따라서 alpha2는 평가 기준을 정리하는 데는 성공했지만, 전략 자체는 아직 pass 상태가 아님
```

## 테스트

```text
uv run pytest tests/unit/domain/futures/strategy/test_ml_diagnostics.py tests/unit/domain/futures/strategy/test_ml_builder.py tests/unit/domain/futures/strategy/test_alpha_evaluation.py tests/unit/domain/futures/forecast/test_compose.py tests/unit/execution/test_opt_main_futures_strategy_mode.py --tb=short
112 passed in 1.69s

UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. timeout 360 uv run python src/execution/opt_main_futures.py --mode alpha --sync-mode skip --trials 1 --tf 4h --reference-date 2026-05-01
latest smoke: `ALPHA_PASS: FALSE` / `policy_economics.validation_net_lcb_non_positive`
```
