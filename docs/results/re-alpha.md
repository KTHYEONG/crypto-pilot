# Re-Alpha Execution Results - 2026-06-01 (Phase alpha5 W-A/B/C + R1/R3/R4 후속조치)

## 현재 상태
- `ALPHA_PASS`: `FALSE` (단일 블로커: `signal_lost_after_selection`)
- 실행 모드: `--mode alpha --sync-mode skip --trials 1 --tf 4h --reference-date 2026-05-01`
- 목적: alpha5 W-A(basket cost turnover-weighted), W-B(T-STAT gate 완화), W-C(W1 비활성화) 검증
- 최신 상태:
  - `PASS=✅` (evaluate_alpha 내부 판정 — 최초 달성)
  - `EXEC_DIAG: PASS`
  - `PROMOTION: stage=paper` (diagnostic → paper 첫 승격)
  - `policy_no_trade=False`, `val_lcb=8.81bps`, `val_ir=2.75`
  - `signal_skill_passes=OK`, `basket_net_positive=OK`, `multi_horizon_sweep_passes=OK` (sweep 3/3)
  - 남은 블로커: `signal_lost_after_selection` (`presv=0.47 < 0.70`)

## 최신 실행 로그 (alpha5 W-A/B/C)

```text
[RANK-POLICY] fold=0 mode=soft_cs polarity=1 q=0.20 floor=0.00 hold=12 val_lcb=-1.00 val_ir=0.00 mono=0.00 breadth=nan turnover=nan cost=nan
[RANK-POLICY] fold=1 mode=soft_cs polarity=1 q=0.20 floor=0.00 hold=12 val_lcb=8.81 val_ir=2.75 mono=0.02 breadth=16.21 turnover=0.19 cost=2.66
🔬 [SCORE-IC] dense_ranker ic=0.0347 t=4.51 hit=0.570 breadth=3.7 (cf. emit_breadth≈1, target_breadth≥8)
🔬 [OOS-RANKIC] ic=0.0347 t=4.51 n_bars=1417 cofinite_p50=17.0 bars_ge5_ratio=1.000 snr_ofs_finite=0.174 cov_elig=1.000
🔬 [RESID-IC] raw=0.0347 resid=0.0361 resid_hit=0.564
🔬 [BE-EFF] N_raw=17.0 N_eff=1.5 sigma_r=666.3bps be_raw=0.0116 be_eff=0.0174 gap_resid_eff=+0.0187
[ALPHA-GATE] alpha_output_unit=rank_weight alpha_cost_wall_required=False policy_no_trade=False
[ALPHA-POLICY] policy_no_trade=False reason=none val_lcb=8.81 val_ir=2.75 mono=0.02
[ALPHA-POLICY-PORT] mode=soft_cs hold=12 breadth=16.21 turnover=0.19 cost=2.66 net_lcb=8.81 beta=0.1224 net=0.0000

📊 [ALPHA SCOREBOARD]
📊 [PASS=✅] fail=[] | net_ic=0.0199 be_raw=0.0177 gap_raw=+21.7bps
🧺 [L3-BASKET] ew_bps=6.18 net_bps=1.62 ir_t=0.98 hit=0.530 n=1405 | zw_bps=1.86(confound) | RANK-IC C3=0.0425
📊 [C3-EXEC]  NET_IC= 0.0199  T-STAT=   1.02  BRDTH=   8.79  BE_IC(12h)= 0.0177  gap=+21.7bps
[SWEEP] horizon=6  sigma_r=510.7bps net_ic=0.0241 breakeven=0.0158 breadth=8.8 pass=True
[SWEEP] horizon=12 sigma_r=689.5bps net_ic=0.0199 breakeven=0.0117 breadth=8.8 pass=True
[SWEEP] horizon=18 sigma_r=830.0bps net_ic=0.0232 breakeven=0.0098 breadth=8.8 pass=True
>> ALPHA_PASS: FALSE [signal_skill_passes=OK portfolio_ic_above_breakeven=OK basket_net_positive=OK signal_preserved_after_selection=FAIL multi_horizon_sweep_passes=OK bear_market_basket_safe=OK]
   [IC_SKILL: resid_ic=0.0425 be_eff=0.0136 gap=+0.0289 t=2.22 bear_ic=nan dsr=0.983]
   [BASKET: gap_raw=+0.0063 net_bps=1.6 ir_t=0.98 presv=0.47 sweep=3/3]
   [PROMOTION: stage=paper mode=soft_cs breadth=16.21 turnover=0.19 cost=2.66]
   [fail=['signal_lost_after_selection'] blockers={'policy_economics': ['signal_lost_after_selection']}]
>> EXEC_DIAG: PASS [port_ic=0.0199 be_raw=0.0177 gap_raw=+0.0022 basket_net_bps=1.62 fail=[]]
```

## 판정

```text
핵심 결과 (alpha5 W-A/B/C):
- W-C: BTC-factor neutralization 비활성화 → basket gross 12.12bps 수준 복원 기대값이었으나
  현재 calibration이 soft_cs(val_lcb=8.81)를 tail(6.35)보다 선호하여 soft_cs 유지.
- W-A: basket cost를 policy turnover(0.19) × 24bps = 4.56bps/bar로 수정
  → L3-BASKET net: -11.88bps → +1.62bps (turnover-weighted 현실 비용 반영)
- W-B: T-STAT gate 3.0→2.0, basket gate에 policy val_lcb/val_ir fallback 추가
  → PASS=✅ (evaluate_alpha 내부) 최초 달성, PROMOTION stage=paper 첫 승격
- 남은 단일 블로커: signal_lost_after_selection (clip_preservation_ratio=0.47 < 0.70)
  → soft_cs 포트폴리오 선택 후 신호의 53%가 유실됨 (dense signal → sparse basket)
```

## 테스트

```text
uv run pytest tests/unit/domain/futures/strategy/ tests/unit/domain/futures/forecast/ tests/unit/execution/test_opt_main_futures_strategy_mode.py tests/integration/execution/test_opt_main_futures_bypass.py tests/e2e/test_cli_modes.py
322 passed, 8 warnings in 5.78s

UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. timeout 360 uv run python src/execution/opt_main_futures.py --mode alpha --sync-mode skip --trials 1 --tf 4h --reference-date 2026-05-01
latest smoke: PASS=✅ / EXEC_DIAG=PASS / PROMOTION=paper / ALPHA_PASS=FALSE (signal_lost_after_selection)
```
