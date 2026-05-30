# Alpha Execution Results - 2026-05-30

## Execution Log
```text
[SYNC-COVERAGE] rows=1 file=/home/kth/my_coin_traider/logs/futures/universe/sync_coverage_report.parquet
Ledger update complete.
.. AUDIT: req=214 load=201 coverage=0.94 | !! FAIL: 1h(engineered: nan=2.8%), 1h(merged: nan=2.8%), 1h(raw: nan=2.9%)
.. DATA: ok=201 keep=96 fail={'warmup_insufficient': 62, 'panel_history_insufficient': 34, 'is_coverage_short': 9}
<< DATA: 30.78s (ok=96)
>> STRATEGY: alpha | lambdamart
.. STRATEGY: panels(inf=96, live=33) trade=20
[RAW-SIGNAL-DIAG] raw_cs_std=0.0369 resid_cs_std=0.0347 var_retention=0.941 n_ts=3704 raw_nz=0.979 resid_nz=1.000
🛡️ [DATA-INT] zero_price=0.000000 ohlc_violation=0.000000 bar_gap=0 nan_decomp={'universe_inactive': 0.998143259160405, 'price_missing': 0.0, 'warmup': 0.0018567408395949983, 'kill': 0.0}
🧬 [FEAT-INT] constant=9 drifted=5 redundant=8 leakage=0
🧬 [FEAT-INT] constant=['basis_1', 'basis_mean_6', 'execution_cost_rank', 'oi_ret_1', 'oi_z_18', 'basis_missing_ind', 'oi_missing_ind', 'adv_missing_ind', 'execution_cost_missing_ind']
🧬 [FEAT-INT] redundant=[('ret_3', 'rev_3', -1.0), ('ret_6', 'rev_6', -1.0), ('ret_12', 'mom_12_skip_1', 0.953), ('ret_12', 'rev_12', -1.0), ('ret_36', 'mom_36_skip_3', 0.955), ('mom_12_skip_1', 'rev_12', -0.953), ('cs_rank_ret_18', 'cs_sharpe_18', 0.961), ('dollar_volume_rank', 'adv_rank', 1.0)]
🧬 [FEAT-SELECT] kept=26/56 names=['ret_1', 'ret_3', 'ret_6', 'ret_12', 'ret_18', 'ret_36', 'momentum_autocorr', 'cs_residual_momentum', 'vwap_deviation', 'rv_6', 'rv_18', 'rv_36', 'downside_rv_18', 'atr_pct_14', 'vol_of_vol_36', 'funding_z_30d', 'funding_sign_persistence_6', 'funding_rate_momentum', 'volume_z_18', 'btc_ret_6', 'market_median_ret_6', 'market_dispersion_6', 'positive_breadth_6', 'micro_hl_spread_1', 'micro_close_to_hl_1', 'funding_missing_ind']
🧠 ML_INIT: feats=26 symbols=96 rows=6462 train_w=24m | 🧩 folds=2
🧠 ML-PARALLEL: Training 6 folds in parallel. LightGBM n_jobs forced to 1.
🧠 ML-PARALLEL: Completed all 3 folds in 23625.23 ms
🔬 [EV-PRECLIP] fold=0 neg=9.3% p50=10.0bps p90=21.5bps p95=24.9bps n=8559
🔬 [EV-PRECLIP] fold=1 neg=41.9% p50=11.4bps p90=74.8bps p95=75.0bps n=8440
🔬 [EV-PRECLIP] vrefit neg=56.0% p50=-4.0bps p90=27.6bps p95=38.3bps n=6624
🧩 ML_OOS_FILL: virtual_refit complete (rows=6624 L_nz=1.000 S=1.000)
🔬 [SCORE-IC] dense_ranker ic=0.0454 t=5.10 hit=0.536 breadth=3.7 (cf. emit_breadth≈1, target_breadth≥8)
🔬 [OOS-RANKIC] ic=0.0454 t=5.10 n_bars=1417 cofinite_p50=17.0 bars_ge5_ratio=1.000 snr_oos_finite=0.174 cov_elig=1.000
🔬 [OOS-DIAG] cause=sufficient_cofinite_check_ic oos_bars=1417 ge5_bars=1417
🔬 [RESID-IC] raw=0.0454 resid=0.0446 resid_hit=0.550
🔬 [BE-EFF] N_raw=17.0 N_eff=1.5 sigma_r=666.3bps be_raw=0.0116 be_eff=0.0174 gap_resid_eff=+0.0273
🔬 [FEATURE-IC] rv_18:ic=-0.0787,gap=-0.0961 | rv_36:ic=-0.0772,gap=-0.0945 | rv_6:ic=-0.0764,gap=-0.0937 | atr_pct_14:ic=-0.0711,gap=-0.0885 | micro_hl_spread_1:ic=-0.0685,gap=-0.0859 | downside_rv_18:ic=-0.0643,gap=-0.0817 | vol_of_vol_36:ic=-0.0622,gap=-0.0796 | funding_z_30d:ic=-0.0409,gap=-0.0583 | ret_12:ic=-0.0407,gap=-0.0581 | cs_residual_momentum:ic=-0.0368,gap=-0.0542 | ret_6:ic=-0.0299,gap=-0.0473 | ret_18:ic=-0.0265,gap=-0.0439 | vwap_deviation:ic=-0.0262,gap=-0.0436 | funding_rate_momentum:ic=-0.0201,gap=-0.0375 | ret_3:ic=-0.0182,gap=-0.0356
📊 ML_EVAL: nz(L=0.025 S=0.014) ic=0.0098 t=3.45 hit=0.203 obs=3704
💰 ML_COST: gate=75.0bps floor=24.0bps pass=true
[ML-IC-GATE] IC gate WARN: mean_ic=0.0098 t_stat=3.45 hit_ratio=0.203
[REGIME-GATE] applied: bull=2431 bear=2905 chop=1096 unlabeled=30
[ML-PIPE-PROF] anchored=False symbols=96 tf=4h elapsed=36.27s alpha_rows=620352
.. ALPHA_MERGE: syms=96 span=2022-10-20 ~ 2025-09-30 L_nz=0.013 S=0.008
.. ALPHA_MERGE: syms=96 span=2022-10-20 ~ 2025-09-30 L_nz=0.011 S=0.007
[RANK-SCOREBOARD] rank_cs_neutral applied: q=0.33 long_nz=0.333 short_nz=0.333
🔬 [IC-DECOMP] dense_c1_raw=0.0031(hit=0.118 br=1.8) dense_c1_resid=0.0023 dense_c3_raw=0.0066 dense_c3_resid=0.0043
🔧 [1B-RESID] market_fwd_std=0.0501 beta_mean=1.000 real_std=0.0692 resid_std=0.0472
🏅 [RANK-QUALITY L1] ic=0.0062 t=3.01 hit=0.130 breadth=3.7 | rank_score_long-short vs forward_gross_ret (C1 dense, unclipped)
🏅 [RANK-GENERALIZE] oos_rank_ic=0.0454 is_rank_ic=0.0098 retention=4.65 decision=continue

📊 [ALPHA SCOREBOARD]
Metric | RESID_IC |  T-STAT  |  N_EFF   |   DSR    | BE_EFF(12h) | BEAR_IC
Value  |  0.0141  |    3.62  |    15.0  |  1.0000  |   0.0131  |  0.0240
Result |    ✅    |    ✅    |  N_eff   |    ✅    |  (gap=+10.3bps)  |    ✅
📊 [PASS=✅] fail=[] | net_ic=-0.0030 be_raw=0.0386 gap_raw=-415.7bps
📊 [RANK-IC C3] ic= 0.0141  t=   3.62  breadth=   1.94 | rank_score_long-short vs beta-resid (dense, unclipped, C3)
🧺 [L3-BASKET] ew_bps=-12.94 net_bps=-36.94 ir_t=-2.61 hit=0.494 n=1417 | zw_bps=-11.18(confound) | RANK-IC C3=0.0141
📊 [C3-EXEC]  NET_IC=-0.0030  T-STAT=  -0.95  BRDTH=   1.74  BE_IC(12h)= 0.0386  gap=-415.7bps
🌐 [REGIME IC] Bull: 0.011 | Bear: 0.024 | Chop: -0.006
[SWEEP] horizon=6 sigma_r=493.4bps net_ic=-0.0043 breakeven=0.0369 breadth=1.7 pass=False
[SWEEP] horizon=12 sigma_r=691.7bps net_ic=-0.0065 breakeven=0.0263 breadth=1.7 pass=False
[SWEEP] horizon=18 sigma_r=845.4bps net_ic=-0.0080 breakeven=0.0216 breadth=1.7 pass=False
📈 SWEEP: [6h: ic=-0.004 ❌] [12h: ic=-0.007 ❌] [18h: ic=-0.008 ❌]
>> ALPHA_PASS: TRUE [phase1 resid_ic=0.0141 be_eff=0.0131 gap=+0.0010(OK) t=3.62(>=2.0:OK) bear_ic=0.0240(>=0:OK) dsr=1.000(>=0.95:OK) fail=[]]
>> EXEC_DIAG: FAIL [port_ic=-0.0030 be_raw=0.0386 gap_raw=-0.0416 basket_net_bps=-36.94 fail=['port_ic_non_positive', 'port_ic_below_raw_breakeven', 'basket_net_bps_non_positive']]
```

### Analysis
*   **PASS Criteria (Phase 1)**: All criteria met (`resid_ic`, `be_eff`, `t-stat`, `bear_ic`, `dsr`).
*   **EXEC Diagnostic**: FAILED due to `port_ic` and `basket_net_bps` being negative after costs.
*   **Key Insight**: Ranking skill (IC) is significant, but transaction costs/funding drag result in negative net PnL.
