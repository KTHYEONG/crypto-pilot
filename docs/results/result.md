# 목적
유효한 signal 들을 여러개 생성 후 기존의 1개의 전략으로만 매매하는 한계점을 보완하여 ML으로 동적으로 적재적소에 전략을 사용하여 복리자산증식 전략 도출해내는 것임

# Mode Full (ALO/Ensemble) — 최신 검증 결과

**최신 갱신:** 2026-06-09 (Regime-Cell Admission 활성화 — `FUTURES_CANDIDATE_REGIME_CELL_ADMISSION_ENABLED=True`)
**현재 상태:** `wf_eligible` — 3-Fold Pass (fold_pass_ratio 3/4 ≥ 60%). **BLOCKED → wf_eligible 전환**
**평가 기준:** `min_variant_oos_edge_bps=10.0`, `breakeven_hard_gate=enabled`, `min_fold_realized_edge_bps=8.0`, `min_oos_rank_ic=0.01`, `min_ic_tstat=0.8`, `max_variant_oos_q10_fail_rate=0.65`, `min_wf_fold_pass_ratio=0.60`
**Regime-Cell 파라미터:** `min_regime_cell_oos_obs=60`, `min_regime_cell_edge_bps=8.0`, `min_regime_cell_tstat=1.0`, `max_admitted_cells_per_variant=2`

**진단 노트 (Regime-Cell Admission 활성화 결과):**
- **RECOMMENDED 3→10종:** dm_24_96, tpc_50_200, tpc_20_100(추세) + dm_12_48, fzs_96, rsi_6, fzs_168, fzs_48, brm_48, donchian_18(cell-admitted). carry(`fzs_*`)/momentum/reversion 계열 혼합 풀 달성.
- **BLOCKED 30→19종:** 11개 구제. 글로벌 평균으로 탈락했던 carry/reversion 신호들이 특정 regime cell(bull_quiet 등)에서 edge 충족으로 admission.
- **WF fold Fold1 반전:** −16.1bps(❌) → +20.4bps(✅). 풀 다양화로 약세장 출혈 해소.
- **Active Signals 0→4960 (sel=662):** 실제 배포 가능 이벤트 생성.
- **Fold4 미통과(정상):** RlzdMean=10.3bps(≥8.0 통과)·selected_count 충족이나 **`ml_lift_bps>0` 실패** — ensemble 선택이 baseline 전체평균을 못 이김(선택 스킬 부재). 버그 아닌 설계대로의 3차 게이트. 3/4 fold pass로 `wf_eligible`.
- 상세 주의사항은 문서 최하단 **[주의사항]** 참조.

---

## Symbols
```text
[SELECTED SYMBOLS] ---------------------------------
| 1000FLOKIUSDT, 1000PEPEUSDT, 1000SATSUSDT, 1000SHIBUSDT, 1000XECUSDT, AAVEUSDT |
| ADAUSDT, ANKRUSDT, APEUSDT, API3USDT, ARBUSDT, ARPAUSDT |
| ARUSDT , ASTRUSDT, ATOMUSDT, AUCTIONUSDT, AVAXUSDT, AXSUSDT |
| BAKEUSDT, BANDUSDT, BCHUSDT, BIGTIMEUSDT, BLZUSDT, BNBUSDT |
| BNXUSDT, BTCUSDT, CKBUSDT, CRVUSDT, DOGEUSDT, DOTUSDT |
| DYDXUSDT, EIGENUSDT, ENSUSDT, ETCUSDT, ETHUSDT, FILUSDT |
| FLMUSDT, FTMUSDT, GALAUSDT, GRTUSDT, ICPUSDT, IOTAUSDT |
| IPUSDT , JASMYUSDT, JTOUSDT, KAITOUSDT, KAVAUSDT, LDOUSDT |
| LEVERUSDT, LINAUSDT, LINKUSDT, LPTUSDT, LTCUSDT, LUNA2USDT |
| MANAUSDT, MKRUSDT, MOODENGUSDT, MTLUSDT, NEARUSDT, NEOUSDT |
| OCEANUSDT, OPUSDT , PEOPLEUSDT, POPCATUSDT, REEFUSDT, RSRUSDT |
| RUNEUSDT, RVNUSDT, SANDUSDT, SEIUSDT, SNXUSDT, SOLUSDT |
| STMXUSDT, STORJUSDT, SXPUSDT, THETAUSDT, TRBUSDT, TRXUSDT |
| UNFIUSDT, UNIUSDT, VETUSDT, VIDTUSDT, WAVESUSDT, WIFUSDT |
| WLDUSDT, XLMUSDT, XRPUSDT, XVGUSDT, YGGUSDT, ZECUSDT |
| ZENUSDT, ZETAUSDT, ZILUSDT, ZRXUSDT              |
----------------------------------------------------
```

## 최신 실행 요약 (4h Timeframe - ALO Phase)

```text
[WINDOW] -------------------------------------------
| Property           | Value                       |
| ------------------ | --------------------------- |
| Range              | 2022-10-01 ~ 2026-03-31     |
| IS Start           | 2023-10-01                  |
| OOS Start          | 2025-10-01                  |
| Elapsed            | 0.00s                       |
----------------------------------------------------

[UNIVERSE REPORT] ----------------------------------
| Metric             | Value                       |
| ------------------ | --------------------------- |
| Selected (Stg6)    | 20                          |
| Panels (Inf/Live)  | 94 / 20                     |
| Windows (Inf)      | 10                          |
----------------------------------------------------
[UNIVERSE] Discovery complete: 94 symbols (2.34s)

[SELECTED SYMBOLS] ---------------------------------
| 1000FLOKIUSDT, 1000PEPEUSDT, 1000SATSUSDT, 1000SHIBUSDT, 1000XECUSDT, AAVEUSDT |
| ADAUSDT, ANKRUSDT, APEUSDT, API3USDT, ARBUSDT, ARPAUSDT |
| ARUSDT , ASTRUSDT, ATOMUSDT, AUCTIONUSDT, AVAXUSDT, AXSUSDT |
| BAKEUSDT, BANDUSDT, BCHUSDT, BIGTIMEUSDT, BLZUSDT, BNBUSDT |
| BNXUSDT, BTCUSDT, CKBUSDT, CRVUSDT, DOGEUSDT, DOTUSDT |
| DYDXUSDT, EIGENUSDT, ENSUSDT, ETCUSDT, ETHUSDT, FILUSDT |
| FLMUSDT, FTMUSDT, GALAUSDT, GRTUSDT, ICPUSDT, IOTAUSDT |
| IPUSDT , JASMYUSDT, JTOUSDT, KAITOUSDT, KAVAUSDT, LDOUSDT |
| LEVERUSDT, LINAUSDT, LINKUSDT, LPTUSDT, LTCUSDT, LUNA2USDT |
| MANAUSDT, MKRUSDT, MOODENGUSDT, MTLUSDT, NEARUSDT, NEOUSDT |
| OCEANUSDT, OPUSDT , PEOPLEUSDT, POPCATUSDT, REEFUSDT, RSRUSDT |
| RUNEUSDT, RVNUSDT, SANDUSDT, SEIUSDT, SNXUSDT, SOLUSDT |
| STMXUSDT, STORJUSDT, SXPUSDT, THETAUSDT, TRBUSDT, TRXUSDT |
| UNFIUSDT, UNIUSDT, VETUSDT, VIDTUSDT, WAVESUSDT, WIFUSDT |
| WLDUSDT, XLMUSDT, XRPUSDT, XVGUSDT, YGGUSDT, ZECUSDT |
| ZENUSDT, ZETAUSDT, ZILUSDT, ZRXUSDT              |
----------------------------------------------------
[CACHE] Backfill: 2022-10-01 ~ 2026-03-31 | Symbols: 94 | Last: 2026-04-01
Sync mode=full targeted_symbols=94
[SYNC-COVERAGE] rows=1 file=/home/kth/my_coin_traider/logs/futures/universe/sync_coverage_report.parquet
Ledger update complete.

[DATA QUALITY] -------------------------------------
| Metric             | Value                       |
| ------------------ | --------------------------- |
| Symbols (Req/Load) | 94 / 91 (96.8%)             |
| Kept (Ready)       | 63                          |
| Fail Reasons       | fetch_window_short:28       |
----------------------------------------------------

[STRATEGY: candidate_ml] ---------------------------
| Component          | Status/Value                |
| ------------------ | --------------------------- |
| Inf Panel          | 63 symbols                  |
| Live Panel         | 12 symbols                  |
| Trade Symbols      | 20                          |
----------------------------------------------------

[CANDIDATE TOP STRATEGIES] --------------------------------------------------------------------------------------------
| Rank | Strategy Name                       | Sample (OOS) | Profit(bps) | Win Rate |    P/L |  Score | Action | Rec |
| ---- | ----------------------------------- | ------------ | ----------- | -------- | ------ | ------ | ------ | --- |
| 1    | trend_pullback_continuation:tpc_... | 760 (309)    |        74.0 |    37.9% |   1.63 |  0.064 | KEEP   | Y   |
| 2    | dual_momentum:dm_24_96              | 1373 (556)   |        68.1 |    42.6% |   1.25 |  0.044 | KEEP   | Y   |
| 3    | btc_corr_regime:bcr_48              | 107 (77)     |        66.4 |    53.2% |   1.94 |  0.039 | DROP   | N   |
| 4    | btc_corr_regime:bcr_96              | 47 (40)      |        64.7 |    52.5% |   2.34 | -0.017 | DROP   | N   |
| 5    | dual_momentum:dm_12_48              | 1953 (764)   |        25.5 |    40.3% |   1.32 | -0.004 | KEEP   | N   |
| 6    | btc_residual_momentum:brm_24        | 1496 (905)   |        18.4 |    44.5% |   1.23 | -0.096 | KEEP   | N   |
| 7    | trend_donchian:donchian_36          | 757 (532)    |        15.2 |    44.0% |   1.40 |  0.010 | KEEP   | N   |
| 8    | residual_reversion:rr_24            | 1212 (494)   |        14.6 |    43.4% |   1.01 | -0.032 | DROP   | N   |
| 9    | rsi_reversion:rsi_14                | 2082 (1371)  |        12.2 |    42.4% |   1.06 | -0.095 | DROP   | N   |
| 10   | cross_sectional_momentum:cs_mom_10  | 5315 (3406)  |        10.0 |    41.9% |   1.25 | -0.064 | KEEP   | N   |
| 11   | residual_reversion:rr_48            | 1250 (511)   |         9.9 |    40.5% |   0.88 | -0.055 | DROP   | N   |
| 12   | funding_zscore_carry:fzs_168        | 1594 (983)   |         9.8 |    42.9% |   1.33 | -0.057 | KEEP   | N   |
| 13   | vol_regime_reversion:vrr_40         | 824 (572)    |         9.2 |    42.3% |   1.19 |  0.007 | DROP   | N   |
| 14   | funding_acceleration_carry:fac_48   | 7376 (4986)  |         8.4 |    42.6% |   1.18 | -0.046 | DROP   | N   |
| 15   | cross_sectional_momentum:cs_mom_20  | 7742 (4929)  |         7.9 |    42.3% |   1.23 | -0.061 | KEEP   | N   |
| 16   | trend_donchian:donchian_18          | 1245 (821)   |         6.4 |    41.9% |   1.36 |  0.010 | KEEP   | N   |
| 17   | cross_sectional_momentum:cs_mom_5   | 5227 (3343)  |         5.9 |    41.1% |   1.12 | -0.052 | DROP   | N   |
| 18   | funding_zscore_carry:fzs_96         | 823 (532)    |         4.8 |    41.0% |   1.22 | -0.095 | KEEP   | N   |
| 19   | funding_acceleration_carry:fac_168  | 7415 (5062)  |         4.7 |    41.7% |   1.10 | -0.021 | DROP   | N   |
| 20   | funding_carry:funding_24            | 1804 (1176)  |         4.5 |    41.5% |   1.38 | -0.053 | KEEP   | N   |
-----------------------------------------------------------------------------------------------------------------------

[SIGNAL DIAGNOSTICS: STRATEGY FILTERING]
----------------------------------------------------------------------------------
| Action       | Count | Details / Selected Strategies                           |
----------------------------------------------------------------------------------
| BLOCKED      | 19    | Fail Reasons: Breakeven Gate (10) | Event Overload (8)  |
|              |       |               Poor Hit/Payoff (6) | Low IC t-stat (11)  |
|              |       |               Low Mean Edge (6) | Low Median (5)        |
|              |       |               Low Obs (2) | Low OOS IC (9)              |
|              |       | Top Blocked: bcr_48, vrr_40, bb_compress_20, vrr_20, fa |
| RECOMMENDED  | 10    | 1. dual_momentum:dm_24_96                               |
| (Eligible)   |       | 2. dual_momentum:dm_12_48                               |
| (Eligible)   |       | 3. trend_pullback_continuation:tpc_50_200               |
| (Eligible)   |       | 4. funding_zscore_carry:fzs_96                          |
| (Eligible)   |       | 5. rsi_reversion:rsi_6                                  |
| (Eligible)   |       | 6. funding_zscore_carry:fzs_168                         |
| (Eligible)   |       | 7. trend_pullback_continuation:tpc_20_100               |
| (Eligible)   |       | 8. funding_zscore_carry:fzs_48                          |
| (Eligible)   |       | 9. btc_residual_momentum:brm_48                         |
| (Eligible)   |       | 10. trend_donchian:donchian_18                           |
----------------------------------------------------------------------------------

[GATE FAILURES: PER-VARIANT]
----------------------------------------------------------------------------------
| Variant                        | Action           | Failed Gates / Cells       |
----------------------------------------------------------------------------------
| btc_corr_regime:bcr_48         | INSUFFICIENT_OBS | Low Obs, Breakeven Gate, … |
| vol_regime_reversion:vrr_40    | KEEP_CANDIDATE   | Low OOS IC, Low IC t-stat  |
| vol_breakout:bb_compress_20    | INSUFFICIENT_OBS | Breakeven Gate, Low OOS I… |
| vol_regime_reversion:vrr_20    | INSUFFICIENT_OBS | Breakeven Gate, Low Media… |
| funding_acceleration_carry:fac | DROP_OR_REWORK   | Event Overload             |
| bollinger_reversion:bollinger_ | DROP_OR_REWORK   | Breakeven Gate, Poor Hit/… |
| funding_carry:funding_24       | KEEP_CANDIDATE   | Breakeven Gate, Low Mean … |
| cross_sectional_momentum:cs_mo | DROP_OR_REWORK   | Event Overload             |
| btc_corr_regime:bcr_24         | INSUFFICIENT_OBS | Breakeven Gate, Low Mean … |
| funding_acceleration_carry:fac | DROP_OR_REWORK   | Event Overload             |
| cross_sectional_momentum:cs_mo | DROP_OR_REWORK   | Event Overload             |
| btc_residual_momentum:brm_24   | DROP_OR_REWORK   | Breakeven Gate, Low Mean … |
| trend_ma:ema_18_108            | DROP_OR_REWORK   | Event Overload             |
| cross_sectional_momentum:cs_mo | DROP_OR_REWORK   | Event Overload             |
| trend_ma:ema_12_72             | DROP_OR_REWORK   | Event Overload             |
| trend_ma:ema_6_36              | DROP_OR_REWORK   | Event Overload             |
| trend_donchian:donchian_36     | DROP_OR_REWORK   | Breakeven Gate, Low Mean … |
| trend_donchian:donchian_72     | INSUFFICIENT_OBS | Breakeven Gate, Low Mean … |
| btc_corr_regime:bcr_96         | INSUFFICIENT_OBS | Low Obs, Breakeven Gate, … |
----------------------------------------------------------------------------------

[WALK-FORWARD FOLD DETAILS]
----------------------------------------------------------------------------------
| Fold | Mode       | IC(diag) |  Events | RlzdMean |  EU_p90 | Pass   |
|      |            |    (ref) |         |  (★gate) | (★gate) |        |
----------------------------------------------------------------------------------
| 1    | ensemble_b0 |    0.023 |   1,300 |     20.4 |   59.94 | ✅      |
| 2    | ensemble_b0 |   -0.015 |   1,736 |      9.8 |   33.06 | ✅      |
| 3    | ensemble_b0 |   -0.034 |   2,158 |     25.3 |   36.00 | ✅      |
| 4    | ensemble_b0 |   -0.027 |   2,679 |     10.3 |   38.04 | ❌      |
----------------------------------------------------------------------------------

[BRIDGE SUMMARY] -----------------------------------
| Metric             | Value                       |
| ------------------ | --------------------------- |
| Active Signals     | 4960 (sel=662)              |
| Status             | wf_eligible                 |
| Execution Time     | 59.00s                      |
----------------------------------------------------

[ABLATION STUDY FRONTIER] ----------------------------------------------------------------
| Model Alias        |    CAGR |   MaxDD |    MAR |     Equity | Trades | Deploy | Pass  |
| ------------------ | ------- | ------- | ------ | ---------- | ------ | ------ | ----- |
| rule_stop_risk     |  -26.5% |   17.5% |  -1.51 |    836,286 |    626 |   1.00 |   N   |
| prior_rank_stop_ri |   -7.5% |    6.7% |  -1.12 |    955,633 |    146 |   0.25 |   N   |
| prior_residual_ran |   -5.8% |    5.9% |  -0.99 |    965,833 |    151 |   0.25 |   N   |
| edge_plus_validate |   -5.8% |    5.9% |  -0.99 |    965,833 |    151 |   0.25 |   N   |
| edge_plus_gate_eve |   -2.2% |    3.0% |  -0.72 |    987,358 |    148 |   0.25 |   N   |
| full_portfolio_cap |   -0.9% |    1.1% |  -0.84 |    994,599 |    149 |   0.25 |   N   |
------------------------------------------------------------------------------------------

[REGIME_C34_GOLD] C3/C4 gold standard 계산 완료: events=3480 (IS=2076, OOS=1404)

----------------------------------------------------------------------------------
| [REGIME_SCORECARD]                                                             |
----------------------------------------------------------------------------------
| Axis                 |  Score  | Key Metrics                                   |
----------------------------------------------------------------------------------
| C2 Persistence       | 3.0/10  | dwell=3.00(micro) macro=3.00 tr=0.162 ent=0.619 |
| C3 Distinctness      | 6.0/10  | kw_p=0.0000  flip=N  mi=0.0382                |
| C4 OOS Stability     | 1.0/10  | rho=-0.086  n_regimes=6                       |
| C5 Coverage          | 10.0/10 | min=0.089 max=0.224 n_eff=5.77                |
----------------------------------------------------------------------------------
| Weighted C2-C5       |  0.315  | C1(hard_gate=pass)  C6-C8(manual)             |
----------------------------------------------------------------------------------
| Occupancy            |   n/a   | bull_q=0.224  bull_vol=0.212  bear_q=0.172    |
|                      |         | bear_vol=0.141  trans=0.089  crash=0.163      |
| C3/C4_proxy (mkt)    |   n/a   | kw_p=0.000 flip=Y rho=0.886                   |
----------------------------------------------------------------------------------
[PHASE] phase=alo completed strategy/candidate evaluation only; optimization/training skipped
```

---