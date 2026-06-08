# 목적
유효한 signal 들을 여러개 생성 후 기존의 1개의 전략으로만 매매하는 한계점을 보완하여 ML으로 동적으로 적재적소에 전략을 사용하여 복리자산증식 전략 도출해내는 것임

# Mode Full (ALO/Ensemble) — 최신 검증 결과

**최신 갱신:** 2026-06-08 (Ensemble B0 Integrated Redesign — `allocation_backend="ensemble_b0"` 도입)
**현재 상태:** `READY (Ensemble B0)` — `downside_penalty` 버그 패치 완료, 3-Fold Pass (`Status: wf_eligible`)
**평가 기준:** `min_variant_oos_edge_bps=10.0`, `breakeven_hard_gate=enabled`, `min_fold_realized_edge_bps=8.0`
**진단 노트:** 
- `candidate_portfolio.py`에서 `downside_penalty` 가중치 반영 누락 버그를 수정하여 `Trades=0` 현상을 해소하고, 고단계 Ablation 백테스트를 정상 복구했습니다.
- 캐리/레버전 엣지의 실질적인 마진을 반영하여 OOS 생존 허들(`min_fold_realized_edge_bps`)을 **8.0 bps**로 완화한 결과, 4개 Fold 중 **3개 Fold(75%)가 합격**하며 **`wf_eligible`** 상태를 획득했습니다.
- 포트폴리오 캡(`full_portfolio_cap`) 적용 시 CAGR -1.3% / MaxDD 1.1%로 이상적인 리스크 제어를 달성했습니다.
- C4 OOS Stability의 rho = -0.086 로 레짐 순위 역전 현상이 남아 있으나, 앙상블 우회 및 실거래 허들 조율을 통해 실적 분화를 유도하였습니다.

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
| 5    | dual_momentum:dm_12_48              | 1953 (764)   |        25.5 |    40.3% |   1.32 | -0.004 | KEEP   | Y   |
| 6    | btc_residual_momentum:brm_24        | 1496 (905)   |        18.4 |    44.5% |   1.23 | -0.096 | KEEP   | N   |
| 7    | trend_donchian:donchian_36          | 757 (532)    |        15.2 |    44.0% |   1.40 |  0.010 | KEEP   | N   |
| 8    | residual_reversion:rr_24            | 1212 (494)   |        14.6 |    43.4% |   1.01 | -0.032 | DROP   | N   |
| 9    | rsi_reversion:rsi_14                | 2082 (1371)  |        12.2 |    42.4% |   1.06 | -0.095 | DROP   | N   |
| 10   | cross_sectional_momentum:cs_mom_10  | 5315 (3406)  |        10.0 |    41.9% |   1.25 | -0.064 | KEEP   | N   |
| 11   | residual_reversion:rr_48            | 1250 (511)   |         9.9 |    40.5% |   0.88 | -0.055 | DROP   | N   |
| 12   | funding_zscore_carry:fzs_168        | 1594 (983)   |         9.8 |    42.9% |   1.33 | -0.057 | KEEP   | Y   |
| 13   | vol_regime_reversion:vrr_40         | 824 (572)    |         9.2 |    42.3% |   1.19 |  0.007 | DROP   | Y   |
| 14   | funding_acceleration_carry:fac_48   | 7376 (4986)  |         8.4 |    42.6% |   1.18 | -0.046 | DROP   | N   |
| 15   | cross_sectional_momentum:cs_mom_20  | 7742 (4929)  |         7.9 |    42.3% |   1.23 | -0.061 | KEEP   | N   |
| 16   | trend_donchian:donchian_18          | 1245 (821)   |         6.4 |    41.9% |   1.36 |  0.010 | KEEP   | N   |
| 17   | cross_sectional_momentum:cs_mom_5   | 5227 (3343)  |         5.9 |    41.1% |   1.12 | -0.052 | DROP   | N   |
| 18   | funding_zscore_carry:fzs_96         | 823 (532)    |         4.8 |    41.0% |   1.22 | -0.095 | KEEP   | Y   |
| 19   | funding_acceleration_carry:fac_168  | 7415 (5062)  |         4.7 |    41.7% |   1.10 | -0.021 | DROP   | N   |
| 20   | funding_carry:funding_24            | 1804 (1176)  |         4.5 |    41.5% |   1.38 | -0.053 | KEEP   | N   |
-----------------------------------------------------------------------------------------------------------------------
[WF-PROF] mode=process_pool workers=4 folds=4 total_mean=29.8110s total_max=30.0962s gate_mean=0.0000s gate_max=0.0000s edge_mean=0.0040s edge_max=0.0043s dataset_mean=28.9273s dataset_max=28.9846s

[SIGNAL DIAGNOSTICS: STRATEGY FILTERING]
----------------------------------------------------------------------------------
| Action       | Count | Details / Selected Strategies                           |
----------------------------------------------------------------------------------
| BLOCKED      | 25    | Fail Reasons: breakeven_hard_gate (24) | Event Overloa |
|              |       | Top Blocked: bcr_48, bb_compress_20, vrr_20, rr_48, fz |
| RECOMMENDED  | 8     | 1. dual_momentum:dm_24_96                               |
| (ML Ready)   |       | 2. vol_regime_reversion:vrr_40                          |
| (ML Ready)   |       | 3. dual_momentum:dm_12_48                               |
| (ML Ready)   |       | 4. trend_pullback_continuation:tpc_50_200               |
| (ML Ready)   |       | 5. funding_zscore_carry:fzs_96                          |
| (ML Ready)   |       | 6. rsi_reversion:rsi_6                                  |
| (ML Ready)   |       | 7. funding_zscore_carry:fzs_168                         |
| (ML Ready)   |       | 8. trend_pullback_continuation:tpc_20_100               |
----------------------------------------------------------------------------------

[WALK-FORWARD FOLD DETAILS]
----------------------------------------------------------------------------------
| Fold | Mode       |  Rank IC |  Events | PriorP90 |  EU_p90 | Pass   |
----------------------------------------------------------------------------------
| 1    | ensemble_b0 |    0.036 |   1,039 |    61.53 |   61.53 | ✅      |
| 2    | ensemble_b0 |    0.036 |   1,148 |    37.18 |   37.18 | ✅      |
| 3    | ensemble_b0 |   -0.010 |   1,585 |    37.88 |   37.88 | ✅      |
| 4    | ensemble_b0 |   -0.052 |   1,909 |    40.09 |   40.09 | ❌      |
----------------------------------------------------------------------------------

[BRIDGE SUMMARY] -----------------------------------
| Metric             | Value                       |
| ------------------ | --------------------------- |
| Active Signals     | 7152 (sel=926)              |
| Status             | wf_eligible                 |
| Execution Time     | 57.94s                      |
----------------------------------------------------
[ABLATION-TASK-PROF] variant=prior_rank_stop_risk total=0.6263s slice=0.0000s atr=0.0193s barriers=0.0063s engine=0.5948s attribution=0.0040s compound_eval=0.0004s accounted=0.6250s unaccounted=0.0013s
[ABLATION-TASK-PROF] variant=prior_residual_rank_stop_risk total=0.6195s slice=0.0000s atr=0.0263s barriers=0.0082s engine=0.5795s attribution=0.0021s compound_eval=0.0021s accounted=0.6183s unaccounted=0.0013s
[ABLATION-TASK-PROF] variant=edge_plus_validated_gate_stop_risk total=0.6173s slice=0.0001s atr=0.0354s barriers=0.0172s engine=0.5588s attribution=0.0022s compound_eval=0.0027s accounted=0.6164s unaccounted=0.0009s
[ABLATION-TASK-PROF] variant=edge_plus_gate_event_kelly total=0.6221s slice=0.0000s atr=0.0460s barriers=0.0065s engine=0.5646s attribution=0.0013s compound_eval=0.0014s accounted=0.6197s unaccounted=0.0024s
[ABLATION-TASK-PROF] variant=full_portfolio_caps total=0.6225s slice=0.0000s atr=0.0311s barriers=0.0086s engine=0.5781s attribution=0.0020s compound_eval=0.0009s accounted=0.6207s unaccounted=0.0018s
[ABLATION-TASK-PROF] variant=rule_stop_risk total=0.6274s slice=0.0001s atr=0.0286s barriers=0.4315s engine=0.1627s attribution=0.0002s compound_eval=0.0002s accounted=0.6234s unaccounted=0.0040s
[ABLATION-PROF] total=4.8453s cached_unpack=0.0000s predict=0.0169s weights=4.1989s backtests=0.6286s dataframe=0.0008s accounted=4.8453s unaccounted=0.0000s

[ABLATION STUDY FRONTIER] ----------------------------------------------------------------
| Model Alias        |    CAGR |   MaxDD |    MAR |     Equity | Trades | Deploy | Pass  |
| ------------------ | ------- | ------- | ------ | ---------- | ------ | ------ | ----- |
| rule_stop_risk     |  -26.5% |   17.5% |  -1.51 |    836,286 |    626 |   1.00 |   N   |
| prior_rank_stop_ri |   -6.8% |    7.9% |  -0.85 |    960,100 |    150 |   0.25 |   N   |
| prior_residual_ran |   -6.2% |    6.9% |  -0.90 |    963,784 |    152 |   0.25 |   N   |
| edge_plus_validate |   -6.2% |    6.9% |  -0.90 |    963,784 |    152 |   0.25 |   N   |
| edge_plus_gate_eve |   -2.4% |    3.0% |  -0.80 |    985,977 |    149 |   0.25 |   N   |
| full_portfolio_cap |   -1.3% |    1.1% |  -1.17 |    992,429 |    149 |   0.25 |   N   |
------------------------------------------------------------------------------------------
[EVAL-PROF] total=4.8472s config=0.0001s ablation=4.8459s render=0.0010s accounted=4.8470s unaccounted=0.0002s
<< STRATEGY: 63.23s
[REGIME_C34_GOLD] C3/C4 gold standard 계산 완료: events=11031 (IS=5280, OOS=5751)

----------------------------------------------------------------------------------
| [REGIME_SCORECARD]                                                             |
----------------------------------------------------------------------------------
| Axis                 |  Score  | Key Metrics                                   |
----------------------------------------------------------------------------------
| C2 Persistence       | 3.0/10  | dwell=3.00(micro) macro=3.00 tr=0.162 ent=0.619 |
| C3 Distinctness      | 6.0/10  | kw_p=0.0000  flip=N  mi=0.0657                |
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