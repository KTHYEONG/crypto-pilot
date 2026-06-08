# 목적
유효한 signal 들을 여러개 생성 후 기존의 1개의 전략으로만 매매하는 한계점을 보완하여 ML으로 동적으로 적재적소에 전략을 사용하여 복리자산증식 전략 도출해내는 것임

# Mode Full (ALO/Ensemble) — 최신 검증 결과

**최신 갱신:** 2026-06-09 (WF Fold Contamination Fix + Signal oos_rank_ic Gate 도입)
**현재 상태:** `READY (Ensemble B0)` — 3-Fold Pass (`Status: wf_eligible`). **불변식 미복원**(아래 참조)
**평가 기준:** `min_variant_oos_edge_bps=10.0`, `breakeven_hard_gate=enabled`, `min_fold_realized_edge_bps=8.0`, `min_oos_rank_ic=0.01`

**진단 노트 (WF Fold 오염 수정 결과):**
- **Fix A (bridge.py):** 실패 fold 예측을 0으로 censoring → 실패 fold의 anti-selection이 combined pool에 오염되지 않도록 차단. Fold 1/2 Rank IC가 0.036→**0.175/0.163**으로 대폭 개선 (오염 제거 효과 직접 확인).
- **Fix B (rule_diagnostics.py):** `min_oos_rank_ic=0.01` 게이트 연결 + `oos_rank_ic` 필드를 recommendation records에 추가. RECOMMENDED 8개→**2개** (dm_24_96, dm_12_48만 생존). `tpc_50_200` 등 IC가 recommendation window 기준 <0.01인 신호 차단.
- **Fix C (opt_main_futures.py):** "Low OOS IC" 실패 라벨 추가.
- **WF Fold 구조 변화:** Fold 3이 新규 실패(IC=-0.016). Fold 4가 Pass 전환(이전 실패). CAGR: -1.3%→**-1.2%** (미미한 개선).
- **C4 OOS Stability 개선:** rho=-0.086→**rho=+0.143** (Score 1.0/10→4.0/10). 오염 제거 후 레짐 예측력이 회복 방향.
- **잔류 이슈:** `oos_rank_ic` 게이트가 recommendation window Spearman 기준이라 sample이 적거나 binary raw_score 신호(tpc, fzs 등)가 과차단됨. min_oos_rank_ic 임계값 또는 IC 계산 window 재검토 필요.
- **구조적 블로커(미해소):** C3 flip=N (모든 레짐 동부호 long-bias). 복리 양전환은 **레짐별 부호 역전 신호 발굴이 선결**.
- 과거 기록: Regime-Alpha conditioning 反證 결과는 `docs/architecture/allocation.md` §4 및 `docs/architecture/regime.md` §1에 영구 기록.

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
| 1    | trend_pullback_continuation:tpc_... | 760 (309)    |        74.0 |    37.9% |   1.63 |  0.064 | KEEP   | N   |
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
[WF-PROF] mode=process_pool workers=4 folds=4 total_mean=29.8110s total_max=30.0962s gate_mean=0.0000s gate_max=0.0000s edge_mean=0.0040s edge_max=0.0043s dataset_mean=28.9273s dataset_max=28.9846s

[SIGNAL DIAGNOSTICS: STRATEGY FILTERING]
----------------------------------------------------------------------------------
| Action       | Count | Details / Selected Strategies                           |
----------------------------------------------------------------------------------
| BLOCKED      | 31    | Fail Reasons: breakeven_hard_gate (25) + Low OOS IC     |
|              |       | Top Blocked: bcr_48, vrr_40, tpc_50_200, bb_compress_2 |
| RECOMMENDED  | 2     | 1. dual_momentum:dm_24_96                               |
| (Eligible)   |       | 2. dual_momentum:dm_12_48                               |
----------------------------------------------------------------------------------

[WALK-FORWARD FOLD DETAILS]
----------------------------------------------------------------------------------
| Fold | Mode       |  Rank IC |  Events | PriorP90 |  EU_p90 | Pass   |
----------------------------------------------------------------------------------
| 1    | ensemble_b0 |    0.175 |     349 |    59.30 |   59.30 | ✅      |
| 2    | ensemble_b0 |    0.163 |     254 |    34.72 |   34.72 | ✅      |
| 3    | ensemble_b0 |   -0.016 |     354 |    37.86 |   37.86 | ❌      |
| 4    | ensemble_b0 |   -0.052 |     345 |    54.35 |   54.35 | ✅      |
----------------------------------------------------------------------------------

[BRIDGE SUMMARY] -----------------------------------
| Metric             | Value                       |
| ------------------ | --------------------------- |
| Active Signals     | 3977 (sel=259)              |
| Status             | wf_eligible                 |
| Execution Time     | 53.35s                      |
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
| prior_rank_stop_ri |   -3.2% |    5.7% |  -0.57 |    981,092 |    151 |   0.25 |   N   |
| prior_residual_ran |   -6.2% |    7.0% |  -0.89 |    963,245 |    150 |   0.25 |   N   |
| edge_plus_validate |   -6.2% |    7.0% |  -0.89 |    963,245 |    150 |   0.25 |   N   |
| edge_plus_gate_eve |   -2.6% |    3.0% |  -0.86 |    984,884 |    128 |   0.25 |   N   |
| full_portfolio_cap |   -1.2% |    1.4% |  -0.87 |    993,047 |    131 |   0.25 |   N   |
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
| C3 Distinctness      | 6.0/10  | kw_p=0.0000  flip=N  mi=0.0342                |
| C4 OOS Stability     | 4.0/10  | rho=+0.143  n_regimes=6                       |
| C5 Coverage          | 10.0/10 | min=0.089 max=0.224 n_eff=5.77                |
----------------------------------------------------------------------------------
| Weighted C2-C5       |  0.375  | C1(hard_gate=pass)  C6-C8(manual)             |
----------------------------------------------------------------------------------
| Occupancy            |   n/a   | bull_q=0.224  bull_vol=0.212  bear_q=0.172    |
|                      |         | bear_vol=0.141  trans=0.089  crash=0.163      |
| C3/C4_proxy (mkt)    |   n/a   | kw_p=0.000 flip=Y rho=0.886                   |
----------------------------------------------------------------------------------
[PHASE] phase=alo completed strategy/candidate evaluation only; optimization/training skipped
```