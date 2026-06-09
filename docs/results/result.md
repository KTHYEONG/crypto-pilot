# 목적
유효한 signal 들을 여러개 생성 후 기존의 1개의 전략으로만 매매하는 한계점을 보완하여 ML으로 동적으로 적재적소에 전략을 사용하여 복리자산증식 전략 도출해내는 것임

# Mode Full (ALO/Ensemble) — 최신 검증 결과

**최신 갱신:** 2026-06-09 (Signal Eval 강화 — IC t-stat 게이트 추가, q10_fail 0.90→0.65, BLOCKED 섹션 렌더링 버그 수정)
**현재 상태:** `BLOCKED (Ensemble B0)` — 2-Fold Pass (`Status: blocked`, fold_pass_ratio 2/4 < 60%). **정상 상태**
**평가 기준:** `min_variant_oos_edge_bps=10.0`, `breakeven_hard_gate=enabled`, `min_fold_realized_edge_bps=8.0`, `min_oos_rank_ic=0.01`, `min_ic_tstat=0.8`, `max_variant_oos_q10_fail_rate=0.65`, `min_wf_fold_pass_ratio=0.60`

**진단 노트 (Signal Eval 강화 결과):**
- **신규 게이트 (IC t-stat):** `min_ic_tstat=0.8` 추가. Fisher 근사 `t = IC × √(N−2) / √(1−IC²)`. **N은 OOS window 표본 수(`oos_ic_n`)** 사용 — IC가 OOS window에서 계산되므로 IS window 표본 수(`oos_n`)를 쓰면 t-stat이 부풀려짐(예: tpc_20_100 N=772→t=1.04 vs 정직한 N=537→t=0.87). 정직한 t-stat: dm_24_96(N=556, t=1.04), tpc_50_200(N=309, t=1.13), tpc_20_100(N=537, t=0.87) 모두 통과.
- **q10_fail 강화:** `max_variant_oos_q10_fail_rate: 0.90 → 0.65`. 이전 0.90은 사실상 무효 게이트. 65%로 꼬리 위험 하한 부여.
- **버그 수정 2건:** (1) `rule_diagnostics.py` 인덴테이션 오류(`recommendation_failure_report` 미설정), (2) `bridge.py` WF fold 실패 조기 반환 경로에 `failure_report` 키 누락 → BLOCKED 섹션 복구.
- **결과:** RECOMMENDED 3개(dm_24_96, tpc_50_200, tpc_20_100) 유지. BLOCKED 30개 정상 표시. Rec 컬럼 정합(tpc_50_200=Y, dm_12_48=N).
- **근본 원인:** 신호 풀이 장세 추세(momentum) 편향 → 약세(2025.10~2026.03) OOS 기간에 수익성 악화. Fold 2/4 pass.
- **다음 선택지:** (1) `min_wf_fold_pass_ratio` 완화(0.60→0.50), (2) mean-reversion/short 신호 추가, (3) 기간/파라미터 재조정.

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
| BLOCKED      | 30    | Fail Reasons: breakeven_hard_gate (25) | Event Overloa |
|              |       | Top Blocked: bcr_48, vrr_40, dm_12_48, bb_compress_20  |
| RECOMMENDED  | 3     | 1. dual_momentum:dm_24_96                               |
| (Eligible)   |       | 2. trend_pullback_continuation:tpc_50_200               |
| (Eligible)   |       | 3. trend_pullback_continuation:tpc_20_100               |
----------------------------------------------------------------------------------

[WALK-FORWARD FOLD DETAILS]
----------------------------------------------------------------------------------
| Fold | Mode       |  Rank IC |  Events | PriorP90 |  EU_p90 | Pass   |
----------------------------------------------------------------------------------
| 1    | ensemble_b0 |   -0.031 |     393 |   139.01 |  139.01 | ❌      |
| 2    | ensemble_b0 |   -0.010 |     295 |    79.51 |   79.51 | ✅      |
| 3    | ensemble_b0 |    0.051 |     341 |    93.50 |   93.50 | ✅      |
| 4    | ensemble_b0 |   -0.132 |     353 |   101.69 |  101.69 | ❌      |
----------------------------------------------------------------------------------

[BRIDGE SUMMARY] -----------------------------------
| Metric             | Value                       |
| ------------------ | --------------------------- |
| Active Signals     | 0 (sel=0)                   |
| Status             | blocked                     |
| Execution Time     | 53.39s                      |
----------------------------------------------------

[ABLATION STUDY FRONTIER] ----------------------------------------------------------------
| Model Alias        |    CAGR |   MaxDD |    MAR |     Equity | Trades | Deploy | Pass  |
| ------------------ | ------- | ------- | ------ | ---------- | ------ | ------ | ----- |
| rule_stop_risk     |  -26.5% |   17.5% |  -1.51 |    836,286 |    626 |   1.00 |   N   |
| prior_rank_stop_ri |   -5.1% |    5.6% |  -0.92 |    969,976 |    145 |   0.25 |   N   |
| prior_residual_ran |   -2.9% |    4.0% |  -0.73 |    982,905 |    122 |   0.24 |   N   |
| edge_plus_validate |   -2.9% |    4.0% |  -0.73 |    982,905 |    122 |   0.24 |   N   |
| edge_plus_gate_eve |   -2.3% |    2.2% |  -1.04 |    986,443 |    120 |   0.24 |   N   |
| full_portfolio_cap |   -0.6% |    0.9% |   0.00 |    996,712 |    118 |   0.24 |   N   |
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