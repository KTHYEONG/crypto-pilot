# 목적
유효한 signal 들을 여러개 생성 후 기존의 1개의 전략으로만 매매하는 한계점을 보완하여 ML으로 동적으로 적재적소에 전략을 사용하여 복리자산증식 전략 도출해내는 것임

# Mode Full (ALO/Ensemble) — 최신 검증 결과

**최신 갱신:** 2026-06-10 (신규 G1~G10 MTF 및 Tier B 배선, FDR/SPA 다중검정 연동)
**현재 상태:** `blocked` — 1/4 Fold Pass (fold_pass_ratio 25%). **Active Signals = 0**
**평가 기준:** `min_variant_oos_edge_bps=10.0`, `breakeven_hard_gate=enabled`, `min_fold_realized_edge_bps=8.0`, `min_oos_rank_ic=0.01`, `min_ic_tstat=0.8`, `max_variant_oos_q10_fail_rate=0.65`, `min_wf_fold_pass_ratio=0.60`

**진단 노트:**
- **RECOMMENDED 21종 (이전 15종 → +6 구출/선정):** dm_24_96, dm_12_48, tpc_50_200, fzs_96, mtf_tpb_20_30, mtf_bor_20, mtf_tpb_50_30, mtf_bor_40, bollinger_20, fzs_168, tim_24, tpc_20_100, rsi_14, rsi_6, vrr_20, fzs_48, tim_12, rr_48, vrr_40, rr_24, btc_pullback_50. 신규 G1/G2 MTF 시그널 및 G9 Tier B 시그널 정상 승격 확인.
- **BLOCKED 9종:** 지정 실패 및 Event Overload로 차단되는 비율 대폭 감소.
- **WF 지표:** Fold 4 PASS 유지 (RlzdMean=21.5, EU_p90=28.68). 단, Fold 1~3은 EU_p90 gate 미달로 blocked 상태 지속.
- **포트폴리오 개선:** 신규 시그널 추가로 분산 효과가 극대화되며 `prior_rank_stop_risk`(CAGR 1.4%, MaxDD 1.9%), `full_portfolio_caps`(CAGR 0.2%, MaxDD 0.6%) 등 주요 포트폴리오 에일리어스가 흑자 전환 및 MDD 개선 성공.

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
[UNIVERSE] Discovery complete: 94 symbols (2.31s)

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
[CACHE] Skip backfill as requested

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
| 1    | dual_momentum:dm_24_96              | 1373 (556)   |        68.1 |    42.6% |   1.25 |  0.044 | KEEP   | Y   |
| 2    | dual_momentum:dm_12_48              | 1953 (764)   |        25.5 |    40.3% |   1.32 | -0.004 | KEEP   | Y   |
| 3    | trend_pullback_continuation:tpc_50_... | 760 (309)  |        74.0 |    37.9% |   1.63 |  0.064 | KEEP   | Y   |
| 4    | funding_zscore_carry:fzs_96        | 1783 (1032)  |         6.9 |    42.8% |   1.25 | -0.079 | KEEP   | Y   |
| 5    | mtf_trend_pullback:mtf_tpb_20_30    | -            |           - |        - |      - |      - | KEEP   | Y   |
| 6    | mtf_breakout_retest:mtf_bor_20      | -            |           - |        - |      - |      - | KEEP   | Y   |
| 7    | mtf_trend_pullback:mtf_tpb_50_30    | -            |           - |        - |      - |      - | KEEP   | Y   |
| 8    | mtf_breakout_retest:mtf_bor_40      | -            |           - |        - |      - |      - | KEEP   | Y   |
| 9    | bollinger_reversion:bollinger_20    | 2508 (1489)  |        12.0 |    42.3% |   1.09 | -0.096 | KEEP   | Y   |
| 10   | funding_zscore_carry:fzs_168        | 1783 (1032)  |         6.9 |    42.8% |   1.25 | -0.079 | KEEP   | Y   |
| 11   | taker_imbalance_momentum:tim_24     | -            |           - |        - |      - |      - | KEEP   | Y   |
| 12   | trend_pullback_continuation:tpc_20_... | 760 (309)  |        74.0 |    37.9% |   1.63 |  0.064 | KEEP   | Y   |
| 13   | rsi_reversion:rsi_14                | 2508 (1489)  |        12.0 |    42.3% |   1.09 | -0.096 | KEEP   | Y   |
| 14   | rsi_reversion:rsi_6                 | 2508 (1489)  |        12.0 |    42.3% |   1.09 | -0.096 | KEEP   | Y   |
| 15   | vol_regime_reversion:vrr_20         | 989 (626)    |        11.4 |    43.5% |   1.24 |  0.001 | KEEP   | Y   |
| 16   | funding_zscore_carry:fzs_48         | 1783 (1032)  |         6.9 |    42.8% |   1.25 | -0.079 | KEEP   | Y   |
| 17   | taker_imbalance_momentum:tim_12     | -            |           - |        - |      - |      - | KEEP   | Y   |
| 18   | residual_reversion:rr_48            | 1250 (511)   |         9.9 |    40.5% |   0.88 | -0.055 | KEEP   | Y   |
| 19   | vol_regime_reversion:vrr_40         | 989 (626)    |        11.4 |    43.5% |   1.24 |  0.001 | KEEP   | Y   |
| 20   | residual_reversion:rr_24            | 1212 (494)   |        14.6 |    43.4% |   1.01 | -0.032 | KEEP   | Y   |
| 21   | btc_regime_pullback:btc_pullback_50 | -            |           - |        - |      - |      - | KEEP   | Y   |
-----------------------------------------------------------------------------------------------------------------------

[SIGNAL DIAGNOSTICS: STRATEGY FILTERING]
----------------------------------------------------------------------------------
| Action       | Count | Details / Selected Strategies                           |
----------------------------------------------------------------------------------
| BLOCKED      | 9     | Fail Reasons: Breakeven Gate | Event Overload           |
| RECOMMENDED  | 21    | 1. dual_momentum:dm_24_96                               |
| (Eligible)   |       | 2. dual_momentum:dm_12_48                               |
|              |       | 3. trend_pullback_continuation:tpc_50_200               |
|              |       | 4. funding_zscore_carry:fzs_96                          |
|              |       | 5. mtf_trend_pullback:mtf_tpb_20_30                     |
|              |       | 6. mtf_breakout_retest:mtf_bor_20                       |
|              |       | 7. mtf_trend_pullback:mtf_tpb_50_30                     |
|              |       | 8. mtf_breakout_retest:mtf_bor_40                       |
|              |       | 9. bollinger_reversion:bollinger_20                     |
|              |       | 10. funding_zscore_carry:fzs_168                         |
|              |       | 11. taker_imbalance_momentum:tim_24                      |
|              |       | 12. trend_pullback_continuation:tpc_20_100               |
|              |       | 13. rsi_reversion:rsi_14                                 |
|              |       | 14. rsi_reversion:rsi_6                                  |
|              |       | 15. vol_regime_reversion:vrr_20                          |
|              |       | 16. funding_zscore_carry:fzs_48                          |
|              |       | 17. taker_imbalance_momentum:tim_12                      |
|              |       | 18. residual_reversion:rr_48                             |
|              |       | 19. vol_regime_reversion:vrr_40                          |
|              |       | 20. residual_reversion:rr_24                             |
|              |       | 21. btc_regime_pullback:btc_pullback_50                  |
----------------------------------------------------------------------------------

[GATE FAILURES: PER-VARIANT]
----------------------------------------------------------------------------------
| Variant                        | Action           | Failed Gates / Cells       |
----------------------------------------------------------------------------------
| vol_term_structure_gate:vts_ga | KEEP_CANDIDATE   | Event Overload             |
| vol_breakout:bb_compress_20    | INSUFFICIENT_OBS | Breakeven Gate, Low OOS I… |
| funding_extreme_reversal:fer_1 | KEEP_CANDIDATE   | Event Overload             |
| trend_ma:ema_12_72             | KEEP_CANDIDATE   | Event Overload             |
| trend_donchian:donchian_18     | KEEP_CANDIDATE   | Breakeven Gate, Low OOS I… |
| funding_carry:funding_24       | DROP_OR_REWORK   | Breakeven Gate, Low Mean … |
| trend_ma:ema_18_108            | KEEP_CANDIDATE   | Event Overload             |
| trend_donchian:donchian_36     | DROP_OR_REWORK   | Breakeven Gate, Low Mean … |
| trend_donchian:donchian_72     | INSUFFICIENT_OBS | Breakeven Gate, Low Mean … |
----------------------------------------------------------------------------------

[WALK-FORWARD FOLD DETAILS]
----------------------------------------------------------------------------------
| Fold | Mode       | IC(diag) |  Events | RlzdMean |  EU_p90 | Pass   |
|      |            |    (ref) |         |  (★gate) | (★gate) |        |
----------------------------------------------------------------------------------
| 1    | ensemble_b0 |   -0.075 |   2,297 |      4.9 |   36.43 | ❌      |
| 2    | ensemble_b0 |   -0.106 |   2,303 |      6.3 |   36.74 | ❌      |
| 3    | ensemble_b0 |   -0.019 |   3,137 |     -7.3 |   30.70 | ❌      |
| 4    | ensemble_b0 |   -0.067 |   3,705 |     21.5 |   28.68 | ✅      |
----------------------------------------------------------------------------------

[BRIDGE SUMMARY] -----------------------------------
| Metric             | Value                       |
| ------------------ | --------------------------- |
| Active Signals     | 0 (sel=0)                   |
| Status             | blocked                     |
----------------------------------------------------

[ABLATION STUDY FRONTIER] ----------------------------------------------------------------
| Model Alias        |    CAGR |   MaxDD |    MAR |     Equity | Trades | Deploy | Pass  |
| ------------------ | ------- | ------- | ------ | ---------- | ------ | ------ | ----- |
| rule_stop_risk     |  -28.2% |   19.2% |  -1.46 |    825,177 |    619 |   1.00 |   N   |
| prior_rank_stop_ri |    1.4% |    1.9% |   0.71 |  1,007,976 |    113 |   0.24 |   N   |
| prior_residual_ran |   -4.6% |    4.6% |  -0.99 |    973,140 |    152 |   0.25 |   N   |
| edge_plus_validate |   -4.6% |    4.6% |  -0.99 |    973,140 |    152 |   0.25 |   N   |
| edge_plus_gate_eve |    0.5% |    0.7% |   0.00 |  1,002,778 |    151 |   0.25 |   N   |
| full_portfolio_cap |    0.2% |    0.6% |   0.00 |  1,001,055 |    151 |   0.25 |   N   |
------------------------------------------------------------------------------------------

[REGIME_C34_GOLD] C3/C4 gold standard 계산 완료: events=23806 (IS=12204, OOS=11602)

----------------------------------------------------------------------------------
| [REGIME_SCORECARD]                                                             |
----------------------------------------------------------------------------------
| Axis                 |  Score  | Key Metrics                                   |
----------------------------------------------------------------------------------
| C2 Persistence       | 10.0/10 | dwell=7.00(micro) macro=10.00 tr=0.075 ent=0.340 |
| C3 Distinctness      | 10.0/10 | kw_p=0.0000  flip=Y  mi=0.0373                |
| C4 OOS Stability     | 9.0/10  | rho=0.829  n_regimes=6                        |
| C5 Coverage          | 10.0/10 | min=0.106 max=0.247 n_eff=5.74                |
----------------------------------------------------------------------------------
| Weighted C2-C5       |  0.680  | C1(hard_gate=pass)  C6-C8(manual)             |
----------------------------------------------------------------------------------
| Occupancy            |   n/a   | bull_q=0.201  bull_vol=0.172  bear_q=0.111    |
|                      |         | bear_vol=0.106  trans=0.247  crash=0.163      |
| C3/C4_proxy (mkt)    |   n/a   | kw_p=0.498 flip=Y rho=0.429                   |
----------------------------------------------------------------------------------
[PHASE] phase=alo completed strategy/candidate evaluation only; optimization/training skipped
```

---