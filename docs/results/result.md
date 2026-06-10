# 목적
유효한 signal 들을 여러개 생성 후 기존의 1개의 전략으로만 매매하는 한계점을 보완하여 ML으로 동적으로 적재적소에 전략을 사용하여 복리자산증식 전략 도출해내는 것임

# Mode Full (ALO/Ensemble) — 최신 검증 결과

**최신 갱신:** 2026-06-10 (Phase 3 Variant Prior Offset + Family Filter 적용)
**현재 상태:** `blocked` — 1/4 Fold Pass (fold_pass_ratio 25.0%). **Active Signals = 0**
**평가 기준:** `min_variant_oos_edge_bps=10.0`, `breakeven_hard_gate=enabled`, `min_fold_realized_edge_bps=8.0`, `min_oos_rank_ic=0.01`, `min_ic_tstat=0.8`, `max_variant_oos_q10_fail_rate=0.65`, `min_wf_fold_pass_ratio=0.60`

**진단 노트:**
- **Phase 3 Variant Prior Offset + Family Filter 결과 분석 (신규):**
  - ✅ **Fold 4 회복 및 통과**: RlzdMean이 10.2에서 **21.5 bps로 급반등하며 ✅ PASS 성공** (자산 배분 분산 문제 해결).
  - ⚠️ **전체 1/4 PASS로 여전히 blocked**: Fold 1, 2, 3이 통과 기준(8.0 bps)에 미달함 (특히 Fold 3 RlzdMean -7.3 bps로 악화).
  - **진단**: Variant Offset 도입으로 고이벤트 Fold 4에서 최적 포지션 비중이 보호되었으며, 필터링을 통해 Reversion 계열 노이즈가 제거되어 안정성이 확보되었습니다. 다만, 특정 Fold(Fold 3 등)의 국면 변화에는 추가적인 대책이 필요합니다.

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

## 최신 실행 요약 (4h Timeframe - ALO Phase, Phase 3 Variant Prior)

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
| 1    | trend_pullback_continuation:tpc_... | 760 (309)    |        74.0 |    37.9% |   1.63 |  0.064 | KEEP   | Y   |
| 2    | dual_momentum:dm_24_96              | 1373 (556)   |        68.1 |    42.6% |   1.25 |  0.044 | KEEP   | Y   |
| 3    | mtf_trend_pullback:mtf_tpb_50_30    | 598 (249)    |        63.9 |    37.2% |   1.57 |  0.218 | KEEP   | Y   |
| 4    | vol_term_structure_gate:vts_gate_20 | 2465 (1027)  |        36.5 |    35.4% |   1.25 |  0.012 | KEEP   | N   |
| 5    | mtf_trend_pullback:mtf_tpb_20_30    | 439 (172)    |        32.8 |    33.9% |   1.42 |  0.257 | KEEP   | Y   |
| 6    | dual_momentum:dm_12_48              | 1953 (764)   |        25.5 |    40.3% |   1.32 | -0.004 | KEEP   | Y   |
| 7    | trend_donchian:donchian_36          | 916 (564)    |        16.0 |    44.0% |   1.50 |  0.007 | KEEP   | N   |
| 8    | residual_reversion:rr_24            | 1212 (494)   |        14.6 |    43.4% |   1.01 | -0.032 | DROP   | Y   |
| 9    | rsi_reversion:rsi_14                | 2508 (1489)  |        12.0 |    42.3% |   1.09 | -0.096 | DROP   | Y   |
| 10   | vol_regime_reversion:vrr_40         | 989 (626)    |        11.4 |    43.5% |   1.24 |  0.001 | KEEP   | Y   |
| 11   | residual_reversion:rr_48            | 1250 (511)   |         9.9 |    40.5% |   0.88 | -0.055 | DROP   | Y   |
| 12   | funding_extreme_reversal:fer_168    | 3045 (1054)  |         9.0 |    42.4% |   1.75 | -0.060 | KEEP   | N   |
| 13   | trend_donchian:donchian_18          | 1489 (894)   |         9.0 |    42.6% |   1.41 |  0.008 | KEEP   | N   |
| 14   | funding_zscore_carry:fzs_168        | 1783 (1032)  |         6.9 |    42.8% |   1.25 | -0.079 | KEEP   | Y   |
| 15   | taker_imbalance_momentum:tim_12     | 621 (170)    |         5.6 |    38.8% |   1.12 | -0.071 | DROP   | Y   |
| 16   | funding_carry:funding_24            | 2042 (1254)  |         4.7 |    41.8% |   1.38 | -0.058 | KEEP   | N   |
| 17   | trend_ma:ema_18_108                 | 13664 (7739) |         3.6 |    40.3% |   1.13 | -0.077 | DROP   | N   |
| 18   | trend_ma:ema_12_72                  | 12732 (7412) |         3.2 |    40.4% |   1.13 | -0.075 | DROP   | N   |
| 19   | trend_donchian:donchian_72          | 539 (358)    |         1.6 |    40.8% |   1.33 | -0.004 | KEEP   | N   |
| 20   | funding_zscore_carry:fzs_48         | 937 (574)    |         0.3 |    40.0% |   1.35 | -0.063 | KEEP   | Y   |
-----------------------------------------------------------------------------------------------------------------------

[SIGNAL DIAGNOSTICS: STRATEGY FILTERING]
----------------------------------------------------------------------------------
| Action       | Count | Details / Selected Strategies                           |
----------------------------------------------------------------------------------
| BLOCKED      | 9     | Fail Reasons: Breakeven Gate (5) | Event Overload (4)   |
| RECOMMENDED  | 21    | 1. trend_pullback_continuation:tpc_50_200               |
| (Eligible)   |       | 2. dual_momentum:dm_24_96                               |
|              |       | 3. mtf_trend_pullback:mtf_tpb_50_30                     |
|              |       | 4. mtf_trend_pullback:mtf_tpb_20_30                     |
|              |       | 5. dual_momentum:dm_12_48                               |
|              |       | 6. residual_reversion:rr_24                             |
|              |       | 7. rsi_reversion:rsi_14                                 |
|              |       | 8. vol_regime_reversion:vrr_40                          |
|              |       | 9. residual_reversion:rr_48                             |
|              |       | 10. funding_zscore_carry:fzs_168                         |
|              |       | 11. taker_imbalance_momentum:tim_12                      |
|              |       | 12. funding_zscore_carry:fzs_48                          |
|              |       | 13. btc_regime_pullback:btc_pullback_50                  |
|              |       | 14. mtf_breakout_retest:mtf_bor_20                       |
|              |       | 15. mtf_breakout_retest:mtf_bor_40                       |
|              |       | 16. bollinger_reversion:bollinger_20                     |
|              |       | 17. rsi_reversion:rsi_6                                  |
|              |       | 18. vol_regime_reversion:vrr_20                          |
|              |       | 19. taker_imbalance_momentum:tim_24                      |
|              |       | 20. trend_pullback_continuation:tpc_20_100               |
|              |       | 21. funding_zscore_carry:fzs_96                          |
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
| 1    | ensemble_b0 |   -0.075 |   2,297 |      4.9 |   34.50 | ❌      |
| 2    | ensemble_b0 |   -0.106 |   2,303 |      6.3 |   35.03 | ❌      |
| 3    | ensemble_b0 |   -0.019 |   3,137 |     -7.3 |   29.84 | ❌      |
| 4    | ensemble_b0 |   -0.067 |   3,705 |     21.5 |   27.58 | ✅      |
----------------------------------------------------------------------------------
(★ Phase 3 Offset + Filter 적용 후: Fold4 RlzdMean 10.2→21.5 복원 및 ✅ PASS 성공)

[ENSEMBLE DIAGNOSTICS] Phase 3 Variant Prior Offset + Family Filter (2026-06-10 ALO Run — Fold 1 기준, 0 variants fitted)
----------------------------------------------------------------------------------
| Metric                       | Value                                           |
| ---------------------------- | ----------------------------------------------- |
| N events (train)             | 5,705                                           |
| Global mu (bps)              | 22.4                                            |
| Validation Rank IC           | -0.044                                          |
| IC sign                      | ❌ NEGATIVE                                     |
| Conditioning chosen          | archetype_regime                                |
| Adaptive shrinkage           | True                                            |
| k_used (EB k_eff)            | 50.0                                            |
----------------------------------------------------------------------------------
| Archetype                    | Shrunk mu (bps) | Sign | N                      |
| ---------------------------- | --------------- | ---- | ---------------------- |
| beta_neutral_reversion       | 20.8            | POS  | 715                    |
| mean_reversion               | 16.8            | POS  | 2,255                  |
| time_series_momentum         | 27.6            | POS  | 1,512                  |
| trend_continuation           | 27.1            | POS  | 1,223                  |
----------------------------------------------------------------------------------
| 진단: Variant Offset 및 패밀리 필터 적용. Fold 4 realized mean 21.5bps로 복원. |
| Fold 1, 2, 3이 여전히 통과선에 도달하지 못해 전체 1/4 PASS 상태.               |
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
| edge_plus_gate_eve |    0.3% |    0.6% |   0.00 |  1,001,718 |    150 |   0.25 |   N   |
| full_portfolio_cap |    0.1% |    0.6% |   0.00 |  1,000,715 |    150 |   0.25 |   N   |
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