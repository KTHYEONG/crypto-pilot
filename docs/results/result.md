# 목적
유효한 signal 들을 여러개 생성 후 기존의 1개의 전략으로만 매매하는 한계점을 보완하여 ML으로 동적으로 적재적소에 전략을 사용하여 복리자산증식 전략 도출해내는 것임

# Mode Full (ALO/Ensemble) — 최신 검증 결과

**최신 갱신:** 2026-06-10 (Phase 3 Variant-Edge Hierarchical Prior 적용)
**현재 상태:** `blocked` — 0/4 Fold Pass (fold_pass_ratio 0%). **Active Signals = 0**
**평가 기준:** `min_variant_oos_edge_bps=10.0`, `breakeven_hard_gate=enabled`, `min_fold_realized_edge_bps=8.0`, `min_oos_rank_ic=0.01`, `min_ic_tstat=0.8`, `max_variant_oos_q10_fail_rate=0.65`, `min_wf_fold_pass_ratio=0.60`

**진단 노트:**
- **RECOMMENDED 21종 (Phase 3 적용 후):** tpc_50_200, dm_24_96, mtf_tpb_50_30, mtf_tpb_20_30, dm_12_48, rr_24, rsi_14, vrr_40, rr_48, fzs_168, tim_12, fzs_48, btc_pullback_50, mtf_bor_20, mtf_bor_40, bollinger_20, rsi_6, vrr_20, tim_24, tpc_20_100, fzs_96. (구성 불변, 순위 재정렬: tpc 1위·mtf_tpb 부상)
- **BLOCKED 9종:** Event Overload(4), Breakeven Gate(5). (profit_floor 적용 변이는 구성 동일)
- **WF 지표:** Fold 3 IC(diag) = **+0.076 (최초 양전환)** — variant prior 효과 확인. 단, Fold 4 RlzdMean 21.5→10.2로 하락해 전체 0/4 PASS (이전 1/4 퇴행).
- **Phase 3 결과 분석:**
  - ✅ variant prior 21종 fit 완료. Fold 3 IC 양전환(+0.076)으로 변이 수준 예측자 복원 부분 성공.
  - ⚠️ **Fold 4 퇴행**: RlzdMean 21.5→10.2. ensemble 재배열로 고이벤트 fold(3705 events)에서 signal allocation이 변경됨 → 이전에 고수익 포지션에 몰렸던 배분이 분산됨.
  - ⚠️ **IC 일관성 미확보**: Fold 1(-0.072), Fold 2(-0.031), Fold 4(-0.005) 여전히 음수/근사0. variant prior가 IS-window에서 fitting되므로 OOS에서는 변이 엣지 OOS 지속성이 fold마다 불균등함.
  - **근본 확인**: 변이 수준 엣지 자체는 일부 존재(Fold 3 성공). 문제는 OOS fold 간 **변이 엣지 지속성(stability)** 불균등 — 진짜 알파는 특정 시장국면에만 집중.
- **다음 병목:** variant 엣지 OOS 지속성 강화. 옵션: (a) variant_min_obs 상향(소표본 노이즈 차단), (b) WF fold별 variant prior 재fit(현재 IS-only 단일 fit), (c) 높은 OOS IC 변이(tpc/dm/mtf)만 variant prior 적용하고 noise 변이는 셀 평균으로 회귀.

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
| 1    | ensemble_b0 |   -0.072 |   2,297 |      7.7 |   43.30 | ❌      |
| 2    | ensemble_b0 |   -0.031 |   2,303 |      4.3 |   37.82 | ❌      |
| 3    | ensemble_b0 |   +0.076 |   3,137 |      8.5 |   36.15 | ❌      |
| 4    | ensemble_b0 |   -0.005 |   3,705 |     10.2 |   30.52 | ❌      |
----------------------------------------------------------------------------------
(★ Phase 3 이전 대비: Fold3 IC -0.019→+0.076 양전환. Fold4 RlzdMean 21.5→10.2 퇴행)

[ENSEMBLE DIAGNOSTICS] Phase 3 Variant Prior (2026-06-10 ALO Run — Fold 1 기준, 21 variants fitted)
----------------------------------------------------------------------------------
| Metric                       | Value                                           |
| ---------------------------- | ----------------------------------------------- |
| N events (train)             | 5,705                                           |
| Global mu (bps)              | 22.4                                            |
| Validation Rank IC           | -0.045 (Fold 1; Fold 3 = +0.074 양전환)         |
| IC sign                      | Fold별 혼재 (Fold 3 양전환 확인)                |
| Conditioning chosen          | archetype_only                                  |
| Adaptive shrinkage           | True                                            |
| k_used (EB k_eff)            | ~0.0 (between_var 압도적)                       |
----------------------------------------------------------------------------------
| Archetype                    | Shrunk mu (bps) | Sign | N                      |
| ---------------------------- | --------------- | ---- | ---------------------- |
| beta_neutral_reversion       | 20.8            | POS  | 715                    |
| mean_reversion               | 16.8            | POS  | 2,255                  |
| time_series_momentum         | 27.6            | POS  | 1,512                  |
| trend_continuation           | 27.1            | POS  | 1,223                  |
----------------------------------------------------------------------------------
| 진단: variant prior 21종 fit. Fold 3 IC 양전환(+0.076) = 변이 예측자 복원 부분 성공. |
| Fold 4 RlzdMean 퇴행(21.5→10.2) = WF fold pass 1/4→0/4.                        |
| 변이 엣지 OOS fold 간 지속성 불균등이 핵심 미해결 과제.                          |
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
| prior_residual_ran |   -1.8% |    3.1% |  -0.57 |    989,588 |    146 |   0.25 |   N   |
| edge_plus_validate |   -1.8% |    3.1% |  -0.57 |    989,588 |    146 |   0.25 |   N   |
| edge_plus_gate_eve |   -0.2% |    0.9% |   0.00 |    998,627 |    144 |   0.25 |   N   |
| full_portfolio_cap |    0.1% |    0.6% |   0.00 |  1,000,805 |    144 |   0.25 |   N   |
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