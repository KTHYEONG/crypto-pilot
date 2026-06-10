# 목적
유효한 signal 들을 여러개 생성 후 기존의 1개의 전략으로만 매매하는 한계점을 보완하여 ML으로 동적으로 적재적소에 전략을 사용하여 복리자산증식 전략 도출해내는 것임

# Mode Full (ALO/Ensemble) — 최신 검증 결과

**최신 갱신:** 2026-06-10 (Phase 1 EB 수축 + Phase 2 Signal Pruning profit_floor=15bps 적용)
**현재 상태:** `blocked` — 1/4 Fold Pass (fold_pass_ratio 25%). **Active Signals = 0**
**평가 기준:** `min_variant_oos_edge_bps=10.0`, `breakeven_hard_gate=enabled`, `min_fold_realized_edge_bps=8.0`, `min_oos_rank_ic=0.01`, `min_ic_tstat=0.8`, `max_variant_oos_q10_fail_rate=0.65`, `min_wf_fold_pass_ratio=0.60`

**진단 노트:**
- **RECOMMENDED 21종 (profit_floor=15bps 적용 후 동일):** dm_24_96, dm_12_48, tpc_50_200, fzs_96, mtf_tpb_20_30, mtf_bor_20, mtf_tpb_50_30, mtf_bor_40, bollinger_20, fzs_168, tim_24, tpc_20_100, rsi_14, rsi_6, vrr_20, fzs_48, tim_12, rr_48, vrr_40, rr_24, btc_pullback_50.
- **BLOCKED 11종 (profit_floor 추가 2종):** Event Overload(4), Breakeven Gate(5), profit_floor(2) 차단. fzs/rsi 등은 Bayesian admission OR-path로 여전히 통과 중 — 개별 regime cell 수준에서는 통계적으로 유효.
- **WF 지표:** Fold 4 PASS (RlzdMean=21.5, EU_p90=27.58). Fold 1~3 변화 없음.
- **Phase 2 결과:** profit_floor=15bps는 BLOCKED 수를 9→11로 늘렸으나 RECOMMENDED 21종 구성 및 Ablation CAGR 동일. **근본 원인 확정: WF IC 음수(-0.075~-0.106)는 archetype/signal 수준이 아닌 ensemble 예측 점수와 실현 수익 간 정렬 실패.** fzs/rsi 등 noise signal은 개별 regime cell에서 양수 edge 보유 → Bayesian admission 통과 → ensemble pool에 포함. 단, ensemble 출력 score가 target(net_return_bps)과 역상관.
- **다음 병목:** ensemble 예측 score ↔ target 정렬 문제 직접 해결 필요. 가능한 접근: (a) score 정규화 재검토, (b) mu_net_decision_bps 계산 경로 추적, (c) walk_forward 모듈에서 IC 음수 발원 fold 집중 분석.

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
| 1    | ensemble_b0 |   -0.075 |   2,297 |      4.9 |   34.50 | ❌      |
| 2    | ensemble_b0 |   -0.106 |   2,303 |      6.3 |   35.03 | ❌      |
| 3    | ensemble_b0 |   -0.019 |   3,137 |     -7.3 |   29.84 | ❌      |
| 4    | ensemble_b0 |   -0.067 |   3,705 |     21.5 |   27.58 | ✅      |
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