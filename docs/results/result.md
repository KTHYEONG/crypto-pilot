# 목적
유효한 signal 들을 여러개 생성 후 기존의 1개의 전략으로만 매매하는 한계점을 보완하여 ML으로 동적으로 적재적소에 전략을 사용하여 복리자산증식 전략 도출해내는 것임

# Mode Full (ALO/Ensemble) — 최신 검증 결과

**최신 갱신:** 2026-06-11 (Direction A 실제 활성화 — opt_config.py 직접 주입, Fix 2/3 적용)
**현재 상태:** `blocked` — 1/4 Fold Pass (fold_pass_ratio 25.0%). **Active Signals = 0**
**평가 기준:** `min_variant_oos_edge_bps=10.0`, `breakeven_hard_gate=enabled`, `min_fold_realized_edge_bps=8.0`, `min_oos_rank_ic=0.01`, `min_ic_tstat=0.8`, `max_variant_oos_q10_fail_rate=0.65`, `min_wf_fold_pass_ratio=0.60`

**진단 노트:**
- **Direction A 실제 활성화 결과 분석 (2026-06-11, `score_calibration: 6 regimes fitted, 3 valid`):**
  - ✅ **실제로 Direction A 가동 확인**: `[ENSEMBLE] score_calibration: 6 regimes fitted, 3 valid` 로그 확인. 이전 run은 `.env` 미반영으로 실제 미가동이었음.
  - ℹ️ **Validation Rank IC -0.046** (Fix 2 적용 후 score path IC 실측): 이전 기준선(-0.004)보다 악화. score path가 in-fold val window에서 anti-predictive.
  - ❌ **Fold 1 RlzdMean 9.7 (✅ PASS)**, Fold 2: 1.7 (❌), Fold 3: 4.4 (❌), Fold 4: 7.3 (❌). 기준선(7.7/4.3/8.5/12.8, Fold 4만 PASS)에서 변화하였으나 pass_ratio 동일(1/4=25%).
  - ❌ **Fold 2 IC +0.045로 양전환에도 RlzdMean 1.7**: score path가 val IC를 양수로 만들었으나 realized mean이 개선 안 됨 → IC-RlzdMean 해리(signal이 있어도 selection→sizing 경로에서 실현 안 됨).
  - **핵심 진단 확정**: `score_z`는 동일 variant 내 시계열 percentile (2160 bars causal window). 서로 다른 variant 간 cross-sectional ranking에 정보력 없음 → slope 기울기가 양수더라도 cross-sectional Rank IC≈0 유지. **알고리즘 교체 필요 (cross-sectional score 설계 또는 대체 ranking model).**

- **Direction A+B 구현 (q90 실산출, 2026-06-11):**
  - ✅ **Ens_Kelly CAGR +0.9%, MaxDD 0.7%** (이전: +2.8%, MaxDD 2.0%): q90 실산출로 Kelly σ 정상화 → 포지션 축소. 리스크 감소 확인.
  - ✅ **회귀 없음**: 기존 경로(`score_calibration=False`) 시 bit-identical 수치 유지.

- **Portfolio Kelly A/B (2026-06-11):**
  - ❌ **Ens_CovKelly -7.4%** — Ledoit-Wolf overlay 반증. Disabled 유지.

- **Phase 3 + Allocation Target Vol Bypass 결과 분석 (2026-06-10):**
  - ✅ **앙상블 변이 우선 매핑 정상화 (21 variants fitted)**: 추천 21종 전략의 패밀리명을 온전히 매핑하여 21개 변이가 정상 적합(fit)되었습니다.
  - ✅ **OOS Rank IC 개선**: 검증 및 예측 공식 불일치(bug) 해결 및 variant prior 적용 활성화로 전 Fold의 OOS Rank IC가 개선되었으며, 특히 Fold 3의 IC는 양수(+0.076)로 반전되었습니다.
  - ⚠️ **여전히 blocked (1/4 PASS)**: [portfolio_constructor.py](file:///home/kth/my_coin_traider/src/domain/futures/portfolio/portfolio_constructor.py) 핫픽스 및 켈리 분모 정밀도 수정에 의해 `Ens_Kelly_Caps`이 **CAGR +2.6%** (원천 켈리 CAGR **+2.8%**)로 정상 수렴함을 확인하였습니다.

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
| 3    | ensemble_b0 |    0.076 |   3,137 |      8.5 |   36.15 | ❌      |
| 4    | ensemble_b0 |   -0.010 |   3,705 |     12.8 |   33.89 | ✅      |
----------------------------------------------------------------------------------
(★ Phase 3 Offset + Filter + Alignment 적용 후: Fold OOS Rank IC 대폭 개선 및 21 variants fitted)

[ENSEMBLE DIAGNOSTICS] Phase 3 Variant Prior Offset + Family Filter (2026-06-10 ALO Run — Fold 1 기준, 21 variants fitted)
----------------------------------------------------------------------------------
| Metric                       | Value                                           |
| ---------------------------- | ----------------------------------------------- |
| N events (train)             | 5,705                                           |
| Global mu (bps)              | 22.4                                            |
| Validation Rank IC           | -0.004                                          |
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
| 진단: Variant Offset, 패밀리 필터 적용 및 검증 IC 공식 정합 완료.              |
| OOS Rank IC는 전 Fold에서 의미 있게 개선되었으며, 특히 Fold 3의 IC는 +0.076으로 |
| 최초 양전환을 확인했습니다. 다만 Fold 1, 2, 3이 realized mean 기준에 미달하여  |
| 전체 1/4 PASS 상태 유지.                                                       |
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
| Base_Rule          |  -28.2% |   19.2% |  -1.46 |    825,177 |    619 |   1.00 |   N   |
| Prior_Filter       |    1.4% |    1.9% |   0.71 |  1,007,976 |    113 |   0.24 |   N   |
| Prior_Residual     |    0.7% |    3.2% |   0.22 |  1,004,009 |    145 |   0.25 |   N   |
| Ens_Gate       |    0.7% |    3.2% |   0.22 |  1,004,009 |    145 |   0.25 |   N   |
| Ens_Kelly     |    0.9% |    0.7% |   0.00 |  1,005,171 |    142 |   0.25 |   N   |
| Ens_Kelly_Caps |    0.9% |    0.7% |   0.00 |  1,005,171 |    142 |   0.25 |   N   |
| Ens_CovKelly |   -7.4% |    7.0% |  -1.05 |    956,470 |    145 |   0.25 |   N   |
------------------------------------------------------------------------------------------
(★ Ens_CovKelly = Ledoit-Wolf 공분산 Kelly, caps bypassed. Disabled 유지)
(★ Ens_Kelly: q90 실산출(Direction B) 적용 후 +2.8%→+0.9%, MaxDD 2.0%→0.7%. score_calibration(Direction A) 미활성)


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