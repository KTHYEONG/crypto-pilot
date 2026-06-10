# 목적
유효한 signal 들을 여러개 생성 후 기존의 1개의 전략으로만 매매하는 한계점을 보완하여 ML으로 동적으로 적재적소에 전략을 사용하여 복리자산증식 전략 도출해내는 것임

# Mode Full (ALO/Ensemble) — 최신 검증 결과

**최신 갱신:** 2026-06-10 (Conditioning Bug Fix 적용 — Regime-Cell Admission 미적용)
**현재 상태:** `blocked` — 0/4 Fold Pass (fold_pass_ratio 0%). **Active Signals = 0**
**평가 기준:** `min_variant_oos_edge_bps=10.0`, `breakeven_hard_gate=enabled`, `min_fold_realized_edge_bps=8.0`, `min_oos_rank_ic=0.01`, `min_ic_tstat=0.8`, `max_variant_oos_q10_fail_rate=0.65`, `min_wf_fold_pass_ratio=0.60`

**진단 노트:**
- **RECOMMENDED 7종:** dm_24_96, dm_12_48, tpc_50_200, fzs_96, fzs_168, tpc_20_100, fzs_48. Momentum + Carry 계열 위주.
- **BLOCKED 17종:** Breakeven Gate(10), Event Overload(8), Low IC t-stat(10) 주요 원인. Regime-cell admission 미적용으로 carry/reversion 계열 다수 탈락.
- **WF 전 fold 실패:** IC(diag) 전 구간 음수(-0.016~-0.120), RlzdMean Fold3만 +43.1bps 기록했으나 EU_p90 gate(44.12) 미달. Ensemble 선택이 baseline 대비 우위 없음.
- **[구조 수정 완료] Conditioning Bug Fix:** `ensemble_conditioning` 기본값 `archetype_regime`→`auto`, lift_proof fail-OPEN→fail-SAFE 수정. 이번 실행에서 `conditioning_path=no_oos_evidence_failsafe`(archetype_only 강등) 확인. **WF 수치 불변이 정상**: 풀이 동질적(추세/carry 7종)이라 archetype_only도 저IC. 핵심 블로커 = admission OFF로 인한 풀 다양성 부재.
- **Ablation Study 전 Variant Fail:** CAGR 최대 −35.0%(rule_stop_risk). 모든 variant가 compound gate + deployment gate 통과 실패.
- **Regime Scorecard:** C2=10.0/10, C3=10.0/10, C4=4.0/10, C5=10.0/10 → Weighted **0.580**.
- **다음 블로커:** (1) admission ON + t_g NW 보정 → 풀 다양성 복원. (2) regime 6→4 state 병합 → C4 rho 상향.<!-- truncate -->

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
| 1    | trend_pullback_continuation:tpc_... | 760 (309)    |        74.0 |    37.9% |   1.63 |  0.064 | KEEP   | Y   |
| 2    | btc_corr_regime:bcr_96              | 51 (39)      |        70.6 |    53.8% |   2.42 | -0.026 | DROP   | N   |
| 3    | dual_momentum:dm_24_96              | 1373 (556)   |        68.1 |    42.6% |   1.25 |  0.044 | KEEP   | Y   |
| 4    | btc_corr_regime:bcr_48              | 108 (73)     |        59.2 |    52.1% |   1.86 |  0.079 | DROP   | N   |
| 5    | dual_momentum:dm_12_48              | 1953 (764)   |        25.5 |    40.3% |   1.32 | -0.004 | KEEP   | Y   |
| 6    | btc_residual_momentum:brm_24        | 1876 (969)   |        17.2 |    44.0% |   1.17 | -0.108 | DROP   | N   |
| 7    | trend_donchian:donchian_36          | 916 (564)    |        16.0 |    44.0% |   1.50 |  0.007 | KEEP   | N   |
| 8    | residual_reversion:rr_24            | 1212 (494)   |        14.6 |    43.4% |   1.01 | -0.032 | DROP   | N   |
| 9    | rsi_reversion:rsi_14                | 2508 (1489)  |        12.0 |    42.3% |   1.09 | -0.096 | DROP   | N   |
| 10   | vol_regime_reversion:vrr_40         | 989 (626)    |        11.4 |    43.5% |   1.24 |  0.001 | KEEP   | N   |
| 11   | residual_reversion:rr_48            | 1250 (511)   |         9.9 |    40.5% |   0.88 | -0.055 | DROP   | N   |
| 12   | trend_donchian:donchian_18          | 1489 (894)   |         9.0 |    42.6% |   1.41 |  0.008 | KEEP   | N   |
| 13   | cross_sectional_momentum:cs_mom_10  | 6118 (3659)  |         8.8 |    41.6% |   1.21 | -0.091 | KEEP   | N   |
| 14   | funding_acceleration_carry:fac_168  | 8679 (5533)  |         8.8 |    42.6% |   1.21 | -0.044 | KEEP   | N   |
| 15   | funding_zscore_carry:fzs_168        | 1783 (1032)  |         6.9 |    42.8% |   1.25 | -0.079 | KEEP   | Y   |
| 16   | cross_sectional_momentum:cs_mom_20  | 8982 (5283)  |         6.6 |    42.1% |   1.18 | -0.081 | DROP   | N   |
| 17   | funding_acceleration_carry:fac_48   | 8716 (5435)  |         6.1 |    42.2% |   1.11 | -0.065 | DROP   | N   |
| 18   | cross_sectional_momentum:cs_mom_5   | 6092 (3604)  |         5.3 |    40.9% |   1.09 | -0.075 | DROP   | N   |
| 19   | trend_ma:ema_6_36                   | 11410 (6581) |         5.2 |    40.6% |   1.17 | -0.055 | DROP   | N   |
| 20   | funding_carry:funding_24            | 2042 (1254)  |         4.7 |    41.8% |   1.38 | -0.058 | KEEP   | N   |
-----------------------------------------------------------------------------------------------------------------------

[SIGNAL DIAGNOSTICS: STRATEGY FILTERING]
----------------------------------------------------------------------------------
| Action       | Count | Details / Selected Strategies                           |
----------------------------------------------------------------------------------
| BLOCKED      | 17    | Fail Reasons: Breakeven Gate (10) | Event Overload (8)  |
|              |       |               Poor Hit/Payoff (4) | Low IC t-stat (10)  |
|              |       |               Low Mean Edge (4) | Low Median (5)        |
|              |       |               Low Obs (2) | Low OOS IC (8)              |
|              |       | Top Blocked: bcr_48, bb_compress_20, bcr_24, vrr_40, fa |
| RECOMMENDED  | 7     | 1. dual_momentum:dm_24_96                               |
| (Eligible)   |       | 2. dual_momentum:dm_12_48                               |
| (Eligible)   |       | 3. trend_pullback_continuation:tpc_50_200               |
| (Eligible)   |       | 4. funding_zscore_carry:fzs_96                          |
| (Eligible)   |       | 5. funding_zscore_carry:fzs_168                         |
| (Eligible)   |       | 6. trend_pullback_continuation:tpc_20_100               |
| (Eligible)   |       | 7. funding_zscore_carry:fzs_48                          |
----------------------------------------------------------------------------------

[GATE FAILURES: PER-VARIANT]
----------------------------------------------------------------------------------
| Variant                        | Action           | Failed Gates / Cells       |
----------------------------------------------------------------------------------
| btc_corr_regime:bcr_48         | INSUFFICIENT_OBS | Low Obs, Breakeven Gate, … |
| vol_breakout:bb_compress_20    | INSUFFICIENT_OBS | Breakeven Gate, Low OOS I… |
| btc_corr_regime:bcr_24         | INSUFFICIENT_OBS | Breakeven Gate, Low Media… |
| vol_regime_reversion:vrr_40    | DROP_OR_REWORK   | Breakeven Gate, Low Media… |
| funding_acceleration_carry:fac | DROP_OR_REWORK   | Event Overload             |
| funding_acceleration_carry:fac | DROP_OR_REWORK   | Event Overload             |
| trend_ma:ema_12_72             | KEEP_CANDIDATE   | Event Overload             |
| cross_sectional_momentum:cs_mo | DROP_OR_REWORK   | Breakeven Gate, Event Ove… |
| trend_donchian:donchian_18     | KEEP_CANDIDATE   | Breakeven Gate, Low OOS I… |
| trend_ma:ema_18_108            | KEEP_CANDIDATE   | Event Overload             |
| cross_sectional_momentum:cs_mo | DROP_OR_REWORK   | Event Overload             |
| trend_ma:ema_6_36              | DROP_OR_REWORK   | Event Overload             |
| cross_sectional_momentum:cs_mo | DROP_OR_REWORK   | Event Overload             |
| trend_donchian:donchian_36     | DROP_OR_REWORK   | Breakeven Gate, Low Mean … |
| trend_donchian:donchian_72     | INSUFFICIENT_OBS | Breakeven Gate, Low Mean … |
| btc_residual_momentum:brm_24   | DROP_OR_REWORK   | Breakeven Gate, Low Mean … |
| btc_corr_regime:bcr_96         | INSUFFICIENT_OBS | Low Obs, Breakeven Gate, … |
----------------------------------------------------------------------------------

[WALK-FORWARD FOLD DETAILS]
----------------------------------------------------------------------------------
| Fold | Mode       | IC(diag) |  Events | RlzdMean |  EU_p90 | Pass   |
|      |            |    (ref) |         |  (★gate) | (★gate) |        |
----------------------------------------------------------------------------------
| 1    | ensemble_b0 |   -0.016 |     908 |    -26.3 |   39.14 | ❌      |
| 2    | ensemble_b0 |   -0.104 |     871 |    -17.5 |   48.61 | ❌      |
| 3    | ensemble_b0 |   -0.059 |   1,140 |     43.1 |   44.12 | ❌      |
| 4    | ensemble_b0 |   -0.120 |   1,360 |      7.2 |   44.59 | ❌      |
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
| rule_stop_risk     |  -35.0% |   23.0% |  -1.52 |    778,883 |    624 |   1.00 |   N   |
| prior_rank_stop_ri |   -3.2% |    3.2% |  -0.98 |    981,569 |    106 |   0.22 |   N   |
| prior_residual_ran |   -2.9% |    3.5% |  -0.84 |    982,797 |    109 |   0.22 |   N   |
| edge_plus_validate |   -2.9% |    3.5% |  -0.84 |    982,797 |    109 |   0.22 |   N   |
| edge_plus_gate_eve |   -1.6% |    1.5% |  -1.07 |    990,717 |    108 |   0.22 |   N   |
| full_portfolio_cap |   -0.6% |    0.8% |   0.00 |    996,479 |    108 |   0.22 |   N   |
------------------------------------------------------------------------------------------

[REGIME_C34_GOLD] C3/C4 gold standard 계산 완료: events=9112 (IS=4785, OOS=4327)

----------------------------------------------------------------------------------
| [REGIME_SCORECARD]                                                             |
----------------------------------------------------------------------------------
| Axis                 |  Score  | Key Metrics                                   |
----------------------------------------------------------------------------------
| C2 Persistence       | 10.0/10 | dwell=7.00(micro) macro=10.00 tr=0.075 ent=0.340 |
| C3 Distinctness      | 10.0/10 | kw_p=0.0000  flip=Y  mi=0.0609                |
| C4 OOS Stability     | 4.0/10  | rho=0.257  n_regimes=6                        |
| C5 Coverage          | 10.0/10 | min=0.106 max=0.247 n_eff=5.74                |
----------------------------------------------------------------------------------
| Weighted C2-C5       |  0.580  | C1(hard_gate=pass)  C6-C8(manual)             |
----------------------------------------------------------------------------------
| Occupancy            |   n/a   | bull_q=0.201  bull_vol=0.172  bear_q=0.111    |
|                      |         | bear_vol=0.106  trans=0.247  crash=0.163      |
| C3/C4_proxy (mkt)    |   n/a   | kw_p=0.498 flip=Y rho=0.429                   |
----------------------------------------------------------------------------------
[PHASE] phase=alo completed strategy/candidate evaluation only; optimization/training skipped
```

---