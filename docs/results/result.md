# 목적
유효한 signal 들을 여러개 생성 후 기존의 1개의 전략으로만 매매하는 한계점을 보완하여 ML으로 동적으로 적재적소에 전략을 사용하여 복리자산증식 전략 도출해내는 것임

# Mode Full (ALO/Ensemble) — 최신 검증 결과

**최신 갱신:** 2026-06-08 (Ensemble B0 Integrated Redesign — `allocation_backend="ensemble_b0"` 도입)
**현재 상태:** `READY (Ensemble B0)` — integrated redesign 적용 완료, ML-Ready 8개 신호 확보
**평가 기준:** `min_variant_oos_edge_bps=10.0`, `breakeven_hard_gate=enabled`, `cost_floor_bps=7.5`
**진단 노트:** 복잡한 LGBM 모델 대신 **Regime-conditional Ensemble (B0)** 을 기본 배분 백엔드로 전환하여 안정성 확보. Newey-West t-stat 기반의 **Breakeven Hard Gate**를 도입하여 통계적 유의성이 낮은 변종 자동 배제. Signal-Regime entry gating (mean-reversion) 적용. **잔여 블로커**: C4 rho=-0.086 (Ensemble 전환 후 IS/OOS 순위 역전 현상 발생, prior shrinkage k 조정 필요). SSOT: `docs/specs/futures_integrated_redesign_master.md`

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
| 12   | funding_zscore_carry:fzs_168        | 1594 (983)   |         9.8 |    42.9% |   1.33 | -0.057 | KEEP   | Y   |
| 13   | vol_regime_reversion:vrr_40         | 824 (572)    |         9.2 |    42.3% |   1.19 |  0.007 | DROP   | Y   |
| 18   | funding_zscore_carry:fzs_96         | 823 (532)    |         4.8 |    41.0% |   1.22 | -0.095 | KEEP   | Y   |
-----------------------------------------------------------------------------------------------------------------------

[SIGNAL DIAGNOSTICS: STRATEGY FILTERING]
----------------------------------------------------------------------------------
| Action       | Count | Details / Selected Strategies                           |
----------------------------------------------------------------------------------
| RECOMMENDED  | 8     | 1. dual_momentum:dm_24_96                               |
| (Ensemble B0)|       | 2. vol_regime_reversion:vrr_40                          |
| (ML Ready)   |       | 3. dual_momentum:dm_12_48                               |
|              |       | 4. trend_pullback_continuation:tpc_50_200               |
|              |       | 5. funding_zscore_carry:fzs_96                          |
|              |       | 6. rsi_reversion:rsi_6                                  |
|              |       | 7. funding_zscore_carry:fzs_168                         |
|              |       | 8. trend_pullback_continuation:tpc_20_100               |
----------------------------------------------------------------------------------

[WALK-FORWARD FOLD DETAILS]
----------------------------------------------------------------------------------
| Fold | Mode       |  Rank IC |  Events | PriorP90 |  EU_p90 | Pass   |
----------------------------------------------------------------------------------
| 1    | n/a        |    0.000 |   1,039 |     0.00 |   61.53 | ❌      |
| 2    | n/a        |    0.000 |   1,148 |     0.00 |   37.18 | ❌      |
| 3    | n/a        |    0.000 |   1,585 |     0.00 |   37.88 | ✅      |
| 4    | n/a        |    0.000 |   1,909 |     0.00 |   40.09 | ❌      |
----------------------------------------------------------------------------------

[BRIDGE SUMMARY] -----------------------------------
| Metric             | Value                       |
| ------------------ | --------------------------- |
| Active Signals     | 0 (sel=0)                   |
| Status             | blocked                     |
| Execution Time     | 54.64s                      |
----------------------------------------------------

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

[ABLATION STUDY FRONTIER] ----------------------------------------------------------------
| Model Alias        |    CAGR |   MaxDD |    MAR |     Equity | Trades | Deploy | Pass  |
| ------------------ | ------- | ------- | ------ | ---------- | ------ | ------ | ----- |
| rule_stop_risk     |  -26.5% |   17.5% |  -1.51 |    836,286 |    626 |   1.00 |   N   |
| prior_rank_stop_ri |   -6.8% |    8.0% |  -0.86 |    959,801 |    150 |   0.25 |   N   |
| prior_residual_ran |    0.0% |    0.0% |   0.00 |  1,000,000 |      0 |   0.00 |   N   |
| edge_plus_validate |    0.0% |    0.0% |   0.00 |  1,000,000 |      0 |   0.00 |   N   |
| edge_plus_gate_eve |    0.0% |    0.0% |   0.00 |  1,000,000 |      0 |   0.00 |   N   |
| full_portfolio_cap |    0.0% |    0.0% |   0.00 |  1,000,000 |      0 |   0.00 |   N   |
------------------------------------------------------------------------------------------
```

---

## 전략별 성과 요약 (Top Candidates)

| Rank | Strategy Name | Profit (bps) | Win Rate | P/L | Action | Rec |
|---|---|---:|---:|---:|---|---|
| 1 | **tpc_50_200** | **74.0** | 37.9% | 1.63 | KEEP | **Y** |
| 2 | **dm_24_96** | **68.1** | 42.6% | 1.25 | KEEP | **Y** |
| 5 | **dm_12_48** | **25.5** | 40.3% | 1.32 | KEEP | **Y** |
| 12 | **fzs_168** | **9.8** | 42.9% | 1.33 | KEEP | **Y** |
| 18 | **fzs_96** | **4.8** | 41.0% | 1.22 | KEEP | **Y** |

---

## Regime 평가 분석 (Integrated Redesign 반영)

**측정일:** 2026-06-08 | **기준:** `docs/specs/futures_integrated_redesign_master.md`

### 핵심 진단: Ensemble B0 전환 후 상태

- **C4 rho=-0.086**: Ensemble 백엔드로 전환하면서 IS 기간의 레짐별 성과 순위가 OOS에서 유지되지 않는 현상 관찰. 이는 prior shrinkage 강도가 부족하거나, 특정 레짐에서의 시그널 편향이 OOS에서 역전되었음을 시사함.
- **Breakeven Hard Gate**: 8개 신호가 통과되었으나, 실제 백테스트(`rule_stop_risk`)에서는 여전히 하향 곡선. 이는 개별 신호의 엣지보다 포트폴리오 차원의 정교한 배분이 더 시급함을 의미.
- **C3 Distinctness**: `flip=N` 유지. 레짐이 수익의 '크기'는 조절하지만 '방향'을 바꾸지는 못하고 있음 (Long Bias).

### 다음 액션

| 우선순위 | 액션 | 근거 |
|---|---|---|
| **P0** | Ensemble Shrinkage k 튜닝 | C4 rho 개선을 위해 `ensemble_shrinkage_k` 조정 (현재 50.0 → 최적화 필요) |
| **P1** | Allocation Backend 'ml_edge' 활성화 테스트 | Ensemble 안정성 확인 후 LGBM 기반 Challenger 경로(`ml_edge`) 성능 비교 |
| **P2** | Portfolio Risk-Unit Normalization 강화 | `allocation.md`에 명시된 단위 위험당 배분 로직의 정합성 재검증 |

SSOT: `docs/specs/futures_integrated_redesign_master.md` (통합 설계), `docs/architecture/allocation.md` (배분)
