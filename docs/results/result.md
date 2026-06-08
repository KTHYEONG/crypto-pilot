# 목적
유효한 signal 들을 여러개 생성 후 기존의 1개의 전략으로만 매매하는 한계점을 보완하여 ML으로 동적으로 적재적소에 전략을 사용하여 복리자산증식 전략 도출해내는 것임

# Mode Full (ML) — 최신 검증 결과

**최신 갱신:** 2026-06-07 (Regime 평가 프레임워크 R1 구현 완료 + Phase R2 결정)
**현재 상태:** `WF_ELIGIBLE (ML Phase)` — 3/4 fold PASS, Active Signals 614개 선정
**평가 기준:** `min_variant_oos_edge_bps=10.0`, `min_deployment_trade_count=20`, `cost_floor_bps=7.5`
**진단 노트:** Fix-1/Fix-2로 복원한 베이스라인. Phase 1(cal prior update)·Phase 2(fit_frac=0.40) 실험 모두 악화 → 롤백. **근본 블로커**: center ML rank-IC=-0.10~+0.04(전 fold, 무예측력). 원인은 ML-Ready 신호 3개×~1500 훈련이벤트로 변별 불가. 해결책: 신호 풀을 6+개로 확장(Priority 1), 이진 분류 타겟(Priority 2), IC 게이트 재보정(Priority 3). SSOT: `docs/specs/ml_dynamic_allocation_roadmap.md`

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

## 최신 실행 요약 (4h Timeframe)

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
[UNIVERSE] Discovery complete: 94 symbols (2.45s)

[DATA QUALITY] -------------------------------------
| Metric             | Value                       |
| ------------------ | --------------------------- |
| Symbols (Req/Load) | 94 / 91 (96.8%)             |
| Kept (Ready)       | 63                          |
| Fail Reasons       | fetch_window_short:28       |
----------------------------------------------------

[REGIME_SCORECARD] (4h, 2026-06-07) ← 유니버스 이후 / 시그널 이전
------------------------------------------------------------
| Axis                 |  Score  | Key Metrics                     |
------------------------------------------------------------
| C2 Persistence       | 8.0/10  | dwell=6.00  tr=0.116  ent=0.477 |
| C3 Distinctness      | 6.0/10* | kw_p=0.000  flip=N  mi=0.039    |
| C4 OOS Stability     | 4.0/10* | rho=0.100  n_regimes=5          |
| C5 Coverage          |10.0/10  | min=0.111  max=0.325  n_eff=4.65|
------------------------------------------------------------
| Weighted C2-C5       |  0.450  | C1(hard_gate=pass)  C6-C8(manual) |
------------------------------------------------------------
| Occupancy            | bull_quiet=0.325  bull_volatile=0.153  bear_quiet=0.248 |
|                      | bear_volatile=0.111  transition=0.000  crash=0.163      |
------------------------------------------------------------
* C3/C4: 레이블 이벤트 기반 사후 계산값 (pre-signal 단계에서는 n/a)

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
| 2    | dual_momentum:dm_24_96              | 9815 (3857)  |        31.3 |    37.4% |   1.11 |  0.025 | DROP   | N   |
| 3    | rsi_reversion:rsi_14                | 4384 (1811)  |        22.6 |    43.8% |   1.30 | -0.082 | KEEP   | N   |
| 4    | dual_momentum:dm_12_48              | 9873 (3937)  |        20.1 |    39.3% |   1.15 |  0.026 | DROP   | N   |
| 5    | funding_zscore_carry:fzs_48         | 1927 (680)   |        17.6 |    42.9% |   1.62 | -0.056 | KEEP   | N   |
| 6    | funding_zscore_carry:fzs_168        | 3678 (1286)  |        17.5 |    44.4% |   1.41 | -0.025 | KEEP   | N   |
| 7    | bollinger_reversion:bollinger_20    | 3207 (1218)  |        16.5 |    41.7% |   0.95 | -0.052 | DROP   | N   |
| 8    | residual_reversion:rr_24            | 1212 (494)   |        14.6 |    43.4% |   1.01 | -0.032 | DROP   | N   |
| 9    | funding_carry:funding_24            | 3743 (1515)  |        13.3 |    43.2% |   1.50 | -0.038 | KEEP   | N   |
| 10   | vol_regime_reversion:vrr_40         | 4254 (1513)  |        12.4 |    42.4% |   1.10 | -0.020 | DROP   | N   |
| 11   | funding_zscore_carry:fzs_96         | 1908 (652)   |        12.1 |    41.9% |   1.37 | -0.096 | KEEP   | Y   |
| 12   | rsi_reversion:rsi_6                 | 4908 (2000)  |        12.1 |    42.3% |   1.08 | -0.078 | DROP   | N   |
| 13   | cross_sectional_momentum:cs_mom_10  | 10547 (4389) |        10.8 |    41.9% |   1.17 | -0.052 | DROP   | N   |
| 14   | trend_donchian:donchian_36          | 1788 (728)   |        10.2 |    43.3% |   1.39 |  0.059 | KEEP   | N   |
| 15   | residual_reversion:rr_48            | 1250 (511)   |         9.9 |    40.5% |   0.88 | -0.055 | DROP   | N   |
| 16   | btc_corr_regime:bcr_48              | 23884 (9479) |         9.8 |    42.8% |   1.23 | -0.002 | KEEP   | N   |
| 17   | funding_acceleration_carry:fac_48   | 15017 (6579) |         9.6 |    42.6% |   1.15 | -0.027 | DROP   | N   |
| 18   | btc_regime_pullback:btc_pullback_50 | 1953 (660)   |         8.9 |    41.1% |   0.99 | -0.141 | DROP   | N   |
| 19   | funding_acceleration_carry:fac_168  | 14978 (6776) |         8.9 |    42.2% |   1.17 |  0.001 | DROP   | N   |
| 20   | cross_sectional_momentum:cs_mom_20  | 15403 (6337) |         8.8 |    42.3% |   1.15 | -0.048 | DROP   | N   |
-----------------------------------------------------------------------------------------------------------------------

[SIGNAL DIAGNOSTICS: STRATEGY FILTERING]
----------------------------------------------------------------------------------
| Action       | Count | Details / Selected Strategies                           |
----------------------------------------------------------------------------------
| RECOMMENDED  | 3     | 1. trend_pullback_continuation:tpc_50_200               |
| (ML Ready)   |       | 2. funding_zscore_carry:fzs_96                          |
| (ML Ready)   |       | 3. trend_pullback_continuation:tpc_20_100               |
----------------------------------------------------------------------------------

[WALK-FORWARD FOLD DETAILS]
----------------------------------------------------------------------------------
| Fold | Mode       |  Rank IC |  Events | PriorP90 |  EU_p90 | Pass   |
----------------------------------------------------------------------------------
| 1    | prior_only |    0.000 |     377 |   103.86 |  103.86 | ❌      |
| 2    | prior_only |    0.000 |     303 |    80.97 |   80.97 | ✅      |
| 3    | prior_only |    0.000 |     383 |    69.31 |   69.31 | ✅      |
| 4    | prior_only |    0.000 |     420 |    61.76 |   61.76 | ✅      |
----------------------------------------------------------------------------------

[BRIDGE SUMMARY] -----------------------------------
| Metric             | Value                       |
| ------------------ | --------------------------- |
| Active Signals     | 4924 (sel=614)              |
| Status             | wf_eligible                 |
| Execution Time     | 73.52s                      |
----------------------------------------------------

[ABLATION STUDY FRONTIER] ----------------------------------------------------------------
| Model Alias        |    CAGR |   MaxDD |    MAR |     Equity | Trades | Deploy | Pass  |
| ------------------ | ------- | ------- | ------ | ---------- | ------ | ------ | ----- |
| rule_stop_risk     |  -20.4% |   16.6% |  -1.23 |    875,924 |    620 |   1.00 |   N   |
| prior_rank_stop_ri |    1.4% |    3.1% |   0.45 |  1,008,093 |    128 |   0.24 |   N   |
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
| 11 | **fzs_96** | **12.1** | 41.9% | 1.37 | KEEP | **Y** |
| n/a | **tpc_20_100** | **~10.0** | n/a | n/a | KEEP | **Y** |

---

## Regime 평가 분석 (Phase R2 결정)

**측정일:** 2026-06-07 | **기준:** `docs/specs/regime_evaluation_and_hardening.md` 8축 루브릭

### 축별 판정

| 축 | 점수 | 임계 | 판정 | 근거 |
|---|---:|---|---|---|
| C1 Look-ahead | 9/10 | hard gate | ✅ PASS | CUSUM/EMA 전부 causal, entry-1 소비 |
| C2 Persistence | 8/10 | dwell≥6, tr≤0.15 | ✅ PASS | dwell=6.0(경계), tr=0.116 |
| C3 Distinctness | 6/10 | KW p<0.05 **AND** 부호반전 | ❌ FAIL | kw_p≈0(통계적 유의), **flip=False**(방향 역전 없음) |
| C4 OOS Stability | 4/10 | ρ≥0.5 | ❌ FAIL | rho=0.100 << 0.5 |
| C5 Coverage | 10/10 | 5%≤occ≤60% | ✅ PASS | n_eff=4.65, transition=0%(dead state) |

**가중 종합 (C2-C5):** `0.450` — spec 예측치(0.355)보다 높으나 C3·C4 실패가 결정적

### Phase R2 결정: **분기 B** (이산 code 폐기 → 연속 overlay 직행)

**근거:**
- C3 `sign_flip=False` — 이산 code가 strategy 수익 **방향**을 전환하지 못함. 크기 차이(bull_quiet=max edge)는 있으나 regime 조건부 배분의 핵심 가치인 "A 레짐에서 매수, B 레짐에서 매도"가 부재
- C4 `rho=0.10` — IS에서 확인된 regime-conditional Sharpe 순위가 OOS에서 붕괴. 분포 fitting 우려
- `transition` 레짐 점유율=0% — 6-state 설계지만 실질 5-state

**주의:** 신호 풀 3개(ML-Ready)로 측정한 C3/C4의 **측정 불확실성 높음**. 신호 풀 6개+ 확장 후 재측정 필요.

### Phase R3-B 다음 액션

| 우선순위 | 액션 | 근거 |
|---|---|---|
| **P0** | 신호 풀 확장 (6개+) | ML 신호 3개 → 변별력 불충분, C3/C4 재측정 선행 |
| **P1** | 연속 overlay → allocation prior 연결 | trend_scale/vol_scale/overlay_mult 조건부 가중 회귀 |
| **P2** | btc_trend_20_100 MA-cross 제거 | overlay.trend_scale로 일원화 (일관성 확보) |
| **P3** | transition=0% dead state 처리 | 5-state로 축소 or hysteresis 재설계 |

SSOT: `docs/specs/regime_evaluation_and_hardening.md` → Phase R3-B
