# 목적
유효한 signal 들을 여러개 생성 후 기존의 1개의 전략으로만 매매하는 한계점을 보완하여 ML으로 동적으로 적재적소에 전략을 사용하여 복리자산증식 전략 도출해내는 것임

# Mode Full (ML) — 최신 검증 결과

**최신 갱신:** 2026-06-08 (Signal Rising-Edge Refactor — 4개 family persistent-state → transition entry)
**현재 상태:** `WF_ELIGIBLE (ML Phase)` — ML-Ready 3→5개, C4 rho 0.029→0.314
**평가 기준:** `min_variant_oos_edge_bps=10.0`, `min_deployment_trade_count=20`, `cost_floor_bps=7.5`
**진단 노트:** bollinger/vol_regime/dual_momentum/btc_corr_regime 4개 family에 `_entry_rising_edge_2d` 적용. raw events 108k→73k 감소, ML-Ready 3→5 (dm_24_96, dm_12_48 신규 추가). C4 rho 0.029→0.314 개선. **잔여 블로커**: C3 flip=N 유지, C4 rho<0.5 미달. 다음 step = mean-reversion 추세장 진입 차단(archetype-regime entry gating). SSOT: `docs/specs/mean_reversion_regime_entry_gating.md`

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
[UNIVERSE] Discovery complete: 94 symbols (2.31s)

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
| 3    | btc_corr_regime:bcr_48              | 183 (84)     |        63.3 |    53.6% |   1.89 |  0.075 | DROP   | N   |
| 4    | btc_corr_regime:bcr_96              | 92 (41)      |        58.1 |    51.2% |   2.20 | -0.003 | DROP   | N   |
| 5    | dual_momentum:dm_12_48              | 1953 (764)   |        25.5 |    40.3% |   1.32 | -0.004 | KEEP   | Y   |
| 6    | rsi_reversion:rsi_14                | 4384 (1811)  |        22.6 |    43.8% |   1.30 | -0.082 | KEEP   | N   |
| 7    | funding_zscore_carry:fzs_48         | 1927 (680)   |        17.6 |    42.9% |   1.62 | -0.056 | KEEP   | N   |
| 8    | funding_zscore_carry:fzs_168        | 3678 (1286)  |        17.5 |    44.4% |   1.41 | -0.025 | KEEP   | N   |
| 9    | residual_reversion:rr_24            | 1212 (494)   |        14.6 |    43.4% |   1.01 | -0.032 | DROP   | N   |
| 10   | funding_carry:funding_24            | 3743 (1515)  |        13.3 |    43.2% |   1.50 | -0.038 | KEEP   | N   |
| 11   | funding_zscore_carry:fzs_96         | 1908 (652)   |        12.1 |    41.9% |   1.37 | -0.096 | KEEP   | Y   |
| 12   | rsi_reversion:rsi_6                 | 4908 (2000)  |        12.1 |    42.3% |   1.08 | -0.078 | DROP   | N   |
| 13   | cross_sectional_momentum:cs_mom_10  | 10547 (4389) |        10.8 |    41.9% |   1.17 | -0.052 | DROP   | N   |
| 14   | vol_regime_reversion:vrr_40         | 2545 (900)   |        10.8 |    42.4% |   1.16 | -0.015 | DROP   | N   |
| 15   | trend_donchian:donchian_36          | 1788 (728)   |        10.2 |    43.3% |   1.39 |  0.059 | KEEP   | N   |
| 16   | residual_reversion:rr_48            | 1250 (511)   |         9.9 |    40.5% |   0.88 | -0.055 | DROP   | N   |
| 17   | funding_acceleration_carry:fac_48   | 15017 (6579) |         9.6 |    42.6% |   1.15 | -0.027 | DROP   | N   |
| 18   | btc_regime_pullback:btc_pullback_50 | 1953 (660)   |         8.9 |    41.1% |   0.99 | -0.141 | DROP   | N   |
| 19   | funding_acceleration_carry:fac_168  | 14978 (6776) |         8.9 |    42.2% |   1.17 |  0.001 | DROP   | N   |
| 20   | cross_sectional_momentum:cs_mom_20  | 15403 (6337) |         8.8 |    42.3% |   1.15 | -0.048 | DROP   | N   |
-----------------------------------------------------------------------------------------------------------------------

[SIGNAL DIAGNOSTICS: STRATEGY FILTERING]
----------------------------------------------------------------------------------
| Action       | Count | Details / Selected Strategies                           |
----------------------------------------------------------------------------------
| BLOCKED      | 27    | Fail Reasons: Event Overload (16) | Poor Hit/Payoff (2) |
|              |       | Top Blocked: vrr_20, bollinger_20, vrr_40, fzs_168, rr |
| RECOMMENDED  | 5     | 1. dual_momentum:dm_24_96                               |
| (ML Ready)   |       | 2. dual_momentum:dm_12_48                               |
| (ML Ready)   |       | 3. trend_pullback_continuation:tpc_50_200               |
| (ML Ready)   |       | 4. funding_zscore_carry:fzs_96                          |
| (ML Ready)   |       | 5. trend_pullback_continuation:tpc_20_100               |
----------------------------------------------------------------------------------

[BRIDGE SUMMARY] -----------------------------------
| Metric             | Value                       |
| ------------------ | --------------------------- |
| Active Signals     | 0 (sel=0)                   |
| Status             | blocked                     |
| Execution Time     | 27.29s                      |
----------------------------------------------------

[SIGNAL VALIDATION: FILTERING IMPACT]
--------------------------------------------------------------------------------------------------------------------------
| Variant                |   Events |   Hit Rate |  Mean Edge | Status          | Fail Reasons                                 |
--------------------------------------------------------------------------------------------------------------------------
| rule_only (Raw)        |   73,592 |      41.0% |    6.3 bps | ❌ FAIL         | mean_net_stress_below_floor,hac_t_below_floor |
| rule_promo (Filter)    |    2,818 |      39.0% |   29.0 bps | ✅ PASS         | -                                            |
--------------------------------------------------------------------------------------------------------------------------
>> Conclusion: Filtering improved Mean Edge by +22.7 bps. Proceeding to ML phase.

----------------------------------------------------------------------------------
| [REGIME_SCORECARD] (4h, 2026-06-08)                                            |
----------------------------------------------------------------------------------
| Axis                 |  Score  | Key Metrics                                   |
----------------------------------------------------------------------------------
| C2 Persistence       | 3.0/10  | dwell=3.00(micro) macro=3.00 tr=0.162 ent=0.619 |
| C3 Distinctness      | 6.0/10  | kw_p=0.0000  flip=N  mi=0.0248                |
| C4 OOS Stability     | 4.0/10  | rho=0.314  n_regimes=6                        |
| C5 Coverage          | 10.0/10 | min=0.089 max=0.224 n_eff=5.77                |
----------------------------------------------------------------------------------
| Weighted C2-C5       |  0.375  | C1(hard_gate=pass)  C6-C8(manual)             |
----------------------------------------------------------------------------------
| Occupancy            |   n/a   | bull_q=0.224  bull_vol=0.212  bear_q=0.172    |
|                      |         | bear_vol=0.141  trans=0.089  crash=0.163      |
| C3/C4_proxy (mkt)    |   n/a   | kw_p=0.000 flip=Y rho=0.886                   |
----------------------------------------------------------------------------------

[ABLATION STUDY FRONTIER] ----------------------------------------------------------------
| Model Alias        |    CAGR |   MaxDD |    MAR |     Equity | Trades | Deploy | Pass  |
| ------------------ | ------- | ------- | ------ | ---------- | ------ | ------ | ----- |
| rule_stop_risk     |  -38.1% |   24.9% |  -1.53 |    756,823 |    630 |   1.00 |   N   |
| prior_rank_stop_ri |   -0.1% |    3.3% |  -0.02 |    999,647 |    129 |   0.24 |   N   |
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
| 11 | **fzs_96** | **12.1** | 41.9% | 1.37 | KEEP | **Y** |
| n/a | **tpc_20_100** | **~10.0** | n/a | n/a | KEEP | **Y** |

---

## Regime 평가 분석 (Phase 6 gold standard 확정)

**측정일:** 2026-06-08 | **기준:** `docs/specs/regime_architecture_audit_and_hardening.md` 8축 루브릭

### 축별 판정 (gold standard 확정값)

| 축 | 점수 | 임계 | 판정 | 근거 |
|---|---:|---|---|---|
| C1 Look-ahead | 9/10 | hard gate | ✅ PASS | CUSUM/EMA causal, entry-1 소비 |
| C2 Persistence | 3/10 | dwell≥6, tr≤0.15 | ❌ FAIL | macro_dwell=3.00 (4h crypto 방향 ≈12h, 구조적 특성) |
| C3 Distinctness | 6/10 | KW p<0.05 AND 부호반전 | ⚠️ PARTIAL | kw_p=0.0000(유의), **flip=N**(방향 역전 없음) |
| C4 OOS Stability | 4/10 | ρ≥0.5 | ❌ FAIL | rho=0.314 (IS→OOS 부분 붕괴, 개선 중) |
| C5 Coverage | 10/10 | 5%≤occ≤60% | ✅ PASS | n_eff=5.77, transition=8.9% |

**가중 종합 (C2-C5):** `0.375` (gold standard events=4013 기준)

### 핵심 진단: Proxy vs Gold 괴리

| 측정 대상 | kw_p | flip | rho | 해석 |
|---|---|---|---|---|
| **Proxy** (BTC 시장수익) | 0.000 | ✅ Y | 0.886 | 레짐이 **시장 방향을 강하게 구분** |
| **Gold** (전략 엣지) | 0.0078 | ❌ N | 0.029 | 레짐이 **전략 엣지 방향을 구분 못함** |

→ 레짐은 시장 상태(BTC 방향)를 잘 분류하지만, 전략 수익의 방향 전환을 일으키지 못한다.  
→ C3 flip=N: 모든 레짐에서 전략이 같은 방향(long bias)으로 수익. 레짐이 크기는 바꾸지만 방향은 못 바꿈.  
→ C4 rho=0.314: IS 레짐별 Sharpe 순위가 OOS에서 부분적으로 유지 (이전 0.029→0.314). events=7370(5개 신호)으로 측정 신뢰도 향상 중. ρ≥0.5 미달.

### 다음 단계

신호 풀 3개(ML-Ready)로 측정한 C3/C4의 **측정 신뢰도 낮음** — 신호 풀 6+개 확장 후 재측정 필요.  
SSOT: `docs/specs/ml_regime_allocation.md` Priority 0.

### Phase 1~5 + C2 macro 재설계 결과 요약 (2026-06-08)

| 결함/작업 | 조치 | 결과 |
|---|---|---|
| D1 transition dead state | percentile band 자기보정 (Phase 1+4) | ✅ transition 0%→8.9% |
| D2 독립 4-state SSOT 위반 | rule_diagnostics CUSUM 단일화 | ✅ 제거 완료 |
| D3 고정 vol 경계 | expanding median 적응 임계 | ✅ 적용 완료 |
| D4 overlay_lift raw 처벌 | Sharpe 차분으로 교체 | ✅ 적용 완료 |
| D5 평가 대상 불일치 | overlay IC + C3 magnitude_sep | ✅ 적용 완료 |
| D8 C2 측정 대상 오류 | macro_dwell(방향 수준)로 교체 (임계 ≥6 유지) | ✅ micro=3/macro=3 — 방향 전환 자체 빈번 진단 |
| D9 C3/C4 pre-signal 불가 | proxy (시장수익 기반) 추가 (Phase 5) | ✅ flip=Y, rho=0.886 |
| C3/C4 gold standard 연결 | 이벤트 데이터 → scorecard 연결 경로 부재 | ❌ 구조적 공백 — 다음 P0 |

### 다음 액션 (Phase 6+7 완료 후 갱신)

| 우선순위 | 액션 | 근거 |
|---|---|---|
| ~~P0~~ | ~~C3/C4 gold standard 연결~~ | ✅ **완료** (Phase 6: events=4013→7370, C3 kw_p=0.0000·C4 rho=0.029→0.314, ML-Ready 3→5) |
| ~~P1~~ | ~~gate-failure diagnostics 추가~~ | ✅ **완료** (BLOCKED 사유 세분화 및 Signal Validation 실패 사유 노출) |
| **P2** | causal signal expansion 후 C3/C4 재측정 | threshold 완화 없이 orthogonal signal 후보를 추가하고, flip=Y·rho≥0.5 달성 시에만 regime-conditional prior 진입 |
| **P3** | regime-conditional prior 설계 | 윈도우 제약 ≤ 3×macro_dwell(≈9 bars) 필수 — 12h 방향 회전 대응 ([[ml_regime_allocation]] P1) |
| P4 (후순위) | `regime_c2_dwell_target` timeframe-relative 파라미터화 | D10 — C2 임계 4h 부적합 (현재 측정은 정확, 임계만 과함) |

SSOT: `docs/specs/ml_regime_allocation.md` (배분), `docs/architecture/regime.md` (아키텍처)
