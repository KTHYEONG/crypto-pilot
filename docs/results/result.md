# 목적
유효한 signal 들을 여러개 생성 후 기존의 1개의 전략으로만 매매하는 한계점을 보완하여 ML으로 동적으로 적재적소에 전략을 사용하여 복리자산증식 전략 도출해내는 것임

# Mode Full (ML) — 최신 검증 결과

**최신 갱신:** 2026-06-08 (Phase 6+7 완료 — C3/C4 gold standard 연결, macro_dwell 함의 문서화)
**현재 상태:** `WF_ELIGIBLE (ML Phase)` — 3/4 fold PASS, Active Signals 531개 선정
**평가 기준:** `min_variant_oos_edge_bps=10.0`, `min_deployment_trade_count=20`, `cost_floor_bps=7.5`
**진단 노트:** Fix-1/Fix-2로 복원한 베이스라인. Phase 1(cal prior update)·Phase 2(fit_frac=0.40) 실험 모두 악화 → 롤백. **근본 블로커**: center ML rank-IC≈0.000(전 fold, 무예측력). 원인은 ML-Ready 신호 3개×~1500 훈련이벤트로 변별 불가. 해결책: 신호 풀을 6+개로 확장(Priority 1). SSOT: `docs/specs/ml_dynamic_allocation_roadmap.md`

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

[REGIME_SCORECARD] (4h, 2026-06-08) ← Phase 6+7 완료 / gold standard 연결 후 최종
------------------------------------------------------------
| Axis                 |  Score  | Key Metrics                                        |
------------------------------------------------------------
| C2 Persistence       | 3.0/10  | dwell=3.00(micro)  macro=3.00  tr=0.162  ent=0.619 |
| C3 Distinctness      | 6.0/10  | kw_p=0.0078  flip=N  mi=0.0250                     |
| C4 OOS Stability     | 4.0/10  | rho=0.029  n_regimes=6                             |
| C5 Coverage          |10.0/10  | min=0.089  max=0.224  n_eff=5.77                   |
------------------------------------------------------------
| Weighted C2-C5       |  0.375  | C1(hard_gate=pass)  C6-C8(manual)                  |
------------------------------------------------------------
| Occupancy            | bull_quiet=0.224  bull_volatile=0.212  bear_quiet=0.172     |
|                      | bear_volatile=0.141  transition=0.089  crash=0.163          |
| C3/C4_proxy (mkt)    |   n/a   | kw_p=0.000  flip=Y  rho=0.886                      |
------------------------------------------------------------
* [Phase 6] C3/C4 gold standard 연결 완료 (events=4013: IS=2512, OOS=1501)
  - C3 kw_p=0.0078 (유의) / flip=N (부호반전 없음) / mi=0.0250
  - C4 rho=0.029 (OOS 불안정) → gold standard 확정치
  - Weighted 0.185 → 0.375 (C3/C4 채워짐)
* [핵심 진단] proxy vs gold 괴리: 레짐이 시장 수익은 강하게 구분(flip=Y, rho=0.886)하지만
  전략 엣지는 약하게 구분(flip=N, rho=0.029). 레짐이 시장 방향은 잡지만 전략 성과 방향은 못 잡음.
  → regime-conditional 배분 전에 신호 풀 확장(6+)으로 측정 신뢰도 확보 필요
* C2: macro_dwell=3.00 = 4h crypto 방향 지속성 ≈ 12h (구조적 특성, 버그 아님)
* C5: n_eff=5.77 (6-state 균등화 양호)

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
| 1    | direct     |    0.000 |     377 |     0.00 |   60.57 | ❌      |
| 2    | prior_only |    0.000 |     303 |    80.97 |   80.97 | ✅      |
| 3    | prior_only |    0.000 |     383 |    69.31 |   69.31 | ✅      |
| 4    | prior_only |    0.000 |     420 |    61.76 |   61.76 | ✅      |
----------------------------------------------------------------------------------

[BRIDGE SUMMARY] -----------------------------------
| Metric             | Value                       |
| ------------------ | --------------------------- |
| Active Signals     | 4461 (sel=531)              |
| Status             | wf_eligible                 |
| Execution Time     | 77.09s                      |
----------------------------------------------------

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
| C3 Distinctness | 6/10 | KW p<0.05 AND 부호반전 | ⚠️ PARTIAL | kw_p=0.0078(유의), **flip=N**(방향 역전 없음) |
| C4 OOS Stability | 4/10 | ρ≥0.5 | ❌ FAIL | rho=0.029 (IS→OOS 붕괴) |
| C5 Coverage | 10/10 | 5%≤occ≤60% | ✅ PASS | n_eff=5.77, transition=8.9% |

**가중 종합 (C2-C5):** `0.375` (gold standard events=4013 기준)

### 핵심 진단: Proxy vs Gold 괴리

| 측정 대상 | kw_p | flip | rho | 해석 |
|---|---|---|---|---|
| **Proxy** (BTC 시장수익) | 0.000 | ✅ Y | 0.886 | 레짐이 **시장 방향을 강하게 구분** |
| **Gold** (전략 엣지) | 0.0078 | ❌ N | 0.029 | 레짐이 **전략 엣지 방향을 구분 못함** |

→ 레짐은 시장 상태(BTC 방향)를 잘 분류하지만, 전략 수익의 방향 전환을 일으키지 못한다.  
→ C3 flip=N: 모든 레짐에서 전략이 같은 방향(long bias)으로 수익. 레짐이 크기는 바꾸지만 방향은 못 바꿈.  
→ C4 rho=0.029: IS 레짐별 Sharpe 순위가 OOS에서 완전 붕괴 — 레짐 조건부 전략 순위의 신호 없음.

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
| ~~P0~~ | ~~C3/C4 gold standard 연결~~ | ✅ **완료** (Phase 6: events=4013, C3 kw_p=0.0078·C4 rho=0.029 확정) |
| **P1** | 신호 풀 6개+ 확장 | ML Rank-IC=0.000 근본 해결 + C3/C4 측정 신뢰도 제고 (3개 신호로는 gold standard 불확실) |
| **P2** | 신호 확장 후 C3/C4 재측정 | flip=Y·rho≥0.5 달성 시에만 regime-conditional prior 진입 |
| **P3** | regime-conditional prior 설계 | 윈도우 제약 ≤ 3×macro_dwell(≈9 bars) 필수 — 12h 방향 회전 대응 ([[ml_regime_allocation]] P1) |
| P4 (후순위) | `regime_c2_dwell_target` timeframe-relative 파라미터화 | D10 — C2 임계 4h 부적합 (현재 측정은 정확, 임계만 과함) |

SSOT: `docs/specs/ml_regime_allocation.md` (배분), `docs/architecture/regime.md` (아키텍처)
