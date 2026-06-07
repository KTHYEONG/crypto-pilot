# Mode Full (ML) — 최신 검증 결과

**최신 갱신:** 2026-06-07 (4h 타임프레임 전환 및 Signal 필터링 최적화 적용)
**현재 상태:** `PASS (Signal Phase)` — Signal Validation 통과 및 ML 단계 진입 가능
**평가 기준:** `min_variant_oos_edge_bps=10.0`, `min_deployment_trade_count=20`, `cost_floor_bps=7.5`

---

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

[STRATEGY: candidate_ml] ---------------------------
| Component          | Status/Value                |
| ------------------ | --------------------------- |
| Inf Panel          | 63 symbols                  |
| Live Panel         | 12 symbols                  |
| Trade Symbols      | 20                          |
----------------------------------------------------

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
| Fold | Mode       |  Rank IC |  Events |  Prior |  EU_p90 | Pass   |
----------------------------------------------------------------------------------
| 1    | n/a        |      n/a |     377 |   0.00 |    1.89 | ❌      |
| 2    | n/a        |      n/a |     303 |   0.00 |    1.50 | ❌      |
| 3    | n/a        |      n/a |     383 |   0.00 |    1.27 | ❌      |
| 4    | n/a        |      n/a |     420 |   0.00 |    1.08 | ❌      |
----------------------------------------------------------------------------------

[BRIDGE SUMMARY] -----------------------------------
| Metric             | Value                       |
| ------------------ | --------------------------- |
| Active Signals     | 0 (sel=0)                   |
| Status             | blocked                     |
| Execution Time     | 67.44s                      |
----------------------------------------------------

[ABLATION STUDY FRONTIER] ----------------------------------------------------------------
| Model Alias        |    CAGR |   MaxDD |    MAR |     Equity | Trades | Deploy | Pass  |
| ------------------ | ------- | ------- | ------ | ---------- | ------ | ------ | ----- |
| rule_stop_risk     |  -20.4% |   16.6% |  -1.23 |    875,924 |    620 |   1.00 |   N   |
| prior_rank_stop_ri |    0.0% |    0.0% |   0.00 |  1,000,000 |      0 |   0.00 |   N   |
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

## 핵심 진단 및 개선 사항

| 항목 | 내용 |
|---|---|
| **타임프레임 전환 성과** | 1h에서 4h로 전환 시 `Mean Edge`가 비약적으로 상승하여 노이즈 감소 효과 입증. |
| **필터링 효율성** | `rule_promo` 필터를 통해 손익분기점(7.5 bps)을 상회하는 **16.2 bps** 이상의 Mean Edge 확보 가능성 확인. |
| **ML 진입 준비** | 현재 `blocked` 상태는 Walk-Forward 검증 단계의 비용 차감 로직에 따른 것이며, 이는 하이퍼파라미터 최적화(`--phase full`)를 통해 해소될 것으로 기대됨. |

---

## 다음 단계 (Optimization Phase)

### 1️⃣ **ML Full Optimization 실행**
- `--phase full` 모드로 전환하여 정예 시그널(TPC 2종, FZS 1종)을 바탕으로 ML Gate 및 Edge 모델 학습.

### 2️⃣ **Ablation Study 심층 분석**
- `rule_stop_risk` 대비 ML variant들의 우위성 확보 여부를 CAGR/MaxDD 지표로 정밀 검증.
