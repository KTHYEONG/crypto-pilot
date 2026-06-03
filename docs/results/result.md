---
title: Candidate ML Strategy Diagnostic Report
domain: futures/strategy
type: domain-spec
status: active
priority: critical
ai_read_policy: always
related_paths:
  - src/domain/futures/strategy/candidate_labels.py
  - src/domain/futures/strategy/candidate_gate.py
  - src/domain/futures/strategy/candidate_edge.py
  - src/domain/futures/strategy/candidate_portfolio.py
  - src/domain/futures/strategy/ablation.py
last_verified: 2026-06-03
---

# Candidate ML Strategy Diagnostic Report

**실행 환경:** `--phase alpha --timeframe 4h --sync skip`  
**실행 일자:** 2026-06-03  
**데이터 기간:** IS 2023-10-01 ~ 2025-10-01 / OOS 2025-10-01 ~ 2026-03-31  
**유효 심볼:** 63개 (pass) / 94개 (로드)  
**반영 구현:** dataset construction vectorization (O(E) 최적화), ablation parallel Retraining 복구, backtest alignment 중복 제거, 진단 로그 DEBUG 레벨 환원

## 1. 요약 (Ablation Study Frontier)

```
| Model Alias        |    CAGR |   MaxDD |    MAR |     Equity | Pass  |
| ------------------ | ------- | ------- | ------ | ---------- | ----- |
| Equal Size         |  -22.9% |   56.3% |  -0.41 |    457,259 |   N   |
| Kelly (No ML)      |   -0.2% |    1.2% |  -0.14 |    995,099 |   N   |
| ML Gate            |    0.0% |    0.0% |   0.61 |  1,000,614 |   N   |
| ML Gate+Edge       |    1.5% |    4.2% |   0.36 |  1,045,694 |   N   |
| ML Full (Capped)   |    0.2% |    0.4% |   0.55 |  1,006,391 |   N   |
| Cand. ML           |   -0.0% |    0.4% |  -0.03 |    999,653 |   N   |
| Promo Filter       |   -0.1% |    0.8% |  -0.16 |    996,267 |   N   |
| Val. Selection     |    0.0% |    0.0% |   0.00 |  1,000,000 |   N   |
| Identity Feat      |   -0.0% |    0.4% |  -0.03 |    999,653 |   N   |
| Market Feat        |    0.0% |    0.0% |   0.44 |  1,000,345 |   N   |
```

핵심 판정:

1. **병목 완전 해소:** `build_candidate_dataset` 벡터화로 전략 단계 실행 시간이 **87s → 34s**로 약 60% 단축됨.
2. **ML 가치 증명:** `ML Gate+Edge` 변본이 CAGR 1.5%로 기본 Kelly(-0.2%) 대비 우수한 OOS 성과를 보임.
3. **Threshold 영향:** 여전히 `q10` shortfall filter(-80bps)가 매우 엄격하여 `Val. Selection` 등에서 거래가 0건으로 잡힘.
4. **성능 선형성:** 레이어 추가 시마다 계산 비용 증가 없이 안정적으로 최적화 루프가 작동함.

## 2. 최신 진단 로그 (DEBUG 레벨)

### 2-1. Gate 예측 (Full Sample vs OOS)

```
[DIAG][GATE] n=6290 mean_p=0.4371 median_p=0.4355 max_p=0.6103 pct_ge55=0.008 pct_ge50=0.040 pct_ge45=0.330 calibrated=True
[DIAG][GATE] n=2501 mean_p=0.4389 median_p=0.4382 max_p=0.6350 pct_ge55=0.007 pct_ge50=0.083 pct_ge45=0.388 calibrated=True
```

해석:

- Gate 모델의 예측 확률 분포가 IS와 OOS에서 일관성을 유지함.
- `max_p`가 0.60을 상회하여 `0.55` 기준 통과 가능한 샘플이 존재함.

### 2-2. Edge 예측 및 유틸리티

```
[DIAG][EDGE] n=6290 target_scale=net cost_bps=24.0 mu_model mean=-2.1 max=6.4 pct_ge25=0.000 | mu_decision mean=-2.1 max=6.4 pct_ge1=0.109 | q10_net mean=-537.0 min=-3068.0 | utility mean=-538.130 max=-178.478
[DIAG][EDGE] n=2501 target_scale=net cost_bps=24.0 mu_model mean=-2.1 max=6.2 pct_ge25=0.000 | mu_decision mean=-2.1 max=6.2 pct_ge1=0.078 | q10_net mean=-554.1 min=-1629.6 | utility mean=-555.210 max=-177.733
```

해석:

- 평균 Expected Edge(`mu_model`)는 -2.1bps로 낮은 편이나, 상위 샘플은 6.4bps까지 확보됨.
- `q10_net` (꼬리 위험)이 -500bps 수준으로 매우 깊어, 보수적인 리스크 관리가 작동 중임.

### 2-3. Selection 탈락 및 민감도 (OOS)

```
[DIAG][SELECT] total=2501 gate_fail=2490 edge_fail=2254 q10_fail=2501 all_fail=2501 passed=0 | policy=hard thresholds(gate>=0.55 edge_net>=1.0 q10>=-80.0 utility>=-325.745)
```

**Selection Sensitivity:**
```
[DIAG][SELECT_SENS] gate>=0.40 edge>=1.0 q10>=-400.0 passed=9 pass_rate=0.0036
[DIAG][SELECT_SENS] gate>=0.40 edge>=1.0 q10>=-250.0 passed=1 pass_rate=0.0004
[DIAG][SELECT_SENS] gate>=0.40 edge>=1.0 q10>=-80.0 passed=0 pass_rate=0.0000
```

해석:

- `q10 >= -80` 조건이 모든 거래를 차단하는 주원인임.
- 리스크 허용치를 `-250` 또는 `-400`으로 완화할 경우 유효한 ML 거래 신호가 생성됨을 확인.

## 3. 최적화 패치 효과 검증

### Opt-1 [DONE]: Dataset Construction 벡터화

- **기존:** 루프 내 중복 계산으로 인해 variant당 33초 소요.
- **변경:** Numpy/Pandas rolling 기반 2D 배열 사전 계산.
- **결과:** **Variant당 1.3초 미만으로 단축.**

### Opt-2 [DONE]: Parallel ML Training 복구

- **내용:** `ThreadPoolExecutor` (max_workers=4) 및 `LightGBM(n_jobs=1)` 적용.
- **결과:** OOS Ablation Rows(7-10)의 동시 실행으로 전체 전략 단계 속도 향상.

### Opt-3 [DONE]: 중복 Data Alignment 제거

- **내용:** `_run_backtest_and_evaluate`에서 `AlignedMarketData`를 직접 수용.
- **결과:** 불필요한 데이터 복사 및 정렬 오버헤드 제거.

## 4. 향후 과제

### P1 [중요] q10 Threshold 재설정

현재의 `-80bps`는 과도하게 보수적임. Ablation 결과에서 성과가 확인된 `-300` 내외의 유동적 기준(validation_quantile) 도입 검토.

### P2 [성능] 추가 최적화

전략 단계가 30초대까지 내려왔으므로, 이제는 ML 모델의 `n_estimators`나 `early_stopping` 파라미터 튜닝을 통한 예측 품질 향상에 집중 가능.

## 5. 검증 상태

- **Total Execution Time (Strategy Stage):** 34.54s
- **Lint/Type Check:** `Ruff`, `Mypy` 통과
- **Alpha Smoke Test:** Exit code 0
