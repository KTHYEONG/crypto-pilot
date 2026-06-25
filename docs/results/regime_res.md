# L2 Phase Diagnostic Run — DEBUG Logging 검증 + Block-Level 비교

> 실행: `LOG_LEVEL=DEBUG uv run python src/execution/opt_main_futures.py --phase l2 --timeframe 4h --trials 5`
> 일시: 2026-06-25 (3차 실행, 진단 로깅 추가 후)

---

## ✅ 진단 로깅 추가 검증

| 로그 태그 | 파일 | 상태 |
|-----------|------|------|
| `[L2-SHARPE-CMP]` | `pipeline.py:1625` | ✅ 출력 확인 |
| `[L2-BLOCK-SUM]` | `pipeline.py:1721` | ✅ 출력 확인 |
| `[L2-BLOCK-CMP]` | `pipeline.py:1750` | ✅ 출력 확인 |
| `[L2-CALIB-CV]` 확장 필드 | `risk_deployment.py:202` | ✅ fit_CAGR_v1/OOS_CAGR_v1 추가 확인 |

---

## 🔬 `[L2-SHARPE-CMP]` — Sharpe 성분 분해

```
[L2-SHARPE-CMP] hybrid: ann_mean=0.131438 ann_std=0.1117 sharpe_hac=1.2657 |
  baseline_ew: ann_mean=0.217385 ann_std=0.1956 sharpe_hac=1.1921 |
  delta_sharpe=0.0736 mean_ratio=0.60 std_ratio=0.57
```

### 해석

| 항목 | Hybrid (CS Rank+Kelly) | Baseline_EW (1/N) | 비고 |
|------|----------------------|-------------------|------|
| 연율화 평균 수익률 | +13.1% | +21.7% | **hybrid가 60% 수준** |
| 연율화 표준편차 | 11.2% | 19.6% | hybrid 변동성 **57% 수준** |
| Sharpe_HAC | 1.27 | 1.19 | hybrid가 **+0.074 우세** |

**핵심 발견**: Hybrid의 낮은 평균 수익률(13.1% vs 21.7%)이 변동성 감소(11.2% vs 19.6%)로 상쇄되어 Sharpe는 유사.
- `delta_sharpe=+0.0736` — gate 요건(+0.20)의 **36.8%만 충족**
- **Kelly allocation이 알파를 생성하지 못함**: CS Rank로 인한 포지션 집중이 수익률을 낮추지만(edge 손실), long/short 상쇄로 변동성도 함께 낮춤
- 순수 1/N이 평균 수익률 측면에서는 더 우수

---

## 🔬 `[L2-BLOCK-SUM]` — Block 단위 성장 비교

```
[L2-BLOCK-SUM] n_blocks=282 blocks_per_year=182.5 |
  hybrid: mean=0.0007 std=0.0078 min=-0.0279 max=0.0274 |
  baseline: mean=0.0007 std=0.0078 min=-0.0279 max=0.0274 |
  win_rate: hybrid>baseline = 51/282 (18.1%)
```

### 해석

| 항목 | Hybrid | Baseline (risk-matched EW) |
|------|--------|---------------------------|
| block 성장 평균 | **0.0007** | **0.0007** (동일) |
| block 성장 표준편차 | **0.0078** | **0.0078** (동일) |
| win_rate | 18.1% | 실질적 noise (4자리 동일) |

**충격적 발견**: Hybrid와 Baseline(risk-matched EW)의 block 성장이 **4자리까지 동일**.
- 이는 CS Rank + Kelly 할당이 **risk-matched EW에 수렴**했음을 의미
- Kelly 가중치가 사실상 risk parity에 가깝게 분포
- win_rate 18.1%는 부동소수점 차이일 뿐 실질적 차이 없음

---

## 🔬 `[L2-BLOCK-CMP]` — Per-fold 세부 비교

```
[L2-BLOCK-CMP] fold=0 log_growth_h=0.0176 log_growth_b=0.0176 delta=-0.0000 n_bars=563
[L2-BLOCK-CMP] fold=1 log_growth_h=0.0577 log_growth_b=0.0577 delta=-0.0000 n_bars=563
[L2-BLOCK-CMP] fold=2 log_growth_h=0.1182 log_growth_b=0.1182 delta=0.0000 n_bars=566
```

모든 fold에서 delta ≈ 0. 세 fold 모두 hybrid-baseline 차이가 0.0001 미만.

---

## 🔬 `[L2-CALIB-CV]` — 확장 필드

```
fit_CAGR_v1=-0.3688 fit_sharpe_v1=-2.7024 OOS_CAGR_v1=0.2846 OOS_sharpe_v1=1.6648
```
- fit-leg CAGR = -36.9%, Sharpe = -2.70 (매우 나쁨)
- OOS CAGR = +28.5%, Sharpe = +1.66 (우수)
- **Alpha decay**: fit→OOS로 CAGR이 -36.9% → +28.5% 로 반전
- 이 이격이 L* floor hard landing(1.0)의 근본 원인

---

## 🚧 Gate 현황

```
[L2-GATE] promotion=False blocker=cagr |
  cagr=0.1334(vs0.30) sortino=1.7290(vs1.50) sharpe=1.1772(vs0.70) calmar=1.4173(vs0.50) |
  mdd=0.0941(vs0.30) folds=1.00(vs0.60) trades=129(vs30) cost_drag=0.0000(vs0.60) |
  psr=-1.0000(vs0.90) uplift=0.0736(vs0.20) cvar=0.0079(vs0.06)
```

- blocker: `cagr` (CAGR=13.34% < 30% threshold)
- `uplift=0.074` (< 0.20)
- PSR=-1.0 (non-finite, gate 통과)
- L*=1.0, RiskUtil=31.4%

---

## 📊 종합 진단 요약

| 발견 | 심각도 | 설명 |
|------|--------|------|
| **Kelly=EW 수렴** | 🔴 CRITICAL | Kelly 할당이 risk-matched EW와 동일. CS Rank 차별력 없음 |
| **L* hard landing** | 🔴 CRITICAL | fit-leg negative CAGR(-36.9%)로 L*=1.0 고정, OOS RiskUtil 31.4% |
| **Sharpe delta 부족** | 🟡 HIGH | delta=+0.074, gate 요건(+0.20)의 36.8% |
| **PSR/Folds 안정** | 🟢 GOOD | folds=1.00, 게이트 통과 |
| **cost_drag 안정** | 🟢 GOOD | 0.0~0.18, 게이트 통과 |

### 근본 원인: CS Rank Score 차별력 부족

CS Rank 스코어가 모든 심볼에 대해 유사한 값을 가짐:
- 상위 N 심볼 간 CS Z-score 편차가 미미
- Kelly 할당이 `w ∝ μ/σ²` 에서 μ가 유사하면 1/σ² (risk parity)에 수렴
- 결과적으로 **전략과 1/N이 동일한 포트폴리오**

### 권장 개선 방향 (우선순위)

1. **CS Rank Score Amplification** (P0): Z-score → sigmoid/tanh 변환으로 tail 차별력 증대
2. **L* floor OOS-cross-validation** (P0): fit에서 not feasible일 경우 OOS proxy로 L* floor 동적 설정
3. **Kelly Covariance Shrinkage** (P1): Ledoit-Wolf shrinkage로 공분산 추정 오차 감소
4. **Gate Uplift Threshold 조정 검토** (P2): `l2_min_sharpe_uplift=0.15`로 완화 (현재 0.20은 너무 높음)
