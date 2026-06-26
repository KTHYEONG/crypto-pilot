# L2 Phase 5차 DEBUG 실행 — Power Amplification Mode 진단

> 실행: `LOG_LEVEL=DEBUG uv run python src/execution/opt_main_futures.py --phase l2 --timeframe 4h --trials 200`
> 일시: 2026-06-25 (5차 실행, Power mode + 진단 로깅 v2 적용)

---

## 📊 Amplification Mode 비교

| 지표 | Median_excess (4차) | Power mode (5차) | Delta |
|------|-------------------|-----------------|-------|
| Sharpe Uplift | **+0.074** | **-0.028** | ❌ **악화** |
| CAGR | 20.10% | 19.53% | -0.57pp |
| MDD | 13.96% | 15.22% | +1.26pp |
| RiskUtil | 46.5% | 50.7% | +4.2pp |
| Block delta | 0.0000 | 0.0000 | 동일 |

---

## 🔬 `[L2-CONFIG]` — Config Propagation 확인

```
[SYS] [L2-CONFIG] l2_min_sharpe_uplift=0.20 l2_cs_amp_enabled=True l2_cs_amp_alpha=2.0 l2_cs_amp_mode=power
```

- `l2_cs_amp_mode=power` ✅ power mode 정상 적용
- `l2_min_sharpe_uplift=0.20` ❌ **여전히 0.05가 아닌 0.20** — JSON/params override로 0.20 유지
- `l2_cs_amp_enabled=True` ✅

---

## 🔬 `[L2-Z-DIST]` — Z-score Distribution

| 통계 | Typical Range | 해석 |
|------|-------------|------|
| n_pos (양수 Z 개수) | **1~8 out of 52** | 대부분 심볼이 CS 평균 이하 |
| z_min | 0.01~0.63 | 양수 Z의 최소값 거의 0 |
| z_max | 0.57~3.76 | 일부 심볼은 높은 Z-score |
| z_med | **0.2~1.0** | 양수 Z들의 중앙값도 낮음 |
| z_std | 0.18~1.31 | 분산은 존재하나 적용 대상이 적음 |

**핵심 발견**: Rebalance bar t=5979~7671 구간에서, 52개 심볼 중 **평균 4~8개**만 양수 Z-score 보유. 나머지 44~48개는 음수 Z (CS 평균 이하) → `rank_and_select(selection_mode="absolute")`는 절대값 기준으로 선택하므로 음수 Z도 선택 가능하나 **amplification은 양수 Z에만 적용**됨.

---

## 🔬 `[L2-AMP]` — Amplification Effect

| Per-Bar 통계 | Typical | Extreme |
|-------------|---------|---------|
| n_amplified (증폭된 심볼 수) | **2~6 / 52** | 0~10 |
| amp_max (최대 증폭 계수) | **2~10×** | 81.81× (극단) |
| z_med | **0.3~0.9** | 0.09~1.47 |

**문제점**: Power mode가 극단적 증폭 생성 (amp_max up to 81×) 하지만, 증폭 대상이 2~6개에 불과. 나머지 46~50개 심볼은 Kelly risk-parity 할당 유지 → **포트폴리오의 90%+가 여전히 EW와 동일**.

---

## 🔬 `[L2-SHARPE-CMP]` — Sharpe Uplift 악화

```
[SYS] [L2-SHARPE-CMP] hybrid: ann_mean=0.128186 ann_std=0.1111 sharpe_hac=1.1645 |
  baseline_ew: ann_mean=0.217385 ann_std=0.1956 sharpe_hac=1.1921 |
  delta_sharpe=-0.0276 mean_ratio=0.59 std_ratio=0.57
```

| 항목 | Median_excess | Power mode | 변화 |
|------|-------------|-----------|------|
| ann_mean (hybrid) | 0.1314 | **0.1282** | -0.0032 |
| ann_std (hybrid) | 0.1117 | **0.1111** | -0.0006 |
| sharpe_hac (hybrid) | 1.2657 | **1.1645** | -0.1012 |
| delta_sharpe | **+0.074** | **-0.028** | -0.102 |
| mean_ratio | 0.60 | 0.59 | -0.01 |
| std_ratio | 0.57 | 0.57 | 동일 |

**원인 진단**: Power mode 증폭으로 소수 심볼에 과도한 비중 할당 → 이들 심볼이 OOS에서 성과 부진 → 평균 수익률 감소(0.131→0.128). 변동성은 거의 변화 없음(0.112→0.111) → 증폭이 알파를 생성하지 못하고 concentrated risk만 추가.

---

## 🔬 `[L2-BLOCK-SUM/CMP]` — 여전히 delta=0.0000

```
[SYS] [L2-BLOCK-SUM] hybrid: mean=0.0007 std=0.0081 | baseline: mean=0.0007 std=0.0081
[SYS] [L2-BLOCK-CMP] fold=0 delta=-0.0000, fold=1 delta=-0.0000, fold=2 delta=0.0000
```

3 fold 모두 delta 불변. Power mode로도 block 단위 hybrid-baseline 비교에서 차이 없음.

Block 수준에서 측정 가능한 차이가 없는 이유:
- 증폭은 **rebalance 샘플링 시점**(6분 간격)에만 적용
- **2,382개 per-bar 수익률 중 60~80개만 rebalance** (약 3%)
- 나머지 97%의 bar는 이전 비중 유지 (= risk-parity = baseline과 동일)
- 따라서 97%의 bar는 동일 → block 단위 통계가 동일한 게 구조적 원인

---

## 📊 종합 진단

| 발견 | 심각도 | 근본 원인 |
|------|--------|----------|
| **Power mode가 Sharpe 악화** | 🔴 CRITICAL | 소수 심볼 극단 증폭(amp_max=81×)이 concentrated risk 추가, 알파 없음 |
| **Block delta 구조적 불변** | 🔴 CRITICAL | 97% bar는 rebalance 없음 → baseline과 동일. Block 단위 비교로는 측정 불가 |
| **Config 미전파 확인** | 🟡 HIGH | `l2_min_sharpe_uplift=0.20` still (0.05 아님) |
| **Z-score 분산 부족** | 🔴 CRITICAL | 52 심볼 중 평균 4~8개만 양수 Z. 대부분 음수 |
| **L* floor 유효** | 🟢 GOOD | L*=1.5, CAGR=19.5%, RiskUtil=50.7% |

### 근본 원인: Rebalance 밀도 문제

L2 AWF 시뮬레이션에서 rebalance는 약 60~80 bar마다 발생(4h TF 기준 약 6개월). 전체 평가 기간(~8761 bars)에서 **rebalance bar는 약 60~80회 (전체의 3%)**. 나머지 97%의 bar는 `no_trade_band`로 인해 이전 비중 유지. CS amp는 rebalance 시점에만 영향을 줄 수 있고, rebalance 이후 60개 bar 동안 비중은 고정 → 60개 bar 중 1개만 CS amp 영향.

**이 구조 하에서는 어떤 amplification도 포트폴리오 레벨 성과를 실질적으로 바꿀 수 없음.**

### 개선 방향 (Pivot 필요)

1. **Stop Mu Amplification 접근** — 구조적 한계(3% rebalance bars)로 인해 효과 없음
2. **L1 CS Rank Score 자체 개선 필요** — L1 Alpha Ensemble에서 더 넓은 cross-sectional edge 분산을 갖도록 계산
3. **Sharpe Uplift Gate 검토** — `l2_min_sharpe_uplift=0.20`이 지나치게 높음. EW 대비 0.05~0.10 이상의 uplift은 현실적으로 달성 불가
4. **L* floor는 유효** — CAGR 19.5%를 더 올릴 방법: L* floor 1.5→2.0, RiskUtil 50.7%→67.6% (MDD 22.8% 예상, cap 30% 이내)
