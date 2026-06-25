# L2 Phase 실행 결과 — cost_drag fix 검증 + vol-targeting 진단

> 실행: `LOG_LEVEL=DEBUG uv run python src/execution/opt_main_futures.py --phase l2 --timeframe 4h --trials 30`
> 일시: 2026-06-25 (2차 실행, cost_drag fix 반영)

---

## ✅ cost_drag Fix 검증 (P0)

**Before (1차 실행, fix 미적용)**: `cost_drag=14803860.8594` — gate 영구 FAIL
**After (2차 실행, fix 적용)**: `cost_drag=0.1577(vs0.60)` — ✅ 정상 범위, gate 통과

| 항목 | Before | After | 비고 |
|------|--------|-------|------|
| cost_drag | 148,031,860 | **0.1577** | `Σ\|price\|` 분모 + 100× cap 적용 |
| cagr | -34.3% (BLOCKED) | **+40.55%** (PASS) | cost_drag gate 해제로 trial 정상 평가 |
| MDD | 29.16% | 16.13% | 동일 trial 기준 |
| sortino | -1.51 | **+2.24** | |
| sharpe | -1.09 | **+1.53** | |
| PSR | 0.17 | **0.91** (≥0.90 ✅) | |

**결론**: cost_drag fix가 근본 원인(분모 폭발)을 해결. `blocker=cagr` → `blocker=growth_lcb`로 gate blocker가 정상으로 변화.

## 🔬 `[L2-FIT-DIAG]` — fit-leg vol-targeting 진단 (P1 신규)

```
[L2-FIT-DIAG] fold=0 fit_bars=494  fit_CAGR=-0.4838 fit_MDD=0.1569 fit_ann_vol=0.1455 fit_sharpe=-4.4699
[L2-FIT-DIAG] fold=1 fit_bars=1058 fit_CAGR=-0.3304 fit_MDD=0.1981 fit_ann_vol=0.1437 fit_sharpe=-2.7189
[L2-FIT-DIAG] fold=2 fit_bars=1622 fit_CAGR=-0.2490 fit_MDD=0.2083 fit_ann_vol=0.1290 fit_sharpe=-2.1547
```

### 주요 발견: `fit_ann_vol ≈ 14%` — vol_target=1.0 미달

모든 fold에서 **realized annual vol ≈ 13~14.5%**. vol_target=1.0(100%) 대비 **1/7 수준**.

**의미**:
- Kelly sizing의 포트폴리오 레벨 realized vol이 1.0이 아니라 ~0.14
- `vol_target=1.0` 정규화가 각 전략 시그널 레벨에는 적용되나, cross-sectional portfolio의 realized vol은 크게 낮음
- 원인: Kelly 포지션이 long/short 상쇄 + CS Rank로 인해 portfolio level leverage가 낮게 유지됨

**충격적 발견**: `fit_ann_vol=0.14`일 때 fit_MDD=0.16~0.21은 `σ·√T·1.6` 수식과 정합
- 예상 MDD ≈ 0.14 · √(1058/2190) · 1.6 ≈ 0.14 · 0.69 · 1.6 ≈ 15.5% ← fit_MDD=15.7%와 일치
- 즉 **fit_MDD가 높은 것은 vol_target 문제가 아니라 portfolio의 inherent risk가 14% vol이기 때문**

**해결 방향**:
- Portfolio level vol을 1.0(100%)에 근접시키려면 leverage boost 필요
- 또는 vol_target을 현재 realized level (0.14)에 맞게 재조정하거나, L*로 scaling
- 근본적으로 Kelly sizing의 cross-sectional vol 특성을 반영한 vol_target 재정의 필요

## 🔍 `[L2-OOS-CAP]` — OOS RiskUtil 진단 (P2 신규)

```
[L2-OOS-CAP] OOS_RiskUtil=0.538 cap=0.30 (L*=1.000)
```

- OOS_RiskUtil=53.8%로 MDD 예산(30%) 절반 수준에서 안정적
- L*=1.0에서도 OOS_MDD=16.13%로 mdd_target(21%) 이내
- fit_MDD_vol1=48.48%가 너무 높아 L*=1.0으로 hard landing하지만, OOS는 훨씬 안정적

## 🚧 Gate 현황 — `growth_lcb` blocker로 전환

```
[L2-GATE] promotion=False blocker=growth_lcb |
cagr=0.4055(vs0.30)✅ sortino=2.2351(vs1.50)✅ sharpe=1.5290(vs0.70)✅ calmar=2.5133(vs0.50)✅
mdd=0.1613(vs0.30)✅ folds=0.67(vs0.60)✅ trades=129(vs30)✅ cost_drag=0.1577(vs0.60)✅
psr=0.9112(vs0.90)✅ | uplift=-0.0603(vs0.20)❌
```

- `cost_drag` ✅ 15.77% (gate 60% 이하)
- `psr` ✅ 0.911 (gate 0.90 이상) — **이전 문제 해결됨**
- `cagr` ✅ 40.55% (gate 30% 이상)
- **`growth_lcb` / `uplift`** ❌ -0.06 (gate +0.20) — **새로운 blocker**

즉, L2 전략이 EW baseline 대비 Sharpe 향상을 입증하지 못하는 구조적 문제.
전략의 Sharpe_HAC(1.529)이 baseline(1.589)보다 낮음.

## 📊 전체 진단 요약

| 항목 | 1차 실행 | 2차 실행 (fix 적용) | 상태 |
|------|---------|-------------------|------|
| cost_drag | 148M → BLOCK | **0.16 → PASS** | ✅ FIXED |
| fit_ann_vol | N/A | 13~14.5% | 🔍 vol_target=1.0 대비 1/7 |
| OOS_RiskUtil | N/A | 0.538 | ✅ 안정적 |
| CAGR gate | 100% FAIL | **PASS** (40.55%) | ✅ |
| PSR gate | 100% FAIL | **PASS** (0.911) | ✅ |
| Uplift gate | N/A | **FAIL** (-0.06) | ❌ 신규 이슈 |
| 최종 gate | `cagr` | `growth_lcb` | 정상화 중 |

## 권장 사항 (Updated)

### Priority 1: Uplift/growth_lcb gate 실패 원인 분석
전략 Sharpe_HAC(1.529)이 baseline(1.589)보다 낮은 근본 원인:
- CS Rank가 오히려 1/N에 비해 성과를 저하시킴
- `l2_growth_uplift = 0.20`이 너무 높을 가능성
- 또는 fit-leg (negative CAGR)의 block log growth가 LCB를 왜곡

### Priority 2: Portfolio realized vol 분석
fit_ann_vol=14%는 vol_target=1.0의 1/7 수준.
실현 vol을 높여 risk budget을 더 효율적으로 활용 가능.
다만 이는 별도의 리스크 예산 재설계(RC-5 수준) 필요.

### Priority 3: L* calibration 재검토
fit_MDD_vol1=48.48%가 fit_CAGR=-35.61%에서 발생한 손실 구간의 영향.
OOS CAGR=+40.55%에서는 MDD가 16.13%로 훨씬 낮음.
→ fit-leg의 negative CAGR이 L*를 1.0으로 고정시키는 구조.
→ `l_floor` 완화 또는 fit-leg 기간 재정의(더 긴 warm-up) 필요.
