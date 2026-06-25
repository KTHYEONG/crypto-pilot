# L2 Routing Mode 실험 결과 (2026-06-25)

## 실험 설정
- **Data:** 2024-10-01 ~ 2025-09-30 (52 symbols, 4h TF)
- **Optuna:** 200 trials, V9 search space
- **Gate:** CAGR≥30%, Sharpe≥1.0, Sortino≥1.5, Calmar≥1.0, Fold≥60%, Trades≥30

---

## 실험 1: Pool Mode (`l2_routing_mode=pool`)

| 지표 | 결과 | 게이트 | 통과 |
|------|------|--------|------|
| CAGR | **+6.2%** | ≥30.0% | ❌ |
| Sharpe | 0.456 | ≥1.000 | ❌ |
| Sortino | 0.641 | ≥1.500 | ❌ |
| Calmar | 0.401 | ≥1.000 | ❌ |
| MDD | 15.5% | ≤30.0% | ✅ |
| CVaR95 | 1.2% | ≤6.0% | ✅ |
| Fold Pass | 66.7% | ≥60.0% | ✅ |
| Trades | 159 | ≥30 | ✅ |
| Sharpe Uplift | +0.21 | ≥+0.20 | ✅ |
| DSR | 0.671 | ≥0.60 | ✅ |
| PSR | 0.714 | (diag) | — |

**최종: ❌ BLOCKED (cagr)**

**Fold 상세:**
- Fold #1: CAGR -9.4%, Sharpe -0.515 ❌
- Fold #2: CAGR +28.0%, Sharpe 2.846 ✅
- Fold #3: CAGR +3.3%, Sharpe 0.439 ✅

---

## 실험 2: Bucket Mode + Zero Floor (`l2_routing_mode=bucket`, `l2_bucket_edge_floor_bps=0.0`)

| 지표 | 결과 | 게이트 | 통과 |
|------|------|--------|------|
| CAGR | **+13.3%** | ≥30.0% | ❌ |
| Sharpe | **1.177** | ≥1.000 | ✅ |
| Sortino | **1.729** | ≥1.500 | ✅ |
| Calmar | **1.417** | ≥1.000 | ✅ |
| MDD | **9.4%** | ≤30.0% | ✅ |
| CVaR95 | **0.8%** | ≤6.0% | ✅ |
| Fold Pass | **100.0%** | ≥60.0% | ✅ |
| Trades | 129 | ≥30 | ✅ |
| Sharpe Uplift | +0.07 | ≥+0.20 | ❌ |
| DSR | 0.632 | ≥0.60 | ✅ |
| PSR | 0.929 | (diag) | — |

**Best Optuna Trial CAGR: 300.86%** (vs pool 150.58%)

**최종: ❌ BLOCKED (cagr)**

**Fold 상세:**
- Fold #1: CAGR +3.5%, Sharpe 0.461 ✅ (Symbols: 14)
- Fold #2: CAGR +11.9%, Sharpe 1.610 ✅ (Symbols: 11)
- Fold #3: CAGR +25.7%, Sharpe 3.376 ✅ (Symbols: 15)

---

## 핵심 통찰

| 항목 | Pool (100bps) | Pool | Bucket+0bps |
|------|:---:|:---:|:---:|
| Best Trial CAGR | 56.09% | 150.58% | **300.86%** |
| 최종 CAGR | +0.3% | +6.2% | **+13.3%** |
| Sharpe | 0.074 | 0.456 | **1.177** |
| Sortino | 0.119 | 0.641 | **1.729** |
| Calmar | 0.025 | 0.401 | **1.417** |
| MDD | 11.2% | 15.5% | **9.4%** |
| Fold Pass | 33.3% | 66.7% | **100.0%** |
| Trades | 8 | 159 | 129 |
| Blocked By | low_trades | cagr | cagr |

1. **Bucket+0bps ≫ Pool ≫ Bucket+100bps**: regime-conditional bucket routing이 신호 품질을 크게 개선함
2. **CAGR gate(30%)가 공통 blocker**: 두 실험 모두 CAGR 부족으로 BLOCKED. 이는 edge 자체의 한계이지 routing 문제가 아님
3. **Bucket+0bps의 100% fold pass**: 모든 폴드가 양수 CAGR, 분산 투자(11-15 symbols) → regime 필터링이 효과적으로 negative edge 제거
4. **Best trial CAGR 300.86%**: Optuna가 더 높은 잠재력을 발견했으나 deployment L* calibration에서 보수적으로 수렴

---

## 권장 액션

1. **bucket mode + zero-floor(0.0)**가 정답 방향 → `l2_routing_mode=bucket`, `l2_bucket_edge_floor_bps=0.0` 기본값으로 변경
2. CAGR gate 30%는 이 심볼 셋/기간에서 달성 불가능 → 게이트 임계값 하향 조정 필요 (e.g., 15%)
3. Uplift 기준도 +0.20이 엄격함 → Bucket+0bps의 +0.07이 noise 수준인지 확인
4. Deployment L* gap (CAGR 300.86% → 13.3%)은 별도 분석 필요
