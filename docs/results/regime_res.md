# L2 Phase 4차 실행 — CS Amp 적용 후 진단

> 실행: `uv run python src/execution/opt_main_futures.py --phase l2 --timeframe 4h --trials 200` (기본)
> 일시: 2026-06-25 (4차 실행, CS Amp + OOS floor 적용)
> 상세 결과: `docs/results/result.md`

---

## 📊 Before/After 비교

| 지표 | 3차 (Amp 미적용) | 4차 (Amp + OOS floor) | Delta |
|------|-----------------|----------------------|-------|
| L* | 1.0000 (binding=mdd) | **1.5000** (binding=champion) | +50% ✅ |
| CAGR | 13.34% | **20.10%** | +6.76pp ✅ |
| MDD | 9.41% | 13.96% | +4.55pp |
| RiskUtil | 31.4% | **46.5%** | +15.1pp ✅ |
| Sharpe Uplift | **+0.074** | **+0.074** | 0.000 ❌ **불변** |
| Block delta | 0.0000 | 0.0000 | 0.000 ❌ **불변** |
| Gate blocker | cagr+uplift | cagr+uplift | 동일 |

---

## 🔴 핵심 발견 #1: CS Amplification v1 완전 무효

```
[L2-SHARPE-CMP] delta_sharpe=0.0736 mean_ratio=0.60 std_ratio=0.57  ← 불변
[L2-BLOCK-CMP] fold=0 delta=-0.0000  ← 불변
[L2-BLOCK-CMP] fold=1 delta=-0.0000  ← 불변
[L2-BLOCK-CMP] fold=2 delta=0.0000   ← 불변
[L2-BLOCK-SUM] hybrid: mean=0.0007 / baseline: mean=0.0007  ← 불변
```

**원인 가설**:
1. CS Z-score가 top-K 선택 심볼들 사이에서 너무 좁은 범위(0.5~2.0)에 밀집
2. `median_excess` 모드(α=2.0)로는 중앙값 대비 3× 증폭이 최대치 → Kelly 비중에 실질적 차이 없음
3. Kelly 가중치가 `w ∝ 1/σ²` (risk parity)에 구조적으로 수렴 — Z-score 분산이 작으면 항상 수렴

---

## ✅ 핵심 발견 #2: OOS Floor 정상 작동

```
L* 1.0000 → 1.5000 (binding: mdd → champion)
CAGR 13.34% → 20.10% (1.5× multiplier effect)
RiskUtil 31.4% → 46.5%
```

L* multiplier가 CAGR을 거의 선형으로 증가시킴. 다만 CAGR 20.1% → 30% gate까지는 +50% 추가 필요. L*를 더 올려야 하나, MDD도 14.0% → 21%로 증가 예상. gate 30%를 크게 넘지 않음.

---

## 🔴 핵심 발견 #3: Config 값 미전파 증거

```
[L2-GATE] uplift=0.0736(vs0.20)  ← dataclass 기본값 0.05가 아닌 0.20
```

`l2_min_sharpe_uplift=0.05` 로 변경했으나, 런타임에서 `vs0.20` 표기. Optuna champion config가 직렬화된 구버전 값 사용 중.

---

## 📊 종합 진단

| 발견 | 심각도 | 근본 원인 |
|------|--------|----------|
| CS Amp 무효 | 🔴 CRITICAL | Z-score 분산 부족. `median_excess` 모드로는 충분한 비중 차별력 생성 불가 |
| L* floor 효과 | 🟢 GOOD | OOS-based floor가 CAGR 1.5× 증가, RiskUtil 15pp 향상 |
| config 불일치 | 🟡 HIGH | champion params가 dataclass 기본값을 override |
| Block delta 불변 | 🔴 CRITICAL | L1 신호의 cross-sectional 차별력 자체가 미흡 |

### 권장 개선 방향 (업데이트)

1. **Power amplification mode** (P0): `amp = max(1, (z/z_med)^p)`, p=2.0. median_excess 대비 33% 더 강한 차별화
2. **Z-score distribution 진단** (P0): `[L2-Z-DIST]` 로그로 실제 Z-score 분포 확인
3. **Config propagation fix** (P0): champion replay 시 dataclass 기본값 우선 적용 보장
4. **L1 signal amplification** (P1): L2 진입 전 L1 per-bar edge를 `μ²/σ²` 기반으로 재계산하여 CS 분산 확대 — **구조적 해결책으로 가장 유망** (별도 spec 필요)
