---
title: ML Alpha 개선 로드맵 (Phase 1-5)
domain: futures-alpha
type: guide
status: active
priority: high
ai_read_policy: when_related
last_verified: 2026-05-30
---

# ML Alpha 개선 로드맵: Phase 1-5

> 현 알파 상태: `ALPHA_PASS=FALSE` (G2 경제거래성 FAIL).
> 개선은 순차적이며, **P1(rank-sizing)이 단일 최고 레버리지**.

---

## Phase 0: 게이트 정정 ✅ 완료
- N_eff emit-floor (미실현 분산 크레딧 제거)
- G2/G3 병합 (진단→하드 게이트 승격)
- clip_preservation_ratio 방향 정정 (post/pre)
- DSR n_trials 정직화 (n_trials=folds×3)
- t_stat 임계 2.0→3.0 상향

**결과:** `ALPHA_PASS: FALSE` (정직한 판정), 344bps 손실 노출

---

## Phase 1: Rank-Sizing + Soft-Hurdle ✅ 구현 완료, 효과 측정 대기

**구현:**
- 경질 `max(EV-hurdle,0)` 제거
- EV(return-fraction) 단위 soft-hurdle → rank-weight(-1,+1) → soft-hurdle gate 순차
- Params: `RANK_WEIGHT_K`(default 3.0), `SOFT_HURDLE_STEEPNESS`(default 5.0)

**목표:** `clip_preservation_ratio ≥ 0.5`, `basket_net_bps > 0`

**진입 조건:** 현재 P1 코드 기본 동작 → Optuna trial 자동 탐색

---

## Phase 2: Beta-Neutral 실행 오버레이

**논리:** 잔차-타깃 학습이 베타 헤지 가정. 실거래 북은 BTC 팩터 노출.

**구현:**
- 포트 수준 net-beta 측정 (회귀)
- BTC-perp 또는 지수 헤지 포지션
- 실행: 전략 시그널 생성 직전

**목표:** resid_ic 엣지 → PnL 전환, G2a gap>0 달성

---

## Phase 3: Idiosyncratic 피처 + Beta-Neutral 신호

**논리:** 현재 피처 전부 음의 IC (베타-적재). Resid-neutral 신호로 대체.

**구현:**
1. 학습 타깃: beta-residualized로 직접 지정 (P2와 일관)
2. 피처: BTC 대비 lead-lag, sector-relative, funding-basis 이격
3. 제거: 베타-프록시 (ret_*, vol_*)

**목표:** pre-clip resid_ic 상향 (현 0.0141 → 0.025+)

---

## Phase 4: 정직한 모델 선택 + 강건성

**구현:**
- DSR n_trials 주입 ✅ (완료)
- Fold 부호 안정성 검증 추가 (G1c)
- Purged/embargoed walk-forward 확인

**목표:** 과적합 방지, 일반화 gap 축소

---

## Phase 5: 비용-호라이즌 공동최적화

**논리:** 현 24bps 비용 vs 11bps p50 EV → 모든 bar가 비용 미달. SWEEP 전 horizon 음수.

**구현:**
- 보유 호라이즌 연장 (비용 상각)
- 고확신 tail(상위 N%) 거래만 선택
- P1-P3 이후 재측정

**목표:** SWEEP ≥1 호라이즌 통과 (현 0/3)

---

## 의사결정 흐름도

```
P0 (게이트 정정) ✅
    ↓
P1 (rank-sizing) — Optuna trial 탐색 중
    ├─ presv ≥ 0.5? → yes: P2 진입
    └─ no: P3 신호 강화 우선
    ↓
P2 (beta-hedge) — P1 효과 측정 후
    ↓
P3 (idiosyncratic) — 병렬 가능
    ↓
P4 (강건성) — 통합
    ↓
P5 (호라이즌) — 최종 거래 최적화
```

---

## 성공 지표

| Phase | G2a gap | presv | SWEEP | basket_net | Status |
|-------|---------|-------|-------|-----------|--------|
| **P0** | -415.7bps | -0.21 | 0/3 | -36.9bps | ✅ 정직화 |
| **P1** | -200bps? | 0.3-0.5 | 1+? | 10-20bps? | 진행 중 |
| **P2** | >0 | ≥0.5 | 2+ | >50bps | 설계 |
| **P3** | +200bps? | 0.7+ | 3/3 | 100bps+ | 목표 |

---

## 구현 순서

1. **P1 효과 측정:** Optuna trial 5-10회 실행, presv/basket 추이 확인
2. **P2 디자인:** Beta-neutral 실행 오버레이 상세 설계 (P1 효과 기반)
3. **P3 신호:** 피처 재설계 + 베타-neutral 학습
4. **P4 강건성:** fold 안정성, purge 검증
5. **P5 호라이즌:** 최종 거래 최적화

---

**참고:** P1부터 각 phase는 이전 단계의 성공 지표를 기반으로 진입. 병렬 작업 가능하나 의존성 확인 필수.
