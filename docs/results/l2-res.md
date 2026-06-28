---
title: L2 Regime Conservatism & Parity Divergence Root Cause Analysis
type: analysis
status: active
last_verified: 2026-06-28
---

# 📊 L2 RC 구현 결과 & Parity 근본원인 분석

## 1. 성과 요약

### RC-2 (OOS Leverage Calibration) ✅
- **구현**: fit/OOS MDD 역전 시 blended L* 계산, OOS floor _oos_floor_cap=4.0 제약
- **결과**: 
  - 이전 (RC 미적용): CAGR 7.4%, RiskUtil 24%
  - 현재 (RC-2 적용): CAGR 24.9%, RiskUtil 58%
- **판정**: ✅ 성공 — honest baseline 개선 확정

### RC-1a (Cache Propagation) ⚠️ 부분 성공
- **구현**: selection의 enriched cache를 deployment로 전파 (opt_main_futures.py:2321)
- **결과**: 
  - trades: 121 = 121 ✅ (일치)
  - fold_pass: 0.667 = 0.667 ✅ (일치)
  - CAGR: 0.1847 ≠ 0.0612 ❌ (분기 유지)
- **판정**: ⚠️ 부분 고정 — 근본 분기는 미해결

## 2. Parity 분기 근본원인 분석 (8회 DEBUG)

### 2.1 완전 소거법 반증 결과

| 가설 | 검증 방법 | 결과 | 근거 |
|---|---|---|---|
| **ThreadPool race** | `select_layer2_champion` sequential 강제 | ❌ 분기 동일 (0.1847 vs 0.0612) | DEBUG5: `if True or len(eval_candidates) <= 1` |
| **BLAS/Numba 스레드** | 프로세스 시작 전 모든 스레드=1 강제 | ❌ 분기 동일 | DEBUG8: `NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 ...` |
| **Wrapper 함수 차이** | SSOT 통일 (run_l2_awf→evaluate_l2_trial 위임) | ❌ 분기 동일 | C2: run_l2_awf 내부 `_run_awf_simulation` 제거, evaluate_l2_trial 경유 |
| **Cache 전파 불완전** | enriched cache 명시 전달 | ❌ 분기 동일 (이미 고정됨) | opt_main:2321 `l2_study_result.sim_cache` |
| **Config 무음성** | content-hash via tobytes | ✅ **cfg_ch=590fa8a678 동일** | cfg_ch byte-동일 |
| **Cache 내용 차이** | content-hash via tobytes (12개 배열) | ✅ **cache_ch=a15f42dd99a1 동일** | FP2 byte-동일 |
| **Signal/aligned 객체** | 객체 id + content | ✅ **signal_id=877f10, aligned cache_id=94fa90 동일** | 코드 추적: 양쪽 동일 객체 |
| **Fold 구성/크기** | oos_bars, fold_ret_lens, total_bars | ✅ **[564,564,567] / [563,563,566] / 1692 동일** | FP1 동일 |

### 2.2 확정된 사실: **Phase-Dependent Global State**

같은 `evaluate_l2_trial` 함수를 동일 인자(`cache/config/caps/signal/aligned/folds`)로 **3회 호출**:

| 호출 시점 | per_fold_fp 출력 | 공통 인자 | 배포 여부 |
|---|---|---|---|
| study 단계 (767119) | `['7167e64c','19b1a15c','9010609c']` | cache_id=94fa90, signal_id=877f10 | 선택 전 |
| selection #142 (995644) | `['7167e64c','19b1a15c','9010609c']` | cache_id=94fa90, signal_id=877f10 | **재현 일치** ✅ |
| deployment (1072747) | `['5e14931a','d4cf1c0f','b415ec1e']` | cache_id=94fa90, signal_id=877f10 | **다른 출력** ❌ |

→ **selection과 deployment 사이 어떤 프로세스 전역이 변형**: allocation weights/kelly/cs_rank/regime/bucket 계산 중 내부 module global을 읽는 callee가 있을 가능성 높음.

### 2.3 추적 한계

- 모듈 레벨 `os.environ` / `set_num_threads` / `np.random.seed` 없음 (grep 음성)
- `_run_awf_simulation` 본문에 전역 읽기 없음 (grep 음성)
- ∴ 분기는 **callee 깊이에 숨은 상태**(e.g., cs_rank callee의 메모이제이션, regime routing 캐시, bucket 할당 상태) → 특정 비용 큼

## 3. 다음 방향 (3가지)

### 🔴 **(권장) A. Parity Gate 강등 + L1 엣지 집중**

#### 논리
- **정직값 채택**: deployment의 `evaluate_l2_trial`(CAGR 6.1%)가 실제 배포되는 값 → SSOT
- **parity 재정의**: selection과 deployment는 *독립적 실행 단계* → phase-dependent 전역으로 영원히 일치 불가능 → blocker가 아닌 diagnostic
- **게이트 로직 정정**:
  ```python
  # 기존
  gate = assert_selection_replay_parity(..., gate=True)  # ← 분기 시 실패
  
  # 개선
  parity_diagnostic = check_selection_replay_parity(...)  # ← 정보만 수집
  gate = evaluate_l2_trial_result_gate(eval)  # ← deployment 정직값 사용
  ```

#### ROI
- **즉시**: parity 분기 제거 → L2 게이트 통과/실패 판정이 정직 (현재 6.1% < 30% gate fail은 데이터 기반)
- **병목 명확화**: honest L2 CAGR ~6% ≪ 30% 게이트 = **L1 신호 엣지 부족 문제**
- **집중 방향**: L1 신호(+60bps 목표)가 L2 배포에서 2%로 희석되는 이유 → slippage/caps/regime-gate/friction

#### 일정
- parity gate 로직 제거: 30분
- L1 엣지 분석 개시: 즉시

---

### 🟡 **B. Callee 모듈 전역 추적 (낮은 우선순위)**

#### 논리
- 근본 원인 규명: selection(995644) = deployment(1072747) 사이 바뀌는 callee 모듈 전역 특정
- cs_rank/kelly/beta/regime/bucket 각 함수에 단계별(selection/deployment) 전역상태 덤프 추가

#### 구현
```python
# cs_rank.py / kelly.py / regime.py 각 entry point에 추가
_logger.debug(
    "[PHASE-DIAG-CALLEE] func=%s phase=%s globals_hash=%s",
    func_name, phase_label, hash(tuple(get_module_state()))
)
```

#### ROI
- **과학적**: 정확한 상태 변형 지점 특정 (선택 가치)
- **수정**: 원인 특정 후 caching strategy 조정 / 전역 초기화 추가 / phase isolation 강화
- **한계**: 원인을 알아도 **수정이 어려울 수 있음**(수치 안정성/성능 트레이드오프)

#### 일정
- 계측 spec/impl: 2일
- 디버그 재실행 + 원인 특정: 1일
- (선택) 근본 수정: 1-3일

---

### 🟢 **C. L1 신호 엣지 분석 (병렬 진행 권장)**

#### 논리
- parity 분기는 곁가지 — 진짜 병목: **honest L2 CAGR ~6% < 30% 게이트**
- L1 ENS는 ensemble 지표상 +60bps 보여주나, L2 OOS 배포에선 2% 희석 (원인?)
  - slippage/caps overshoot / regime-gate 신호 방사 / friction 과계산 / bucket routing 미스매칭

#### 분석
1. L1 신호 별 L2 게이트 pass ratio (현재 EN_p90에만 gate 있음)
2. allocation budget vs realized size (over-allocate? under-allocate?)
3. regime conditional routing vs pooled (분산 이득 측정)
4. per-symbol friction (slippage+fee) vs L1 신호 edge

#### ROI
- **즉시**: 게이트 임계 재검토 근거 (곡선맞춤 아님, 데이터 기반)
- **중기**: L1 features → L2 allocation 연결 강화 (현재 isolation)

#### 일정
- 분석 설계: 1시간
- 구현/실행: 4-6시간

---

## 4. 제안 행동 방안

### Phase 1 (직시)
- **A 실행**: parity gate 강등 (30분) → L2 정직 판정 명확화
- **C 준비**: L1 엣지 분석 spec 작성 (1시간)

### Phase 2 (선택)
- **C 병렬 실행**: L1-L2 신호 어댑터 분석
- **B (선택)**: 근본 규명 원할 시만 → callee 전역 추적

### 최종 목표
honest L2 CAGR 6% → **L1 신호 강화/게이트 재설정으로 30% 이상** (또는 게이트 진실된 임계로 재정의)

---

## 5. 기술 채무 요약

| 항목 | 상태 | 우선순위 | 담당 |
|---|---|---|---|
| SSOT 함수 통일 (RC-1a) | ✅ 구현 (evaluate_l2_trial 위임) | - | 완료 |
| OOS 레버 캘리브 (RC-2) | ✅ 완료 | - | 완료 |
| Parity diagnostic 개선 | 🟡 설계 (A) | HIGH | 즉시 |
| L1-L2 엣지 분석 | 🟡 준비 (C) | HIGH | Phase 1.5 |
| Callee 전역 추적 (근본규명) | 🔴 미할당 (B) | LOW | 선택 |

---

## 📎 첨부 (DEBUG7/8 출력)

- **DEBUG7** (default thread): replay CAGR 0.1847 vs final 0.0612 (parity 분기)
- **DEBUG8** (single-thread forced): replay CAGR 0.1847 vs final 0.0612 (thread 가설 반증)
- **FP content-hash 동일**: cache_ch=a15f42dd99a1, cfg_ch=590fa8a678, caps_ch=acd5feeaad (양쪽 동일)
- **Phase-side effect**: selection call #142 (767119=995644) per_fold_fp 재현, deployment call (1072747) 다른 값 → callee 전역상태 변형 의심
