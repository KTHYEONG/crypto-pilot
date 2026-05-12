# 2026-05-12: Optimizer Pipeline Bug Investigation & Fix Attempts

## 목표
`opt_main_futures.py` Phase A `complete=0 pass=0 best=10.0` 원인 진단 및
백테스팅 파이프라인 복리 자산 극대화 구조 개편.

---

## 근본 문제 진단 결과

### Bug #1 (FIXED): Zero-Trade Hack
**위치**: `optimizer.py` `_evaluate_awf_phase_d_aggregate`
```python
# Before (BUG): 80 trials 이하이면 가짜 점수 10.0 반환
if n_trials_eff <= 80:
    return float(10.0), {"pruned": False, "robust_val": float(-1e9)}
# After (FIXED): 항상 TrialPruned()
raise optuna.TrialPruned()
```
**효과**: Phase A best=10.0 → 실제 점수 산출 가능

### Bug #2 (FIXED): Calibrator Look-Ahead Leakage
**위치**: `optimizer.py` Platt scaling calibrator 학습 구간
- **Before**: `window_lo=first_awf_anchor` (OOS pool과 동일 구간 → 미래 참조)
- **After**: IS-only 구간 `(0, first_awf_anchor - embargo)` 사용, 최소 bars 부족 시 경고 후 fallback

### Bug #3 (FIXED): `_apply_ls_balance` 방향성 포트폴리오 zeroing
**위치**: `portfolio_constructor.py`
```python
# Before (BUG): 단방향 포지션 시 weights 전부 0
if short_m < 1e-12:
    return out * 0.0
# After (FIXED): 단방향 허용 (HMM이 방향성 결정)
if short_m < 1e-12 or long_m < 1e-12:
    return out
```
**효과**: CHOP 구간(61.3%)에서 발생하던 weight zeroing 해소

### Bug #4 (FIXED): AWF K-legs 과대 설정
**위치**: `config/opt_config.py`
- **Before**: `k=6` → 각 leg = 데이터의 5% → 학습 span 부족
- **After**: `k=4` → 각 leg = 7.5% → 충분한 학습 window 확보

### Bug #5 (FIXED): NSGA-II 목적함수 동일 상관
**위치**: `optimizer.py` `_evaluate_awf_phase_d_aggregate`
```python
# Before (BUG): (obj, -robust_val) — robust_val=-obj이므로 완전 상관
return (float(obj), float(-robust_val))
# After (FIXED): 독립적 두 목적함수
obj1 = -float(np.mean(leg_arr))  # Kelly 복리 성장
obj2 = -float(np.min(leg_arr))   # Worst leg tail 방어
```

### Bug #6 (FIXED): Objective 함수 Kelly 부정합
**위치**: `evaluator.py` `compute_awf_robust_objective_score`
```python
# Before: median + MAD (Kelly와 부정합)
# After: mean + semi_deviation (Kelly E[log(1+r)] 정합)
mu = float(np.mean(arr))
downside = arr[arr < mu]
semi_dev = float(np.sqrt(np.mean((downside - mu) ** 2)))
return float(mu - lambda_mad * semi_dev - dd_term)
```

### Bug #7 (ROOT CAUSE, FIXED): ATR=0 → 전체 거래 skip
**위치**: `optimizer.py` `_build_prebuilt_full_arrays`
```python
# Before (BUG): ATR 컬럼 없으면 zeros → backtest_target_weights_numba가
#              atr_prev <= 0.0 조건으로 모든 rebalance entry skip → n_tr=0
trimmed_sig["atr"] = np.zeros(len(raw_full))

# After (FIXED): compute_atr_numpy로 on-the-fly 계산
_atr_col = compute_atr_numpy(high, low, close, period=30)
trimmed_sig["atr"] = np.where(finite & >0, _atr_col, close * 0.01)
```
**추가**: `opt_main_futures.py`에서 ML merge 직후 data_maps 전체에 ATR injection

### Bug #8 (FIXED): NSGA-II에서 trial.report() 호출
**위치**: `optimizer.py` `_evaluate_awf_phase_d_aggregate`
- Optuna multi-objective study는 `trial.report()` 미지원 → NotImplementedError
- **Fix**: `len(trial.study.directions) == 1` 조건 체크로 guard

### Bug #9 (FIXED): Phase C ValueError crash
**위치**: `portfolio_constructor.py` `rolling_ledoit_wolf_cov`
- `REBALANCE_BARS=1` trial에서 returns_hist shape 오류 발생 → 전체 study crash
- **Fix**: `study_ml.optimize(..., catch=(ValueError,))` 추가

---

## 현재 상태 (2026-05-12 기준)

| 지표 | 값 | 진단 |
|------|------|------|
| Phase A complete | 80 | ✅ 정상 (이전 0에서 개선) |
| Phase A pass | 0 | ❌ 미해결 |
| Phase A best | nan | ⚠️ NSGA-II display 이슈 (기능적 무해) |
| Phase B complete | 120 | ✅ 정상 |
| Phase B pass | 0 | ❌ 미해결 |
| Phase C | ValueError crash | ❌ catch로 임시 처리 중 |

**pass=0 진단 중**: 
`awf_pos_frac` (4 legs 중 양수 비율) 기반 pseudo-PBO 게이트가
`pbo_max=0.45` → `awf_pos_frac >= 0.55` (3/4 legs 이상 양수) 필요.
실제 user_attrs 값 확인 필요 (DIAG 로그 추가됨).

---

## 미해결 이슈 및 다음 단계

### Priority 1: pass=0 근본 원인 확인
DIAG 로그를 통해 Phase A trial들의 실제 user_attrs 값 확인 필요:
- `awf_pos_frac` 분포 → PBO gate 통과 여부
- `avg_trades` 값 → 최소 거래 횟수 gate
- `gate1_dsr` 값 → DSR gate (min 0.40)
- `awf_worst_leg_log_tw` → worst leg gate (min -0.10)

**다음 실행**: `uv run python src/execution/opt_main_futures.py --ops-profile smoke --tf 4h 2>&1 | grep DIAG`

### Priority 2: Phase C ValueError 근본 수정
`rolling_ledoit_wolf_cov` 진입 전 `returns_hist.ndim == 2 and shape[0] >= 2` 검증
또는 `precompute_rebalance_weights` 에서 단일 심볼 edge case 처리

### Priority 3: pass=0 원인 수정
진단 결과에 따라:
- gate 임계값 완화 검토 (smoke 환경의 짧은 데이터 기간 감안)
- AWF leg 품질 이슈 → 데이터 기간 확장 또는 leg 수 재조정

### Priority 4: NSGA-II best=nan 디스플레이 수정
`_trial_progress_callback`에서 multi-objective study를 위한 별도 best 계산 로직
(TOPSIS 선택 trial의 obj1 값으로 표시)

---

## 아키텍처 개선 방향 (미구현)

### A. Coordinate Ascent → Joint NSGA-II
현재: Phase A(포트폴리오) → B(신호 beta) → C(실행) 순차 최적화
문제: 비시너지적, 국소 최적해 수렴 위험
방향: 전 파라미터를 NSGA-II 단일 joint study로 동시 최적화

### B. Walk-Forward 구조 강화
현재: k=4 legs, IS pool 70%, embargo 24 bars
방향: 각 leg OOS 구간을 더 길게 → leg당 최소 500 bars 보장

### C. 백테스팅 마찰 현실화
- funding rate 반영 강화
- 거래량 기반 slippage 모델 (고정 BPS → 유동적)

