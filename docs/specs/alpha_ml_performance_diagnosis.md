# Spec: Futures Alpha ML Performance Diagnosis & Optimization (docs/specs/alpha_ml_performance_diagnosis.md)

## 1. 목적 (Purpose)
`src/execution/opt_main_futures.py`를 `--mode alpha`로 실행했을 때 나타난 **ALPHA_PASS: FALSE** 판정과 **230.58초의 과도한 실행 시간** 문제를 정밀 진단하고, 이에 대한 최적화 및 튜닝 방안을 수립합니다.

---

## 2. 현상 분석 및 원인 진단 (Root Cause Analysis)

### 2.1. 왜 ALPHA_PASS: FALSE 판정이 나왔는가?
실행 로그에 따르면, LGBMRanker 자체의 OOS 성능인 `[OOS-RANKIC]`은 매우 우수합니다:
- **`OOS-RANKIC`:** `ic=0.0731`, `t=9.71` (문서상의 최소 기준인 `ic >= 0.03`, `t >= 3.0`을 쉽게 초과)

그러나 최종 슬리브 포지션(C3-EXEC)을 평가하는 `[ALPHA SCOREBOARD]`에서 최종 탈락했습니다:
- **`NET_IC`:** `0.0027` (기준치 `0.03` 대비 크게 미달)
- **`T-STAT`:** `1.06` (기준치 `2.0` 대비 미달)
- **`BRDTH`:** `0.95` (기준치 `3.0` 대비 미달)
- **`BE_IC`:** `0.0508` (비용 장벽 대비 `gap = -481.3 bps`로 대폭 실패)

**핵심 실패 원인:**
1. **레짐 게이트(Regime Gate)에 의한 신호 희소성:**
   - 전체 6,462개 바 중 Bear 레짐(2,905 bars), Chop 레짐(1,096 bars)이 60% 이상을 차지합니다.
   - 현재 설정 상 Bear/Chop 레짐에서는 포지션 노출도가 `0.0` 또는 크게 제한되어, 수많은 구간의 익스포저가 `0`이 되면서 전체 평균 `NET_IC`가 극도로 희석되었습니다.
2. **과도하게 엄격한 EV 허들 게이트 (Strict EV Hurdle Gate):**
   - 횡단면 랭킹 상위 K개(4개) 중 보수적 EV가 `EV_HURDLE_BPS(10.0bps)`를 상회하는 자산만 진입하도록 강제하고 있습니다.
   - 이로 인해 실제 진입하는 포지션의 수가 극도로 한정되어 유효 독립 베팅 수(`Effective Breadth = 0.95`, 평균 1개 자산 미만)가 급감하고, 결과적으로 브레이크이븐 IC 기준선(`BE_IC = 0.0508`)이 매우 높게 치솟아 패스를 실패하게 만듭니다.
3. **타깃 평가 불일치 (Beta-Residualization Mismatch):**
   - 모델은 raw log return의 횡단면 순위(`forward_gross_rank`)를 학습하였으나, 스코어보드 평가 시에는 1차 벤치마크 시장 성분을 제거한 베타 잔차 수익률(`real_resid = real - beta * market_fwd`)을 대상으로 `NET_IC`를 측정하기 때문에 성능 저하가 발생합니다.

### 2.2. 왜 230.58초나 걸렸는가?
1. **데이터 동기화 오버헤드 (Data Sync Overhead):**
   - `🔄 SYNC: processing 123 symbols (Parallel)... | ⏱️ 47.84s`
   - 이미 과거 데이터가 모두 로컬에 저장되어 있음에도 불구하고, 123개 심볼 전체에 대해 거래소 히스토리컬 싱크 검사를 매번 수행하여 약 **48초**의 시간이 무의미하게 소요되었습니다.
2. **데이터 로딩 및 준비 오버헤드 (Data Audit & Processing):**
   - `<< DATA: 36.29s (ok=96)`
   - 96개 심볼에 대해 3.5개년 분량의 데이터 클렌징 및 적격성 감사를 매번 수행하는 데 약 **36초**가 소요되었습니다.
3. **피처 엔지니어링 및 앙상블 피팅 (Feature Calculation & Fit):**
   - `[ML-PIPE-PROF] elapsed=98.30s`
   - 59개 다차원 피처 연산 및 2개 Fold의 3개 시드 앙상블(총 6회 피팅)을 수행하는 데 약 **98초**가 소요되었습니다.

---

## 3. 해결 및 개선 방안 (Actionable Solutions)

### 3.1. 시간 단축 (Performance Speed-up)
- **명령어 아규먼트 최적화:**
  - `--skip-data-sync` 플래그를 추가하여 이미 최신화된 로컬 데이터베이스를 바로 사용하게 만듭니다. (실행 시간 **48초 단축**)
  - `--skip-universe` 플래그를 결합하여 유니버스 타임라인 계산 단계를 생략합니다.

### 3.2. ALPHA_PASS 튜닝 및 정합성 개선 (Alpha Pass Optimization)
- **`EV_HURDLE_BPS` 조정:**
  - 10.0bps의 보수적인 진입 장벽을 하향 튜닝(예: 5.0bps ~ 7.0bps)하거나, 횡단면 앙상블 시드의 융합 가중치(`ev_tail_blend_weight`)를 조절하여 유효 포지션 다각화(`Effective Breadth >= 3.0`)를 복구하고 브레이크이븐 IC 장벽을 하향 안정화시킵니다.
- **`--mode strategy` 또는 `full`을 통한 Hyperparameter Search 연동:**
  - `--mode alpha`는 최적화(Optuna Search) 없이 단일 기본 구성(`ml_lambdamart_v1`)으로만 테스트하므로 `--trials 100`이 무시됩니다.
  - 실제 하이퍼파라미터 튜닝을 반영하려면 `--mode strategy` 또는 `--mode full`을 실행해야 합니다.

---

## 4. Surgical Plan (Surgical Plan for speed-up run)
실행 시 아래와 같이 동기화를 스킵하는 명령어를 제안합니다:
```bash
PYTHONPATH=. uv run python src/execution/opt_main_futures.py \
    --mode alpha \
    --strategy ml_lambdamart_v1 \
    --skip-data-sync
```
