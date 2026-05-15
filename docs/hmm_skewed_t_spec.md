# Skewed-t HMM & Parallel Scan (GPU-Optimized) 구현 사양서

본 사양서는 기존 CPU 기반의 순차적 Student-t HMM을 대체하여, 코인 파생시장의 비대칭 꼬리 위험을 포착하고 RTX 4070 Ti(GPU)의 병렬 연산 성능을 100% 활용하기 위한 **Skewed-t HMM 기반 아키텍처 재설계**에 대한 구체적인 구현 방안을 정의합니다.

## 1. 아키텍처 개요 및 목표

- **핵심 목표 1 (모델링):** Skewed-t 분포(비대칭 스튜던트 T)를 적용하여 하방 폭락(Negative Skew)과 강제 청산에 대한 조기 탐지 능력 극대화 (`Crisis-Prec > 20%`).
- **핵심 목표 2 (속도 혁신):** `jax.lax.associative_scan`을 활용한 **Parallel Markov Smoothing**을 도입하여 1500 Iteration 학습을 수 초 이내로 단축.
- **핵심 목표 3 (최적화 안정성):** Softplus, Sigmoid 등 부드러운 함수를 이용한 Reparameterization Trick으로 JAX XLA 컴파일러 최적화 효율 극대화.

---

## 2. 수학적 모델 사양: Skewed-t Emission (방출 확률)

일반적인 Student-t 분포에 비대칭 파라미터 $\lambda$(Skewness)를 결합한 Azzalini Skewed-t 확률 밀도 함수를 방출 모델로 사용합니다.

### 2.1 PDF 수학적 정의
다변량(대각 공분산 가정을 위한 독립 차원) Skewed-t의 로그 확률 밀도는 다음과 같이 근사하여 계산합니다.
$$ \log f(x) = \log(2) + \log t_{\nu}(z) + \log T_{\nu+1}(\lambda z \sqrt{\frac{\nu+1}{\nu+z^2}}) $$
- $z = \frac{x - \mu}{\sigma}$ (표준화된 관측치)
- $t_{\nu}$: 자유도 $\nu$인 표준 Student-t의 PDF (기존 코드 재사용)
- $T_{\nu+1}$: 자유도 $\nu+1$인 표준 Student-t의 CDF (JAX 텐서 연산으로 근사 구현)
- $\lambda$: 비대칭성을 결정하는 파라미터 (음수면 하방 꼬리가 긺)

### 2.2 파라미터 텐서 (Reparameterization Trick 적용)
- `mu`: $(K, F)$ - 실수 전체.
- `log_sig`: $(K, F)$ - 분산. 부드러운 클리핑 적용 (`jnp.clip` 대신 부드러운 $\tanh$ 스케일링 권장하나 기존대로 $[-6, 3]$ 제한 사용).
- `nu_raw`: $(K,)$ - $nu = \text{softplus}(\nu_{raw}) + 2.1$. 최소 자유도 보장.
- **`lambda_raw` (New):** $(K, F)$ - 상태/피처별 비대칭성 파라미터. (예: BEAR 상태의 $f_3$(Downside Vol) 차원에 대해 $\lambda < 0$으로 유도되도록 초기화/Prior 설정).

---

## 3. GPU 병렬화 사양: Parallel Associative Scan

HMM의 순차적(Sequential) 연산 병목인 `_hmm_forward`를 병렬 연산 트리로 전환합니다. 이는 Viterbi 추론과 Forward 확률 계산 속도를 GPU에서 극단적으로 끌어올립니다.

### 3.1 Operator (이항 연산자) 정의
행렬 곱의 결합 법칙(Associative Property)을 이용하여, 두 구간의 결합 확률 행렬을 융합하는 Operator를 정의합니다. 로그 공간(Log-space)에서 연산하므로 Log-Sum-Exp 트릭을 사용합니다.

```python
@jax.vmap
def _combine_fn(a, b):
    # a, b는 각각 (K, K) 크기의 전환(Transition) + 방출(Emission) 결합 로그 확률 텐서
    # c_ij = logsumexp_k(a_ik + b_kj)
    return jax.scipy.special.logsumexp(a[:, :, None] + b[:, None, :], axis=1)
```

### 3.2 Associative Scan 적용 파이프라인
1. **Local Emission & Transition 계산:** 모든 시간 $t$에 대해 동시에 독립적으로 로그 방출 확률 `log_emit[t]`과 로그 전환 확률 `log_trans[t]` (TVTP 적용 시)를 계산. $\rightarrow$ GPU 코어 100% 병렬 계산.
2. **Initial Matrix 구성:** 각 $t$에 대해 `(K, K)` 크기의 초기 상태 결합 행렬을 생성.
3. **Parallel Scan:** `jax.lax.associative_scan(_combine_fn, initial_matrices)` 실행. $O(N)$ 순차 작업이 GPU 내에서 트리 구조로 분배되어 $O(\log N)$ 단계 만에 완료.
4. **Marginalization:** 최종 산출된 행렬에서 초기 확률(`log_init`)을 곱해 최종 `log_alpha` 도출.

---

## 4. 기존 코드베이스 수정 계획 (Implementation Plan)

### 4.1 대상 파일 및 클래스
- **수정 대상 파일:** `src/domain/futures/ml_pipeline/regime/student_t_hmm.py` (또는 신규 `skewed_t_hmm.py` 생성 후 덮어쓰기)
- **대상 클래스:** `StudentTMultivariateHMM` $\rightarrow$ `SkewedTMultivariateHMM`

### 4.2 주요 변경 컴포넌트

**1. `_skewed_t_log_pdf` 함수 신설:**
- 기존 `_student_t_log_pdf`를 호출한 뒤, 비대칭 요소(CDF 근사)를 덧셈 연산.
- JAX에 Student-t CDF가 내장되어 있지 않으므로, 정규분포 CDF(`jax.scipy.special.ndtr`) 기반 근사식이나 자체 수치 적분(T-distribution CDF approximation)을 JIT 컴파일 최적화하여 작성.

**2. `_hmm_forward_parallel` 구현 (GPU 핵심):**
- 기존 `jax.lax.scan` 기반의 `_hmm_forward`를 **`jax.lax.associative_scan`**으로 전면 교체.

**3. `_hmm_nll` 함수 변경:**
- 새로운 파라미터 `lambda_raw`에 대한 L2 정규화 Prior 추가 (과도한 Skewness로 인한 불안정 방지).
- `BEAR` 상태의 주요 피처(예: $f_3$, $f_5$)에 대해 음의 $\lambda$를 갖도록 유도하는 Semantic Prior 추가.

**4. GPU 환경 변수 해제:**
- 파일 최상단에 있는 `JAX_PLATFORMS=cpu` 강제 설정 코드를 제거 또는 조건부(GPU 가용 시 해제)로 변경하여 RTX 4070 Ti 인식을 허용. (`os.environ` 조작 부분 제거)

---

## 5. 단계별 검증 (Validation) 계획

1. **Unit Test (수학적 검증):**
   - Skewed-t PDF가 정상 적분(합 1)에 가까운지, $\lambda=0$일 때 기존 Student-t와 정확히 동일한 값(허용 오차 1e-6)을 반환하는지 테스트.
   - `associative_scan`의 Forward 확률 결과와 기존 순차 `scan`의 결과가 정확히 일치하는지 `jnp.allclose`로 검증.
2. **Performance Test (학습 속도):**
   - `JAX_PLATFORMS=gpu` 환경에서 3000개 시계열 × 1500 iter 기준, 기존 CPU 연산 속도와 GPU Parallel Scan의 속도(wall-time) 벤치마킹 비교. 목표는 최소 5x 이상 속도 향상.
3. **Metric Test (도메인 목표):**
   - `opt_main_futures.py`를 연동하여 `Tail-Capture` 및 `Crisis-Prec` 지표 변화 확인.

---
*위 사양서는 JAX의 함수형 프로그래밍 철학 및 GPU 텐서 병렬화 아키텍처에 완벽하게 부합하도록 설계되었습니다.*
