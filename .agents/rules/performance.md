---
trigger: glob
priority: 10
---

# HPC & Memory Management Directives (WSL-Optimized Quant)

본 문서는 WSL 환경의 하드웨어 제약 하에서 대용량 퀀트 데이터 연산의 속도와 메모리를 극대화하기 위한 물리적 성능 가이드라인입니다.

---

## 1. 물리적 자원 예산 & 환경 제약 (WSL Budget Constraints)

- **가용 CPU**: 8 Processors
  - **지침**: 병렬 연산(`ProcessPoolExecutor`, `multiprocessing`) 사용 시 `max_workers` 상한선은 **4~6**으로 제한합니다. 8개 전체 코어 점유 시 컨텍스트 스위칭 오버헤드 및 WSL 네트워크/디바이스 통신 중단이 발생할 수 있습니다.
- **가용 RAM**: 18GB Physical Memory (20GB Swap)
  - **지침**: Swap 영역 진입 시 디스크 I/O 병목으로 백테스팅 속도가 10배 이상 저하됩니다. 실행 중인 프로세스의 최대 상주 메모리(RSS)는 **12GB**를 초과할 수 없습니다.

---

## 2. 메모리 최적화 지침 (Memory Safety Guardrails)

- ** float32 스토리지 다운캐스팅**
  - 원시 피처 데이터, 대용량 보조 지표 패널 등 연산 중간 과정에 저장되는 행렬은 `float64` 대신 **`float32`**로 캐스팅하여 메모리 점유율을 50% 절감합니다.
  - 단, 포트폴리오 최적화(공분산 행렬 역연산), 누적 수익률 복리 연산은 수치적 안정성을 위해 `float64`를 유지합니다.
- **불필요한 Panel Deepcopy 금지**
  - Pandas `DataFrame` 또는 NumPy `ndarray`를 수정할 때 불필요한 `.copy(deep=True)` 사용을 차단하고, inplace 연산 또는 view 참조(`slice`)를 적극 활용합니다.
- **수동 Garbage Collection (GC) 제어**
  - 타임프레임(TF) 전환 단계, 또는 에포크(Fold) 연산 완료 즉시 `del` 키워드로 미사용 대형 인스턴스를 소멸시키고 `gc.collect()`를 호출하여 WSL 커널에 메모리를 즉시 반환합니다.

---

## 3. 소요시간 및 HPC 연산 최적화 지침 (HPC Computation)

- **Pandas Loop (.iloc, .iterrows) 절대 금지**
  - Pandas DataFrame의 로우 단위 순회(loop)는 절대 금지합니다.
  - 모든 수치 계산은 NumPy vectorized operation 또는 Polars expression을 통해 C-level 벡터 연산으로 수행합니다.
- **Numba JIT 및 Array Contiguity 확보**
  - DataFrame 또는 복잡한 파이썬 객체는 Numba 함수에 전달하기 전 원시 NumPy 배열(`ndarray`)로 완전히 언패킹하여 전달해야 합니다.
  - Numba JIT 함수(`@njit`)로 넘기는 모든 NumPy 배열은 메모리 연속성(`C_CONTIGUOUS`)을 확보해야 합니다. 슬라이싱이나 Resample 등으로 쪼개진 배열은 반드시 **`np.ascontiguousarray(arr)`**를 거쳐 Numba에 전달하여, 메모리 복사 오버헤드와 캐시 미스(Cache Miss)를 예방합니다.
- **병렬화 의사결정 수칙 (Optimized Parallelism)**
  - 대용량 Grid Search 등 연산량이 매우 크고 상호 독립적인 무거운 루프에만 `ProcessPoolExecutor`나 `parallel=True`를 사용합니다. 프로세스 생성 및 데이터 직렬화/역직렬화(IPC) 오버헤드가 연산 속도보다 큰 경량 연산에는 멀티프로세싱을 금지합니다. (WSL 코어 가드 준수)
- **조기 탈락 (Early-Exit) 아키텍처**
  - L0 Cheap Gate 등 저비용 필터링 단계에서 탈락한 가설은 무거운 후속 연산(Bootstrap, Triple-barrier 등)에 집입하기 전 즉시 제외(`early-exit`)시킵니다.

---

## 4. 스킬 연동 검증 규칙 (TDD & Check Integration)

- **성능 퇴보(Performance Regression) 감시**
  - 기능 개발을 완료하고 `lean_check.py`를 실행할 때, 전체 소요 시간이 이전 벤치마크 대비 **15% 이상 증가**하거나 RSS 메모리가 **12GB 가드를 터치**하는 경우, AI는 반드시 코드 작성을 멈추고 복잡도(Big-O) 분석을 통해 병목을 리팩토링해야 합니다.
