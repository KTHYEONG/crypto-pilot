# ADR: CPCV Pipeline and Portfolio Selection Optimization

## Context
- `phase=signal` 실행 시 15개 CPCV Fold 순차 훈련 연산과 중복되는 시계열 기술적 지표/리스크 오버레이/레지임 컨텍스트 연산, 그리고 shadow selection 루프 내부의 판다스 복사/Datetime 파싱 오버헤드로 인해 약 29분(1750초)의 극심한 대기 시간 병목이 발생함.

## Decisions & Rationale
- **글로벌 캐싱 및 사전 계산**: `AlignedMarketData` 시계열 지표 및 마켓 레지임/리스크 오버레이 컨텍스트 계산 결과를 모듈 수준 `_ALIGNED_FEATURE_CACHE`에 최초 1회만 캐싱하고, shadow selection 루프 전 `pd.to_datetime` 및 컴포넌트 프레임을 사전 계산하여 중복 연산을 원천 차단함.
- **병렬 처리**: `ProcessPoolExecutor` (fork context)를 활용하여 15개 CPCV Fold의 모델 피팅과 예측 과정을 독립된 멀티프로세스로 병렬 실행함.
- **Numba `@njit` 적용**: 순수 파이썬 루프로 인해 병목이 심했던 가중치 연산(`_event_uniqueness_weights`, `_block_bootstrap_tstat` 블록 부트스트랩) 및 감도 조사 3중 루프(`compute_selection_sensitivity`)를 컴파일 가속화하여 밀리초 단위로 단축함.
- **결과**: 전체 소요 시간이 **63.64초**(약 96% 단축)로 획기적으로 개선됨.
