# Alpha 컴포넌트 목표 문서 (G-ALPHA, Factory V1 기준)

본 문서는 현재 구현된 `AlphaFactoryV1` 아키텍처를 기준으로, 4h 비-ML 알파 산출/게이트 규약을 정의한다.

## 1. 아키텍처 원칙 (실구현 정렬)
- **4h Non-ML 고정축**: 1차 알파 산출과 생존 판정은 `timeframe="4h"`에서만 수행한다.
- **Factor Sleeves 조합**: `trend/reversal/carry/flow/idio` 5개 sleeve score를 기반으로 raw alpha를 생성한다.
- **Regime Router 적용**: HMM posterior(`bull/bear/chop/crisis`)로 sleeve 가중치와 confidence/exposure를 동적으로 라우팅한다.
- **Cost-aware 보정**: turnover 힌트와 비용 민감도를 반영해 알파를 보정하고 과도한 회전을 억제한다.
- **Gate 분리 운용**: 통과 불가 조건은 hard gate로 즉시 탈락, 품질 가중/우선순위 조정은 soft gate로 처리한다.

## 2. 출력 계약 (alpha_panel contract)
- 필수 컬럼: `alpha_long_00`, `alpha_short_00`, `alpha_long`, `alpha_short`, `alpha_net`, `alpha_confidence`
- 값 범위:
  - `alpha_long`, `alpha_short`, `alpha_confidence`는 `[0, 1]`
  - `alpha_net`은 `[-1, 1]`
- 메타(`alpha_component_filter`) 필수 키:
  - `n_components`, `n_surviving`, `n_surviving_long`, `n_surviving_short`
  - `post_agg_selected_long_count`, `post_agg_selected_short_count`
  - `survived_long_cols`, `survived_short_cols`
  - `post_agg_selected_long_cols`, `post_agg_selected_short_cols`
  - `elite_zero_after_survival`

## 3. Gate 운영 정책 (Hard/Soft)
- **Hard Gate**: 4h 시간축 위반, 계약 컬럼 누락, 핵심 리스크 지표 하한 미달은 즉시 `FAIL`.
- **Soft Gate**: regime별 품질 변동, confidence 저하, 비용 민감도 상승은 감점/축소 반영.
- **Audit 규칙**: 로그는 hard/soft gate 근거를 분리 기록하며, OOS 기준 판정을 우선한다.

## 4. 최적화 진입 조건
- 4h Non-ML hard gate를 통과한 artifact가 최소 1개 이상일 때만 후속 최적화 단계(Phase C/D) 진입을 허용한다.
