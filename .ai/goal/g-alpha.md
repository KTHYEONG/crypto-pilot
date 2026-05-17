# Alpha 컴포넌트 평가 기준 및 목표 (G-ALPHA)

이 문서는 크립토 선물 시장에서 살아남을 수 있는 정예 알파를 선별하기 위한 정량적 목표와 평가 기준을 정의한다. 모든 알파 생성 및 개선 작업은 본 지표를 통과하는 것을 목표로 한다.

## 1. 핵심 철학 (v8.0.0 기준: PairLogit Optimization & Survival)
- **순위 최적화 대응**: CatBoost PairLogit 도입으로 인한 IS IC 인플레이션을 경계하고, 순수 OOS 성능만으로 실력을 증명한다.
- **예외 없는 엄격함**: 통계적 유의성(T-stat)이 높더라도 리스크 관리 지표(Tail, Short, Half-life)에서 타협하지 않는다.
- **비용 극복**: 높은 Taker 수수료와 슬리피지를 이길 수 있는 고마진 IC(Rank IC ≥ 0.015)를 지향한다.
- **방향 균형**: 롱(Long) 편향을 지양하고, 하락장에서도 작동하는 숏(Short) 예측력을 확보한다.

## 2. 정량적 평가 지표 및 합격 기준 (Pass/Fail)

### A. 통계 및 적합성 (Stats & Recency)
| 평가 지표 | 합격 기준 (Pass) | 실무적 의미 |
| :--- | :--- | :--- |
| **FDR q-value** | **< 0.10** | 다중 검정 보정을 통해 노이즈에 의한 우연한 성과 배제. |
| **DSR (Deflated Sharpe)** | **> 50.0%** | 데이터 마이닝 편향을 제거한 실질적 샤프 지수 신뢰도. |
| **OOS IC Floor** | **≥ 0.015** | **(Hard Floor)** 혼합(Blend) 점수를 배제하고 순수 OOS 구간 Rank IC 확인. |
| **IS-OOS Retention** | **≥ 50.0%** | (Decay < 50%) PairLogit의 과적합을 방지하기 위한 최소 보존율. |
| **ICIR (OOS)** | **≥ 0.5** | OOS 구간 내에서 예측력의 꾸준함(Consistency) 확인. |

### B. 리스크 및 실무 (Risk & Operational)
| 평가 지표 | 합격 기준 (Pass) | 실무적 의미 |
| :--- | :--- | :--- |
| **Tail IC (Decile)** | **≥ 0.005** | 극단 변동성 구간에서 역방향 베팅 방지 및 최소한의 수익 기여. |
| **Short-side IC** | **≥ 0.015** | 하락장(Bear Regime)에서도 작동하는 실질적인 숏 예측력 확보. |
| **Half-life** | **≥ 3.0 bars** | **(No Exception)** 잦은 포지션 교체로 인한 슬리피지 방어 (4h 기준 12시간). |
| **Symbol Balance** | **≤ 3.0** | 특정 종목에 편중되지 않은 유니버스 전체에 대한 일반화 성능. |

## 3. 로그 출력 및 감사 가이드라인
- 알파 요약 로그는 위 지표를 기반으로 `[PASS]`, `[FAIL]`, `[REJECTED]` 사유를 명시한다.
- **Short-side IC** 또는 **Tail IC**가 허들 미만인 경우, 전체 IC가 아무리 높더라도 부적격 판정한다.
- **Fast-track Bypass 금지**: T-stat이 높다는 이유로 Half-life나 리스크 게이트를 건너뛰는 것을 금지한다.
- **최적화 진입 전제**: 본 문서의 모든 게이트를 통과한 alpha artifact가 1개 이상 존재해야만 Optuna 최적화(Phase C/D) 파이프라인으로의 진입을 허용한다.
