# Optuna 전략 통합 및 복리 엔진 평가 기준 (G-OPTUNA)

이 문서는 개별 컴포넌트(Alpha, HMM)가 결합된 최종 포트폴리오 전략이 '24시간 무인 자동매매' 환경에서 자산을 안전하게 불릴 수 있는지 판정하는 최종 통합 가이드라인을 정의한다. 모든 최적화 및 전략 승격 작업은 본 지표를 통과해야 한다.

> **데이터 윈도우 전제**: 최적화는 **trailing 24개월** 윈도우 기준(~4,380 4h봉). 분기(~540봉) 윈도우에서는 fold당 표본이 수십 봉 수준으로 Calmar·Sortino·CAGR 추정이 통계적으로 무의미함.

## 1. 핵심 철학 (v4.0.0 기준: Crypto-Native Mechanical Compounder)
- **무인 운용 (Mechanical 24/7)**: 인간의 심리적 개입 없이 수학적 우위만으로 365일 작동하는 전략을 지향한다.
- **수학적 생존 (Mathematical Survival)**: 복리 회복이 불가능한 '파산 지점'에 도달하지 않도록 낙폭(MDD)과 레버리지·청산 리스크를 엄격히 통제한다.
- **비용 효율 (Friction-Aware)**: 수수료·슬리피지·**Funding Rate(perp 선물 고유 비용)**를 압도하는 edge를 확보한다. **EV/Cost Ratio가 최우선 지표**이다.
- **크립토 적합 기준**: 주식 stat-arb 지표(trade당 raw expectancy, ICIR ≥ 2.0)는 빈도·universe 규모 의존적이므로 사용하지 않는다. Regime 의존적인 절대수익(CAGR)보다 risk-adjusted 효율이 hard gate의 중심이다.

## 2. 통합 평가지표 및 하드 게이트 (Pass/Fail)

### A. 마찰 극복 및 엣지 (Friction & Edge) — 최우선 게이트
| 평가 지표 | 합격 기준 (Pass) | 실무적 의미 |
| :--- | :--- | :--- |
| **EV/Cost Ratio** | **≥ 3.0** | **최우선.** 순손익 ÷ 총비용(수수료+슬리피지+funding). "엣지가 friction을 3배 이상 압도" |
| **Funding Drag** | **≤ 25% of gross PnL** | Perp 선물 고유 보유비용 비율. Long 편향 전략의 근본 열화 방지 |
| *Net Expectancy/trade* | *> 0.40% (참고)* | Advisory. 빈도 의존적 — EV/Cost 통과 시 자동 함의되는 경향 |

### B. 복리 엔진 (Compound Engine)
| 평가 지표 | 합격 기준 (Pass) | 실무적 의미 |
| :--- | :--- | :--- |
| **CAGR** | **≥ 30.0%** | 24개월 trailing 윈도우 기준. 크립토 위험 프리미엄 최소 상회 |
| **Sortino Ratio** | **≥ 1.8** (목표 2.5) | 하방 변동성 대비 수익성. 크립토 고변동성 구조 반영 (구 기준 2.0 → 1.8) |
| **Calmar Ratio** | **≥ 1.5** (목표 2.5) | 수익/낙폭 비율. `CAGR_floor(30%) / MDD_cap(20%) = 1.5` — 세 게이트 수학적 정합 |

> **Calmar·CAGR·MDD 정합 조건**: `Calmar_floor ≤ CAGR_floor / MDD_cap`. 현재 1.5 = 30% / 20%. 구 기준(2.5)은 실질 CAGR ≥ 50% 요구로 CAGR 게이트를 무력화하던 모순을 수정함.

### C. 생존 및 리스크 (Survival & Risk)
| 평가 지표 | 합격 기준 (Pass) | 실무적 의미 |
| :--- | :--- | :--- |
| **Max Drawdown** | **≤ 20.0%** | 복리 회복 탄력성 마지노선 |
| **MDD Duration** | **≤ 180 days** | 자산 정체 기간 상한. 자동매매 기계적 인내심 고려 |
| **CVaR (Tail)** | **≤ 1.3× MDD** | 극단 구간(하위 5%)에서 MDD를 크게 벗어나지 않는 안정성 |
| **Flash-crash 생존** | **단일봉 −25% 충격 후 MDD ≤ 게이트(20%)** | 크립토 gap 리스크. 순간 급락 시 청산·MDD 붕괴 방지 |
| **레버리지·청산 안전** | **백테스트 청산 이벤트 0 + gross ≤ 레버리지 상한** | 선물 고유 파산 경로 차단. 마진 계산 포함 검증 |

### D. 시장 방향 균형 (Directional Balance)
| 평가 지표 | 합격 기준 (Pass) | 실무적 의미 |
| :--- | :--- | :--- |
| **Long/Short minority** | **≥ 15%** | 숨은 long-only(베타 라이딩) 차단. 하락장에서도 수익 경로 확보 |

### E. ML 견고성 (ML Robustness)
| 평가 지표 | 합격 기준 (Pass) | 실무적 의미 |
| :--- | :--- | :--- |
| **PBO (Candidate)** | **< 10.0%** | 전략 과적합 확률 통제 |
| **PBO (Champion)** | **< 15.0%** | 실전 투입용 챔피언의 통계적 유의성 최종 하한 |
| **OOS Retention** | **> 50.0%** (**Sortino 기준**) | IS 대비 OOS Sortino 유지율. 분모(IS Sortino)가 0에 가까울 때 ratio 무효 — 분모 가드 필수 |

## 3. 전략 승격 및 감사 가이드라인
- **Hard Gate 우선순위**: `EV/Cost Ratio`, `MDD`, `PBO` 중 하나라도 FAIL 시 다른 지표에 관계없이 즉시 탈락.
- **24/7 Dashboard**: 모든 결과는 `[MECHANICAL 24/7 DASHBOARD]` 형식으로 리포팅.
- **Champion Swap**: 새 후보가 현 챔피언을 교체하려면 모든 지표에서 동등 이상 + `PBO < 챔피언의 PBO` + `EV/Cost Ratio > 챔피언의 EV/Cost Ratio`.

## 4. 재최적화 주기
| 주기 | 실행 내용 | 비고 |
| :--- | :--- | :--- |
| **반년 (Semi-Annual)** | alpha + HMM 학습 → Optuna A→B→C 전체 → Champion Swap 판단. Trailing 24개월 윈도우 | 챔피언 파라미터는 해당 실행의 alpha/HMM 아티팩트와 쌍으로 고정 |
| **즉시 (On-Demand)** | §9.1 경보 발령 시 전체 파이프라인 재실행 | 비상 완료 전 포지션 50% 유지 |

> **반년 사이에는 마지막 전체 실행의 alpha/HMM 아티팩트를 그대로 사용한다.** 파라미터는 해당 아티팩트 기준으로 최적화됐으므로, 모델만 교체하면 파라미터-모델 일관성이 깨진다.

## 5. 기계적 예외 조항
- `MDD Duration`이 90일을 초과하더라도 `EV/Cost Ratio`가 유효하면 승격 가능. (단, 180일 초과 금지)
