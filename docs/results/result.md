# L1→L2 Handoff 실측 결과

## 1. 실행 메타데이터

| 항목 | 값 |
|---|---:|
| 실행일 | 2026-07-24 |
| reference date | 2026-07-23 |
| 데이터 기간 | 730일 |
| 4h bar | 4,380 |
| 1h bar | 17,520 |
| 종목 | 120 Binance perpetual |
| seed | 42 |
| data integrity | `true` |
| full artifact | `logs/futures/compound/20260724_114634/result.json` |

## 2. 최신 L1 handoff 결과

### 2.1 내부 평가 후보

최종 통과가 아닌 fold별 평가 후보:

`basis_gap:fast`, `breakout_donchian:fast`, `momentum_ts:fast`, `reversal_st:fast`, `smart_money_divergence:fast`, `trend_ema:fast`, `xs_momentum_slow:slow`, `xs_reversal:fast`

### 2.2 최종 admission

| 지표 | 값 |
|---|---:|
| outer folds | 5 |
| dev-only diagnostic bars | 3,300 (마지막 1,080 bars holdout 제외) |
| positive risk-adjusted folds | 0/5 |
| active signal 수 | 8 |
| admitted signal 수 | **0** |
| growth LCB90 | 0.000 |
| effective breadth 보고값 | 8.0 |
| admission | **false** |
| reasons | `growth_lcb90_not_positive`, `positive_folds_0_below_4` |

### 2.3 Dev fold 범위

| Fold | OOS index |
|---:|---:|
| 0 | [633, 1158) |
| 1 | [1158, 1683) |
| 2 | [1683, 2208) |
| 3 | [2208, 2733) |
| 4 | [2733, 3258) |

## 3. 최신 L2/L3 결과

| 지표 | 값 |
|---|---:|
| target weight non-zero rows | 0 / 4,380 |
| L2 annualized log growth | 0.000 |
| L2 equity multiple | 1.000 |
| L2 MDD | 0.00% |
| L2 annual volatility | 0.00% |
| L2 turnover | 0.000 |
| L2 integrity | `true` |
| L3 verdict | `REJECT` |
| L3 reason | `low_growth_probability` |

해석: L2가 본전으로 거래한 것이 아니라, L1 admission 실패로 전 구간 cash-only fallback이 실행됐다.

## 4. 계산 유효성 경고

실행 중 다음 경고가 발생했다.

`handoff.py:210 RuntimeWarning: invalid value encountered in log1p`

| 확인 항목 | 상태 |
|---|---|
| 일부 handoff weight가 gross cap 없이 표준화됨 | 수정 필요 |
| `log1p(return)` 입력이 -1 이하가 될 가능성 | 존재 |
| dev MDD | `NaN` 발생 |
| dev 성장률 수치 | 0으로 보수적 붕괴 |
| full runner fold 입력 | 4,380 (holdout 경계 미적용) |
| dev-only diagnostic fold 입력 | 3,300 |
| 전체 시계열 상관 계산 | holdout 포함 가능성, 수정 필요 |
| 현재 수치로 alpha 수익성 확정 가능 | 불가 |

따라서 최신 실행의 결론은 **“거래 차단은 정상 작동”**이며, **“신호가 수익성이 없다”는 통계적 확정은 아님**이다.

## 5. 이전 기준선과의 비교

| 버전/상태 | L2 log growth | MDD | Ann. vol | 상태 |
|---|---:|---:|---:|---|
| v6 dyn-Kelly | -6.90 | -100.0% | 102.8% | reject |
| v5 quarter-Kelly | -3.14 | -98.4% | 45.6% | reject |
| v6.1 price-risk sizing | -0.384 | -16.55% | 15.99% | L3 shadow |
| 최신 handoff full run | 0.000 | 0.00% | 0.00% | cash-only / reject |

## 6. 신호·데이터 참고값

| 항목 | 결과 |
|---|---:|
| metrics_5m partitions | 2,196 |
| metrics_5m null rate (OI / LSR) | 0.0% / 0.01% |
| smart-money divergence 730d meanIC | -0.0156 @24h, -0.0179 @72h |
| smart-money divergence split-half | -0.0113 / -0.0196 @24h |
| OI momentum 730d meanIC | +0.0041 @24h, +0.0080 @72h |
| OI vs price momentum rank corr | +0.075~+0.076 |

## 7. 현재 범위

- L2 simulator는 종가 기반 수익률·funding·turnover 비용을 반영한다.
- 고정 손절, 익절, ATR stop, trailing stop, intrabar stop hit는 아직 반영하지 않는다.
- 다음 데이터 보고 전 필수 수정: gross/per-symbol cap, `return > -1` 검증, holdout 이전 dev 경계, fit 구간만 correlation 계산.
