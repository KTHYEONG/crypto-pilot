## CORE 축 정합성 수정 후 분기 백테스트 결과 — 2026-07-26

- 실행일: `2026-07-26`
- 실행 명령: `L2_DRY_RUN=1 uv run python src/execution/opt_main_futures.py --phase full --sync local --date 2026-07-26`
- 결과 artifact: `logs/futures/compound/20260726_102018/`
- 검증 창: 2024-01-03 ~ 2026-07-01
  - 워밍업 90일
  - L1 365일
  - L2 365일
  - L3 봉인 홀드아웃 90일
- 데이터 축: **51개 CORE 완전 이력 심볼 × 5,460개 1시간 봉**
- 데이터 무결성: `true`
- 드라이런: `true` — 봉인 홀드아웃은 소모하지 않음

### 치명적 오류와 수정

CORE 데이터 커버리지 검증은 51개 심볼만 통과했지만 PIT 유니버스 축은 120개를 유지했다. 이 불일치로 누락 심볼이 L1 클러스터·신호 승인 입력에 남았고, 결과가 전 기간 cash-only(`target_weights=0`)가 됐다.

수정 내용:

- CORE 완전 이력 심볼만 PIT 유니버스에 유지
- `eligible`, `entry_block`, `exit_required`, `capacity`, `risk_scale`, `cost_bps`의 열 축을 동일하게 축소
- 누락 심볼을 0 가격·비적격 행렬로 후단에 전달하지 않도록 차단

수정 후 목표 비중은 `(5,460, 51)` 배열에서 비영 셀 **228,117개**로 생성됐고, `weight_abs_sum=2705.5249`, `max_abs_weight=0.0764`였다.

### L2 결과

| 지표 | 결과 |
|---|---:|
| verdict | `FAIL` |
| equity multiple | 1.2581 |
| absolute CAGR | 26.69% |
| benchmark-relative CAGR | -22.16% |
| annualized log growth | -25.05% |
| Sharpe | -1.0172 |
| Sharpe probability | 0.245 |
| deflated Sharpe probability | 0.500 |
| excess growth probability | 0.261 |
| excess growth LCB90 | -0.6752 |
| max drawdown | 10.10% |
| daily CVaR95 | -1.05% |
| annual volatility | 11.64% |
| annual turnover | 28.75x |
| cost drag ratio | 12.16% |
| capacity utilisation p95 | 6.35% |
| integrity | `true` |

L2 미통과 사유:

- `excess_growth_lcb90=-0.675205 not strictly positive`
- `excess_growth_probability=0.2610<0.9`
- `deflated_sharpe_probability=0.5000<0.9`
- `sharpe_probability=0.2450<0.9`

### L3 결과

| 지표 | 결과 |
|---|---:|
| verdict | `REJECT` |
| posterior growth probability | 0.8242 |
| holdout days | 90 |
| max drawdown | 5.23% |
| daily CVaR95 | -0.74% |
| reason | `l2_not_pass` |

### 최종 판정

- cash-only 및 0 비중 버그는 해소됐다.
- 현재 결과는 실행·원장 관점에서는 정상이며, L2가 실제 거래 포지션을 평가했다.
- 그러나 벤치마크 대비 성장성과 위험조정 성과가 부족해 L2/L3 모두 배포 기준을 통과하지 못했다.
- 따라서 현재 전략은 실전 매매에 사용하지 않는다.
- 이전 문서의 120심볼·4,380봉·absolute CAGR 41.01% 결과는 현재의 51심볼·5,460봉 분기 검증 창과 동일 조건이 아니므로 직접적인 성과 우열 비교에 사용하지 않는다.
