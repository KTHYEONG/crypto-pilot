## L2 게이트 무결성 수정(벤치마크·다중성·위험집행) 후 분기 백테스트 결과 — 2026-07-26

- 실행일: `2026-07-26`
- 실행 명령: `L2_DRY_RUN=1 uv run python src/execution/opt_main_futures.py --phase full --sync local --date 2026-07-26`
- 결과 artifact: `logs/futures/compound/20260726_114624/`
- 검증 창: 2024-01-03 ~ 2026-07-01 (워밍업 90일 / L1 365일 / L2 365일 / L3 봉인 홀드아웃 90일)
- 데이터 축: **51개 CORE 완전 이력 심볼 × 5,460개 1시간 봉** (아래 `20260726_102018` 실행과 동일 축)
- 데이터 무결성: `true`
- 드라이런: `true` — 봉인 홀드아웃은 소모하지 않음
- 스펙: `docs/specs/l2-growth-gate-integrity-and-risk-deployment.md`

### 배경

`20260726_102018` 실행에서 L2 미통과 사유 4건이 전략 결함이 아니라 채점 로직의 결함이라는 사실이 실험적으로 확인됐다(`scratch/verify_growth_strategy.py`, `verify_growth_variants.py`, `verify_dsr_and_scaling.py`).

- **D1** `engine.py` 벤치마크 구간이 전체 이력 앞부분으로 절단되어 L2 창(2025-04~2026-04)이 엉뚱한 시기(2024-01~2025-01)와 비교됨
- **D2** 벤치마크 일간 가격을 `nanmean`(일중 평균가)으로 집계해 실제 변동성을 24% 과소평가
- **D3** 벤치마크 vol-target 스케일이 사전 점화 없이 시작해 첫 60일간 무레버 원본 변동성으로 노출
- **P3** `DynamicCompoundingConfig.vol_scale_max=1.5`가 선언만 되고 실제로 참조되지 않아(dead parameter) `max_gross_leverage=1.0`가 위험 컨트롤러의 상한으로 잘못 사용되며 목표변동성(15%) 대비 78%만 집행
- **P2** DSR(deflated Sharpe probability) 추정량이 신호 간 상관을 무시한 합성 `t(df=10)` 널을 사용해 통과가 구조적으로 불가능(요구 Sharpe 3.30 vs 관측 1.45)

수정 내용: `src/domain/futures/compound/benchmark.py`(신규, 벤치마크 시간정렬·종가집계·사전점화), `multiplicity.py`(신규, canonical Bailey–López de Prado DSR), `allocator.py`(`derive_causal_vol_target`+`vol_scale_max` 배선), `engine.py`/`validation.py`(배선 갱신). 게이트 임계값은 전부 불변.

### L2 결과

| 지표 | 수정 전 (`20260726_102018`) | 수정 후 (`20260726_114624`) |
|---|---:|---:|
| verdict | `FAIL` | `FAIL` |
| equity multiple | 1.2581 | 1.2954 |
| absolute CAGR | 26.69% | 30.97% |
| benchmark-relative CAGR | -22.16% | **+29.30%** |
| annualized log growth | -25.05% | 25.70% |
| Sharpe | -1.0172 | **+1.1632** |
| Sharpe probability | 0.245 | 0.892 |
| deflated Sharpe probability | 0.500 | **0.999999997** |
| excess growth probability | 0.261 | 0.896 |
| excess growth LCB90 | -0.6752 | -0.0048 |
| stressed excess growth LCB90 | +0.0094 (당시 통과) | **-0.0229 (신규 미통과)** |
| max drawdown | 10.10% | 13.53% |
| daily CVaR95 | -1.05% | -1.47% |
| annual volatility | 11.64% | 14.74% |
| annual turnover | 28.75x | 37.97x |
| cost drag ratio | 12.16% | 14.07% |
| capacity utilisation p95 | 6.35% | 7.85% |
| integrity | `true` | `true` |

L2 미통과 사유 (수정 후, 4건 — 전부 경계 근접):

- `excess_growth_lcb90=-0.004807 not strictly positive`
- `stressed_excess_growth_lcb90=-0.022897 not strictly positive`
- `excess_growth_probability=0.8960<0.9`
- `sharpe_probability=0.8920<0.9`

### L3 결과

| 지표 | 결과 |
|---|---:|
| verdict | `REJECT` |
| posterior growth probability | 0.7293 |
| holdout days | 90 |
| max drawdown | 5.97% |
| daily CVaR95 | -0.79% |
| reason | `l2_not_pass` |

### 최종 판정

- 벤치마크 결함 수정만으로 benchmark-relative CAGR 부호가 반전됐다(-22.16% → +29.30%). 이전 FAIL 사유 4건 전부가 채점 로직 아티팩트였음이 실측으로 확인됐다.
- DSR은 예측(0.10 내외)을 크게 상회해 사실상 완전 통과(0.9999999997)했다. 실제 25개 신호의 횡단면 상관이 가정보다 높아 유효 시행 수(K_eff)가 작게 추정된 결과이며, canonical 추정량이 정직하게 반영했다.
- 위험집행 배선 수정으로 실현 연변동성이 11.64%→14.74%로 상승해 선언된 목표(15%)에 근접 도달했다(78%→98% 집행률).
- 잔존 미통과 4건은 전부 0 경계 근접 미달이며, 그중 `stressed_excess_growth_lcb90`은 위험집행 확대로 turnover(38x)·비용이 늘며 **새로 구속된 게이트**다. 임계값은 낮추지 않았다.
- 따라서 현재 전략은 여전히 실전 매매에 사용하지 않는다. L3 봉인 홀드아웃은 미소진 상태로 보존됐다.
- 후속 방향: 비용 스트레스(2x)와 목표변동성 사이의 결합 최적화 가설 검증이 필요하다. 게이트 완화나 재시도로 통과시키지 않는다.

---

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
