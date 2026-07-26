## 사이징 래칫 제거 + L1 결합층 재설계 — 2026-07-26 (드라이런)

- 실행: `L2_DRY_RUN=1 uv run python src/execution/opt_main_futures.py --phase full --sync local --date 2026-07-26`
- 데이터: **120 symbols × 4,380 bars**, integrity `true`
- 드라이런 모드: `dry_run: true` — L2 PASS가 나와도 봉인 홀드아웃을 `consume()`하지 않는 안전가드 적용, 실행 후 `sealed_holdouts.first_consumed_at_ns` 미변경(`None`) 확인

### 배경: 발견된 3층 구조적 결함

1. **사이징 래칫**: drawdown overlay가 EWMA 스무더의 내부 상태값(state)을 곱셈으로 오염시켜, "10% 노출 감축" 지시가 고정점 방정식(`w* = s·α/(1−(1−α)s)`) 구조상 실효 79% 감축으로 폭주. 회복 불가능한 단방향 소멸.
2. **L2 게이트 배선 결함**: L1 fold가 봉인 홀드아웃 구간과 겹쳐 look-ahead 발생, L2 `fold_ids_1d`가 전부 0으로 배선되어 `positive_outer_folds` 게이트가 구조적으로 통과 불가, DSR(deflated Sharpe) null이 표본 길이를 반영하지 않아 후보 27개 기준 임의로 높은 문턱(Sharpe>3.3) 요구.
3. **L1 결합층 결함**: `combine_posterior_sleeves`가 admitted sleeve 개수(=신호 반응 속도, 경제적 가치와 무관)에 비례 가중 → 최악 신호(net edge −0.83)가 최대 가중(6.6%)을, 최고 신호(+0.48)가 최소 가중(1.7%)을 받는 역상관 배분.

### 조치

- `allocator.py`: drawdown overlay를 출력 전용·회복 가능 구조로 재작성(EWMA state 미오염), rank-conviction 사이징 도입.
- `engine.py`: L1 fold를 L1 윈도우(`l1_window_end`)로 절단, L2 `fold_ids_1d` 시간 5분할 배선, 종목별 실측 `cost_bps_4h`를 시뮬레이터에 전달, `count_effective_candidates`로 DSR 후보 수를 유효 descriptor(20개)로 정정, `L2_DRY_RUN` 안전가드 추가.
- `l1_sleeves.py`: sleeve 채택 조건에서 OOS `fold_return > 0`(look-ahead) 제거, `select_non_redundant_signals`(fit-window 상관 기반 구조적 중복 제거) 도입, `combine_posterior_sleeves`를 "신호당 1표 등가중"으로 재작성.
- `validation.py`: DSR null을 표본 길이(`sqrt(periods_per_year/n_obs)`)로 스케일링, `cost_drag_ratio`의 로그공간/지수공간 혼용 차원 버그 수정(감사 중 발견), `absolute_cagr` 필드 추가.

### L2/L3 성과 — 수정 전후 비교

| 지표 | 수정 전 (2026-07-26 최초) | 사이징만 수정 | **사이징+결합층 수정 (본 실행)** |
|---|---:|---:|---:|
| L2 verdict | `FAIL` | `FAIL` | `FAIL` |
| equity multiple | 0.9686 | 1.2954 | **1.7573** |
| absolute CAGR | 5.98%(*) | — | **41.01%** |
| excess CAGR (vs 벤치마크) | — | 25.95% | **50.96%** |
| Sharpe | 0.2375 | 0.8884 | **1.6555** |
| max drawdown | 18.01% | 8.05% | 8.75% |
| annual volatility | 14.55% | — | 13.22% |
| annual turnover | 108.68x | 158.17x | **50.21x** |
| cost drag ratio | 23.83%(*) | 9.31%(*) | 15.87% |
| excess growth probability | 0.572 | 0.833 | **0.984 (기준 0.90 통과)** |
| deflated Sharpe probability | 0.000 | 0.021 | **0.553 (기준 0.90 미달)** |
| positive outer folds | 0/5 | 1/5 | **게이트 통과(사유 목록에서 제외)** |
| **L2 미통과 사유 개수** | 6개 | 6개 | **1개** |
| L3 posterior growth probability | 0.351 | 0.879 | **0.901** |
| L3 verdict | `REJECT` | `REJECT` | `REJECT` (`l2_not_pass`) |

*(\*) 수정 전 `cost_drag_ratio`는 로그공간/지수공간 혼용 버그로 왜곡된 값이었음(본 실행에서 수정).*

### 판정

- **L2 verdict: `FAIL`** — 유일한 미통과 사유는 `deflated_sharpe_probability=0.5530<0.9`. 성장성(초과성장 확률, LCB90, stressed LCB90), 위험(MDD, CVaR95, 변동성), 효율성(비용 손실 비율, 용량) 게이트는 전부 통과.
- **L3 verdict: `REJECT`** (`l2_not_pass`) — L2가 막혀 있어 봉인 홀드아웃은 평가되었으나 소모되지 않음(dry-run).
- 구조적 버그(래칫, fold 배선, 결합층 역상관 가중) 3층을 모두 제거한 결과, 미통과 원인이 **6개의 구조적/통계적 결함에서 순수 통계적 유의성 부족 1개**로 좁혀졌다. 이는 배포 승인이 아니라, 이제 파이프라인이 정직하게 측정되고 있다는 뜻이다.

### 남은 병목

- `deflated_sharpe_probability`는 후보 20개(유효 descriptor) 기준 다중검정 보정을 반영한 통계량으로, 현재 관측 Sharpe(1.656)가 아직 문턱을 넘지 못한다. 신호 다양화(carry/OI/taker) 증분 기여는 실측상 전부 음수였고, 사후 계열 선택은 윈도우 비강건성이 확인되어 기각된 상태(`docs/decisions/decisions.md` `ADR_20260726_L1L2_GROWTH_RECOVERY_AND_COMBINATION_REDESIGN` 참조).
- 봉인 홀드아웃 실제 소모(`L2_DRY_RUN` 없이 재실행) 여부는 별도 승인 대기 중.

결과 파일: `logs/futures/compound/20260726_065236/result.json`
