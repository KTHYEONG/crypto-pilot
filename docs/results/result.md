## L2 growth-evidence & L1 sizing integrity — production replay

### 1. 실행 조건

| 항목 | 값 |
|---|---|
| 실행 디렉터리 | `logs/futures/compound/20260730_085150/` |
| reference date / seed | `2026-07-15 / 42` |
| data manifest | `0048c160d459209c959006389a269441c6d2d33c6dc079e9bd1659398cffc6b5` |
| 입력 | 5,442 bars × 51 symbols |
| 실행 | `L2_DRY_RUN=1`, local sync, full phase, entrypoint `src/execution/opt_main_futures.py` |
| integrity / dry-run | `true / true` |
| L2 / L3 | `no_evidence / reject` |

`docs/specs/l2_growth_evidence_and_l1_sizing_integrity.md` 스펙을 적용한 재실행이다.
P0(단일 posterior 성장 게이트, degenerate cash prior 제거)와 P2(L1 admission이
L2 평가창을 미리 보지 못하도록 인과성 경계 적용)를 함께 배선했다.

검증 결과:

- `lean_check.py` (mypy/pytest/coverage 전체): **PASS** (Cov 89%)
- 관련 allocator/engine/validation/runner 테스트 135건: **전부 PASS**
- 실행 로그: optional `smart_money_divergence` 필드 경고와 NumPy 상관행렬 warning만 존재

### 2. 무엇이 바뀌었나

- **측정 버그 제거 확인.** `logs/l1_admission.jsonl` 최신 leg 기록: `evidence_weight`가
  순수 스크린 `{0.0, 1.0}`으로 복원(과거 모든 leg가 `0.5/0.5`로 강제 동률이었음),
  `alpha_sharpe`가 정상 연율화(1.698~2.257; 과거 0.037~0.044는 연율/bar 단위 혼동 오기재).
  두 leg(`trend_momentum:xs`, `vol_regime:ts`) 모두 개별 통계는 여전히 유의(`t≈2.0~2.6`,
  breakeven 대비 3~5배 비용 헤드룸).
- **인과성 경계가 실제로 결과를 뒤집었다.** 과거 admission은 15개 traded fold를
  사용했고 그중 79%가 L2 평가창과 겹쳤다(=자기가 채점할 데이터를 미리 보고 admit).
  경계를 `l2_start` 이전으로 제한하자 깨끗한 pre-L2 증거는 **5 fold/150일**뿐이며,
  그 창에서 포트폴리오 posterior가 `0.795`로 0.90 문턱을 넘지 못했다.

### 3. L2 / L3 결과

| Metric | 값 |
|---|---|
| verdict | `no_evidence` |
| L1 admitted | `False` — `posterior_0.795_below_0.9` |
| active_days_ratio | 0.0000 (<0.10) |
| rebalances | 0 (<30) |
| target_weights | 전 구간 0 (cash-only) |
| L3 verdict | `reject` — `low_growth_probability`, `l2_not_pass` |

이전 실행(`20260730_041009`)이 보고했던 CAGR 27.29% / excess growth probability 32.90%는
LCB90(+0.0919)과 확률(0.3290)이 동일 분포에서 산술적으로 양립 불가능한 measurement
결함(전액 cash인 L1 prior window를 `[:90]`으로 오래된 구간부터 슬라이스해 확률 0으로
채점 후 2배 가중치로 블렌딩)과, 자기 평가창을 미리 본 admission의 결합 산물이었다.
두 결함을 제거하자 그 수치들은 재현되지 않는다 — 임계값은 그대로(`min_excess_growth_probability=0.90`
유지)이며, 실제로 통과 가능한 근거 자체가 없었다.

### 4. 결론

1. **signal 자체는 여전히 방어 가능하다.** leg 단위 t-stat, 비용 헤드룸은 유효하며
   0.795는 0.90에 근접했다 — edge 부재가 아니라 얇은 깨끗한 증거 창(150일)이 원인이다.
2. **배포 승인은 아니다.** 이전 27.29% CAGR / cash-only가 아니라는 판단은 모두 측정
   결함에서 나온 허위 신호였다. 현재 정직한 상태는 `no_evidence` cash-only이다.
3. **다음 단일 개선 후보**는 두 갈래: (a) 더 이른 L1 시작점 또는 fold 재조정으로
   pre-L2 증거 기간을 150일보다 늘리는 것, (b) 0.795→0.90 격차의 부트스트랩 신뢰구간
   폭을 확인해 표본 부족 대 진짜 약한 edge를 구분하는 것. 임계값 완화는 여전히 금지.
4. `target_weights.npy`의 `float32` 저장 정밀도 이슈(과거 short-cap 반올림 초과)는
   이번 실행이 cash-only라 관측 불가 — 별도 실행에서 재확인 필요.

### 5. Artifact

- [result.json](../../logs/futures/compound/20260730_085150/result.json)
- [spec](../specs/l2_growth_evidence_and_l1_sizing_integrity.md)
- [contract](../specs/l2_growth_evidence_and_l1_sizing_integrity_contract.json)
- [L1 leg evidence](../../logs/l1_admission.jsonl)
