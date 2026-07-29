## L1 family-screen 도입 실전 검증 — 이중 부호반전 버그 발견·수정, 그러나 라우터는 여전히 공집합 — 2026-07-29

- 실행일(KST): `2026-07-29`
- 실행 명령: `UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. L2_DRY_RUN=1 L1_DEBUG=1 LOG_LEVEL=DEBUG timeout 1800 uv run python src/execution/opt_main_futures.py --phase full --sync local --date 2026-07-15 --seed 42`
- 프로세스: 2회 실행 전부 `exit_code=0` (크래시 없음)
- 검증 규모: `51` symbols × `5,442` 1h bars(내부 L1 4h), `L2_DRY_RUN=1`, sealed L3 holdout 미소비
- 관련 스펙: `docs/specs/l1-signal-edge-measurement-redesign.md` (+ `_contract.json`), `/check` PASS(Cov 88%)

### 배경 — 스펙 구현 후 `/check` PASS만으로는 프로덕션 데이터에서 실제로 엣지가 도달하는지 증명되지 않음

선행 세션에서 `no_evidence`의 원인이 신호 엣지 부재가 아니라 측정 구조 결함(측정 자체 미실행, 반정보 부호 추정기, 파손된 증거 tape, family 단위 다중검정 미통제)임을 실측 확정하고, family 단위 엣지 스크리너(`screen_family_edge`)를 도입하는 스펙을 작성·구현·`/check` PASS까지 마쳤다. 이번 실행은 그 실전 확인이다.

### 1차 실행 — family screen은 작동했으나 유일한 두 유의 엣지가 전부 거부됨

`logs/l1_admission.jsonl` 태그 분포: `ALGO=455`, `SCREEN=8`, `EVAL=1`, `REGIME=0`

| family | n_signals | mean_ic | t_NW | declared_orientation | 판정 |
|---|---:|---:|---:|---:|:--|
| **xs_reversal** | 2 | **+0.0332** | **+2.942** | −1 | **`declared_orientation_contradicted`** |
| **reversal_st** | 1 | **+0.0199** | **+2.753** | −1 | **`declared_orientation_contradicted`** |
| breakout_donchian | 5 | −0.0218 | −2.251 | +1 | contradicted |
| basis_gap | 5 | −0.0230 | −2.524 | +1 | contradicted |
| trend_ema | 5 | +0.0018 | +0.134 | +1 | not_significant_after_sidak |
| momentum_ts | 5 | −0.0136 | −1.260 | +1 | not_significant_after_sidak |
| xs_momentum_slow | 2 | −0.0028 | −0.141 | +1 | not_significant_after_sidak |
| smart_money_divergence | 2 | 0.0 | 0.0 | +1 | insufficient_ic_samples(n_ic_bars=0) |

`sidak_alpha=0.0061`(n_eff 기반), `n_ic_bars=1700`(모든 family 공통).

### 근본 원인 — signal_bank.py의 원시 z 계산과 카탈로그 부호 선언의 이중 반전

```python
# src/domain/futures/compound/signal_bank.py
def _compute_reversal_st(...): return -log_ret / vol            # 원시 z 계산 자체에 이미 mean-reversion 반전 내장
def _compute_xs_reversal(...): return _compute_xs_rank_signal(..., sign=-1.0)  # 마찬가지
def _compute_xs_momentum_slow(...): ... sign=+1.0                # 대조: 반전 없음, 그대로 랭크
```

`reversal_st`/`xs_reversal`은 z를 만드는 시점에 이미 "buy high z"가 맞는 방향이 되도록 부호를 내장했다. 그런데 스펙 구현이 추가한 `_family_orientation()`이 "이름이 reversal이니 -1"이라는 경제적 이름만 보고 **또 한 번 부호를 뒤집었다** — 이중 반전으로 원위치되어 실제로는 정반대 방향이 배포 대상이 됨. `trend_ema`/`momentum_ts`/`breakout_donchian`/`basis_gap`/`xs_momentum_slow`는 원시 계산에 내장 반전이 없어 `+1`이 옳았음을 소스 전수 확인.

### 수정

`_family_orientation()`을 전 family `+1` 반환으로 통일 (z 구성 자체가 이미 각 family의 경제적 가설 부호를 인코딩하므로, declared_orientation은 항상 "buy high z"가 정답).

### 2차 실행(수정 후) — 목표한 엣지가 정확히 admit됨

| family | t_NW | 수정 전 | 수정 후 |
|---|---:|:--|:--|
| **xs_reversal** | +2.942 | 거부 | **`admitted=True`** |
| **reversal_st** | +2.753 | 거부 | **`admitted=True`** |
| breakout_donchian | −2.251 | 거부 | 거부(정당 — momentum형 방향인데 실현이 반대) |
| basis_gap | −2.524 | 거부 | 거부(정당) |

family screen 로직 자체는 이제 스펙 의도대로 정확히 작동함을 실전 데이터로 확인.

### 그런데도 여전히 `no_evidence`, CAGR=0.0% — 라우터가 여전히 공집합을 받는 구조 결함

수정 후 재실행 결과:

| 항목 | 값 |
|---|---:|
| `l2.verdict` | `no_evidence` |
| `l2.reasons` | `active_days_ratio=0.0000<0.1`, `rebalances=0<30` |
| `l2.cagr` | `0.0%` |
| `l3.verdict` | `reject` |
| `logs/l1_admission.jsonl` 태그 분포 | `ALGO=455`, `SCREEN=8`, `EVAL=1`, **`REGIME=0`**(라우터가 evidence row를 단 1개도 생성 못함) |
| 구식 sleeve gate(`estimate_cluster_sleeve_posteriors`, HAC-beta 0.95)가 admit한 signal_id | `momentum_ts:*` 뿐 (medium×1, slow×1, very_slow×3 = 5 sleeve) |
| family screen이 admit한 signal_id | `xs_reversal`, `reversal_st` 뿐 |

**원인**: 라우터(`build_prequential_expert_route`)는 "family screen이 admit한 것" **AND** "구식 `estimate_cluster_sleeve_posteriors`가 admit한 것"의 **교집합**만 사용한다. 구식 게이트는 이번 스펙에서 손대지 않아 여전히 `momentum_ts` 계열만 통과시키고, family screen은 `xs_reversal`/`reversal_st` 계열만 통과시켜 **두 집합이 완전히 disjoint** — 교집합이 공집합이 되어 `collect_fold_expert_contributions`가 매 fold 빈 튜플을 반환한다. `tested_hypotheses`는 실질적으로 여전히 0.

### 판정 및 다음 착수점

1. 이중 부호반전 버그는 실측·수정·재검증 완료 — family screen이 이제 정확한 엣지(xs_reversal t=+2.94, reversal_st t=+2.75)를 admit한다.
2. **여전히 `no_evidence`** — 병목이 "부호 오류"에서 "두 개의 독립적인 게이트가 AND로 걸려 있고 그 결과가 disjoint 집합이라 교집합이 공집합"으로 완전히 재분류됨.
3. 다음 스펙 범위: family screen이 라우터의 **유일한** 게이트가 되도록 배선 변경 필요 — 구식 `estimate_cluster_sleeve_posteriors`의 HAC-beta 기반 admission(및 그 자체 부호 추정 `fitted_beta`)을 라우팅 경로에서 제거하고, family screen이 admit한 신호는 `declared_orientation`으로 직접 방향을 정한 sleeve(cluster/member 구조)를 별도로 구성해야 한다. 현재는 서로 다른 두 부호·선택 로직이 걸려 있어 한쪽만 고쳐서는 전진하지 못한다는 것이 이번 실측의 핵심 결론이다.
4. 임계값은 이번 세션에서도 완화하지 않았다.

- 결과 원본: `logs/futures/compound/20260729_042341/result.json`, `logs/l1_admission.jsonl`
