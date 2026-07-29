## L1 유효 horizon 재탐색 게이트(P5) 실전 적용 결과 — 2026-07-29

### 1. 실행 식별자

| 항목 | 값 |
|---|---|
| 기준 실행 | `logs/futures/compound/20260729_122330/` |
| 기준 명령 | `UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. L2_DRY_RUN=1 L1_DEBUG=1 LOG_LEVEL=DEBUG timeout 1800 uv run python src/execution/opt_main_futures.py --phase full --sync local --date 2026-07-15 --seed 42` |
| reference date / seed | `2026-07-15` / `42` |
| 입력 규모 | `5,442` bars × `51` symbols × `37` signals, 18-fold, pooled OOS 3,240봉 |
| data manifest hash | `0048c160d459209c959006389a269441c6d2d33c6dc079e9bd1659398cffc6b5` |
| process | `exit_code=0` |
| L2/L3 보호 | `L2_DRY_RUN=1`, sealed L3 holdout 미소비 |
| 적용 스펙 | `docs/specs/l1_effective_horizon_screening.md` (P5: declared-horizon IC 탈락자에게 스무딩 배포북을 7개 horizon 후보(24/48/96/144/216/432/648h)에 Sidak 보정 재검정하는 5번째 게이트) |

`/check` 과정에서 발견·수정한 배선 결함: `declared_orientation_contradicted`로 탈락한 신호가 반대 방향에서 유효 horizon을 찾아도 후속 net-of-cost 재검증이 원래 `declared_orientation`을 그대로 쓰던 버그(비대칭 long/short 레버리지 캡 때문에 부호를 잘못 넣으면 단순 부호반전이 아니라 다른 포트폴리오가 됨) 수정. `SignalEdgeRecord`의 신규 3개 필드에 대한 `__post_init__` 검증 누락도 보완.

### 2. 37종 전 신호 실측 결과 (`logs/l1_admission.jsonl` SCREEN 레코드)

| signal | t_IC(declared) | 1차 탈락 사유 | eh_horizon | eh_orientation | eh_t_stat | 최종 admitted |
|---|---:|---|---:|---:|---:|:---:|
| trend_ema:fast | −4.250 | declared_orientation_contradicted | 0 | 0 | 0.000 | ✗ (no_effective_horizon_found) |
| trend_ema:medium | +1.498 | not_significant_after_sidak | 0 | 0 | 0.000 | ✗ (no_effective_horizon_found) |
| trend_ema:moderate | +1.561 | not_significant_after_sidak | 0 | 0 | 0.000 | ✗ (no_effective_horizon_found) |
| trend_ema:slow | +1.191 | not_significant_after_sidak | 0 | 0 | 0.000 | ✗ (no_effective_horizon_found) |
| **trend_ema:very_slow** | +1.177 | not_significant_after_sidak | **24** | **1** | **3.278** | ✗ (net_edge_not_significant_after_cost) |
| **momentum_ts:fast** | −5.925 | declared_orientation_contradicted | 0 | 0 | 0.000 | ✗ (no_effective_horizon_found) |
| momentum_ts:medium | −1.534 | not_significant_after_sidak | 0 | 0 | 0.000 | ✗ (no_effective_horizon_found) |
| momentum_ts:moderate | +2.051 | not_significant_after_sidak | 0 | 0 | 0.000 | ✗ (no_effective_horizon_found) |
| momentum_ts:slow | +2.308 | not_significant_after_sidak | 0 | 0 | 0.000 | ✗ (no_effective_horizon_found) |
| momentum_ts:very_slow | +0.595 | not_significant_after_sidak | 0 | 0 | 0.000 | ✗ (no_effective_horizon_found) |
| breakout_donchian:fast | −6.239 | declared_orientation_contradicted | 0 | 0 | 0.000 | ✗ (no_effective_horizon_found) |
| breakout_donchian:medium | −2.933 | declared_orientation_contradicted | 0 | 0 | 0.000 | ✗ (no_effective_horizon_found) |
| breakout_donchian:moderate | −0.246 | not_significant_after_sidak | 0 | 0 | 0.000 | ✗ (no_effective_horizon_found) |
| breakout_donchian:slow | +1.492 | not_significant_after_sidak | 0 | 0 | 0.000 | ✗ (no_effective_horizon_found) |
| breakout_donchian:very_slow | +0.289 | not_significant_after_sidak | 0 | 0 | 0.000 | ✗ (no_effective_horizon_found) |
| basis_gap:fast~very_slow (5종) | −1.914~+0.189 | not_significant_after_sidak | 0 | 0 | 0.000 | ✗ (no_effective_horizon_found) |
| reversal_st:fast | +6.861 | *(declared 게이트 통과)* | — | — | — | ✗ (net_edge_not_significant_after_cost, 기확정) |
| reversal_st:medium | +8.698 | *(declared 게이트 통과)* | — | — | — | ✗ (net_edge_not_significant_after_cost, 기확정) |
| reversal_st:moderate | +5.880 | *(declared 게이트 통과)* | — | — | — | ✗ (net_edge_not_significant_after_cost, 기확정) |
| reversal_st:slow | +3.653 | *(declared 게이트 통과)* | — | — | — | ✗ (net_edge_not_significant_after_cost, 기확정) |
| reversal_st:very_slow/ultra_slow | +1.520/−0.225 | not_significant_after_sidak | 0 | 0 | 0.000 | ✗ (no_effective_horizon_found) |
| xs_reversal:fast~slow (4종) | +7.315~+4.373 | *(declared 게이트 통과)* | — | — | — | ✗ (net_edge_not_significant_after_cost, 기확정) |
| xs_reversal:very_slow/ultra_slow | +2.414/+0.358 | not_significant_after_sidak | 0 | 0 | 0.000 | ✗ (no_effective_horizon_found) |
| xs_momentum_slow:slow | +2.221 | not_significant_after_sidak | 0 | 0 | 0.000 | ✗ (no_effective_horizon_found) |
| xs_momentum_slow:very_slow | +0.675 | not_significant_after_sidak | 0 | 0 | 0.000 | ✗ (no_effective_horizon_found) |
| xs_momentum_slow:ultra_slow | +0.324 | not_significant_after_sidak | 0 | 0 | 0.000 | ✗ (no_effective_horizon_found) |
| smart_money_divergence:fast/medium | n/a | insufficient_ic_samples | 0 | 0 | 0.000 | ✗ (데이터 소스 부재, 기지 결함) |

**admitted = 0/37.** `l1_admission.jsonl` 최종 `EVAL`: `admitted_sleeves=0, oos_bars=0, ann_growth=0.0, fold_growths=[]`.

### 3. P5 게이트가 실제로 한 일

1. **trend_ema:very_slow — 유효 horizon 발견, 그러나 경제성 미달**: declared horizon(432h)에서는 유의하지 않던 신호가, 스무딩된 배포북 기준으로는 **24h에서 t=+3.278**(Sidak 보정 통과)로 유의했다. 즉 발견 자체는 성공했다. 그러나 그 24h에서 net-of-cost를 재실행하니 여전히 탈락 — **P5의 2단계 게이트(통계적 발견 → 경제적 재검증)가 설계대로 독립적으로 작동**함을 실전 데이터로 확인했다.
2. **momentum_ts:very_slow, xs_momentum_slow:slow/ultra_slow — 이전 세션의 "유망 신호" 재검증 결과, 정당하게 admit 안 됨**: 직전 세션의 비공식 census(37종 무보정 net-of-cost 스캔)에서 P(net>0)=0.90~0.96으로 통과선에 근접했던 이 3종이, 이번 P5의 **Sidak 보정 7-horizon 통계적 게이트를 통과하지 못해 애초에 net-of-cost 재검증 단계까지 가지도 못했다**(`no_effective_horizon_found`). 원인은 스펙 작성 시점에 이미 실측해 기록한 대로다 — 이 신호들의 스무딩북 IC t-stat은 어느 후보 horizon에서도 최대 t≈2.0 수준으로, Sidak 보정 임계값(약 t>2.68)에 못 미친다. **직전 세션의 "37종 무보정 스캔에서 우연히 걸릴 기대값(~3.5)과 실제 발견(4종)이 의심스럽게 근접"이라는 우려가 실제로 맞았음이 이번 엄격한 재검정으로 확인됐다.**
3. **momentum_ts:fast — 스펙 설계 자체의 한계 발견**: `declared_orientation_contradicted`로 탈락(raw z가 declared_orientation=1과 반대 방향으로 유의, t=−5.93)했으므로 P5는 규칙([LIMIT-02])대로 **반대 방향**(`search_orientation=-1`)에서 유효 horizon을 탐색했으나 아무것도 찾지 못했다(`no_effective_horizon_found`). 그러나 스펙 작성 전 사전 실측(`scratch/verify_smoothed_signal_true_horizon.py`)에서는 momentum_ts:fast의 스무딩북이 **원래 declared_orientation(+1, 반대가 아닌 그대로)** 방향으로 horizon이 길어질수록 t가 단조 증가(24h +0.24 → 648h +2.65)함을 이미 확인한 바 있다. 즉 **스무딩이 신호를 "반대로 뒤집는" 것이 아니라 "전혀 다른(더 느린) 신호로 변환"시키는데, P5의 `[LIMIT-02]` 규칙은 "방향 모순 탈락자는 반대 방향만 재탐색"으로 설계되어 있어 이 케이스의 진짜 기회(있다면)를 애초에 탐색 범위 밖에 둔다.** 이는 구현 버그가 아니라 스펙의 방향탐색 규칙 자체가 이번에 실측으로 새로 드러난 메커니즘(스무딩에 의한 horizon 변환)을 완전히 반영하지 못했다는 설계 갭이다.

### 4. L1 / L2 / L3 결과

| metric | value |
|---|---:|
| L1 admitted_sleeves | 0 / 37 |
| L2 verdict | `no_evidence` |
| L2 reasons | `active_days_ratio=0.0000<0.1`; `rebalances=0<30` |
| L3 verdict | `reject` |
| L3 reasons | `low_growth_probability`; `l2_not_pass` |
| `target_weights.npy` | `(5442, 51)` float32, `nonzero=0` |

### 5. 결론

1. **P5 인프라는 정확하게 작동한다.** 37종 전량이 올바른 순서(IC→방향성→[탈락 시] 유효 horizon 재탐색→net-of-cost)로 재검정됐고, 통계적 발견(trend_ema:very_slow의 24h)과 경제적 재검증이 설계대로 독립적으로 분리되어 동작함을 실전 데이터로 확인했다.
2. **직전 세션이 "약한 증거"로 유보했던 3종(momentum_ts:very_slow, xs_momentum_slow:slow/ultra_slow)은 엄격한 Sidak 보정 하에서 정당하게 탈락한다.** 즉시 admit하지 않고 인프라만 구축해 재측정에 맡긴 판단이 옳았음이 확인됐다 — 다중검정 우려가 실제로 근거 있었다.
3. **momentum_ts:fast에서 스펙의 방향탐색 규칙(`[LIMIT-02]`) 자체의 한계가 새로 드러났다.** "방향 모순 탈락자는 반대 방향만 재탐색"이라는 규칙이, 이번 조사로 새로 발견된 메커니즘(스무딩이 신호를 반대가 아니라 전혀 다른 effective horizon의 신호로 변환)과 맞지 않는다. 후속 스펙에서 `declared_orientation_contradicted` 탈락자도 **양쪽 방향(원래+반대) 모두** 재탐색하도록 규칙을 확장하는 것이 다음 후보다.
4. **cash-only 유지, admitted=0/37.** 임계값 완화 근거 없음. 코드 배선과 통계적 엄격함은 검증됐으나, 실제로 admit되는 신호는 이번 실행에서 발견되지 않았다.

원본 artifact:

- [result.json](../../logs/futures/compound/20260729_122330/result.json)
- [manifest.json](../../logs/futures/compound/20260729_122330/manifest.json)
- [target_weights.npy](../../logs/futures/compound/20260729_122330/target_weights.npy)
- [l1_admission.jsonl](../../logs/l1_admission.jsonl)
- [l1_effective_horizon_screening.md](../specs/l1_effective_horizon_screening.md)
