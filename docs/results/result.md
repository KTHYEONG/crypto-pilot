## L1 book-cost 회계 수정 + signal net-of-cost 스크리닝 실제 데이터 측정 결과 — 2026-07-29

### 1. 실행 식별자와 원자료

| 항목 | 값 |
|---|---|
| 기준 실행 | `logs/futures/compound/20260729_104757/` |
| 진단 계측 스크립트 | `scratch/verify_signal_cost_accounting.py` (앙상블 cost 회계 항등식 위반 실측), `scratch/verify_net_edge_screen.py`/`verify_net_edge_screen_v2.py` (screen 설계 bake-off) |
| 기준 명령 | `UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. L2_DRY_RUN=1 L1_DEBUG=1 LOG_LEVEL=DEBUG timeout 1800 uv run python src/execution/opt_main_futures.py --phase full --sync local --date 2026-07-15 --seed 42` |
| reference date / seed | `2026-07-15` / `42` |
| base timeframe | `1h` (L1 내부 결정 grid `4h`) |
| 입력 규모 | `5,442` bars × `51` symbols |
| model version | `quarterly-v1` |
| data manifest hash | `0048c160d459209c959006389a269441c6d2d33c6dc079e9bd1659398cffc6b5` |
| process | `exit_code=0` |
| L2/L3 보호 | `L2_DRY_RUN=1`, sealed L3 holdout 미소비 |
| 적용 스펙 | `docs/specs/l1_book_cost_accounting_and_net_edge_screen.md` (P0: 앙상블 cost 회계 항등식 복원, P1: signal screen에 net-of-cost 게이트 추가) |

artifact hash (직전 실행과 동일 — 둘 다 all-zero cash-only 산출물이라 해시가 우연히 일치):

- `result.json`: `216e76fd9567318d5d4e5b4b6270f81082845d288b34a502953efe764419afde`
- `target_weights.npy`: `2a161f690b5593fc026fbf44c11205b0afd6e228295cb756bb4c779092e6c102` (`float32`, shape `(5442, 51)`, `nonzero=0`, `max_abs=0.0`, `mean_abs=0.0`)

### 2. 진단 — 이전 result.md의 "비용이 총알파 3~7배 압도"는 게이트 회계 결함이었다

직전 실행(`20260729_075352`)에서 관측된 gross 연 +9~31% vs cost 연 −68~70% 워터폴을 프로덕션 코드로 재현·분해한 결과(`scratch/verify_signal_cost_accounting.py`), 경제 현실이 아니라 **앙상블 게이트의 cost 이중청구**였다:

`build_fold_expert_books`가 `expert_w[e] = share_e × weights_2d`(`share = |z_e|/Σ|z|`)로 신호별 가상 서브북을 만들고 `aggregate_ensemble_evidence`가 성분을 단순 합산한다. `Σ_e share_e = 1`이라 gross/funding은 정확히 telescoping하지만, cost는 삼각부등식(`Σ_e |Δ(share_e·w)| ≥ |Δw|`) 때문에 telescoping하지 않는다 — share는 원시 z 기반이라 매 봉 요동치는 반면 실제 배포북은 `alpha_smooth=0.08`/`band_frac=0.60`으로 평활화되어 있다.

| | charged (구 게이트) | real book (실제 배포) |
|---|---:|---:|
| gross 연환산 | −0.0071 | −0.0071 (정확히 일치) |
| **cost 연환산** | **−0.5225** | **−0.0051** |
| turnover/bar | 0.2982 | 0.0029 |

**과다청구 102배.** charged turnover 0.2982는 직전 실행 cost −0.69에서 역산되는 0.394와 동일 자릿수로 동일 메커니즘임을 확증한다.

### 3. P0 — 회계 항등식 복원

`allocate_book_turnover_cost`를 신규 도입해 배포북 실제 비용을 turnover 책임비율로 배분하도록 수정, `Σ_e cost_e ≡ book_cost` 항등식을 구조적 불변식으로 강제(`[RULE-P0-4]`, 위반 시 `ValueError`)했다. `score_expert_returns`는 비용을 재계산하지 않고 주입값을 그대로 사용한다.

### 4. P1 — net-of-cost 신호 스크리닝

기존 gross rank-IC t-stat 게이트는 실현 net과 **음의 상관(−0.157)** 을 보였다(`scratch/verify_net_edge_screen.py`, 프로덕션 커널 재현) — 불완전이 아니라 반정보. 무평활 프록시 북(C1)은 turnover 50배 과대로 흑자 신호까지 전량 기각해 반증됐고, 평활 프록시(C2)는 근사오차(corr +0.403)가 남아, **프로덕션 allocator 실전 replay(C3)** 를 채택했다(37종 카탈로그 19.3s, 프록시 오차 0).

`screen_signal_edge`에 4번째 게이트로 배선: `replay_signal_standalone_book`(단독 신호를 실제 allocator에 태워 net 시계열 산출) → `screen_signal_net_edge`(bootstrap `P(net>0) ≥ config.min_growth_posterior_probability` AND `ann_net_growth > 0`). 신규 임계값 없이 기존 `min_growth_posterior_probability`(0.90) 필드를 재사용했다.

검증 중 발견·수정한 배선 결함: `screen_signal_edge`가 반환하는 `SignalEdgeRecord`에는 net-edge 필드가 정확히 채워졌으나, `L1AdmissionRecorder.record_family_screen` 호출에 해당 4개 키워드 인자가 누락되어 JSONL 진단 로그엔 항상 0.0이 찍히던 관측성 결함을 수정(회귀 테스트 2건 추가: 정상 전파 + `except ValueError` fail-closed 경로).

### 5. 실측 결과 — 37종 signal 카탈로그 전량 재검정

| 판정 | signal 수 | family |
|---|---:|---|
| `declared_orientation_contradicted` | 4 | trend_ema/momentum_ts/breakout_donchian fast/medium |
| `not_significant_after_sidak` | 23 | trend_ema/momentum_ts/breakout_donchian/basis_gap 전 speed, reversal_st/xs_reversal very_slow·ultra_slow, xs_momentum_slow 전 speed |
| `insufficient_ic_samples` | 2 | smart_money_divergence |
| **`net_edge_not_significant_after_cost`** | **8** | reversal_st/xs_reversal fast~slow (아래 표) |
| **admit** | **0** | — |

이전 회계 결함 상태에서 gross IC로 admit됐던 8개 신호의 **실측 net-of-cost 성과** (회계 수정 후 진짜 배포북 기준):

| signal | t_nw (gross IC) | turnover/bar | net 연환산 | P(net>0) | edge/turnover |
|---|---:|---:|---:|---:|---:|
| `reversal_st:fast` (8h) | +6.861 | 0.0073 | **−8.03%** | 0.003 | −50.6bps |
| `reversal_st:medium` (12h) | +8.698 | 0.0125 | **−12.49%** | 0.013 | −45.5bps |
| `reversal_st:moderate` (24h) | +5.880 | 0.0033 | **−6.97%** | 0.003 | −98.1bps |
| `reversal_st:slow` (48h) | +3.653 | 0.0036 | **−14.47%** | 0.003 | −182.7bps |
| `xs_reversal:fast` (8h) | +7.315 | 0.0158 | **−6.60%** | 0.000 | −19.1bps |
| `xs_reversal:medium` (12h) | +8.150 | 0.0344 | **−8.89%** | 0.012 | −11.8bps |
| `xs_reversal:moderate` (24h) | +4.883 | 0.0257 | **−7.06%** | 0.014 | −12.6bps |
| `xs_reversal:slow` (48h) | +4.373 | 0.0193 | −3.94% | 0.136 | −9.3bps |

강한 gross rank-IC(t up to +8.70)는 실재하나, 회계 수정 후 실제 배포북(스무딩된 랭크 컨빅션 북) 기준으로는 8개 전부 net 연환산이 음수이고 `P(net>0)`가 0.000~0.136으로 낮다.

### 6. L1 / L2 / L3 결과

`logs/l1_admission.jsonl` 최종 `EVAL` record:

```json
{
  "admitted_sleeves": 0,
  "distinct_series": 0,
  "oos_bars": 0,
  "ann_growth": 0.0,
  "ann_lcb90": 0.0,
  "pw_block": 0.0,
  "turnover": 0.0,
  "cost_drag": 0.0,
  "positive_folds": 0,
  "fold_growths": [],
  "mean_abs_net": 0.0,
  "admitted": false
}
```

| metric | value |
|---|---:|
| L2 verdict | `no_evidence` |
| annualized log growth / CAGR / Sharpe / max drawdown | `0.0` |
| integrity_ok | `true` |
| L2 reasons | `active_days_ratio=0.0000<0.1`; `rebalances=0<30` |
| L3 verdict | `reject` |
| L3 reasons | `low_growth_probability`; `l2_not_pass` |

### 7. 결론

1. **P0(회계 항등식 복원)는 실제 데이터에서 정확히 작동했다** — cost 과다청구 102배가 해소되고 `Σ_e cost_e == book_cost` 항등식이 프로덕션 경로에서 구조적으로 보장된다.
2. **P1(net-of-cost 스크리닝)도 정상 배선됐다** — 37종 카탈로그 전량이 4단 게이트(IC 표본 → Sidak 유의성 → 방향성 → net-of-cost)를 통과해 재검정됐고, 8개 신호가 gross IC 통과 후 net 게이트에서 실측값(turnover/net_ann/prob)과 함께 정직하게 탈락했다.
3. **admitted_sleeves가 8 → 0으로 줄었으나 이는 회귀가 아니라 설계대로다.** 이전엔 회계 버그로 비용이 과다청구되어 우연히 안전하게 거부됐지만, 이제는 회계가 정확한 상태에서 진짜 net 계산으로 거부된다. gross t-stat과 실현 net의 음의 상관(−0.157, `scratch/verify_net_edge_screen.py`)이 37종 전체 재검정에서도 재확인됐다.
4. **cash-only는 유지되나 사유가 근본적으로 바뀌었다.** 이전엔 "게이트 회계 결함으로 인한 도달 불가능한 임계값"이었고, 이번엔 "회계가 정확한 상태에서 실측된 net 엣지 부재"다. 임계값을 낮추거나 cash-only를 해제할 근거는 없다.
5. **다음 착수점**: gross rank-IC(t 최대 8.70)가 net으로 왜 소실되는지 — `alpha_smooth=0.08`/`band_frac=0.60` 완충이 8~48h 반전 신호군에 부적합한지, 혹은 이 horizon대 반전 자체가 이 유니버스에서 비용을 구조적으로 감당 못하는지가 다음 스펙의 질문이다.

원본 artifact:

- [result.json](../../logs/futures/compound/20260729_104757/result.json)
- [manifest.json](../../logs/futures/compound/20260729_104757/manifest.json)
- [target_weights.npy](../../logs/futures/compound/20260729_104757/target_weights.npy)
- [l1_admission.jsonl](../../logs/l1_admission.jsonl)
- [l1_book_cost_accounting_and_net_edge_screen.md](../specs/l1_book_cost_accounting_and_net_edge_screen.md)
