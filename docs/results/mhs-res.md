# MHS Horizon Diagnostic Quantitative Performance & Resource Report

- **Document Date**: 2026-08-10 (2차 구조 수정 후 재실행 결과로 완전 재작성)
- **Registered ADRs**:
  - `ADR_20260810_MHS_EXECUTION_ROSTER_RENORMALIZATION` (1차: 실행 roster 마스킹 후 dollar-neutral/unit-gross 재정규화)
  - `ADR_20260810_MHS_ROSTER_HYSTERESIS_VOL_TILT` (2차: roster 진입/이탈 히스테리시스 + causal inverse-vol tilt)
- **Domain**: Research / MHS (Multi-Horizon Market State)
- **Source Diagnostic File**: [`docs/results/mhs_horizon_diagnostic.json`](file:///home/kth/crypto-pilot/docs/results/mhs_horizon_diagnostic.json) (compact tier)
- **Execution Status**: `COMPLETE`(파이프라인 자체는 완주) — 단, 3북 중 2북(`fast_reversal`, `blend`)과 anchored fold 2는 `CAPITAL_INVARIANT_BREACH`로 리플레이 중도 실패
- **Run Metadata**: 2021-01-01~2025-12-31, `execution_universe_size=30`, `execution_timeframe=5m`, `eligible_symbols=445`, `run_elapsed_seconds≈261s`, peak RSS ≈ 6.45 GB

---

## 0. 이 리포트를 읽는 법 — 왜 숫자가 이전과 다른가

이전 버전의 이 문서(및 `mhs_horizon_opt_spec_summary.md`)는 **실행 roster 마스킹
버그**가 있던 상태의 결과였다: `_pit_execution_mask()`로 상위 30개 유동성
종목만 남기면서도 가중치를 재정규화하지 않아, 실제 배포 gross exposure가
의도한 100%가 아니라 우연히 ~7%로 희석되어 있었다. 이 버그가 자본 잠식을
가려주고 있었기 때문에, 당시 수치는 "그럭저럭 완주하지만 성과가 나쁜" 것처럼
보였다(MDD -10~-95%, 전부 완주).

두 차례 구조 수정을 거치며 이 착시가 걷혔다:

1. **1차 수정**(`renormalize_within_mask`): gross를 의도한 100%로 복원 →
   3북 전부와 fold 2가 곧바로 `CAPITAL_INVARIANT_BREACH`(자본이 0 이하로
   떨어져 원장이 fail-closed)로 중도 실패, 완주한 fold 0/1도 MDD가
   -58~-66%로 급등.
2. **2차 수정**(roster 히스테리시스 + causal inverse-vol tilt): roster
   유출입으로 인한 강제 전량매매와, 밈코인 등 최고-변동성 종목에 대한
   균등 rank-slot 과대 배분을 구조적으로 줄임 → `slow_momentum`이 5년
   전체를 파산 없이 완주, fold 0/1 MDD가 다소 개선(-43.4%/-55.8%). 그러나
   `fast_reversal`·`blend`·fold 2는 **여전히 자본잠식으로 완주 실패**.

**결론적으로 이 리포트가 보여주는 것은 "성과가 좋아졌다"가 아니라 "이전에
숨겨져 있던 진짜 문제(약한 알파 + 100% gross에서 감당 불가능한 실전 체결
비용)가 두 차례 수정을 거치며 점점 더 정직하게 드러나고 있다"는 것이다.**
자세한 진단·근거는 `docs/decisions/task_index.json`의 두 ADR과, 정리 전
spec 문서를 참고(spec 파일 자체는 `/sync`로 정리되어 저장소에는 남아있지
않음, ADR 요약에 핵심 로직 흐름이 기록됨).

---

## 1. Full-Period Primary Replay — 세 북의 현재 상태

| Book | 완주 여부 | Autocorr Sharpe | Naive Sharpe | Stress Naive Sharpe | MDD | Annualized Turnover | Research GO Target |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **fast_reversal** | ❌ **`CAPITAL_INVARIANT_BREACH`** ("pre-trade equity must be positive and finite") | — | — | — | — | — | ≥ +0.60 |
| **slow_momentum** | ✅ 완주 | **-1.8215** | -0.6913 | **+0.0991** | **-99.2375%** | 5.6231 | ≥ +0.60 |
| **blend** | ❌ **`CAPITAL_INVARIANT_BREACH`** ("simulated inventory equity must be finite and strictly positive") | — | — | — | — | — | ≥ +0.60 |

- `slow_momentum`은 5년 전체를 파산 없이 완주한 유일한 북이지만, MDD
  -99.24%는 사실상 초기 자본이 거의 전멸했다는 뜻이며 autocorr Sharpe도
  여전히 크게 음수다. "완주"를 "건전"으로 착각하면 안 된다.
- `fast_reversal`/`blend`는 리플레이 도중 자본이 0 이하로 떨어져 원장이
  `DataIntegrityError`를 fail-closed로 던지며 중단됐다 — primary/stress
  결과 객체 자체가 생성되지 않아 이 두 북은 Autocorr Sharpe·MDD·Turnover를
  보고할 수 없다(표의 "—"는 결측이 아니라 "측정 자체가 불가능"을 의미).
- Intent Shortfall(주문당 슬리피지)/Fill·Unfilled count는 `--output-tier
  full` 재실행 시에만 `_full/report.json`에 기록된다. 현재 저장된
  `_full/` 아티팩트는 이번 두 차례 수정 **이전** 상태의 것이라 이 표와
  정합하지 않으므로, 혼동을 피하기 위해 이 리포트에는 신지 않는다 — 필요 시
  `--output-tier full`로 재실행 후 갱신할 것.

---

## 2. Anchored Folds Quantitative Performance

| Metric | Fold 0 (2023) | Fold 1 (2024) | Fold 2 (2025) | Research GO Target |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Validity** | `True` (완주) | `True` (완주) | **`False`** — `CAPITAL_INVARIANT_BREACH` | All `True` |
| **Autocorrelation Sharpe** | **-4.6261** | **-7.1676** | — | ≥ +0.60 |
| **Naive Sharpe** | -1.4489 | -2.0418 | — | > 0.00 |
| **Stress Naive Sharpe** | -0.2461 | -0.5205 | — | > 0.00 |
| **Maximum Drawdown (MDD)** | **-43.4226%** | **-55.7931%** | — | Low MDD |
| **Failures** | `PRIMARY_AUTOCORR_SHARPE_BELOW_0_6`, `STRESS_SHARPE_NOT_POSITIVE` | 동일 | `CAPITAL_INVARIANT_BREACH` | — |

Fold 0/1은 완주했지만 Autocorr Sharpe가 각각 -4.63/-7.17로, gross가
소량이던 이전 상태(-0.90/-3.56)보다도 훨씬 나쁘다 — 100% gross에서는 단
1년짜리 검증 구간에서도 비용·변동성 노출이 압도적임을 보여준다. Fold 2는
2025년 구간 리플레이 중 자본잠식으로 아예 완주하지 못했다.

---

## 3. Books Prescreen & Tail Analysis (이론적 전체 유니버스 기준, 마스킹 이전)

Prescreen은 실제 roster 마스킹·실행비용을 적용하지 않고 445개 eligible
유니버스 전체·flat 8bps 비용 가정으로 계산한 **이론적 상한**이다. Research
GO 판정에는 쓰이지 않지만, 신호 자체의 알파 존재 여부를 가늠하는 참고
지표다.

| Book | Prescreen Net Sharpe (2.64bps) | Prescreen t-stat | Phase Ensemble Sharpe | Tail Base Sharpe | Leave-Worst-Out Sharpe |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **fast_reversal** | **-0.0974** | -0.218 | -2.3647 | -0.5417 | -0.5565 |
| **slow_momentum** | **+0.7507** | **1.679** | -0.0366 | +0.3284 | +0.3002 |
| **blend** | **+0.5836** | 1.305 | -2.3647 | -0.1266 | -0.1117 |

- `fast_reversal`은 이론적 전체 유니버스·비용 이전 단계에서도 edge가
  사실상 없다(t≈-0.2). 이 북은 실행 로직을 아무리 고쳐도 Research GO를
  기대하기 어렵고, 신호 자체의 재설계가 필요하다.
- `slow_momentum`은 이론상 약하지만 실재하는 edge(t≈1.68)를 보이는데,
  §1의 실제 리플레이(autocorr Sharpe -1.82, MDD -99.2%)와의 격차가 여전히
  극단적이다 — 실행/사이징 레이어가 이 이론적 edge를 심각하게 파괴하고
  있다는 뜻이며, 두 차례 구조 수정 이후에도 이 격차가 다 메워지지 않았다.
- `slow_momentum`의 `phase.degenerate=True`(phase_spread 0.152 >
  |mean_phase_ann| 0.0063)는 이 이론적 edge조차 특정 clock-offset에
  의존적일 수 있다는 별도 경고이며, 아직 해소되지 않았다.

Cross-sectional 진단(마스킹과 무관, 신호 자체의 통계적 유의성):
`xs_rank_ic`: mean IC = 0.0939, t = 68.4 (n=43,753일) — 신호와 미래
수익률 간 순위 상관은 통계적으로 매우 유의하다. 문제는 신호의 존재
여부가 아니라, 그 신호를 **실행 가능한 형태로 자본화**하는 과정(roster
선정·사이징·비용)에 있다는 진단을 다시 확인시켜 준다.

---

## 4. Research GO Evaluation

- **Research GO Final Status**: **`eligible: false`**
- **Reason Codes**: `CAPITAL_INVARIANT_BREACH`, `INCOMPLETE_ANCHORED_FOLD`, `PRIMARY_AUTOCORR_SHARPE_BELOW_0_6`, `STRESS_SHARPE_NOT_POSITIVE`, `UNSPECIFIED_POLICY` (evaluated_folds=3, folds_passed=0)
- 이전 버전(마스킹 버그 상태)의 사유 코드보다 항목이 **늘었다** —
  `CAPITAL_INVARIANT_BREACH`와 `INCOMPLETE_ANCHORED_FOLD`가 새로 추가된
  것은 회귀가 아니라, 재정규화로 실제 gross exposure가 정상화되며 이전에
  가려져 있던 자본잠식 실패 모드가 새로 드러난 결과다.

---

## 5. 진단 요약과 다음 단계

### 5.1 지금까지 확인된 근본 원인 (수정 완료)

1. **실행 roster 마스킹 후 미재정규화** (1차 수정, `ADR_20260810_MHS_EXECUTION_ROSTER_RENORMALIZATION`):
   `rank_weight_book()`의 dollar-neutral/unit-gross 북을 top-30 roster로
   마스킹하면서 재정규화하지 않아 gross가 몰래 ~7%로 붕괴했던 버그. 수정
   완료.
2. **roster 유출입의 강제 전량매매 + 변동성 무시 균등가중** (2차 수정,
   `ADR_20260810_MHS_ROSTER_HYSTERESIS_VOL_TILT`): roster 경계 flicker가
   2% 리밸런스 데드밴드를 우회해 강제 전량 진입/청산을 유발하고, 순수
   rank-slot 가중이 밈코인 등 최고-변동성 roster 멤버에 신호 신뢰도와
   무관하게 동일 비중을 배정하던 결함. 히스테리시스+inverse-vol tilt로
   부분 완화, `slow_momentum` 완주 전환·fold 0/1 MDD 개선까지는 확인,
   `fast_reversal`/`blend`/fold 2 자본잠식은 **미해결**.

### 5.2 남은 문제 (미해결, 후속 과제)

- 100% gross에서 여전히 자본잠식이 발생한다는 것은, 두 차례 수정이
  방향은 맞지만 크기가 부족함을 시사한다. 다음 순서로 재검토 필요:
  1. `_regime_cash_scale()`의 `vol_mean` 분모를 eligible 전체(445종목)
     평균이 아니라 실제 30종목 roster 자체의 변동성으로 좁히고, floor
     (현재 0.5)를 재검토.
  2. base gross target(현재 암묵적 100%) 자체를 하향하는 사이징 정책
     변경 — 별도 spec 대상.
  3. `ExecutionSpec.passive_timeout_minutes=30`(공유 비용 모델 상수)이
     6h/24h 결정 주기 대비 과도하게 짧아 timeout-taker fallback을
     남발시키는지 별도 검증.
- `fast_reversal`은 prescreen 단계부터 edge가 없다(§3). 실행/사이징을
  아무리 고쳐도 이 북 단독으로는 Research GO를 기대할 수 없으며, 신호
  재설계가 필요하다는 진단이 이번 재실행으로 재확인됐다.
- `MHS_EXECUTION_ROSTER_EXIT_MULTIPLIER=2.0`은 사전측정 상수가 아니라
  엔지니어링 기본값이다 — 이번 재실행에서 fold 0/1 MDD가 일부 개선된
  것으로 방향성은 확인됐으나, 최적값 탐색은 아직 하지 않았다.
