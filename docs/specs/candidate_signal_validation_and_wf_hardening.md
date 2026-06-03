---
title: Candidate 레이블 정합성 · Signal-Only 검증모드 · 비용모델 현실화 · WF 견고화
domain: futures-strategy
type: bug-fix + refactor + prd
status: ready_for_implementation
priority: critical
created_at: 2026-06-03
related_paths:
  - src/domain/futures/strategy/candidate_labels.py
  - src/domain/futures/strategy/config.py
  - src/domain/futures/strategy/ablation.py
  - src/domain/futures/strategy_runtime/bridge.py
  - src/domain/futures/strategy/candidate_evaluation.py
  - src/domain/futures/optimization/objectives.py
  - src/execution/opt_main_futures.py
dependencies:
  documents:
    - docs/specs/candidate_ml_edge_attribution_plan.md
    - docs/specs/alpha1.md
last_verified: 2026-06-03
---

# Candidate 레이블 정합성 · Signal-Only 검증모드 · 비용모델 현실화 · WF 견고화

## 0. 결론 요약 (Why this spec exists)

사용자 4개 요청을 코드 검증 기반으로 설계한다. 핵심은 **"ML이 알파를 추출하는 게 아니라
낙관 편향된 레이블 위에서 거래회피로 통과한다"**는 attribution plan(P0-C)의 의심을 **레이블·비용·WF의
구조적 결함**으로 환원해 봉합하는 것이다.

| # | 요청 | 검증된 결함 | 조치 |
|---|---|---|---|
| 1 | 레이블 버그 점검 | barrier 감지는 high/low인데 **실현가는 close** → SL 낙관·TP 비관(L-2). `triple_barrier_label`이 edge-conditioned로 오염(L-1). `min_holding_bars` 무시(L-3). ATR 폴백 매직넘버(L-4) | 레이블러 정합화 |
| 2 | `mode=signal` 검증단계 컷오프 | 현재 phase=`strategy`/`alpha` 모두 ML 학습까지 무조건 진행 | `--mode {full,signal}` 추가, ML 학습 직전 단락 |
| 3 | 24bps 과도 여부 | candidate 경로는 **평탄 24bps 상수**. `objectives.py`는 이미 maker/taker/slippage 블렌드(RT≈12.8bps) 보유 → SSOT 불일치. 지정가 위주·자본≤10억이면 24는 ~2× 과보수 | 비용을 maker/taker/slippage·보유기간·ADV 함수로 도출, 기본 RT≈12.8bps + 1.5× 스트레스 |
| 4 | WF 약점 보완 | `train/valid/test_months` **전부 dead config**. 실제는 단일 fraction split(0.6/0.2/0.2), 비-WF. 단일 OOS 블록 → 표본부족·비정상성 무방비 | Purged Anchored Walk-Forward(K-fold)+교차폴드 일관성 게이트 |

---

## 1. 정확한 근본 원인 (Verified)

### L. 레이블링 결함 — `candidate_labels.py`

**L-1 (semantic 오염, 확정).** L174에서
`barrier_first_label = 1 if (barrier_label==1 AND edge_after_hurdle_bps>0)`.
L189는 `barrier_label_list.append(barrier_first_label)`, L206 `triple_barrier_label = barrier_label_list`.
→ `triple_barrier_label`은 **원시 삼중장벽 결과(어느 장벽 선도착)가 아니라 비용·헐들 차감 후 라벨**이다.
이름과 의미가 불일치한다. 원시 장벽 라벨을 소비하는 진단/메타라벨링은 모두 오염된다.

**L-2 (실현가 불일치 → SL 낙관, 확정·중대).** L137-141 장벽 도달은 `high`/`low`(장중 극값)로 판정하지만,
L156 `exit_px = close_path[exit_off]`는 **도달 바의 종가**를 쓴다.
- SL: `low <= -sl_thr`로 손절 판정했으나 실현 손실은 (반등한) close 기준 → **손실 과소평가 = edge 낙관 편향**.
- TP: `high >= tp_thr`로 익절 판정했으나 실현 이익은 (되돌림) close 기준 → 이익 과소평가.
- 부작용: `barrier_label==1`(TP)인데 `gross_ret_bps<0`이 공존 가능 → `barrier_first_label`이 0으로 뒤집힘(내부 모순).

이 SL 낙관이 라벨 edge를 부풀려 ML/selection을 "실제로 없는 알파"로 유도한다.
이는 attribution plan P0-C(capture<<1 가설)의 유력한 정량 원인이다.

**L-3 (`min_holding_bars` 무시, 확정).** events는 `min_holding_bars`를 보유(dataset feature col로 사용)하나,
레이블러는 `entry_idx`부터 **즉시** 장벽을 스캔한다. `exit_policy_mode="engine_aligned"`인데 엔진이 최소보유를
강제하면 라벨/엔진이 괴리한다.

**L-4 (ATR 폴백 매직넘버).** L126 `atr = entry_px * 0.01`. 평탄 1% 프록시는 1m~1d 전 타임프레임에서 부정확.
요청 #3(타임프레임 무관)과 충돌. 상수 분리·로그 대체 필요.

비누수 확인(수정 불필요): ATR는 `decision_idx=entry_idx-1`까지만(누수 없음), 진입은 `open[entry_idx]`,
경로는 `entry_idx..exit_limit`, same-bar 충돌은 SL 우선(보수적). 모두 정상.

### M. `mode=signal` 컷오프 부재

`bridge.run_candidate_strategy_for_universe()`(L290-296)와 `ablation.run_candidate_ablation()`(L430-436)는
조건 없이 `fit_candidate_gate`/`fit_candidate_edge_models`로 진행한다. "유효 전략 검증까지만"의 단락점이 없다.
검증에 필요한 산출물(no-leak 룰 승격 + 비용생존)은 이미 `compute_rule_diagnostics`와
ablation `rule_promo_no_leak` 변형에 존재 → 학습 직전에서 단락하면 된다.

### C. 비용 24bps — SSOT 불일치 & 과보수

- candidate: `cost_floor_bps=24.0`, `expected_cost_bps=24.0` (config.py L105,181) — **평탄 round-trip 상수**.
- objectives.py L77-85: `maker_ratio=0.20, maker_fee=2, taker_fee=5, slippage=2` →
  one-way `0.2*2+0.8*5+2=6.4bps`, **RT=12.8bps**, baseline_rt=14bps. 이미 더 정교한 모델이 별도 경로에 존재.
- Binance USDT-M 선물 기본: maker 2bps / taker 5bps. 엔진이 "지정가 위주"면 maker_ratio↑(예 0.7~0.8).
- 자본 ≤ 10억원(~$700k)을 유동성 상위 종목에 분산 → 1주문 notional ≪ ADV → **시장충격 ≈ 0**.
  지정가의 진짜 비용은 슬리피지가 아니라 **미체결/역선택**이며, 이는 `(1-maker_ratio)` taker 폴백으로 모델링.

결론: 24bps RT는 지정가·소자본 전제에서 ~2× 과보수. 단, **타임프레임이 짧을수록 거래빈도↑**라
"숫자만 낮추기"가 아니라 **보유기간 인지 net-edge**(amortize)가 본질. (objectives의 `COST_GATE_AMORTIZE` 참조.)

### W. WF 구조 약점

- **W-1 (dead config, 확정).** `train_months/valid_months/test_months`는 `src` 내 config.py 외 **미사용**.
  실제 분할은 `_candidate_ml_split_indices`(bridge.py L16): 단일 `fit=0..0.6n / cal / oos=..n`. 24개월 의도 미반영.
- **W-2 (단일 OOS 블록).** fit/cal/oos 1회 분할 → walk-forward 아님. `candidate_evaluation`의 DSR/PBO는
  단일 equity의 6개월 인위 서브블록에서 계산(L159-173) → 진짜 폴드별 OOS가 아님.
- **W-3 (표본부족).** 4h=2190bars/yr, 1d=365/yr. 제한된 히스토리에서 fit set이 과소. 단일 split은 한 레짐만 학습.
- **W-4 (비정상성).** 고정 split = 과거 레짐 학습→미래 레짐 평가. 폴드 교차·재학습 부재. `min_symbol_oos_blocks=3`은
  존재하나 단일 split이라 실효 없음.

---

## 2. Target Files

Modify:
- `src/domain/futures/strategy/candidate_labels.py` — L-1~L-4 정합화
- `src/domain/futures/strategy/config.py` — 비용모델 필드, signal-only, WF 필드 추가
- `src/domain/futures/strategy_runtime/bridge.py` — signal-only 단락, WF 분할 사용
- `src/domain/futures/strategy/ablation.py` — signal-only 검증경로, WF 폴드 집계
- `src/domain/futures/strategy/candidate_evaluation.py` — 폴드별 OOS 기반 DSR/PBO
- `src/execution/opt_main_futures.py` — `--mode` 인자, signal-only 리포트 출력
- `src/domain/futures/optimization/objectives.py` — 비용 도출 헬퍼 SSOT화(또는 공용 모듈로 추출)

New:
- `src/domain/futures/strategy/execution_cost.py` — `ExecutionCostModel` (SSOT)
- `src/domain/futures/strategy/walk_forward.py` — purged anchored WF 폴드 생성기

Tests (co-modification):
- `tests/unit/domain/futures/strategy/test_candidate_labels.py`
- `tests/unit/domain/futures/strategy/test_execution_cost.py`
- `tests/unit/domain/futures/strategy/test_walk_forward.py`
- `tests/unit/domain/futures/strategy/test_ablation.py` (signal-only)
- `tests/unit/execution/test_opt_main_futures_strategy_mode.py` (--mode)

---

## 3. Contracts

### 3.1 `candidate_labels.py` (레이블 정합화)

상수 분리:
```python
_ATR_FALLBACK_FRACTION = 0.01   # L-4: was inline 0.01
_TP_FILL_SLIPPAGE_BPS = 0.0     # 장벽가 체결 가정(보수적으로 0)
```

`label_candidate_events` 내부 변경:
- **L-2 fix:** 장벽 도달 시 실현가를 **장벽가**로 사용(close 아님):
  - TP 도달: `exit_px = entry_px * (1 + side*tp_thr)` (지정가 익절 체결 가정)
  - SL 도달: `exit_px = entry_px * (1 - side*sl_thr)` (스톱가 체결 — 보수적)
  - time_exit: 기존대로 `close_path[exit_off]`
  - short(side<0)은 동일 부호규약으로 대칭 적용.
- **L-1 fix:** 원시 장벽 라벨과 비용차감 라벨을 분리:
  - `triple_barrier_label` = 원시 결과(TP선도착=1, SL선도착/time=0)
  - `barrier_first_label` = 기존(TP AND edge>0) 유지
- **L-3 fix:** `min_holding_bars`(있으면) 이전 바의 장벽 도달 무시 — 스캔 시작 오프셋 적용:
  `scan_from = max(0, min_holding_bars - 0)` 기준 인덱스에서 `tp_hits`/`sl_hits` 필터.
  (`exit_policy_mode=="engine_aligned"`일 때만 적용; `label_only`면 기존 동작 유지하되 동일 가정 적용.)

### 3.2 `ExecutionCostModel` (NEW, `execution_cost.py`)

```python
@dataclass(slots=True, frozen=True)
class ExecutionCostModel:
    maker_fee_bps: float = 2.0
    taker_fee_bps: float = 5.0
    maker_ratio: float = 0.75       # 지정가 위주 엔진 반영
    slippage_bps: float = 1.0       # 지정가 체결 잔여 슬리피지(타임/큐)
    impact_coeff_bps: float = 0.0   # k in k*sqrt(notional/ADV); 소자본·메이저면 0
    stress_multiplier: float = 1.5  # fail-closed 비용 스트레스 배수

    def one_way_bps(self) -> float:
        fee = self.maker_ratio * self.maker_fee_bps + (1 - self.maker_ratio) * self.taker_fee_bps
        return fee + self.slippage_bps + self.impact_coeff_bps

    def round_trip_bps(self) -> float:
        return 2.0 * self.one_way_bps()

    def stress_round_trip_bps(self) -> float:
        return self.stress_multiplier * self.round_trip_bps()
```

기본값 검산: one-way = 0.75*2 + 0.25*5 + 1 = 3.75bps → **RT = 7.5bps**, stress = 11.25bps.
(보수 밴드를 원하면 maker_ratio=0.5 → RT=10.5bps, stress=15.75bps.)

### 3.3 `CandidateStrategyConfig` (config.py) — ADD

```python
# Execution cost (replaces flat 24bps; SSOT via ExecutionCostModel)
maker_fee_bps: float = 2.0
taker_fee_bps: float = 5.0
maker_ratio: float = 0.75
slippage_bps: float = 1.0
impact_coeff_bps: float = 0.0
cost_stress_multiplier: float = 1.5
cost_amortize_by_holding: bool = True   # 보유기간 인지 net-edge
# Signal-only validation mode
signal_only: bool = False
# Walk-forward
wf_enabled: bool = True
wf_scheme: Literal["anchored", "rolling", "single"] = "anchored"
wf_n_folds: int = 4
min_fit_obs: int = 200                  # 미만이면 prior-only fallback(fail-closed)
min_wf_fold_pass_ratio: float = 0.60    # 폴드 중 비용생존 비율 하한
```

검증:
- `0.0 <= maker_ratio <= 1.0`; 모든 fee/slippage/impact `>= 0`; `cost_stress_multiplier >= 1.0`
- `wf_scheme in {anchored, rolling, single}`; `wf_n_folds >= 1`; `min_fit_obs >= 1`
- `0.0 <= min_wf_fold_pass_ratio <= 1.0`
- `cost_floor_bps`/`expected_cost_bps`는 **유지하되 deprecated 주석**; 기본값을 `ExecutionCostModel().round_trip_bps()`로 도출(하드코딩 24 제거). 명시 오버라이드 시에만 상수 사용.

### 3.4 Walk-Forward 폴드 (NEW, `walk_forward.py`)

```python
@dataclass(slots=True, frozen=True)
class WFFold:
    fit_start: int; fit_end: int
    cal_start: int; cal_end: int
    oos_start: int; oos_end: int

def build_walk_forward_folds(
    *, n_bars: int, cfg: CandidateStrategyConfig,
) -> tuple[WFFold, ...]:
    """Purged + embargoed anchored/rolling WF folds.

    anchored: fit_start=0 고정, OOS 윈도우를 n_folds로 시간순 전진.
    rolling : fit 윈도우 길이 고정 이동.
    single  : 기존 _candidate_ml_split_indices 1폴드(하위호환).
    각 폴드: fit_end→cal_start에 purge_bars, cal_end→oos_start에 embargo_bars 삽입.
    """
```
- `single`은 `_candidate_ml_split_indices`를 래핑해 회귀 보장.

### 3.5 Signal-only 검증 리포트

```python
@dataclass(slots=True, frozen=True)
class SignalValidationReport:
    variant: str
    n_events: int
    net_edge_bps_p50: float        # no-leak 실현 net (cost RT 차감)
    net_edge_bps_stress_p50: float # stress cost 차감
    hit_rate: float
    ir_t_stat: float
    survives_cost: bool            # net_edge_stress_p50 > 0 AND t>=min_rule_ir_t
    deployment_count: int
```
`validate_candidate_signals()`가 변형별 리스트 + 종합 PASS/FAIL(하나라도 survives_cost) 반환.

---

## 4. Step-by-Step Logic

### Step 1 — 레이블 정합화 (요청 #1)
1. `triple_barrier_label`을 원시 결과로 복원(별도 리스트 `raw_barrier_list`).
2. 장벽 도달 시 `exit_px`를 장벽가로 치환(§3.1). time_exit만 close.
3. `min_holding_bars` 스캔 오프셋 적용(engine_aligned).
4. ATR 폴백 매직넘버 상수화.
5. 회귀: 기존 컬럼 이름/타입 유지, 의미만 정정. 다운스트림(`candidate_dataset.y_gate`는 `gate_label_column`
   기본 `profitable_after_hurdle_label` → 영향 없음). `triple_barrier_label` 소비처 grep 후 영향 점검.

### Step 2 — Signal-only 컷오프 (요청 #2)
1. `opt_main_futures.build_arg_parser()`에 `--mode`(`choices=["full","signal"], default="full"`) 추가 →
   `FuturesRunConfig`/`build_run_config_from_args`로 전파 → `strategy_cfg.candidate.signal_only`로 매핑.
2. `bridge.run_candidate_strategy_for_universe()`: `compute_rule_diagnostics` + 승격 직후,
   `if cfg.signal_only:` → `validate_candidate_signals(...)` 호출하고 **학습 전 반환**
   (`CandidatePipelineOutput.rule_report["signal_validation"]=[...]`, target_weights=0).
3. `ablation.run_candidate_ablation()`: `if cfg.signal_only:` → 변형 1(equal_size)·1b(rule_promo_no_leak)만
   실행하고 `EdgeAttribution` 없이 `SignalValidationReport` 테이블 반환.
4. `opt_main_futures._run_strategy_stage()`: signal_only면 `[SIGNAL-VALIDATION]` 테이블 출력 후 ML 단계 skip.

### Step 3 — 비용모델 현실화 (요청 #3)
1. `execution_cost.py` 신설(§3.2). `objectives.py`의 maker/taker/slippage 블렌드를 이 모델로 **위임**(중복 제거, SSOT).
2. candidate 경로: `cost_floor_bps`/`expected_cost_bps` 기본을 `ExecutionCostModel(...).round_trip_bps()`로 도출.
   레이블러/데이터셋의 `ex_ante_cost_bps`는 동일 모델 사용.
3. `cost_amortize_by_holding=True`면 per-event 비용을 `round_trip_bps / max(expected_holding_bars,1)`로 분할
   계상하지 않고(이중차감 금지), **selection 게이트의 edge floor를 보유기간 인지로** 적용
   (objectives `COST_GATE_AMORTIZE`와 동일 의미). 비용은 진입·청산 1회만 차감.
4. **fail-closed**: 최종 승격/selection은 `stress_round_trip_bps()`(1.5×) 차감 후에도 net>0이어야 통과.
   기본 비용은 낮추되 안전마진은 스트레스로 보존.

### Step 4 — WF 견고화 (요청 #4)
1. `walk_forward.build_walk_forward_folds()` 신설(§3.4).
2. bridge/ablation의 `_candidate_ml_split_indices` 호출을 `wf_enabled`/`wf_scheme`에 따라 폴드 루프로 교체.
   `single`은 기존과 동일(회귀 보장).
3. 폴드별: fit→cal→oos 학습/예측, OOS 예측을 시간순으로 **concat → 단일 OOS equity** 구성.
4. `candidate_evaluation`: DSR/PBO를 인위 6개월 서브블록이 아니라 **폴드별 OOS 수익**에서 계산.
5. 표본부족 fallback: `fit_set.X.shape[0] < cfg.min_fit_obs` → 해당 폴드는 **prior-only**(룰 진단 prior)로 강등,
   `gate=base_rate`. 전 폴드 강등 시 변형은 자동 탈락(fail-closed).
6. 비정상성: 변형은 `wf_n_folds` 중 `min_wf_fold_pass_ratio` 이상에서 비용생존해야 최종 승격(교차폴드 일관성).
   `min_symbol_oos_blocks`를 폴드 수와 정합.

---

## 5. Surgical Plan (요약)

- `candidate_labels.py`: REPLACE 장벽가 산출·라벨 분리·min_hold 오프셋·상수화 (§3.1, Step1).
- `execution_cost.py`: ADD `ExecutionCostModel` (§3.2).
- `walk_forward.py`: ADD `WFFold`,`build_walk_forward_folds` (§3.4).
- `config.py`: ADD §3.3 필드+검증, 24 하드코딩 → 모델 도출.
- `bridge.py`: ADD signal_only 단락(Step2-2), WF 폴드 루프(Step4-2/3/5).
- `ablation.py`: ADD signal_only 경로(Step2-3), WF 집계.
- `candidate_evaluation.py`: REPLACE 블록수익 산출을 폴드 OOS 기반으로.
- `opt_main_futures.py`: ADD `--mode`, signal 리포트 출력(Step2-1/4).
- `objectives.py`: REFACTOR 비용 블렌드를 `ExecutionCostModel` 위임.

---

## 6. Verification

L1:
```bash
uv run ruff check --fix src/domain/futures/strategy/candidate_labels.py src/domain/futures/strategy/config.py \
  src/domain/futures/strategy/execution_cost.py src/domain/futures/strategy/walk_forward.py \
  src/domain/futures/strategy_runtime/bridge.py src/domain/futures/strategy/ablation.py \
  src/domain/futures/strategy/candidate_evaluation.py src/execution/opt_main_futures.py
uv run mypy src/domain/futures/strategy/candidate_labels.py src/domain/futures/strategy/execution_cost.py \
  src/domain/futures/strategy/walk_forward.py src/domain/futures/strategy/config.py
```

Focused tests:
```bash
uv run pytest tests/unit/domain/futures/strategy/test_candidate_labels.py \
  tests/unit/domain/futures/strategy/test_execution_cost.py \
  tests/unit/domain/futures/strategy/test_walk_forward.py \
  tests/unit/domain/futures/strategy/test_ablation.py \
  tests/unit/execution/test_opt_main_futures_strategy_mode.py --tb=short
```

Integration (signal-only 재현):
```bash
FUTURES_STRATEGY_NAME=candidate_ml PYTHONPATH=. uv run python src/execution/opt_main_futures.py \
  --phase strategy --mode signal --sync skip --timeframe 4h --trials 1 --date 2026-05-01
```

기대:
- `--mode signal`: `[SIGNAL-VALIDATION]` 테이블 출력 후 ML 학습 없이 종료. 변형별 `survives_cost` 표기.
- 레이블: SL 실현가가 스톱가로 → 라벨 edge 분포가 하향(현실화). `triple_barrier_label`이 원시 결과로 복원.
- 비용: 기본 RT≈7.5bps(또는 설정값)로 표시, 최종 게이트는 1.5× 스트레스 차감 후 판정.
- WF: 다폴드 OOS concat equity + 폴드별 DSR/PBO. 변형은 ≥60% 폴드 비용생존 시에만 승격.

---

## 7. Acceptance Criteria

- [ ] 장벽 도달 실현가가 장벽가로 산출되어 `barrier_label==1 AND gross_ret<0` 모순이 제거된다.
- [ ] `triple_barrier_label`이 원시 삼중장벽 결과이고 `barrier_first_label`과 의미가 분리된다.
- [ ] `min_holding_bars` 이전 장벽 도달이 무시된다(engine_aligned).
- [ ] `--mode signal`이 ML 학습 전 검증 리포트만 산출하고 종료한다(기존 `full` 회귀 0).
- [ ] 평탄 24bps 하드코딩이 제거되고 비용이 `ExecutionCostModel`에서 도출된다(maker/taker/slippage/impact).
- [ ] 최종 승격/selection이 1.5× 스트레스 비용 차감 후 net>0을 요구한다(fail-closed 유지).
- [ ] WF가 K-fold purged/anchored로 동작하고 OOS가 폴드 concat으로 구성된다(`single` 회귀 보장).
- [ ] `min_fit_obs` 미만 폴드가 prior-only로 강등되고, 교차폴드 일관성(`min_wf_fold_pass_ratio`)이 승격 조건이다.
- [ ] `train/valid/test_months` dead config가 WF 윈도우로 연결되거나 명시적으로 deprecated 처리된다.

---

## 8. 확정된 결정 (User-confirmed 2026-06-03)

1. **WF 범위**: ✅ **K-fold WF 전면 도입**(anchored purged + 폴드 concat OOS + 교차폴드 일관성 게이트). `single`은 회귀용으로만 유지.
2. **비용 기본 밴드**: ✅ **`maker_ratio=0.75` → RT≈7.5bps**, 최종 게이트는 1.5× 스트레스(≈11.25bps) 차감. (엔진 실제 maker fill율 로그로 사후 캘리브레이션 권장.)
3. **레이블 SL 체결 가정**: ✅ **장벽가 체결**(TP=익절가, SL=스톱가). de Prado 삼중장벽 표준 정합.
</content>
</invoke>
