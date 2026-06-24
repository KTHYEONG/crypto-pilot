# 🎯 Objective
L2 AWF 시뮬레이션에 **per-fold 엣지 귀속(attribution) DEBUG 로깅**을 추가하여, `+60bps(L1) → -3.6% CAGR(L2)` 붕괴가 어느 항(알파 감쇠 / sizing 붕괴 / 비용 drag / funding)에서 발생하는지 정량 분리한다.

# 🧭 Diagnosis Context (왜 이 spec인가)
- L1→L2 체인 코드 검증 완료: 부호 반전·비용 이중차감·look-ahead·프로모션 전파 누락·sizing 단위 버그 **전부 없음**.
- L2 음수는 정직한 OOS 결과(2023-2024 검증 → 2024.12~2025.09 홀드아웃, 2025-Q2 모멘텀 반전 fold = -20.1%).
- 그러나 **현 로그(게이트 결과만)로는 "수식/설정/구간 중 어디서 엣지가 새는지" 분리 불가** → 본 spec은 *판정*이 아니라 *귀속 분해*를 로깅한다.
- 핵심 분해식:
  - `realized_total_f = realized_price_f + realized_funding_f − realized_cost_f`  (== `Σ fold_rets_hybrid[f]`, 자기검증)
  - `expected_net_f   = Σ_block (n_bars_block · dot(w_block, mu_arr_block · 1e-4))`  (sizing이 기대한 net 수익)
  - `alpha_gap_f      = realized_total_f − expected_net_f`
  - 해석: `expected_net>0 & realized_total<0` → **알파 감쇠**(코드 무죄). `expected_net≈0` → **pooling/throttle/cap이 엣지 소거**(설정). `realized_price>0 & realized_total<0` → **비용·funding drag**.

# 📦 Context & Dependencies
- **Target file (단일)**: `src/domain/futures/strategy/tiered_workflow/awf_sim.py`
- **Config 추가**: `src/domain/futures/strategy/tiered_workflow/dataclasses.py` → `Layer2AllocationConfig`
- **Imports (awf_sim.py 상단에 추가)**:
  ```python
  import logging  # isEnabledFor 가드용 (기존 함수내 import logging 재사용 가능)
  ```
- 기존 사용 심볼: `PERF` (이미 import), `logger = logging.getLogger("src.domain.futures.strategy.tiered_workflow")` (함수 내부 정의됨).
- **Data Shapes**: `w: NDArray[float64] [N]`, `mu_arr: NDArray[float64] [N] (bps, per-bar net)`, `bar_ret/funding: [N]`.

# ✍️ Contract Changes

## C1. 신규 dataclass (awf_sim.py, `Layer2ExpectedEdge` 아래 삽입)
```python
@dataclass(slots=True, frozen=True)
class Layer2FoldAttribution:
    fold_idx: int
    oos_bars: int
    n_rebal: int
    realized_total: float
    realized_price: float
    realized_funding: float
    realized_cost: float
    expected_net: float
    alpha_gap: float            # realized_total - expected_net
    mean_gross_exp: float
    mean_net_exp: float
    sleeves_active_mean: float
    friction_pass_ratio: float
    throttle_mult_mean: float
    dropped_below_cost: int
    netting_events: int
```

## C2. 신규 순수 함수 (awf_sim.py) — TDD 1급 타깃
```python
def _assemble_fold_attribution(
    *,
    fold_idx: int,
    oos_bars: int,
    n_rebal: int,
    realized_price: float,
    realized_funding: float,
    realized_cost: float,
    expected_net: float,
    gross_exps: list[float],
    net_exps: list[float],
    throttle_mults: list[float],
    sleeves_active: list[int],
    friction_pass_total: int,
    signal_total: int,
    dropped_below_cost: int,
    netting_events: int,
) -> Layer2FoldAttribution: ...
```
- `realized_total = realized_price + realized_funding - realized_cost`
- `alpha_gap = realized_total - expected_net`
- `friction_pass_ratio = friction_pass_total / signal_total if signal_total > 0 else 0.0`
- `throttle_mult_mean = mean(throttle_mults) if throttle_mults else 1.0`
- `mean_gross_exp = mean(gross_exps) if gross_exps else 0.0` (net 동일)
- `sleeves_active_mean = mean(sleeves_active) if sleeves_active else 0.0`
- 모든 float은 `np.isfinite` 미통과 시 `0.0` 치환(NaN 방어).

## C3. `_resolve_sleeve_signals_at_bar` 반환 확장 (2-tuple → 3-tuple)
```python
def _resolve_sleeve_signals_at_bar(...) -> tuple[
    dict[tuple[str, str], SymbolSignal],
    dict[tuple[str, str], tuple[float, float]],
    int,  # n_dropped_below_cost: signed_net==0 또는 non-finite로 탈락한 active sleeve 수
]:
```
- `n_dropped = len(active_sleeves_tradeable) - len(result)` 형태로 산출(루프 내 카운터).
- **caller 2곳 갱신**: fit-leg `_sleeve_signals, _ = ...` → `_sleeve_signals, _, _ = ...`; OOS `_oos_sleeve_sigs, _oos_sleeve_edges = ...` → `..., _oos_dropped = ...`.

## C4. `Layer2AllocationConfig` 신규 필드 (dataclasses.py)
| 필드 | 타입 | 기본값 | 역할 |
|---|---|---|---|
| `l2_diag_attribution_enabled` | `bool` | `False` | per-fold attribution 누적·로깅 on/off |
| `l2_diag_sleeve_top_k` | `int` | `15` | sleeve 샘플 로그 시 \|w\| 상위 K |
| `l2_diag_sleeve_sample_every` | `int` | `0` | 0=각 fold 첫 rebalance만, N=N rebalance마다 |
- `from_mapping`에 3필드 파싱 추가(`_as_int`/존재여부 가드, bool은 `bool(mapping.get(...))`).

## C5. `_AwfSimResult` 필드 추가 (default 빈 튜플 → 하위호환)
```python
fold_attributions: tuple[Layer2FoldAttribution, ...] = ()
```

# 🛠️ Algorithmic Plan (`_run_awf_simulation` 수정)

**Target**: `src/domain/futures/strategy/tiered_workflow/awf_sim.py` → `_run_awf_simulation`

### Step 0 — diag 플래그 + 설정 스냅샷
- Anchor (함수 초입, `k_rank = int(config.k_rank)` 부근):
  ```python
  k_rank = int(config.k_rank)
  ```
- 추가: `_diag = bool(getattr(config, "l2_diag_attribution_enabled", False))`
- `if _diag and logger.isEnabledFor(logging.DEBUG):` → `[L2-ATTR-CFG]` 1회 로그(V9 9-param + `fixed_cost_safety_mult`, `deploy_cost_safety_mult`, `edge_*`, `risk_budget_*`, `l2_sleeve_combine_method`).
- `fold_attributions: list[Layer2FoldAttribution] = []`

### Step 1 — fold 루프 진입 시 accumulator 초기화
- Anchor:
  ```python
  _fold_h: list[float] = []
  _fold_b: list[float] = []
  ```
- 추가(diag일 때만 의미, 비용 무시 가능하므로 항상 누적 가능):
  ```python
  _attr_price = 0.0; _attr_funding = 0.0; _attr_cost = 0.0; _attr_expected = 0.0
  _attr_gross_exps: list[float] = []; _attr_net_exps: list[float] = []
  _attr_throttle: list[float] = []; _attr_sleeves_active: list[int] = []
  _attr_friction_pass = 0; _attr_signal_total = 0
  _attr_dropped = 0; _attr_netting = 0
  ```

### Step 2 — rebalance 시 expected/throttle/netting/drop 누적
- Anchor (OOS sleeve resolve):
  ```python
  _oos_sleeve_sigs, _oos_sleeve_edges = _resolve_sleeve_signals_at_bar(
  ```
  → 3-tuple 언팩 + `_attr_dropped += _oos_dropped`.
- throttle: `_book_edge_score`/`_edge_throttle_multiplier`로 산출되는 스칼라 `m`(또는 disabled 시 `1.0`)를 `_attr_throttle.append(float(m))`.
- expected_net: 최종 cap·throttle 적용된 `w`와 `mu_arr` 사용:
  ```python
  _attr_expected += float(t_end - t) * float(np.dot(w, mu_arr * 1e-4))
  ```
- exposure: `_attr_gross_exps.append(float(np.sum(np.abs(w))))`, `_attr_net_exps.append(float(np.sum(w)))` (기존 `all_gross_exposures` 재사용 가능하나 fold-local 별도 누적).
- sleeves_active: `_attr_sleeves_active.append(len(_oos_sleeve_sigs))`.
- friction/signal: `_attr_friction_pass += friction_pass`, `_attr_signal_total += len(selected)`.
- **netting 검출**(diag일 때만): `_oos_sleeve_sigs`를 symbol별 그룹화 → 부호 혼재 & `abs(pooled_mu) < 0.5*max(abs(raw_mu_i))`인 symbol 수를 `_attr_netting`에 가산. pooled_mu는 `valid_signals[sym].raw_mu` 참조.

### Step 3 — 실현 항 분해 누적 (per-bar 루프)
- Anchor:
  ```python
  gross_ret = compute_futures_bar_return(
      weights=w, price_returns=bar_ret, funding_rates=funding_rates,
  )
  ```
- 분해(동일 입력 재사용, 추가 dot 2회):
  ```python
  _attr_price += float(np.dot(w, bar_ret))
  _attr_funding += -float(np.dot(w, funding_rates))
  ```
- cost: `_attr_cost += cost`  (cost는 `rebal_cost if t2==t else 0.0`, 기존 변수 그대로).

### Step 4 — fold 종료 시 attribution 조립 + 로깅
- Anchor:
  ```python
  fold_rets_hybrid.append(_fold_h)
  ```
- 추가:
  ```python
  if _diag:
      _attr = _assemble_fold_attribution(
          fold_idx=_fold_idx, oos_bars=fold.oos_end - fold.oos_start,
          n_rebal=len(_attr_throttle) or rebalance_count, ...
      )
      fold_attributions.append(_attr)
      if logger.isEnabledFor(logging.DEBUG):
          logger.debug(
              "[L2-ATTR] fold=%d oos_bars=%d n_rebal=%d realized_total=%.6f "
              "realized_price=%.6f realized_funding=%.6f realized_cost=%.6f "
              "expected_net=%.6f alpha_gap=%.6f mean_gross_exp=%.4f mean_net_exp=%.4f "
              "sleeves_active_mean=%.1f friction_pass_ratio=%.3f throttle_mult_mean=%.3f "
              "dropped_below_cost=%d netting_events=%d",
              _attr.fold_idx, _attr.oos_bars, _attr.n_rebal, _attr.realized_total,
              _attr.realized_price, _attr.realized_funding, _attr.realized_cost,
              _attr.expected_net, _attr.alpha_gap, _attr.mean_gross_exp, _attr.mean_net_exp,
              _attr.sleeves_active_mean, _attr.friction_pass_ratio, _attr.throttle_mult_mean,
              _attr.dropped_below_cost, _attr.netting_events,
          )
  ```

### Step 5 — sleeve-level 샘플 로깅 (선택적, top-K)
- 위치: OOS 루프 rebalance 직후, `if _diag and logger.isEnabledFor(logging.DEBUG)` 가드.
- 발화 조건: fold 첫 rebalance(=`t == fold.oos_start` 근사) 또는 `sample_every>0 and rebalance_count % sample_every == 0`.
- `valid_signals`/`w`/`_oos_sleeve_edges`에서 `|w|` 상위 `l2_diag_sleeve_top_k` symbol에 대해:
  ```
  [L2-ATTR-SLEEVE] fold=%d t=%d sym=%s side=%d raw_mu_pb=%.4f qw=%.3f w=%.4f friction_pass=%s
  ```

### Step 6 — 반환에 attribution 부착
- Anchor: `fit_rets_hybrid=tuple(all_fit_rets_hybrid),`
- 추가: `fold_attributions=tuple(fold_attributions),`

# 🧪 TDD Test Scenario Matrix
- **Test file**: `tests/unit/domain/futures/strategy/tiered_workflow/test_awf_attribution.py`
- **Mock/Fixture**: 순수 함수 위주 → mock 불필요. dataclass 직접 호출.

### S1 — `_assemble_fold_attribution` reconciliation (Happy)
- Input: `realized_price=0.05, realized_funding=-0.01, realized_cost=0.02, expected_net=0.03`
- Expect: `realized_total == pytest.approx(0.02)`, `alpha_gap == pytest.approx(-0.01)`.
- name: `test_assemble_fold_attribution_reconciles_total_and_gap`

### S2 — friction ratio zero-division 방어 (Edge)
- Input: `signal_total=0, friction_pass_total=0`
- Expect: `friction_pass_ratio == 0.0` (no ZeroDivisionError).
- name: `test_assemble_fold_attribution_zero_signal_no_div_error`

### S3 — throttle/exposure 빈 리스트 fallback (Edge)
- Input: `throttle_mults=[], gross_exps=[], sleeves_active=[]`
- Expect: `throttle_mult_mean == 1.0`, `mean_gross_exp == 0.0`, `sleeves_active_mean == 0.0`.
- name: `test_assemble_fold_attribution_empty_lists_use_safe_defaults`

### S4 — NaN 입력 방어 (Error/Robust)
- Input: `expected_net=float("nan")`
- Expect: `expected_net == 0.0` 및 `alpha_gap` 유한.
- name: `test_assemble_fold_attribution_nan_input_coerced_to_zero`

### S5 — `_resolve_sleeve_signals_at_bar` drop count (Boundary)
- Setup: 최소 `L2SimulationCache`에 2 sleeve — 하나는 `gross_bps > cost`(생존), 하나는 `gross_bps < cost`(net→0 탈락).
- Expect: 반환 3번째 값 `n_dropped == 1`, `len(result) == 1`.
- name: `test_resolve_sleeve_signals_reports_below_cost_drop_count`
- **Mock 보일러플레이트**: 기존 `build_l2_simulation_cache` 사용하는 통합 fixture가 있으면 재사용; 없으면 `L2SimulationCache`를 직접 구성(필드: `signal_mask_2d`, `side_2d`, `expected_gross_bps_2d`, `expected_net_bps_2d`, `holding_bars_2d`, `quality_weight_2d`, `sleeve_ids`, `sleeve_to_sym`). 1 bar × 2 sleeve 최소 shape.

### S6 — netting 검출 정확성 (Logic)
- 순수 헬퍼로 추출 권장: `_count_netting_symbols(sleeve_sigs, pooled, *, cancel_ratio=0.5) -> int`.
- Input: sym A에 sleeve(+10bps, qw=1), sleeve(-10bps, qw=1) → pooled≈0.
- Expect: `netting == 1`. 동일 부호 두 sleeve → `0`.
- name: `test_count_netting_symbols_flags_opposite_sign_cancellation`

### S7 — diag off 시 무로깅 (Integration, optional `@pytest.mark`)
- `caplog`로 `l2_diag_attribution_enabled=False` 실행 시 `[L2-ATTR]` 미발생, `True`+DEBUG 시 fold 수만큼 발생.
- name: `test_run_awf_emits_attribution_only_when_diag_enabled`

# 🚨 Risk & Constraints
- **성능**: per-bar `np.dot` 2회 추가 → diag off여도 누적은 경량 float; sleeve 샘플/netting은 `_diag` 가드로 off 시 0비용. 로깅은 `isEnabledFor(DEBUG)` 이중가드.
- **자기검증 불변식**: `realized_total_f ≈ sum(fold_rets_hybrid[f])` — 불일치 시 분해 로직 버그(테스트로 강제).
- **하위호환**: 모든 신규 필드 default 보유 → 기존 caller/직렬화 무영향.
- **No look-ahead**: 순수 사후 집계(이미 계산된 `w`, 실현 `ret` 재사용) → 시뮬 결정성 불변.
- **Scope**: 본 spec은 *관측*만 추가. 전략/게이트/수식 변경 없음(엣지 절대 미변경).
