# 🎯 Objective
L1 신호 다양성(regime-조건화 비추세 패밀리)과 TF 커버리지(1h/2h/1d)를 measure-first로 확장하고, 미시구조(오더북 깊이) 기반 비용 현실성을 강화하되, 이미 반증된 접근(XS 팩터, L2 regime×family×TF 버킷 라우팅, breadth 레벨 판별, 무조건 mean-reversion 풀링)을 재시도하지 않는다.

**진행 상태 (2026-07-01)**: Phase 0 measure-first 완료(실제 파이프라인 실행, 16 fold). "추세 실패 구간 대안 신호" 원가설은 반증. `residual_revision`의 regime 게이트가 코드상 전혀 작동하지 않고 있었다는 사실을 발견 → Phase 2 구현 완료(beta_neut_gating_enabled opt-in 플래그, `("bull_quiet",)` only 허용, 기본값 False 무회귀). Phase 1은 여전히 연율화 하드코딩 이슈로 블로킹.

# 🚫 명시적 중단선 (재시도 금지 — 이번 spec에서도 유효)
- **XS cross-sectional factor** (xs_momentum/carry/flow/oi_skew) 승격 재시도 금지 — 2회 독립 반증.
- **L2 단의 regime×family×TF 버킷 라우팅 신설** 금지 — fit/oos edge 상관 음수(2/3 fold) 반증 완료 (`docs/results/next.md` L16). 이 spec은 라우팅이 아니라 **L1 개별 신호의 regime-조건부 승격**만 다룬다.
- **Breadth 레벨 기반 시장상태 판별** 재시도 금지 — 병목/평상 구간 판별력 없음 반증.
- **무조건(regime 비조건화) mean-reversion 전체 풀링** 재시도 금지 — 이미 rsi_reversion/bollinger_reversion/vol_regime_reversion 삭제 이력의 원인. **단, regime-조건부 재도입은 허용** (아래 Phase 2/3).
- **RC-2 `oos_blend` 하드닝**은 `docs/results/next.md` P1②로 이미 별도 스코프됨 — 이 spec과 병행 가능하나 중복 명세하지 않음.

## ⛔ 신규 블로킹 종속성 (자체 검증 후 추가, 2026-07-01)
**`calibrate_deployment_leverage`(risk_deployment.py:222-225, allocation/deployment.py:222-225)가 `bars_per_year=2190`(4h 가정)을 하드코딩.** `_resolve_l2_master_tf`/`_tf_edge_quality`(pipeline.py:2178-2224)는 Σ oos_edge_bps만으로 master TF를 선택하며 **TF 종류 제약이 전혀 없고**, `l2_master_tf` config 기본값이 `None`(자동선택이 실제 운영 기본 경로). 1h가 선택되면 실제 bars_per_year≈8760인데 2190으로 계산(4배 과소평가), 1d 선택 시 실제≈365인데 2190(6배 과대평가) — 메모리 기록된 기존 사고(`project_l2_parity_annualization_tf_2026_06_30`, study 4h vs final 8h 연율화 불일치로 챔피언 거짓 승격, **아직 미수정**)와 동일한 결함 클래스. **Phase 1은 이 중 하나가 선행되기 전까지 착수 금지**: (a) `_resolve_l2_master_tf`에 1h/1d 제외 allowlist 추가, 또는 (b) `calibrate_deployment_leverage` 호출부에서 `bars_per_year`를 실제 resolved TF로 파라미터화.

# 📦 Context & Dependencies

**Imports (측정/구현 대상 모듈)**
```python
from src.domain.futures.strategy.timeframe_contracts import HOURS_PER_BAR, RESAMPLE_ALIAS, PROBE_SOURCE_TFS
from src.domain.futures.strategy.config import CandidateStrategyConfig  # l1_tfs, per_tf_candidate_families
from src.domain.futures.signals.rules import build_rule_signal_panels  # family 등록부
from src.domain.futures.strategy.tiered_workflow.dataclasses import Layer1Result
from src.domain.futures.strategy.market_regime import ...  # code_1d (6-state), regime_side_split
from src.core.settings import SLIPPAGE_RATE, MAKER_FEE_RATE, TAKER_FEE_RATE
from src.core.exchange.binance_vision import BinanceVisionClient  # fetch_book_depth (기존, 미사용)
```

**Data Shapes & Types**
- `regime_side_split`: `dict[int, tuple[long_fraction: float, long_real_mean_bps: float, short_real_mean_bps: float, n_long: int, n_short: int]]` — 이미 `L1_PROBE_DIAG` 환경변수로 존재 (layer1.md §92(f)).
- `code_1d: NDArray[np.int8]` shape `[T]`, 0-5 discrete regime.
- `book_depth_df`: columns `[timestamp, percentage, depth, notional]` (Binance bookDepth 아카이브 스키마, `binance_vision.py:308` 참조 — 실제 페이로드는 raw fetch로 확인 필요, 가정 금지).

---

# Phase 0 — Measure-First: 기존 신호의 Regime별 차등 엣지 (신규 코드 최소)

## ✍️ Contract Changes
기존 `L1_PROBE_DIAG` 경로의 `regime_side_split`을 **family 단위**로 세분화하는 진단 전용 확장.

```python
def compute_family_regime_edge_diagnostics(
    *,
    realized_event_results: pd.DataFrame,  # = outer_events. 필수 컬럼: family/archetype, decision_idx,
                                            # entry_regime_code, realized_side_adjusted_gross_bps
                                            # (entry_regime_code 존재는 candidate_dataset.py:333 기준 유력 —
                                            #  outer_events 실제 컬럼셋 1줄 검증 후 착수할 것, 가정 금지)
    cfg: CandidateStrategyConfig,
    fold_id: int,
    seed: int = 0,
    min_bars: int = 8,
) -> FamilyRegimeDiagnostics | None:
    """L1_FAMILY_REGIME_DIAG env 게이트. 게이트 무영향, DEBUG 로그 전용."""
```

```python
@dataclass(slots=True, frozen=True)
class FamilyRegimeDiagnostics:
    """compute_xs_factor_spread_diagnostics의 XsFactorSpreadDiagnostics와 동일 패턴."""
    fold_id: int
    by_family_regime: dict[tuple[str, int], tuple[int, int, float, float, float, float, float]]
    # key=(family, entry_regime_code), value=(n_bars, n_events, mean_gross_bps, std, sharpe, lcb_gross_bps, rank_ic)
```

## 🛠️ Algorithmic Plan
- **Target Location**: `src/domain/futures/allocation/signal_batch.py` (미러: `src/domain/futures/strategy/tiered_workflow/signal_selection.py`) -> `compute_xs_factor_spread_diagnostics` **바로 옆에 신규 함수 추가**. ~~`signals/diagnostics.py`~~ **자체 검증 결과 정정: 해당 파일엔 `L1_XS_SPREAD_DIAG` 패턴이 존재하지 않음 — grep으로 실제 위치를 `allocation/signal_batch.py:1119`로 확인.**
- **Anchor**: `signal_batch.py:1119-1163` `compute_xs_factor_spread_diagnostics` 함수 전체 (그룹핑 `strategy_id`→`decision_idx` 구조, `moving_block_bootstrap_mean` LCB, `_xs_rank_ic` 호출 패턴 그대로 복제).
- **Logic Flow**:
  1. `realized_event_results`를 `(family, entry_regime_code)`로 group (기존 함수는 `strategy_id`로 그룹핑 — family/regime 조합으로 교체).
  2. 각 셀에 대해 기존 `moving_block_bootstrap_mean` + `_xs_rank_ic` 재사용 (신규 통계 함수 작성 금지).
  3. **모든 산출값은 gross** (`realized_side_adjusted_gross_bps` 그대로, 비용 미차감) — 필드명에 `_gross_` 명시해 Phase 2에서 net과 혼동 방지.
  4. `residual_reversion`, `dual_momentum` 등 비추세 패밀리가 특정 regime에서 양의 gross 엣지를 갖는지, `trend_donchian` 계열이 그 regime에서 음의 엣지를 갖는지 교차 확인.
  5. 결과를 DEBUG 로그로만 출력, 게이트 미입력.

## 🧪 TDD Test Scenario Matrix

**Test Environment & Fixtures**: `tests/unit/domain/futures/signals/test_diagnostics.py`에 이미 존재하는 `realized_event_results` 픽스처 패턴 재사용 (`L1_XS_SPREAD_DIAG` 테스트 인접 배치).

**Scenario 1 (Happy Path)**
- Input: 2개 family × 2개 regime_code, 각 30 events, family A regime 0에서 mean_gross_bps=+50, family B regime 1에서 mean_gross_bps=+30.
- Expected: 각 셀의 `mean_gross_bps`가 입력과 `pytest.approx` 일치, `n_events=30`.
- Test name: `test_compute_family_regime_edge_diagnostics_success`

**Scenario 2 (Edge Cases)**
- Input: 셀 `n_bars < min_bars`(8 미만, 기존 `compute_xs_factor_spread_diagnostics`와 동일 임계).
- Expected: 해당 셀은 `by_family_regime`에서 제외 (기존 함수의 `if n_bars < min_bars: continue` 패턴과 동일).

**Scenario 3 (Error Handling)**
- Input: `realized_event_results.empty == True`.
- Expected: `None` 반환 (기존 `compute_xs_factor_spread_diagnostics`의 `if realized_event_results.empty: return None`과 동일 계약 — 예외 아님).

---

# Phase 1 — TF 확장 (1h/2h/1d): 인프라 존재, config-level 실험

## ✍️ Contract Changes
```python
# src/domain/futures/strategy/config.py
@dataclass
class CandidateStrategyConfig:
    l1_tfs: tuple[str, ...] = ("4h", "6h", "8h", "12h")  # 기존
    # 실험 브랜치: l1_tfs_experimental: tuple[str, ...] = ("1h", "4h", "6h", "8h", "12h", "1d")
```

## ⚠️ Risk (필수 — quant.md §4 Realistic Crypto Microstructure)
- **[BLOCKING] 연율화 하드코딩 상호작용** — 상단 "신규 블로킹 종속성" 참조. 이 항목 해소 전 1h/1d를 `l1_tfs`에 추가하지 말 것.
- **1h TF는 회전율 급증** → RT cost(taker fee + slippage) 비중이 커져 기존 `l1_breakeven_floor_bps`(~7.5bps) 게이트가 자동으로 저품질 1h 신호를 걸러낼 것으로 예상되나 **실측 전 가정 금지**.
- **1d TF는 샘플 수 급감** → 기존 readiness gate의 `min_effective_obs`/`min_folds`가 1d에서 항상 미달할 위험. layer1.md의 "12h tightened" 패턴처럼 **1d 전용 완화 오버라이드 설계 필요** (1h는 이미 relaxed 오버라이드 존재, 1d는 없음).
- **PROBE_SOURCE_TFS 확장 불필요** — 1h는 이미 base 소스, 1d는 1h→1D resample로 `is_resample_compatible` 통과 (24시간 배수).
- **메모리/OOM 예산 재검토 필요** — 1h는 동일 윈도우에서 4h 대비 바 개수 ~4배. `l1_nested_result_soft_cap_mb`/`estimated_proc_gb` 공식이 4h~12h 기준으로 튜닝되어 있어, 1h 추가 시 soft cap 재산정 없이는 OOM 위험 (자체 검증 후 추가 — 원본 스펙 누락).
- **TF 우선순위 정정**: 연율화 오차는 4h와의 시간 거리에 비례(2h≈2배, 1h≈4배, 1d≈6배 잠재 오차) → 블로킹 종속성 해소 전이라면 **2h가 가장 안전한 파일럿**, 1h/1d 동시 우선 배정은 위 blocking 리스크를 정면으로 키움. 원본 스펙의 "2h 1순위 제외"는 재검토 대상.

## 🛠️ Algorithmic Plan
- **Target Location**: `src/domain/futures/strategy/config.py` -> `l1_tfs`, `per_tf_candidate_families`
- **Anchor**:
  ```python
  l1_tfs: tuple[str, ...] = ("4h", "6h", "8h", "12h")
  ```
- **Logic Flow**:
  1. **선행**: blocking 종속성(연율화 파라미터화 또는 allowlist) 해소.
  2. `l1_tfs`에 `"2h"` 우선 추가(가장 낮은 연율화 오차 잠재치), 검증 후 `"1h"`/`"1d"` 순차 확장.
  3. `per_tf_candidate_families`에 신규 TF 전용 pool 정의: 1h는 저지연 추세추종(`macd_4h`류 파라미터 스케일링) + 회전율 낮은 패밀리만, 1d는 `trend_donchian`/`dual_momentum` 등 저빈도 검증된 패밀리 위주로 시작(광범위 신규 아님).
  4. Readiness gate에 `"1d"` 전용 오버라이드 추가: `min_effective_obs`/`min_folds` 완화 폭은 기존 1h 패턴 참조.
  5. **Gate B(synthetic crash) 회귀 확인**: TF 추가가 기존 champion 승격 로직에 회귀를 일으키지 않는지 `synthetic_crash_defense_verdict` 재검증 (next.md P1 방침 재사용).

## 🧪 TDD Test Scenario Matrix
**Fixtures**: `tests/unit/domain/futures/optimization/test_opt_config_layered.py` 패턴 확장.

**Scenario 1**: `l1_tfs=("1h","4h","6h","8h","12h","1d")` config 생성 → `__post_init__` validation 통과.
- Test name: `test_candidate_strategy_config_accepts_extended_tfs`

**Scenario 2 (Edge)**: `l1_tfs`에 미지원 TF(`"3h"`) 포함 시 `ValueError` (HOURS_PER_BAR/RESAMPLE_ALIAS에 없는 키).
- Test name: `test_candidate_strategy_config_rejects_unsupported_tf`

**Scenario 3 (Error)**: 1d readiness gate에서 `min_effective_obs` 미달 시 `hard_eligible=False`로 정상 차단되는지 (신규 alpha 아님, 기존 게이트 로직 검증).
- Test name: `test_readiness_gate_1d_insufficient_obs_blocks`

---

# Phase 2 — `residual_reversion`(beta_neut archetype) Regime Gate 활성화

## 📐 Phase 0 실측 결과 (2026-07-01, 실제 파이프라인 실행 — `--phase l1`, 4 TF × 4 calendar fold)

### ⚠️ 통계적 재검증 (2026-07-01 2차 검토) — "16 fold"는 과대표기였음, 정정
1차 분석에서 "4 TF × 4 fold = 16 fold"라고 집계했으나, **로그의 실제 `Fold #N (FitEnd: ...)` 타임스탬프를 TF별로 대조한 결과 4개 TF(4h/6h/8h/12h) 모두 완전히 동일한 캘린더 구간을 재사용**함을 확인(Fold#0: 2023-12-30~2024-03-30, Fold#1: ~2024-06-30, Fold#2: ~2024-09-30, Fold#3: ~2024-12-30, 4개 TF 전부 동일). 즉 **독립적인 시장 관측치는 4개뿐**이며, TF는 같은 구간을 다른 해상도로 재관측한 것 — 이를 "16개 독립 fold"로 표기하면 pseudo-replication(의사 반복)으로 신뢰도를 부풀리는 것.

**per-family TF 소속 재확인**: `residual_reversion`은 `_DEFAULT_PER_TF_FAMILIES`(`config.py:1000`) 설정상 **1h/2h/4h 전용이며 6h/8h/12h 풀에는 애초에 포함되지 않음**(기존 설계, 버그 아님) — 로그 재검증 결과도 정확히 이와 일치(residual_reversion 이벤트는 4h 패스에서만 관측, 6h/8h/12h에는 전혀 등장하지 않음). 따라서 아래 표의 모든 residual_reversion 수치는 **애초부터 4개의 진짜 독립 캘린더 fold**(4h TF 내 Fold#0-3)이며 중복 계수 문제가 없음 — 우연히 1차 집계값과 최종값이 같지만, 근거가 이제 명시적으로 확인됨.

(대조: `dual_momentum@R1`은 4개 TF 전부에 존재해 16개 관측치가 잡히지만, 이는 **4개의 독립 캘린더 구간 × 4개의 TF-해상도 재확인**으로 해석해야 함. 다행히 이 셀은 16/16 전부 동일 부호(양수)라 TF-중복을 감안해도 결론은 변하지 않음 — "4개 독립 구간 전부에서, 4개 해상도 전부로 재확인된 양의 엣지"로 재서술.)

### Regime별 실측 (residual_reversion, 4개 독립 캘린더 fold, 전부 4h TF)

| regime_code | regime name | fold pass | per-fold LCB (fold0→3) | mean LCB | 판정 |
|---|---|---|---|---|---|
| 0 | bull_quiet | 3/4 (75%) | +29.1 / **+2.8** / +14.5 / +51.1 | **+24.4bps** | ✅ 기준 충족 |
| 1 | bull_volatile | 3/4 (75%) | +19.9 / +14.4 / +16.0 / **-41.4** | +2.2bps | ❌ fold3 단일 급락이 평균을 왜곡 — 불안정 |
| 2 | bear_quiet | 1/2 (50%)* | n/a | -14.5bps | ❌ 음수, *R2는 4개 fold 중 2개만 이벤트 발생(희소 regime) |
| 3 | bear_volatile | 1/4 (25%) | 개별값 미표시 | -43.1bps | ❌ 음수 |
| 4 | transition | 1/4 (25%) | **-7.3** / +40.1 / +1.3 / **-68.5** | -8.6bps | ❌ fold별 부호 전환 심함 — 음수 |
| 5 | crash | 2/4 (50%) | +42.6 / -16.3 / **-51.9** / +61.0 | +8.9bps | ⚠️ fold2→fold3에서 -51.9→+61.0 부호 급반전 — 4개뿐인 crash 관측치가 서로 모순, 확장 보류 |

**결론(Go, 좁게 한정, 통계적으로 재확인됨)**: R0(bull_quiet)만 기준 충족하며, 4개 fold 중 fold1(+2.8bps)이 가장 약한 지지 — 완벽하진 않지만 3/4가 breakeven을 확실히 상회(+29/+14.5/+51.1)하고 fold1도 최소 양수는 유지(부호 반전 없음). R2/R3/R4는 명확히 음수, R5는 fold 간 부호가 극단적으로 반전(-51.9→+61.0)해 표본 4개로는 방향성 자체를 신뢰할 수 없음. "추세 실패 구간(R2/R3)의 대안 신호"라는 원래 가설은 **반증**(`residual_reversion`도 그 구간에서 손실).

## 🔍 코드 검증으로 발견한 근본 사실 (자체 검증, 원본 스펙에 없던 내용)
`residual_reversion`의 regime 게이트는 **현재 완전히 비활성 상태**임을 코드로 확인:
- `_resolve_panel_archetype`(`rules.py:358`)는 `residual_reversion`의 메타데이터 `archetype="beta_neut"`을 그대로 반환.
- `_allowed_regimes_for_archetype`(`rules.py:386`)는 `xs_alpha`/`trend`/`ts_mom`/`flow_rev`/`unwind`/`carry_rev`만 명시적으로 분기하고, `beta_neut`은 분기 없이 마지막 fallback(`("bull_quiet","bear_quiet","transition")`)으로 빠짐.
- `_attach_signal_context`(`rules.py:438`)의 게이트 조건은 `cfg.regime_signal_gating_enabled or (cfg.mean_rev_gating_enabled and archetype == "mean_rev")` — `beta_neut != "mean_rev"`이고, `regime_signal_gating_enabled`는 **전체 코드베이스에서 단 한 곳도 True로 설정된 적이 없음**(grep 확인, opt_config.py/search_space.py Optuna 탐색공간에도 없음). 즉 **beta_neut 아케타입은 regime 게이트가 한 번도 실제로 작동한 적이 없다** — fallback 리스트 자체가 도달 불가능한 죽은 분기.
- `config.py:349` 주석: "regime 게이트는 sizing multiplier 레이어로 이전됨(`regime_as_size_multiplier`)" — 그러나 이 대체 메커니즘도 기본값 `False`(전역 미사용)이고, 활성화되어도 아케타입 무관 **전 신호 공통 배수**(`regime_size_multipliers`)라 beta_neut 전용 조정이 불가능.

## ✍️ Contract Changes
```python
# src/domain/futures/strategy/config.py (CandidateStrategyConfig)
mean_rev_gating_enabled: bool = True  # 기존
beta_neut_gating_enabled: bool = False  # 신규, mean_rev_gating_enabled와 동일 패턴의 opt-in 플래그
```

```python
# src/domain/futures/signals/rules.py (+ 미러 src/domain/futures/strategy/rule_signals.py)
def _allowed_regimes_for_archetype(archetype: str) -> tuple[str, ...]:
    ...
    if archetype == "beta_neut":
        return ("bull_quiet",)  # Phase 0 실측: R0만 breakeven 상회, R2/R3/R4 음수 확인
    ...
```

## 🛠️ Algorithmic Plan
- **Target Location**: `src/domain/futures/signals/rules.py`(+ 미러 `src/domain/futures/strategy/rule_signals.py`) → `_allowed_regimes_for_archetype`, `_attach_signal_context`. `src/domain/futures/strategy/config.py` → `CandidateStrategyConfig`.
- **Anchor 1** (`rules.py:386-395`):
  ```python
  def _allowed_regimes_for_archetype(archetype: str) -> tuple[str, ...]:
      if archetype == "xs_alpha":
          return ()
      if archetype in {"trend", "ts_mom"}:
          return ("bull_quiet", "bull_volatile", "bear_quiet", "bear_volatile")
      if archetype in {"flow_rev", "unwind"}:
          return ("bull_volatile", "bear_volatile", "crash")
      if archetype == "carry_rev":
          return ("bull_quiet", "bear_quiet", "transition")
      return ("bull_quiet", "bear_quiet", "transition")
  ```
- **Anchor 2** (`rules.py:438-440`):
  ```python
  if cfg.regime_signal_gating_enabled or (
      cfg.mean_rev_gating_enabled and archetype == "mean_rev"
  ):
  ```
- **Logic Flow**:
  1. `_allowed_regimes_for_archetype`에 `beta_neut` 전용 분기 추가(`("bull_quiet",)`) — 현재 도달 불가능한 generic fallback에서 명시적 분기로 승격.
  2. `CandidateStrategyConfig`에 `beta_neut_gating_enabled: bool = False` 추가(기본값 False → 기존 챔피언 무회귀 보장).
  3. `_attach_signal_context`의 게이트 조건에 `or (cfg.beta_neut_gating_enabled and archetype == "beta_neut")` 추가.
  4. **두 미러 파일(`rules.py`/`rule_signals.py`) 동일 적용** — Phase 0에서 확인한 기존 프로젝트 컨벤션(두 경로 모두 live import됨).
  5. FDR/SPA multiplicity 게이트는 기존 로직 그대로 통과(신규 예외 금지) — 이 변경은 side_hint 마스킹만 건드리고 promotion 게이트 로직은 무변경.

## ⚠️ Risk

### ✅ 설계 방향 결정 확정: 하드 마스킹 채택 (sizing-multiplier 기각)
두 대안을 검토한 결과 **하드 마스킹(`mean_rev_gating_enabled` 패턴 재사용)으로 확정**. 근거:
1. **증거의 형태가 이산적(discrete)**: Phase 0 실측은 "R0는 breakeven 상회, R2/R3/R4는 명확히 음수"라는 **통과/실패 이분법** 결과이지, "regime별로 신뢰도가 연속적으로 변한다"는 근거가 아님. `regime_size_multipliers`(연속 배수)는 애초에 이 증거 형태와 맞지 않음 — 있지도 않은 연속성을 가정하게 됨.
2. **Blast radius 비교**: `regime_size_multipliers`는 **아케타입 무관 전역 배수**라 활성화 시 residual_reversion뿐 아니라 현재 운영 중인 모든 신호의 사이징에 동시 영향(`regime_as_size_multiplier` 자체가 현재 전역 미사용 상태) — 하드 마스킹은 `beta_neut_gating_enabled` 플래그로 정확히 residual_reversion(beta_neut)만 격리 타격, 다른 아케타입 무영향.
3. **패턴 재사용성**: `mean_rev_gating_enabled`가 이미 동일 구조로 프로덕션에 존재·테스트됨(`test_mean_rev_gated_out_of_trending_regime`) — 신규 개념 도입 없이 검증된 패턴을 archetype 하나 늘려 복제하는 최소 변경.
4. `config.py:349` 주석("regime 게이트는 sizing multiplier로 이전")은 방향성 힌트일 뿐 강제 규칙이 아니며, 실제로 `mean_rev` 아케타입은 지금도 하드 마스킹 경로를 계속 사용 중(주석대로 완전히 이전되지 않음) — 기존 코드가 이미 두 메커니즘을 병행하고 있어, 하드 마스킹 추가가 기존 관행과 배치되지 않음.

이번 spec은 **하드 마스킹 단일 경로로 확정**하고 아래 Contract Changes/TDD를 그대로 구현 대상으로 삼는다. sizing-multiplier 경로는 채택하지 않음(향후 별도 필요성이 입증되면 재검토).

- **단일 셀 근거의 multiplicity 위험**: 128개 셀 스크리닝 중 1개 — Phase 0는 FDR/SPA 보정 없는 DEBUG 진단이므로, 이 변경이 실제 promotion 파이프라인(FDR/SPA 포함)을 통과하는지 별도 outer-fold 재검증 필요.
- **R1(bull_volatile) 제외 근거**: fold count(75%)는 기준 충족처럼 보이나 mean_lcb(+2.2)가 breakeven 미달 — 한 fold의 극단치가 평균을 왜곡했을 가능성. 보수적으로 제외.

## 🧪 TDD Test Scenario Matrix

**Test Environment & Fixtures**: `tests/unit/domain/futures/strategy/test_rule_signals.py`의 `test_mean_rev_gated_out_of_trending_regime` 패턴 재사용 (동일 `MarketRegimeContext`/`CandidateSignalPanel` 구성 방식).

**Mock Boilerplate Snippet**:
```python
def _beta_neut_panel() -> CandidateSignalPanel:
    return CandidateSignalPanel(
        family="residual_reversion", variant="unit", params={},
        datetimes=np.array([np.datetime64("2025-01-01T00"), np.datetime64("2025-01-01T04")]),
        symbols=("BTCUSDT",),
        signed_score_2d=np.ones((2, 1), dtype=np.float64),
        side_hint_2d=np.ones((2, 1), dtype=np.int8),
        expected_holding_bars=4, min_holding_bars=1,
        stop_atr_mult=1.0, take_profit_atr_mult=1.0,
        turnover_proxy_2d=np.zeros((2, 1), dtype=np.float64),
        valid_mask_2d=np.ones((2, 1), dtype=bool),
        metadata={"archetype": "beta_neut"},
    )

def _regime_ctx(codes: list[int]) -> MarketRegimeContext:
    return MarketRegimeContext(
        code_1d=np.array(codes, dtype=np.int8),
        name_by_code=("bull_quiet", "bull_volatile", "bear_quiet", "bear_volatile", "transition", "crash"),
        trend_score_1d=np.zeros(len(codes), dtype=np.float64),
        vol_z_1d=np.zeros(len(codes), dtype=np.float64),
        dispersion_z_1d=np.zeros(len(codes), dtype=np.float64),
    )
```

**Scenario 1 (Happy Path — 게이트 활성화, bull_quiet 통과)**
- Input: `_beta_neut_panel()`, `_regime_ctx([0, 0])`(bull_quiet), `cfg=CandidateStrategyConfig(beta_neut_gating_enabled=True)`.
- Expected: `side_hint_2d` 원본 그대로 유지(마스킹 안 됨, `bull_quiet`는 허용 리스트에 있음).
- Test name: `test_beta_neut_gated_allowed_in_bull_quiet`

**Scenario 2 (핵심 회귀 — bear_quiet에서 차단)**
- Input: `_beta_neut_panel()`, `_regime_ctx([2, 2])`(bear_quiet), `cfg=CandidateStrategyConfig(beta_neut_gating_enabled=True)`.
- Expected: `side_hint_2d`가 전부 0으로 마스킹됨 — Phase 0에서 확인한 음수 엣지 구간 차단.
- Test name: `test_beta_neut_gated_blocked_in_bear_quiet`

**Scenario 3 (Edge — 기본값 False, 무회귀 확인)**
- Input: `_beta_neut_panel()`, `_regime_ctx([2, 2])`(bear_quiet), `cfg=CandidateStrategyConfig()`(기본값, `beta_neut_gating_enabled=False`).
- Expected: `side_hint_2d` 마스킹 안 됨 — 기존 챔피언 동작 무변경 보장(회귀 방지 핵심 테스트).
- Test name: `test_beta_neut_gating_disabled_by_default_no_regression`

**Scenario 4 (Error/상호작용 — 전역 오버라이드와 정합성)**
- Input: `_beta_neut_panel()`, `_regime_ctx([2, 2])`(bear_quiet), `cfg=CandidateStrategyConfig(regime_signal_gating_enabled=True, beta_neut_gating_enabled=False)`.
- Expected: 전역 `regime_signal_gating_enabled=True`만으로도 마스킹됨(OR 조건의 첫 항) — 기존 `mean_rev` 패턴과 동일한 전역 오버라이드 정합성 확인.
- Test name: `test_beta_neut_masked_by_global_regime_signal_gating_override`

---

# Phase 3 — 미시구조: bookDepth 기반 유동성-인지 슬리피지 (알파 아님, 비용 모델)

## ✍️ Contract Changes
```python
# src/domain/futures/backtest/engine.py 대상 확장 (신규 함수)
def liquidity_adjusted_slippage_rate(
    base_slippage_rate: float,
    notional: float,
    book_depth_notional: float | None,  # None = 기존 고정치로 fallback
    depth_stress_mult_cap: float = 3.0,
) -> float:
    """오더북 깊이 대비 주문 규모 비율로 슬리피지 승수 조정. 데이터 없으면 기존 고정 SLIPPAGE_RATE 그대로."""
```

## ⚠️ Risk
- **[승격] bookDepth 스키마 오인 위험** — Binance bookDepth 아카이브의 `notional`은 실시간 오더북이 아니라 **mid가 대비 특정 %밴드 내 일간 평균 스냅샷**. 이를 실시간 depth인 것처럼 슬리피지 공식에 대입하면 체계적으로 틀린 비용 모델이 됨. **코드 작성 전 raw ZIP 1개 다운로드해 실제 컬럼/스케일을 확인하는 것이 하드 전제조건** (Context 섹션의 스키마는 미검증 가정).
- bookDepth 아카이브는 **일간 스냅샷**(daily ZIP) — 4h~1h 이하 TF의 인트라데이 유동성 변화 반영 불가. 저빈도(1d) 근사치로만 사용 가능, 과신 금지.
- 데이터 커버리지 미확인 상태 — 65개 승격 심볼 전체에 대해 bookDepth 아카이브가 존재하는지 먼저 확인 필요 (일부 저유동성 알트는 아카이브 자체가 없을 가능성).
- **[신규] 기존 Capacity Clip과의 중복 가능성** — layer1.md에 이미 `intended_notional < 5 USDT → w=0`, `> capacity → 비례 clip` 메커니즘 존재(`portfolio_nav` 활성 시). 이 clip이 유동성 정보를 이미 반영하는지 미확인 — Phase 3 착수 전 `deployment.py`의 capacity 정의를 먼저 확인해 중복/상충 여부 정리할 것.

## 🛠️ Algorithmic Plan
- **Target Location**: `src/core/exchange/binance_vision.py` -> `fetch_book_depth` (이미 존재, 호출부만 신규) + `src/domain/futures/backtest/engine.py`
- **Anchor**: `engine.py:117` `slip_eff = float(self.slippage_rate) * max(buf, 1e-9)` — 이 라인의 `self.slippage_rate`를 notional/depth 비율로 동적 대체.
- **Logic Flow**:
  1. **선행 작업(코드 아님)**: 65개 심볼에 대해 `fetch_book_depth` 아카이브 커버리지 실측 — 존재하지 않으면 Phase 3 전체 보류.
  2. 커버리지 확인되면 `depth_stress_ratio = notional / book_depth_notional`, `mult = clip(1 + depth_stress_ratio, 1.0, depth_stress_mult_cap)`.
  3. `book_depth_notional=None`(누락 심볼) → 기존 고정 `SLIPPAGE_RATE` fallback, 회귀 없음 보장.

## 🧪 TDD Test Scenario Matrix
**Scenario 1**: `book_depth_notional=1_000_000`, `notional=10_000` → mult≈1.01, base와 거의 동일.
- Test name: `test_liquidity_adjusted_slippage_deep_book_near_base`

**Scenario 2 (Edge)**: `book_depth_notional=None` → 정확히 `base_slippage_rate` 반환 (fallback, 회귀 방지 핵심 테스트).
- Test name: `test_liquidity_adjusted_slippage_missing_depth_data_falls_back`

**Scenario 3 (Edge)**: `notional > book_depth_notional * 10` (초저유동성 스트레스) → `mult == depth_stress_mult_cap`로 캡.
- Test name: `test_liquidity_adjusted_slippage_caps_at_depth_stress_mult_cap`

**Scenario 4 (Error)**: `book_depth_notional=0` → `ZeroDivisionError` 대신 안전 분모 처리(quant.md §3 Safe Division Guardrails — `np.where` 또는 epsilon 패턴 필수).
- Test name: `test_liquidity_adjusted_slippage_zero_depth_no_division_error`

---

# Phase 4 (신규, 사용자 요청 범위 밖 — 코드 검증으로 발견) — 포트폴리오 사이징의 상관관계 미반영

## 🔍 발견 경위
Phase 0/2 작업 중 실제 사이징 로직(`portfolio_constructor.py`)을 확인하는 과정에서, 사용자가 요청한 범위(L1 신호/TF/미시구조) 밖의 구조적 이슈를 발견. "효과적 자산증식"이라는 최상위 목표(CLAUDE.md §1)에 직접 연관되어 별도 항목으로 기록.

## 📐 사실 확인 (코드, 추측 아님)
`diagonal_kelly_weights`(`portfolio_constructor.py:810`)의 사이징 공식(docstring 원문):
```
w_raw_i = kelly_fraction * mu_bps_i / max(sigma_i^2, VOL_FLOOR^2)
```
- 이는 **교과서적 diagonal(대각) Kelly** — 심볼 i의 비중을 오직 자기 자신의 분산(`sigma_i^2`)만으로 결정하고, **심볼 간 공분산(상관관계)을 전혀 사용하지 않음**. Docstring도 명시: "기존 LW/BL/full-cov 경로와 독립적인 신규 함수" — full-covariance 경로가 코드베이스에 존재는 하되(LW=Ledoit-Wolf, BL=Black-Litterman 추정), 현재 라이브 경로(`diagonal_kelly_weights`)는 이를 쓰지 않음.
- 부분 완화책 존재: `PortfolioCaps.beta=0.50`(`portfolio_constructor.py:653`)이 `|w @ btc_beta| ≤ 0.5`로 **BTC 단일 팩터 기준** 순노출을 제한 — 그러나 이는 "BTC 베타로 설명되는 상관관계"만 통제하고, 알트코인 섹터 클러스터링 등 **BTC 베타 외 잔여 상관관계는 여전히 미반영**.
- L2 스코어카드 실측(result.md)에서 fold당 19~29개 심볼을 **동시 보유** — 이 심볼들이 실제로 얼마나 독립적인지(혹은 BTC 베타로 설명 안 되는 공통 요인에 얼마나 노출됐는지)는 현재 사이징 수식에 반영되지 않음.

## ⚠️ 왜 "치명적 버그"가 아니라 "검토 후보"인가 (과장 금지)
- 공분산 행렬 기반 Kelly는 well-known 실무 딜레마: N=50~150 심볼, 제한된 히스토리로 full-covariance 추정 시 노이즈가 신호보다 커지는 게 더 흔한 실패 모드(추정오차가 진짜 상관관계보다 큼) — quant.md §0 "Logic Robustness Over Metrics"와 "Do not over-engineer" 원칙상, **정교한 공분산 모델이 현재의 단순한 diagonal+beta-cap보다 항상 나은 것은 아님**.
- 이미 Ledoit-Wolf 등 shrinkage 기법이 코드베이스 어딘가(LW 경로)에 존재한다는 docstring 힌트가 있으므로, 이 팀이 이 트레이드오프를 이미 인지하고 의도적으로 단순 경로를 라이브에 채택했을 가능성이 높음 — **"놓친 버그"가 아니라 "알고 있는 단순화"일 가능성을 배제할 수 없음**. 확정 전 LW/BL 경로의 존재 이유와 diagonal 채택 배경을 먼저 조사(git blame/decisions 이력) 권장.

## 🛠️ 제안 (조사 우선, 코드 변경 아님)
1. **1차**: `docs/decisions/`에 diagonal Kelly 채택 근거가 이미 기록되어 있는지 확인(git log/blame으로 `diagonal_kelly_weights` 도입 커밋 추적) — 의도된 결정이면 이 항목 종결.
2. **기록 없으면**: 저비용 measure-first 진단 제안 — 현재 사이징 하에서 실현된 포트폴리오 vol이 diagonal 가정(Σ w_i² σ_i²) 대비 실측 vol(공분산 포함)이 얼마나 괴리되는지 사후 비교(신규 alpha 아닌 진단 전용, Phase 0와 동일한 DEBUG env-gate 패턴).
3. 괴리가 유의미하면 그때 Ledoit-Wolf shrinkage 공분산 기반 사이징을 별도 spec으로 설계 — 이 문서 범위 밖.

## 📎 부수 발견 (하우스키핑, 별도 조치 불요)
`tests/unit/domain/futures/strategy/tiered_workflow/test_layer2_gate_fixes.py`의 4개 테스트(`test_l2_alloc_space_v3_*`, `test_l2_alloc_space_alias_*`)가 `src.domain.futures.optimization.opt_config`에 존재하지 않는 `L2_ALLOC_SPACE_V3`/`L2_ALLOC_SPACE` 심볼을 import — 현재 `opt_config.py`에 `ALLOC_SPACE` 계열 심볼이 전혀 없음(과거 리팩터로 제거되고 테스트만 잔존한 것으로 추정). Kelly 관련 실제 로직(`kelly_fraction` 필드, `diagonal_kelly_weights`)은 정상 작동 중이라 자산증식과 무관한 **순수 테스트 부채** — 별도 커밋으로 삭제 또는 갱신 권장(이 spec 범위 밖).

---

# 📌 실행 순서 권고 (2026-07-01 2차 갱신 — 통계 재검증 + Phase 4 추가 반영)
1. **~~Phase 0~~ [완료, 통계 재검증 완료]** — 실제 파이프라인 실행(`--phase l1`) + 사후 재검증으로 "16 fold"가 실제로는 "4 독립 캘린더 fold × 4 TF-재관측"임을 확인, 결론(수치)은 불변이나 근거 표기를 정정. 결과: 신규 패밀리 발견 0건, "추세 실패 구간 대안 신호" 가설 반증. `residual_reversion@bull_quiet`(R0) 1건만 기준 통과(4개 fold 중 3개 확실, 1개 약함) → Phase 2 좁게 GO.
2. **Phase 2**(스코프·방향 확정, 구현 착수 가능) — `residual_reversion`(beta_neut archetype) regime 게이트를 **하드 마스킹**(`mean_rev_gating_enabled` 패턴 복제)으로 활성화. `_allowed_regimes_for_archetype`/`_attach_signal_context`/`CandidateStrategyConfig` 3곳 수정, opt-in 플래그로 기본 무회귀. sizing-multiplier 대안은 증거 형태(이산적 통과/실패) 불일치 + blast radius(전역 배수) 이유로 기각 확정.
3. Phase 2 구현 후 반드시 **outer-fold 재검증**(FDR/SPA 포함 정식 promotion 파이프라인)으로 단일-셀 multiplicity 위험 해소 확인. n=4인 소표본이므로 특히 신중히.
4. **Phase 1 blocking 종속성 해소**(연율화 파라미터화 또는 1h/1d allowlist 배제) — **이것 없이 Phase 1 착수 금지**. `next.md` P1②(RC-2)와 같은 파일(`risk_deployment.py`/`deployment.py`)을 건드리므로 통합 검토 권장.
5. **Phase 1**(TF 확장) — 4단계 완료 후 착수. `2h`부터 파일럿, `1h`/`1d`는 순차.
6. **Phase 3**(bookDepth 스키마 실증 + 커버리지 실측 + Capacity Clip 중복 확인) — 선행 조사 후 Go/No-Go, 독립적으로 병행 가능.
7. **Phase 4(신규)**(포트폴리오 diagonal Kelly 상관관계 미반영) — **코드 변경 아닌 조사부터**: `diagonal_kelly_weights` 채택 배경을 git log/decisions에서 먼저 확인. 기록 없으면 저비용 진단(실현 vol vs diagonal 가정 vol 괴리 측정)으로 후속 여부 결정. 다른 Phase와 완전 독립, 우선순위 최하위(사용자 원 요청 범위 밖 발견).
