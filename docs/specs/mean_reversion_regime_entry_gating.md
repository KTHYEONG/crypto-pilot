# Objective

Mean-reversion 계열 신호가 **추세장(trending regime)에서 진입하는 구조적 오류**를 차단해, ML-Ready gate 통과 품질을 높이고 레짐별 edge 부호 분화(C3 flip)의 전제 조건을 만든다. Threshold 완화 없이 진입 semantics만 정정한다.

# Strategy

## 진단 (rising-edge refactor 후 잔존 병목)

Signal Rising-Edge Refactor(2026-06-08) 결과:
- raw events 108k→73k, rule_promo mean edge 16.2→29.0 bps, ML-Ready 3→5, C4 rho 0.029→0.314.
- 그러나 `bollinger_20`, `vrr_20`, `vrr_40`은 여전히 BLOCKED (Event Overload / Poor Hit-Payoff).
- C3 gold flip=N 불변: ML-Ready 5개가 전부 방향성(추세/모멘텀) → 모든 레짐에서 동방향 수익.

## 근본 원인 (코드로 확인)

`rule_signals.py:286` — regime entry gating이 전역 비활성:

```python
if cfg.regime_signal_gating_enabled:   # config.py:204 = False
    allowed_mask = np.isin(regime_names[regime_ctx.code_1d], allowed_regimes)
    side_hint_2d[~allowed_mask, :] = 0
```

`regime_signal_gating_enabled=False`는 **배분(allocation) gating**을 끄기 위한 설정이다 (C4 rho<0.5라 배분 근거 부족). 그러나 이로 인해 **진입 유효성(entry validity) gating**까지 함께 꺼져 있다.

→ `bollinger_reversion`, `vol_regime_reversion` (archetype=`mean_reversion`)이 `bull_volatile`, `bear_volatile`, `crash` 레짐에서도 진입한다. 추세장에서 평균회귀는 구조적으로 실패(가격이 계속 추세 추종)하므로 hit_rate≈42%, payoff≈1.1로 gate 미달.

## 핵심 구분: 진입 유효성 ≠ 배분

| 구분 | 질문 | C4 rho≥0.5 필요? |
|---|---|---|
| **진입 유효성 gating** | "지금 mean-reversion 진입이 경제적으로 타당한가" | ❌ 불필요 (경제 로직) |
| **배분 gating** | "어느 레짐에 얼마를 배분할 것인가" | ✅ 필요 (통계 신뢰도) |

"추세장에서 mean-reversion 진입 금지"는 C4 rho가 0이어도 경제적으로 타당하다. 이 둘을 코드에서 분리한다.

# Cold Review (결과 끼워맞추기 방지)

- 이 작업은 "bollinger/vrr를 통과시키기 위한" 것이 **아니다**. gating 후 이벤트가 min_obs 미달로 떨어져 INSUFFICIENT_OBS가 되어도 수용한다 — 나쁜 진입을 제거하는 것 자체가 목적이다.
- 통과 개수 목표 없음. flip=Y 강제 없음. 측정 결과가 flip=N으로 유지돼도 "추세장 mean-reversion 차단"은 독립적으로 타당하다.
- threshold(z=2.0, vol_z, hit_rate=0.50 등) 일절 변경 금지.
- 전역 `regime_signal_gating_enabled=True`로 켜지 않는다 (추세 신호까지 영향받음). archetype-selective로 적용한다.

# Current Evidence

Source: `docs/results/result.md` (rising-edge refactor 후)

```text
RECOMMENDED: 5 (dm_24_96, dm_12_48, tpc_50_200, fzs_96, tpc_20_100)
BLOCKED: 27 — Top Blocked: vrr_20, bollinger_20, vrr_40, fzs_168, rr_24

bollinger_20: 16.5 bps, 41.7% hit, P/L 0.95  → DROP
vrr_40:       10.8 bps, 42.4% hit, P/L 1.16  → DROP

C3 gold: kw_p=0.0000, flip=N
C4 gold: rho=0.314 (events=7370)
```

# Target Files

- `src/domain/futures/strategy/rule_signals.py`
- `src/domain/futures/strategy/config.py`
- `tests/unit/domain/futures/strategy/test_rule_signals.py`
- `docs/results/result.md` (rerun 후에만)

Do not modify:
- `src/domain/futures/strategy/regime_evaluation.py`
- `src/domain/futures/strategy/market_regime.py`
- signal thresholds / recommendation thresholds / cost stress / deployment gates
- 추세 archetype(`trend_continuation`, `time_series_momentum`)의 진입 로직

# Contract Changes

## 1. `config.py` — archetype-selective entry gating 플래그

기존 `regime_signal_gating_enabled`(배분용, False 유지)와 **독립된** 진입 유효성 게이트를 추가:

```python
# 진입 유효성 게이팅: archetype별로 경제적으로 부적합한 레짐 진입을 차단.
# regime_signal_gating_enabled(배분 게이팅)와 독립. C4 rho 신뢰도와 무관하게 적용 가능.
mean_reversion_regime_entry_gating_enabled: bool = True
```

기본값 `True` 근거: 추세장 mean-reversion 진입 차단은 경제적으로 항상 타당하며 배분 통계와 무관하다.

## 2. `rule_signals.py` — `_allowed_regimes_for_archetype` 의미 명확화

현재 (L236-243):
```python
def _allowed_regimes_for_archetype(archetype: str) -> tuple[str, ...]:
    if archetype in {"trend_continuation", "time_series_momentum"}:
        return ("bull_quiet", "bull_volatile", "bear_quiet", "bear_volatile")
    if archetype in {"forced_flow_reversal", "position_unwind"}:
        return ("bull_volatile", "bear_volatile", "crash")
    if archetype == "carry_reversion":
        return ("bull_quiet", "bear_quiet", "transition")
    return ("bull_quiet", "bear_quiet", "transition")  # mean_reversion default
```

`mean_reversion` default가 이미 `(bull_quiet, bear_quiet, transition)`으로 추세장(volatile/crash)을 제외하고 있다 — **로직은 이미 올바르나 호출 경로(L286)가 비활성**이다.

# Surgical Plan

## 1. `config.py`

ACTION: UPDATE — `CandidateStrategyConfig`에 플래그 1개 추가

```python
mean_reversion_regime_entry_gating_enabled: bool = True
```

위치: `regime_signal_gating_enabled` 인접. `__post_init__` 검증 불필요(bool).

## 2. `rule_signals.py` — `_attach_signal_context` 게이팅 분기 추가

ACTION: UPDATE — L286 분기를 archetype-selective로 확장

현재:
```python
side_hint_2d = np.asarray(panel.side_hint_2d, dtype=np.int8).copy()
if cfg.regime_signal_gating_enabled:
    allowed_mask = np.isin(regime_names[regime_ctx.code_1d], np.asarray(allowed_regimes, dtype=object))
    side_hint_2d[~allowed_mask, :] = 0
```

교체:
```python
side_hint_2d = np.asarray(panel.side_hint_2d, dtype=np.int8).copy()
# 진입 유효성 게이팅: mean_reversion 계열은 추세장(volatile/crash) 진입이
# 구조적으로 부적합 → allowed_regimes 밖 진입을 차단. 배분 게이팅과 독립.
_entry_gated = cfg.regime_signal_gating_enabled or (
    cfg.mean_reversion_regime_entry_gating_enabled and archetype == "mean_reversion"
)
if _entry_gated:
    allowed_mask = np.isin(regime_names[regime_ctx.code_1d], np.asarray(allowed_regimes, dtype=object))
    side_hint_2d[~allowed_mask, :] = 0
```

설계 노트:
- 추세 archetype(`trend_continuation`, `time_series_momentum`)은 `mean_reversion`이 아니므로 영향 없음 — dm/tpc 동작 불변.
- `regime_signal_gating_enabled=True`로 켜면 기존처럼 전 archetype 게이팅(상위 호환).
- `regime_code_1d`는 `[t-1]` 소비(causal) — 기존 경로 그대로라 look-ahead 신규 유입 없음.

## 3. `test_rule_signals.py`

ACTION: UPDATE

추가 테스트:
- `test_mean_reversion_gated_out_of_trending_regime`: mean_reversion 패널이 volatile/crash 레짐 bar에서 side_hint=0
- `test_trend_archetype_not_affected_by_mean_reversion_gating`: trend/momentum 패널은 게이팅 플래그와 무관하게 모든 레짐 진입 유지
- `test_mean_reversion_gating_preserves_allowed_regime_entries`: bull_quiet/bear_quiet/transition 레짐 진입은 보존
- `test_gating_disabled_flag_restores_ungated_behavior`: `mean_reversion_regime_entry_gating_enabled=False`면 게이팅 미적용

assertion 금지: "ML-Ready 증가", "bollinger 통과" 등 결과 예측.

## 4. `docs/results/result.md`

ACTION: rerun 후에만 사실 갱신 (event count, BLOCKED, ML-Ready, C3/C4). 개선 주장은 실측 후에만.

# Verification

L1:
```bash
uv run ruff check --fix src/domain/futures/strategy/rule_signals.py src/domain/futures/strategy/config.py tests/unit/domain/futures/strategy/test_rule_signals.py
uv run mypy src/domain/futures/strategy/rule_signals.py src/domain/futures/strategy/config.py
uv run pytest tests/unit/domain/futures/strategy/test_rule_signals.py --tb=short
```

Regression:
```bash
uv run pytest tests/unit/domain/futures/strategy/ --tb=short
```

Signal-mode rerun:
```bash
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. timeout 300 uv run python src/execution/opt_main_futures.py --phase signal --sync skip --timeframe 4h --date 2026-05-01
```

# Success Criteria

- mean_reversion 패널이 volatile/crash 레짐에서 진입하지 않는다 (테스트로 증명)
- trend/momentum 패널 동작 불변 (dm/tpc 이벤트 수·side 동일)
- look-ahead 신규 유입 없음 (`[t-1]` 소비 유지)
- threshold 일절 미변경
- bollinger/vrr 이벤트 수 감소는 정상 — 통과 여부는 성공 기준이 아님
- (관찰 지표) rerun 후 C3 flip / C4 rho 재측정 — flip=Y 전환 여부는 강제하지 않고 기록만

# Non-Goals

- 전역 `regime_signal_gating_enabled=True` 전환 (배분 게이팅은 C4 rho≥0.5 달성 후 별도 진행)
- threshold 완화로 bollinger/vrr 구제
- 신규 signal family/variant 추가
- regime-conditional prior/배분 설계 (C4 rho≥0.5 선결)
- flip=Y / 특정 ML-Ready 개수 목표화
