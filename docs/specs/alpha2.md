---
title: Alpha 게이트 정직화 + OOS 평가 파이프라인 회귀 수정
domain: futures-alpha
type: spec-lite
status: active
priority: critical
ai_read_policy: when_related
last_verified: 2026-06-01
related_paths:
  - src/domain/futures/strategy/alpha_evaluation.py
  - src/execution/opt_main_futures.py
  - src/domain/futures/forecast/compose.py
  - src/domain/futures/strategy/ml_builder.py
dependencies:
  documents: [docs/specs/alpha0.md, docs/specs/alpha1.md, docs/results/re-alpha.md]
change_triggers:
  - src/execution/opt_main_futures.py
  - src/domain/futures/strategy/alpha_evaluation.py
---

# Alpha 게이트 정직화 + OOS 평가 파이프라인 회귀 수정

## 0. 현재 상태 요약

### 완료된 게이트 정직화 (2026-06-01)

| 변경 | 코드 위치 | 상태 |
|---|---|---|
| clip_pres 임계 0.5 → 0.7 | `opt_main_futures.py:145` | ✅ 적용됨 |
| G2 breakeven → N_eff 기준 | `opt_main_futures.py:132` | ✅ 적용됨 |
| DSR 기준 pre-clip → post-clip | `alpha_evaluation.py:699` | ✅ 적용됨 |
| COST_GATE_AMORTIZE 복원 | `opt_config.py:195` | ✅ True 유지 |

### 미해결 블로커: OOS 평가 파이프라인 회귀

```
smoke run 결과 (--mode alpha --sync-mode skip --trials 1 --tf 4h --reference-date 2026-05-01):
  ALPHA_PASS: FALSE
  basket n=0, resid_ic=0.0000, NET_IC=0.0000

git stash 후 동일 명령 재실행 → 동일 결과 확인
→ 내 게이트 변경이 아닌 선행 파이프라인 회귀
```

---

## 1. OOS 평가 회귀 근본 원인 분석

### 데이터 흐름 (코드 검증)

```
pick_strategy_data_maps()       # IS-only data (up to 2025-10-01)
    ↓
build_ml_strategy_alpha()       # features.datetimes = IS dates only
    ↓
score_grid = NaN everywhere     # ml_builder:912
    ↓
Fill only fold test bars        # ml_builder:1087-1099 (rank_score_long_grid)
    ↓
alpha_panel span 2022-10 ~ 2025-09-30  # IS fold test coverage
    ↓
opt_main:935-943: common_idx 추출 경로
    _raw_rs_long = pivot(rank_score_long).reindex(columns=common_syms)
    _oos_dt_mask = any(isfinite(_raw_rs_long), axis=1)
    _oos_idx = dates[_oos_dt_mask]
    common_idx = _oos_idx ∩ realized_df.index
```

### 핵심 가설 (우선순위 순)

**H1 (HIGH): C3 trading 심볼이 rank_score_long 커버리지 밖**
- `common_syms` = C3 trading symbols (20개)
- `rank_score_long.reindex(columns=common_syms)` → 20개 심볼에 대해 전부 NaN
- `_oos_dt_mask` = all False → `_oos_idx` = empty → `common_idx` = empty
- 증거: basket n=0, `[RANK-QUALITY L1] breadth=0.0`, SCORE-IC는 정상(3.7 breadth)

**H2 (MED): Phase 3 commit `7fa05eb` 이후 심볼 필터링 변경**
- `7fa05eb feat(futures): Phase 3 Idiosyncratic Redesign` 이후 inference symbols가
  C3 trading symbols와 다른 경로로 처리될 수 있음
- `pick_strategy_data_maps`가 `full_strategy_maps` 내 trading symbols를 올바르게 포함하지
  않을 경우 rank_score_long의 심볼 커버리지 불일치 발생

**H3 (LOW): 타임존 불일치**
- `panel_reset["datetime"]`: `pd.to_datetime(utc=True).dt.tz_localize(None)`
- `realized_df.index`: `pivot_long.index` (동일 변환)
- 동일 경로이므로 불일치 가능성 낮음

### 진단 방법 (opt_main:935-943 직후 로그 추가)

```python
_logger.info(
    "[OOS-DIAG] raw_rs_long_syms=%d finite_rows=%d oos_idx_len=%d common_idx_len=%d",
    len(_raw_rs_long.columns),
    int(_oos_dt_mask.sum()),
    len(_oos_idx),
    len(common_idx),
)
_logger.info("[OOS-DIAG] common_syms=%s", list(common_syms)[:5])
_logger.info("[OOS-DIAG] rank_score_cols=%s", list(_raw_rs_long.columns)[:5])
```

---

## 2. 수정 계획 (Step-by-Step)

### Step A: 진단 로그 추가 및 H1 검증 (1h)

**목표:** `_oos_idx` empty 원인 특정 — C3 심볼 커버리지 vs 타임존 중 어느 것인지.

**파일:** `src/execution/opt_main_futures.py`

**위치:** `opt_main_futures.py:943` (`common_idx = ...` 바로 다음)

```python
# AS-IS (현재):
common_idx = _oos_idx.intersection(realized_df.index)
al = pivot_long.loc[common_idx].to_numpy(dtype=np.float64)

# TO-BE (진단 로그 추가):
common_idx = _oos_idx.intersection(realized_df.index)
_logger.info(
    "[OOS-DIAG] rank_cols=%d finite_rows=%d oos_idx=%d common_idx=%d | "
    "common_syms[:3]=%s rank_score_cols[:3]=%s",
    len(_raw_rs_long.columns),
    int(_oos_dt_mask.sum()),
    len(_oos_idx),
    len(common_idx),
    list(common_syms)[:3],
    list(panel_reset["symbol"].unique())[:3],
)
al = pivot_long.loc[common_idx].to_numpy(dtype=np.float64)
```

**검증 명령:**
```bash
PYTHONPATH=. uv run python src/execution/opt_main_futures.py \
  --mode alpha --sync-mode skip --trials 1 --tf 4h \
  --reference-date 2026-05-01 2>&1 | grep "OOS-DIAG"
```

**예상 결과 분기:**
- `finite_rows=0` → H1 확정: rank_score_long에 C3 심볼 없음 → Step B-1 진행
- `finite_rows>0, common_idx=0` → H3 확정: 타임존 불일치 → Step B-2 진행
- `common_idx>0` → 다른 원인 → Step B-3 진행

---

### Step B-1: rank_score_long C3 커버리지 수정 (H1 확정 시)

**원인:** `_raw_rs_long.reindex(columns=common_syms)`에서 C3 trading 심볼이 alpha_panel의 symbol 컬럼과 불일치.

**수정 방향 A (권장):** `_oos_dt_mask` 를 C3 필터 없이 전체 심볼로 계산

```python
# AS-IS (문제):
_raw_rs_long = (
    panel_reset.pivot(index="datetime", columns="symbol", values="rank_score_long")
    .reindex(columns=common_syms)  # ← C3 필터가 빈 교집합 유발
)
_oos_dt_mask = np.any(np.isfinite(_raw_rs_long.to_numpy()), axis=1)

# TO-BE (수정):
_raw_rs_all = (
    panel_reset.pivot(index="datetime", columns="symbol", values="rank_score_long")
    # 전체 심볼로 finite 날짜 감지 (C3 교집합 전에 시간 인덱스 확보)
)
_oos_dt_mask = np.any(np.isfinite(_raw_rs_all.to_numpy()), axis=1)
_oos_idx = _raw_rs_all.index[_oos_dt_mask]
# C3 필터는 al/as_ 추출 시만 적용 (pivot_long이 이미 common_syms로 필터됨)
```

**수정 방향 B (대안):** alpha_panel 빌드 시 C3 심볼을 명시적으로 포함

```python
# ml_builder: build_ml_strategy_alpha에 trading_symbols 파라미터 전달 보장
# + rank_score_long_grid를 trading_symbols에 대해서도 채우도록
```

방향 A가 더 간단하며 alpha_panel 구조 변경 없이 수정 가능.

---

### Step B-2: 타임존 불일치 수정 (H3 확정 시)

```python
# AS-IS:
panel_reset["datetime"] = pd.to_datetime(panel_reset["datetime"], utc=True).dt.tz_localize(None)

# TO-BE: tz_localize(None) → tz_convert("UTC").dt.tz_localize(None)
# 또는: 명시적 floor to microsecond resolution
panel_reset["datetime"] = (
    pd.to_datetime(panel_reset["datetime"], utc=True)
    .dt.tz_convert(None)  # tz_localize(None) 대신 tz_convert(None) 사용
)
```

---

### Step C: 수정 후 smoke 검증

```bash
PYTHONPATH=. uv run python src/execution/opt_main_futures.py \
  --mode alpha --sync-mode skip --trials 1 --tf 4h \
  --reference-date 2026-05-01 2>&1 | \
  grep -E "OOS-DIAG|NET_IC|basket|ALPHA_PASS|DSR"
```

**복원 기준 (re-alpha.md 대비):**

| 지표 | 목표 | re-alpha.md 실측 |
|---|---|---|
| basket n | > 200 | 254 |
| NET_IC | > 0.025 | 0.0380 |
| T-STAT | > 2.0 | 2.45 |
| DSR | < 0.99 (포화 방지) | 1.0000 (이전 포화) |
| ALPHA_PASS | TRUE | TRUE |

> **DSR 목표 변경 주의:** 게이트 정직화(변경 3)로 DSR은 post-clip 기준 계산됨.
> IC=0.038, t=2.45 신호의 정직한 DSR은 0.7~0.9 범위 예상 (1.0 아님).
> 이전 ALPHA_PASS=TRUE가 DSR=1.0000 포화 덕분이었다면, 수정 후 ALPHA_PASS는
> DSR이 낮아져 결과가 달라질 수 있음 — 이는 "버그"가 아니라 "정직화"의 결과.

---

## 3. 중장기 로드맵 (OOS 회귀 수정 완료 후)

### 3-A. OOS 12개월 AWF 확장

**현재 구조 (코드 검증):**
- IS=24개월, OOS=6개월 (get_quarterly_window)
- AWF: anchored 5-leg, IS_POOL_FRAC=0.65

**변경 방향 (단일 holdout 확장 금지 — 데이터 낭비):**
```python
# opt_config.py (조정 대상)
"FUTURES_AWF_K_LEGS": 5,          # 유지
"FUTURES_AWF_IS_POOL_FRAC": 0.70,  # 0.65 → 0.70 (OOS pool 30% 확보)
# get_quarterly_window: oos=6 → oos=12 months (OOS pool 확장)
```

**ML fold 노이즈 감소:**
- 현재: fold test ~3개월 × 2 folds = 6개월
- 개선: `FUTURES_INFERENCE_MIN_HISTORY_MONTHS` 33 → 39로 상향 (fold 3개 가능)

### 3-B. 포트폴리오/리스크 배선 audit

기존 portfolio/ 스택(Kelly, LW covariance, 5-cap)은 이미 active.
`docs/architecture/portfolio_risk_architecture.md` §4 갭 항목만 검증:

1. `compute_drawdown_gross_scale` dead 중복 제거
2. `DYNAMIC_RA_BEAR_COEF` → `dyn_leverage` 경로 1-line 확인
3. COST_GATE_AMORTIZE 정합 재검토 (alpha unit 해명 후)

---

## 4. 즉시 실행 체크리스트

```
[ ] Step A: 진단 로그 추가 + smoke run → H1/H2/H3 특정
[ ] Step B (진단 결과에 따라): rank_score_long 커버리지 수정
[ ] Step C: smoke 재실행 → basket n>0, NET_IC>0 확인
[ ] 단위테스트: uv run pytest tests/unit/domain/futures/ -q (580 passed 유지)
[ ] 문서: last_verified 업데이트 + re-alpha.md에 최신 결과 기록
```

---

## 5. 코드 레퍼런스

| 지점 | 위치 | 설명 |
|---|---|---|
| 게이트 verdict | `opt_main_futures.py:117-228` | `_summarize_alpha_phase1_verdict` |
| OOS index 추출 | `opt_main_futures.py:935-943` | `common_idx` 계산 핵심 경로 |
| DSR 계산 | `alpha_evaluation.py:699` | `pred_2d` 기준 (수정됨) |
| rank_score 채우기 | `ml_builder.py:1086-1099` | fold test 루프 |
| IS-only data 선택 | `strategy_service.py:21-43` | `pick_strategy_data_maps` |
| clip_pres 임계 | `opt_main_futures.py:145` | 0.7 (수정됨) |
| breakeven G2 | `opt_main_futures.py:132` | `breakeven_ic_eff` (수정됨) |
