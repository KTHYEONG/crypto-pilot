# MHS Execution Data Coverage Gate & Data-Integrity/Alpha Reason Split

## 1. Problem (Empirical Evidence)

`docs/results/mhs_run_history/active.jsonl` 3건 비교:
- run0(03:34)/run1(07:42, `committee_capital` 미적용): fold1(2024)만 `PRIMARY_AUTOCORR_SHARPE_BELOW_0_6`/`STRESS_SHARPE_NOT_POSITIVE`로 실패 (순수 알파 품질 실패).
- run2(09:05, commit `35e85649` "committee_capital 폴드 측정 경로" 적용 후): fold2(2025-01-08~2025-12-31 validation)가 처음으로 `RELEVANT_EXECUTION_DATA_GAP`으로 실패 — `evaluation.py:3914-3918`의 `primary.termination_counts["MISSING_DATA"] > 0` 게이트.

`committee_capital=True`가 fold 자본을 실제 5m `simulated_inventory_ledger`로 흘려보내면서, 이전엔 측정 경로에 잡히지 않던 **기존 5m 캐시 결손이 처음 노출**됐다 (신규 버그가 아니라 신규 계측).

로컬 캐시 직접 대조(`data/futures/ohlcv/{5m,1m}`):
- 5m 255개 심볼 vs 1m 472개 심볼. **246개 심볼이 1m엔 있으나 5m엔 전혀 없음** (`1000SHIBUSDT`, `AEROUSDT`, `0GUSDT` 등 — 모두 funding 데이터 보유, `MHS_SOURCE_GAP_EXCLUDED_SYMBOLS`에도 없어 `funded` 유니버스에 정상 편입됨).
- 겹치는 226개 심볼은 전부 정확히 `HOLDOUT_CUTOFF`(`2025-12-31 23:59:59`, `src/research/evaluation/policy.py:8`)까지 채워져 있음 — staleness 문제 아님, 순수 커버리지(존재 여부) 문제.

`_coverage()`(`src/application/data/mhs_execution_collection.py:106`)는 `observed`의 min~max 구간 내부 홀만 `missing_internal_bars`로 잡고, 파일 자체가 없거나 구간 전체가 비어있는 경우만 `"MISSING"`으로 본다 — 이는 이미 정확하다(트레일링 결손 오탐 없음). 문제는 **이 커버리지 판정이 진단 실행 전에 한 번도 호출되지 않는다**는 것: `run_mhs_horizon_diagnostic`은 `funded` 유니버스를 확정한 뒤(~520초) 전체 리플레이를 끝까지 돌리고 나서야 `MISSING_DATA` 카운트로 결손을 사후 발견한다.

`build_mhs_execution_plan`/`collect_mhs_execution_data`/`refresh_mhs_execution_manifest`(`mhs_execution_collection.py`)로 PIT 심볼 유니버스 산출·백필·로컬 재검증 인프라는 이미 구현되어 있으나, 진단 실행과 연결되어 있지 않다.

## 2. Root Cause

`[mhs_execution_collection.py] -> [진단 실행 전(pre-flight) 5m 커버리지 검증 미수행]`: 246개 funded 심볼의 5m 캐시 완전 결손이 리플레이 도중(수백 초 경과 후) `MISSING_DATA` termination으로만 드러나, 원인 심볼을 알 수 없는 채로 fold 전체가 fail-closed 처리됨.

부수 문제: `MhsResearchGoResult.reason_codes`가 데이터 결손성 실패(`RELEVANT_EXECUTION_DATA_GAP` 등)와 순수 알파 성과 실패(`PRIMARY_AUTOCORR_SHARPE_BELOW_0_6` 등)를 구분 없이 flat list로 섞어, "committee_capital 같은 측정 경로 변경이 성과를 바꿨다"와 "데이터 결손이 드러났다"를 리포트만 봐서는 혼동하기 쉽다.

## 3. Solution

### 3.1 Pre-flight coverage gate (fail-closed, actionable)
`mhs_execution_collection.py`에 `assert_execution_data_coverage(symbols, timeframe, start, end, root=None)` 추가 — 기존 `_coverage()`를 심볼별로 재사용해 `status != "PRESENT"`인 심볼을 모아 `DataIntegrityError`로 **결손 심볼 목록과 상태를 그대로 노출**하며 즉시 중단시킨다. `_coverage()`에 `root: str | None = None` 파라미터를 추가해(기본값은 기존 `FUTURES_DATA_DIR / "ohlcv"`와 동일 — 하위호환) 테스트의 합성 `data_root`와 연결한다.

`run_mhs_horizon_diagnostic`은 `funded` 유니버스가 확정된 직후(`evaluation.py` L4213 부근, 리플레이 시작 전) 신규 opt-in 플래그 `MhsDiagnosticRequest.execution_coverage_gate`(기본 `False`, 이 저장소의 기존 옵트인 플래그 컨벤션과 동일)가 `True`일 때만 이 게이트를 호출한다. 기본값 `False`는 기존 모든 호출부/유닛 테스트를 byte-identical로 유지하기 위함이며, 실사용 시(`--execution-coverage-gate`, 특히 `--committee-capital`과 함께) 활성화를 권장한다.

**Why opt-in, not always-on**: 이 저장소는 `touch_diagnostic`/`ladder_diagnostic`/`multi_feature_book`/`committee_book`/`committee_capital` 등 모든 신규 진단 기능을 옵트인·기본 byte-identical로 도입해왔다(§CLAUDE.md Prefer Minimal Change). 항상 켜는 것으로 바꾸려면 기존 통과 조합들에 대한 회귀 확인이 별도로 필요하므로 이번 스펙 범위에서는 제외하고, 검증 후 CLI 기본값 전환은 후속 ADR로 분리한다.

### 3.2 Data-integrity vs alpha-quality reason split
`evaluation.py`에 `MHS_GO_REASON_DATA_INTEGRITY_CODES` frozenset 상수 추가(`{INCOMPLETE_ANCHORED_FOLD, INVALID_PRIMARY_LEDGER, NONFINITE_EQUITY, RELEVANT_EXECUTION_DATA_GAP, CAPITAL_INVARIANT_BREACH, RESOURCE_BUDGET_BREACH}` — 이미 정의된 `MHS_GO_REASON_*` 상수만 참조, 신규 매직 문자열 없음). `MhsResearchGoResult`에 파생 필드 `data_integrity_reason_codes: tuple[str, ...] = ()` 추가, `_mhs_research_go()`가 최종 `reasons`에서 이 집합과 교집합을 계산해 채운다. `PRIMARY_AUTOCORR_SHARPE_BELOW_0_6`/`STRESS_SHARPE_NOT_POSITIVE`(알파 품질)와 `UNSPECIFIED_POLICY`(정책 미등록)는 제외되므로, 리포트 소비자는 `research_go.data_integrity_reason_codes`가 비어있는지만 봐도 "데이터는 정상, 알파가 부족했다" vs "데이터 자체가 결손이었다"를 즉시 구분한다. `build_mhs_run_history_record`의 `research_go` dict에 동일 키를 추가해 `active.jsonl`/`latest.json`에도 반영한다.

### 3.3 Scope-out: 3m timeframe / 5m→1m 전환
- `execution_timeframe`은 코드 계약상 `Literal["1m", "5m"]`만 지원(`evaluation.py:317,347`, `mhs_execution_collection.py:58`). 3m 데이터 수집 파이프라인 자체가 없어 도입하려면 신규 수집 인프라 구축이 필요한 별개 스코프 — 이번 스펙에서 제외.
- 1m 캐시가 5m보다 넓은 것은 5m 백필이 방치된 **운영 결손**이지 5m의 구조적 열등성이 아니다(§1 근거). 진단 리포트의 `participation_warnings`(~1e-9, 무시 가능한 시장충격)를 볼 때 체결 그리드를 1m으로 세분화해도 체결 현실성 이득은 미미한 반면, 바 수 5배 증가로 `ram_guard`/RSS budget 인프라가 존재하는 이유인 연산·메모리 비용만 커진다. **5m→1m 전면 전환은 근거 부족으로 비권장**; 올바른 해법은 §3.1 게이트로 결손을 드러낸 뒤 §3.4 백필로 5m 커버리지를 1m 수준으로 맞추는 것이다.

### 3.4 Backfill (operational, out of contract)
게이트가 드러낸 246개 결손 심볼은 기존 `build_mhs_execution_plan(start, end, "5m", 30)` → `collect_mhs_execution_data(plan, execute=True)`로 백필 가능(신규 코드 불필요). 이는 코드 변경이 아닌 1회성 운영 작업이므로 이번 구현 계약에는 포함하지 않고, 스펙 완료 후 별도로 실행한다.

## 4. Non-goals
- 3m 타임프레임 신규 도입 (§3.3)
- 5m→1m 전면 전환 (§3.3)
- 246개 심볼 백필 스크립트 실행 자체 (§3.4, 운영 작업)
- `execution_coverage_gate` 기본값 `True` 전환 (§3.1, 후속 ADR)
