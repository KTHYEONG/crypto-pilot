# L0/L1 버그 발견 및 수정 결과 (2026-07-15)

## 측정 범위

- **실행일**: 2026-07-15
- **실행군**: `control` 단일 run (`scripts/run_l1_cross_tf_replay.py control`)
- **기간**: 2023-07-31 ~ 2026-03-31 | IS/OOS split: 2025-10-01
- **Universe**: Pool 377 → Selected 150 → Loaded 106

## 1. 진단 파이프라인 무결성 버그 (수정 완료)

### 증상
`docs/results/result.md`(구버전) artifact가 TF별 판정 결과를 신뢰할 수 없는 상태였음 — `RunnerResult`가 폐기되어 프로세스 exit code가 항상 0, canonical 10-stage 중 `terminal_event_audit`/`outer_folds` 2개 누락.

### 근본 원인
- `scripts/run_l1_cross_tf_replay.py::run_once()`가 `run_pipeline()`의 `RunnerResult`를 변수에 담지 않고 버림.
- `trace` dict가 canonical stage 8/10만 캡처.
- `cross_tf_diagnostics.py`에 정식 진단 계약(`diagnose_snapshots`/`write_cross_tf_diagnosis`)이 이미 구현되어 있었으나 어떤 프로덕션 코드도 호출하지 않는 고아 코드.
- `run_tiered_pipeline_outcome()`의 `diagnostic_sink` 파라미터는 정의만 되고 미사용, 호출부 자체가 레포 전체에 0개.
- `RunnerResult`가 `models.py`/`active_pipeline.py`에 이중 정의되어 매 호출마다 상호 변환.

### 수정
`run_once()`가 caller-owned trace를 참조로 받아 RunnerResult + 10-stage 전체 기록, `main()`이 `RunnerResult.exit_code`를 프로세스 exit code로 반영, 예외 시에도 partial trace 보존. `STAGE_ORDER` 공개 상수로 SSOT화. 신규 `scripts/run_l1_cross_tf_diagnosis.py` — 4-run(`control`/`control_repeat`/`treatment`/`fusion_ablation`) 순차 supervisor가 `diagnose_snapshots()`에 실제로 연결. 미사용 `diagnostic_sink` 파라미터 제거. `RunnerResult` 이중정의를 `models.py` 단일 클래스로 통합.

### 실측 검증
control 재실행 결과 artifact에 `runner_result` + 10/10 stage 전부 기록됨. 2h `n_valid=74`, fold edge(160.48/191.68/108.26/61.52bps) 등 수치는 기존 측정과 완전 일치(회귀 없음).

## 2. L1 pair 자격 게이트 TF-밀도 역방향 임계값 버그 (수정 완료)

### 증상
4h/6h/8h/12h/1d가 첫 fold 이후 전부 `registry_empty`로 BLOCKED — 과거 조사에서는 "진짜 시장 비정상성"으로만 판정되어 있었음.

### 근본 원인 (실행 계측 3회로 확정)
- `evaluate_outer_signal_opportunities` → `QualifiedSignalRegistry.ready_symbols`가 비어서 발생.
- 실측: `n_evidence_out`(evidence row 수, 590~808)은 2h(601~604)와 비슷한 규모 — "데이터 부족" 아님.
- 실측(bootstrap 확률 분포): `probability_positive>0.5` pair 비율은 전 TF 동일(~0.44~0.53) — bootstrap 편향 아님.
- **결정적 원인**: `config.py`의 `_DEFAULT_PER_TF_GATE_OVERRIDES["l1_pair_min_effective_obs"]`가 TF 속도와 반대 방향(1h=3.0 → 2h=4.0 → 4h=5.0(누락 폴백) → 6h=5.0 → 8h=5.0 → 12h=6.0 → 1d=7.0)으로 설정됨. 느린 TF일수록 pair당 관측치(n_obs)는 자연히 줄어드는데(2h fold3 중앙값 100 vs 12h 21) 문턱값은 오히려 더 높게 요구되어 4h~1d가 구조적으로 거의 통과 불가능.

### 수정
`scripts/calibrate_l1_pair_gate.py` 신규 작성 — 측정(control replay + `effective_n_sink` 훅)과 채택(config.py 수동 반영)을 분리(quant.md anti-overfitting 원칙 준수). 구현 중 크래시 2건 발견/수정: (1) `run_once(trace={})` 계약 위반 크래시, (2) `pipeline.py`가 `compute_symbol_strategy_evidence`를 자체 재import하므로 `signal_selection` 모듈 패치는 무효(측정치 전부 빈 값으로 나왔던 원인) — `pipeline` 모듈 자신의 바인딩을 패치하도록 수정.

### 실측 결과
```
measured_effective_n_p10_by_tf: 2h=28.4, 4h=12.7, 6h=16.1, 8h=14.4, 12h=9.6, 1d=4.9
```
6개 TF 전부 ceiling(4.0, 기존 2h 값)을 초과 → 전부 4.0으로 수렴. `config.py` 갱신: 6h/8h/12h/1d `5.0/5.0/6.0/7.0` → `4.0`, 4h 누락 엔트리 신설(`4.0`). 어떤 TF도 이제 2h보다 엄격한 문턱값을 요구받지 않음.

## 3. L0 TF-probe majors-scope 축소 버그 (발견, 미수정)

### 증상
`[L0-PROBE] 0 winning cells across 0 tf` — 6개 TF 전부 t-stat 게이트에서 전멸.

### 근본 원인 (실행 계측으로 확정)
- `_TF_PROBE_FALLBACK_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT")`로 3-메이저 스코프를 의도했으나, 실제 `full_strategy_maps`(거래 유니버스)에 BTCUSDT/ETHUSDT가 **아예 존재하지 않음**(실측: `BTCUSDT_present=MISSING_FROM_MAPS`, `ETHUSDT_present=MISSING_FROM_MAPS`).
- `opt_config.py`의 `FUTURES_ANCHOR_SYMBOLS`/`FUTURES_MACRO_INDEX_SYMBOLS`에 BTC/ETH가 "레짐 벤치마크"용으로만 별도 수집되고, "Pool 377 → Selected 150 → Loaded 106" 거래 유니버스 선정 단계에서는 제외되는 것으로 보임(의도된 정책 추정).
- 결과: 534개 프로브 셀 전부가 BNBUSDT 단일 심볼 — "메이저 3종 교차검증"이 실질적으로 "BNB 단독 판정"으로 조용히 축소됨. `scan_timeframe_alpha`가 누락 심볼을 DEBUG 로그로만 남기고 조용히 스킵해서 지금까지 발견되지 않았던 것으로 추정.
- 셀 단위 데이터 자체는 NaN/degenerate 없이 정상 계산됨(계산 버그 아님) — 표본 구성의 문제.

### 상태
**미수정.** 후속 조사/spec 필요: `_TF_PROBE_FALLBACK_SYMBOLS`를 실제 거래 유니버스 내 유동성 상위 N종으로 동적 선정하거나, 요청 스코프 대비 실제 사용 가능 심볼 수를 INFO 레벨로 명시 경고하는 안 검토.

## 다음 단계

1. L0 TF-probe majors-scope 축소 버그 spec 작성 및 수정.
2. `scripts/run_l1_cross_tf_diagnosis.py`로 4-run(control/control_repeat/treatment/fusion_ablation) 전체 실행 — 1h 추가가 6h/12h에 미치는 영향 및 cross-TF divergence 판정(이번 세션 미실행, 4배 소요 예상).
3. 재보정된 `l1_pair_min_effective_obs=4.0` 적용 후 4h~1d가 실제로 registry_empty를 벗어나는지, economic edge(LCB/incremental bps)가 유의미한지 재검증.
