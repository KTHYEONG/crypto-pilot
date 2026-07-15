# L0/L1 Discovery Snapshot

- **최신 측정일**: `2026-07-15` (run `--phase l1 --timeframe 4h --sync skip`, `LOG_LEVEL=DEBUG`, post 게이트 TF-바이어스 교정 + TF-스케일 설정값 거버넌스 + 1h 재도입)
- 이 문서는 **현재 상태와 최신 관측 데이터만** 담는다. 과거 세션의 반복 로그는 `docs/decisions/decisions.md`/`decisions_archive.md`에 보존.
- **⚠️ 이 스냅샷은 이전(그리드-수정 직후) 버전을 완전히 대체한다.** 이전 스냅샷의 2h 압도적 결과는 부분적으로 계측 버그(부트스트랩 block 미스케일, sym_count 임계값 우회, probe_bps 손익분기 미반영)에 의한 과대평가였음이 이번 세션에 확정됨(§2). 버그 교정 후에도 2h는 마스터 TF로 남았으나, 그 우위 폭은 줄었다.

## 1. L0 → L1 TF별 배포 현황 (최종, 이번 세션 전체 수정 반영)

| TF | 구조 게이트 | L1 n_ready | 주요 블로커 | 비고 |
| --- | --- | ---: | --- | --- |
| **1h**(신규 재도입) | ❌ BLOCKED | 0 | `probe_lcb_bps:4.582`(<7.50 손익분기 미달) | Symbol-Breadth 20.8로 밀도상 여유 통과하나 경제성 게이트에서 탈락 — 밀도 착시 아님, 진짜 엣지 부재로 판정 |
| **2h** | ✅ PASSED | **77** | none | **master TF**. 새 임계값(Symbol-Breadth≥5.00, probe_lcb>7.50bps)에서도 여유 통과 |
| 4h | 구조 통과(advisory만 실패) | 7 | `fold_ratio:0.250` | `l1_structural_gate_only=True`로 배포는 됨, quality_weight만 25% 페널티 |
| 6h | ❌ BLOCKED(**이번 세션 중 회귀**) | 0 | `sym_count:2.273, probe_lcb_bps:-1.510, fold_ratio:0.250` | 직전엔 PASSED(n_ready=1)였으나 1h 재도입의 부작용으로 구조 게이트 실패로 전환(§2.3, 원인 미규명) |
| 8h | ❌ BLOCKED | 0 | `sym_count:2.000, probe_lcb_bps:-inf, match_ratio:0.855, fold_ratio:0.000` | 무변화 |
| 12h | ❌ BLOCKED | 0 | `probe_lcb_bps:-inf, match_ratio:0.863, fold_ratio:0.000` | sym_count는 1.0→**3.0(통과)**로 개선됐으나 probe_lcb_bps는 여전히 -inf |
| 1d | 구조 통과(advisory만 실패) | 12 | `fold_ratio:0.250` | 무변화(TF-스케일 버그의 왕복 검증으로 재확인 완료) |

- `would_resolve_master_tf`: 전 TF 공통 **`2h`** — 게이트 계측 버그를 걷어낸 뒤에도 유지.
- 배포 가능 TF: **3/7 유효 배포**(2h/4h/1d, 후2개는 25% advisory 페널티 상태). 1h/6h/8h/12h는 구조 게이트 실패.

## 2. 이번 세션 수정 내역 및 실측 검증

### 2.1 L1 게이트 TF-바이어스 교정 (`ADR_20260715_L1_TF_BIAS_GATE_CALIBRATION`)
2h 단독 압도(n_ready=103, probe_lcb_bps=108.2)의 원인 3가지를 코드 레벨에서 확인 후 교정:
1. `l1_bootstrap_block_bars=6`(bar-count 고정)이 TF/보유기간 미스케일 → `_resolve_block_bars_eff` 도입.
2. `l1_sym_count_mode="effective_n"`이 TF별 sym_count 오버라이드를 우회, 전 TF 공통 임계값 3.0만 적용 → `l1_min_effective_sym_n` per-TF 오버라이드(1h/2h=5.0) 추가.
3. `probe_lcb_bps` 구조 게이트가 손익분기(round-trip cost ≈7.5bps) 미반영, `>0.00`만 요구 → `max(l1_min_probe_bps, l1_breakeven_floor_bps)`로 교정.
- **실측**: 2h n_ready 103→101(교정만 적용 시), Symbol-Breadth/probe_lcb_bps 모두 새 기준에서도 여유 통과 확인.

### 2.2 TF-스케일 설정값 거버넌스 (`ADR_20260715_TF_SCALED_CONFIG_FIELD_GOVERNANCE`)
- 2.1 교정 중 `max_holding_bars`(4h 기준 36bar 상수)를 부트스트랩 스케일링에 재사용하다 **1d에서 재현 가능한 회귀 발견**(block 크기 6배 폭증 → n_ready 12→0, Symbol-Breadth 3.0→2.0, probe_lcb_bps 227→376).
- 전수 감사 결과 config.py 전역에 "base-TF 캘리브레이션 상수 vs TF-네이티브 값" 구분 컨벤션 부재가 15개+ 필드에 퍼져있음 확인.
- `dataclasses.field(metadata={"tf_scale_base": "4h"|None})` 컨벤션 도입, `apply_tf_gate_overrides`에 스케일링 로직 통합, 신규 필드 미분류 시 실패하는 구조 테스트 추가.
- **실측**: 1d n_ready 0→**12(완전 복원)**, 3회 반복 재현(byte-identical) 확인.

### 2.3 1h 재도입 + 미해결 이상 현상 (`ADR_20260715_L1_TF_COVERAGE_1H_REINTRO`)
- L0→L1 병목 재진단: L0는 8h(14개)/12h(12개) 레시피를 4h(11개)와 대등하게 통과시킴 — **universe/L0 admission 문제가 아님**. 병목은 L1 fold-level 매칭 게이트.
- `l1_min_matched_events_per_fold=20`(TF-불변 플랫 상수)을 용의자로 지목, TF-스케일 메타데이터 태깅 + 1h를 `DEFAULT_L1_TFS`에 재도입.
- **⚠️ check 단계에서 반증**: 1h를 뺀 격리 재실행 결과 6h/8h/12h가 스케일링 이전 원값과 **완전히 일치** — `l1_min_matched_events_per_fold` 스케일링은 **단독으로는 효과가 없었다.** 12h의 sym_count 개선(1.0→3.0)과 6h의 회귀(PASSED→BLOCKED)는 전부 **1h를 TF 목록에 추가한 것 자체의 부작용**이었음이 실측으로 확인됨. 정확한 인과 경로(cross-TF 공유 연산 의심, `tf_idx` 기반 seed 의존은 배제 확인)는 **미규명 상태로 다음 세션 과제**.
- 1h 자체의 결과: Symbol-Breadth 20.8로 밀도 덕에 손쉽게 통과하지만 probe_lcb_bps=4.58bps(<7.5bps 손익분기)로 최종 탈락 — **밀도 착시가 아니라 경제성 게이트가 독립적으로 정상 작동**한 사례로 판단(설계 세이프가드 검증됨).

## 3. TF별 상세 (fold 레벨, 2026-07-15 최종 재실행)

#### [TF 1H] - ❌ BLOCKED (신규 재도입, 손익분기 미달)
- Ready Folds: 2/4
- Cov: 1.000 (>=0.80) | Symbol-Breadth: 20.829 (>=5.00, 통과) | probe_lcb_bps: 4.582 (<=7.50, **탈락**)
- entry_idx 경계 드롭: 4/101,910건 (0.004%)

#### [TF 2H] - ✅ PASSED (master TF)
- Ready Folds: 4/4
- Cov: 1.000 (>=0.80) | Symbol-Breadth: 24.398 (>=5.00) | probe_lcb_bps: 123.343 (>7.50)
- **승격 결과**: n_ready=77
- entry_idx 경계 드롭: 6/131,427건 (0.005%)

#### [TF 4H] - ⚠️ PARTIAL (advisory fold_ratio만 위반, base TF)
- Ready Folds: 1/4 (F0 통과)
- Cov: 1.000 (>=0.80) | Symbol-Breadth: 8.000 (>=3.00) | probe_lcb_bps: 95.796 (>7.50)
- **승격 결과**: 7개 승격, `fold_ratio:0.250` 어드바이저리(25% quality_weight 페널티, 배포는 유지)
- entry_idx 경계 드롭: 34/177,930건

#### [TF 6H] - ❌ BLOCKED (이번 세션 중 회귀, 원인 미규명)
- Ready Folds: 1/4
- Cov: 1.000 (>=0.80) | Symbol-Breadth: 2.273 (<3.00, 탈락) | probe_lcb_bps: -1.510 (<=7.50, 탈락)
- 직전 스냅샷(1h 재도입 이전)엔 Symbol-Breadth 3.769 / probe_lcb_bps 13.478로 PASSED(n_ready=1)였음 — 1h 추가만으로 재현(§2.3)
- entry_idx 경계 드롭: 9/91,661건

#### [TF 8H] - ❌ BLOCKED
- Ready Folds: 0/4
- Cov: 1.000 (>=0.80) | Symbol-Breadth: 2.000 (<3.00) | probe_lcb_bps: -inf
- **블로커**: `sym_count:2.000, probe_lcb_bps:-inf, match_ratio:0.855, fold_ratio:0.000`
- entry_idx 경계 드롭: 43/83,681건

#### [TF 12H] - ❌ BLOCKED (sym_count는 개선)
- Ready Folds: 0/4
- Cov: 1.000 (>=0.80) | Symbol-Breadth: **3.000 (>=3.00, 통과)** | probe_lcb_bps: -inf(여전히 탈락)
- **블로커**: `probe_lcb_bps:-inf, match_ratio:0.863, fold_ratio:0.000`
- entry_idx 경계 드롭: 60/84,747건

#### [TF 1D] - ⚠️ PARTIAL (advisory fold_ratio만 위반)
- Ready Folds: 1/4 (F0 통과)
- Cov: 1.000 (>=0.80) | Symbol-Breadth: 3.000 (>=3.00) | probe_lcb_bps: 227.324 (>7.50)
- **승격 결과**: 12개 승격, `fold_ratio:0.250` 어드바이저리
- entry_idx 경계 드롭: 21/24,611건

## 4. 다음 분석 우선순위

1. **[최우선] 1h 추가가 6h/12h에 영향을 주는 정확한 인과 경로 규명**: cross-TF 공유 연산(family pruning/redundancy/diversity audit 등 `l1_tfs` 전체 집합을 보는 로직) 의심되나 미확정. TF 목록 변경이 무관한 다른 TF의 결과를 바꾸는 건 그 자체로 파이프라인 모듈성 문제 — 별도 스펙 필요.
2. **2h/1h 격차의 경제적 의미 재확인**: 1h는 2h보다 촘촘한데도 손익분기를 못 넘겼다 — 이는 "촘촘할수록 유리하다"는 단순 밀도 가설을 반증하는 긍정적 신호. 2h의 엣지가 진짜인지에 대한 확신이 강화됐으나, fold 밀도 정규화(직전 세션에 미해결로 남긴 LIMIT-04)는 여전히 미시행.
3. **8h/12h 완전 차단의 최종 처리**: sym_count는 12h만 개선(원인 불명, 1h 부작용과 동일 메커니즘일 가능성), probe_lcb_bps는 둘 다 여전히 -inf — 실질적 배포 가능성 없음. quant.md 원칙에 따라 "현재 데이터로 구조적 불가능" 결론을 굳히고 배포 제외 유지할지 결정 필요.
4. **6h 회귀의 실질적 영향 평가**: n_ready=1(사실상 무의미)이었던 TF가 완전 차단으로 바뀐 것 자체의 실무 영향은 작으나, 근본 원인(1번 항목)을 모르는 채로 두면 향후 다른 TF 추가/제거 시 동일 현상이 반복될 위험.
