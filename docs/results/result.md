# L0/L1 Discovery Snapshot

- **최신 측정일**: `2026-07-15` (run `--phase l1 --timeframe 4h --sync skip`, `LOG_LEVEL=DEBUG`, post per-TF native `labeled_events_by_tf` fix)
- 이 문서는 **현재 상태와 최신 관측 데이터만** 담는다. 과거 세션의 반복 로그는 `docs/decisions/decisions.md`/`decisions_archive.md`에 보존.
- **⚠️ 이 스냅샷은 이전(2026-07-14) 버전을 완전히 대체한다 — 숫자가 반전됐다.** 이전 스냅샷의 "느린 TF일수록 성과 좋음"이라는 결론은 그리드 불일치 아티팩트였음이 이번 수정으로 확정됨(§3).

## 1. L0 → L1 TF별 배포 현황 (최종, per-TF native grid 수정 이후)

| TF | L0 gate_passed | L1 n_ready | L1 blockers | 비고 |
| --- | ---: | ---: | --- | --- |
| **2h** | ✅ | **103** | none | **master TF 자동선정**. 이전 17 → 103 (6배 이상 증가) |
| 4h | ✅ | 7 | `fold_ratio:0.250` | base TF, 이전과 동일(self-consistent라 영향 없음) |
| 6h | ✅ | 1 | none | 이전 16 → 1 (거의 소멸) |
| 8h | ❌ | 0 | `sym_count:2.000, probe_lcb_bps:-inf, match_ratio:0.886, fold_ratio:0.000` | 이전 70 → **완전 차단** |
| 12h | ❌ | 0 | `sym_count:1.000, probe_lcb_bps:-inf, match_ratio:0.425, fold_ratio:0.000` | 이전 151 → **완전 차단** |
| 1d | ✅ | 12 | `fold_ratio:0.250` | 이전 153 → 12 (대폭 감소) |
| 1h | — | — | — | L0 단계 제외 (변동 없음) |

- `would_resolve_master_tf`: 전 TF 공통으로 **`2h`** (이전엔 항상 `1d`).
- 배포 가능 TF: **3/6 유효 배포**(2h/4h/1d), 6h는 사실상 무의미(1건), 8h/12h는 완전 차단.

## 2. 근본 원인 및 수정 내역 (2026-07-14~15 세션)

### 2.1 문제의 실제 근원 — 3단계 누적 결함

1. **그리드 공유 버그** (`aligned_by_tf` 미배선): `CandidatePipelineOutput.aligned_by_tf`가 `TieredL1Handoff`/`run_tiered_pipeline` 호출부에 전달되지 않아 6개 TF 전부가 base(4h) 그리드 하나를 공유. → `_build_output` 팩토리 도입 + 4개 반환지점 통일 + `__post_init__` 가드로 해결.
2. **이벤트 데이터가 애초에 TF-네이티브가 아니었음** (`labeled_events_by_tf` 구조적 결함): L0 게이트는 TF별 네이티브 grid를 이미 정확히 썼지만, 실제 L1 워크포워드가 소비하는 `labeled_events`는 **항상 base(4h) grid 하나로만** 만들어지고 있었음 — 다른 TF의 신호는 `project_htf_panels_to_base`로 base grid에 투영된 뒤 `native_tf` 태그만 원래 TF명으로 붙어 있었음(`entry_idx`는 base grid 기준 위치값). 그리드 공유 버그를 고치자, 이 투영된 이벤트가 (a) 더 작은 native grid에선 `IndexError`로, (b) 더 큰 native grid(2h)에선 **크래시 없이 조용히 잘못된 시점의 봉을 가리키는 채로** 드러남.
   → `labeled_events_by_tf: dict[str, pd.DataFrame]`를 신설, 각 TF 고유의 `panels_for_l1`(L0 admission 통과분, recipe_id 스탬프 완료)로 그 TF 고유 grid 위에서 직접 라벨링 — `entry_idx`가 원천적으로 다른 grid로 넘어갈 일이 없도록 구성.
3. **구현 중 발견된 2차 결함(같은 세션 내 즉시 수정)**: 신규 라벨링 루프를 L0 recipe-binding **이전** 시점(원시 `panels_by_tf`)에 배치해 `l0_recipe_id`가 전부 빈 문자열로 나와 모든 TF가 "delivery route has no labeled events"로 전면 차단됨 → 라벨링 시점을 `pruned_multi_results[tf].panels_for_l1`(recipe_id 스탬프 완료 후)로 재배치하고 `native_tf` 컬럼 명시적 설정으로 해결.
- **방어 장치**: `run_per_tf_l1`에 `entry_idx` 경계 체크 추가 — 범위 밖 이벤트는 크래시 대신 `[DATA]` WARNING 후 드롭(수정 후 실측: 2h 262,638건 중 14건=0.005%만 드롭 — 정상적인 경계 케이스 수준).

### 2.2 왜 이전 스냅샷(8h=70, 12h=151, 1d=153)이 틀렸는가
- 이전 수치들은 **base(4h) grid 위에 투영된 다른 TF들의 이벤트를 그 TF 고유의 성과인 것처럼 집계한 결과** — 실제로는 base grid의 봉 배열 구조(추세/변동성 패턴)가 우연히 그 TF의 "명목상" 이벤트 개수와 뒤섞여 인위적으로 높은 edge/n_ready를 만들어냄.
- 8h/12h는 native grid로 재평가하자 `sym_count`(1~2)와 `probe_lcb_bps=-inf`로 완전 붕괴 — 실질적인 고유 신호가 사실상 없었다는 뜻.
- 2h는 반대로 저평가돼 있었음 — base grid(4h, 6949봉)보다 촘촘한 자기 grid(11736봉)에서 재평가되며 n_ready 17→103.

## 3. TF별 상세 (fold 레벨, 2026-07-15 재실행)

#### [TF 2H] - ✅ PASSED (master TF)
- Cov: 1.000 (>=0.80) | Symbol-Breadth: 22.407 (>=3.00) | probe_lcb_bps: 108.209 (>0.00)
- **승격 결과**: 총 229개 후보 중 Top 5 기준 **103개 최종 승격**
- entry_idx 경계 드롭: 14/262,638건 (0.005%)

#### [TF 4H] - ⚠️ PARTIAL (fold_ratio 어드바이저리만 위반, base TF)
- Ready Folds: 1/4 (F0 통과)
- Cov: 1.000 (>=0.80) | Symbol-Breadth: 8.000 (>=3.00) | probe_lcb_bps: 95.796 (>0.00)
- **승격 결과**: 7개 승격, `fold_ratio:0.250` 어드바이저리로 최종 미배포(구조적 게이트는 통과)
- entry_idx 경계 드롭: 34/177,930건

#### [TF 6H] - ✅ PASSED (사실상 무의미)
- Ready Folds: 2/4 (F0, F3 통과 / F1, F2 블로킹)
- Cov: 1.000 (>=0.80) | Symbol-Breadth: 3.769 (>=3.00) | probe_lcb_bps: 13.478 (>0.00)
- **승격 결과**: 총 1개만 승격
- entry_idx 경계 드롭: 46/201,441건

#### [TF 8H] - ❌ BLOCKED
- Ready Folds: 0/4 (전 폴드 블로킹)
- Cov: 1.000 (>=0.80) | Symbol-Breadth: 2.000 (<3.00 사실상 미달) | probe_lcb_bps: -inf
- **블로커**: `sym_count:2.000, probe_lcb_bps:-inf, match_ratio:0.886, fold_ratio:0.000`
- entry_idx 경계 드롭: 143/174,618건

#### [TF 12H] - ❌ BLOCKED
- Ready Folds: 0/4 (전 폴드 블로킹)
- Cov: 1.000 (>=0.80) | Symbol-Breadth: 1.000 (<3.00 미달) | probe_lcb_bps: -inf
- **블로커**: `sym_count:1.000, probe_lcb_bps:-inf, match_ratio:0.425, fold_ratio:0.000`
- entry_idx 경계 드롭: 105/125,347건

#### [TF 1D] - ⚠️ PARTIAL
- Ready Folds: 1/4 (F0 통과)
- Cov: 1.000 (>=0.80) | Symbol-Breadth: 3.000 (>=3.00) | probe_lcb_bps: 227.324 (>0.00)
- **승격 결과**: 12개 승격, `fold_ratio:0.250` 어드바이저리
- entry_idx 경계 드롭: 21/24,611건

## 4. 다음 분석 우선순위

1. **2h가 왜 이렇게 강한가 실질 검증**: master TF 자동선정이 2h로 바뀐 것이 진짜 경제적 엣지인지, 아니면 촘촘한 grid 특유의 통계적 아티팩트(과최적화·다중검정)인지 별도 검증 필요 — `docs/decisions/decisions.md` ADR 참고 후 홀드아웃/워크포워드 안정성 재확인.
2. **8h/12h 완전 차단의 타당성 확인**: `sym_count` 1~2로 붕괴한 게 진짜 "이 TF엔 고유 신호 없음"인지, 아니면 `panels_for_l1`(L0 admission 통과분)이 너무 협소하게 필터링된 부작용인지 — L0 admission 임계값과 L1 native 이벤트 수 사이의 관계 재검토 필요.
3. **6h의 애매한 위치**(1건 승격) 재검토 — 사실상 8h/12h와 같은 "무의미" 그룹인지 확인.
4. `entry_idx` 경계 드롭 비율(0.005~0.08%)이 TF마다 다른 이유 확인 — 진짜 경계 케이스인지 추가 구조적 이슈의 잔여 신호인지.
