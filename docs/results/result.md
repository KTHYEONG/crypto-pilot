# L0 Multi-TF Gate Redesign - Run Comparison (HTF Bypass Closed)

- New run id: `4h_1783427649` (per-TF suffix: `4h_1783427649_{4h,6h,8h,12h}`)
- Previous baseline: `4h_1783419659` (`docs/specs/l0_multi_tf_gate_redesign*.md` 작성 이전 스냅샷 — canonical gate는 고쳤지만 base TF 하나만 게이트에 태우던 상태)
- Timeframe: `4h` (base), `6h`/`8h`/`12h`는 이번에 처음으로 동일 L0 게이트에 편입됨
- Command: `UV_CACHE_DIR=/tmp/uv-cache LOG_LEVEL=DEBUG PYTHONPATH=. timeout 900 uv run python src/execution/opt_main_futures.py --phase l1 --sync skip --timeframe 4h --trials 1 --seed 42 --alpha-foundry gate --alpha-foundry-total-l1-budget 30 --alpha-foundry-min-conviction-lcb-bps 5.0`
  - artifact write는 기본값(`artifact_write_enabled=False`)이라 CLI 자체로는 파일이 남지 않음. 데이터 확보를 위해 `build_alpha_foundry_runtime_config()`를 1회성으로 monkeypatch해서 `artifact_write_enabled=True`로 강제한 별도 실행에서 TF별 parquet 4개를 남김.
- Source: `logs/futures/alpha_foundry/4h_1783427649_{tf}_evidence.parquet` (4h: 42 rows/29 families, 6h/8h/12h: 각 16 rows/11 families)
- Exit: `0`

## Executive Summary

`docs/specs/l0_multi_tf_gate_redesign.md`의 fan-out(Phase1 cheap-gate) → fuse(Phase2 cross-TF corroboration) → fan-in(Phase3 canonical gate) 오케스트레이션이 `run_candidate_strategy_for_universe()`에 실제로 배선됐다. 이전 세션에서 새 함수들(`run_alpha_foundry_l0_gate_multi_tf` 등)은 정확히 동작함을 unit test로 확인했지만 **실제 호출 경로에 연결되지 않아 프로덕션 데이터에는 전혀 영향이 없었다** — 이번에 그 배선을 완료하고 실행한 결과, 그동안 L0 경제성 게이트를 완전히 우회해온 HTF(6h/8h/12h) 후보들이 처음으로 동일 기준에 걸러지면서 **최종 L1 승격 수가 (5+50+45+99=199) → 43으로 급감**했다.

## 1. TF별 Readiness 변화 (배선 전 → 후)

| TF | 배선 전 (게이트 우회) | 배선 후 (L0 게이트 적용) | HTF native panel 투영 수 |
| --- | --- | --- | --- |
| 4h | BLOCKED, probe_lcb_bps 88.9 | BLOCKED (4/5 Passed), probe_lcb_bps 87.6 — 거의 불변(원래도 게이트 적용) | - |
| 6h | **✅ READY, 50 promoted** | **❌ BLOCKED (0/5 Passed), probe_lcb_bps=-inf, 0 promoted** | `Proj=0` |
| 8h | ✅ READY, 45 promoted | ❌ BLOCKED (4/5 Passed), probe_lcb_bps 24.2 | `Proj=1` |
| 12h | ✅ READY, 99 promoted | ✅ PASSED (5/5 Passed), probe_lcb_bps 27.8 | `Proj=2` |
| **최종 L1 Promoted 합계** | **~199** (5+50+45+99, TF별 개별 집계) | **43** (통합 "Top 5 / 43 Promoted") | - |

```
🧬 [L1: MULTI-TF PANEL INJECTION]
  └─ Active : [6h] Proj=0 Syms=126 | [8h] Proj=1 Syms=126 | [12h] Proj=2 Syms=126
[ALPHA-FOUNDRY] mode=gate n_panels_in=42 n_bound=42 n_passed=4 n_rejected=38 elapsed=4.2505s
```

Interpretation: 6h는 게이트를 통과하는 신호가 하나도 없어(`Proj=0`) L1 후보 자체가 사라졌다. 이전까지 "6h에서 50개 승격"으로 보고됐던 것은 L0 경제성 검증(cost_drag/tstat/insufficient_events 등)을 전혀 거치지 않은 착시였다는 뜻이다. 8h/12h도 투영되는 패널 수가 각각 1개, 2개로 격감(HTF native panel 다수가 자체 recipe 바인딩·cheap-gate 단계에서 이미 탈락).

## 2. TF별 L0 Evidence 요약

| TF | rows | families | gate_passed | selected_for_l1 | handoff_tier 분포 | selected_for_l1∧blocked (leak) |
| --- | --- | --- | --- | --- | --- | --- |
| 4h | 42 | 29 | 1 | 4 | blocked=37, seed=5 | 0 |
| 6h | 16 | 11 | 1 | 0 | blocked=16 | 0 |
| 8h | 16 | 11 | 1 | 1 | blocked=15, seed=1 | 0 |
| 12h | 16 | 11 | 2 | 2 | blocked=14, seed=2 | 0 |

- 4개 TF 전부 `selected_for_l1=True`인 행이 `handoff_tier=blocked`인 leak 0건 — 지난 세션에서 고친 Gap 3(`selected_for_l1` leak)가 멀티-TF 확장 이후에도 유지됨을 재확인.
- HTF(6h/8h/12h)의 family pool은 `resolve_tf_signal_pool()`에 의해 11종으로 제한되어 4h(29종)보다 훨씬 좁다 — 이는 기존 설계(HTF 전용 signal pool)이며 이번 변경과 무관.

## 3. 잔여 관찰: `tf_corroboration`이 이번 실행에서도 0.0 — 단, 원인이 다르다

이전 세션에서 발견한 버그(2-tuple vs 3-tuple key 불일치)는 unit test로 수정 확인됐지만, 이번 실데이터 실행에서도 4개 TF 전부 `tf_corroboration=0.0`, `corroboration_tier="insufficient_coverage"`로 나온다. 원인을 추적한 결과 **버그가 아니라 실측 데이터 특성**임을 확인했다:

- 4h와 6h/8h/12h 사이에 실제로 겹치는 `(family, variant)` 조합이 존재한다(예: `trend_ma/ema_12_72`, `mtf_breakout_retest/mtf_bor_20` 등 9개 페어 확인).
- 하지만 `fuse_multi_timeframe_evidence()`는 다른 TF의 해당 페어가 `reject_reasons`에 `insufficient_events`/`insufficient_effective_n`을 포함하면 그 TF를 `tf_coverage_count`에서 제외한다. HTF는 bar 수가 적어(예: 12h는 4h 대비 bar 수가 1/3) 동일 캘린더 구간 내 이벤트 수가 구조적으로 적고, 실제 `AlphaGateConfig`(운영 임계치, unit test의 완화된 설정이 아님) 기준으로는 겹치는 페어들도 다수가 `insufficient_events`로 걸러진다.
- 결과적으로 "겹치는 페어는 있으나 유효 커버리지가 0"이 되어 `insufficient_coverage`로 판정 — 이는 키 불일치 버그가 아니라 **HTF 데이터 볼륨 자체의 한계**다. 코드는 정확히 설계대로 동작 중이다.

## Current Conclusion

- **핵심 아키텍처 리스크(HTF가 L0 경제성 게이트를 우회해 L1로 직행) 해소 확인.** 6h는 이제 완전히 차단되고, 8h/12h도 큰 폭으로 후보가 줄었다.
- 3개 배선 갭(canonical gate 미호출/미배선/`selected_for_l1` leak) + 이번 멀티-TF 오케스트레이션 배선까지, `alpha_signal_generation.md` → `l0_multi_tf_gate_redesign.md` 스펙 체인 전체가 실측으로 종결됐다.
- `tf_corroboration`이 여전히 0인 것은 남은 개선 여지이지만, 그 원인이 "배선 안 됨"에서 "HTF 이벤트 수 부족"으로 바뀌었다 — 다음 단계가 있다면 HTF cheap-gate의 `min_events`/`min_effective_n` 임계치를 TF별로 완화하거나, `bars_per_year` 정규화된 이벤트 밀도 기준으로 바꾸는 것이 후보가 된다(이번 스코프 밖).
