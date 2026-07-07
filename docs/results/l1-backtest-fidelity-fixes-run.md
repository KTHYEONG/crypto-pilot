# L1 Backtest Fidelity Fixes — Real Run Verification

- Run date: 2026-07-07
- 관련 spec: `docs/specs/l1-backtest-fidelity-fixes.md` (sync 시 제거됨, 이 문서가 실측 기록의 SSOT)
- 비교 대상: `4h_1783345440`(수정 전) vs `4h_1783384093`(수정 후), 동일 커맨드/시드로 재실행

## 실행 명령
```
UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. timeout 900 uv run python src/execution/opt_main_futures.py \
  --phase l1 --sync skip --timeframe 4h --trials 1 --seed 42 \
  --alpha-foundry gate --alpha-foundry-total-l1-budget 30 --alpha-foundry-min-conviction-lcb-bps 5.0
```

## 실측 결과 — `btc_regime_pullback` 아키타입 수정 효과

| 지표 | 수정 전 (mean_rev/snapback 오분류) | 수정 후 (trend/trend_grind 정정) |
|---|---:|---:|
| mean_net_bps | -55.77 | **-9.19** |
| block_lcb_bps | -89.94 | **-38.35** |
| nw_tstat | -1.64 | -0.32 |
| discovery_tier | blocked | blocked (판정 불변) |
| reject_reasons | non_positive_lcb\|weak_tstat\|excess_cost_drag | non_positive_lcb\|weak_tstat\|excess_cost_drag |

**해석**: 손실 폭이 약 6배 축소(mean_net_bps -55.8→-9.2bps, LCB -89.9→-38.3bps). 잘못된 archetype(타이트 손절 0.90×ATR/보유≤6bar)이 이 family의 실제 경제성을 심하게 과소평가하고 있었음을 실측으로 확증. 4h에서는 여전히 LCB 음수라 최종 판정(blocked)은 바뀌지 않았음 — 버그 수정이 "판정을 뒤집는" 것이 아니라 "더 정확하게 평가하는" 효과였다는 점이 중요(사전에 특정 방향으로 결과를 예단하지 않았음을 실측으로 재확인).

## 회귀 확인

- `n_panels_in=36`, `n_passed=3` — 수정 전후 동일.
- `selected_for_l1=True` 3건 불변: `lsr_oi_regime_filter`(seed), `mtf_breakout_retest`(candidate), `trend_pullback_continuation`(seed).
- Fix 2(dead config 제거), Fix 3(TF-generic 연율화)는 4h 실행에서 관측 가능한 차이 없음(각각 원래 미사용/원래도 정확한 값이었음) — 6h/8h/12h에서의 Fix 3 효과는 유니버스 디스커버리가 4h 외 TF를 네이티브 실행 지원하지 않아 이번 실측 범위에서 검증 불가(`docs/results/l0-signal-family-diversity-run.md` 참고).

## 결론

3개 fix 중 실측으로 직접 값 변화가 확인된 것은 archetype 재분류 1건이며, 효과가 상당히 크다(6배 손실 축소). 나머지 2건은 코드 정합성 개선(dead code 제거, TF 일반화)으로 이번 실행에서는 중립적(무변화) 확인.
