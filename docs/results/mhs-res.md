# MHS Horizon Diagnostic — Latest Result

- **Document Date**: 2026-08-13 (22차, `mhs_return_source_breadth_expansion` 실측 완료 — `ADR_20260813_MHS_RETURN_SOURCE_BREADTH_EXPANSION`. 이전 1-21차 이력은 이번 갱신에서 제거됨, git 이력으로 복구 가능)
- **Domain**: Research / MHS (Multi-Horizon Market State)
- **Run Metadata**: `start=2021-01-01`, `end=2025-12-31`, `execution_timeframe=1m`, `execution_universe_size=30`, `eligible_symbols=446`, `run_elapsed_seconds=728.98`
- **CLI**: `research run portfolio mhs-horizon-diagnostic --slow-book-mode horizon_ensemble --rebalance-filter portfolio_trigger --fold-safe-horizon --discovery-gate --output-tier full`
- **Source**: [`report.json`](file:///home/kth/crypto-pilot/docs/results/mhs_horizon_diagnostic_artifacts/_full/report.json)
- **Research GO 판정 기준**: `daily autocorr-adjusted Sharpe >= 0.6` (primary) AND `stress Sharpe > 0`, 3-fold anchored 전부 통과
- **성격**: `docs/specs/mhs_strategy_foundation_reset.md`(RC-1 계측기 정합화) → `mhs_execution_friction_and_exposure_layers.md`(P1~P4 실행 마찰) → `mhs_return_source_breadth_expansion.md`(호라이즌 축 포화 확정 + 신규 수익원 검증)로 이어지는 3연속 스펙 사이클의 최신 실측. **이번 22차의 핵심은 "21차례 반복 개선이 왜 정체됐는가"의 구조적 원인(breadth 포화)을 MHS 자체 데이터로 확정하고, 유일한 돌파구 후보(펀딩비 캐리)를 검증한 것.**

---

## 1. 진단 배경 — 21차까지의 계측기 정합화 요약

20차까지는 모든 유의성 계측기(`prescreen`/`phase`/`tail`/`xs_rank_ic`)가 **자본 0%인 참조책**을 측정하고 있었고, `research_go.eligible`은 성과와 무관하게 구조적으로 영구 `False`였다(`UNSPECIFIED_POLICY` 무조건 append). 21차가 이를 고쳤다:

- `executed_prescreen` 신설 — 실제 100% 자본이 도는 집행책을 참조책과 나란히 계측. `slow_momentum` net_t가 참조책 0.642 → 집행책 **1.505**로, `fast_reversal`은 참조책 -1.230 → 집행책 **-1.669**로 재확인(0% 배분 결정 유지).
- `deflated_sharpe_ratio` 최초 노출 — 20차에 걸친 순차 탐색(`trials_attempted=20`)을 반영하면 진짜 Sharpe가 양수일 확률은 **53.2%**(동전던지기 수준).
- `MHS_REGISTERED_POLICY_THRESHOLDS` 신설 — 게이트가 이제 성과에 반응 가능한 구조가 됐으나 임계값 미등록으로 `eligible=False` 유지(의도됨).
- 로스터 재정규화 회전율 가설 **실측 기각**(로스터 네이티브 랭킹으로 반증, 회전율 42.74→42.28, -1%뿐) — 회전율은 gross에 비례함을 확인, `_regime_cash_scale`은 동일 gross에서 Sharpe +14% 확인(재평가), `_pnl_vol_target_scale`은 Sharpe를 깎는 것으로 확인돼 플래그화(기본값 `True` 유지, 무회귀).
- 실현 집행비용 노출(`primary_realized_shortfall_bps=10.70` vs 모델 8.0bps) 및 `realized_execution_roster_size=41.93`(선언값 30 대비 +40%) 노출.

## 2. 22차 신규 진단 — breadth 포화 확정과 펀딩비 캐리 검증

### 2.1 근본 질문: 21차례를 반복해도 왜 Sharpe가 0.5선에서 벗어나지 못하는가

$$IR = IC \times \sqrt{BR}, \qquad BR = n_{\text{eff}} \times (\text{연간 리밸런스 횟수})$$

16차 알파엔진 재구축(`mhs_alpha_engine.md` RC-2)은 단일 호라이즌 argmax를 19-호라이즌 동일가중 앙상블로 바꿨고, 18차는 fast 밴드도 7-후보 그리드로 넓혔다. **이 확장이 실제로 독립 베팅(breadth)을 늘렸는지 MHS 자신의 데이터로 측정된 적이 없었다.** 이번에 처음 측정했다 — 신규 `effective_breadth` 통계(참여비율, participation ratio)를 프로덕션 진단 경로(`--discovery-gate`)에 배선해 실제 후보 가중치북으로 직접 계측:

| 축 | 명목 개수 | **$n_{\text{eff}}$ (실측, 22차 프로덕션 경로)** |
| :--- | ---: | ---: |
| `slow_momentum` 19-호라이즌 | 19 | **1.41** (7.4%) |
| `fast_reversal` 7-호라이즌 | 7 | **1.57** (22.4%) |

**19개 호라이즌을 동시에 운용해도 유효 독립 베팅은 1.41개뿐이다.** 16차가 Sharpe를 0.182→0.526으로 올린 것은 앙상블이 breadth를 늘려서가 아니라 argmax의 선택 분산을 제거해서였을 가능성이 높다. 21차례 동안 반복된 "호라이즌 그리드 확장", "로스터 크기 조정" 류의 튜닝은 **같은 유효 베팅 ~1.5개를 재포장**해 온 것으로 확정됐다. 이 축을 더 눌러도 Sharpe 0.6 플로어에 닿지 않는다.

### 2.2 신규 수익원 후보 — 펀딩비 캐리, leak-free 검증에서 탈락

MHS 파이프라인은 모든 replay에 `bar_funding_panel`(인과적으로 정렬된 심볼별 펀딩비율)을 이미 로드하지만, 지금까지 원장의 비용/캐리 항목으로만 쓰였지 **횡단면 신호로 쓰인 적이 없었다.** 22차는 `src/mhs/funding.py`를 신설해 이 데이터를 신호로 승격하고, **기존 discovery/qualification 게이트를 코드 변경 없이 그대로 재사용**(sign-agnostic API)해 leak-free 검증을 배선했다.

**전 구간 예비측정(fold 미분리, 진단 전용)**은 강했다:

| lookback | net_t@4.18bps | net_ann@4.18bps | slow_momentum과의 일간수익 상관 |
| ---: | ---: | ---: | ---: |
| 72h | +4.15 | +29.3%/년 | +0.13 |
| 168h | +4.02 | +27.4%/년 | +0.22 |
| 336h | +2.67 | +17.5%/년 | +0.23 |

그러나 **fold-train-only leak-free 재검증(3-fold, sign=+1/-1 양쪽 모두 시도)에서는 3개 fold 전부 탈락**했다:

| fold | validation | `funding_carry_lookback_hours` | `funding_carry_sign` | `funding_carry_source` |
| ---: | :--- | ---: | ---: | :--- |
| 0 | 2023 | `null` | `null` | `frozen_default` |
| 1 | 2024 | `null` | `null` | `frozen_default` |
| 2 | 2025 | `null` | `null` | `frozen_default` |

전 구간 집계에서 강하게 보였던 신호가 fold-local(1년 남짓) 표본으로 쪼개니 admission floor(|t|≥2.0)를 넘지 못했다 — fast_reversal이 18차에서 정확히 같은 방식으로 탈락했던 패턴과 동일하다. **자본 배분 근거 없음. `PHASE_1_BOOK_SPECS`/`PHASE_1_BOOK_BLEND_WEIGHTS`는 무변경.**

**주의 — 미해결 질문**: 이 admission floor는 fold-local 표본(1년 남짓)엔 과도하게 엄격하다는 기존 우려가 있다(`ADR_20260811_MHS_FOLD_SAFE_HORIZON_SELECTION`) — momentum(168h) 자체도 같은 이유로 fold-train discovery에서 매번 `frozen_default`로 폴백해 왔다. 따라서 "펀딩비 캐리에 edge가 전혀 없다"는 확정 판정이 아니라, **"이 게이트 설정으로는 아직 못 살렸다"**가 정확한 결론이다. 다음 갈래 중 하나로 후속 판단 필요:
1. admission floor를 fold-local 표본에 맞게 재조정한 뒤 재검증 (momentum도 같은 문제를 겪고 있어 펀딩비 캐리만의 문제가 아닐 수 있음)
2. 이 신호는 접고 다른 수익원 후보(OI, 청산, 현물-선물 베이시스)로 이동

### 2.3 회귀 불변식 확인

`slow_momentum.primary_autocorr_sharpe=0.525673922813482`, `blend.primary_autocorr_sharpe=0.5196163403815739`, `realized_execution_roster_size=41.93` — 21차와 바이트 동일. 신규 진단 코드(effective_breadth, funding_carry discovery)가 자본 배분·리플레이 경로를 전혀 건드리지 않았다는 계약이 실측으로 재확인됨.

### 2.4 Research-GO 게이트 — 무변화

```
research_go.eligible = False
research_go.reason_codes = [PRIMARY_AUTOCORR_SHARPE_BELOW_0_6, STRESS_SHARPE_NOT_POSITIVE, UNSPECIFIED_POLICY]
research_go.folds_passed = 2/3
trials_attempted = 20
deflated_sharpe_ratio = 0.5321328197543407
```

## 3. 22차 요약

| 항목 | 상태 |
| :--- | :--- |
| `effective_breadth` 계측 신설 + 프로덕션 경로 배선 | ✅ 완료, 실측 확인 |
| 호라이즌 축(19-slow/7-fast) breadth 포화 확정 | ✅ $n_{\text{eff}}$=1.41/19, 1.57/7 — 21차례 튜닝이 재포장해온 대상이 확정됨 |
| `funding_carry_signal` + discovery 게이트 배선 (자본 배분 없음) | ✅ 완료 |
| 펀딩비 캐리 leak-free 검증 | ❌ 3-fold 전부 admission 실패 — 자본 배분 근거 없음, 원인(신호 부재 vs floor 과엄격) 미해결 |
| 회귀 불변식 | ✅ 21차와 바이트 동일 확인 |

## 4. 다음 스텝 후보

| 후보 | 상태 |
| :--- | :--- |
| funding_carry admission floor의 fold-local 표본 민감도 재검토 (momentum 168h도 동일 증상) | 미착수, 사용자 판단 대기 |
| 신규 수익원 후보 탐색 — OI, 청산, 현물-선물 베이시스 | 미착수 |
| `MHS_REGISTERED_POLICY_THRESHOLDS`(`cap_30_roster`, `primary_annual_return`) 등록 여부 | 미착수, 성과 무관 정책 결정 필요 |
| `pnl_vol_target` 기본값 전환 여부 (사전등록 fold-train-only 기준, `mhs_execution_friction_and_exposure_layers.md` §6.1) | 미착수 |
| `primary_fill_count=0` 필드명 오독 이슈 (버그 아님, `OHLCV_IMMEDIATE_TAKER` 경로 구조상 0) | 미해결, 명명 정정 또는 taker 전용 카운터 필요 |
| `primary_notional_weighted_shortfall_bps` 부호가 단순평균과 반대(-46.6bps vs +10.7bps) | 원인 미상, 초대형 명목가 체결 영향 추정 |
| P3 — 20차까지 참조책 기준으로 내려진 결론 재판정 (`phase.degenerate`, `xs_rank_ic` 등 집행책 버전 미배선) | 미착수 |
