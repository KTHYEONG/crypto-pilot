---
title: Layer 2 AWF Engineering History
domain: futures.strategy.tiered_workflow
type: adr
status: active
priority: high
ai_read_policy: when_related
---

## 2026-06-15 Layer2 이벤트 계약 분리 및 support-preserving projection
- **Delta:** `ValidatedSignalBatch` 기반 event schedule을 L2 입력 SSOT로 고정하고, `rank_and_select`는 `signed`/`absolute` 모드로 분리했다.
- **Rationale:** L1의 방향·holding·net edge를 bar-level 포지션 계약으로 보존해야 short symmetry와 기존 legacy rank를 동시에 유지할 수 있다.
- **Edge Cases:** `project_all_caps`는 support 밖 신규 non-zero를 만들지 않으며, malformed mock mask/funding 입력은 fail-open으로 처리한다.

## 2026-06-15 L2 AWF 정합성 강화 (P0+P1)
- **Delta:** 5개 결함 수정 — 복리 CAGR, 로그 필드명, taker 비용 차감, net edge 핸드오프, AWF 윈도우 look-ahead 제거 + 4중 게이트 도입
- **Rationale:** audit 점수 43/100 → 구조 정합성 확보. 복리 목적(사용자 요구)과 비용 현실성(quant.md §4) 직접 충돌 제거
- **Edge Cases:**
  - `fold_sharpes_h` 전체 fold 기준 계산 후 별도 `_nonempty_sharpes`로 pass_ratio 분모 산출 (zip 길이 불일치 방지)
  - taker 비용은 리밸런싱 첫 bar에만 차감 (`t2 == t` 조건); baseline(1/N)은 무비용 유지 (게이트 보수성)

## 2026-06-15 L2 게이트 재설계 (8조건 절대+상대 이중기준)
- **Delta:** 곱셈식 게이트(×1.20, 절대Sharpe0.30, fold>50%) → 8조건 AND: Stage0(sanity)+A(CAGR/MAR/Sharpe절대)+B(MDD상대+절대)+C(복리fold≥60%)+D(가산Uplift+0.20). `Layer2Result`에 cagr/mar/fold_pass_ratio/blocker_reason 필드 추가.
- **Rationale:** 음수 baseline에서 ×1.20이 임계를 역전(로그 `>=-1.02` 버그). CAGR/MAR 게이트 부재로 절대손실 전략 통과 가능. 복리자산증식 목적과 직접 충돌.
- **Edge Cases:**
  - Stage A CAGR>0: `<=` 비교 (0.0 정확히는 FAIL — 원금 유지만으론 불충분).
  - fold pass = `prod(1+r)>1.0` (변동성드래그 반영); 빈 fold nonempty 분모 분리.
  - config 6키 `l2_params.get(key,default)` 노출 — magic number 금지.

## 2026-06-15 AWF fold pass_ratio zip 버그 수정
- **Delta:** `fold_sharpes_h = [_sharpe(fr) for fr in sim.fold_rets_hybrid if fr]` → `if fr` 필터 제거, `zip(strict=True)` 길이 불일치 런타임 오류 수정
- **Rationale:** 빈 fold 존재 시 `ValueError` 발생. 전체 정렬 유지 + 분모 별도 분리로 수정

## 2026-06-15 L2 AWF 신호 동적 매핑 및 수치적 안정성 확보 (P0)
- **Delta:** `run_l1_nested_swf`에서 `signals_per_fold` 수집 및 AWF 백테스팅 연동. 시점 $t$ 기준 L1 fold 시간 매핑 적용. 비용 허들, 베타 및 수익률 NaN 방어 추가.
- **Rationale:** L1 Nested SWF의 동적 예측 신호 유실로 인한 고정 신호 강제 및 오매핑 버그 해결. 거래 비용 및 수익률 계산에 NaN 유입 시 가중치가 0.0으로 유실되어 스코어카드가 nan이 되는 현상 방지.
