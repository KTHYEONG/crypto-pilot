---
title: Layer 2 AWF Engineering History
domain: futures.strategy.tiered_workflow
type: adr
status: active
priority: high
ai_read_policy: when_related
---

## 2026-06-15 L2 AWF 정합성 강화 (P0+P1)
- **Delta:** 5개 결함 수정 — 복리 CAGR, 로그 필드명, taker 비용 차감, net edge 핸드오프, AWF 윈도우 look-ahead 제거 + 4중 게이트 도입
- **Rationale:** audit 점수 43/100 → 구조 정합성 확보. 복리 목적(사용자 요구)과 비용 현실성(quant.md §4) 직접 충돌 제거
- **Edge Cases:**
  - `fold_sharpes_h` 전체 fold 기준 계산 후 별도 `_nonempty_sharpes`로 pass_ratio 분모 산출 (zip 길이 불일치 방지)
  - taker 비용은 리밸런싱 첫 bar에만 차감 (`t2 == t` 조건); baseline(1/N)은 무비용 유지 (게이트 보수성)

## 2026-06-15 AWF fold pass_ratio zip 버그 수정
- **Delta:** `fold_sharpes_h = [_sharpe(fr) for fr in sim.fold_rets_hybrid if fr]` → `if fr` 필터 제거, `zip(strict=True)` 길이 불일치 런타임 오류 수정
- **Rationale:** 빈 fold 존재 시 `ValueError` 발생. 전체 정렬 유지 + 분모 별도 분리로 수정
