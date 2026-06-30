# L1/L2 다음 Spec 방향성 (2026-06-29 진단 결과 압축)

SSOT 상세: `docs/specs/l1-alpha-redesign.md`. 측정 훅: `L1_PROBE_DIAG`, `L2_DIAG_ATTR`(+`[L2-ATTR-REGIME]`).

## 1. 확정된 진단 (측정 기반)

- **L1 edge ≈ 100% 시계열 trend-beta, 횡단면 alpha≈0**: beta_edge +76.6bps vs selection_alpha +3.7(부호진동), residual_ic +0.028(median +0.01). gross의 103%가 beta.
- **비용·selection-inflation 아님**: rt_cost 7.5≪gross, top-k≈breadth. magnitude gap 주범은 **realized_price 음전**(OOS beta 반전).
- **regime별 OOS(2025) price/bar**: bull +0.91 / **bear −1.13(주범)** / crisis +0.16. crisis는 ≈0(범인 아님).
- **regime 라벨 IS↔OOS 부호반전**: bear IS(2023-24) **+82.5** ↔ OOS(2025) **−1.13**. → 정적 regime-gating은 필요하나 불충분.

## 2. 검증된 개선/기각 (A/B, 200 trials)

| 변경 | 결과 | 판정 |
|---|---|---|
| breadth_mode (selection 제거) | CAGR −9.2%, bull +0.71→−1.19, fold#2 −24.8% | **기각**. residual_ic≈0(IS·Arch-Only degenerate)은 잘못된 프록시. `rank_and_select`는 배포 실가치 보유. |
| regime cap↓ (bear 0.75→0.35, crisis 0.55→0.25) + select | CAGR +6.1→+7.1%, Sharpe 0.36→0.47, **Uplift +0.11→+0.25(통과)**, bear −1.43→−1.13 | **유지**. 소폭이나 실질 개선. |
| IC 하드 게이트 | 프로덕션 무조건 BLOCK(입력 debug 종속) | **기각**. DEBUG 모니터링(`[L1-IC-DIAG]`)만. |

현 확정 config: `l2_selection_breadth_mode=False` + bear/crisis cap↓. (banked win, 단 여전히 CAGR≪30% BLOCKED)

## 3. 다음 Spec 우선순위

### P1 — bear regime 방향성 ✅ 측정 완료 (2026-06-29) — 가설 기각, P3로 승격
- **계측 결과** (`SIDE[regime]` 훅, 16 fold-diag): bear는 **net-long이 아니라 90~99% 숏**.
  - IS f1/f2(23-24): bear 숏 `sr+185~+207`(강수익), 롱 `lr−151~−276`.
  - **OOS f3(2025): bear 숏 전부 음전 `sr−4~−41`**, 소수 롱은 오히려 `lr+99~+322`.
- **결론**: "net-long 편향 → 롱 붕괴" 가설 **거짓**. 진짜 출혈 = **숏 쪽**. §1 "IS+82.5↔OOS−1.13"의 정체 = **추세-숏이 OOS에서 역전**(2025 bear 라벨=반등/숏스퀴즈) → side-bias 아닌 **within-regime 비정상성(P3)**.
- **판정**: `flip_short` NO-GO(신호 이미 숏). `flat`은 fold3 방어되나 bear cap↓ 강화판일 뿐. SSOT: `docs/specs/l1-bear-side-directionality.md`.

### P2 — bull 자본효율
- bull +0.91/bar 견고, L*=1.69·RiskUtil 62% → **bull 한정 레버리지 상향** 여지. 단 P1 방어 하에서만.

### P3 — within-regime 비정상성 (구조적 한계) ✅ 구현·A/B 완료 (2026-06-30) — 성과 NULL
- **walk-forward 적응형 regime-reliability 구현**: trailing N-fold 실현 bear edge로 bear cap online 강등(look-ahead 안전, env `L2_REGIME_RELIABILITY`, 기본 off).
- **A/B(순차, seed42, 40trials) 결과**: 최종 3-fold **byte-identical**(−17.1/+16.7/+10.4), Best CAGR 55.23→55.10%. 메커니즘 정상 발화(78회 강등)하나 성과 무변.
- **원인**: best-param 포트폴리오 실현 bear edge≈0(±0.4bps) — 최적 config가 이미 기존 L2 routing으로 bear 출혈 제거 → 강등할 잔여 없음. **cap/sizing 레버 고갈 확정**. SSOT: `docs/specs/l3-adaptive-regime-reliability.md`.

## 4. 정직한 기대치 (P1·P3 실증으로 확정)
- 사이징/cap/selection/side-policy는 **완화**일 뿐 음(−) OOS edge를 양(+)으로 못 바꿈 — **P3 A/B가 실증**(reliability 발화해도 성과 byte-identical).
- 병목 = **Fold#1(24Q4-25Q1) −17.1%**, regime-노출이 아닌 **L1 횡단면 alpha 부재**. 30% CAGR엔 **신호 자체의 OOS-지속 edge** 개선이 유일 경로.
- ⬆️ **다음 우선순위 = L1 cross-sectional alpha 재설계** (P2 bull 자본효율은 부차). cap/regime 레버는 고갈 — 더 파지 말 것.
