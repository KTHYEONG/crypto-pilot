# L1/L2 다음 방향성 (2026-07-01 갱신 — P0 가드레일 구현 완료 반영)

## 1. 확정된 전제 (재검증 불필요, 압축)

- **L1 edge는 100% 시계열 trend-beta, 횡단면 alpha 없음** — XS factor(momentum/carry/flow/oi_skew) 2회 독립 측정으로 승격 0 확정. 재시도 금지.
- **L\* 레버리지 옵티마이저가 균일/비례 크기 레버를 전부 흡수** — regime cap, trend-efficiency gate, XS 계열은 byte-identical. 오직 **시간분포를 바꾸는 선택적 de-gross**(reversal kill-switch)만 L\* 탈출 가능. 매그니튜드 레버 추가 탐색 금지.
- **breadth 레벨(cross-sectional downside breadth) 기반 시장상태 판별은 반증됨** — 병목 구간과 평상 구간이 breadth 레벨만으로 구분 안 됨(암호화폐 유니버스 상시 고-breadth baseline). `reversal_mode="panel"`은 opt-in 코드로만 남기고 재시도 금지.
- **reversal kill-switch + recovery cooldown(exit hysteresis) 구현·유닛테스트 완료** — 코드 레벨 동작은 검증됨(entry/exit 신호 분리, 단조 확장, look-ahead 없음, 동적 병목-fold 식별).
- **✅ P0 완료(2026-07-01): "위기 부재 PASS" measure-first 가드레일 구현·감사·라이브 검증 완료.** `docs/specs/l2-promotion-crisis-guardrail.md` 3종 게이트 실装:
  1. **Fold MDD 리포팅 버그 수정** — 이전엔 전 fold MDD가 0.0%로 하드코딩(버그). 이제 실측값 표시(예: 11.6%/17.5%/13.3%).
  2. **Gate A(진단, non-blocking)**: 평가 윈도우에 병목-caliber fold(실측 MDD≥15% ∧ CAGR≤0)가 있는지 스코어카드에 자동 배너로 노출. 강세장 조정(MDD 높지만 CAGR 양수)을 위기로 오인하지 않음(P2 사실상 해소).
  3. **Gate B(blocking, calendar-independent)**: synthetic crash shape("Scenario 8": ATH→지속하락)에 reversal-kill 탐지 로직이 실제로 발화하는지 챔피언 승격 직전 매번 재검증. 미발화 시 챔피언 스토어 갱신 자체를 차단.
  - **라이브 파이프라인 재실행(2026-07-01)으로 3종 전부 실증**: NO-CRISIS-WINDOW 배너가 실제로 스코어카드에 노출됐고, synthetic crash defense는 발화(정상)해 챔피언은 정상 갱신됨. "위기 부재 PASS"가 이제 시스템 화면에 항상 명시적으로 드러나며, 더 이상 사람이 기억에 의존해 조심할 필요가 없음.
- **단, Gate B가 검증하는 것은 "탐지 로직이 죽지 않았다"이지 "실제 크래시를 경제적으로 방어한다"가 아니다.** economic replay로 실제 크래시 방어력을 입증한 적은 여전히 없음 — 관련 평가 윈도우(2024-12-31~2025-12-30)가 기존 병목 구간(24Q4-25Q1)을 벗어나 있어 검증 자체가 불가능한 상태는 그대로다. **P0는 "오인 방지" 문제를 해결한 것이지 "크래시 방어력 실증"(P1) 문제를 해결한 게 아님 — 혼동 금지.**

## 2. 다음 스텝 (우선순위 순)

### P1 — 크래시 방어력 실증 경로 확보 (최우선으로 승격, 양자택일 또는 병행)
P0 가드레일 덕분에 이제부터는 "위기 없는 PASS"가 자동으로 투명하게 드러나므로, 다음 목표는 **실제로 크래시를 방어하는 로직을 만들고 이를 검증할 데이터/경로를 확보하는 것**이다.
- **① 마이크로구조 데이터 확장**: bookDepth / half-spread / liquidation proxy / funding-OI stress 등, BTC 가격 단일 축을 넘어서는 causal 조기탐지 입력 확보. reversal-kill의 entry 조건을 가격 후행 지표가 아닌 포지셔닝 선행 지표로 보강하는 방향.
- **③ horizon 확장**: 일/주 단위 regime·macro state를 조건화해, 4h~12h 단일 스케일 detector가 놓치는 구조적 위기 신호(유동성 위축, 매크로 리스크오프) 포착.
- **판단 기준**: 어느 쪽이든 "선택적·시간집중 de-gross" 원칙(L\* 흡수 회피)을 유지해야 하며, 새 입력을 추가하기 전 반드시 measure-first(H1 가설 → 계측 → 반증/채택) 절차를 거칠 것 — breadth 레벨 반증 사례처럼 그럴듯한 가설도 실측 전엔 신뢰하지 말 것.
- **Gate B를 이 작업의 회귀 방지망으로 재사용**: 새 크래시 방어 로직을 추가할 때마다 `synthetic_crash_defense_verdict` 패턴을 확장(다양한 synthetic 위기 형태 — flash crash, 완만한 약세장, 유동성 위축 시뮬레이션 등)해 "여전히 발화하는가"를 계속 자동 검증할 것.

### P2 — 실제 위기 구간 재유입 시점의 economic replay 확보 (기회 포착형, 저강도 모니터링)
- 평가 윈도우가 롤링되며 24Q4-25Q1급 병목이 다시 윈도우에 들어오는 시점(또는 신규 크래시 발생 시점)을 Gate A의 NO-CRISIS-WINDOW 배너로 자동 감지 가능 — 배너가 사라지는(covered=True) 순간이 곧 "이번엔 진짜 economic replay로 크래시 방어력을 검증할 수 있는 기회"임.
- 별도 능동 작업 불필요 — 정기 파이프라인 실행 시 배너 상태만 확인하면 됨. 배너가 사라지는 시점 발생 시 즉시 P1에서 만든 방어 로직의 실제 fold 성과(defense ratio, non-stress fold 손상 여부)를 분석할 것.

### P3 — 목표 재검토 (P1 반복 실패 시)
- trend-beta가 유일한 edge이고 reversal-kill은 신규 alpha가 아니라 방어 레버일 뿐이라는 점을 감안, P1이 반복 실패하면 **현재 strategy class의 구조적 CAGR 상한을 재평가**할 것 — 목표(예: 30%)가 현재 아키텍처로 달성 불가능한 수준인지 정직하게 재검토.

## 3. 명시적 중단선

- XS cross-sectional factor, regime cap, trend-efficiency gate, breadth 레벨 판별 — **전부 반증 완료, 재시도 금지**.
- **"위기 없는 윈도우에서의 PASS"를 프로덕션 승격 근거로 인용하는 것 — P0 가드레일(Gate A 배너 + Gate B 하드 게이트)로 구조적으로 방지됨. 더 이상 별도 수동 체크 불필요.**
- 새 레버는 반드시 **economic replay 기반 measure-first**로 검증할 것 — 코드 구현 완료가 곧 채택을 의미하지 않는다. Gate B의 synthetic 검증은 "메커니즘 생존"만 증명하며 "경제적 효과"의 대체재가 아님 — 이 둘을 절대 혼동하지 말 것.
- `reversal_kill_live`(이번 run에 크래시 방어가 켜져 있었는가) 같은 env-상태 기반 승격 조건은 **폐기 확정(2026-07-01 자체 리뷰)** — 프로덕션 기본값에서 챔피언 스토어를 영구 동결시키는 결함이 있었음. 재도입 금지, 대신 Gate B(메커니즘 헬스체크)만 유지.
