# Architecture Decision Records (ADRs)

본 문서는 `crypto-pilot` 시스템의 핵심 기술적 의사결정 기록(ADR)과 엔지니어링 트레이드오프를 정리합니다.

---

## 1. Executive Decision Matrix

| ADR | 의사결정 주제 | 기각된 대안 | 채택된 솔루션 | 핵심 엔지니어링 근거 및 트레이드오프 |
| :--- | :--- | :--- | :--- | :--- |
| **ADR-01** | **체결 회계 모델** | • 벡터화 가상 곱셈<br>• L2/L3 틱 시뮬레이터 | **3분봉 모의 체결 원장 + 메이커/테이커 집행** | 8h 펀딩비 및 3-tier 수수료 실정산, 메이커 집행으로 CAGR +10.8%p 개선. 계산 시간(~6분) 증가 수용 |
| **ADR-02** | **시계열 검증 체계** | • 단순 K-Fold<br>• 무엠바고 워크포워드 | **168시간 엠바고 16-Fold Walk-Forward & DSR** | 최대 168h 신호 잔여 자기상관 차단 및 96회 탐색 다중 검정 통계 보정 |
| **ADR-03** | **유니버스 및 턴오버** | • 고정 Top-N 순위 컷<br>• 비히스테리시스 필터 | **Schmitt-Trigger 히스테리시스 (60위 진입/120위 방출)** | 유니버스 경계선 부근의 잦은 진입/퇴출 진동을 억제하여 불필요한 턴오버 30% 절감 |
| **ADR-04** | **신호 단순화 & Frozen** | • 19개 지평 복합 앙상블<br>• 과적합 파라미터 튜닝 | **Frozen Top-20 직교 횡단면 알파 결합** | 5개 직교 신호 랭크 평균으로 1h 120일 창에서 인프로세스 직접 산출, 런타임 코드 1,500줄 절감 |
| **ADR-05** | **저사양 클라우드 운용** | • 650개 심볼 전수 수집<br>• 수동 디스크 정리 | **Tail 증분 동기화(~20초) & 슬라이딩 보존** | 갱신 시간 29분 $\to$ ~20초로 단축, OCI A1.Flex 컨테이너 `mem_limit: 2g` 예산 내 무인 운용 |
| **ADR-06** | **배포 보안 및 봉인** | • Mozilla SOPS 암호화<br>• GitHub Secrets 평문 주입 | **서버 .env 정본 + `LIVE_ARTIFACT_KEY` AES-256-GCM** | SOPS 도구 복잡성을 제거하고, 공개 리포에 과거 단위수익률 성과 노출을 암호학적으로 차단 |
| **ADR-07** | **인과적 자본 사이징** | • 표본 전체 사후 고정 배율<br>• 주관적 MDD 예산 | **인과적 베이지안 수축 Kelly 노출 정책** | 사전 $\mu_0=0$(730일 가중) 기반 일별 인과적 모멘트 갱신, ₩300만 실계좌 무청산 복리 극대화 |
| **ADR-08** | **실거래 마진/청산 회계** | • 무한 증거금 가정<br>• 거래소 주문 규칙 무시 | **거래소 규칙 실계좌 규모 원장 (`replay_account`)** | 바이낸스 실거래 브래킷(MMR/cum), 최소주문단위 반영으로 실거래 괴리 0% 달성 |

---

## 2. Detailed Architecture Decision Records

## ADR-01: 3분봉 단위 체결 원장 및 메이커/테이커 집행 회계

### Decision
벡터화된 가상 목표 가중치(Target Weight) 곱셈 방식 대신, 바이낸스 3분봉(`3m`) 프록시 체결과 현금·계약·수수료·펀딩비 결제를 통합한 [`SimulatedInventoryLedger`](file:///home/kth/crypto-pilot/src/mhs/execution/ledger.py) 및 30분 앵커 지정가 대기 후 테이커 폴백하는 메이커(`strict_passive`) 집행 경로를 채택함.

### Context
기존 암호화폐 퀀트 백테스트는 1시간봉 종가 시점에 즉시 체결된다고 가정하여 8시간 펀딩비와 테이커 수수료/슬리피지에 따른 손실을 반영하지 못했습니다. 또한 백테스트에서 유일하게 측정된 Sharpe/CAGR의 개선 요인은 메이커 집행(CAGR 140.7% $\rightarrow$ 151.5%)이었으므로 이를 계정 원장과 라이브 런타임에 일관되게 지원해야 했습니다.

### Selected Approach
1. 3분봉 단위 MTM 평가 $\rightarrow$ 8시간 펀딩비 정산 $\rightarrow$ 주문 체결 $\rightarrow$ 3-tier 수수료 차감의 회계 순서 강제.
2. 메이커 집행 시 발주 시점 기준가(anchor)에 지정가를 배치하고 최대 10개 봉(30분) 대기 후 미체결 잔량을 테이커로 폴백(`strict_passive_execution_policy`).

### Trade-offs
5개년 백테스트 시 연산 시간(~6분)이 소요되나, 가상 체결 PnL 착시를 원천 배제함.

---

## ADR-02: 168시간 엠바고 16-Fold Purged Walk-Forward 검증 및 DSR

### Decision
1주일(168시간)의 시계열 퍼징/엠바고를 강제한 16-Fold 확장(Anchored) Walk-Forward 교차 검증 및 Deflated Sharpe Ratio (DSR) 도입.

### Context
금융 시계열은 자기상관을 가지므로 최대 168시간 모멘텀 피처를 사용할 때 엠바고 없는 교차 검증은 미래 정보 누출(Data Leakage)을 유발합니다. 또한 다중 탐색에 따른 과적합(Data Snooping)을 통계적으로 보정해야 합니다.

### Selected Approach
분기별 16개 확장 폴드 사이에 168시간 엠바고 구간을 물리적으로 격리하고, 탐색 경로의 표본 분산과 왜도/첨도를 반영한 DSR 지표를 산출함.

### Trade-offs
폴드 경계선마다 약 2~3%의 검증 가용 데이터가 소실되나 미래 정보 누출을 엄격히 방지함.

---

## ADR-03: Schmitt-Trigger 히스테리시스 (Top-60/120) 동적 유니버스

### Decision
거래대금 상위 60위 종목 진입 후 120위(`60 * 2.0x`) 밖으로 밀려날 때만 방출하는 2.0x Schmitt-Trigger 이중 임계값 도입.

### Context
매 시간 거래대금 순위 상위 N개를 단순 재선정하면 60위 부근 종목들의 잦은 진입/퇴출 진동(Boundary Churning)으로 거래비용이 급증합니다.

### Selected Approach
과거 720시간 거래대금 중앙값 상위 50% 중 상위 60위 이내 진입 시 편입, 120위 이탈 시에만 방출하는 히스테리시스 적용으로 불필요한 턴오버를 30% 이상 절감함.

---

## ADR-04: Frozen Top-20 직교 횡단면 알파 전략 단순화

### Decision
복잡한 런타임 의존성을 가졌던 19개 지평 위원회 스택을 5개 직교 횡단면 피처의 Frozen Top-20 결합([`frozen_mhs_top20_v2`](file:///home/kth/crypto-pilot/src/mhs/frozen_research_candidate.py))으로 단순화.

### Context
라이브 데몬이 옛 horizon-diagnostic 봉인 파라미터 스택(약 1,500줄)으로 구동되어 백테스트 코드와 드리프트가 발생할 위험이 있었습니다. 또한 1h 120일 패널 창에서 인프로세스로 신호를 직접 계산할 수 있는 결정론적 구조가 요구되었습니다.

### Selected Approach
테이커 불균형(168h/720h), 모멘텀(336h), 잔차 모멘텀(336h), 왜도(168h)의 5개 직교 피처 랭크 평균 결합 및 종목별 5% 클립 정책을 고정 사양으로 채택.

### Rationale
신호 계산 경로를 1,500줄 이상 감축하고 16초/1GB 메모리 예산 내에서 완주하여 런타임 안정성을 비약적으로 향상시킴.

---

## ADR-05: 디스크 Tail 증분 갱신 및 원자적 보존 관리 (OCI A1.Flex Cloud Ops)

### Decision
단일 클라우드(OCI Ampere A1.Flex, 호스트 2 OCPU/12GB 공유, `mhs-live` 컨테이너 `mem_limit: 2g`)에서 로컬 Parquet 디스크 tail 증분 갱신과 슬라이딩 보존 프루닝 체계를 구축함.

### Context
650개 심볼 전수 시세를 매번 동기화하면 29분이 소요되어 크론 타임아웃이 발생했습니다. 또한 메모리 한도 설정 오류(과거 1200m 고정)로 피크 시 OOM(exit 137)이 발생한 사례가 있어 실측 피크 기반의 정밀 예산 관리가 필요했습니다.

### Selected Approach
1. 로컬 Parquet의 마지막 타임스탬프 기준 미수집 구간만 멀티스레드 패치하여 갱신 소요 시간을 약 20초로 단축.
2. 재생성 가능 데이터는 임시 파일 원자적 대체(`tmp.replace`) 방식으로 슬라이딩 절단.
3. 컨테이너 `mem_limit`은 실측 피크에 맞추어 `2g`로 최적화.

---

## ADR-06: 서버 .env 단일 정본 + `LIVE_ARTIFACT_KEY` AES-256-GCM 아티팩트 봉인

### Decision
Mozilla SOPS 도구 의존성을 폐기하고, 서버 로컬에 안전하게 배치된 `.env`를 단일 정본으로 삼으며 전략 부트스트랩 단위수익률을 `LIVE_ARTIFACT_KEY` 기반 AES-256-GCM 봉투([`crypto.py`](file:///home/kth/crypto-pilot/src/live/crypto.py))로 암호화하여 배포.

### Context
SOPS 도구 체인은 키 관리와 배포 파이프라인의 복잡도를 가중시켰으며, 공개 리포지토리에 과거 백테스트 단위수익률 데이터가 평문으로 노출되는 보안 위험이 있었습니다.

### Selected Approach
서버 환경변수 `.env`를 정본으로 직접 마운트하고, 배포 아티팩트(`frozen_unit_returns_*.parquet.enc`)만 대칭키로 봉인하여 커밋.

---

## ADR-07: 인과적 베이지안 수축 Kelly 노출 정책

### Decision
사후 확증 편향이 존재하는 고정 레버리지 대신, 사전 $\mu_0 = 0$(730일 가중)에서 매일 전일까지의 실적만으로 사후 모멘트를 인과적으로 갱신하는 베이지안 Kelly 정책([`account_policy.py`](file:///home/kth/crypto-pilot/src/mhs/account_policy.py)) 도입.

### Context
기존 레버리지 방식은 전체 백테스트 기간의 사후 통계량에 의존해 사실상 룩어헤드 바이어스를 내포하고 있었으며 시장 국면 변화 시 과도한 레버리지로 파산할 위험이 있었습니다.

### Selected Approach
매일 전일까지의 단위북 실적으로 사후 $\mu, \sigma$를 추정하고, 기대수익률을 50% 할인한 켈리 공식과 거래소 증거금 상한을 결합하여 동적 노출을 결정함. 실측 결과 소액(₩300만) 계좌에서도 청산 없이 CAGR 202.2% 달성.

---

## ADR-08: 거래소 규칙 실계좌 규모 원장 (`replay_account`)

### Decision
바이낸스 USDT-M 실거래 브래킷(Tier별 누적유지증거금 MMR/cum) 및 주문필터(최소주문금액, 수량 단위)를 전수 반영한 실계좌 원장 모델링 채택.

### Context
단위북(unlevered)이나 무한 증거금을 가정한 백테스트는 실제 소액 계좌에서 강제 청산이나 최소 주문단위 미달로 인한 주문 거부 위험을 포착할 수 없었습니다.

### Selected Approach
1. 실거래 베뉴 룰 스냅샷 수집기를 통해 MMR/cum 사다리와 주문필터 로드.
2. 3분봉의 저가/고가 기준 증거금 위반 시 즉시 전액 청산 처리하는 엄격한 계정 원장 구현.
3. 매 백테스트마다 공식 원장과의 재조정 격차를 투명하게 공시.
