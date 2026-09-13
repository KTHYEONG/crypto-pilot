# 아키텍처 결정 기록 (Architectural Decisions)

## ADR-01: 3분봉 단위 체결 원장 기반 손익 측정 vs. 벡터화 목표 가중치 수익률

### Decision
벡터화된 가상 목표 가중치(Target Weight) 곱셈 방식 대신, 바이낸스 3분봉(`3m`) 프록시 체결과 현금·계약·수수료·펀딩비 결제를 통합한 `SimulatedInventoryLedger`(`src/mhs/execution/ledger.py`)를 포트폴리오 PnL 산출의 기준 원장(Single Source of Truth)으로 채택함.

### Context
기존 암호화폐 퀀트 백테스트는 1시간봉 종가 시점에 슬리피지 없이 즉시 체결된다고 가정하고 목표 비중 변동만으로 수익률을 계산하여 심각한 과최적화(Over-optimism)를 겪었습니다. 실거래 환경에서는 8시간마다 강제 결제되는 선물 펀딩비, 테이커 호가 스프레드, 3계층 수수료(2.64~6.07 bps)가 장기 PnL의 30% 이상을 좌우합니다.

### Alternatives
- **Vectorized Backtesting (Fast Backtest)**: 1시간봉 종가 기준 단순 벡터 연산. 수 초 만에 끝나지만 슬리피지와 펀딩비 결제에 따른 현금 고갈을 전혀 포착하지 못함.
- **Tick-Level Orderbook Simulator (L2/L3)**: 호가창 틱 데이터를 전수 리플레이. 다년간 수백 개 심볼 백테스트 시 메모리 및 연산량이 과도하여 실용성이 떨어짐.

### Selected Approach
바이낸스 네이티브 3분봉(`3m`) 바 단위의 즉시 테이커 체결 프록시. 매 인터벌마다 1) 보유 수량에 대한 마크 가격 MTM 평가 $\rightarrow$ 2) 직전 보유량 기준 8시간 펀딩비 정산 $\rightarrow$ 3) 주문 의도 상쇄 및 3-tier 수수료 차감 $\rightarrow$ 4) 현금 잔고와 재고 수량 갱신의 회계 순서를 엄격히 집행.

### Rationale
3분봉은 기존 5분봉 대비 체결 정밀도를 약 +27% 향상시키며, 5개년 364개 심볼 패널을 약 6분 16초 만에 완주할 수 있어 계산 비용 대비 실거래 일치도가 극대화됨.

### Trade-offs
백테스트 연산 시간이 수 초에서 수 분으로 증가하므로, 효율적인 배치 메모리 관리와 Arrow IPC 및 NumPy 버퍼 스트리밍 기법이 요구됨.

---

## ADR-02: 168시간 엠바고가 적용된 16-Fold Purged Walk-Forward 검증 및 DSR

### Decision
1주일(168시간)의 시계열 퍼징/엠바고를 강제한 16-Fold 확장(Anchored) Walk-Forward 교차 검증 및 Lopez de Prado의 Deflated Sharpe Ratio (DSR) 도입.

### Context
금융 시계열은 자기상관(Autocorrelation)을 가집니다. 전략이 최대 168시간 모멘텀 피처를 사용할 때, 단순 K-Fold 검증이나 엠바고가 없는 Walk-Forward 분석은 Train 종료 시점의 시계열 잔여 정보가 Test 셋으로 누출(Data Leakage)되는 치명적인 결함을 유발합니다. 또한 여러 파라미터를 탐색하는 과정에서 다중 가설 검정 과적합(Data Snooping)이 발생합니다.

### Alternatives
- **Standard K-Fold Cross Validation**: 시계열 셔플링으로 인한 미래 정보 누출로 시계열 백테스트에 부적합.
- **Unpurged Walk-Forward Analysis**: 인접한 테스트 윈도우가 직전 훈련 윈도우의 자기상관을 그대로 흡수하여 성능 왜곡.

### Selected Approach
분기별 16개 확장 폴드 사이에 168시간의 퍼지/엠바고 구간을 물리적으로 제거하여 완전 격리. 또한 96개 탐색 경로의 표본 분산과 왜도/첨도를 반영하여 통계적 유의성을 검정하는 DSR 지표 산출.

### Rationale
미래 참조 편향과 시계열 누출을 차단함. OOS 구간 16/16개 전수 통과(100%) 및 DSR 1.0 달성을 통해 전략 성과의 통계적 유의성을 객관적으로 입증함.

### Trade-offs
폴드 경계선마다 약 2~3%의 검증 가용 데이터가 소실됨.

---

## ADR-03: Schmitt-Trigger 히스테리시스 (Top-60/120) 동적 유니버스 선정

### Decision
거래대금 상위 60위 종목 진입 후 120위(`60 * 2.0x`) 밖으로 밀려날 때만 방출하는 2.0x Schmitt-Trigger 이중 임계값(Hysteresis) 도입.

### Context
매 시간 거래대금 순위 상위 N개를 단순 재선정하면, 60위 경계선 부근의 종목들이 59위 $\leftrightarrow$ 61위를 오가며 불필요한 포지션 청산과 신규 매수 주문을 쏟아내는 진동 매매(Boundary Churning)가 발생하여 슬리피지와 수수료로 net PnL이 파괴됩니다.

### Alternatives
- **고정 심볼 유니버스**: 특정 시점 이후 상장된 우량 자산을 누락하거나 생존 편향 유발.
- **히스테리시스 없는 Top-N 랭킹**: 경계 부근 자산 교체 비용으로 연간 턴오버가 50% 이상 급증.

### Selected Approach
최근 30일(720시간) 거래대금 중앙값 상위 50%를 선별한 뒤, 상위 60위 이내로 진입한 심볼만 로스터에 편입하고 순위가 120위 밖으로 하락할 때만 방출.

### Rationale
로스터의 유동성을 안정적으로 유지하면서, 경계선 진동에 따른 불필요한 포지션 교체 비용과 턴오버를 30% 이상 절감함.

### Trade-offs
활성 로스터의 보유 종목 수가 60~80개 사이로 동적 변동함.

---

## ADR-04: Causal Autocorr 기반 적응형 트랜치 평활 (Adaptive Tranche Smoothing)

### Decision
위원회 북의 Causal Trailing Lag-1 자기상관(Autocorrelation)을 실시간 측정하여 신호 평활 여부를 동적으로 전환함.

### Context
고정 다중 바 평활(Tranche Smoothing)은 횡보장의 휩소(Whipsaw) 손실과 거래 수수료를 줄여주지만, 강한 추세가 시작될 때 포지션 진입을 지연시켜 알파 기회비용을 유발합니다. 반대로 미평활 신호는 추세에는 빠르지만 횡보장에서 잦은 슬리피지로 자본이 갉아먹힙니다.

### Alternatives
- **고정 미평활 (Tranche = 1)**: 추세 민감도는 높으나 횡보장에서 막대한 수수료 손실.
- **고정 평활 (Tranche = 3)**: 횡보장 방어는 되나 추세 변곡점 진입 지연.

### Selected Approach
위원회 북의 최근 15일 Lag-1 자기상관을 인과적으로 계산하여, 음수(휩소 레짐)에서는 3행 트랜치 평활을 적용하고 양수(추세 레짐)에서는 Raw 신호를 지연 없이 집행.

### Rationale
평활과 미평활 간의 근본적인 딜레마를 레짐 적응형으로 해결하여, 비용 3배 스트레스 상황에서도 Stress Sharpe +1.98의 우수한 내구성을 입증함.

### Trade-offs
레짐 판정을 위한 최소 웜업 윈도우가 필요하며 상태 전환 임계값에 대한 모니터링이 요구됨.

---

## ADR-05: 디스크 Tail 증분 갱신 및 원자적 보존 프루닝 (1.2GB Cloud Ops)

### Decision
저사양 단일 클라우드(1 vCPU, 1.2GB RAM)에서 24/7 무인 가동을 위해 무거운 DB 인프라를 배제하고 로컬 Parquet 디스크 tail 증분 갱신과 220일 원자적 프루닝 체계를 구축함.

### Context
오라클 클라우드 프리티어 환경에서 650개 선물 심볼의 전수 시세를 매 시간 동기화하면 갱신에만 29분이 소요되어 크론 타임아웃이 발생했습니다. 또한 축적되는 Parquet 파일이 15GB를 초과하여 디스크 고갈 위기에 직면했습니다.

### Alternatives
- **RDBMS / 시계열 DB (TimescaleDB / PostgreSQL)**: 유휴 상태에서도 1GB 이상의 메모리를 점유하여 1.2GB 저사양 박스에서 OOM 발생.
- **매 사이클 전체 시세 재수집**: hourly 크론 허용 시간 초과.

### Selected Approach
1. 로컬 Parquet의 마지막 타임스탬프를 읽어 `max(tail - 2h, now - lookback)` 구간만 멀티스레드로 증분 패치.
2. 재생성 가능한 시계열(`ohlcv/1h`, `markPriceKlines/1h`, `funding`)에 한해 220일 초과 데이터를 원자적 임시 파일 대체(`tmp.replace(dest)`) 방식으로 안전하게 절단.

### Rationale
시세 갱신 소요 시간을 29분에서 20초로 단축(98.8% 절감)하고, 디스크 사용량을 15GB에서 150MB 수준으로 경량화하여 단일 저사양 서버에서 안정적인 24/7 무인 운영을 달성함.

### Trade-offs
복잡한 애드혹 SQL 쿼리는 불가능하며 분석 시 Parquet 파일을 메모리에 적재해야 함.

---

## ADR-06: Tailscale 사설망 + Mozilla SOPS 기반 제로 트러스트 암호학적 배포

### Decision
외부 인바운드 포트를 완전 차단한 Tailscale VPN 사설망을 경유하고, Mozilla SOPS + Age 비대칭 암호화와 전략 파라미터 SHA-256 불변 봉인(v2 Policy)을 결합한 제로 트러스트 CI/CD 구축.

### Context
실제 거래소 API 키와 자금을 다루는 트레이딩 봇은 키 탈취 공격의 표적이 되기 쉬우며, 연구 환경의 파라미터가 배포 과정에서 실수로 변조되거나 설정이 누락되는 설정 드리프트(Config Drift) 위험이 큽니다.

### Alternatives
- **GitHub Secrets 평문 주입 및 공인 IP SSH**: 인바운드 방화벽 개방에 따른 해킹 위험.
- **비암호화 로컬 설정 파일**: 저장소 유출 시 거래소 자산 탈취 위험.

### Selected Approach
1. GitHub Actions 러너가 Tailscale WireGuard 사설 터널을 통해서만 원격 오라클 서버에 접속.
2. Git 저장소 내 `.env.enc` 및 전략 파라미터는 Age 공개키로 암호화하고 서버 배포 시 인메모리 복호화.
3. 라이브 데몬 구동 시 전략 파라미터의 SHA-256 해시를 대조하여 1바이트라도 불일치하면 `ArtifactSealError`로 즉시 기동 중단.

### Rationale
자격증명 평문 노출을 방지하고, 백테스트에서 검증된 파라미터가 실거래 런타임에 동일하게 적용되도록 보장함.

### Trade-offs
배포 파이프라인에 SOPS 복호화 및 키 관리 단계가 추가됨.
