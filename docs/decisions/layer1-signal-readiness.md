# ADR: Layer1 Signal Readiness Workflow Refactor

- **Delta**: Layer1 검증 파이프라인을 Nested Anchored Walk-Forward 구조로 개편하고, target contract를 net에서 gross로 단일화.
- **Rationale**: Inner selection과 Outer evaluation을 격리하여 selection bias를 완전 제거하고, 비용(fee, slippage 등)을 Layer2로 이관하여 도메인 간의 책임 한계를 확실히 분리하기 위함.
