---
title: Futures Universe Architecture
domain: futures.universe
type: architecture
status: active
priority: high
ai_read_policy: when_related
related_paths:
  - src/domain/futures/universe/
  - src/application/futures/optimization/universe_service.py
change_triggers:
  - src/domain/futures/universe/**
last_verified: 2026-06-10
---

# 1. Purpose
Generates a Point-In-Time (PIT) valid, survivorship-bias-free trading universe through a strict 7-stage filtration funnel.

# 2. Core Logic & Math

**Execution Cost Estimation (Stage 4)**
- $\text{cost\_bps} = 2 \cdot \text{taker\_fee} + 2 \cdot \text{half\_spread} + \text{impact} + \text{tick\_cost}$
- Post-2020: `half_spread` = empirical median of `bookDepth`.
- Pre-2020: `half_spread` = Modified Corwin-Schultz OHLC model.

**Snapshot Quality Score**
- $\text{Score}_{\text{universe}} = \text{fill\_rate} \times \log_{10}\left(\frac{\text{median\_adv\_usdt}}{10^6}\right) \times \frac{1}{\text{mAEC\_bps}}$

**PIT Constraints**
- Strict requirement: `knowledge_date <= as_of`. No forward-looking metadata or delisting knowledge is allowed.

**7-Stage Funnel Hurdles**
- **S1 (Structure):** `listing_age_days >= 90`
- **S2 (Quality):** `min_is_coverage >= 0.80`, `min_coverage_60d >= 0.95`
- **S3 (Liquidity):** `adv_usdt_median >= 25M`, `max_amihud_30d <= 1.63e-9`
- **S4 (Cost):** `execution_cost_bps <= 50.0`
- **S5 (Risk):** $0.05 \leq \text{vol\_30d} \leq 4.0$, $|\text{funding\_zscore}| \leq 2.5$

# 3. Architecture Flow

```mermaid
graph TD
    A[Exchange Symbol List] --> B[S0: Eligibility]
    B --> C[S1: Structure & Listing Age]
    C --> D[S2: Data Quality]
    D --> E[S3: Liquidity]
    E --> F[S4: Execution Cost]
    F --> G[S5: Risk Events]
    G --> H[S6: Selection & Ranking]
    H --> I[PIT Universe Snapshot]
    I --> J[Downstream ML & Candidate Pipeline]
```

# 4. Core Variables & I/O

| Type | Variable | Description |
|------|----------|-------------|
| **Input** | `knowledge_date` | Point-in-time barrier for all historical data queries |
| **Param** | `min_listing_days` | Minimum days since listing to bypass structural gate (default: 90) |
| **Param** | `min_adv_usdt` | Minimum 30-day median trading volume (default: 25M) |
| **Param** | `max_exec_cost_bps` | Maximum tolerated round-trip execution cost (default: 50.0) |
| **Output**| `Universe Snapshot` | Static, timestamped set of valid symbols |
| **Output**| `Static Metadata` | Metrics passed downstream: `vol_30d`, `friction_score`, `beta_vs_market`, etc. |
