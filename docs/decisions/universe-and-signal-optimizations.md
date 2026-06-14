# ADR: Universe SQLite Migration & 2D Alignment Cache

## Context
During `phase signal` runs, the universe ledger's repeated full Parquet read/write rewrites and redundant 2D time-series data alignment (`align_data_maps`) operations caused severe disk I/O throttling and pipeline bottleneck.

## Decision
1. **SQLite Migration**: Replaced the Parquet ledger with SQLite database (`universe_ledger.db`) utilizing a composite unique index `(symbol, tf, date, knowledge_date)` for idempotent UPSERTs and index-sliced query loading.
2. **Short-circuit Bypass**: Implemented early return checks in the data synchronizer using life-cycle metadata so that already updated symbols skip all Parquet disk scanning and network API calls entirely.
3. **Alignment Cache**: Injected a data map lookup cache inside `align_data_maps` to bypass repetitive Outer Join and matrix sorting calculations.

## Consequences
- Pre-synchronization scans and database load times dropped from tens of seconds to under 0.01s (instant skip).
- Redundant 2D alignment computation time is eliminated, enabling immediate transition to downstream L1 model training.
