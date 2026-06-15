---
title: Futures Universe Ledger Backend Compatibility
domain: futures.universe
type: adr
status: active
priority: high
ai_read_policy: when_related
---

## 2026-06-15 Ledger backend compatibility recovery
- **Delta:** `load_ledger_slice(...)` now dispatches by backend suffix and supports both SQLite and parquet fixtures through the same PIT filter path.
- **Rationale:** universe tests and offline snapshots depend on parquet inputs; the loader must not collapse existing files into silent empty stage0 results.
- **Edge Cases:** missing files may still return empty frames, but readable files that fail backend-specific loading now raise with explicit backend context.
