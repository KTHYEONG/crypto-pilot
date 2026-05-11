# System Architecture & Evolution Repository

This directory (`/.ai/`) maintains the **Source of Truth** for the my-coin-traider system architecture.

## Files

| File | Purpose | Audience |
|------|---------|----------|
| **DNA.json** | Current architectural state, parameters, performance metrics | Architects, Agents evaluating systems |
| **EVOLUTION.md** | Chronological log of major changes, lessons learned, experimental results | Developers, Agents planning improvements |
| **evaluation_criteria.md** | HMM evaluation standards & scoring framework | **Agents evaluating HMM regime classifier** |
| **archive/** | Legacy evolution logs (v1-v7) | Historical reference only |
| **experiments/** | Individual trial results, ablation studies | Research archive |

## For AI Agents: HMM Evaluation Protocol

When evaluating the HMM regime classifier:

1. **Read this README** to understand the repo structure
2. **Read `evaluation_criteria.md`** — this defines the PASS/FAIL criteria for HMM metrics
3. **Run test**: `uv run python tests/test_universe_to_hmm.py --tf 4h`
4. **Extract metrics** from AUDIT output (CRISIS MU, BEAR MU, CRISIS Share, Tail Capture, Avg Duration)
5. **Score against criteria** in `evaluation_criteria.md` table
6. **Update DNA.json** with new results if running experiment

## Current System State

**Version**: v9.0.0 (2026-05-12)  
**Architecture**: 3-Layer Hierarchical (MS-GARCH Vol Regime + Direction HMM + Rules Crisis Detector)  
**Status**: Production-ready with ongoing optimization  
**Bottleneck**: CRISIS MU -0.144% (target -0.2%), Lead-Lag Tail Capture not yet measured  

See `DNA.json` for full details.

## Key Architectural Decisions (v9.0)

1. **Decomposed single 5-state HMM → 3-Layer**: Separates vol clustering (GARCH, 12h) from direction detection (HMM, 6h) from crisis events (rules, 4h).
2. **MS-GARCH instead of Gaussian**: Native vol persistence and skew-t for natural left-tail emphasis. No penalty stacking.
3. **Rules-based Crisis Detector**: 6-rule soft scorer overrides HMM output. Preventive hedging via forward-looking triggers.

See `EVOLUTION.md` for rationale.

## Contact

For questions about evaluation framework, consult `evaluation_criteria.md`.  
For architectural questions, consult `DNA.json` and `EVOLUTION.md`.
