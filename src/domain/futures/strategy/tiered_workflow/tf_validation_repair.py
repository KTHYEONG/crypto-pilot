from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal

from src.domain.futures.strategy.config import PerTfL1Result
from src.domain.futures.strategy.tiered_workflow.awf_sim import (
    MAJOR_DIAG_SYMBOLS,
    MajorSymbolRegistryCensusEntry,
    MajorSymbolSleeveContributionSummary,
)
from src.domain.futures.strategy.timeframe_probe import TfProbeManifest

logger = logging.getLogger("opt_main_futures")

TfProbeDecision = Literal["diagnostic_only"]
TfCandidateDecision = Literal["keep_existing", "reject_candidate", "review_candidate"]
MajorGapClass = Literal["admission_gap", "activation_gap", "outvoted", "no_gap"]
RepairAction = Literal[
    "none",
    "major_sign_rank_combiner_spec",
    "l1_family_evidence_spec",
    "activation_context_spec",
]


@dataclass(frozen=True, slots=True)
class TfProbeDiagnosticVerdict:
    tf: str
    source_ready: bool
    n_cells: int
    n_pass_tstat: int
    n_pass_fdr: int
    n_pass_net_edge: int
    n_pass_fold_consistency: int
    n_winning: int
    top_fail_reason: str
    decision: TfProbeDecision = "diagnostic_only"


@dataclass(frozen=True, slots=True)
class MainCompatibleTfEvidence:
    tf: str
    gate_passed: bool
    edge_quality_bps: float
    n_winning_signals: int
    n_ready_symbols: int
    registry_present: bool
    top_symbols: tuple[str, ...]
    top_families: tuple[str, ...]
    candidate_decision: TfCandidateDecision


@dataclass(frozen=True, slots=True)
class MajorSymbolGapEvidence:
    symbol: str
    tf: str
    family: str
    in_registry: bool
    hard_eligible: bool
    observed_active: bool
    registry_mean_incremental_bps: float
    regime_adverse_mismatch_pct: float
    mean_raw_mu: float
    mean_quality_weight: float
    gap_class: MajorGapClass
    repair_action: RepairAction


@dataclass(frozen=True, slots=True, init=False)
class ValidationParityReport:
    """Final TF validation parity report with scan-diagnostics, main, and major-gap evidence.

    [ADR_20260705_TF_VALIDATION_ROOT_CAUSE_CAPTURE]
    """

    probe: tuple[TfProbeDiagnosticVerdict, ...]
    main_tf: tuple[MainCompatibleTfEvidence, ...]
    major_gaps: tuple[MajorSymbolGapEvidence, ...]
    decision: Literal["diagnostic_only", "candidate_review_required"]
    blockers: tuple[str, ...]

    def __init__(
        self,
        probe: tuple[TfProbeDiagnosticVerdict, ...] | None = None,
        main_tf: tuple[MainCompatibleTfEvidence, ...] | None = None,
        major_gaps: tuple[MajorSymbolGapEvidence, ...] | None = None,
        decision: Literal["diagnostic_only", "candidate_review_required"] = "diagnostic_only",
        blockers: tuple[str, ...] = (),
        **kwargs: Any,
    ) -> None:
        scan_diagnostics = kwargs.pop("scan_diagnostics", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected keyword arguments: {unexpected}")
        final_probe = scan_diagnostics if scan_diagnostics is not None else probe
        if final_probe is None:
            final_probe = ()
        if main_tf is None:
            main_tf = ()
        if major_gaps is None:
            major_gaps = ()
        object.__setattr__(self, "probe", final_probe)
        object.__setattr__(self, "main_tf", main_tf)
        object.__setattr__(self, "major_gaps", major_gaps)
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "blockers", blockers)

    @property
    def scan_diagnostics(self) -> tuple[TfProbeDiagnosticVerdict, ...]:
        return self.probe


ValidationPhase = Literal["l1", "l2", "l3"]


@dataclass(frozen=True, slots=True, init=False)
class ValidationParityCapture:
    """Pre-clear TF validation parity capture used to finalize parity reports.

    [ADR_20260705_TF_VALIDATION_ROOT_CAUSE_CAPTURE]
    """

    probe: tuple[TfProbeDiagnosticVerdict, ...]
    main_tf: tuple[MainCompatibleTfEvidence, ...]
    registry_census: tuple[tuple[str, MajorSymbolRegistryCensusEntry], ...]
    blockers: tuple[str, ...]
    decision: Literal["diagnostic_only", "candidate_review_required"]
    candidate_tfs: tuple[str, ...]
    symbols: tuple[str, ...] = MAJOR_DIAG_SYMBOLS

    def __init__(
        self,
        probe: tuple[TfProbeDiagnosticVerdict, ...] | None = None,
        main_tf: tuple[MainCompatibleTfEvidence, ...] | None = None,
        registry_census: tuple[tuple[str, MajorSymbolRegistryCensusEntry], ...] | None = None,
        blockers: tuple[str, ...] = (),
        decision: Literal["diagnostic_only", "candidate_review_required"] = "diagnostic_only",
        candidate_tfs: tuple[str, ...] = (),
        symbols: tuple[str, ...] = MAJOR_DIAG_SYMBOLS,
        **kwargs: Any,
    ) -> None:
        scan_diagnostics = kwargs.pop("scan_diagnostics", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected keyword arguments: {unexpected}")
        final_probe = scan_diagnostics if scan_diagnostics is not None else probe
        if final_probe is None:
            final_probe = ()
        if main_tf is None:
            main_tf = ()
        if registry_census is None:
            registry_census = ()
        object.__setattr__(self, "probe", final_probe)
        object.__setattr__(self, "main_tf", main_tf)
        object.__setattr__(self, "registry_census", registry_census)
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "candidate_tfs", candidate_tfs)
        object.__setattr__(self, "symbols", symbols)

    @property
    def scan_diagnostics(self) -> tuple[TfProbeDiagnosticVerdict, ...]:
        return self.probe


def _edge_quality_bps(r: PerTfL1Result) -> float:
    """Replicate _tf_edge_quality logic for summary."""
    panel = r.l1_result.strategy_panel
    if not panel:
        return float(r.n_winning_signals)
    total: float = 0.0
    for s in panel:
        valid = getattr(s, "valid", False)
        edge = getattr(s, "oos_edge_bps", 0.0)
        if valid and edge > 0.0:
            total += edge
    return total


def _pass_counts_for_tf(
    cells: Sequence[Any],
    *,
    min_ic_tstat: float,
    require_fdr: bool,
    min_net_edge_bps: float,
    min_fold_sign_consistency: float,
) -> dict[str, int]:
    n = len(cells)
    n_tstat = sum(1 for c in cells if c.ic_tstat_hac >= min_ic_tstat)
    n_fdr = sum(1 for c in cells if c.passed_fdr) if require_fdr else n
    n_edge = sum(1 for c in cells if c.net_edge_bps >= min_net_edge_bps)
    n_fold = sum(1 for c in cells if c.ic_fold_sign_consistency >= min_fold_sign_consistency)
    n_win = sum(
        1
        for c in cells
        if c.ic_tstat_hac >= min_ic_tstat
        and (not require_fdr or c.passed_fdr)
        and c.net_edge_bps >= min_net_edge_bps
        and c.ic_fold_sign_consistency >= min_fold_sign_consistency
    )
    return {
        "n_cells": n,
        "n_pass_tstat": n_tstat,
        "n_pass_fdr": n_fdr,
        "n_pass_net_edge": n_edge,
        "n_pass_fold_consistency": n_fold,
        "n_winning": n_win,
    }


def _top_fail_reason(
    cells: Sequence[Any],
    *,
    min_ic_tstat: float,
    require_fdr: bool,
    min_net_edge_bps: float,
    min_fold_sign_consistency: float,
) -> str:
    for c in cells:
        reasons: list[str] = []
        if c.ic_tstat_hac < min_ic_tstat:
            reasons.append(f"tstat({c.ic_tstat_hac:.4f})")
        if require_fdr and not c.passed_fdr:
            reasons.append("fdr")
        if c.net_edge_bps < min_net_edge_bps:
            reasons.append(f"edge({c.net_edge_bps:.4f})")
        if c.ic_fold_sign_consistency < min_fold_sign_consistency:
            reasons.append(f"fold_cons({c.ic_fold_sign_consistency:.4f})")
        if reasons:
            return ";".join(reasons)
    return "all_passed"


def summarize_tf_probe_diagnostics(
    manifest: TfProbeManifest | None,
    *,
    min_ic_tstat: float = 2.0,
    require_fdr: bool = True,
    min_net_edge_bps: float = 0.0,
    min_fold_sign_consistency: float = 0.75,
) -> tuple[TfProbeDiagnosticVerdict, ...]:
    if manifest is None:
        return ()
    tf_cells: dict[str, list[Any]] = {}
    for cell in manifest.cells:
        tf_cells.setdefault(cell.tf, []).append(cell)

    verdicts: list[TfProbeDiagnosticVerdict] = []
    for tf in manifest.tf_grid:
        cells = tf_cells.get(tf, [])
        counts = _pass_counts_for_tf(
            cells,
            min_ic_tstat=min_ic_tstat,
            require_fdr=require_fdr,
            min_net_edge_bps=min_net_edge_bps,
            min_fold_sign_consistency=min_fold_sign_consistency,
        )
        fail_reason = _top_fail_reason(
            cells,
            min_ic_tstat=min_ic_tstat,
            require_fdr=require_fdr,
            min_net_edge_bps=min_net_edge_bps,
            min_fold_sign_consistency=min_fold_sign_consistency,
        )
        verdicts.append(
            TfProbeDiagnosticVerdict(
                tf=tf,
                source_ready=len(cells) > 0,
                n_cells=counts["n_cells"],
                n_pass_tstat=counts["n_pass_tstat"],
                n_pass_fdr=counts["n_pass_fdr"],
                n_pass_net_edge=counts["n_pass_net_edge"],
                n_pass_fold_consistency=counts["n_pass_fold_consistency"],
                n_winning=counts["n_winning"],
                top_fail_reason=fail_reason,
            )
        )
    return tuple(verdicts)


def _read_registry_top_symbols(
    registry: Any,
    *,
    max_symbols: int = 3,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    by_symbol = getattr(registry, "by_symbol", {})
    top_symbols = tuple(sorted(by_symbol.keys())[:max_symbols])
    families: set[str] = set()
    for ev_list in by_symbol.values():
        if ev_list:
            for ev in ev_list:
                fam = getattr(ev, "family", None) or getattr(getattr(ev, "key", None), "strategy_id", "-").split(":")[0]
                families.add(fam)
    return top_symbols, tuple(sorted(families))


def _decide_candidate(
    tf: str,
    candidate_tfs: Sequence[str],
    edge_quality_bps: float,
    registry_present: bool,
    n_ready_symbols: int,
) -> TfCandidateDecision:
    if tf in candidate_tfs:
        if n_ready_symbols > 0 and edge_quality_bps > 0:
            return "review_candidate"
        return "reject_candidate"
    if edge_quality_bps > 0 or n_ready_symbols > 0:
        return "keep_existing"
    return "review_candidate"


def _raw_probe_to_manifest(
    raw: list[dict[str, Any]] | None,
) -> Any:
    """Convert raw list[dict] probe from pipeline to TfProbeManifest-like SimpleNamespace."""
    if raw is None:
        return None
    cells = [SimpleNamespace(**c) for c in raw]
    tfs: list[str] = []
    seen: set[str] = set()
    for c in cells:
        tf = getattr(c, "tf", "")
        if tf and tf not in seen:
            seen.add(tf)
            tfs.append(tf)
    return SimpleNamespace(
        cells=tuple(cells),
        tf_grid=tuple(tfs),
        coverage_by_tf={},
        diversity_corr={},
    )


def summarize_main_compatible_tf_evidence(
    per_tf_l1: Mapping[str, PerTfL1Result],
    *,
    candidate_tfs: Sequence[str] = ("1h", "2h"),
) -> tuple[MainCompatibleTfEvidence, ...]:
    evidence: list[MainCompatibleTfEvidence] = []
    for tf in sorted(per_tf_l1.keys()):
        r = per_tf_l1[tf]
        edge = _edge_quality_bps(r)
        registry = r.l1_result.deployment_registry
        registry_present = registry is not None
        ready_symbols = getattr(registry, "ready_symbols", ()) if registry is not None else ()
        n_ready = len(ready_symbols)
        top_symbols, top_families = _read_registry_top_symbols(registry) if registry_present else ((), ())
        decision = _decide_candidate(
            tf=tf,
            candidate_tfs=candidate_tfs,
            edge_quality_bps=edge,
            registry_present=registry_present,
            n_ready_symbols=n_ready,
        )
        evidence.append(
            MainCompatibleTfEvidence(
                tf=tf,
                gate_passed=r.l1_result.gate_passed,
                edge_quality_bps=edge,
                n_winning_signals=r.n_winning_signals,
                n_ready_symbols=n_ready,
                registry_present=registry_present,
                top_symbols=top_symbols,
                top_families=top_families,
                candidate_decision=decision,
            )
        )
    return tuple(evidence)


def build_multi_tf_major_registry_census(
    per_tf_l1: Mapping[str, PerTfL1Result],
    observed_sleeve_summaries: Sequence[MajorSymbolSleeveContributionSummary],
    *,
    symbols: tuple[str, ...] = MAJOR_DIAG_SYMBOLS,
) -> tuple[tuple[str, MajorSymbolRegistryCensusEntry], ...]:
    census: list[tuple[str, MajorSymbolRegistryCensusEntry]] = []
    for tf in sorted(per_tf_l1.keys()):
        r = per_tf_l1[tf]
        registry = r.l1_result.deployment_registry
        if registry is None:
            artifact = getattr(r.l1_result, "inference_artifact", None)
            if artifact is not None:
                registry = getattr(artifact, "deployment_registry", None)
        if registry is None:
            continue
        by_symbol = getattr(registry, "by_symbol", {})
        if not by_symbol:
            continue
        for sym in symbols:
            ev_list = by_symbol.get(sym, ())
            if not ev_list:
                continue
            for ev in ev_list:
                fam = getattr(ev, "family", None) or getattr(getattr(ev, "key", None), "strategy_id", "-").split(":")[0]
                mean_bps = getattr(ev, "mean_incremental_bps", 0.0)
                hard = getattr(ev, "hard_eligible", False)
                observed = any(s.symbol == sym and s.family == fam for s in observed_sleeve_summaries)
                entry = MajorSymbolRegistryCensusEntry(
                    symbol=sym,
                    family=fam,
                    registry_mean_incremental_bps=mean_bps,
                    hard_eligible=hard,
                    observed_active_in_holdout=observed,
                )
                census.append((tf, entry))
    return tuple(census)


def classify_major_symbol_gap_evidence(
    *,
    tf: str,
    entry: MajorSymbolRegistryCensusEntry | None,
    symbol: str,
    family: str,
    observed_sleeve_summaries: Sequence[MajorSymbolSleeveContributionSummary],
    adverse_sign_mismatch_threshold: float = 0.50,
) -> MajorSymbolGapEvidence:
    obs = _find_observed_summary(symbol, family, observed_sleeve_summaries)

    if entry is None:
        gap_class: MajorGapClass = "admission_gap"
        repair: RepairAction = "l1_family_evidence_spec"
        in_registry = False
        hard_eligible = False
        observed_active = obs is not None
        reg_bps = 0.0
        mismatch = obs.regime_adverse_sign_mismatch_pct if obs is not None else 0.0
        mu = obs.mean_raw_mu_sleeve if obs is not None else 0.0
        qw = obs.mean_quality_weight_sleeve if obs is not None else 0.0
    elif not entry.hard_eligible:
        gap_class = "admission_gap"
        repair = "l1_family_evidence_spec"
        in_registry = True
        hard_eligible = False
        observed_active = entry.observed_active_in_holdout
        reg_bps = entry.registry_mean_incremental_bps
        mismatch = obs.regime_adverse_sign_mismatch_pct if obs is not None else 0.0
        mu = obs.mean_raw_mu_sleeve if obs is not None else 0.0
        qw = obs.mean_quality_weight_sleeve if obs is not None else 0.0
    elif not entry.observed_active_in_holdout:
        gap_class = "activation_gap"
        repair = "activation_context_spec"
        in_registry = True
        hard_eligible = True
        observed_active = False
        reg_bps = entry.registry_mean_incremental_bps
        mismatch = obs.regime_adverse_sign_mismatch_pct if obs is not None else 0.0
        mu = obs.mean_raw_mu_sleeve if obs is not None else 0.0
        qw = obs.mean_quality_weight_sleeve if obs is not None else 0.0
    else:
        mismatch = obs.regime_adverse_sign_mismatch_pct if obs is not None else 0.0
        if mismatch >= adverse_sign_mismatch_threshold:
            gap_class = "outvoted"
            repair = "major_sign_rank_combiner_spec"
        else:
            gap_class = "no_gap"
            repair = "none"
        in_registry = True
        hard_eligible = True
        observed_active = True
        reg_bps = entry.registry_mean_incremental_bps
        mu = obs.mean_raw_mu_sleeve if obs is not None else 0.0
        qw = obs.mean_quality_weight_sleeve if obs is not None else 0.0

    return MajorSymbolGapEvidence(
        symbol=symbol,
        tf=tf,
        family=family,
        in_registry=in_registry,
        hard_eligible=hard_eligible,
        observed_active=observed_active,
        registry_mean_incremental_bps=reg_bps,
        regime_adverse_mismatch_pct=mismatch,
        mean_raw_mu=mu,
        mean_quality_weight=qw,
        gap_class=gap_class,
        repair_action=repair,
    )


def _find_observed_summary(
    symbol: str,
    family: str,
    summaries: Sequence[MajorSymbolSleeveContributionSummary],
) -> MajorSymbolSleeveContributionSummary | None:
    for s in summaries:
        if s.symbol == symbol and s.family == family:
            return s
    return None


def _extract_target_pairs_from_registry(
    r: PerTfL1Result,
    symbols: tuple[str, ...],
) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    registry = r.l1_result.deployment_registry
    if registry is None:
        return pairs
    by_symbol = getattr(registry, "by_symbol", {})
    for sym in symbols:
        ev_list = by_symbol.get(sym, ())
        for ev in ev_list:
            fam = getattr(ev, "family", None) or getattr(getattr(ev, "key", None), "strategy_id", "-").split(":")[0]
            pairs.add((sym, fam))
    return pairs


def _extract_target_pairs_from_summaries(
    summaries: Sequence[MajorSymbolSleeveContributionSummary],
) -> set[tuple[str, str]]:
    return {(s.symbol, s.family) for s in summaries}


def build_major_symbol_gap_evidence(
    *,
    per_tf_l1: Mapping[str, PerTfL1Result],
    observed_sleeve_summaries: Sequence[MajorSymbolSleeveContributionSummary],
    symbols: tuple[str, ...] = MAJOR_DIAG_SYMBOLS,
    families: tuple[str, ...] | None = None,
    adverse_sign_mismatch_threshold: float = 0.50,
) -> tuple[MajorSymbolGapEvidence, ...]:
    target_pairs: set[tuple[str, str]] = _extract_target_pairs_from_summaries(observed_sleeve_summaries)
    for tf in sorted(per_tf_l1.keys()):
        target_pairs |= _extract_target_pairs_from_registry(per_tf_l1[tf], symbols)

    evidence: list[MajorSymbolGapEvidence] = []
    census = {
        (tf, sym, fam): entry
        for tf, entry in build_multi_tf_major_registry_census(
            per_tf_l1,
            observed_sleeve_summaries,
            symbols=symbols,
        )
        for sym in [entry.symbol]
        for fam in [entry.family]
    }

    for sym, fam in sorted(target_pairs):
        if families is not None and fam not in families:
            continue
        for tf in sorted(per_tf_l1.keys()):
            entry = census.get((tf, sym, fam))
            gap = classify_major_symbol_gap_evidence(
                tf=tf,
                entry=entry,
                symbol=sym,
                family=fam,
                observed_sleeve_summaries=observed_sleeve_summaries,
                adverse_sign_mismatch_threshold=adverse_sign_mismatch_threshold,
            )
            evidence.append(gap)

    return tuple(evidence)


def build_validation_parity_report(
    *,
    probe_manifest: TfProbeManifest | None,
    per_tf_l1: Mapping[str, PerTfL1Result],
    observed_sleeve_summaries: Sequence[MajorSymbolSleeveContributionSummary] = (),
    candidate_tfs: Sequence[str] = ("1h", "2h"),
    symbols: tuple[str, ...] = MAJOR_DIAG_SYMBOLS,
) -> ValidationParityReport:
    blockers: list[str] = []

    probe = summarize_tf_probe_diagnostics(probe_manifest)

    if not per_tf_l1:
        blockers.append("missing_per_tf_l1")

    main_tf = summarize_main_compatible_tf_evidence(per_tf_l1, candidate_tfs=candidate_tfs)

    if not main_tf:
        blockers.append("missing_main_tf_evidence")

    blockers.extend(f"candidate_tf_missing_main_l1:{ctf}" for ctf in candidate_tfs if ctf not in per_tf_l1)

    major_gaps = build_major_symbol_gap_evidence(
        per_tf_l1=per_tf_l1,
        observed_sleeve_summaries=observed_sleeve_summaries,
        symbols=symbols,
    )

    needs_review = any(ev.candidate_decision == "review_candidate" for ev in main_tf)
    decision: Literal["diagnostic_only", "candidate_review_required"] = (
        "candidate_review_required" if needs_review else "diagnostic_only"
    )

    return ValidationParityReport(
        scan_diagnostics=probe,
        main_tf=main_tf,
        major_gaps=major_gaps,
        decision=decision,
        blockers=tuple(blockers),
    )


def log_validation_parity_report(
    report: ValidationParityReport,
    *,
    phase: ValidationPhase = "l1",
    log_level: int = logging.DEBUG,
) -> None:
    """Emit the parity report as structured logger lines.

    [ADR_20260705_TF_VALIDATION_ROOT_CAUSE_CAPTURE]
    """

    for v in report.probe:
        logger.log(
            log_level,
            "[TF-VALIDATION-PARITY] tf=%s probe_winning=%d decision=%s",
            v.tf,
            v.n_winning,
            v.decision,
        )
    for ev in report.main_tf:
        logger.log(
            log_level,
            "[TF-VALIDATION-PARITY] tf=%s main_edge=%.2f n_ready=%d decision=%s",
            ev.tf,
            ev.edge_quality_bps,
            ev.n_ready_symbols,
            ev.candidate_decision,
        )
    for g in report.major_gaps:
        logger.log(
            log_level,
            "[L1-MAJOR-GAP] symbol=%s tf=%s family=%s gap=%s action=%s",
            g.symbol,
            g.tf,
            g.family,
            g.gap_class,
            g.repair_action,
        )
    if report.blockers:
        for b in report.blockers:
            logger.log(log_level, "[TF-VALIDATION-PARITY] blocker=%s", b)


def build_validation_parity_capture(
    *,
    probe_manifest: TfProbeManifest | None,
    per_tf_l1: Mapping[str, PerTfL1Result],
    candidate_tfs: Sequence[str] = ("1h", "2h"),
    symbols: tuple[str, ...] = MAJOR_DIAG_SYMBOLS,
) -> ValidationParityCapture:
    """Capture pre-clear probe/main/census evidence without retaining per_tf_l1 objects.

    [ADR_20260705_TF_VALIDATION_ROOT_CAUSE_CAPTURE]
    """
    probe = summarize_tf_probe_diagnostics(probe_manifest)
    main_tf = summarize_main_compatible_tf_evidence(per_tf_l1, candidate_tfs=candidate_tfs)
    registry_census = build_multi_tf_major_registry_census(
        per_tf_l1,
        (),
        symbols=symbols,
    )
    blockers: list[str] = []
    if not per_tf_l1:
        blockers.append("missing_per_tf_l1")
    if not main_tf:
        blockers.append("missing_main_tf_evidence")
    blockers.extend(f"candidate_tf_missing_main_l1:{ctf}" for ctf in candidate_tfs if ctf not in per_tf_l1)
    needs_review = any(ev.candidate_decision == "review_candidate" for ev in main_tf)
    decision: Literal["diagnostic_only", "candidate_review_required"] = (
        "candidate_review_required" if needs_review else "diagnostic_only"
    )
    return ValidationParityCapture(
        scan_diagnostics=probe,
        main_tf=main_tf,
        registry_census=registry_census,
        blockers=tuple(blockers),
        decision=decision,
        candidate_tfs=tuple(candidate_tfs),
        symbols=symbols,
    )


def build_major_symbol_gap_evidence_from_census(
    *,
    registry_census: Sequence[tuple[str, MajorSymbolRegistryCensusEntry]],
    observed_sleeve_summaries: Sequence[MajorSymbolSleeveContributionSummary],
    symbols: tuple[str, ...] = MAJOR_DIAG_SYMBOLS,
    families: tuple[str, ...] | None = None,
    adverse_sign_mismatch_threshold: float = 0.50,
) -> tuple[MajorSymbolGapEvidence, ...]:
    """Build gap evidence from pre-clear census plus later observed sleeve summaries."""
    observed_pairs: set[tuple[str, str]] = _extract_target_pairs_from_summaries(observed_sleeve_summaries)
    evidence: list[MajorSymbolGapEvidence] = []
    seen_pairs: set[tuple[str, str, str]] = set()
    for tf, entry in registry_census:
        if families is not None and entry.family not in families:
            continue
        key = (entry.symbol, tf, entry.family)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        is_observed = (entry.symbol, entry.family) in observed_pairs
        if not entry.observed_active_in_holdout and is_observed:
            entry = MajorSymbolRegistryCensusEntry(
                symbol=entry.symbol,
                family=entry.family,
                registry_mean_incremental_bps=entry.registry_mean_incremental_bps,
                hard_eligible=entry.hard_eligible,
                observed_active_in_holdout=True,
            )
        gap = classify_major_symbol_gap_evidence(
            tf=tf,
            entry=entry,
            symbol=entry.symbol,
            family=entry.family,
            observed_sleeve_summaries=observed_sleeve_summaries,
            adverse_sign_mismatch_threshold=adverse_sign_mismatch_threshold,
        )
        evidence.append(gap)
    return tuple(evidence)


def finalize_validation_parity_capture(
    capture: ValidationParityCapture,
    *,
    observed_sleeve_summaries: Sequence[MajorSymbolSleeveContributionSummary] = (),
    adverse_sign_mismatch_threshold: float = 0.50,
) -> ValidationParityReport:
    """Convert pre-clear capture into final report using later L2/L3 observed sleeve evidence.

    [ADR_20260705_TF_VALIDATION_ROOT_CAUSE_CAPTURE]
    """
    major_gaps = build_major_symbol_gap_evidence_from_census(
        registry_census=capture.registry_census,
        observed_sleeve_summaries=observed_sleeve_summaries,
        symbols=capture.symbols,
        adverse_sign_mismatch_threshold=adverse_sign_mismatch_threshold,
    )
    return ValidationParityReport(
        scan_diagnostics=capture.scan_diagnostics,
        main_tf=capture.main_tf,
        major_gaps=major_gaps,
        decision=capture.decision,
        blockers=capture.blockers,
    )


def _should_emit_validation_parity_report(
    *,
    verbose: bool,
    phase: ValidationPhase,
    report: ValidationParityReport | None,
) -> bool:
    """Return True when the report should be visible in real runs."""
    return bool(verbose and report is not None)
