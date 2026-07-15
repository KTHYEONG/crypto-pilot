from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.application.futures.run_contracts import FuturesRunConfig
    from src.application.futures.runner.active_pipeline import TfProbeStageResult
    from src.domain.futures.strategy.config import CandidateStrategyConfig
    from src.domain.futures.strategy.timeframe_probe import (
        TfCellEvidence,
        TfProbeManifest,
    )

_TF_PROBE_FALLBACK_SYMBOLS: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "BNBUSDT")


@dataclass(frozen=True, slots=True)
class TfDiagnosticSamplingPolicy:
    confidence_level: float
    target_margin: float
    max_symbols: int
    seed: int


@dataclass(frozen=True, slots=True)
class TfDiagnosticSample:
    population_size: int
    required_size: int
    selected_symbols: tuple[str, ...]
    stratum_population: Mapping[str, int]
    stratum_sample: Mapping[str, int]
    representative: bool
    achieved_margin: float


@dataclass(slots=True, frozen=True)
class SymbolMeta:
    symbol: str
    rank: int
    cluster: int


def resolve_tf_diagnostic_sample(
    *,
    symbol_metadata: Sequence[SymbolMeta],
    available_symbols: Collection[str],
    policy: TfDiagnosticSamplingPolicy,
) -> TfDiagnosticSample:
    import math

    pop_size = len(symbol_metadata)
    z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(policy.confidence_level, 1.96)
    e = policy.target_margin
    n_req = math.ceil((pop_size * z**2 * 0.25) / (e**2 * (pop_size - 1) + z**2 * 0.25)) if pop_size > 1 else pop_size
    n_cap = min(n_req, policy.max_symbols, len(available_symbols))

    rng = __import__("random").Random(policy.seed)
    strata: dict[int, list[SymbolMeta]] = {}
    for meta in symbol_metadata:
        if meta.symbol in available_symbols:
            strata.setdefault(meta.cluster, []).append(meta)

    stratified: list[SymbolMeta] = []
    stratum_pop: dict[str, int] = {}
    stratum_samp: dict[str, int] = {}
    remaining = n_cap
    for cluster_id in sorted(strata):
        group = strata[cluster_id]
        rng.shuffle(group)
        skey = str(cluster_id)
        stratum_pop[skey] = len(group)
        alloc = max(1, round(n_cap * len(group) / pop_size)) if pop_size > 0 else 0
        alloc = min(alloc, len(group), remaining)
        stratified.extend(group[:alloc])
        stratum_samp[skey] = alloc
        remaining -= alloc
    if remaining > 0:
        extra = [m for m in symbol_metadata if m.symbol in available_symbols and m not in stratified]
        rng.shuffle(extra)
        stratified.extend(extra[:remaining])
        for m in extra[:remaining]:
            skey = str(m.cluster)
            stratum_samp[skey] = stratum_samp.get(skey, 0) + 1

    selected = tuple(m.symbol for m in stratified)
    achieved = e * math.sqrt(n_req / max(n_cap, 1)) if n_cap > 0 else 1.0
    representative = n_cap >= n_req and len(strata) == len(stratum_samp)
    return TfDiagnosticSample(
        population_size=pop_size,
        required_size=n_req,
        selected_symbols=selected,
        stratum_population=stratum_pop,
        stratum_sample=stratum_samp,
        representative=representative,
        achieved_margin=achieved,
    )


def _run_tf_probe_stage_scoped(
    *,
    run_config: FuturesRunConfig,
    full_strategy_maps: dict[str, dict[str, Any]],
    probe_cfg: CandidateStrategyConfig,
    scope_symbols: Sequence[str] | None = None,
) -> TfProbeStageResult | None:
    """[ADR_20260705_TF_PROBE_SCOPED_SYNC] Probe를 clear 이전 full_strategy_maps로 실행하고
    심볼 스코프를 majors-only 기본값으로 제한하는 안전 래퍼.

    내부적으로 기존 probe_timeframe_alpha, select_tf_family_cells,
    summarize_tf_probe_gate_audit를 그대로 재사용 — 신규 수학 없음.
    """
    from src.application.futures.runner.active_pipeline import (
        _format_counter_items,
        _logger,
    )
    from src.domain.futures.optimization.opt_config import OPT_FUTURES_CONFIG

    resolved_scope: Sequence[str] = scope_symbols if scope_symbols is not None else _TF_PROBE_FALLBACK_SYMBOLS
    _max_scope = 20
    if len(resolved_scope) > _max_scope:
        raise ValueError(f"scope_symbols has {len(resolved_scope)} symbols, exceeds max {_max_scope} (OOM guard)")
    if not OPT_FUTURES_CONFIG.get("ENABLE_TF_PROBE", False):
        return None

    from src.domain.futures.strategy.execution_cost import ExecutionCostModel
    from src.domain.futures.strategy.timeframe_probe import (
        probe_timeframe_alpha,
        select_tf_family_cells,
        summarize_tf_probe_gate_audit,
    )

    tf_grid: list[str] = list(OPT_FUTURES_CONFIG.get("TF_PROBE_GRID", ["4h"]))
    max_workers: int = int(OPT_FUTURES_CONFIG.get("TF_PROBE_MAX_WORKERS", 8))
    min_tstat: float = float(OPT_FUTURES_CONFIG.get("TF_PROBE_MIN_TSTAT", 2.0))
    require_fdr: bool = bool(OPT_FUTURES_CONFIG.get("TF_PROBE_REQUIRE_FDR", True))
    min_fold_cons: float = float(OPT_FUTURES_CONFIG.get("TF_PROBE_MIN_FOLD_CONSISTENCY", 0.75))

    try:
        manifest: TfProbeManifest = probe_timeframe_alpha(
            data_maps=full_strategy_maps,
            symbols=list(resolved_scope),
            base_cfg=probe_cfg,
            tf_grid=tf_grid,
            max_workers=max_workers,
            round_trip_cost_bps=ExecutionCostModel().round_trip_bps(),
        )
        winning: tuple[TfCellEvidence, ...] = select_tf_family_cells(
            manifest,
            min_ic_tstat=min_tstat,
            require_fdr=require_fdr,
            min_fold_sign_consistency=min_fold_cons,
        )
        selected_tfs: frozenset[str] = frozenset(c.tf for c in winning)
        winning_by_tf: dict[str, list[TfCellEvidence]] = {}
        for cell in winning:
            winning_by_tf.setdefault(cell.tf, []).append(cell)
        from src.domain.futures.strategy.tiered_logging import format_layer_header
        _logger.info(format_layer_header(0, "TF-Probe Gate Survivorship & Selection"))
        _logger.info(
            "[L0-PROBE] %d winning cells across %d tf: %s",
            len(winning),
            len(selected_tfs),
            sorted(selected_tfs),
        )
        _logger.info("")

        gate_rows = summarize_tf_probe_gate_audit(
            manifest,
            min_ic_tstat=min_tstat,
            require_fdr=require_fdr,
            min_fold_sign_consistency=min_fold_cons,
        )
        gate_by_tf = {row.tf: row for row in gate_rows}

        for idx, tf_i in enumerate(manifest.tf_grid):
            row = gate_by_tf.get(tf_i)
            if row is None:
                continue
            status_emoji = "🟢 SELECT" if row.winning > 0 else "🔴 REJECT"
            funnel_text = (
                f"Funnel: Candidates {row.computed} ➔ t-stat {row.pass_tstat} "
                f"➔ FDR {row.pass_fdr} ➔ Edge {row.pass_net_edge} ➔ Fold {row.pass_fold_consistency} "
                f"(Win: {row.winning})"
            )
            if row.winning == 0:
                _logger.info(
                    f"  ├── {tf_i:<4} : {status_emoji} ({row.winning}/{row.computed} cells) "
                    f"| Top Fail: {row.top_fail_reason}"
                )
                _logger.info(f"  │         └── {funnel_text}")
            else:
                tf_cells = winning_by_tf.get(tf_i, [])
                top_variants = _format_counter_items(
                    Counter(f"{c.family}:{c.variant}" for c in tf_cells),
                    limit=2,
                )
                _logger.info(f"  ├── {tf_i:<4} : {status_emoji} ({row.winning}/{row.computed} cells)")
                _logger.info(f"  │         ├── {funnel_text}")
                _logger.info(f"  │         └── Top: {top_variants}")
            if idx < len(manifest.tf_grid) - 1:
                _logger.info("  │")
        from src.application.futures.runner.active_pipeline import TfProbeStageResult

        return TfProbeStageResult(
            scan_manifest=manifest,
            qualified_cells=winning,
            selected_timeframes=selected_tfs,
        )
    except Exception as exc:
        _logger.warning("[TF-PROBE-SCOPED] probe stage failed (fallback to None): %s", exc)
        return None


def probe_stage_result_to_raw_manifest(
    probe_result: TfProbeStageResult | None,
) -> list[dict[str, Any]] | None:
    """Convert scoped TfProbeStageResult into raw probe_manifest rows for run_tiered_pipeline.

    [ADR_20260705_TF_VALIDATION_ROOT_CAUSE_CAPTURE]
    """
    if probe_result is None:
        return None
    rows: list[dict[str, Any]] = []
    for cell in probe_result.manifest.cells:
        tf = getattr(cell, "tf", None)
        if not tf:
            continue
        try:
            from dataclasses import asdict

            row = asdict(cell)
        except (TypeError, ValueError):
            row = {
                k: getattr(cell, k)
                for k in (
                    "symbol",
                    "family",
                    "variant",
                    "archetype",
                    "tf",
                    "n_obs",
                    "n_events",
                    "ic_mean",
                    "ic_tstat_hac",
                    "ic_fold_sign_consistency",
                    "alpha_half_life_h",
                    "net_edge_bps",
                    "turnover_per_year",
                    "vr_label",
                    "hurst",
                    "passed_fdr",
                )
                if hasattr(cell, k)
            }
        rows.append(row)
    return rows
