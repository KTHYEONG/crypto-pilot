from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.application.futures.optimization.config import FuturesRunConfig
    from src.application.futures.runner.active_pipeline import TfProbeStageResult
    from src.domain.futures.strategy.config import CandidateStrategyConfig
    from src.domain.futures.strategy.timeframe_probe import (
        TfCellEvidence,
        TfProbeManifest,
    )

_TF_PROBE_FALLBACK_SYMBOLS: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "BNBUSDT")


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
        _log_ascii_table,
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
        _logger.info(
            "[TF-PROBE-SCOPED] %d winning cells across %d tf: %s",
            len(winning),
            len(selected_tfs),
            sorted(selected_tfs),
        )
        rows: list[list[str]] = []
        for tf_i in manifest.tf_grid:
            tf_cells = winning_by_tf.get(tf_i, [])
            top_families = _format_counter_items(Counter(c.family for c in tf_cells), limit=2)
            top_variants = _format_counter_items(
                Counter(f"{c.family}:{c.variant}" for c in tf_cells),
                limit=2,
            )
            decision = "SELECT" if tf_cells else "REJECT"
            rows.append([tf_i, str(len(tf_cells)), top_families, top_variants, decision])
        _log_ascii_table(
            "[TF-PROBE-SCOPED AUDIT] TIMEFRAME SELECTION",
            ("TF", "Winning", "Families", "Variants", "Decision"),
            rows,
            (8, 10, 24, 34, 10),
        )
        gate_rows = summarize_tf_probe_gate_audit(
            manifest,
            min_ic_tstat=min_tstat,
            require_fdr=require_fdr,
            min_fold_sign_consistency=min_fold_cons,
        )
        audit_table_rows: list[list[str]] = [
            [
                row.tf,
                str(row.computed),
                str(row.pass_tstat),
                str(row.pass_fdr),
                str(row.pass_net_edge),
                str(row.pass_fold_consistency),
                str(row.winning),
                row.top_fail_reason,
            ]
            for row in gate_rows
        ]
        _log_ascii_table(
            "[TF-PROBE-SCOPED AUDIT] GATE SURVIVORSHIP",
            ("TF", "Cells", "Pass t", "Pass FDR", "Pass Edge", "Pass Fold", "Winning", "Top Fail"),
            audit_table_rows,
            (8, 8, 8, 10, 10, 10, 8, 16),
        )
        from src.application.futures.runner.active_pipeline import TfProbeStageResult

        return TfProbeStageResult(
            manifest=manifest,
            winning_cells=winning,
            selected_tfs=selected_tfs,
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
