"""S7: Anchored purged fold pool + post-book concurrent diagnostics.

Extracted verbatim from ``evaluation.py`` lines 4061-4172 (``_run_post_book_concurrently``
call, post-fold committee/multi-feature diagnostic opt-ins, deflated-sharpe
evidence, fold blend parity / growth concentration, the research-GO gate, and
the deployment-readiness patch). Calls the existing isolated functions unchanged;
only variable threading via ``ctx`` moves. The ``del eligible`` / ``del opens,
bar_funding`` / ``del funding_window, minute_grid`` + ``gc.collect()`` at 4169-4172
are preserved at the end of this function.
"""

from __future__ import annotations

import dataclasses
import gc
from types import SimpleNamespace

import pandas as pd

from src.application.research.mhs import research_go as _research_go
from src.application.research.mhs import scaling as _scaling
from src.application.research.mhs import statistics as _statistics
from src.application.research.mhs.evaluation import (
    COMMITTEE_MEMBERS,
    COMMITTEE_OOS_START,
    FEATURE_REGISTRY,
    DataIntegrityError,
    compute_deployment_readiness,
    feature_registry_panel_columns,
    phase_1_anchored_purged_folds,
)
from src.application.research.mhs.marks import _get_symbol_mark_frame
from src.application.research.mhs.resources import _assert_stage_rss_budget
from src.application.research.mhs.stage_services import (
    _committee_diagnostic,
    _fold_blend_parity,
    _fold_growth_concentration,
    _fold_realized_risk_parity,
    _guard_stage_or_breach,
    _load_feature_panels,
    _multi_feature_diagnostic,
    _pooled_fold_evidence,
    _run_post_book_concurrently,
)
from src.mhs.calibration import NullShareCalibration, calibrate_max_share_null
from src.mhs.evidence import selection_overlap_fraction
from src.mhs.params import PERIODS_PER_YEAR_1H as _PERIODS_PER_YEAR_1H
from src.mhs.pipeline.context import PipelineContext
from src.mhs.run_history import derive_trials_attempted
from src.mhs.telemetry import StageTelemetry


def run_folds(ctx: PipelineContext, telemetry: StageTelemetry) -> None:
    """Run the fold pool and all post-book statistical diagnostics."""
    ctx.trials_attempted, ctx.trials_attempted_source = derive_trials_attempted()
    ctx.deflated_sharpe_ratio = None
    # Observational disclosure of the defaults' selection window overlap.
    ctx.selection_overlap_fraction = selection_overlap_fraction(ctx.start, ctx.end)

    ctx.bootstrap_ci = None
    ctx.placebo_percentile = None
    ctx.participation = {}
    ctx.termination_counts = {}
    ctx.unsupported = (
        "partial_fill", "queue_position", "post_only_rejection",
        "cancel_replace_latency", "order_size_impact",
    )

    # Folds, statistical diagnostics, and deployment readiness are independent
    # post-book streams: the fold pool runs in fork workers while a background
    # thread computes the diagnostics + deployment tail (spec Phase 3, P14).
    # The top-level feature matrices stay alive through that thread and are
    # released after it joins so the wide multi-year panels never coexist with
    # the final assembly.
    ctx._fold_growth_budget_target_vol = None
    ctx._fold_exposure_warmup_returns = None
    ctx._fold_blend_exposure_scale = None
    if (
        ctx.config.pnl_vol_target_mode in ("growth_budget", "constant_risk")
        and ctx.blend_report is not None
        and ctx.blend_report.pre_vol_target_reference is not None
    ):
        # I2/I3/I4: each boundary's target vol is fit once here, on reference
        # rows strictly before that boundary's train_end, and only the small
        # float mapping crosses into the fork workers.
        _reference_daily_returns = (
            ctx.blend_report.pre_vol_target_reference.ledger.equity.resample("1D").last().pct_change()
        )
        if ctx.config.pnl_vol_target_mode == "constant_risk":
            # I-SCALE-IS-DEPLOYED-OVERLAY: exposure_scale은 blend가 배치 확정한
            # 리스크 오버레이로, fold는 자신의 검증 구간만큼 슬라이스해 읽기만
            # 한다 -- fold-local EWMA 재적합은 FOLD_BLEND_PATH_DIVERGENCE의
            # 실측 원인이므로 금지. 누락 시 침묵 폴백 없이 fail-closed.
            _exposure_scale = ctx.blend_report.exposure_scale
            if ctx.blend_report is None or _exposure_scale is None:
                raise DataIntegrityError(
                    "constant_risk folds require the blend book's deployed "
                    f"exposure_scale (pnl_vol_target_mode={ctx.config.pnl_vol_target_mode})"
                )
            ctx._fold_blend_exposure_scale = {
                _i: _exposure_scale.loc[
                    (_exposure_scale.index >= _f.validation_start)
                    & (_exposure_scale.index <= _f.validation_end)
                ]
                for _i, _f in enumerate(phase_1_anchored_purged_folds())
            }
        else:
            _train_ends = {"top_level": COMMITTEE_OOS_START}
            _train_ends.update({
                f"fold_{_i}": _f.train_end
                for _i, _f in enumerate(phase_1_anchored_purged_folds())
            })
            _boundary_target_vols = _scaling._growth_budget_target_vol_by_boundary(
                _reference_daily_returns, _research_go._resolved_growth_envelope(ctx.config), _train_ends,
            )
            ctx._fold_growth_budget_target_vol = {
                _i: _boundary_target_vols[f"fold_{_i}"]
                for _i in range(len(phase_1_anchored_purged_folds()))
            }
        ctx._fold_exposure_warmup_returns = _reference_daily_returns
    (
        ctx.bootstrap_ci, ctx.placebo_percentile, ctx.participation, ctx.termination_counts,
        fold_reports, ctx.deployment,
    ) = _run_post_book_concurrently(
        ctx.blend_report, ctx.root, ctx.config, ctx.execution_symbols, ctx.minute_grid,
        ctx.signal_48h, ctx.eligible, ctx.opens, ctx.bar_funding, ctx.grid_1h, ctx.fast,
        ctx.fold_funding, ctx.initial_equity, ctx.recorder, ctx.fold_slow_horizons, ctx.fold_fast_horizons,
        ctx.fold_funding_carry, ctx._fold_committee_weights, ctx._fold_growth_budget_target_vol,
        ctx._fold_exposure_warmup_returns,
        fold_blend_exposure_scale=ctx._fold_blend_exposure_scale,
    )
    ctx.folds = tuple(fold_reports)
    # Free mark frame cache so opt-in diagnostics run with minimal parent memory.
    _get_symbol_mark_frame.cache_clear()
    gc.collect()
    _terminal = _guard_stage_or_breach(
        "post_folds", ctx.rss_budget_bytes, ctx.rss_reserve_bytes,
        ctx.config, ctx.recorder, str(ctx.resolved_end), str(ctx.start), str(ctx.end),
    )
    if _terminal is not None:
        ctx._terminal_report = _terminal
        return
    if ctx.config.multi_feature_book or ctx.config.committee_book:
        if ctx.config.multi_feature_book:
            _diag_panel_columns = feature_registry_panel_columns(FEATURE_REGISTRY)
        else:
            _diag_panel_columns = feature_registry_panel_columns(
                [
                    spec for spec in FEATURE_REGISTRY
                    if spec.name in set(COMMITTEE_MEMBERS)
                ],
            )
        _diag_panels = _load_feature_panels(
            ctx.root, ctx.start, ctx.end, ctx.grid_1h, ctx.aligned_symbols, columns=_diag_panel_columns,
        )
        ctx.recorder.record("diagnostic_feature_panels")
        _assert_stage_rss_budget("diagnostic_feature_panels", ctx.rss_budget_bytes, ctx.rss_reserve_bytes)
        if ctx.config.committee_book:
            ctx.committee_diagnostic = _committee_diagnostic(
                ctx.root, ctx.start, ctx.end, ctx.grid_1h, ctx.aligned_symbols, ctx.execution_mask, ctx.opens,
                ctx.bar_funding, panels=_diag_panels,
                rss_budget_bytes=ctx.rss_budget_bytes,
                rss_reserve_bytes=ctx.rss_reserve_bytes,
                telemetry=ctx.recorder,
                sizing_mode="kelly_blend" if ctx.config.committee_kelly_sizing else "vol_target",
                growth_diagnostic=ctx.config.committee_growth_diagnostic,
            )
            ctx.recorder.record("committee_diagnostic")
        if ctx.config.multi_feature_book:
            ctx.multi_feature_diagnostic = _multi_feature_diagnostic(
                ctx.root, ctx.start, ctx.end, ctx.grid_1h, ctx.aligned_symbols, ctx.execution_mask, ctx.opens,
                ctx.bar_funding, panels=_diag_panels,
                rss_budget_bytes=ctx.rss_budget_bytes,
                rss_reserve_bytes=ctx.rss_reserve_bytes,
                telemetry=ctx.recorder,
            )
            ctx.recorder.record("multi_feature_diagnostic")
        del _diag_panels
        gc.collect()
    ctx.deflated_sharpe_ratio = _statistics._deflated_sharpe_evidence(
        ctx.blend_report, ctx.folds, ctx.trials_attempted,
    )
    ctx.fold_blend_parity, parity_reasons = _fold_blend_parity(ctx.blend_traces, ctx.folds)
    # I-FAMILY level 증거: fold별 min이 아니라 pooled 하한으로 판정한다.
    ctx._pooled_fold_evidence = _pooled_fold_evidence(ctx.folds)
    # 관측치 선계산(임계값 무관 경계): 보정 임계값 도출의 observed_share 입력.
    _observed_concentration, _observed_reasons = _fold_growth_concentration(ctx.folds, 1.0)
    _share_calibration: NullShareCalibration | SimpleNamespace
    if ctx._pooled_fold_evidence["n_measured_folds"] >= 2:
        _measured_reports = [
            f for f in ctx.folds
            if f.fold_index not in set(ctx._pooled_fold_evidence["unmeasured"])
        ]
        _measured_anchor = _measured_reports[0]
        _fold_days = int(
            (
                pd.Timestamp(_measured_anchor.validation_end)
                - pd.Timestamp(_measured_anchor.validation_start)
            ).days
        )
        # I-CALIB/I-DETERMINISTIC: 임계값은 등록 alpha와 자기 null에서 파생된다.
        _share_calibration = calibrate_max_share_null(
            ctx._pooled_fold_evidence["pooled_strict_returns"],
            len(ctx.folds),
            _fold_days,
            float(_observed_concentration["max_fold_share"]),
        )
        ctx.evidence_calibration = {
            "null_alpha": _share_calibration.alpha,
            "max_share_threshold": _share_calibration.threshold,
            "max_share_null_percentile": _share_calibration.observed_percentile,
            "pooled_sharpe_lcb": ctx._pooled_fold_evidence["pooled_sharpe_lcb"],
            "pooled_stress_sharpe_lcb": ctx._pooled_fold_evidence["pooled_stress_sharpe_lcb"],
            "n_pooled_days": ctx._pooled_fold_evidence["n_pooled_days"],
        }
        _share_threshold = _share_calibration.threshold
    else:
        # 측정 불가: dispersion 게이트를 구경계(threshold=1.0, share<=1 불가능
        # 영역)로 비활성한다. 보정값을 어떤 기본 임계값으로도 대체하지 않는다.
        ctx.evidence_calibration = None
        _share_calibration = SimpleNamespace(threshold=1.0)
    ctx.fold_growth_concentration, concentration_reasons = _fold_growth_concentration(ctx.folds, _share_calibration.threshold)
    level_reasons = _research_go._pooled_level_gate_reasons(ctx._pooled_fold_evidence)
    # 관측 전용 진단: 항상 빈 튜플이며 Research-GO reason 합산에 더하지 않는다.
    ctx.fold_realized_risk_parity, _risk_parity_reasons = _fold_realized_risk_parity(ctx.folds)
    # 관측 전용 공시 코드(선택창 겹침)만 조건부로 추가된다. 차단 코드는 아니다.
    extra_reasons: tuple[str, ...] = parity_reasons + concentration_reasons + level_reasons
    if ctx.selection_overlap_fraction > 0:
        extra_reasons = (
            *extra_reasons,
            _research_go.GO_REASON_SELECTION_WINDOW_OVERLAP,
        )
    ctx.research_go = _research_go._mhs_research_go(
        ctx.folds, ctx.book_reasons, extra_reasons,
        blend_primary_max_drawdown=(
            ctx.blend_report.primary_max_drawdown if ctx.blend_report is not None else None
        ),
        max_drawdown=_research_go._resolved_growth_envelope(ctx.config).max_drawdown,
        deflated_sharpe_ratio=ctx.deflated_sharpe_ratio,
    )

    if ctx.blend_report is not None and ctx.blend_report.primary is not None:
        if ctx.minute_grid is None:
            raise DataIntegrityError("blend report requires a minute replay grid")
        # The deployment tail was computed with ``research_go_eligible=None``;
        # patch in the fold-derived gate decision now that it is resolved.
        assert ctx.deployment is not None
        ctx.deployment = dataclasses.replace(
            ctx.deployment, research_go_eligible=ctx.research_go.eligible,
        )
        ctx.recorder.record(
            "blend_participation",
            fill_count=len(ctx.blend_report.primary.simulated_fills),
        )
        ctx.recorder.record("statistical_diagnostics")
    else:
        ctx.deployment = compute_deployment_readiness(
            pd.Series(
                [1.0, 1.0],
                index=pd.DatetimeIndex([ctx.start, ctx.start + pd.Timedelta(hours=1)]),
            ),
            _PERIODS_PER_YEAR_1H,
            research_go_eligible=ctx.research_go.eligible,
            n_bootstrap=_statistics._BOOTSTRAP_REPLICATES,
        )

    del ctx.eligible
    del ctx.opens, ctx.bar_funding
    del ctx.funding_window, ctx.minute_grid
    gc.collect()
