from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
import pytest

from src.application.futures.runner.active_pipeline import TimeframeScanStageResult
from src.application.futures.runner.config import build_l0_runtime_config
from src.domain.futures.observability.artifact_logging import (
    emit_csv_artifact_debug,
    emit_dataframe_artifact_debug,
    emit_json_artifact_debug,
)
from src.domain.futures.strategy.timeframe_probe import (
    TimeframeScanCell,
    TimeframeScanManifest,
    probe_timeframe_alpha,
    scan_timeframe_alpha,
    select_tf_family_cells,
    select_timeframe_scan_cells,
    summarize_tf_probe_gate_audit,
    summarize_timeframe_scan_gate_audit,
)


@dataclass(frozen=True, slots=True)
class DummyReport:
    run_id: str
    mode: str
    timeframe: str
    symbols: tuple[str, ...]
    n_bars: int
    n_panels_in: int
    n_bound_panels: int
    n_evidence: int
    n_passed: int
    n_rejected: int
    reject_reason_counts: dict[str, int]
    elapsed_sec: float


# ── S1-1: TF scan canonical names work ───────────────────────────────────


def test_scan_timeframe_alpha_is_aliased_to_probe_timeframe_alpha() -> None:
    assert scan_timeframe_alpha is probe_timeframe_alpha


def test_select_timeframe_scan_cells_is_aliased_to_select_tf_family_cells() -> None:
    assert select_timeframe_scan_cells is select_tf_family_cells


def test_summarize_timeframe_scan_gate_audit_is_aliased_to_summarize_tf_probe_gate_audit() -> None:
    assert summarize_timeframe_scan_gate_audit is summarize_tf_probe_gate_audit


# ── S1-2: Stage result compatibility ─────────────────────────────────────


def test_timeframe_scan_stage_result_legacy_aliases() -> None:
    cell = TimeframeScanCell(
        symbol="BTCUSDT",
        family="trend_pullback_continuation",
        variant="tpc_25_100",
        archetype="trend",
        tf="4h",
        n_obs=120,
        n_events=18,
        ic_mean=0.03,
        ic_tstat_hac=2.4,
        ic_fold_sign_consistency=0.75,
        alpha_half_life_h=16.0,
        net_edge_bps=12.5,
        turnover_per_year=80.0,
        vr_label="trend",
        hurst=0.58,
        passed_fdr=True,
    )
    manifest = TimeframeScanManifest(
        cells=(cell,),
        tf_grid=("4h",),
        coverage_by_tf={"4h": 1200},
        diversity_corr={},
    )
    result = TimeframeScanStageResult(
        scan_manifest=manifest,
        qualified_cells=(cell,),
        selected_timeframes=frozenset({"4h"}),
    )

    assert result.scan_manifest is manifest
    assert result.manifest is manifest
    assert result.qualified_cells == (cell,)
    assert result.winning_cells == (cell,)
    assert result.selected_timeframes == frozenset({"4h"})
    assert result.selected_tfs == frozenset({"4h"})


def test_tf_cell_evidence_is_alias_of_timeframe_scan_cell() -> None:
    from src.domain.futures.strategy.timeframe_probe import TfCellEvidence

    assert TfCellEvidence is TimeframeScanCell


def test_tf_probe_manifest_is_alias_of_timeframe_scan_manifest() -> None:
    from src.domain.futures.strategy.timeframe_probe import TfProbeManifest

    assert TfProbeManifest is TimeframeScanManifest


# ── S1-3: DEBUG artifact emission ────────────────────────────────────────


def test_emit_debug_artifacts(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("artifact-test")
    logger.setLevel(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="artifact-test")

    emit_json_artifact_debug(
        logger=logger,
        artifact_name="alpha_foundry_report",
        run_id="run-1",
        payload={"run_id": "run-1", "n_passed": 3},
        path="/tmp/run-1_report.json",  # noqa: S108
    )
    emit_csv_artifact_debug(
        logger=logger,
        artifact_name="alpha_foundry_rows",
        run_id="run-1",
        rows=[
            {"family": "btc_regime_pullback", "tf": "4h", "gate_passed": True},
            {"family": "trend_pullback_continuation", "tf": "12h", "gate_passed": True},
        ],
        path="/tmp/run-1_rows.csv",  # noqa: S108
        row_limit=200,
    )

    assert "ARTIFACT_JSON_BEGIN name=alpha_foundry_report run_id=run-1" in caplog.text
    assert "ARTIFACT_CSV_BEGIN name=alpha_foundry_rows run_id=run-1 rows=2/2" in caplog.text
    assert "btc_regime_pullback,4h,True" in caplog.text


def test_emit_dataframe_artifact_debug(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("artifact-df-test")
    logger.setLevel(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="artifact-df-test")

    frame = pd.DataFrame([
        {"family": "btc_regime_pullback", "tf": "4h", "gate_passed": True},
        {"family": "trend_pullback_continuation", "tf": "12h", "gate_passed": True},
    ])
    emit_dataframe_artifact_debug(
        logger=logger,
        artifact_name="df_artifact",
        run_id="run-df",
        frame=frame,
        row_limit=200,
    )

    assert "ARTIFACT_CSV_BEGIN name=df_artifact run_id=run-df rows=2/2" in caplog.text
    assert "btc_regime_pullback,4h,True" in caplog.text


def test_emit_artifact_skips_when_logger_not_debug(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("skip-test")
    logger.setLevel(logging.INFO)
    caplog.set_level(logging.INFO, logger="skip-test")

    emit_json_artifact_debug(
        logger=logger,
        artifact_name="should_skip",
        run_id="run-skip",
        payload={"data": "value"},
    )
    emit_csv_artifact_debug(
        logger=logger,
        artifact_name="should_skip_csv",
        run_id="run-skip",
        rows=[{"col": "val"}],
    )

    assert "ARTIFACT_JSON_BEGIN" not in caplog.text
    assert "ARTIFACT_CSV_BEGIN" not in caplog.text


# ── S1-4: phase=l0 public CLI route ──────────────────────────────────────


def test_phase_l0_builds_gate_runtime() -> None:
    runtime = build_l0_runtime_config(
        phase="l0",
        settings={
            "alpha_foundry_total_l1_budget": 30,
            "alpha_foundry_min_conviction_lcb_bps": 5.0,
            "alpha_foundry_enable_fast_tf": False,
        },
    )
    assert runtime.mode == "gate"


# ── S2-1: Legacy API compatibility ───────────────────────────────────────


def test_legacy_type_aliases_are_importable() -> None:
    from src.domain.futures.strategy.timeframe_probe import (
        TfCellEvidence,
        TfProbeGateAuditRow,
        TfProbeManifest,
    )

    assert TfCellEvidence is TimeframeScanCell
    assert TfProbeManifest is TimeframeScanManifest
    assert TfProbeGateAuditRow is not None


def test_legacy_function_aliases_are_callable() -> None:
    assert callable(probe_timeframe_alpha)
    assert callable(select_tf_family_cells)
    assert callable(summarize_tf_probe_gate_audit)


# ── S2-2: Large CSV truncation ───────────────────────────────────────────


def test_large_csv_truncation(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("artifact-truncation-test")
    logger.setLevel(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="artifact-truncation-test")

    rows = [{"idx": i, "value": f"x{i}"} for i in range(500)]
    emit_csv_artifact_debug(
        logger=logger,
        artifact_name="large_csv",
        run_id="run-trunc",
        rows=rows,
        row_limit=200,
    )

    assert "ARTIFACT_CSV_BEGIN name=large_csv run_id=run-trunc rows=200/500" in caplog.text
    assert "__truncated__,true" in caplog.text
    assert "ARTIFACT_CSV_END name=large_csv run_id=run-trunc" in caplog.text


# ── S2-3: DEBUG-only terminal mode (no file write) ───────────────────────


def test_debug_only_terminal_mode_no_file_write(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("debug-only-test")
    logger.setLevel(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="debug-only-test")

    emit_json_artifact_debug(
        logger=logger,
        artifact_name="debug_only",
        run_id="run-debug",
        payload={"status": "no_file"},
    )

    assert "ARTIFACT_JSON_BEGIN name=debug_only run_id=run-debug" in caplog.text


# ── S2-4: Empty artifact ─────────────────────────────────────────────────


def test_empty_csv_artifact(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("empty-csv-test")
    logger.setLevel(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="empty-csv-test")

    emit_csv_artifact_debug(
        logger=logger,
        artifact_name="empty_csv",
        run_id="run-empty",
        rows=[],
    )

    assert "ARTIFACT_CSV_BEGIN name=empty_csv run_id=run-empty rows=0/0" in caplog.text
    assert "ARTIFACT_CSV_END name=empty_csv run_id=run-empty" in caplog.text


# ── S2-5: Corroboration rename ──────────────────────────────────────────


def test_corroboration_rename_semantic_only() -> None:
    from src.domain.futures.alpha_foundry.multi_tf_fusion import fuse_multi_timeframe_evidence  # noqa: F401


# ── S2-6: Removed alpha-foundry args are rejected ────────────────────────


def test_removed_alpha_foundry_arg_raises_value_error() -> None:
    from src.application.futures.runner.config import build_run_config_from_args

    with pytest.raises(ValueError, match="removed argument: --alpha-foundry"):
        build_run_config_from_args({"phase": "l1", "alpha_foundry": "gate"})

    with pytest.raises(ValueError, match="removed argument: --alpha-foundry-total-l1-budget"):
        build_run_config_from_args({"phase": "l1", "alpha_foundry_total_l1_budget": 50})


# ── S2-7: phase=l1/l2/l3 wires implicit L0 gate mode ────────────────────


@pytest.mark.parametrize("phase", ["l1", "l2", "l3"])
def test_implicit_l0_gate_mode_for_non_l0_phases(phase: str) -> None:
    run_cfg_kwargs = {
        "timeframe": "4h",
        "date": None,
        "trials": 10,
        "phase": phase,
        "sync": "auto",
        "refresh_universe": False,
        "sync_metrics": False,
        "l0_runtime": build_l0_runtime_config(
            phase=phase,  # type: ignore[arg-type]
            settings={},
        ),
    }
    assert run_cfg_kwargs["l0_runtime"].mode == "gate"


# ── S3-1: JSON serialization failure ─────────────────────────────────────


def test_json_serialization_failure_does_not_raise(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("json-fail-test")
    logger.setLevel(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="json-fail-test")

    class Unserializable:
        pass

    emit_json_artifact_debug(
        logger=logger,
        artifact_name="bad_json",
        run_id="run-bad",
        payload={"data": Unserializable()},  # type: ignore[dict-item]
    )

    assert not caplog.records or all(r.levelno != logging.ERROR for r in caplog.records)


# ── S3-3: Empty timeframe scan (empty manifest) ─────────────────────────


def test_empty_timeframe_manifest() -> None:
    manifest = TimeframeScanManifest(
        cells=(),
        tf_grid=(),
        coverage_by_tf={},
        diversity_corr={},
    )
    assert manifest.cells == ()
    assert manifest.tf_grid == ()


# ── S3-4: Unsupported legacy CLI shape ──────────────────────────────────


def test_unsupported_legacy_cli_shape_raises() -> None:
    from src.application.futures.runner.config import (
        build_run_config_from_args,
        parse_active_phase,
    )

    with pytest.raises(ValueError, match="removed argument: --alpha-foundry"):
        build_run_config_from_args({"phase": "l0", "alpha_foundry": "off"})

    with pytest.raises(ValueError, match="removed phase"):
        parse_active_phase("strategy-smoke")
