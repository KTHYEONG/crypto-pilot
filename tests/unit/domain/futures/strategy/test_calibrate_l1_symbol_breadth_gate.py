"""Tests for calibrate_l1_symbol_breadth_gate.py — propose_thresholds logic and measurement hook."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from pytest_mock import MockerFixture

from src.application.futures.runner.models import RunnerResult


def test_propose_thresholds_happy_path() -> None:
    """Scenario 1 (Happy Path): values clipped within [_FLOOR, _CEIL]."""
    from src.domain.futures.strategy.calibrate_l1_symbol_breadth_gate import propose_thresholds

    proposals = propose_thresholds({"12h": 1.4, "1d": 1.1})
    assert proposals["12h"] == pytest.approx(1.4)
    assert proposals["1d"] == pytest.approx(1.1)


def test_propose_thresholds_omits_nan() -> None:
    """Scenario 2 (Edge — LIMIT-03): NaN values omitted from output."""
    from src.domain.futures.strategy.calibrate_l1_symbol_breadth_gate import propose_thresholds

    proposals = propose_thresholds({"4h": float("nan")})
    assert "4h" not in proposals


def test_propose_thresholds_clamps_to_ceiling() -> None:
    """Scenario 2b (Edge — LIMIT-02 ceiling): measured 5.0 clips to _CEIL=3.0."""
    from src.domain.futures.strategy.calibrate_l1_symbol_breadth_gate import propose_thresholds

    proposals = propose_thresholds({"12h": 5.0, "1d": 0.5})
    assert proposals["12h"] == pytest.approx(3.0)
    assert proposals["1d"] == pytest.approx(1.0)


def test_propose_thresholds_empty_input() -> None:
    """Empty input dict returns empty proposals."""
    from src.domain.futures.strategy.calibrate_l1_symbol_breadth_gate import propose_thresholds

    assert propose_thresholds({}) == {}


def test_measure_effective_sym_n_by_tf_seeds_trace_with_stage_order(mocker: MockerFixture) -> None:
    """Regression guard: run_once must receive trace pre-seeded with STAGE_ORDER keys."""
    from src.domain.futures.strategy.tiered_workflow.cross_tf_diagnostics import STAGE_ORDER

    captured_trace: dict[str, Any] = {}

    def _fake_run_once(
        *, label: str, tfs: tuple[str, ...], ablate_1h_fusion: bool, trace: dict[str, Any]
    ) -> RunnerResult:
        captured_trace.update(trace)
        return RunnerResult(exit_code=0, reason="l1_mode_done")

    mocker.patch("src.domain.futures.strategy.calibrate_l1_symbol_breadth_gate.run_once", side_effect=_fake_run_once)

    from src.domain.futures.strategy.calibrate_l1_symbol_breadth_gate import measure_effective_sym_n_by_tf

    measure_effective_sym_n_by_tf()

    assert set(STAGE_ORDER).issubset(captured_trace.keys())
    assert all(isinstance(v, dict) for v in captured_trace.values())


def test_measure_effective_sym_n_by_tf_captures_per_tf(mocker: MockerFixture) -> None:
    """Integration: the run_per_tf_l1/evaluate_layer1_readiness wiring fires and captures per-TF."""
    from src.domain.futures.strategy.candidate_contracts import Layer1FoldReadiness
    from src.domain.futures.strategy.config import CandidateStrategyConfig
    from src.domain.futures.strategy.tiered_workflow import pipeline as _pipeline_mod

    _pipeline = cast(Any, _pipeline_mod)

    captured_tfs: list[str] = []

    def _fake_orig_per_tf(*, tf: str, **kwargs: Any) -> None:
        captured_tfs.append(tf)
        report = Layer1FoldReadiness(
            fold_id=0,
            registry_source_end_idx=0,
            outer_oos_start_idx=0,
            outer_oos_end_idx=100,
            ready_symbols=("BTCUSDT", "ETHUSDT"),
            matched_event_count=10,
            unmatched_event_count=0,
            realized_match_ratio=1.0,
            unique_decision_count=5,
            prediction_unique_count=0,
            opportunity_ic=None,
            opportunity_ic_tstat=0.0,
            probe_bps=10.0,
            probe_lcb_bps=5.0,
            probe_series_bps=(10.0, 12.0),
            effective_symbol_count=2.0,
            passed=True,
            blockers=(),
        )
        _pipeline.evaluate_layer1_readiness(
            fold_reports=(report,),
            fold_cov=1.0,
            trade_scope_count=10,
            cfg=CandidateStrategyConfig(),
            seed=42,
        )

    mocker.patch.object(_pipeline_mod, "run_per_tf_l1", side_effect=_fake_orig_per_tf)

    def _fake_run_once(
        *, label: str, tfs: tuple[str, ...], ablate_1h_fusion: bool, trace: dict[str, Any]
    ) -> RunnerResult:
        for tf in tfs:
            _pipeline.run_per_tf_l1(tf=tf)
        return RunnerResult(exit_code=0, reason="l1_mode_done")

    mocker.patch("src.domain.futures.strategy.calibrate_l1_symbol_breadth_gate.run_once", side_effect=_fake_run_once)

    from src.domain.futures.strategy.calibrate_l1_symbol_breadth_gate import measure_effective_sym_n_by_tf

    result = measure_effective_sym_n_by_tf()

    assert set(result.keys()) == set(captured_tfs)


def test_main_writes_calibration_artifact_and_does_not_touch_config(
    mocker: MockerFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() writes the proposal artifact and never imports/mutates config.py directly."""
    mocker.patch(
        "src.domain.futures.strategy.calibrate_l1_symbol_breadth_gate.measure_effective_sym_n_by_tf",
        return_value={"4h": 2.5, "12h": 1.2, "1d": 1.1},
    )
    monkeypatch.chdir(tmp_path)

    from src.domain.futures.strategy.calibrate_l1_symbol_breadth_gate import main

    exit_code = main()

    assert exit_code == 0
    out_path = tmp_path / "logs" / "futures" / "diagnostics" / "l1_symbol_breadth_calibration.json"
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    assert artifact["note"].startswith("NOT auto-applied")
    assert "4h" in artifact["proposed_l1_min_effective_sym_n"]
    assert "12h" in artifact["proposed_l1_min_effective_sym_n"]
    assert "1d" in artifact["proposed_l1_min_effective_sym_n"]


def test_propose_cross_section_thresholds_happy_path() -> None:
    """Scenario 1 (Happy Path): values clipped within [1, 2]."""
    from src.domain.futures.strategy.calibrate_l1_symbol_breadth_gate import propose_cross_section_thresholds

    proposals = propose_cross_section_thresholds({"12h": 1.0, "1d": 1.4})
    assert proposals["12h"] == pytest.approx(1.0)
    assert proposals["1d"] == pytest.approx(1.4)


def test_propose_cross_section_thresholds_clamps_to_ceiling() -> None:
    """Scenario 2b (Edge — LIMIT-01 ceiling): measured 3.0 clips to ceiling=2.0."""
    from src.domain.futures.strategy.calibrate_l1_symbol_breadth_gate import propose_cross_section_thresholds

    proposals = propose_cross_section_thresholds({"8h": 3.0, "12h": 0.5})
    assert proposals["8h"] == pytest.approx(2.0)
    assert proposals["12h"] == pytest.approx(1.0)


def test_propose_cross_section_thresholds_omits_nan() -> None:
    """Scenario 2 (Edge): NaN values omitted from output."""
    from src.domain.futures.strategy.calibrate_l1_symbol_breadth_gate import propose_cross_section_thresholds

    proposals = propose_cross_section_thresholds({"6h": float("nan")})
    assert "6h" not in proposals


def test_measure_fold_min_ready_symbols_by_tf_excludes_registry_empty_folds(mocker: MockerFixture) -> None:
    """Scenario 4 (Integration — LIMIT-01): registry_empty folds are excluded from
    the p10 computation; only non-empty folds' raw ready_symbols count contributes."""
    from src.domain.futures.strategy.candidate_contracts import Layer1FoldReadiness
    from src.domain.futures.strategy.config import CandidateStrategyConfig
    from src.domain.futures.strategy.tiered_workflow import pipeline as _pipeline_mod

    _pipeline = cast(Any, _pipeline_mod)

    def _fake_orig_per_tf(*, tf: str, **kwargs: Any) -> None:
        empty_report = Layer1FoldReadiness(
            fold_id=0,
            registry_source_end_idx=0,
            outer_oos_start_idx=0,
            outer_oos_end_idx=100,
            ready_symbols=(),
            matched_event_count=0,
            unmatched_event_count=0,
            realized_match_ratio=0.0,
            unique_decision_count=0,
            prediction_unique_count=0,
            opportunity_ic=None,
            opportunity_ic_tstat=0.0,
            probe_bps=0.0,
            probe_lcb_bps=0.0,
            probe_series_bps=(),
            effective_symbol_count=0.0,
            passed=False,
            blockers=("empty_opportunities:registry_empty",),
        )
        non_empty_report = Layer1FoldReadiness(
            fold_id=1,
            registry_source_end_idx=0,
            outer_oos_start_idx=0,
            outer_oos_end_idx=100,
            ready_symbols=("LUNA2USDT",),
            matched_event_count=10,
            unmatched_event_count=0,
            realized_match_ratio=1.0,
            unique_decision_count=5,
            prediction_unique_count=0,
            opportunity_ic=None,
            opportunity_ic_tstat=0.0,
            probe_bps=229.12,
            probe_lcb_bps=100.0,
            probe_series_bps=(229.12,),
            effective_symbol_count=1.0,
            passed=False,
            blockers=("insufficient_ready_symbols",),
        )
        _pipeline.evaluate_layer1_readiness(
            fold_reports=(empty_report, non_empty_report),
            fold_cov=1.0,
            trade_scope_count=10,
            cfg=CandidateStrategyConfig(),
            seed=42,
        )

    mocker.patch.object(_pipeline_mod, "run_per_tf_l1", side_effect=_fake_orig_per_tf)

    def _fake_run_once(
        *, label: str, tfs: tuple[str, ...], ablate_1h_fusion: bool, trace: dict[str, Any]
    ) -> RunnerResult:
        for tf in tfs:
            _pipeline.run_per_tf_l1(tf=tf)
        return RunnerResult(exit_code=0, reason="l1_mode_done")

    mocker.patch("src.domain.futures.strategy.calibrate_l1_symbol_breadth_gate.run_once", side_effect=_fake_run_once)

    from src.domain.futures.strategy.calibrate_l1_symbol_breadth_gate import measure_fold_min_ready_symbols_by_tf

    result = measure_fold_min_ready_symbols_by_tf()

    for tf, p10 in result.items():
        assert p10 == pytest.approx(1.0), f"tf={tf} should reflect only the non-empty fold (1 ready symbol)"


def test_main_writes_cross_section_proposal_keys(
    mocker: MockerFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() also writes the fold-level cross-section measurement/proposal keys."""
    mocker.patch(
        "src.domain.futures.strategy.calibrate_l1_symbol_breadth_gate.measure_effective_sym_n_by_tf",
        return_value={"4h": 2.5},
    )
    mocker.patch(
        "src.domain.futures.strategy.calibrate_l1_symbol_breadth_gate.measure_fold_min_ready_symbols_by_tf",
        return_value={"12h": 1.0, "1d": 1.0},
    )
    monkeypatch.chdir(tmp_path)

    from src.domain.futures.strategy.calibrate_l1_symbol_breadth_gate import main

    exit_code = main()

    assert exit_code == 0
    out_path = tmp_path / "logs" / "futures" / "diagnostics" / "l1_symbol_breadth_calibration.json"
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    assert artifact["proposed_l1_min_cross_section"] == {"12h": 1.0, "1d": 1.0}
    assert artifact["measured_p10_fold_ready_symbols_by_tf"] == {"12h": 1.0, "1d": 1.0}


def test_script_importable_via_direct_invocation_path(tmp_path: Path) -> None:
    """Regression guard: `python scripts/calibrate_l1_symbol_breadth_gate.py` must not ModuleNotFoundError."""
    import subprocess
    import sys as _sys

    repo_root = Path(__file__).resolve().parents[4]
    script_path = repo_root / "src" / "domain" / "futures" / "strategy" / "calibrate_l1_symbol_breadth_gate.py"
    try:
        result = subprocess.run(  # noqa: S603
            [_sys.executable, str(script_path)],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            timeout=5,
        )
    except subprocess.TimeoutExpired:
        return
    assert "ModuleNotFoundError" not in result.stderr, result.stderr
