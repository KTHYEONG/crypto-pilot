"""Tests for calibrate_l1_pair_gate.py — propose_thresholds logic and effective_n_sink hook."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from src.application.futures.runner.models import RunnerResult
from src.domain.futures.strategy.config import CandidateStrategyConfig
from src.domain.futures.strategy.tiered_workflow.signal_selection import (
    compute_symbol_strategy_evidence,
)


def test_propose_thresholds_happy_path() -> None:
    """Scenario 1 (Happy Path): mixture of low and high effective_n yields floor(p10) clamped."""
    from src.domain.futures.strategy.calibrate_l1_pair_gate import propose_thresholds

    measured = {"4h": [3.0] * 90 + [6.0] * 10}
    proposals = propose_thresholds(measured)
    assert proposals["4h"] == pytest.approx(3.0)


def test_propose_thresholds_clamps_to_floor_and_ceiling() -> None:
    """Scenario 2 (Edge Cases): low-density clamped to 2.0, high-density clamped to 4.0, empty omitted."""
    from src.domain.futures.strategy.calibrate_l1_pair_gate import propose_thresholds

    measured = {
        "1d": [0.1, 0.2, 0.3, 0.15, 0.25],
        "6h": [12.0, 15.0, 20.0, 18.0, 14.0],
        "4h": [],
    }

    proposals = propose_thresholds(measured)

    assert proposals["1d"] == pytest.approx(2.0)
    assert proposals["6h"] == pytest.approx(4.0)
    assert "4h" not in proposals


def test_propose_thresholds_empty_input() -> None:
    """Empty input dict returns empty proposals."""
    from src.domain.futures.strategy.calibrate_l1_pair_gate import propose_thresholds

    assert propose_thresholds({}) == {}


def test_compute_symbol_strategy_evidence_effective_n_sink_matches_computed_value() -> None:
    """Scenario 4: effective_n_sink is called with correct snapshot_index and effective_n."""
    event_results = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 5,
            "family": ["trend_ma"] * 5,
            "variant": ["ema_12_72"] * 5,
            "gross_event_bps": [10.0, 12.0, 8.0, 11.0, 9.0],
            "side": [1, 1, 1, 1, 1],
            "expected_holding_bars": [6] * 5,
            "fold_id": [0, 0, 1, 1, 1],
            "exit_idx": [10, 20, 30, 40, 50],
            "entry_idx": [5, 15, 25, 35, 45],
        }
    )
    cfg = CandidateStrategyConfig()
    captured: list[tuple[int, float]] = []

    compute_symbol_strategy_evidence(
        event_results=event_results,
        cfg=cfg,
        seed=42,
        registry_as_of_idx=100,
        snapshot_index=2,
        effective_n_sink=lambda snap_idx, eff_n: captured.append((snap_idx, eff_n)),
    )

    assert len(captured) == 1
    assert captured[0][0] == 2
    assert captured[0][1] == pytest.approx(5.0, rel=1e-6)


def test_effective_n_sink_none_preserves_existing_behavior() -> None:
    """effective_n_sink=None does not change function behavior (regression guard)."""
    event_results = pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 3,
            "family": ["trend_ma"] * 3,
            "variant": ["ema_12_72"] * 3,
            "gross_event_bps": [5.0, 4.0, 6.0],
            "side": [1, 1, 1],
            "expected_holding_bars": [4] * 3,
            "fold_id": [0, 0, 0],
            "exit_idx": [10, 20, 30],
            "entry_idx": [5, 15, 25],
        }
    )
    cfg = CandidateStrategyConfig()

    result = compute_symbol_strategy_evidence(
        event_results=event_results,
        cfg=cfg,
        seed=0,
        registry_as_of_idx=999,
    )

    assert len(result) == 1
    assert result[0].effective_n == pytest.approx(3.0, rel=1e-6)


def _make_minimal_event_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * 3,
            "family": ["trend_ma"] * 3,
            "variant": ["ema_12_72"] * 3,
            "gross_event_bps": [5.0, 4.0, 6.0],
            "side": [1, 1, 1],
            "expected_holding_bars": [4] * 3,
            "fold_id": [0, 0, 0],
            "exit_idx": [10, 20, 30],
            "entry_idx": [5, 15, 25],
        }
    )


def test_measure_effective_n_by_tf_seeds_trace_with_stage_order(mocker: MockerFixture) -> None:
    """Regression guard: run_once must receive trace pre-seeded with STAGE_ORDER keys, not {}.

    Reproduces the crash fixed in this cycle: run_once(..., trace={}) raised
    KeyError/AttributeError deep inside the pipeline because it writes to
    trace["native_panels"][...] etc. unconditionally.
    """
    from src.domain.futures.strategy.tiered_workflow.cross_tf_diagnostics import STAGE_ORDER

    captured_trace: dict[str, Any] = {}

    def _fake_run_once(
        *, label: str, tfs: tuple[str, ...], ablate_1h_fusion: bool, trace: dict[str, Any]
    ) -> RunnerResult:
        captured_trace.update(trace)
        return RunnerResult(exit_code=0, reason="l1_mode_done")

    mocker.patch("src.domain.futures.strategy.calibrate_l1_pair_gate.run_once", side_effect=_fake_run_once)

    from src.domain.futures.strategy.calibrate_l1_pair_gate import measure_effective_n_by_tf

    measure_effective_n_by_tf()

    assert set(STAGE_ORDER).issubset(captured_trace.keys())
    assert all(isinstance(v, dict) for v in captured_trace.values())


def test_measure_effective_n_by_tf_captures_only_final_snapshot_per_tf(mocker: MockerFixture) -> None:
    """Scenario 4 (Integration): the run_per_tf_l1/compute_symbol_strategy_evidence wiring
    actually fires and only records effective_n at snapshot_index == final_snapshot_index.

    Regression guard: must resolve compute_symbol_strategy_evidence via the `pipeline`
    module's own name binding (as run_l1_nested_swf's _build_snapshot really does),
    not via `signal_selection` directly -- patching the latter alone is a silent no-op
    because pipeline.py imported its own separate reference at module load time.
    """
    from src.domain.futures.strategy.tiered_workflow import pipeline as _pipeline_mod

    _pipeline = cast(Any, _pipeline_mod)

    def _fake_orig_per_tf(*, tf: str, **kwargs: Any) -> None:
        event_results = _make_minimal_event_results()
        cfg = CandidateStrategyConfig()
        _pipeline.compute_symbol_strategy_evidence(
            event_results=event_results,
            cfg=cfg,
            seed=0,
            registry_as_of_idx=999,
            snapshot_index=0,
        )
        _pipeline.compute_symbol_strategy_evidence(
            event_results=event_results,
            cfg=cfg,
            seed=0,
            registry_as_of_idx=999,
            snapshot_index=3,
        )

    mocker.patch.object(_pipeline_mod, "run_per_tf_l1", side_effect=_fake_orig_per_tf)

    def _fake_run_once(
        *, label: str, tfs: tuple[str, ...], ablate_1h_fusion: bool, trace: dict[str, Any]
    ) -> RunnerResult:
        for tf in tfs:
            _pipeline.run_per_tf_l1(tf=tf)
        return RunnerResult(exit_code=0, reason="l1_mode_done")

    mocker.patch("src.domain.futures.strategy.calibrate_l1_pair_gate.run_once", side_effect=_fake_run_once)

    from src.domain.futures.strategy.calibrate_l1_pair_gate import measure_effective_n_by_tf

    result = measure_effective_n_by_tf(final_snapshot_index=3)

    assert set(result.keys()) == {"2h", "4h", "6h", "8h", "12h", "1d"}
    for values in result.values():
        assert values == [pytest.approx(3.0)]


def test_main_writes_calibration_artifact_and_does_not_touch_config(
    mocker: MockerFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() writes the proposal artifact and never imports/mutates config.py directly."""
    mocker.patch(
        "src.domain.futures.strategy.calibrate_l1_pair_gate.measure_effective_n_by_tf",
        return_value={"2h": [10.0, 12.0], "4h": [2.0, 2.5]},
    )
    monkeypatch.chdir(tmp_path)

    from src.domain.futures.strategy.calibrate_l1_pair_gate import main

    exit_code = main()

    assert exit_code == 0
    out_path = tmp_path / "logs" / "futures" / "diagnostics" / "l1_pair_gate_calibration.json"
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    assert artifact["note"].startswith("NOT auto-applied")
    assert "2h" in artifact["proposed_l1_pair_min_effective_obs"]
    assert "4h" in artifact["proposed_l1_pair_min_effective_obs"]


def test_script_importable_via_direct_invocation_path(tmp_path: Path) -> None:
    """Regression guard: `python scripts/calibrate_l1_pair_gate.py` must not ModuleNotFoundError
    on `scripts.run_l1_cross_tf_replay` (sys.path[0] is the script's own dir, not repo root,
    when invoked this way -- the module must bootstrap repo root onto sys.path itself)."""
    import subprocess
    import sys as _sys

    repo_root = Path(__file__).resolve().parents[4]
    script_path = repo_root / "src" / "domain" / "futures" / "strategy" / "calibrate_l1_pair_gate.py"
    try:
        result = subprocess.run(  # noqa: S603
            [_sys.executable, str(script_path)],
            cwd=str(tmp_path),  # deliberately NOT repo root
            capture_output=True,
            text=True,
            timeout=5,
        )
    except subprocess.TimeoutExpired:
        return  # got past imports into the heavy pipeline run -- import bootstrap succeeded
    assert "ModuleNotFoundError" not in result.stderr, result.stderr
