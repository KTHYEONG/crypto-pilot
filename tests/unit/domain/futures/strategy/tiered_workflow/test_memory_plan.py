from __future__ import annotations

from unittest.mock import patch

import pytest

import psutil

from src.domain.futures.strategy.tiered_workflow.memory import (
    GIB,
    MIB,
    ProcessTreeMemory,
    estimate_unique_array_bytes,
    fit_worker_private_linear_model,
    get_worker_private_observations,
    measure_worker_private_bytes,
    predict_calibrated_worker_private_mb,
    record_worker_private_observation,
    reset_worker_private_calibration,
    resolve_l1_memory_plan,
    snapshot_process_tree_memory,
)


def test_memory_plan_fails_closed_near_tree_cap() -> None:
    snapshot = ProcessTreeMemory(
        parent_rss_bytes=7 * GIB,
        tree_pss_bytes=9 * GIB,
        tree_uss_bytes=8 * GIB,
        available_bytes=6 * GIB,
    )

    plan = resolve_l1_memory_plan(
        n_tasks=12,
        shared_input_bytes=2 * GIB,
        result_soft_cap_bytes=512 * MIB,
        snapshot=snapshot,
        stage_cap=3,
        cpu_cap=6,
        pinned=4,
    )

    assert plan.workers == 1
    assert plan.reason == "memory_floor_serial"
    assert plan.projected_tree_bytes <= 10 * GIB + plan.estimated_worker_private_bytes


def test_memory_plan_returns_one_when_pss_unavailable() -> None:
    snapshot = ProcessTreeMemory(
        parent_rss_bytes=7 * GIB,
        tree_pss_bytes=None,
        tree_uss_bytes=None,
        available_bytes=6 * GIB,
    )

    plan = resolve_l1_memory_plan(
        n_tasks=12,
        shared_input_bytes=2 * GIB,
        result_soft_cap_bytes=512 * MIB,
        snapshot=snapshot,
        stage_cap=3,
        cpu_cap=6,
        pinned=4,
    )

    assert plan.workers == 1
    assert plan.reason == "memory_metrics_unavailable"


def test_memory_plan_pinned_overrides_workers() -> None:
    snapshot = ProcessTreeMemory(
        parent_rss_bytes=1 * GIB,
        tree_pss_bytes=2 * GIB,
        tree_uss_bytes=1 * GIB,
        available_bytes=32 * GIB,
    )

    plan = resolve_l1_memory_plan(
        n_tasks=12,
        shared_input_bytes=512 * MIB,
        result_soft_cap_bytes=256 * MIB,
        snapshot=snapshot,
        stage_cap=3,
        cpu_cap=6,
        pinned=2,
    )

    assert plan.workers == 2
    assert plan.reason == "ok"


def test_snapshot_process_tree_memory_integration() -> None:
    import os

    mem = snapshot_process_tree_memory(os.getpid())
    assert isinstance(mem, ProcessTreeMemory)


def test_estimate_unique_array_bytes_numpy() -> None:
    import numpy as np

    arr = np.zeros((100, 10), dtype=np.float64)
    expected = 100 * 10 * 8
    assert estimate_unique_array_bytes(arr) == expected


def test_estimate_unique_array_bytes_dataframe() -> None:
    import numpy as np
    import pandas as pd

    df = pd.DataFrame({"a": np.zeros(50), "b": np.ones(50)})
    total = estimate_unique_array_bytes(df)
    assert total > 0


def test_estimate_unique_array_bytes_series() -> None:
    import numpy as np
    import pandas as pd

    s = pd.Series(np.zeros(50))
    total = estimate_unique_array_bytes(s)
    assert total > 0


def test_estimate_unique_array_bytes_dataframe_shared_base() -> None:
    import numpy as np
    import pandas as pd

    base = np.zeros((50, 2))
    df = pd.DataFrame({"a": base[:, 0], "b": base[:, 1]})
    total = estimate_unique_array_bytes(df)
    assert total > 0


def test_estimate_unique_array_bytes_dataframe_no_base() -> None:
    import numpy as np
    import pandas as pd

    arr_a = np.zeros(50)
    arr_b = np.ones(50)
    df = pd.DataFrame({"a": arr_a, "b": arr_b})
    total = estimate_unique_array_bytes(df)
    assert total > 0


def test_estimate_unique_array_bytes_unsupported_type() -> None:
    assert estimate_unique_array_bytes("not an array") == 0


def test_snapshot_process_tree_memory_handles_child_error() -> None:
    import os

    mem = snapshot_process_tree_memory(os.getpid())
    assert isinstance(mem, ProcessTreeMemory)
    assert mem.parent_rss_bytes > 0


def test_snapshot_process_tree_memory_handles_root_failure() -> None:
    with patch("psutil.Process", side_effect=psutil.NoSuchProcess(pid=99999)):
        mem = snapshot_process_tree_memory(99999)
        assert isinstance(mem, ProcessTreeMemory)
        assert mem.parent_rss_bytes == 0
        assert mem.tree_pss_bytes is None


def test_snapshot_process_tree_memory_handles_root_failure_no_memory() -> None:
    with (
        patch("psutil.Process", side_effect=psutil.NoSuchProcess(pid=99999)),
        patch("psutil.virtual_memory", side_effect=RuntimeError("mock")),
    ):
        mem = snapshot_process_tree_memory(99999)
        assert mem.available_bytes == 0


def test_snapshot_process_tree_memory_with_children() -> None:
    import os

    with patch("psutil.Process") as mock_proc_cls:
        mock_proc = mock_proc_cls.return_value
        mock_proc.memory_info.return_value.rss = 1000
        mock_proc.children.return_value = [mock_proc]

        class FakeFullInfo:
            pss = 500
            uss = 400

        mock_proc.memory_full_info.return_value = FakeFullInfo()

        mem = snapshot_process_tree_memory(os.getpid())
        assert isinstance(mem, ProcessTreeMemory)
        assert mem.parent_rss_bytes == 1000
        assert mem.tree_pss_bytes == 1000
        assert mem.tree_uss_bytes == 800


def test_snapshot_process_tree_memory_handles_child_failure() -> None:
    import os

    with patch("psutil.Process") as mock_proc_cls:
        mock_proc = mock_proc_cls.return_value
        mock_proc.memory_info.return_value.rss = 1000

        class FakeFullInfo:
            pss = 500
            uss = 400

        good_child = type("GoodChild", (), {"memory_full_info": lambda self: FakeFullInfo()})()

        def _raise_no_such_process(_self: object) -> None:
            raise psutil.NoSuchProcess(pid=999)

        bad_child = type(
            "BadChild",
            (),
            {"memory_full_info": _raise_no_such_process},
        )()

        mock_proc.children.return_value = [good_child, bad_child]

        mem = snapshot_process_tree_memory(os.getpid())
        assert isinstance(mem, ProcessTreeMemory)
        assert mem.tree_pss_bytes is not None
        assert mem.tree_pss_bytes >= 500


def test_estimate_unique_array_bytes_pandas_series_fallback() -> None:
    import numpy as np
    import pandas as pd

    arr = np.array([1.0, 2.0, 3.0])
    s = pd.Series(arr, name="x")
    mem = estimate_unique_array_bytes(s)
    assert mem > 0


# ── binding_constraint diagnosis tests (Phase 1) ────────────────────────


def test_binding_constraint_memory_workers_happy_path() -> None:
    snapshot = ProcessTreeMemory(
        parent_rss_bytes=0,
        tree_pss_bytes=int(7 * GIB),
        tree_uss_bytes=None,
        available_bytes=int(20 * GIB),
    )
    plan = resolve_l1_memory_plan(
        n_tasks=12, shared_input_bytes=60 * 1024 * 1024,
        result_soft_cap_bytes=100 * 1024 * 1024, snapshot=snapshot,
        stage_cap=3, cpu_cap=6,
    )
    assert plan.workers == 2
    assert plan.reason == "ok"
    assert plan.binding_constraint == "memory_workers"


def test_binding_constraint_tree_pss_cap() -> None:
    snapshot = ProcessTreeMemory(
        parent_rss_bytes=0,
        tree_pss_bytes=int(9.5 * GIB),
        tree_uss_bytes=None,
        available_bytes=int(20 * GIB),
    )
    plan = resolve_l1_memory_plan(
        n_tasks=12, shared_input_bytes=60 * 1024 * 1024,
        result_soft_cap_bytes=100 * 1024 * 1024, snapshot=snapshot,
        stage_cap=3, cpu_cap=6,
    )
    assert plan.workers == 1
    assert plan.reason == "memory_floor_serial"
    assert plan.binding_constraint == "tree_pss_cap"


def test_binding_constraint_system_available() -> None:
    snapshot = ProcessTreeMemory(
        parent_rss_bytes=0,
        tree_pss_bytes=int(2 * GIB),
        tree_uss_bytes=None,
        available_bytes=int(0.5 * GIB),
    )
    plan = resolve_l1_memory_plan(
        n_tasks=12, shared_input_bytes=60 * 1024 * 1024,
        result_soft_cap_bytes=100 * 1024 * 1024, snapshot=snapshot,
        stage_cap=3, cpu_cap=6,
    )
    assert plan.workers == 1
    assert plan.binding_constraint == "system_available"


def test_binding_constraint_metrics_unavailable() -> None:
    snapshot = ProcessTreeMemory(
        parent_rss_bytes=0,
        tree_pss_bytes=None,
        tree_uss_bytes=None,
        available_bytes=6 * GIB,
    )
    plan = resolve_l1_memory_plan(
        n_tasks=12, shared_input_bytes=2 * GIB,
        result_soft_cap_bytes=512 * MIB, snapshot=snapshot,
        stage_cap=3, cpu_cap=6,
    )
    assert plan.workers == 1
    assert plan.reason == "memory_metrics_unavailable"
    assert plan.binding_constraint == "metrics_unavailable"


def test_binding_constraint_stage_cap() -> None:
    snapshot = ProcessTreeMemory(
        parent_rss_bytes=0,
        tree_pss_bytes=int(2 * GIB),
        tree_uss_bytes=None,
        available_bytes=int(20 * GIB),
    )
    plan = resolve_l1_memory_plan(
        n_tasks=12, shared_input_bytes=60 * 1024 * 1024,
        result_soft_cap_bytes=100 * 1024 * 1024, snapshot=snapshot,
        stage_cap=2, cpu_cap=6,
    )
    assert plan.workers > 1
    assert plan.binding_constraint == "stage_cap"


def test_worker_plan_log_includes_binding_token(
    caplog: pytest.LogCaptureFixture,
    mocker: pytest.MockFixture,
) -> None:
    import logging

    from src.domain.futures.strategy.tiered_workflow.memory import (
        GIB as _GIB,
        ProcessTreeMemory,
    )
    from src.domain.futures.strategy.tiered_workflow.pipeline import (
        resolve_safe_nested_workers,
    )

    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.memory.snapshot_process_tree_memory",
        return_value=ProcessTreeMemory(
            parent_rss_bytes=0,
            tree_pss_bytes=int(2 * _GIB),
            tree_uss_bytes=None,
            available_bytes=int(20 * _GIB),
        ),
    )
    caplog.set_level(logging.DEBUG)
    resolve_safe_nested_workers(12, 60 * 1024 * 1024, stage="evidence")

    assert any("binding=" in r.message for r in caplog.records)


def test_measure_worker_private_bytes_computes_average_growth() -> None:
    before = ProcessTreeMemory(
        parent_rss_bytes=0, tree_pss_bytes=int(7 * GIB), tree_uss_bytes=None, available_bytes=int(10 * GIB)
    )
    after = ProcessTreeMemory(
        parent_rss_bytes=0, tree_pss_bytes=int(7.6 * GIB), tree_uss_bytes=None, available_bytes=int(10 * GIB)
    )

    measured = measure_worker_private_bytes(before, after, workers=3)

    assert measured == int(0.6 * GIB) // 3


def test_measure_worker_private_bytes_clamps_negative_delta_to_zero() -> None:
    before = ProcessTreeMemory(
        parent_rss_bytes=0, tree_pss_bytes=int(7 * GIB), tree_uss_bytes=None, available_bytes=int(10 * GIB)
    )
    after = ProcessTreeMemory(
        parent_rss_bytes=0, tree_pss_bytes=int(6.5 * GIB), tree_uss_bytes=None, available_bytes=int(10 * GIB)
    )

    measured = measure_worker_private_bytes(before, after, workers=3)

    assert measured == 0


def test_measure_worker_private_bytes_falls_back_to_uss_when_pss_unavailable() -> None:
    before = ProcessTreeMemory(
        parent_rss_bytes=0, tree_pss_bytes=None, tree_uss_bytes=int(4 * GIB), available_bytes=int(10 * GIB)
    )
    after = ProcessTreeMemory(
        parent_rss_bytes=0, tree_pss_bytes=None, tree_uss_bytes=int(4.5 * GIB), available_bytes=int(10 * GIB)
    )

    measured = measure_worker_private_bytes(before, after, workers=1)

    assert measured == int(0.5 * GIB)


def test_measure_worker_private_bytes_returns_none_when_metrics_unavailable() -> None:
    before = ProcessTreeMemory(
        parent_rss_bytes=0, tree_pss_bytes=None, tree_uss_bytes=None, available_bytes=int(10 * GIB)
    )
    after = ProcessTreeMemory(
        parent_rss_bytes=0, tree_pss_bytes=int(4 * GIB), tree_uss_bytes=None, available_bytes=int(10 * GIB)
    )

    assert measure_worker_private_bytes(before, after, workers=1) is None


def test_measure_worker_private_bytes_returns_none_when_workers_zero() -> None:
    before = ProcessTreeMemory(
        parent_rss_bytes=0, tree_pss_bytes=int(4 * GIB), tree_uss_bytes=None, available_bytes=int(10 * GIB)
    )
    after = ProcessTreeMemory(
        parent_rss_bytes=0, tree_pss_bytes=int(4.5 * GIB), tree_uss_bytes=None, available_bytes=int(10 * GIB)
    )

    assert measure_worker_private_bytes(before, after, workers=0) is None


def test_resolve_l1_memory_plan_lower_floor_unlocks_more_workers() -> None:
    """Lowering worker_private_floor_bytes below the 1GiB default (calibrated against
    real [SYS] stage=worker_private_measured observations) should raise achievable
    workers for the same headroom, without changing default (1GiB) behavior."""
    snapshot = ProcessTreeMemory(
        parent_rss_bytes=0,
        tree_pss_bytes=int(7.1 * GIB),
        tree_uss_bytes=None,
        available_bytes=int(20 * GIB),
    )

    default_plan = resolve_l1_memory_plan(
        n_tasks=12, shared_input_bytes=60 * 1024 * 1024,
        result_soft_cap_bytes=100 * 1024 * 1024, snapshot=snapshot,
        stage_cap=3, cpu_cap=6,
    )
    calibrated_plan = resolve_l1_memory_plan(
        n_tasks=12, shared_input_bytes=60 * 1024 * 1024,
        result_soft_cap_bytes=100 * 1024 * 1024, snapshot=snapshot,
        stage_cap=3, cpu_cap=6,
        worker_private_floor_bytes=200 * MIB,
    )

    assert default_plan.workers == 1
    assert calibrated_plan.workers == 3
    assert calibrated_plan.binding_constraint == "stage_cap"


# ── Calibration store & linear-model tests (adaptive worker sizing) ──────

_TODAY_EVIDENCE_OBSERVATIONS: list[tuple[float, float]] = [
    (61, 731), (74, 880), (95, 920), (54, 690), (48, 639), (46, 608), (14, 439),
]


def test_fit_worker_private_linear_model_returns_none_when_less_than_2_obs() -> None:
    assert fit_worker_private_linear_model([]) is None
    assert fit_worker_private_linear_model([(10.0, 100.0)]) is None


def test_fit_worker_private_linear_model_returns_none_when_zero_variance() -> None:
    assert fit_worker_private_linear_model([(10.0, 100.0), (10.0, 200.0)]) is None


def test_fit_worker_private_linear_model_fits_ols() -> None:
    result = fit_worker_private_linear_model(_TODAY_EVIDENCE_OBSERVATIONS)
    assert result is not None
    intercept, slope = result
    # OLS from spec: measured_mb ~= 342 + 6.41 * shared_mb
    assert intercept == pytest.approx(342.0, rel=0.15)
    assert slope == pytest.approx(6.41, rel=0.3)


def test_predict_calibrated_worker_private_mb_cold_start_returns_default() -> None:
    reset_worker_private_calibration()

    predicted = predict_calibrated_worker_private_mb(
        observations=get_worker_private_observations("evidence"),
        shared_mb=60.0, default_mb=1024.0,
    )

    assert predicted == 1024.0


def test_predict_calibrated_worker_private_mb_fits_real_observed_data() -> None:
    reset_worker_private_calibration()
    for shared_mb, measured_mb in _TODAY_EVIDENCE_OBSERVATIONS:
        record_worker_private_observation("evidence", shared_mb, measured_mb)

    predicted = predict_calibrated_worker_private_mb(
        observations=get_worker_private_observations("evidence"),
        shared_mb=80.0, default_mb=1024.0, margin=1.3,
    )

    assert predicted == pytest.approx(342 + 6.41 * 80.0, rel=0.15) or predicted > 800


def test_predict_calibrated_worker_private_mb_clamps_below_observed_max() -> None:
    reset_worker_private_calibration()
    record_worker_private_observation("outer", 10.0, 500.0)
    record_worker_private_observation("outer", 20.0, 600.0)

    predicted = predict_calibrated_worker_private_mb(
        observations=get_worker_private_observations("outer"),
        shared_mb=5.0, default_mb=1024.0, margin=1.3,
    )

    assert predicted >= 600.0 * 1.3


def test_get_worker_private_observations_returns_empty_for_nonexistent_stage() -> None:
    reset_worker_private_calibration()
    assert get_worker_private_observations("nonexistent_stage") == []


def test_reset_worker_private_calibration_clears_all_stages() -> None:
    record_worker_private_observation("evidence", 10.0, 100.0)
    record_worker_private_observation("outer", 20.0, 200.0)
    assert len(get_worker_private_observations("evidence")) == 1
    assert len(get_worker_private_observations("outer")) == 1

    reset_worker_private_calibration()
    assert get_worker_private_observations("evidence") == []
    assert get_worker_private_observations("outer") == []


def test_predict_calibrated_worker_private_mb_single_observation_returns_default() -> None:
    reset_worker_private_calibration()
    record_worker_private_observation("evidence", 10.0, 100.0)

    predicted = predict_calibrated_worker_private_mb(
        observations=get_worker_private_observations("evidence"),
        shared_mb=60.0, default_mb=1024.0,
    )

    assert predicted == 1024.0


def test_predict_calibrated_worker_private_mb_zero_variance_fallback() -> None:
    """predict falls back to default_mb when linear model is None (zero shared_mb variance)."""
    reset_worker_private_calibration()
    record_worker_private_observation("evidence", 50.0, 400.0)
    record_worker_private_observation("evidence", 50.0, 500.0)

    predicted = predict_calibrated_worker_private_mb(
        observations=get_worker_private_observations("evidence"),
        shared_mb=60.0, default_mb=1024.0,
    )

    assert predicted == 1024.0


def test_l1_tfs_sorted_by_hours_per_bar() -> None:
    """Scenario 5 [LIMIT-07]: l1_tfs sorting by ascending hours_per_bar."""
    from src.domain.futures.strategy.timeframe_contracts import hours_per_bar

    l1_tfs = ("4h", "1h", "12h", "2h")
    sorted_tfs = tuple(sorted(l1_tfs, key=hours_per_bar))
    assert sorted_tfs == ("1h", "2h", "4h", "12h")


def test_calibrated_worker_private_applied_after_two_observations(
    caplog: pytest.LogCaptureFixture,
    mocker: pytest.MockFixture,
) -> None:
    """Scenario 4: resolve_safe_nested_workers 2회 호출 — 관측치 주입 후
    두 번째 호출에서 worker_mb가 기본값(1024)이 아닌 캘리브레이션 값으로 변경됨."""
    import logging

    from src.domain.futures.strategy.tiered_workflow.memory import (
        GIB as _GIB,
        MIB as _MIB,
        ProcessTreeMemory,
    )
    from src.domain.futures.strategy.tiered_workflow.pipeline import (
        resolve_safe_nested_workers,
    )

    mocker.patch(
        "src.domain.futures.strategy.tiered_workflow.memory.snapshot_process_tree_memory",
        return_value=ProcessTreeMemory(
            parent_rss_bytes=0,
            tree_pss_bytes=int(2 * _GIB),
            tree_uss_bytes=None,
            available_bytes=int(20 * _GIB),
        ),
    )

    reset_worker_private_calibration()
    caplog.set_level(logging.DEBUG)

    # First call: cold start -> workers based on default 1024MB floor
    resolve_safe_nested_workers(12, 60 * _MIB, stage="evidence")

    # Inject a lower-than-default observation (simulates measured ~400MB/worker)
    record_worker_private_observation("evidence", 60.0, 400.0)
    record_worker_private_observation("evidence", 80.0, 500.0)

    # Second call: should now use calibration (smaller floor -> same/more workers)
    resolve_safe_nested_workers(12, 70 * _MIB, stage="evidence")

    assert any("binding=" in r.message for r in caplog.records)
