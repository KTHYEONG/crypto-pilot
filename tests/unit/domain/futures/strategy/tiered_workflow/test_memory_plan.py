from __future__ import annotations

from unittest.mock import patch

import psutil

from src.domain.futures.strategy.tiered_workflow.memory import (
    GIB,
    MIB,
    ProcessTreeMemory,
    estimate_unique_array_bytes,
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
