"""Same-workload warm benchmark for mhs_arrow_ipc_numpy_restore (SLOW).

Measures the contracted ``ChunkedArray.to_numpy(zero_copy_only=False)``
restore path against the pre-change ``to_pylist()`` path on identical
spilled IPC files built from the existing representative 90d/64-symbol/5m
workload (``_mhs_benchmark_sample.py``).

Gate (contract requirements 38-39): after a warm-up pass, the warm median
wall time of the contracted path must decrease versus the legacy path on the
same workload/hardware with no measured peak-RSS increase; bit-exactness is
verified via uint64 patterns. A measured record (wall/CPU/peak-RSS,
rows/windows, workers/threads, repetitions, median/p10/p90) is written to
``logs/scratch/mhs_arrow_ipc_numpy_restore_benchmark.json``; the committed
reference copy lives at
``tests/benchmark/mhs/mhs_arrow_ipc_numpy_restore_benchmark.json``.
"""

from __future__ import annotations

import json
import multiprocessing
import platform
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as pa_ipc
import psutil

from src.mhs.evaluation.windows import _load_window_from_ipc, _spill_window_to_ipc
from src.mhs.execution import ExecutionSpec
from src.mhs.execution.contracts import ExecutionReplayWindow
from src.mhs.resources import _TreeMemorySampler

SAMPLE_DIR = Path(__file__).parent
ROOT = Path(__file__).resolve().parents[3]
SCRATCH_DIR = ROOT / "logs" / "scratch"

sys.path.insert(0, str(SAMPLE_DIR))

from _mhs_benchmark_sample import _partition_windows, build_workload  # noqa: E402

REPETITIONS = 11


def _load_window_legacy_pylist(target_path: str) -> ExecutionReplayWindow:
    """Pre-change restore path, kept here as the A/B comparator only."""
    with zipfile.ZipFile(target_path, "r") as zf:
        meta = json.loads(zf.read("meta.json"))
        buffers = {
            name: zf.read(f"{name}.arrow")
            for name, spec in meta["frames"].items()
            if spec is not None
        }
    minute_grid = pd.DatetimeIndex(
        pd.to_datetime(np.asarray(meta["minute_grid_ns"], dtype="int64"), unit="ns", utc=True)
    )
    signal_available_at = pd.DatetimeIndex(
        pd.to_datetime(np.asarray(meta["signal_ns"], dtype="int64"), unit="ns", utc=True)
    )
    frames: dict[str, pd.DataFrame | None] = {}
    for name, spec in meta["frames"].items():
        if spec is None:
            frames[name] = None
            continue
        reader = pa_ipc.open_stream(pa.py_buffer(buffers[name]))
        table = reader.read_all()
        idx = pd.DatetimeIndex(
            pd.to_datetime(np.asarray(spec["index_ns"], dtype="int64"), unit="ns", utc=True)
        )
        data = {
            c: np.asarray(table.column(c).to_pylist(), dtype="float64")
            for c in spec["columns"]
        }
        frames[name] = pd.DataFrame(data, index=idx, columns=spec["columns"])
        frames[name] = frames[name].astype("float64")
    return ExecutionReplayWindow(
        window_start=pd.Timestamp(meta["window_start_ns"], unit="ns", tz="UTC"),
        window_end=pd.Timestamp(meta["window_end_ns"], unit="ns", tz="UTC"),
        columns=tuple(meta["columns"]),
        symbols=tuple(meta["symbols"]),
        minute_grid=minute_grid,
        highs=frames["highs"],
        lows=frames["lows"],
        closes=frames["closes"],
        marks=frames["marks"],
        bar_funding=frames["bar_funding"],
        target_weights=frames["target_weights"],
        signal_available_at=signal_available_at,
    )


def _tree_rss_bytes() -> int:
    process = psutil.Process()
    return sum(
        int(proc.memory_info().rss)
        for proc in [process, *process.children(recursive=True)]
        if proc.is_running()
    )


def _time_full_passes(paths: list[str], loader, reps: int) -> tuple[list[float], list[float], int]:
    walls: list[float] = []
    cpus: list[float] = []
    peak_tree_rss = 0
    for _ in range(reps):
        t0 = time.perf_counter()
        c0 = time.process_time()
        for p in paths:
            loader(p)
            peak_tree_rss = max(peak_tree_rss, _tree_rss_bytes())
        walls.append(time.perf_counter() - t0)
        cpus.append(time.process_time() - c0)
    return walls, cpus, peak_tree_rss


def _measure_variant(paths: list[str], variant: str, result_pipe) -> None:
    loader = _load_window_legacy_pylist if variant == "legacy" else _load_window_from_ipc
    sampler = _TreeMemorySampler(interval_seconds=0.01)
    sampler.start()
    walls, cpus, tree_rss = _time_full_passes(paths, loader, REPETITIONS)
    memory = sampler.stop()
    result_pipe.send({
        "wall_seconds": _summarize(walls),
        "cpu_seconds": _summarize(cpus),
        "tree_rss_peak_bytes": tree_rss,
        "tree_pss_peak_bytes": memory.tree_pss_peak_bytes,
        "tree_uss_peak_bytes": memory.tree_uss_peak_bytes,
        "samples_taken": memory.samples_taken,
    })
    result_pipe.close()


def _run_variant_in_fresh_process(paths: list[str], variant: str) -> dict[str, object]:
    context = multiprocessing.get_context("fork")
    parent_pipe, child_pipe = context.Pipe(duplex=False)
    process = context.Process(target=_measure_variant, args=(paths, variant, child_pipe))
    process.start()
    child_pipe.close()
    result = parent_pipe.recv()
    process.join()
    assert process.exitcode == 0
    return result


def _summarize(values: list[float]) -> dict[str, object]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p90": float(np.quantile(arr, 0.90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "values": [float(v) for v in values],
    }


def test_mhs_ipc_numpy_restore_warm_benchmark(tmp_path) -> None:
    workload = build_workload()
    windows = _partition_windows(
        workload["grid"],
        workload["weights"],
        workload["signal_available_at"],
        workload["highs"],
        workload["lows"],
        workload["closes"],
        workload["marks"],
        workload["funding"],
        ExecutionSpec(),
    )
    spill_dir = tmp_path / "spill"
    spill_dir.mkdir()
    paths: list[str] = []
    total_cells = 0
    for i, w in enumerate(windows):
        p = str(spill_dir / f"window_{i:05d}.arrow")
        _spill_window_to_ipc(w, p)
        paths.append(p)
        total_cells += len(w.minute_grid) * max(len(w.symbols), 1)
    assert len(paths) >= 3

    # Warm-up: one full pass per variant, discarded.
    for p in paths:
        _load_window_legacy_pylist(p)
    for p in paths:
        _load_window_from_ipc(p)

    # Exactness gate on the identical workload before timing counts.
    for p in paths:
        old = _load_window_legacy_pylist(p)
        new = _load_window_from_ipc(p)
        assert old.minute_grid.equals(new.minute_grid)
        assert old.signal_available_at.equals(new.signal_available_at)
        for name in ("highs", "lows", "closes", "marks", "bar_funding", "target_weights"):
            a, b = getattr(old, name), getattr(new, name)
            assert (a is None) == (b is None)
            if a is not None:
                assert a.index.equals(b.index)
                assert list(a.columns) == list(b.columns)
                np.testing.assert_array_equal(
                    a.to_numpy(dtype=np.float64).view(np.uint64),
                    b.to_numpy(dtype=np.float64).view(np.uint64),
                )

    legacy = _run_variant_in_fresh_process(paths, "legacy")
    numpy = _run_variant_in_fresh_process(paths, "numpy")
    legacy_wall = legacy["wall_seconds"]
    numpy_wall = numpy["wall_seconds"]

    # ADOPT gate: same-workload warm median wall decreases, no tree PSS/RSS increase.
    assert float(numpy_wall["median"]) < float(legacy_wall["median"])
    assert int(numpy["tree_pss_peak_bytes"]) <= int(legacy["tree_pss_peak_bytes"])
    assert int(numpy["tree_rss_peak_bytes"]) <= int(legacy["tree_rss_peak_bytes"])

    record = {
        "feature": "mhs_arrow_ipc_numpy_restore",
        "workload": {
            "source": "tests/benchmark/mhs/_mhs_benchmark_sample.py (90d/64-symbol/5m)",
            "grid_bars": len(workload["grid"]),
            "n_symbols": 64,
            "decisions": len(workload["weights"]),
            "n_windows": len(paths),
            "total_minute_symbol_cells": int(total_cells),
            "frames_per_window": ["highs", "lows", "closes", "marks", "bar_funding", "target_weights"],
        },
        "config": {
            "workers": 1,
            "threads": "pyarrow-single-process",
            "repetitions": REPETITIONS,
            "warmup": "1 full pass per variant, discarded",
            "hardware": platform.platform(),
            "python": platform.python_version(),
        },
        "legacy_to_pylist": {
            "wall_seconds": legacy_wall,
            "cpu_seconds": legacy["cpu_seconds"],
        },
        "numpy_restore": {
            "wall_seconds": numpy_wall,
            "cpu_seconds": numpy["cpu_seconds"],
        },
        "memory": {
            "legacy_tree_pss_peak_bytes": legacy["tree_pss_peak_bytes"],
            "numpy_tree_pss_peak_bytes": numpy["tree_pss_peak_bytes"],
            "legacy_tree_uss_peak_bytes": legacy["tree_uss_peak_bytes"],
            "numpy_tree_uss_peak_bytes": numpy["tree_uss_peak_bytes"],
            "legacy_tree_rss_peak_bytes": legacy["tree_rss_peak_bytes"],
            "numpy_tree_rss_peak_bytes": numpy["tree_rss_peak_bytes"],
            "legacy_samples_taken": legacy["samples_taken"],
            "numpy_samples_taken": numpy["samples_taken"],
        },
        "exactness": "PASS (uint64 bit patterns equal on all frames/windows)",
        "wall_median_delta_seconds": float(numpy_wall["median"]) - float(legacy_wall["median"]),
        "decision": "ADOPT",
    }
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    (SCRATCH_DIR / "mhs_arrow_ipc_numpy_restore_benchmark.json").write_text(json.dumps(record, indent=2))
