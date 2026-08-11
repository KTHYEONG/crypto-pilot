"""Slow benchmark for the MHS discovery gate (spec §6.3).

Runs the densified 19-candidate momentum grid against the fixed 4-year /
64-symbol / 1-hour synthetic panel three times in fresh subprocesses and
publishes workload shape, the deterministic result checksum, and median wall
seconds to ``logs/scratch/`` for before/after comparisons.  No absolute time
gate is enforced (performance.md variance-tolerance guidance): the file exists
for manual comparison only.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SAMPLE_DIR = Path(__file__).parent
SCRATCH_DIR = ROOT / "logs" / "scratch"
N_SAMPLES = 3

SAMPLE_SCRIPT = """
import json, sys
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[2])
from _mhs_discovery_gate_sample import run_sample
sys.stdout.write(json.dumps(run_sample()) + "\\n")
"""


@pytest.mark.slow
def test_mhs_discovery_gate_benchmark() -> None:
    """SCENARIO_MHS_DISCOVERY_GATE_BENCHMARK_10: deterministic-discovery-gate
    wall-clock baseline; see module docstring."""
    samples: list[dict[str, object]] = []
    for _ in range(N_SAMPLES):
        proc = subprocess.run(  # noqa: S603 - fully static interpreter + fixed sample script
            [sys.executable, "-c", SAMPLE_SCRIPT, str(ROOT), str(SAMPLE_DIR)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        samples.append(json.loads(proc.stdout.strip().splitlines()[-1]))

    for sample in samples[1:]:
        assert sample["checksum"] == samples[0]["checksum"]
        assert sample["selected_horizon"] == samples[0]["selected_horizon"]
        assert sample["admitted"] == samples[0]["admitted"]
        assert sample["discovery_scores"] == samples[0]["discovery_scores"]
    assert samples[0]["checksum"]
    assert samples[0]["grid_bars"] == 4 * 365 * 24
    assert samples[0]["n_symbols"] == 64
    assert samples[0]["n_candidates"] == 19
    assert samples[0]["elapsed_seconds"] > 0.0

    summary = {
        "workload": {
            "grid_bars": samples[0]["grid_bars"],
            "n_symbols": samples[0]["n_symbols"],
            "n_candidates": samples[0]["n_candidates"],
        },
        "checksum": samples[0]["checksum"],
        "n_samples": N_SAMPLES,
        "median_elapsed_seconds": statistics.median(
            float(s["elapsed_seconds"]) for s in samples
        ),
        "samples": samples,
    }
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SCRATCH_DIR / "mhs_discovery_gate_benchmark.json"
    out_path.write_text(json.dumps(summary, indent=2))
    assert summary["median_elapsed_seconds"] > 0.0
