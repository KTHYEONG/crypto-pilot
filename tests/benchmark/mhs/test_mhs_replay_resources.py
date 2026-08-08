"""Slow resource benchmark for the MHS replay (MHS-32).

Runs the fixed 90-day / 64-symbol / 5-minute workload three times in fresh
processes and publishes workload shape, the deterministic result checksum,
median wall seconds, and peak RSS to ``logs/scratch/`` for before/after
comparisons.  No absolute time/RSS gate is enforced: the same host and the same
workload must show a strictly lower optimized median.
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
from _mhs_benchmark_sample import run_sample
sys.stdout.write(json.dumps(run_sample()) + "\\n")
"""


@pytest.mark.slow
def test_mhs_replay_resource_benchmark() -> None:
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
        assert sample["fill_count"] == samples[0]["fill_count"]
    assert samples[0]["checksum"]
    assert samples[0]["grid_bars"] == 90 * 24 * 12
    assert samples[0]["n_symbols"] == 64
    assert samples[0]["decisions"] == 360
    assert samples[0]["fill_count"] > 0

    summary = {
        "workload": {
            "grid_bars": samples[0]["grid_bars"],
            "n_symbols": samples[0]["n_symbols"],
            "decisions": samples[0]["decisions"],
            "fill_count": samples[0]["fill_count"],
        },
        "checksum": samples[0]["checksum"],
        "n_samples": N_SAMPLES,
        "median_elapsed_seconds": statistics.median(
            float(s["elapsed_seconds"]) for s in samples
        ),
        "median_peak_rss_kb": statistics.median(
            int(s["peak_rss_kb"]) for s in samples
        ),
        "samples": samples,
    }
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SCRATCH_DIR / "mhs_replay_resources.json"
    out_path.write_text(json.dumps(summary, indent=2))
    assert summary["median_elapsed_seconds"] > 0.0
    assert summary["median_peak_rss_kb"] > 0
