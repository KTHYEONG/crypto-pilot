#!/usr/bin/env python3
# ruff: noqa: T201
"""Regenerate the live-decision golden reference frames (unit, levered, targets).

Requires ``--write``; without it the script exits without touching disk. Before
overwriting, it prints the max abs/rel diff against each stored frame plus any
sign or zero-pattern mismatch, so a regeneration can be reviewed in the commit.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.golden.test_live_decision_golden import (  # noqa: E402
    _REFERENCE_DIR,
    build_reference_frames,
)


def _describe_diff(name: str, new: pd.DataFrame, old: pd.DataFrame) -> None:
    if not (new.index.equals(old.index) and list(new.columns) == list(old.columns)):
        raise SystemExit(f"{name}: index or columns differ from stored frame; refusing to overwrite")
    a = new.to_numpy(dtype="float64")
    b = old.to_numpy(dtype="float64")
    delta = np.abs(a - b)
    nonzero = np.abs(b) > 0.0
    max_abs = float(delta.max()) if delta.size else 0.0
    max_rel = float((delta[nonzero] / np.abs(b[nonzero])).max()) if nonzero.any() else 0.0
    sign_flips = int((np.sign(a) != np.sign(b)).sum())
    zero_diff = int(((a == 0.0) != (b == 0.0)).sum())
    print(
        f"{name}: max_abs={max_abs:.3e} max_rel={max_rel:.3e} "
        f"sign_flips={sign_flips} zero_pattern_diff={zero_diff}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="overwrite the stored reference frames")
    args = parser.parse_args()
    if not args.write:
        print("refusing to regenerate without --write", file=sys.stderr)
        raise SystemExit(2)

    with tempfile.TemporaryDirectory(prefix="live_golden_") as tmp:
        frames = build_reference_frames(Path(tmp))

    _REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        path = _REFERENCE_DIR / f"{name}.parquet"
        if path.exists():
            _describe_diff(name, frame, pd.read_parquet(path))
        else:
            print(f"{name}: no stored frame, writing new reference")
        frame.to_parquet(path, index=True)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
