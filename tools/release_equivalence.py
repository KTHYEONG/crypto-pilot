#!/usr/bin/env python3
# ruff: noqa: T201, S607
"""Prove that a candidate running-code digest decides the same positions as the accepted one.

Workstation only. Rebuilds the flow_mom_top20 unit book on a baseline commit (its own
``uv.lock``) and on the candidate tree, then requires identical decision frames within
``RELEASE_EQUIVALENCE_ATOL``, a payload that differs only under the implementation keys,
and a baseline digest equal to the accepted release. The pre-clip ensemble must match on
every row; a post-clip row may differ only when the baseline pre-clip row held a
float-cancellation noise cell AND the candidate clip replayed on that baseline row
reproduces the candidate decision (the difference is then the noise fix itself). No return, PnL or statistic is ever
computed; only counts and the maximum absolute weight difference are printed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Float summation-order noise from a pandas bump is ~1e-16 per cell; 1e-12 absorbs it while
# any real decision change (a weight moved by a seat or a sign) is orders of magnitude larger.
RELEASE_EQUIVALENCE_ATOL: Final[float] = 1e-12
_STRATEGY_ID: Final[str] = "flow_mom_top20"
_IMPLEMENTATION_KEYS: Final[frozenset[str]] = frozenset({"implementation", "shared_parameters"})
_IMPLEMENTATION_FILES: Final[tuple[str, ...]] = ("features.py", "targets.py", "books.py", "universe.py", "sizing.py")
_EVIDENCE_DIR: Final[Path] = Path("data/research/release_amendments")

_DUMPER = r'''
import json, sys
tree, data_root, start, end, out = sys.argv[1:6]
sys.path.insert(0, tree)
from pathlib import Path
import pandas as pd
import src.strategy.release as release_mod
import src.live.strategy_book as strategy_book_mod
from src.live.strategy_book import build_live_strategy_book, crypto_census
from src.strategy.release import load_release, strategy_spec_digest
from src.strategy.targets import FLOW_MOM_TOP20
root = Path(data_root)
census = crypto_census([p.stem for p in (root / "ohlcv" / "1h").glob("*.parquet")], frozenset())
pre_clip = []
_clip = strategy_book_mod.clip_names_preserving_gross
def _capture(weights, clip):
    pre_clip.append(weights.copy())
    return _clip(weights, clip)
strategy_book_mod.clip_names_preserving_gross = _capture
book = build_live_strategy_book(root, census, panel_start=pd.Timestamp(start), panel_end=pd.Timestamp(end))
if len(pre_clip) != 1:
    raise SystemExit(f"expected one live clip call, saw {len(pre_clip)}")
unit = book.unit_weights
unit.to_parquet(out)
pre = pre_clip[0].reindex(unit.index + pd.Timedelta(days=1))
pd.DataFrame(pre.to_numpy(dtype="float64"), index=unit.index, columns=pre.columns).to_parquet(out + ".preclip.parquet")
release = load_release(FLOW_MOM_TOP20.strategy_id)
sizing = dict(release.sizing)
payload = release_mod.strategy_spec_payload(FLOW_MOM_TOP20, sizing) if hasattr(release_mod, "strategy_spec_payload") else None
print(json.dumps({"digest": strategy_spec_digest(FLOW_MOM_TOP20, sizing), "payload": payload, "pandas": pd.__version__}, sort_keys=True, default=str))
'''


class EquivalenceError(ValueError):
    """Raised when any equivalence clause fails; the amendment is then never recorded."""


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _pattern_counts(
    baseline: pd.DataFrame, candidate: pd.DataFrame, atol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Per-cell sign/zero/size mismatches; cells with ``|w| <= atol`` count as zero on both sides."""
    if not baseline.index.equals(candidate.index) or list(baseline.columns) != list(candidate.columns):
        raise EquivalenceError("decision index or columns differ")
    a = baseline.to_numpy(dtype="float64")
    b = candidate.to_numpy(dtype="float64")
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        raise EquivalenceError("NaN pattern differs")
    observed = ~np.isnan(a)
    # Sub-atol values are summation noise around an exact zero; their sign carries no decision.
    fa = np.where(np.abs(a) <= atol, 0.0, a)
    fb = np.where(np.abs(b) <= atol, 0.0, b)
    sign_bad = (np.sign(fa) != np.sign(fb)) & observed
    zero_bad = ((fa == 0.0) != (fb == 0.0)) & observed
    deltas = np.where(observed, np.abs(a - b), 0.0)
    max_abs_diff = float(deltas.max()) if deltas.size else 0.0
    return sign_bad, zero_bad, deltas > atol, max_abs_diff


def divergent_rows(
    baseline: pd.DataFrame, candidate: pd.DataFrame, *, atol: float = RELEASE_EQUIVALENCE_ATOL,
) -> tuple[np.ndarray, float]:
    """Row positions whose sign/zero pattern or values differ beyond ``atol``, plus the max diff."""
    sign_bad, zero_bad, size_bad, max_abs_diff = _pattern_counts(baseline, candidate, atol)
    return np.flatnonzero((sign_bad | zero_bad | size_bad).any(axis=1)), max_abs_diff


def compare_unit_weights(
    baseline: pd.DataFrame, candidate: pd.DataFrame, *, atol: float = RELEASE_EQUIVALENCE_ATOL,
) -> dict[str, Any]:
    """Require identical shape, sign and zero pattern, and ``max |diff| <= atol``; return counts only."""
    sign_bad, zero_bad, _, max_abs_diff = _pattern_counts(baseline, candidate, atol)
    sign_diffs = int(np.count_nonzero(sign_bad))
    zero_diffs = int(np.count_nonzero(zero_bad))
    if sign_diffs or zero_diffs:
        raise EquivalenceError(f"sign or zero pattern differs: sign={sign_diffs} zero={zero_diffs}")
    if max_abs_diff > atol:
        raise EquivalenceError(f"max abs weight difference {max_abs_diff:.3e} exceeds atol {atol:.0e}")
    return {
        "rows": int(baseline.shape[0]), "cols": int(baseline.shape[1]), "max_abs_diff": max_abs_diff,
        "sign_diffs": sign_diffs, "zero_diffs": zero_diffs,
    }


def attribute_rows_to_noise(
    baseline_pre_clip: pd.DataFrame,
    candidate_unit: pd.DataFrame,
    rows: np.ndarray,
    *,
    clip_fn: Callable[[pd.DataFrame, float], pd.DataFrame],
    clip: float,
    noise_rtol: float,
    atol: float = RELEASE_EQUIVALENCE_ATOL,
) -> list[str]:
    """Prove each divergent decision row is the noise fix; return their labels or raise.

    A row qualifies only when the baseline pre-clip row holds a nonzero cell within
    ``noise_rtol`` of its gross (summation noise the old clip counted as a held name) and the
    candidate clip applied to that same baseline row reproduces the candidate decision.
    """
    labels: list[str] = []
    for r in rows:
        pre = baseline_pre_clip.iloc[[int(r)]]
        values = pre.to_numpy(dtype="float64")
        values = np.where(np.isfinite(values), values, 0.0)
        gross = float(np.abs(values).sum())
        noise = (values != 0.0) & (np.abs(values) <= noise_rtol * gross)
        label = str(baseline_pre_clip.index[int(r)])
        if not bool(noise.any()):
            raise EquivalenceError(f"decision {label} diverges without a baseline noise cell")
        replay = clip_fn(pre, clip)
        replay = pd.DataFrame(replay.to_numpy(dtype="float64"), index=pre.index, columns=pre.columns)
        if divergent_rows(replay, candidate_unit.iloc[[int(r)]], atol=atol)[0].size:
            raise EquivalenceError(f"decision {label}: candidate clip on the baseline row does not reproduce it")
        labels.append(label)
    return labels


def check_payload_scope(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> frozenset[str]:
    """Return the differing top-level keys; any key outside the implementation set is an error."""
    differing = frozenset(
        key for key in set(candidate) | set(baseline) if candidate.get(key) != baseline.get(key)
    )
    leaked = differing - _IMPLEMENTATION_KEYS
    if leaked:
        raise EquivalenceError(f"payload differs outside implementation keys: {sorted(leaked)}")
    return differing


def baseline_payload_from_candidate(
    candidate: Mapping[str, Any], baseline_src: Path,
) -> dict[str, Any]:
    """Candidate payload with the implementation hashes replaced by the baseline tree's file bytes."""
    payload = dict(candidate)
    payload["implementation"] = {
        name: hashlib.sha256((baseline_src / "strategy" / name).read_bytes()).hexdigest()
        for name in _IMPLEMENTATION_FILES
    }
    payload["shared_parameters"] = hashlib.sha256((baseline_src / "core" / "params.py").read_bytes()).hexdigest()
    return payload


def default_window(data_root: Path, warmup_days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Earliest common 1h bar plus warmup through the latest bar; timestamps only, no values read."""
    firsts: list[int] = []
    lasts: list[int] = []
    for path in sorted((Path(data_root) / "ohlcv" / "1h").glob("*.parquet")):
        stamps = pd.to_numeric(pd.read_parquet(path, columns=["timestamp"])["timestamp"], errors="coerce").dropna()
        if len(stamps):
            firsts.append(int(stamps.min()))
            lasts.append(int(stamps.max()))
    if not firsts:
        raise EquivalenceError("no 1h bars under data root")
    # Panels may start at the earliest bar: the builder itself withholds rows for the warmup.
    start = pd.Timestamp(min(firsts), unit="ms", tz="UTC").normalize() + pd.Timedelta(days=warmup_days)
    end = pd.Timestamp(max(lasts), unit="ms", tz="UTC")
    return start, end


def decision_chunks(
    start: pd.Timestamp, end: pd.Timestamp, chunk_days: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Contiguous ``[chunk_start, chunk_end)`` decision windows covering ``[start, end)``."""
    if chunk_days < 1:
        raise EquivalenceError("chunk_days must be positive")
    out: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    while cursor < end:
        stop = min(cursor + pd.Timedelta(days=chunk_days), end)
        out.append((cursor, stop))
        cursor = stop
    return out


def _run_dumper(tree: Path, data_root: Path, start: pd.Timestamp, end: pd.Timestamp, out: Path,
                script: Path) -> dict[str, Any]:
    result = subprocess.run(  # noqa: S603 - fixed uv invocation with a generated dumper
        ["uv", "run", "--project", str(tree), "python", "-I", str(script),
         str(tree), str(data_root), start.isoformat(), end.isoformat(), str(out)],
        cwd=tree, capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise EquivalenceError(f"dumper failed in {tree}: {result.stderr.strip()[-400:]}")
    parsed: dict[str, Any] = json.loads(result.stdout.strip().splitlines()[-1])
    return parsed


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()  # noqa: S603


def run(args: argparse.Namespace) -> int:
    from src.core.params import BOOK_DEGENERATE_GROSS_RTOL, LIVE_SIGNAL_WARMUP_DAYS
    from src.strategy.books import clip_names_preserving_gross
    from src.strategy.release import (
        ImplementationAmendment,
        append_implementation_amendment,
        effective_spec_digest,
        load_release,
    )
    from src.strategy.store import refuse_example_store, strategy_values

    data_root = Path(args.data_root).resolve()
    baseline_commit = _git("rev-parse", "--verify", f"{args.baseline_commit}^{{commit}}")
    candidate_commit = _git("rev-parse", "HEAD")
    clean = _git("status", "--porcelain", "--untracked-files=no") == ""
    if not clean:
        print("warning: candidate tree has tracked changes; candidate_commit will be null", file=sys.stderr)

    if args.start and args.end:
        start, end = pd.Timestamp(args.start, tz="UTC"), pd.Timestamp(args.end, tz="UTC")
    else:
        start, end = default_window(data_root, int(LIVE_SIGNAL_WARMUP_DAYS))
    release = load_release(_STRATEGY_ID)
    accepted = effective_spec_digest(release)

    warmup = pd.Timedelta(days=int(LIVE_SIGNAL_WARMUP_DAYS))
    chunks = decision_chunks(start, end, int(args.chunk_days))
    with tempfile.TemporaryDirectory(prefix="release_equivalence_") as tmp:
        tmp_path = Path(tmp)
        worktree = tmp_path / "baseline"
        _git("worktree", "add", "--detach", str(worktree), baseline_commit)
        try:
            script = tmp_path / "dump_unit_book.py"
            script.write_text(_DUMPER, encoding="utf-8")
            base_frames: list[pd.DataFrame] = []
            cand_frames: list[pd.DataFrame] = []
            base_pre: list[pd.DataFrame] = []
            cand_pre: list[pd.DataFrame] = []
            base: dict[str, Any] = {}
            cand: dict[str, Any] = {}
            for n, (c_start, c_end) in enumerate(chunks):
                # Each chunk panel starts one warmup earlier so its first decision row is c_start.
                base_out = tmp_path / f"baseline_unit_{n}.parquet"
                cand_out = tmp_path / f"candidate_unit_{n}.parquet"
                base = _run_dumper(worktree, data_root, c_start - warmup, c_end, base_out, script)
                cand = _run_dumper(ROOT, data_root, c_start - warmup, c_end, cand_out, script)
                base_frames.append(pd.read_parquet(base_out))
                cand_frames.append(pd.read_parquet(cand_out))
                base_pre.append(pd.read_parquet(f"{base_out}.preclip.parquet"))
                cand_pre.append(pd.read_parquet(f"{cand_out}.preclip.parquet"))
                print(f"chunk {n + 1}/{len(chunks)} {c_start.date()}..{c_end.date()} ok", file=sys.stderr)
            if base["digest"] != accepted:
                raise EquivalenceError("baseline digest does not equal the effective accepted digest")
            if cand["digest"] == accepted:
                raise EquivalenceError("candidate digest equals the accepted digest: nothing to amend")
            if cand["payload"] is None:
                raise EquivalenceError("candidate has no strategy_spec_payload")
            reconstructed = baseline_payload_from_candidate(cand["payload"], worktree / "src")
            if _canonical_digest(reconstructed) != base["digest"]:
                raise EquivalenceError("baseline payload is not the candidate payload with implementation keys replaced")
            check_payload_scope(cand["payload"], reconstructed)
        finally:
            _git("worktree", "remove", "--force", str(worktree))

    stats: dict[str, Any] = {"rows": 0, "cols": 0, "max_abs_diff": 0.0, "pre_clip_max_abs_diff": 0.0}
    attributed: list[str] = []
    for b_frame, c_frame, b_pre, c_pre in zip(base_frames, cand_frames, base_pre, cand_pre, strict=True):
        pre_part = compare_unit_weights(b_pre, c_pre)
        rows, _ = divergent_rows(b_frame, c_frame)
        attributed += attribute_rows_to_noise(
            b_pre, c_frame, rows, clip_fn=clip_names_preserving_gross, clip=float(strategy_values().name_clip),
            noise_rtol=float(BOOK_DEGENERATE_GROSS_RTOL),
        )
        kept = np.setdiff1d(np.arange(len(b_frame)), rows)
        stats["rows"] = int(stats["rows"]) + len(b_frame)
        stats["cols"] = max(int(stats["cols"]), b_frame.shape[1])
        stats["max_abs_diff"] = max(
            float(stats["max_abs_diff"]),
            float(compare_unit_weights(b_frame.iloc[kept], c_frame.iloc[kept])["max_abs_diff"]) if kept.size else 0.0,
        )
        stats["pre_clip_max_abs_diff"] = max(float(stats["pre_clip_max_abs_diff"]), float(pre_part["max_abs_diff"]))
    print(f"window={start.date()}..{end.date()} rows={stats['rows']} cols={stats['cols']} "
          f"max_abs_diff={stats['max_abs_diff']:.3e} pre_clip_max_abs_diff={stats['pre_clip_max_abs_diff']:.3e} "
          f"noise_attributed={attributed}")

    evidence = {
        "strategy_id": _STRATEGY_ID,
        "from_spec_digest": base["digest"],
        "to_spec_digest": cand["digest"],
        "baseline_commit": baseline_commit,
        "candidate_commit": candidate_commit,
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "decision_rows": stats["rows"],
        "decision_cols": stats["cols"],
        "max_abs_diff": stats["max_abs_diff"],
        "sign_diffs": 0,
        "zero_diffs": 0,
        "pre_clip_max_abs_diff": stats["pre_clip_max_abs_diff"],
        "noise_attributed_decisions": attributed,
        "atol": RELEASE_EQUIVALENCE_ATOL,
        "pandas": {"baseline": base["pandas"], "candidate": cand["pandas"]},
        "payload_digests": {"baseline": _canonical_digest(reconstructed), "candidate": cand["digest"]},
    }
    evidence_path = ROOT / _EVIDENCE_DIR / f"{cand['digest'][:12]}.json"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evidence_sha256 = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    print(f"evidence={evidence_path} sha256={evidence_sha256}")

    if args.record:
        refuse_example_store("release equivalence record")
        amendment = ImplementationAmendment(
            from_spec_digest=base["digest"],
            to_spec_digest=cand["digest"],
            baseline_commit=baseline_commit,
            candidate_commit=candidate_commit if clean else None,
            evidence_sha256=evidence_sha256,
            window_start=start.isoformat(),
            window_end=end.isoformat(),
            decision_rows=int(stats["rows"]),
            max_abs_diff=float(stats["max_abs_diff"]),
            recorded_at=dt.datetime.now(dt.UTC).isoformat(),
        )
        append_implementation_amendment(_STRATEGY_ID, amendment)
        print(f"recorded amendment {base['digest'][:12]} -> {cand['digest'][:12]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--strategy", required=True, choices=[_STRATEGY_ID])
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--start", default=None, help="UTC date, default: earliest common bar plus warmup")
    parser.add_argument("--end", default=None, help="UTC date, default: latest bar")
    parser.add_argument("--record", action="store_true", help="append the amendment to the release record")
    parser.add_argument("--chunk-days", type=int, default=365, help="decision days per dumper run (bounds memory)")
    args = parser.parse_args(argv)
    if bool(args.start) != bool(args.end):
        parser.error("--start and --end must be given together")
    try:
        return run(args)
    except EquivalenceError as exc:
        print(f"REJECTED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
