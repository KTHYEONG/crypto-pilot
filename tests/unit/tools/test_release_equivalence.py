"""Release implementation-equivalence tool: pure comparison and payload rules, no git or worktree."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.core.params import BOOK_DEGENERATE_GROSS_RTOL
from src.strategy.books import clip_names_preserving_gross
from tools.release_equivalence import (
    RELEASE_EQUIVALENCE_ATOL,
    EquivalenceError,
    attribute_rows_to_noise,
    baseline_payload_from_candidate,
    check_payload_scope,
    compare_unit_weights,
    default_window,
    divergent_rows,
    main,
)

TOOL = Path(__file__).resolve().parents[3] / "tools" / "release_equivalence.py"


def _frame(values: list[list[float]]) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=len(values), freq="D", tz="UTC")
    return pd.DataFrame(values, index=index, columns=["AAA", "BBB", "CCC"])


def test_compare_accepts_ulp_noise() -> None:
    base = _frame([[0.5, -0.25, 0.0], [0.0, 0.5, -0.5]])
    noisy = base * (1.0 + 4e-16)
    stats = compare_unit_weights(base, noisy)
    assert stats["rows"] == 2
    assert stats["cols"] == 3
    assert stats["sign_diffs"] == 0
    assert stats["zero_diffs"] == 0
    assert 0.0 < stats["max_abs_diff"] <= RELEASE_EQUIVALENCE_ATOL


def test_compare_rejects_sign_flip() -> None:
    base = _frame([[0.5, -0.25, 0.0]])
    flipped = base.copy()
    flipped.iloc[0, 1] = 0.25
    with pytest.raises(EquivalenceError, match="sign or zero"):
        compare_unit_weights(base, flipped)


def test_compare_rejects_zero_pattern_change() -> None:
    base = _frame([[0.5, 0.0, -0.5]])
    touched = base.copy()
    touched.iloc[0, 1] = 1e-9
    with pytest.raises(EquivalenceError, match="sign or zero"):
        compare_unit_weights(base, touched)


@pytest.mark.parametrize(("left", "right"), [(0.0, 1e-17), (-2.4e-18, 7.3e-19)])
def test_compare_treats_sub_atol_cells_as_zero(left: float, right: float) -> None:
    base = _frame([[0.5, left, -0.5]])
    other = _frame([[0.5, right, -0.5]])
    stats = compare_unit_weights(base, other)
    assert stats["sign_diffs"] == 0
    assert stats["zero_diffs"] == 0


def _clip_case() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Baseline pre-clip row with a noise cell, its old (noise-held) and new (noise-free) clips."""
    index = pd.date_range("2022-02-26", periods=1, freq="D", tz="UTC")
    cols = [f"S{i}" for i in range(6)]
    pre = pd.DataFrame([[0.3, 0.1, -0.05, -0.15, -0.2, 1.6653345369377347e-17]], index=index, columns=cols)

    def old_clip(weights: pd.DataFrame, clip: float) -> pd.DataFrame:
        values = weights.to_numpy(dtype="float64")
        held = values != 0.0
        clipped = np.sign(values) * np.minimum(np.abs(values), clip)
        centered = np.where(held, clipped - clipped.sum(axis=1, keepdims=True) / held.sum(axis=1, keepdims=True), 0.0)
        scale = np.abs(values).sum(axis=1, keepdims=True) / np.abs(centered).sum(axis=1, keepdims=True)
        return pd.DataFrame(centered * scale, index=weights.index, columns=weights.columns)

    return pre, old_clip(pre, 0.2), clip_names_preserving_gross(pre, 0.2)


def test_noise_attribution_accepts_the_noise_fix() -> None:
    pre, old, new = _clip_case()
    rows, _ = divergent_rows(old, new)
    assert rows.tolist() == [0]
    labels = attribute_rows_to_noise(
        pre, new, rows, clip_fn=clip_names_preserving_gross, clip=0.2, noise_rtol=BOOK_DEGENERATE_GROSS_RTOL,
    )
    assert labels == [str(pre.index[0])]


def test_noise_attribution_rejects_row_without_noise_cell() -> None:
    pre, _, new = _clip_case()
    clean = pre.copy()
    clean.iloc[0, 5] = 0.0
    with pytest.raises(EquivalenceError, match="without a baseline noise cell"):
        attribute_rows_to_noise(
            clean, new, np.array([0]), clip_fn=clip_names_preserving_gross, clip=0.2,
            noise_rtol=BOOK_DEGENERATE_GROSS_RTOL,
        )


def test_noise_attribution_rejects_unreproduced_decision() -> None:
    pre, _, new = _clip_case()
    moved = new.copy()
    moved.iloc[0, 0] += 1e-6
    moved.iloc[0, 1] -= 1e-6
    with pytest.raises(EquivalenceError, match="does not reproduce"):
        attribute_rows_to_noise(
            pre, moved, np.array([0]), clip_fn=clip_names_preserving_gross, clip=0.2,
            noise_rtol=BOOK_DEGENERATE_GROSS_RTOL,
        )


def test_compare_rejects_index_or_column_change() -> None:
    base = _frame([[0.5, -0.25, 0.0]])
    renamed = base.rename(columns={"CCC": "DDD"})
    with pytest.raises(EquivalenceError, match="index or columns"):
        compare_unit_weights(base, renamed)
    shifted = base.set_axis(base.index + pd.Timedelta(days=1))
    with pytest.raises(EquivalenceError, match="index or columns"):
        compare_unit_weights(base, shifted)


def test_compare_rejects_difference_above_atol_and_nan_pattern() -> None:
    base = _frame([[0.5, -0.25, 0.0]])
    drifted = base + np.array([[1e-9, 0.0, 0.0]])
    with pytest.raises(EquivalenceError, match="exceeds atol"):
        compare_unit_weights(base, drifted)
    nan_base = _frame([[np.nan, -0.25, 0.0]])
    with pytest.raises(EquivalenceError, match="NaN pattern"):
        compare_unit_weights(nan_base, base)
    same_nan = compare_unit_weights(nan_base, nan_base)
    assert same_nan["max_abs_diff"] == 0.0


def test_payload_diff_only_implementation_keys() -> None:
    candidate = {"implementation": {"books.py": "new"}, "shared_parameters": "new", "sizing": {"exposure_cap": 10.0}, "members": [["m", 1]]}
    baseline = {"implementation": {"books.py": "old"}, "shared_parameters": "old", "sizing": {"exposure_cap": 10.0}, "members": [["m", 1]]}
    assert check_payload_scope(candidate, baseline) == frozenset({"implementation", "shared_parameters"})
    resized = {**baseline, "sizing": {"exposure_cap": 3.0}}
    with pytest.raises(EquivalenceError, match="outside implementation keys"):
        check_payload_scope(candidate, resized)
    reseated = {**baseline, "members": [["m", -1]]}
    with pytest.raises(EquivalenceError, match="members"):
        check_payload_scope(candidate, reseated)


def test_baseline_payload_takes_baseline_file_bytes(tmp_path) -> None:
    src = tmp_path / "src"
    (src / "strategy").mkdir(parents=True)
    (src / "core").mkdir(parents=True)
    for name in ("features.py", "targets.py", "books.py", "universe.py", "sizing.py"):
        (src / "strategy" / name).write_text(f"# baseline {name}\n", encoding="utf-8")
    (src / "core" / "params.py").write_text("BASE = 1\n", encoding="utf-8")
    candidate = {"strategy_id": "flow_mom_top20", "implementation": {"books.py": "candidate"}, "shared_parameters": "candidate"}
    rebuilt = baseline_payload_from_candidate(candidate, src)
    assert rebuilt["strategy_id"] == "flow_mom_top20"
    assert rebuilt["implementation"]["books.py"] == hashlib.sha256(b"# baseline books.py\n").hexdigest()
    assert rebuilt["shared_parameters"] == hashlib.sha256(b"BASE = 1\n").hexdigest()
    assert candidate["implementation"] == {"books.py": "candidate"}


def test_default_window_starts_after_earliest_bar_plus_warmup(tmp_path) -> None:
    bars = tmp_path / "ohlcv" / "1h"
    bars.mkdir(parents=True)
    hour = 3_600_000
    base = int(pd.Timestamp("2026-01-01", tz="UTC").value // 1_000_000)
    pd.DataFrame({"timestamp": [base + 5 * hour, base + 9 * hour]}).to_parquet(bars / "AAAUSDT.parquet")
    pd.DataFrame({"timestamp": [base, base + 100 * hour]}).to_parquet(bars / "BBBUSDT.parquet")
    start, end = default_window(tmp_path, warmup_days=1)
    assert start == pd.Timestamp(base, unit="ms", tz="UTC").normalize() + pd.Timedelta(days=1)
    assert end == pd.Timestamp(base + 100 * hour, unit="ms", tz="UTC")


def test_default_window_without_bars_fails_closed(tmp_path) -> None:
    (tmp_path / "ohlcv" / "1h").mkdir(parents=True)
    with pytest.raises(EquivalenceError, match="no 1h bars"):
        default_window(tmp_path, warmup_days=1)


def test_main_rejects_partial_window() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--strategy", "flow_mom_top20", "--baseline-commit", "HEAD", "--data-root", "data/futures",
              "--start", "2025-01-01"])
    assert exc.value.code == 2


def test_never_imports_return_metrics() -> None:
    """The tool sees decision weights only: no evaluation, backtest, CLI or return-statistic symbol."""
    tree = ast.parse(TOOL.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden = ("src.evaluation", "src.cli", "src.lab", "src.live.runner")
    assert not [name for name in imported if name.startswith(forbidden)]
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    banned = ("sharpe", "cagr", "pnl", "drawdown", "statistic", "evaluat", "backtest", "dsr")
    assert not [name for name in names if any(token in name.lower() for token in banned)]


def test_decision_chunks_are_contiguous_and_cover_window() -> None:
    import itertools

    from tools.release_equivalence import decision_chunks

    start, end = pd.Timestamp("2021-05-01", tz="UTC"), pd.Timestamp("2023-02-10", tz="UTC")
    chunks = decision_chunks(start, end, 365)
    assert chunks[0][0] == start
    assert chunks[-1][1] == end
    assert all(prev[1] == nxt[0] for prev, nxt in itertools.pairwise(chunks))
    assert all(c_end - c_start <= pd.Timedelta(days=365) for c_start, c_end in chunks)
