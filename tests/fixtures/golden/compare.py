"""Golden identity comparison helpers for MHS report bit-exact verification.

Floats are compared via ``repr()`` string equality (no tolerance).  A
``renames`` mapping allows the golden to carry old key names that are
renamed before diffing.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Closed set of named golden fixtures produced by capture_matrix.py and exercised
# by the golden-identity test (one report payload per entry).
GOLDEN_MATRIX_NAMES: tuple[str, ...] = (
    "baseline",
    "committee",
    "discovery",
    "trend_sleeve",
    "fold_safe",
)


def _apply_renames(obj: Any, renames: Mapping[str, str]) -> Any:
    """Recursively rename dict keys using the renames mapping."""
    if isinstance(obj, dict):
        return {renames.get(k, k): _apply_renames(v, renames) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_apply_renames(item, renames) for item in obj]
    return obj


def _diff(
    golden: Any,
    actual: Any,
    path: str,
    renames: Mapping[str, str],
    exclude: frozenset[str],
) -> str | None:
    """Return the first divergent json-pointer path, or None if identical."""
    if isinstance(golden, dict) and isinstance(actual, dict):
        all_keys = set(golden) | set(actual)
        for k in sorted(all_keys):
            if k in exclude:
                continue
            if k not in golden:
                return f"{path}/{k}: missing in golden"
            if k not in actual:
                return f"{path}/{k}: missing in actual"
            result = _diff(golden[k], actual[k], f"{path}/{k}", renames, exclude)
            if result is not None:
                return result
        return None
    if isinstance(golden, list) and isinstance(actual, list):
        if len(golden) != len(actual):
            return f"{path}: list length mismatch {len(golden)} != {len(actual)}"
        for i, (g, a) in enumerate(zip(golden, actual, strict=True)):
            result = _diff(g, a, f"{path}[{i}]", renames, exclude)
            if result is not None:
                return result
        return None
    # Leaf comparison: floats via repr() string equality
    g_repr = repr(golden)
    a_repr = repr(actual)
    if g_repr != a_repr:
        return f"{path}: {g_repr} != {a_repr}"
    return None


def assert_report_identical(
    golden: Mapping[str, Any],
    actual: Mapping[str, Any],
    renames: Mapping[str, str] | None = None,
    exclude: frozenset[str] | None = None,
) -> None:
    """Assert two report dicts are bit-exact after applying key renames.

    Floats are compared via ``repr()`` string equality (no tolerance).
    ``exclude`` is a set of key names to skip (e.g. wall-time fields).
    Raises ``AssertionError`` with the first divergent json-pointer path.
    """
    renames = renames or {}
    exclude = exclude or frozenset()
    renamed_golden = _apply_renames(dict(golden), renames)
    result = _diff(renamed_golden, dict(actual), "", renames, exclude)
    if result is not None:
        raise AssertionError(f"Golden identity mismatch: {result}")


def assert_report_digest_identical(
    golden: Mapping[str, Any],
    actual_report: Any,
    exclude: frozenset[str] | None = None,
) -> None:
    """Assert every replay ledger series of ``actual_report`` matches ``golden``.

    ``golden`` is a ``build_report_digest`` payload. On a sha256 mismatch the
    actual series is recomputed in memory and the first divergent
    ``(replay_id, series, index, golden_repr, actual_repr)`` landmark is named,
    preserving first-divergence diagnosability without a 1 GB file.
    ``exclude`` skips replay_ids (e.g. opt-in diagnostic replays absent from
    the golden). A missing replay/series raises as well — never silently passes.
    """
    from tests.fixtures.golden.digest import build_report_digest, first_divergent_index

    actual = build_report_digest(actual_report)
    excluded = exclude or frozenset()
    for replay_id in sorted(set(golden) - set(actual) - set(excluded)):
        raise AssertionError(f"Golden digest mismatch: replay '{replay_id}' missing from actual")
    for replay_id in sorted(set(actual) - set(golden)):
        raise AssertionError(f"Golden digest mismatch: unexpected replay '{replay_id}' in actual")
    for replay_id, series_map in golden.items():
        if replay_id in excluded:
            continue
        for series_name, block in series_map.items():
            actual_block = actual[replay_id].get(series_name)
            if actual_block is None:
                raise AssertionError(
                    f"Golden digest mismatch: {replay_id}/{series_name} missing from actual"
                )
            if (
                actual_block["n"] == block["n"]
                and actual_block["sha256"] == block["sha256"]
            ):
                continue
            if actual_block["n"] != block["n"]:
                raise AssertionError(
                    f"Golden digest mismatch: {replay_id}/{series_name} length "
                    f"{block['n']} != {actual_block['n']}"
                )
            from src.mhs.report.persist import _collect_replay_entries

            series = next(
                getattr(replay.ledger, series_name)
                for rid, replay in _collect_replay_entries(actual_report)
                if rid == replay_id
            )
            idx, g_repr, a_repr = first_divergent_index(block, series)
            if idx is None:
                idx = min(int(k) for k in block["landmarks"]) if block["landmarks"] else -1
                g_repr = block["landmarks"].get(str(idx), "<between-landmarks>")
                a_repr = "sha256-divergent between landmarks"
            raise AssertionError(
                f"Golden digest mismatch: {replay_id}/{series_name}[{idx}]: "
                f"{g_repr} != {a_repr}"
            )
