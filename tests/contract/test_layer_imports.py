"""Layer import contract for the part-2 packages (spec 38 part 2).

Import direction is strictly downward: core <- strategy <- engine <- evaluation,
with live on core+strategy only. Static AST analysis covers lazy (function-level)
imports and TYPE_CHECKING blocks.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_CORE_ALLOW = frozenset({"core", "common", "market_data", "quant"})
_STRATEGY_ALLOW = frozenset({"core", "common", "market_data", "quant", "strategy"})
_ENGINE_ALLOW = _STRATEGY_ALLOW | {"engine"}
_EVALUATION_ALLOW = _ENGINE_ALLOW | {"evaluation"}
_LIVE_ALLOW = frozenset({"core", "strategy", "common", "market_data", "live", "capture"})

# Built by concatenation so the acceptance grep for stale ``src.mhs.*`` imports
# stays empty; the test asserts none of these remain importable.
_LEGACY_PREFIX = "src." + "mhs."
_MOVED_SOURCES = tuple(
    _LEGACY_PREFIX + name
    for name in (
        "params",
        "types",
        "resources",
        "tree_memory",
        "parallel",
        "bootstrap",
        "panel",
        "data_policy",
        "data_provenance",
        "marks",
        "source_gaps",
        "instrument_settlements",
        "venue_halts",
        "settlement_evidence",
        "books",
        "features",
        "feature_admission",
        "horizons",
        "frozen_research_candidate",
        "frozen_research_universe",
        "account_policy",
        "account_liquidity",
        "execution",
        "account_ledger",
        "account_sources",
        "frozen_research_run",
        "frozen_research_windows",
        "frozen_research_evidence",
        "frozen_research_report",
        "growth_exposure",
    )
)


def module_imports(path: Path) -> frozenset[str]:
    """Absolute ``src.*`` modules imported by one file (lazy and TYPE_CHECKING included)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rel = path.relative_to(REPO_ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    owner = ".".join(parts)
    package = owner if path.name == "__init__.py" else owner.rpartition(".")[0]
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                pkg = package
                for _ in range(node.level - 1):
                    pkg = pkg.rpartition(".")[0]
                base = pkg + ("." + node.module if node.module else "")
            else:
                base = node.module or ""
            if base.startswith("src."):
                found.add(base)
            for alias in node.names:
                if alias.name == "*":
                    continue
                if base == "src" or base.startswith("src."):
                    found.add(f"{base}.{alias.name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("src."):
                    found.add(alias.name)
    return frozenset(found)


def _layer_of_src_module(dotted: str) -> str | None:
    segments = dotted.split(".")
    if len(segments) < 2 or segments[0] != "src":
        return None
    return segments[1]


def _iter_layer_files(layer: str) -> list[Path]:
    root = REPO_ROOT / "src" / layer
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _module_name(path: Path) -> str:
    rel = path.relative_to(REPO_ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _check_allow(layer: str, allow: frozenset[str]) -> list[str]:
    offenders: list[str] = []
    for path in _iter_layer_files(layer):
        for target in sorted(module_imports(path)):
            target_layer = _layer_of_src_module(target)
            if target_layer is not None and target_layer not in allow:
                offenders.append(f"{_module_name(path)} ({layer}) imports {target} ({target_layer})")
    return sorted(offenders)


def test_core_imports_no_higher_layer() -> None:
    """No src.core module imports strategy|engine|evaluation|live|application|cli|mhs."""
    assert _check_allow("core", _CORE_ALLOW) == []


def test_strategy_imports_only_core() -> None:
    """src.strategy imports only core/common/market_data/quant (plus itself)."""
    assert _check_allow("strategy", _STRATEGY_ALLOW) == []


def test_engine_never_imports_live_evaluation_or_lab() -> None:
    """src.engine never imports live, evaluation, application, cli or mhs."""
    assert _check_allow("engine", _ENGINE_ALLOW) == []


def test_evaluation_never_imports_live_or_lab() -> None:
    """src.evaluation never imports live, mhs, application or cli."""
    assert _check_allow("evaluation", _EVALUATION_ALLOW) == []


def test_live_never_imports_engine_evaluation_or_lab() -> None:
    """src.live imports only core/strategy/common/market_data/live/capture."""
    assert _check_allow("live", _LIVE_ALLOW) == []


@pytest.mark.parametrize("statement", [
    "from src import live",
    "from ..live import runner",
    "def build():\n    import src.live.runner",
    "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from src.live import runner",
])
def test_layer_checker_rejects_all_import_forms(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, statement: str) -> None:
    monkeypatch.setattr(sys.modules[__name__], "REPO_ROOT", tmp_path)
    root = tmp_path / "src" / "core"
    root.mkdir(parents=True)
    (root / "probe.py").write_text(statement + "\n", encoding="utf-8")
    offenders = _check_allow("core", _CORE_ALLOW)
    assert offenders
    assert all("src.core.probe (core) imports src.live" in message for message in offenders)


@pytest.mark.parametrize(("layer", "allow"), [
    ("core", _CORE_ALLOW), ("engine", _ENGINE_ALLOW), ("evaluation", _EVALUATION_ALLOW),
])
def test_layer_checker_rejects_unlisted_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layer: str, allow: frozenset[str],
) -> None:
    monkeypatch.setattr(sys.modules[__name__], "REPO_ROOT", tmp_path)
    root = tmp_path / "src" / layer
    root.mkdir(parents=True)
    (root / "probe.py").write_text("from src.backtests.contracts import JsonValue\n", encoding="utf-8")
    assert _check_allow(layer, allow)


def test_no_module_remains_importable_at_old_path() -> None:
    """Every move-map source is gone: find_spec returns None (no shims)."""
    offenders = [name for name in _MOVED_SOURCES if importlib.util.find_spec(name) is not None]
    assert offenders == [], f"old paths still importable: {sorted(offenders)}"


def test_moved_registries_keep_content_digests() -> None:
    """Policy jsonl files moved byte-identical; registry digests equal pre-move values."""
    from src.core.instrument_settlements import load_instrument_settlement_registry
    from src.core.source_gaps import _default_registry_path as _gaps_path
    from src.core.venue_halts import load_venue_halt_registry

    assert (
        load_instrument_settlement_registry().digest
        == "sha256:98668d7d99b85a6c702c21bbdc76265ac1d82dee83c0961f4a92e98999162964"
    )
    assert (
        load_venue_halt_registry().digest
        == "sha256:67b420f149af3f192788192cdba81d262c5744093f1281a8bde6527c9867409c"
    )
    import hashlib

    assert (
        hashlib.sha256(_gaps_path().read_bytes()).hexdigest()
        == "c77df78a0c79586f4012ed3ace4735c7a88987c54e51814e3395faf846a2ebf6"
    )
