"""Part 3 naming contract: no strategy-meaning ``frozen`` remains in ``src/``.

Every identifier, module name, CLI parser name, docstring and log tag where
"frozen"/"Frozen" meant the deployed strategy (or its backtest) was renamed to
strategy/release naming. What stays is explicit below:

- ``frozenset`` (builtin container; only shares the substring).
- ``frozen=True`` / ``frozen=False`` / ``frozen+slots`` (dataclass spellings).
- Keep-list with a different meaning: ``frozen_gc_heap`` / ``_INHERITED_FROZEN``
  (gc.freeze), ``frozen_at`` (preregistration timestamp), ``risk_increase_frozen``,
  ``frozen_default``, ``frozen_symbols`` / ``frozen_blocked`` /
  ``frozen_blocked_count`` (unresolved-order freeze, never the strategy roster).
- ``LEGACY_STRATEGY_IDS`` keys (pre-rename evidence keeps resolving).
- ``LEGACY_FROZEN_BACKTESTS_DIR`` path segment ``"frozen"`` (old runs root,
  read-only) and its docstring.
- Persisted catalog kinds ``"mhs_frozen"`` / ``"mhs_frozen_account"``.
- Wire string values: ``"frozen_signal_report.json"``,
  ``"frozen_unit_forward.parquet"``,
  ``"deploy/mhs/frozen_unit_returns_maker.parquet.enc"``.
- Sealed procedure value ``"boundary_frozen_warmup_excluded_v1"`` (renaming it
  would re-key trial identities; it denotes a fixed warmup rule, not the strategy).
- gc-freeze prose in ``src/core/parallel.py``.
"""

from __future__ import annotations

import io
import re
import tokenize
from collections.abc import Iterator
from pathlib import Path
from typing import Final

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_SRC_ROOT: Final[Path] = _REPO_ROOT / "src"

_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9_]*frozen[A-Za-z0-9_]*", re.IGNORECASE)

_GLOBAL_ALLOWED_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "frozenset",
        "frozen_gc_heap",
        "frozen_at",
        "risk_increase_frozen",
        "frozen_default",
        "frozen_symbols",
        "frozen_blocked",
        "frozen_blocked_count",
    }
)

_FILE_ALLOWED_TOKENS: Final[dict[str, frozenset[str]]] = {
    "src/strategy/targets.py": frozenset({
        "frozen_mhs_top20_v2", "frozen_mhs_top40_control_v2", "frozen_mhs_top20_growth_v2",
    }),
    "src/core/parallel.py": frozenset({"_INHERITED_FROZEN"}),
    "src/lab/mhs/params.py": frozenset({"boundary_frozen_warmup_excluded_v1"}),
    "src/common/paths.py": frozenset({"LEGACY_FROZEN_BACKTESTS_DIR"}),
    "src/application/strategy_account.py": frozenset({"mhs_frozen"}),
    "src/application/strategy_account_persist.py": frozenset({"mhs_frozen_account"}),
    "src/backtests/catalog.py": frozenset({"mhs_frozen", "mhs_frozen_account"}),
    "src/cli/commands/backtest.py": frozenset({"mhs_frozen"}),
    "src/live/strategy_signal.py": frozenset({"frozen_signal_report"}),
    "src/live/scheduler.py": frozenset({"frozen_unit_forward"}),
    "src/live/settings.py": frozenset({"frozen_unit_returns_maker"}),
}

# relpath -> bare-"frozen" occurrences that are not dataclass spellings.
_FILE_ALLOWED_BARE: Final[dict[str, int]] = {
    "src/common/paths.py": 2,
    "src/core/parallel.py": 5,
}

_DATACLASS_SUFFIXES: Final[tuple[str, ...]] = ("=True", "=False", "+slots")


def _code_text(path: Path) -> str:
    """File text with every comment token removed (identifiers and strings stay)."""
    text = path.read_text(encoding="utf-8")
    comment_cols: dict[int, int] = {}
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.COMMENT:
            comment_cols[tok.start[0]] = tok.start[1]
    lines = text.splitlines()
    return "\n".join(
        line[: comment_cols[lineno]] if lineno in comment_cols else line
        for lineno, line in enumerate(lines, 1)
    )


def _iter_offenders() -> Iterator[str]:
    """Yield human-readable descriptions of every non-allowlisted ``frozen`` hit."""
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(_REPO_ROOT).as_posix()
        if _TOKEN_PATTERN.search(path.name) and "frozen" in path.name.lower():
            yield f"{rel}: module name carries frozen"
        text = _code_text(path)
        bare = 0
        for match in _TOKEN_PATTERN.finditer(text):
            hit = match.group()
            if hit in _GLOBAL_ALLOWED_TOKENS or hit in _FILE_ALLOWED_TOKENS.get(rel, frozenset()):
                continue
            if hit == "frozen" and text[match.end() :].startswith(_DATACLASS_SUFFIXES):
                continue
            if hit == "frozen":
                bare += 1
                continue
            yield f"{rel}: non-allowlisted frozen token {hit!r}"
        allowed_bare = _FILE_ALLOWED_BARE.get(rel, 0)
        if bare != allowed_bare:
            yield f"{rel}: bare 'frozen' occurrences {bare} != allowlisted {allowed_bare}"


def _iter_cli_names() -> Iterator[str]:
    """Every CLI subcommand name registered on the production root parser."""
    from src.cli.main import build_root_parser

    def _walk(parser: object, prefix: str) -> Iterator[str]:
        for action in getattr(parser, "_actions", []):
            choices = getattr(action, "choices", None)
            if not isinstance(choices, dict):
                continue
            for name, sub in choices.items():
                full = f"{prefix} {name}".strip()
                yield full
                yield from _walk(sub, full)

    yield from _walk(build_root_parser(), "")


def test_no_frozen_identifier_module_or_string_in_src() -> None:
    """Only the explicit allowlist above may carry ``[Ff]rozen`` under ``src/``."""
    offenders = list(_iter_offenders())
    assert offenders == [], "non-allowlisted frozen hits:\n" + "\n".join(offenders)


def test_no_frozen_cli_parser_name() -> None:
    """Renamed CLI surface parses; no subcommand name carries ``[Ff]rozen``."""
    offenders = [name for name in _iter_cli_names() if _TOKEN_PATTERN.search(name)]
    assert offenders == [], f"frozen CLI names: {offenders}"
