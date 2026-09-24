"""Content fingerprint of the market-recorder source closure for selective deploy recreates."""

from __future__ import annotations

import argparse
import ast
import hashlib
import platform
from collections.abc import Sequence
from pathlib import Path

RECORDER_ENTRY_MODULE: str = "src.market_data.streams.recorder_main"
# 런타임 동작을 바꾸는 비-소스 입력: 의존성 잠금, 프로젝트 메타, 이미지 레시피.
FINGERPRINT_EXTRA_FILES: tuple[str, ...] = ("uv.lock", "pyproject.toml", "Dockerfile")


class FingerprintError(RuntimeError):
    """The recorder closure cannot be resolved; the build must fail rather than emit a fingerprint."""


def _module_file(root: Path, dotted: str) -> Path:
    """Resolve a first-party dotted module to its source file under ``root``."""
    parts = dotted.split(".")
    module_path = (root / Path(*parts)).with_suffix(".py")
    if module_path.is_file():
        return module_path
    package_init = root / Path(*parts) / "__init__.py"
    if package_init.is_file():
        return package_init
    raise FingerprintError(f"unresolvable src module: {dotted}")


def _ancestor_inits(root: Path, relative: Path) -> list[Path]:
    """Every ancestor package ``__init__.py`` of an included module that exists on disk."""
    inits: list[Path] = []
    parts = relative.parts
    for depth in range(1, len(parts)):
        candidate = root / Path(*parts[:depth]) / "__init__.py"
        if candidate.is_file():
            inits.append(candidate)
    return inits


def _package_of(module: str, path: Path) -> str:
    """Containing package of ``module`` (itself when ``path`` is an ``__init__.py``)."""
    if path.name == "__init__.py":
        return module
    return module.rpartition(".")[0]


def _resolve_relative(current_module: str, current_path: Path, level: int, module: str | None) -> str:
    """Anchor a relative import to its absolute dotted base."""
    package = _package_of(current_module, current_path)
    base = package
    for _ in range(level - 1):
        base = base.rpartition(".")[0]
    if not base:
        raise FingerprintError(f"relative import escapes src: {current_module}")
    if module:
        return f"{base}.{module}"
    return base


def _referenced_modules(tree: ast.AST, current_module: str, current_path: Path) -> list[tuple[str, bool]]:
    """Import statements in ``tree`` as ``(absolute dotted name, required)`` pairs.

    Plain ``import src.X`` and ``from`` bases must resolve; a ``from P import N`` full name
    (``P.N``) is best-effort and falls back to ``P`` when ``N`` is a plain symbol, not a submodule.
    """
    found: list[tuple[str, bool]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(
                (alias.name, True)
                for alias in node.names
                if alias.name == "src" or alias.name.startswith("src.")
            )
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = _resolve_relative(current_module, current_path, node.level, node.module)
            elif node.module is not None and (node.module == "src" or node.module.startswith("src.")):
                base = node.module
            else:
                continue
            found.append((base, True))
            found.extend(
                (f"{base}.{alias.name}", False) for alias in node.names if alias.name != "*"
            )
    return found


def recorder_source_closure(root: Path, entry_module: str = RECORDER_ENTRY_MODULE) -> tuple[Path, ...]:
    """Every first-party source file the recorder process can execute, sorted by relative path.

    Static AST walk (not runtime import) so function-local and ``TYPE_CHECKING`` imports are
    included; over-inclusion only costs an unnecessary restart, under-inclusion would run stale code.

    Raises:
        FingerprintError: when a referenced ``src.*`` module has no file under ``root``.
    """
    base = Path(root)
    entry_path = _module_file(base, entry_module)
    entry_rel = entry_path.relative_to(base)
    included: dict[str, Path] = {entry_rel.as_posix(): entry_path}
    queue: list[tuple[str, Path]] = [(entry_module, entry_path)]
    while queue:
        module, path = queue.pop()
        try:
            tree = ast.parse(path.read_bytes(), filename=str(path))
        except (OSError, SyntaxError) as exc:
            raise FingerprintError(f"cannot parse {path}: {exc}") from exc
        for dotted, required in sorted(set(_referenced_modules(tree, module, path))):
            try:
                resolved = _module_file(base, dotted)
            except FingerprintError:
                if not required:
                    # `from P import symbol`: P 는 필수 항목으로 이미 검증·포함되므로 심볼은 건너뛴다.
                    continue
                raise
            key = resolved.relative_to(base).as_posix()
            if key not in included:
                included[key] = resolved
                rel_no_suffix = resolved.relative_to(base).with_suffix("")
                if resolved.name == "__init__.py":
                    child_module = rel_no_suffix.parent.as_posix().replace("/", ".")
                else:
                    child_module = rel_no_suffix.as_posix().replace("/", ".")
                queue.append((child_module, resolved))
    for path in list(included.values()):
        for init in _ancestor_inits(base, path.relative_to(base)):
            key = init.relative_to(base).as_posix()
            if key not in included:
                included[key] = init
    return tuple(included[key] for key in sorted(included))


def compute_recorder_fingerprint(root: Path) -> str:
    """``"sha256:<hex>"`` over the closure, ``FINGERPRINT_EXTRA_FILES`` and the interpreter version."""
    base = Path(root)
    digest = hashlib.sha256()
    for path in recorder_source_closure(base):
        relative = path.relative_to(base).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(path.read_bytes())
        digest.update(b"\x00")
    for extra in FINGERPRINT_EXTRA_FILES:
        extra_path = base / extra
        if not extra_path.is_file():
            raise FingerprintError(f"missing fingerprint input: {extra}")
        digest.update(extra.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(extra_path.read_bytes())
        digest.update(b"\x00")
    digest.update(platform.python_version().encode("utf-8"))
    return f"sha256:{digest.hexdigest()}"


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m src.application.ops.recorder_fingerprint [--root P] [--list]``; prints one line."""
    parser = argparse.ArgumentParser(description="Print the market-recorder content fingerprint.")
    parser.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--list", action="store_true", help="Print closure paths before the fingerprint")
    args = parser.parse_args(argv)
    root = Path(args.root)
    if args.list:
        for path in recorder_source_closure(root):
            print(path.relative_to(root).as_posix())  # noqa: T201 - contract-mandated CLI output
    print(compute_recorder_fingerprint(root))  # noqa: T201 - contract-mandated CLI output
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
