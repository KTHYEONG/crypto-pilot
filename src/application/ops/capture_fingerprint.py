"""Content fingerprint of the raw-first capture closure for selective deploy handovers."""

from __future__ import annotations

import argparse
import ast
import hashlib
import platform
import re
from collections.abc import Sequence
from pathlib import Path

CAPTURE_PACKAGE_DIR: str = "src/capture"
CAPTURE_LOCK_ROOT_PACKAGE: str = "aiohttp"


class FingerprintError(RuntimeError):
    """The capture closure cannot be resolved; the image build must fail rather than emit a fingerprint."""


def _first_party_refs(tree: ast.AST) -> list[tuple[str | None, int, str | None]]:
    """Raw import statements as ``(module, level, name)`` triples for first-party screening."""
    refs: list[tuple[str | None, int, str | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            refs.extend(
                (alias.name, 0, None)
                for alias in node.names
                if alias.name == "src" or alias.name.startswith("src.")
            )
        elif isinstance(node, ast.ImportFrom):
            refs.append((node.module, node.level, None))
    return refs


def _check_file(path: Path, module: str) -> None:
    """Raise when ``path`` imports a first-party module outside ``src.capture``."""
    try:
        tree = ast.parse(path.read_bytes(), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise FingerprintError(f"cannot parse {path}: {exc}") from exc
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    for mod, level, _ in _first_party_refs(tree):
        if level:
            base = package
            for _ in range(level - 1):
                base = base.rpartition(".")[0]
            full = f"{base}.{mod}" if mod else base
            if full != "src.capture" and not full.startswith("src.capture."):
                raise FingerprintError(f"first-party import outside src.capture in {path}: {full}")
        elif mod is not None and (mod == "src" or mod.startswith("src.")):
            if mod != "src.capture" and not mod.startswith("src.capture."):
                raise FingerprintError(f"first-party import outside src.capture in {path}: {mod}")


def capture_source_files(root: Path) -> tuple[Path, ...]:
    """Every ``*.py`` under ``src/capture`` plus ``src/__init__.py``, sorted by relative POSIX path.

    The capture package is stdlib + aiohttp only (enforced by spec 01's boundary test), so its runtime
    closure is exactly this directory; no AST walk into other first-party packages is needed.

    Raises:
        FingerprintError: ``src/capture`` is missing or any file imports a first-party module outside
            ``src.capture`` (a boundary breach would otherwise let unfingerprinted code run in capture).
    """
    base = Path(root)
    package_dir = base / CAPTURE_PACKAGE_DIR
    if not package_dir.is_dir():
        raise FingerprintError(f"missing capture package: {package_dir}")
    files = sorted(package_dir.rglob("*.py"))
    init = base / "src" / "__init__.py"
    if init.is_file():
        files = [init, *files]
    for path in files:
        rel = path.relative_to(base).with_suffix("")
        module = rel.as_posix().replace("/", ".")
        _check_file(path, module)
    return tuple(sorted(files, key=lambda p: p.relative_to(base).as_posix()))


def _split_lock_blocks(lock_text: str) -> list[str]:
    """Raw ``[[package]]`` blocks in lock order, each prefixed with its header line."""
    parts = re.split(r"(?m)^\[\[package\]\]\s*$", lock_text)
    return ["[[package]]\n" + chunk.strip("\n") + "\n" for chunk in parts[1:]]


def _block_name(block: str) -> str | None:
    match = re.search(r'(?m)^name\s*=\s*"([^"]+)"', block)
    return match.group(1) if match else None


def _block_dep_names(block: str) -> list[str]:
    return re.findall(r'\{\s*name\s*=\s*"([^"]+)"', block)


def locked_dependency_blocks(lock_text: str, root_package: str = CAPTURE_LOCK_ROOT_PACKAGE) -> tuple[str, ...]:
    """The ``[[package]]`` blocks of ``root_package`` and its transitive runtime dependencies in ``uv.lock``.

    Only these blocks change capture's runtime behavior; the rest of the lock (pandas, pyarrow, …) must not
    trigger a capture handover.

    Raises:
        FingerprintError: ``root_package`` or a referenced dependency has no block in the lock.
    """
    blocks = _split_lock_blocks(lock_text)
    by_name: dict[str, str] = {}
    for block in blocks:
        name = _block_name(block)
        if name is not None and name not in by_name:
            by_name[name] = block
    if root_package not in by_name:
        raise FingerprintError(f"missing lock block for {root_package}")
    wanted: list[str] = []
    queue: list[str] = [root_package]
    seen = {root_package}
    while queue:
        current = queue.pop(0)
        current_block = by_name[current]
        wanted.append(current)
        for dep in _block_dep_names(current_block):
            if dep not in seen:
                if dep not in by_name:
                    raise FingerprintError(f"missing lock block for {dep}")
                seen.add(dep)
                queue.append(dep)
    wanted_set = set(wanted)
    return tuple(block for block in blocks if _block_name(block) in wanted_set)


def compute_capture_fingerprint(root: Path) -> str:
    """``"sha256:<hex>"`` over capture sources (path + bytes), the locked aiohttp dependency blocks and the
    interpreter version (``platform.python_version()``), in that fixed order with NUL separators."""
    base = Path(root)
    digest = hashlib.sha256()
    for path in capture_source_files(base):
        relative = path.relative_to(base).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(path.read_bytes())
        digest.update(b"\x00")
    lock_path = base / "uv.lock"
    if not lock_path.is_file():
        raise FingerprintError("missing fingerprint input: uv.lock")
    for block in locked_dependency_blocks(lock_path.read_text(encoding="utf-8")):
        digest.update(block.encode("utf-8"))
        digest.update(b"\x00")
    digest.update(platform.python_version().encode("utf-8"))
    return f"sha256:{digest.hexdigest()}"


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m src.application.ops.capture_fingerprint [--root P] [--list]``; prints the closure (with
    ``--list``) then one fingerprint line. Returns 0."""
    parser = argparse.ArgumentParser(description="Print the raw-first capture content fingerprint.")
    parser.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--list", action="store_true", help="Print closure paths before the fingerprint")
    args = parser.parse_args(argv)
    root = Path(args.root)
    if args.list:
        for path in capture_source_files(root):
            print(path.relative_to(root).as_posix())  # noqa: T201 - contract-mandated CLI output
    print(compute_capture_fingerprint(root))  # noqa: T201 - contract-mandated CLI output
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
