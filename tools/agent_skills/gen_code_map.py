#!/usr/bin/env python3
# ruff: noqa: T201, S110
from __future__ import annotations

import contextlib
import json
import os
import pathlib
import sys

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())


def _all_test_files() -> list[str]:
    return sorted(
        str(p)
        for p in pathlib.Path("tests").rglob("*.py")
        if "__pycache__" not in p.parts
    )


def _matching_tests(source_file: str, test_files: list[str]) -> list[str]:
    """Return every repository test that covers ``source_file``."""
    parts = source_file.split("/")
    module_name = parts[-1]
    module_stem = module_name[:-3] if module_name.endswith(".py") else module_name
    test_name = f"test_{module_stem}.py"
    exact = {
        f"tests/{category}/{'/'.join(parts[1:-1])}/{test_name}" if parts[1:-1]
        else f"tests/{category}/{test_name}"
        for category in ("unit", "integration", "contract", "e2e")
    }
    matched = [tp for tp in test_files if tp in exact or tp.endswith(f"/{test_name}")]
    if matched:
        return sorted(set(matched))

    dotted = ".".join(parts).removesuffix(".py")
    refs: list[str] = []
    for tp in test_files:
        try:
            content = pathlib.Path(tp).read_text(encoding="utf-8")
            if dotted in content or f"import {module_stem}" in content:
                refs.append(tp)
        except Exception:
            pass
    return sorted(set(refs))


def main() -> None:
    py_files: list[str] = []
    for root, dirs, files in os.walk("src"):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        py_files.extend(
            os.path.join(root, filename)
            for filename in sorted(files)
            if filename.endswith(".py")
        )
    py_files = sorted(py_files)
    test_files = _all_test_files()

    code_map: dict[str, object] = {}
    for source_file in py_files:
        if source_file.endswith("__init__.py"):
            continue
        matched = _matching_tests(source_file, test_files)
        # Keep only existing test files
        matched = [tp for tp in matched if pathlib.Path(tp).exists()]
        entry: dict[str, object] = {}
        if matched:
            entry["testing"] = matched[0] if len(matched) == 1 else matched
        code_map[source_file] = entry

    docs_path = pathlib.Path("docs/code_map.json")
    with contextlib.suppress(FileNotFoundError):
        if not docs_path.parent.exists():
            docs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(docs_path, "w", encoding="utf-8") as handle:
        json.dump(code_map, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"regenerated docs/code_map.json with {len(code_map)} canonical sources")


if __name__ == "__main__":
    main()
