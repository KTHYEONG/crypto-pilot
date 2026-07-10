#!/usr/bin/env python3
# scripts/update_index.py
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys


def update_index(
    index_path: str = "docs/index.json",
    *,
    source_file: str,
    test_file: str | None = None,
    doc_file: str | None = None,
) -> int:
    """Add or update file mappings inside docs/index.json.

    Args:
        index_path: Path to the index JSON metadata database.
        source_file: Relative path to the source module file.
        test_file: Relative path to the unit test file.
        doc_file: Relative path to the architecture documentation.

    Returns:
        0 on success, non-zero error code on failure.
    """
    # 1. Source file validation (LIMIT-03)
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Source file does not exist: {source_file}")

    # 2. Read existing index with broken-JSON failback (LIMIT-04)
    data: dict[str, dict[str, str | list[str]]] = {}
    if os.path.exists(index_path):
        try:
            with open(index_path, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                data = {}
        except json.JSONDecodeError:
            # Backup broken file and initialize
            backup_path = f"{index_path}.bak"
            shutil.copyfile(index_path, backup_path)
            data = {}

    # 3. Add or update mapping
    if source_file not in data:
        data[source_file] = {}

    entry = data[source_file]

    # Handle doc update
    if doc_file is not None:
        entry["architecture"] = doc_file

    # Handle test update (with deduplicated list merging)
    if test_file is not None:
        current_test = entry.get("testing")
        if current_test is None:
            entry["testing"] = test_file
        elif isinstance(current_test, str):
            if current_test != test_file:
                entry["testing"] = [current_test, test_file]
        elif isinstance(current_test, list) and test_file not in current_test:
            current_test.append(test_file)

    # 4. Write back pretty formatted JSON
    os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Update file mappings in index.json.")
    parser.add_argument("--index-path", default="docs/index.json")
    parser.add_argument("--source", required=True, help="Relative path to source file.")
    parser.add_argument("--test", default=None, help="Relative path to test file.")
    parser.add_argument("--doc", default=None, help="Relative path to doc file.")
    args = parser.parse_args()

    try:
        return update_index(
            index_path=args.index_path,
            source_file=args.source,
            test_file=args.test,
            doc_file=args.doc,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
