#!/usr/bin/env python3
# scripts/archive_decisions.py
from __future__ import annotations

import argparse
import os
import re
import sys


def archive_decisions(
    decisions_path: str = "docs/decisions/decisions.md",
    archive_path: str = "docs/decisions/decisions_archive.md",
    max_entries: int = 15,
) -> int:
    """Read decisions.md, keep latest max_entries, and prepend excess to decisions_archive.md.

    Args:
        decisions_path: Path to the active decisions log.
        archive_path: Path to the permanent decisions archive.
        max_entries: Limit of entries to maintain in the active log.

    Returns:
        0 on success, non-zero error code on failure.
    """
    if not os.path.exists(decisions_path):
        raise FileNotFoundError(f"Decisions log not found: {decisions_path}")

    with open(decisions_path, encoding="utf-8") as f:
        content = f.read()

    # Split by active ADR headers: ## [YYYY-MM-DD]
    # Header format: ## [2026-07-10] [TASK_ID] [ADR_ID]
    parts = re.split(r"(?=^##\s+\[\d{4}-\d{2}-\d{2}\])", content, flags=re.MULTILINE)
    
    if len(parts) <= 1:
        raise ValueError("No valid ADR entries found in decisions.md")

    pre_header = parts[0]
    entries = parts[1:]

    # Remove trailing empty lines/spaces from entries
    entries = [entry.strip() + "\n\n" for entry in entries]

    if len(entries) <= max_entries:
        # No need to archive
        return 0

    active_entries = entries[:max_entries]
    excess_entries = entries[max_entries:]

    # Read existing archive or initialize one
    archive_header = "# Decisions Archive (Permanent Log)\n\n"
    existing_archive_content = ""
    if os.path.exists(archive_path):
        with open(archive_path, encoding="utf-8") as f:
            archive_content = f.read()
        # Separate archive header
        archive_parts = re.split(r"(?=^##\s+\[\d{4}-\d{2}-\d{2}\])", archive_content, flags=re.MULTILINE)
        if len(archive_parts) > 1:
            archive_header = archive_parts[0]
            existing_archive_content = "".join(archive_parts[1:])
        else:
            existing_archive_content = archive_content

    # Extract already archived ADR IDs to prevent duplicate prepend (LIMIT-02)
    # Match pattern: [ADR_YYYYMMDD_...]
    adr_id_pattern = re.compile(r"\[ADR_\w+\]")
    archived_ids = set(adr_id_pattern.findall(existing_archive_content))

    new_archive_entries = []
    for entry in excess_entries:
        entry_ids = adr_id_pattern.findall(entry)
        if not entry_ids or not all(eid in archived_ids for eid in entry_ids):
            new_archive_entries.append(entry)

    # Write back decisions.md with only the active entries
    with open(decisions_path, "w", encoding="utf-8") as f:
        f.write(pre_header.rstrip() + "\n\n" + "".join(active_entries).rstrip() + "\n")

    # Prepend new entries to archive
    if new_archive_entries:
        updated_archive_content = (
            archive_header.rstrip() + "\n\n" +
            "".join(new_archive_entries) +
            existing_archive_content.lstrip()
        )
        with open(archive_path, "w", encoding="utf-8") as f:
            f.write(updated_archive_content.rstrip() + "\n")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Archive decisions sliding window.")
    parser.add_argument("--decisions-path", default="docs/decisions/decisions.md")
    parser.add_argument("--archive-path", default="docs/decisions/decisions_archive.md")
    parser.add_argument("--max-entries", type=int, default=15)
    args = parser.parse_args()

    try:
        return archive_decisions(
            decisions_path=args.decisions_path,
            archive_path=args.archive_path,
            max_entries=args.max_entries,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
