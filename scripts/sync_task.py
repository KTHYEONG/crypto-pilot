#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime


def _read_file(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _write_file(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _path_exists(path: str) -> bool:
    return os.path.exists(path)


def _resolve_test_path(source_file: str) -> str | None:
    if not source_file.startswith("src/") or source_file.endswith("__init__.py"):
        return None
    parts = source_file.split("/")
    module_name = parts[-1]
    test_name = f"test_{module_name}"
    for category in ["unit", "integration", "e2e"]:
        sub = "/".join(parts[1:-1])
        test_dir = f"tests/{category}/{sub}" if sub else f"tests/{category}"
        tp = f"{test_dir}/{test_name}"
        if os.path.exists(tp):
            return tp
    return None


def _append_adr(task: str, title: str, why: str, what: str, impact: str) -> str:
    date_str = datetime.now().strftime("%Y-%m-%d")
    adr_date = datetime.now().strftime("%Y%m%d")
    adr_id = f"ADR_{adr_date}_{task.replace('TASK_', '')}"

    decisions_path = "docs/decisions/decisions.md"
    if not _path_exists(decisions_path):
        return f"ERROR: {decisions_path} not found"

    new_entry = (
        f"## [{date_str}] [{task}] [{adr_id}]\n"
        f"- **Context/Why:** {why}\n"
        f"- **Resolution/What:** {what}\n"
        f"- **Impact:** {impact}\n\n"
    )

    content = _read_file(decisions_path)
    header_re = re.compile(
        r"^#\s*Active\s*Decisions\s*Log\s*\(\s*Sliding\s*Window\s*\)",
        re.IGNORECASE | re.MULTILINE,
    )
    m = header_re.search(content)
    if m:
        end = m.end()
        nl = content.find("\n", end)
        if nl == -1:
            nl = end
        before = content[: nl + 1]
        after = content[nl + 1 :].lstrip()
        updated = f"{before}\n{new_entry}{after}"
    else:
        updated = f"{new_entry}{content}"

    _write_file(decisions_path, updated)
    return adr_id


def _prune_archive(max_entries: int = 15) -> tuple[int, int]:
    decisions_path = "docs/decisions/decisions.md"
    archive_path = "docs/decisions/decisions_archive.md"

    if not _path_exists(decisions_path):
        return (0, 0)

    content = _read_file(decisions_path)
    parts = re.split(r"(?=^##\s+\[\d{4}-\d{2}-\d{2}\])", content, flags=re.MULTILINE)
    if len(parts) <= 1:
        return (0, 0)

    pre_header = parts[0]
    entries = [e.strip() + "\n\n" for e in parts[1:]]

    if len(entries) <= max_entries:
        return (0, 0)

    active = entries[:max_entries]
    excess = entries[max_entries:]

    _write_file(decisions_path, pre_header.rstrip() + "\n\n" + "".join(active).rstrip() + "\n")

    archive_header = "# Decisions Archive (Permanent Log)\n\n"
    existing = ""
    if _path_exists(archive_path):
        existing = _read_file(archive_path)
        ap = re.split(r"(?=^##\s+\[\d{4}-\d{2}-\d{2}\])", existing, flags=re.MULTILINE)
        if len(ap) > 1:
            archive_header = ap[0]
            existing = "".join(ap[1:])

    archived_ids = set(re.findall(r"\[ADR_\w+\]", existing))
    new_entries = [e for e in excess if not all(
        eid in archived_ids for eid in re.findall(r"\[ADR_\w+\]", e)
    )]

    if new_entries:
        updated = archive_header.rstrip() + "\n\n" + "".join(new_entries) + existing.lstrip()
        _write_file(archive_path, updated.rstrip() + "\n")

    return (len(excess), len(new_entries))


def _update_index(source_file: str, test_file: str | None, doc_file: str | None) -> None:
    index_path = "docs/index.json"
    data: dict[str, dict[str, str | list[str]]] = {}

    if _path_exists(index_path):
        try:
            data = json.loads(_read_file(index_path))
            if not isinstance(data, dict):
                data = {}
        except (json.JSONDecodeError, ValueError):
            shutil.copyfile(index_path, f"{index_path}.bak")
            data = {}

    if source_file not in data:
        data[source_file] = {}

    entry = data[source_file]
    if doc_file is not None:
        entry["architecture"] = doc_file
    if test_file is not None:
        cur = entry.get("testing")
        if cur is None:
            entry["testing"] = test_file
        elif isinstance(cur, str) and cur != test_file:
            entry["testing"] = [cur, test_file]
        elif isinstance(cur, list) and test_file not in cur:
            cur.append(test_file)

    _write_file(index_path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


EXCLUDED_DIRS = frozenset({".git", ".venv", ".mypy_cache", ".ruff_cache", "__pycache__", "node_modules"})


def _wipe_temp_artifacts() -> int:
    count = 0
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        for f in files:
            if f.endswith((".tmp", ".bak")):
                fpath = os.path.join(root, f)
                try:
                    os.remove(fpath)
                    count += 1
                except OSError:
                    pass
    return count


def _clean_specs(task: str) -> int:
    specs_dir = "docs/specs"
    if not _path_exists(specs_dir):
        return 0
    count = 0
    for fname in os.listdir(specs_dir):
        fpath = os.path.join(specs_dir, fname)
        try:
            content = _read_file(fpath)
            if task not in content:
                continue
            os.remove(fpath)
            count += 1
            json_path = fpath.replace(".md", "_contract.json")
            if os.path.exists(json_path):
                os.remove(json_path)
                count += 1
        except OSError:
            pass
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync: ADR + Index + Cleanup in one command.")
    parser.add_argument("--task", required=True, help="Task ID (e.g. TASK_L0_MTF_FUSION)")
    parser.add_argument("--title", required=True, help="Decision title")
    parser.add_argument("--why", required=True, help="Context/Why")
    parser.add_argument("--what", required=True, help="Resolution/What")
    parser.add_argument("--impact", required=True, help="Impact")
    parser.add_argument("--source", required=True, help="Modified source file path")
    parser.add_argument("--test", default=None, help="Test file path (auto-resolved if omitted)")
    parser.add_argument("--doc", default=None, help="Architecture doc path")
    args = parser.parse_args()

    logs: list[str] = []
    errors: list[str] = []

    # 1. Read context (top 3 decisions)
    decisions_path = "docs/decisions/decisions.md"
    if _path_exists(decisions_path):
        content = _read_file(decisions_path)
        entries = re.split(r"(?=^##\s+\[\d{4}-\d{2}-\d{2}\])", content, flags=re.MULTILINE)
        recent = [e.strip() for e in entries[-3:]] if len(entries) > 1 else []
        if recent:
            print("# Recent Decisions (context)")
            for e in recent:
                print(e[:200] + ("..." if len(e) > 200 else ""))
            print()

    # 2. ADR Append
    try:
        adr_id = _append_adr(args.task, args.title, args.why, args.what, args.impact)
        logs.append(f"ADR added: {adr_id}")
    except Exception as e:
        errors.append(f"ADR append failed: {e}")
        adr_id = "N/A"

    # 3. Archive Pruning
    try:
        pruned, archived = _prune_archive(15)
        if pruned > 0:
            logs.append(f"Pruned {pruned} entries ({archived} new to archive)")
        else:
            logs.append("No pruning needed")
    except Exception as e:
        errors.append(f"Archive prune failed: {e}")

    # 4. Index Update
    test_file = args.test or _resolve_test_path(args.source)
    try:
        _update_index(args.source, test_file, args.doc)
        logs.append(f"Index updated for {args.source}")
    except Exception as e:
        errors.append(f"Index update failed: {e}")

    # 5. Temp Artifact Wipe (.tmp, .bak, .pyc)
    try:
        wiped = _wipe_temp_artifacts()
        if wiped > 0:
            logs.append(f"Wiped {wiped} temp artifacts")
    except Exception as e:
        errors.append(f"Temp wipe failed: {e}")

    # 6. Spec Cleanup
    try:
        cleaned = _clean_specs(args.task)
        if cleaned > 0:
            logs.append(f"Cleaned {cleaned} spec files")
        else:
            logs.append("No spec files to clean")
    except Exception as e:
        errors.append(f"Spec cleanup failed: {e}")

    # 7. Summary
    status = "OK" if not errors else "PARTIAL"
    summary = f"### 🏁 [SYNC:{status}] [{adr_id}] | {' | '.join(logs)}"
    if errors:
        summary += f" | ERRORS: {'; '.join(errors)}"

    print(summary)
    if errors:
        print(json.dumps({"status": "PARTIAL", "adr_id": adr_id, "logs": logs, "errors": errors}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
