#!/usr/bin/env python3
# ruff: noqa: T201, S607
"""Fast, token-efficient local verification gate: Lint, Type, Tests & 100% Diff-Coverage."""

from __future__ import annotations

import argparse
import ast
import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Final

JsonDiag = dict[str, Any]

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())


def _emit_json(
    status: str,
    phase: str,
    diagnostics: list[JsonDiag],
    coverage: int | None = None,
) -> str:
    return json.dumps(
        {
            "status": status,
            "phase": phase,
            "exit_code": 0 if status == "PASS" else 1,
            "coverage": coverage,
            "diagnostics": diagnostics,
        }
    )


def _exit_with_diags(phase: str, header: str, diags: list[JsonDiag], exit_code: int = 1) -> None:
    print(header)
    for d in diags:
        err = d.get("error", "")
        if err:
            print(f"FAIL | {err}")
    print(_emit_json("FAIL", phase, diags), file=sys.stderr)
    sys.exit(exit_code)


def run_cmd(
    cmd: list[str], timeout: int = 120, *, env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    # Strip unnecessary 'uv run' prefix when already running inside virtualenv
    if len(cmd) >= 3 and cmd[0] == "uv" and cmd[1] == "run" and os.environ.get("VIRTUAL_ENV"):
        cmd = cmd[2:]
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(env_overrides or {})
    with subprocess.Popen(  # noqa: S603
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=env, start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Forked replay workers inherit the pipes and must terminate with the runner.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            return subprocess.CompletedProcess(
                args=cmd, returncode=124, stdout=stdout,
                stderr=f"{stderr}\nError: timed out after {timeout}s.",
            )
        return subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)


def _available_memory_gb() -> float:
    """Return available system RAM in gigabytes using Linux procfs or sysconf."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except (OSError, ValueError):
        pass
    try:
        return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")) / (1024**3)
    except (ValueError, OSError, AttributeError):
        return 8.0


# ---------------------------------------------------------------------------
# Scaffolding Leak Guard: Block temporary spec/recipe text from production code
# ---------------------------------------------------------------------------

_SCAFFOLDING_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\[(?:STEP-BY-STEP\s+)?RECIPE", re.IGNORECASE),
        "Recipe directive leaked into code/docstring",
    ),
    (
        re.compile(r"\[ALGORITHM\s+RECIPE\]", re.IGNORECASE),
        "Algorithm recipe placeholder leaked into code",
    ),
    (
        re.compile(r"^\s*(?:#|/{2})?\s*Step\s+\d+\.\s+[A-Z]", re.MULTILINE),
        "Spec Step-by-step numbering leaked into code/comment",
    ),
    (
        re.compile(r"\b(?:TODO|FIXME)\b\s*:", re.IGNORECASE),
        "TODO/FIXME placeholder found in modified code",
    ),
)


def _check_scaffolding_leaks(py_files: list[str]) -> list[JsonDiag]:
    """Verify that no temporary spec recipes or placeholders remain in production code."""
    diags: list[JsonDiag] = []
    src_files = [f for f in py_files if (f.startswith("src/") or "/src/" in f) and os.path.isfile(f)]

    for fpath in src_files:
        try:
            with open(fpath, encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except OSError:
            continue

        for idx, line in enumerate(lines, start=1):
            for pat, desc in _SCAFFOLDING_PATTERNS:
                if pat.search(line):
                    diags.append(
                        {
                            "file": fpath,
                            "line": idx,
                            "error": f"Scaffolding Leak: {desc} -> '{line.strip()}'",
                            "fix_hint": "Remove temporary spec/recipe directives and write clean production code/docstring",
                        }
                    )
                    break
    return diags


# ---------------------------------------------------------------------------
# Direct Test Matching (Predictable, zero-cascade convention mapping)
# ---------------------------------------------------------------------------


def _find_test_files(py_files: list[str]) -> tuple[list[str], list[str]]:
    """Find direct unit tests corresponding to modified source files.

    Returns ``(mapped, unmapped)``: files explicitly passed through are kept,
    ``src/<pkg>/<mod>.py`` resolves to ``tests/unit/<pkg>/test_<mod>.py`` and
    then to ``tests/unit/<pkg>/<mod>/`` when that package directory exists.
    Source modules without a mapped test are reported as ``unmapped``.
    """
    test_files = [f for f in py_files if f.startswith("tests/") or "test_" in f]
    source_files = [f for f in py_files if f.startswith(("src/", "tools/")) and not f.endswith("__init__.py")]
    unmapped: list[str] = []

    # 1. Direct path convention: src/path/module.py -> tests/unit/path/test_module.py
    #    then tests/unit/path/module/ when that nested package exists.
    for sf in source_files:
        rel = sf[4:] if sf.startswith("src/") else sf
        parts = rel.split("/")
        mod_name = parts[-1]
        test_name = f"test_{mod_name}"
        sub_path = "/".join(parts[:-1])

        candidates = [
            f"tests/unit/{sub_path}/{test_name}" if sub_path else f"tests/unit/{test_name}",
            f"tests/unit/{test_name}",
        ]
        nested_dir = f"tests/unit/{sub_path}/{mod_name[:-3]}" if sub_path and mod_name.endswith(".py") else ""
        found = False
        for cand in candidates:
            if cand in test_files:
                found = True
                break
            if os.path.isfile(cand):
                test_files.append(cand)
                found = True
                break
        if not found and nested_dir and os.path.isdir(nested_dir):
            nested_tests = sorted(
                f"tests/unit/{sub_path}/{mod_name[:-3]}/{p}"
                for p in os.listdir(nested_dir)
                if p.startswith("test_") and p.endswith(".py")
            )
            if nested_tests:
                test_files.extend(t for t in nested_tests if t not in test_files)
                found = True
        if not found:
            unmapped.append(sf)


    return sorted(dict.fromkeys(test_files)), sorted(dict.fromkeys(unmapped))


def _test_node_targets(file: str, changed_lines: set[int] | None) -> list[str]:
    """Narrow body-only integration edits; shared or uncertain changes retain the whole file."""
    if not changed_lines:
        return [file]
    try:
        text = Path(file).read_text(encoding="utf-8")
        tree = ast.parse(text)
    except (OSError, SyntaxError):
        return [file]
    spans: list[tuple[int, int, str]] = []
    for node in tree.body:
        candidates = [(node, node.name)] if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else []
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            candidates = [
                (method, f"{node.name}::{method.name}")
                for method in node.body
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
        for test, name in candidates:
            if test.name.startswith("test_"):
                start = min([test.lineno, *(d.lineno for d in test.decorator_list)])
                spans.append((start, test.end_lineno or test.lineno, name))
    targets: set[str] = set()
    for line in changed_lines:
        enclosing = [name for start, end, name in spans if start <= line <= end]
        if not enclosing:
            return [file]
        targets.update(f"{file}::{name}" for name in enclosing)
    return sorted(targets) or [file]


def _integration_targets(test_files: list[str]) -> list[str]:
    """Select edited integration tests without narrowing explicitly requested files."""
    targets: list[str] = []
    for file in test_files:
        if not file.startswith("tests/integration/"):
            targets.append(file)
            continue
        diff = run_cmd(["git", "diff", "--unified=0", "HEAD", "--", file])
        if diff.returncode != 0:
            targets.append(file)
            continue
        lines: set[int] = set()
        for match in re.finditer(r"^@@.*?\+(\d+)(?:,(\d+))? @@", diff.stdout, re.MULTILINE):
            start = int(match[1])
            count = int(match[2]) if match[2] is not None else 1
            # Deletions can affect either adjacent scope; uncertainty expands coverage.
            lines.update(range(start, start + count) if count else (max(1, start), start + 1))
        targets.extend(_test_node_targets(file, lines))
    return targets


HEAVY_MARKER: Final[str] = "e2e_heavy"
PERIODIC_HEAVY_GATE: Final[str] = "uv run pytest tests/integration -m e2e_heavy -n 8 --dist loadgroup"
PERIODIC_SLOW_GATE: Final[str] = "uv run pytest tests/unit tests/integration -m slow -n 0 -p no:cacheprovider -rfEs"
_HEAVY_TIMEOUT_SECONDS: Final[int] = 3600


def _marker_expression(*, run_slow: bool, include_heavy: bool) -> str:
    """Pytest ``-m`` expression for the verification run.

    Returns:
        ``"slow"`` when ``run_slow`` (the slow and heavy tiers are disjoint);
        ``"not slow"`` when ``include_heavy``; otherwise ``"not slow and not e2e_heavy"``.
    """
    if run_slow:
        return "slow"
    if include_heavy:
        return "not slow"
    return "not slow and not e2e_heavy"


def _is_integration_target(target: str) -> bool:
    """Whether a pytest target (path or node id) lies under a ``tests/integration/`` directory.

    The path part before the first ``::`` is compared in POSIX form: it starts with
    ``tests/integration/`` or contains ``/tests/integration/``.
    """
    path_part = target.split("::", 1)[0].replace("\\", "/")
    return path_part.startswith("tests/integration/") or "/tests/integration/" in path_part


def _collect_deferred_heavy(targets: list[str]) -> list[str]:
    """Node ids among ``targets`` that the default run would skip as ``e2e_heavy``.

    Runs ``<python> -m pytest --collect-only -p no:cacheprovider -p no:warnings
    -m "not slow and e2e_heavy" <targets>`` through ``run_cmd`` only when at least one target
    satisfies ``_is_integration_target`` (heavy marks exist only there); otherwise returns
    ``[]`` without spawning a process. Do NOT pass ``-q``: ``addopts`` already supplies one,
    and ``-qq`` collapses the listing to per-file counts with no node ids (a silent drop).

    Returns:
        Sorted unique stdout lines (whole line, right-stripped; parametrised ids may contain
        spaces) matching ``^\\S+\\.py::`` (node ids relative to rootdir).
        Exit status 5 (nothing selected) yields ``[]``.

    Raises:
        RuntimeError: Any other non-zero exit (collection error, guard ``UsageError``,
            timeout 124); the message carries the first 10 lines of stdout/stderr.
    """
    if not any(_is_integration_target(t) for t in targets):
        return []
    cmd = [
        sys.executable, "-m", "pytest", "--collect-only",
        "-p", "no:cacheprovider", "-p", "no:warnings",
        "-m", "not slow and e2e_heavy", *targets,
    ]
    res = run_cmd(cmd)
    if res.returncode == 0:
        ids: list[str] = []
        for line in (res.stdout or "").splitlines():
            stripped = line.rstrip()
            if re.match(r"^\S+\.py::", stripped):
                ids.append(stripped)
        return sorted(set(ids))
    if res.returncode == 5:
        return []
    combined = ((res.stdout or "") + "\n" + (res.stderr or "")).splitlines()[:10]
    raise RuntimeError(f"heavy-tier collection failed (exit {res.returncode}):\n" + "\n".join(combined))


def _deferral_lines(deferred: list[str]) -> tuple[list[str], list[JsonDiag]]:
    """Human lines and JSON diagnostics announcing deferred heavy tests.

    Returns:
        ``([], [])`` when ``deferred`` is empty; otherwise a header
        ``"DEFERRED | <n> e2e_heavy test(s) not run; include with --run-heavy or run the periodic gate: <PERIODIC_HEAVY_GATE>"``
        followed by one ``"DEFERRED | <nodeid>"`` per id, and one diagnostic per id
        ``{"file": <path part>, "line": 0, "error": "deferred e2e_heavy: <nodeid>",
        "fix_hint": "uv run python tools/verify.py --run-heavy --files <path part>  (or the periodic gate)"}``.
    """
    if not deferred:
        return [], []
    header = (
        f"DEFERRED | {len(deferred)} e2e_heavy test(s) not run; "
        f"include with --run-heavy or run the periodic gate: {PERIODIC_HEAVY_GATE}"
    )
    lines = [header] + [f"DEFERRED | {nodeid}" for nodeid in deferred]
    diags: list[JsonDiag] = []
    for nodeid in deferred:
        path_part = nodeid.split("::", 1)[0]
        diags.append({
            "file": path_part,
            "line": 0,
            "error": f"deferred e2e_heavy: {nodeid}",
            "fix_hint": f"uv run python tools/verify.py --run-heavy --files {path_part}  (or the periodic gate)",
        })
    return lines, diags


def _slow_marked_files(targets: list[str]) -> list[str]:
    """Target files whose module code references ``pytest.mark.slow``.

    Static AST scan (no subprocess, no import). A file is reported when it contains an
    ``Attribute`` chain ``pytest.mark.slow`` anywhere in code (decorators, ``pytestmark``
    assignments, lists); string literals never match. The path part before ``::`` is used;
    missing, unreadable or unparsable files are skipped (pytest reports them itself).

    Returns:
        Sorted unique path parts, as given.
    """
    found: set[str] = set()
    for target in targets:
        path = target.split("::", 1)[0]
        try:
            with open(path, encoding="utf-8") as f:
                tree = ast.parse(f.read())
        except (OSError, SyntaxError, UnicodeError):
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "slow"
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "mark"
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "pytest"
            ):
                found.add(path)
                break
    return sorted(found)


def _slow_deferral_lines(files: list[str]) -> tuple[list[str], list[JsonDiag]]:
    """Human lines and JSON diagnostics announcing slow tests the default run excludes.

    Returns:
        ``([], [])`` when ``files`` is empty; otherwise a header
        ``"DEFERRED | <n> target file(s) carry slow tests not run; include with --run-slow or run the periodic slow gate: <PERIODIC_SLOW_GATE>"``
        followed by one ``"DEFERRED | slow: <path>"`` per file, and one diagnostic per file
        ``{"file": <path>, "line": 0, "error": "deferred slow: <path>",
        "fix_hint": "uv run python tools/verify.py --run-slow --files <path>  (or the periodic slow gate)"}``.
    """
    if not files:
        return [], []
    header = (
        f"DEFERRED | {len(files)} target file(s) carry slow tests not run; "
        f"include with --run-slow or run the periodic slow gate: {PERIODIC_SLOW_GATE}"
    )
    lines = [header] + [f"DEFERRED | slow: {path}" for path in files]
    diags = [
        {
            "file": path,
            "line": 0,
            "error": f"deferred slow: {path}",
            "fix_hint": f"uv run python tools/verify.py --run-slow --files {path}  (or the periodic slow gate)",
        }
        for path in files
    ]
    return lines, diags


# ---------------------------------------------------------------------------
# Diff Coverage Gate
# ---------------------------------------------------------------------------


def _git_diff_added_lines(file: str) -> set[int] | None:
    """1-indexed line numbers this working-tree diff adds to `file`."""
    status_res = subprocess.run(  # noqa: S603
        ["git", "status", "--porcelain", "--", file],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if status_res.stdout.strip().startswith("??"):
        return None
    diff_res = subprocess.run(  # noqa: S603
        ["git", "diff", "--unified=0", "HEAD", "--", file],
        capture_output=True,
        text=True,
        timeout=10,
    )
    added: set[int] = set()
    cur_line = 0
    for line in diff_res.stdout.splitlines():
        if line.startswith("@@"):
            m = re.search(r"\+(\d+)", line)
            if m:
                cur_line = int(m.group(1))
            continue
        if line.startswith("+") and not line.startswith("+++"):
            added.add(cur_line)
            cur_line += 1
        elif not line.startswith("-"):
            cur_line += 1
    return added


def _coverage_args(src_files: list[str], cov_json_path: str) -> list[str]:
    """Build pytest-cov arguments that measure changed sources without importing them.

    Coverage resolves any ``--cov`` value that is not an existing directory as an
    importable package via ``importlib.util.find_spec``; for a dotted submodule that
    imports and then evicts its parent packages, which re-initializes numpy's C
    extension ("cannot load module more than once per process") and executes
    ``src`` import-time side effects before ``tests/conftest.py`` redirects storage
    roots. A file path is likewise treated as a package name and measures nothing.
    Directory sources are matched by path only, so each changed file is measured
    through its parent directory.

    Args:
        src_files: Repo-relative POSIX paths of changed ``src/`` Python files.
        cov_json_path: Destination of the JSON coverage report.
    Returns:
        ``["--cov=<dir>", ..., "--cov-report=json:<cov_json_path>"]`` with one
        entry per distinct parent directory in ascending order, or ``[]`` when
        ``src_files`` is empty.
    """
    if not src_files:
        return []
    dirs = sorted({sf.rpartition("/")[0] for sf in src_files if "/" in sf})
    if not dirs:
        return []
    return [f"--cov={d}" for d in dirs] + [f"--cov-report=json:{cov_json_path}"]


def _check_diff_coverage(
    src_files: list[str], cov_json_path: str, unmapped: list[str] | None = None
) -> tuple[list[JsonDiag], int | None]:
    """Verify that every line added to touched src/ files is executed by tests.

    The report is keyed by cwd-relative POSIX paths because coverage sources are
    directories under the repository root, so each ``src_files`` entry is looked
    up verbatim. A missing or unreadable report, or a mapped file absent from it,
    means the gate measured nothing and fails closed instead of passing.
    Modules without a mapped test are reported as ``unmapped`` diagnostics by the
    caller instead: the mapped-test run cannot be expected to cover them.

    Returns:
        ``(diagnostics, percent)``; ``percent`` is None when no added line was
        measurable.
    """
    try:
        with open(cov_json_path, encoding="utf-8") as f:
            cov_data = json.load(f)
    except (OSError, ValueError):
        return (
            [
                {
                    "file": "",
                    "line": 0,
                    "error": f"coverage report missing or unreadable: {cov_json_path}",
                    "fix_hint": "Re-run verification with coverage enabled and ensure the report is written",
                }
            ],
            None,
        )

    skipped = set(unmapped or [])
    files_data = cov_data.get("files", {})
    diags: list[JsonDiag] = []
    total_added = 0
    total_covered = 0
    for sf in src_files:
        if sf in skipped:
            continue
        entry = files_data.get(sf)
        if not entry:
            diags.append(
                {
                    "file": sf,
                    "line": 0,
                    "error": f"no coverage data for {sf}",
                    "fix_hint": "Add or update tests exercising these lines",
                }
            )
            continue
        missing = set(entry.get("missing_lines", []))
        executed = set(entry.get("executed_lines", []))
        added = _git_diff_added_lines(sf)
        if added is None:
            added = executed | missing
        added &= executed | missing
        if not added:
            continue
        total_added += len(added)
        total_covered += len(added - missing)
        uncovered_new = sorted(added & missing)
        if uncovered_new:
            shown = uncovered_new[:10]
            diags.append(
                {
                    "file": sf,
                    "line": shown[0],
                    "error": f"{len(uncovered_new)} newly-added line(s) not executed by tests: {shown}",
                    "fix_hint": "Add or update tests exercising these lines",
                }
            )
    pct = round(100 * total_covered / total_added) if total_added else None
    return diags, pct


# ---------------------------------------------------------------------------
# Main CLI Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    started = time.monotonic()
    parser = argparse.ArgumentParser(description="Fast, token-efficient local verification gate.")
    parser.add_argument("--files", nargs="*", default=[], help="Explicit files to check")
    parser.add_argument("--fast", action="store_true", help="Run static checks only (scaffolding, ruff, mypy)")
    parser.add_argument("--skip-lint", action="store_true", help="Skip Ruff linting")
    parser.add_argument("--skip-mypy", action="store_true", help="Skip Mypy static check")
    parser.add_argument("--no-cov", action="store_true", help="Disable diff-coverage gate")
    parser.add_argument("--no-xdist", action="store_true", help="Force serial pytest execution (-n 0)")
    parser.add_argument("--timeout", type=int, default=None, help="Pytest timeout in seconds")
    parser.add_argument(
        "--run-slow",
        action="store_true",
        help="Run tests marked slow (real-data suite) instead of excluding them",
    )
    parser.add_argument(
        "--run-heavy",
        action="store_true",
        help="Include e2e_heavy tests instead of deferring and listing them",
    )
    parser.add_argument(
        "--pre-impl",
        action="store_true",
        help="Validate spec blueprint paths and wiring anchors before implementation",
    )
    args = parser.parse_args()


    # 1. File discovery from git if not explicitly passed
    auto_discovered = not args.files
    if auto_discovered:
        try:
            diff_res = subprocess.run(
                ["git", "status", "--porcelain", "-uall"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            git_files = [
                line[3:].strip()
                for line in diff_res.stdout.splitlines()
                if "D" not in line[:2]
                and line[3:].strip().endswith(".py")
                and not line[3:].strip().endswith("conftest.py")
                and os.path.exists(line[3:].strip())
            ]
            args.files = git_files
        except Exception:
            args.files = []

    py_files = [f for f in args.files if f.endswith(".py")]
    if not py_files:
        print("ALLCHECKS:PASS | No modified .py files detected")
        sys.exit(0)

    # 2. Scaffolding Leak Guard
    scaffolding_diags = _check_scaffolding_leaks(py_files)
    if scaffolding_diags:
        _exit_with_diags(
            "scaffolding-guard",
            f"FAIL | Scaffolding Leak: {len(scaffolding_diags)} temporary spec/recipe artifact(s) found in code",
            scaffolding_diags,
        )

    # 3. Parallel Static Checks (Ruff, Mypy)
    def check_ruff() -> tuple[str, int, list[JsonDiag], str]:
        if args.skip_lint or not py_files:
            return "ruff", 0, [], ""
        res = run_cmd(["uv", "run", "ruff", "check", *py_files, "--quiet"])
        if res.returncode != 0:
            out = "\n".join((res.stdout or res.stderr).strip().splitlines()[:10])
            return "ruff", 1, [{"file": py_files[0], "line": 0, "error": out, "fix_hint": "Fix ruff lint errors"}], "FAIL | Ruff Lint Failed"
        return "ruff", 0, [], ""

    def check_mypy() -> tuple[str, int, list[JsonDiag], str]:
        if args.skip_mypy or not py_files:
            return "mypy", 0, [], ""
        target_mypy = [f for f in py_files if f.startswith(("src/", "tools/"))] or py_files
        res = run_cmd(["uv", "run", "mypy", *target_mypy, "--ignore-missing-imports"])
        if res.returncode != 0:
            out = "\n".join((res.stdout or res.stderr).strip().splitlines()[:10])
            return "mypy", 1, [{"file": target_mypy[0], "line": 0, "error": out, "fix_hint": "Fix mypy type errors"}], "FAIL | Mypy Type Check Failed"
        return "mypy", 0, [], ""

    # 3. Sequential Static Checks (Ruff Fail-Fast, then Mypy)
    for check_fn in (check_ruff, check_mypy):
        phase, code, diags, msg = check_fn()
        if code != 0:
            _exit_with_diags(phase, msg, diags)

    if args.fast:
        print("PASS | Fast Check Passed (Scaffolding, Ruff, Mypy verified)")
        print(_emit_json("PASS", "fast-check", [], None), file=sys.stderr)
        return

    # 4. Direct Test Discovery
    test_files, unmapped = _find_test_files(py_files)
    if auto_discovered:
        test_files = _integration_targets(test_files)
    if not test_files:
        if unmapped:
            diags = [
                {"file": m, "line": 0, "error": f"unmapped: no test covers {m}", "fix_hint": "Add tests/unit coverage for this module"}
                for m in unmapped
            ]
            print(_emit_json("PASS", "all", diags, None), file=sys.stderr)
            print(f"PASS | Lint & Type check passed ({len(unmapped)} unmapped module(s) reported)")
            return
        print("PASS | Lint & Type check passed (no tests to run)")
        print(_emit_json("PASS", "all", [], None), file=sys.stderr)
        return

    deferred: list[str] = []
    if not (args.run_heavy or args.run_slow):
        try:
            deferred = _collect_deferred_heavy(test_files)
        except RuntimeError as exc:
            _exit_with_diags("pytest-collect", "FAIL | Heavy-tier collection failed",
                             [{"file": "", "line": 0, "error": str(exc), "fix_hint": "Fix collection errors or tier marks"}])
    deferral_lines, deferral_diags = _deferral_lines(deferred)
    slow_files: list[str] = []
    if not args.run_slow:
        slow_files = _slow_marked_files(test_files)
        slow_lines, slow_diags = _slow_deferral_lines(slow_files)
        deferral_lines += slow_lines
        deferral_diags += slow_diags

    # 5. Smart Pytest Execution (Resource Safety Guard)
    # 5. Smart Pytest Execution (Resource Safety Guard: Serial Execution Default)
    # 다중 프로젝트 및 로컬 동시성 환경 안정성을 위해 기본값은 항상 단일 프로세스(-n 0)로 고정.
    # CI 등에서 명시적으로 LEAN_CHECK_WORKERS 환경변수가 2 이상으로 지정된 경우에만 제한적 병렬 허용.
    env_workers = os.environ.get("LEAN_CHECK_WORKERS")
    avail_mem_gb = _available_memory_gb()

    if (
        args.no_xdist
        or not env_workers
        or not env_workers.isdigit()
        or int(env_workers) <= 1
        or avail_mem_gb < 2.0
    ):
        xdist_args = ["-p", "no:cacheprovider", "-n", "0"]
    else:
        target_workers = int(env_workers)
        worker_count = min(target_workers, os.cpu_count() or 2, len(test_files))
        xdist_args = ["-p", "no:cacheprovider", "-n", str(worker_count)]
        if worker_count > 1:
            xdist_args += ["--dist", "loadgroup"]

    src_files = [f for f in py_files if f.startswith("src/")]
    Path("scratch").mkdir(exist_ok=True)
    workspace = tempfile.TemporaryDirectory(prefix="verify_", dir="scratch")
    cov_json_path = str(Path(workspace.name) / "coverage.json")
    cov_args: list[str] = []

    if src_files and not args.no_cov:
        cov_args = _coverage_args(src_files, cov_json_path)

    pytest_cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-m",
        _marker_expression(run_slow=args.run_slow, include_heavy=args.run_heavy),
        *test_files,
        *xdist_args,
        *cov_args,
        "-vv",
        "--tb=line",
    ]
    if args.timeout is not None:
        pytest_timeout = args.timeout
    elif args.run_heavy:
        pytest_timeout = _HEAVY_TIMEOUT_SECONDS
    else:
        pytest_timeout = max(60, min(240, 20 * len(test_files)))
    with workspace:
        pt_res = run_cmd(
            pytest_cmd, timeout=pytest_timeout,
            env_overrides={"COVERAGE_FILE": str(Path(workspace.name) / ".coverage")},
        )
        cov_diags, cov_pct = (
            _check_diff_coverage(src_files, cov_json_path, unmapped)
            if pt_res.returncode == 0 and cov_args else ([], None)
        )

    for line in deferral_lines:
        print(line)

    if pt_res.returncode == 124:
        active_tests = [line.strip() for line in pt_res.stdout.splitlines() if line.startswith("tests/")]
        active_test = active_tests[-1] if active_tests else "unknown test"
        _exit_with_diags(
            "pytest-timeout",
            f"FAIL | Pytest Timed Out ({pytest_timeout}s)",
            [{
                "file": "",
                "line": 0,
                "error": f"pytest timed out after {pytest_timeout}s across {len(test_files)} target(s); last: {active_test}",
                "fix_hint": "Use --files to scope checks, investigate slow tests, or pass --timeout with a larger value.",
            }, *deferral_diags],
        )

    if pt_res.returncode == 5 and (deferred or slow_files):
        unmapped_diags = [
            {"file": m, "line": 0, "error": f"unmapped: no test covers {m}", "fix_hint": "Add tests/unit coverage for this module"}
            for m in unmapped
        ]
        deferred_suffix = f", {len(deferred)} deferred e2e_heavy" if deferred else ""
        slow_suffix = f", {len(slow_files)} deferred slow file(s)" if slow_files else ""
        unmapped_suffix = f", {len(unmapped_diags)} unmapped" if unmapped_diags else ""
        elapsed = time.monotonic() - started
        print(f"PASS | All checks passed (Scaffolding-Clean, Lint, Type, Tests 0 passed{unmapped_suffix}{deferred_suffix}{slow_suffix}) in {elapsed:.2f}s")
        print(_emit_json("PASS", "all", unmapped_diags + deferral_diags, None), file=sys.stderr)
        return

    if pt_res.returncode == 0:
        if cov_diags:
            _exit_with_diags(
                "coverage",
                f"FAIL | Diff Coverage: {len(cov_diags)} file(s) with untested new lines",
                cov_diags + deferral_diags,
            )
        unmapped_diags = [
            {"file": m, "line": 0, "error": f"unmapped: no test covers {m}", "fix_hint": "Add tests/unit coverage for this module"}
            for m in unmapped
        ]
        cov_suffix = f", Diff-Coverage {cov_pct}%" if cov_pct is not None else ""
        unmapped_suffix = f", {len(unmapped_diags)} unmapped" if unmapped_diags else ""
        deferred_suffix = f", {len(deferred)} deferred e2e_heavy" if deferred else ""
        deferred_suffix += f", {len(slow_files)} deferred slow file(s)" if slow_files else ""
        passed = re.search(r"\b(\d+) passed\b", pt_res.stdout)
        test_summary = f"Tests {passed[1]} passed" if passed else "Tests"
        elapsed = time.monotonic() - started
        print(f"PASS | All checks passed (Scaffolding-Clean, Lint, Type, {test_summary}{cov_suffix}{unmapped_suffix}{deferred_suffix}) in {elapsed:.2f}s")
        print(_emit_json("PASS", "all", unmapped_diags + deferral_diags, cov_pct), file=sys.stderr)
    else:
        last_err = [
            line
            for line in (pt_res.stdout or "").splitlines()
            if any(x in line for x in ("FAIL", "Error", "AssertionError"))
        ]
        cause = last_err[-1] if last_err else (pt_res.stderr or "Check pytest output.").strip()
        cause_sliced = "\n".join(cause.splitlines()[:10])
        _exit_with_diags(
            "pytest",
            f"FAIL | Pytest Failed: {cause_sliced}",
            [{"file": "", "line": 0, "error": cause_sliced, "fix_hint": "Fix failing pytest assertions"}, *deferral_diags],
        )


if __name__ == "__main__":
    main()
