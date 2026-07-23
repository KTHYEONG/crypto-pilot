#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from typing import Any

JsonDiag = dict[str, Any]


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


def run_cmd(cmd: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(  # noqa: S603
            cmd, capture_output=True, text=True, shell=False, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=124,
            stdout="",
            stderr=f"Error: timed out after {timeout}s.",
        )


def _fail_exit(phase: str, msg: str, diag: JsonDiag) -> None:
    print(msg)
    print(_emit_json("FAIL", phase, [diag]), file=sys.stderr)
    sys.exit(1)


def _find_test_files(py_files: list[str]) -> list[str]:
    test_files = [f for f in py_files if f.startswith("tests/") or "test_" in f]
    source_files = [f for f in py_files if not (f.startswith("tests/") or "test_" in f)]
    for sf in source_files:
        if sf.startswith("src/") and not sf.endswith("__init__.py"):
            parts = sf.split("/")
            module_name = parts[-1]
            test_name = f"test_{module_name}"
            for category in ["unit", "integration", "e2e"]:
                sub_path = "/".join(parts[1:-1])
                td = f"tests/{category}/{sub_path}" if sub_path else f"tests/{category}"
                tp = f"{td}/{test_name}"
                if os.path.exists(tp) and tp not in test_files:
                    test_files.append(tp)
                    break
    return test_files


def _get_source_files(py_files: list[str]) -> list[str]:
    return [f for f in py_files if not (f.startswith("tests/") or "test_" in f)]


def _get_target_coverage(file_path: str) -> int:
    # Tier 1 (Core): domain, signal, sizing, portfolio
    if any(k in file_path for k in ("domain/", "signal/", "sizing/", "portfolio/")):
        return 85
    # Tier 2 (Adapter/Runner): adapters, runners, repository, services, db
    if any(k in file_path for k in ("adapters/", "runners/", "repository/", "services/", "db/")):
        return 65
    return 50  # Fallback target for other source paths


def _parse_file_coverage(stdout: str, file_path: str) -> int | None:
    mkey = file_path.replace(".py", "")
    module_key = mkey.replace("/", ".")
    for line in stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        first_token = parts[0]
        if first_token in (file_path, mkey, module_key):
            for part in parts[1:]:
                if "%" in part:
                    try:
                        return int(part.replace("%", ""))
                    except ValueError:
                        pass
    return None


def _get_changed_lines(file_path: str) -> set[int]:
    """Return line numbers added/modified by git diff (uncommitted changes)."""
    try:
        res = subprocess.run(  # noqa: S603
            ["git", "diff", "--unified=0", "HEAD", "--", file_path],  # noqa: S607
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        return set()
    changed: set[int] = set()
    for line in res.stdout.splitlines():
        if line.startswith("@@"):
            m = re.search(r"\+(\d+)(?:,(\d+))?", line)
            if m:
                start = int(m.group(1))
                count = int(m.group(2)) if m.group(2) else 1
                for i in range(start, start + count):
                    changed.add(i)
    return changed


def _is_new_file(file_path: str) -> bool:
    """Return True if the file is new (does not exist in Git HEAD)."""
    try:
        res = subprocess.run(  # noqa: S603
            ["git", "cat-file", "-e", f"HEAD:{file_path}"],  # noqa: S607
            capture_output=True, text=True, timeout=10
        )
        return res.returncode != 0
    except Exception:
        return False


def _get_missing_lines(stdout: str, file_path: str) -> set[int]:
    """Parse the Missing column from coverage term-missing output."""
    mkey = file_path.replace(".py", "")
    module_key = mkey.replace("/", ".")
    for line in stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        first_token = parts[0]
        if first_token in (file_path, mkey, module_key) and len(parts) >= 5:
            missing_raw = parts[-1]
            missing: set[int] = set()
            for token in missing_raw.split(","):
                token = token.strip()
                if not token:
                    continue
                try:
                    if "-" in token:
                        a, b = token.split("-", 1)
                        for i in range(int(a.strip()), int(b.strip()) + 1):
                            missing.add(i)
                    else:
                        missing.add(int(token))
                except ValueError:
                    continue
            return missing
    return set()


def _is_stub_node(node: ast.AST) -> bool:
    """Check if an AST node (function/method body) is a stub implementation."""
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    body = node.body
    # Filter out docstrings and logger calls
    filtered_body = [
        stmt for stmt in body
        if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str))
        and not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) and getattr(getattr(stmt.value, "func", None), "attr", "") in ("debug", "info", "warning", "error", "critical"))
    ]
    if not filtered_body:
        return True
    if len(filtered_body) == 1:
        single = filtered_body[0]
        if isinstance(single, ast.Pass):
            return True
        if isinstance(single, ast.Expr) and isinstance(single.value, ast.Constant) and single.value.value == Ellipsis:
            return True
        if isinstance(single, ast.Raise):
            if isinstance(single.exc, ast.Call) and getattr(single.exc.func, "id", None) == "NotImplementedError":
                return True
            if isinstance(single.exc, ast.Name) and single.exc.id == "NotImplementedError":
                return True
        # Check dummy single return (e.g. return None, return {}, return [], return "", return True, return False)
        if isinstance(single, ast.Return):
            if single.value is None:
                return True
            if isinstance(single.value, ast.Constant) and (single.value.value in (None, "", 0, False, True) or isinstance(single.value.value, (int, float, str))):
                return True
            if isinstance(single.value, (ast.List, ast.Dict, ast.Tuple, ast.Set)) and not getattr(single.value, "elts", getattr(single.value, "keys", None)):
                return True
    return False


def _check_spec_compliance(spec_path: str) -> tuple[int, list[JsonDiag]]:
    diagnostics: list[JsonDiag] = []
    try:
        with open(spec_path) as f:
            contract = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return (1, [{"file": spec_path, "line": 0, "error": f"Spec file error: {e}", "fix_hint": ""}])

    for c in contract.get("contracts", []):
        fh: str = c.get("file_hint", "")
        kind: str = c.get("kind", "function")
        name: str = c.get("name", "")
        assertions: list[Any] = c.get("assertions", [])
        if not fh or not name:
            continue
        if not os.path.exists(fh):
            d = {"file": fh, "line": 0, "error": f"Spec: file not found ({kind} {name})", "fix_hint": f"Create {fh}"}
            diagnostics.append(d)
            continue
        
        # Check assertions requirement
        if not assertions:
            msg = f"Spec contract '{name}' in {fh} must define at least one exact input/output assertion in 'assertions'"
            d = {"file": fh, "line": 0, "error": msg, "fix_hint": "Add 'assertions' array in contract.json"}
            diagnostics.append(d)

        with open(fh) as sf:
            sf_content = sf.read()
            if kind == "field":
                field_name = name.split(".")[-1] if "." in name else name
                pat = rf"\b{re.escape(field_name)}[\"']?\s*(?::|=)"
                if not re.search(pat, sf_content, re.MULTILINE):
                    msg = f"Spec: {kind} '{name}' not implemented"
                    d = {"file": fh, "line": 0, "error": msg, "fix_hint": f"Implement {kind} {name} in {fh}"}
                    diagnostics.append(d)
            else:
                pat = rf"^(?:class|def)\s+{re.escape(name)}\b"
                if not re.search(pat, sf_content, re.MULTILINE):
                    msg = f"Spec: {kind} '{name}' not implemented"
                    d = {"file": fh, "line": 0, "error": msg, "fix_hint": f"Implement {kind} {name} in {fh}"}
                    diagnostics.append(d)
                else:
                    # AST Stub Check
                    try:
                        tree = ast.parse(sf_content, filename=fh)
                        for node in ast.walk(tree):
                            if (
                                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                                and node.name == name
                                and _is_stub_node(node)
                            ):
                                msg = f"Spec: {kind} '{name}' is a stub or dummy implementation (pass / Ellipsis / NotImplementedError / logger+dummy return)"
                                d = {"file": fh, "line": node.lineno, "error": msg, "fix_hint": f"Implement real logic in {name}"}
                                diagnostics.append(d)
                    except Exception:  # noqa: S110
                        pass

    for s in contract.get("scenarios", []):
        test_name: str = s.get("name", "")
        if not test_name:
            continue
        found = False
        for root, _dirs, fnames in os.walk("tests"):
            for fn in fnames:
                if not fn.endswith(".py"):
                    continue
                with open(os.path.join(root, fn)) as tf:
                    if re.search(rf"^[ \t]*def\s+{re.escape(test_name)}\b", tf.read(), re.MULTILINE):
                        found = True
                        break
            if found:
                break
        if not found:
            d = {"file": "", "line": 0, "error": f"Spec: missing test '{test_name}'", "fix_hint": f"Write {test_name}"}
            diagnostics.append(d)

    wirings = contract.get("wiring", [])
    if not wirings:
        msg = "Spec: contract.json missing mandatory 'wiring' section defining pipeline integration"
        diagnostics.append({"file": spec_path, "line": 0, "error": msg, "fix_hint": "Add 'wiring' array to contract.json"})

    for w in wirings:
        wf: str = w.get("file", "")
        anchor: str = w.get("anchor", "")
        import_symbol: str = w.get("import_symbol", "")
        invocation_symbol: str = w.get("invocation_symbol", "")
        invocation_regex: str = w.get("invocation_regex", "")
        if not wf:
            continue
        if not os.path.exists(wf):
            d = {"file": wf, "line": 0, "error": f"Spec wiring target file not found: {wf}", "fix_hint": f"Create {wf}"}
            diagnostics.append(d)
            continue
        with open(wf) as f:
            wf_content = f.read()
            if anchor and anchor not in wf_content:
                hint = f"Add ref to {anchor} in {wf}"
                d = {"file": wf, "line": 0, "error": f"Spec wiring: missing anchor '{anchor}'", "fix_hint": hint}
                diagnostics.append(d)
            if import_symbol and import_symbol not in wf_content:
                hint = f"Import or reference '{import_symbol}' in {wf}"
                d = {"file": wf, "line": 0, "error": f"Spec wiring: missing reference to '{import_symbol}' in {wf}", "fix_hint": hint}
                diagnostics.append(d)
            if invocation_symbol and invocation_symbol not in wf_content:
                hint = f"Invoke or instantiate '{invocation_symbol}' in {wf}"
                d = {"file": wf, "line": 0, "error": f"Spec wiring: missing invocation of '{invocation_symbol}' in {wf}", "fix_hint": hint}
                diagnostics.append(d)
            if invocation_regex and not re.search(invocation_regex, wf_content, re.MULTILINE):
                hint = f"Invoke matching pattern '{invocation_regex}' in {wf}"
                d = {"file": wf, "line": 0, "error": f"Spec wiring: missing pattern match '{invocation_regex}' in {wf}", "fix_hint": hint}
                diagnostics.append(d)

    return (1 if diagnostics else 0, diagnostics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lean Check with JSON diagnostics.")
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--spec", default=None, help="Path to spec contract JSON for compliance verification")
    parser.add_argument("--skip-lint", action="store_true", help="Skip Ruff linting")
    parser.add_argument("--skip-mypy", action="store_true", help="Skip Mypy static check")
    args = parser.parse_args()

    py_files = [f for f in args.files if f.endswith(".py")]
    if not py_files and not args.spec:
        print("ALLCHECKS:PASS")
        sys.exit(0)

    # 0. Spec Compliance (optional, runs first — most fundamental)
    if args.spec:
        ec, diags = _check_spec_compliance(args.spec)
        if ec != 0:
            for d in diags:
                print(f"FAIL | {d.get('error', '')}")
            print(_emit_json("FAIL", "spec-compliance", diags), file=sys.stderr)
            sys.exit(1)
        print("PASS | Spec compliance verified")

    # 1. Co-modification Mapping Verification (scripts/ excluded)
    for pf in py_files:
        if not pf.startswith("src/") or pf.endswith("__init__.py") or pf.startswith("scripts/"):
            continue
        parts = pf.split("/")
        module_name = parts[-1]
        test_name = f"test_{module_name}"
        found = any(
            os.path.exists(
                f"tests/{cat}/{'/'.join(parts[1:-1])}/{test_name}" if parts[1:-1] else f"tests/{cat}/{test_name}"
            )
            for cat in ["unit", "integration", "e2e"]
        )
        if not found:
            d = {"file": pf, "line": 0, "error": f"No matching test for {pf}", "fix_hint": ""}
            _fail_exit("co-modification", f"FAIL | {pf}: test file missing", d)

    # 2. print() Detection (scripts/ excluded)
    if not args.skip_lint:
        print_re = re.compile(r"(?<!#)\bprint\s*\(")
        for pf in py_files:
            if pf.startswith("scripts/"):
                continue
            with open(pf, encoding="utf-8") as f:
                for idx, line in enumerate(f, 1):
                    if print_re.search(line):
                        d = {"file": pf, "line": idx, "error": "Unsanctioned print()", "fix_hint": ""}
                        _fail_exit("print-check", f"FAIL | {pf}:{idx} print() detected", d)

    # 3. Ruff
    if not args.skip_lint:
        ruff_res = run_cmd(["uv", "run", "ruff", "check", *py_files, "--quiet"])
        if ruff_res.returncode != 0:
            out = (ruff_res.stdout or ruff_res.stderr).strip()
            # Slice error output to max 10 lines for token efficiency
            out_sliced = "\n".join(out.splitlines()[:10])
            d = {"file": py_files[0] if py_files else "", "line": 0, "error": out_sliced, "fix_hint": "Resolve ruff errors"}
            _fail_exit("ruff", "FAIL | Ruff Lint Failed", d)

    # 4. Mypy
    if not args.skip_mypy:
        mypy_res = run_cmd(["uv", "run", "mypy", *py_files, "--ignore-missing-imports"])
        if mypy_res.returncode != 0:
            out = (mypy_res.stdout or mypy_res.stderr).strip()
            # Slice error output to max 10 lines for token efficiency
            out_sliced = "\n".join(out.splitlines()[:10])
            d = {"file": py_files[0] if py_files else "", "line": 0, "error": out_sliced, "fix_hint": "Resolve type errors"}
            _fail_exit("mypy", "FAIL | Mypy Type Check Failed", d)

    # 5. Single pytest with coverage
    test_files = _find_test_files(py_files)
    source_files = _get_source_files(py_files)

    if not test_files:
        print("PASS | Lint & Type check passed (no tests to run)")
        print(_emit_json("PASS", "all", [], None), file=sys.stderr)
        return

    cov_args = (
        [f"--cov={sf.replace('.py', '').replace('/', '.')}" for sf in source_files] if source_files else ["--cov=src"]
    )

    core_cmd = ["uv", "run", "pytest", *cov_args, *test_files, "-q", "--tb=line", "--cov-report=term-missing"]
    pt_res = run_cmd(core_cmd, timeout=180)

    cov_val: int | None = None
    missing_infos: list[str] = []

    if pt_res.returncode == 0:
        for line in pt_res.stdout.splitlines():
            if "TOTAL" in line:
                m = re.search(r"(\d+)%", line)
                if m:
                    cov_val = int(m.group(1))

        # Check coverage targets per file
        coverage_violations: list[JsonDiag] = []
        for sf in source_files:
            target_cov = _get_target_coverage(sf)
            if target_cov == 0:
                continue

            actual_cov = _parse_file_coverage(pt_res.stdout, sf)
            if actual_cov is None:
                actual_cov = 0  # Fail-safe: if file not executed at all, count as 0%

            is_new = _is_new_file(sf)
            if is_new:
                # New files must meet the target coverage globally
                if actual_cov < target_cov:
                    d = {
                        "file": sf,
                        "line": 0,
                        "error": f"Coverage target violation (New File): actual {actual_cov}% < target {target_cov}%",
                        "fix_hint": f"Add test cases to cover the new file {sf} up to {target_cov}%",
                    }
                    coverage_violations.append(d)
            else:
                # Existing files: ensure 100% of modified lines are covered AND total coverage is at or above 40% floor (for files <= 1000 lines to prevent legacy token waste)
                num_lines = 0
                try:
                    with open(sf, encoding="utf-8") as f_sf:
                        num_lines = len(f_sf.readlines())
                except Exception:  # noqa: S110
                    pass
                if actual_cov < 40 and num_lines <= 1000:
                    d = {
                        "file": sf,
                        "line": 0,
                        "error": f"Coverage target violation: actual coverage {actual_cov}% is below minimum safety floor (40%)",
                        "fix_hint": f"Add test cases to bring total coverage of {sf} above 40% (current: {actual_cov}%)",
                    }
                    coverage_violations.append(d)
                
                changed = _get_changed_lines(sf)
                if changed:
                    missing = _get_missing_lines(pt_res.stdout, sf)
                    uncovered_changed = missing & changed
                    if uncovered_changed:
                        d = {
                            "file": sf,
                            "line": 0,
                            "error": f"Coverage target violation (Modified File): changed lines {sorted(uncovered_changed)} are not covered by tests",
                            "fix_hint": f"Add test cases targeting the modified lines in {sf}: {sorted(uncovered_changed)}",
                        }
                        coverage_violations.append(d)


            mkey = sf.replace(".py", "")
            for line in pt_res.stdout.splitlines():
                if mkey in line or sf in line:
                    parts = line.split()
                    if len(parts) >= 5 and "%" in parts[-2]:
                        missing_infos.append(f"{sf.split('/')[-1]}:{parts[-1]}")

        if coverage_violations:
            first_fail = coverage_violations[0]
            _fail_exit(
                "coverage-target",
                f"FAIL | Coverage target missed: {first_fail['file']} ({first_fail['error']})",
                first_fail,
            )

        suffix = f", Missing: [{', '.join(missing_infos)}]" if missing_infos else ""
        cov_s = f"{cov_val}%" if cov_val is not None else "N/A"
        print(f"PASS | All checks passed (Cov {cov_s}{suffix})")
        print(_emit_json("PASS", "all", [], cov_val), file=sys.stderr)
    else:
        last_err = [
            line for line in pt_res.stdout.splitlines() if any(x in line for x in ("FAIL", "Error", "AssertionError"))
        ]
        cause = last_err[-1] if last_err else "Check pytest output."
        d = {"file": "", "line": 0, "error": cause, "fix_hint": "Fix failing assertions in tests"}
        _fail_exit("pytest", f"FAIL | Pytest Failed: {cause}", d)


if __name__ == "__main__":
    main()
