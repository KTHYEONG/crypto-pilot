#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from typing import Any

JsonDiag = dict[str, Any]


def _emit_json(
    status: str, phase: str,
    diagnostics: list[JsonDiag],
    coverage: int | None = None,
) -> str:
    return json.dumps({
        "status": status,
        "phase": phase,
        "exit_code": 0 if status == "PASS" else 1,
        "coverage": coverage,
        "diagnostics": diagnostics,
    })


def run_cmd(cmd: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(  # noqa: S603
            cmd, capture_output=True, text=True, shell=False, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            args=cmd, returncode=124, stdout="",
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
        if not fh or not name:
            continue
        if not os.path.exists(fh):
            d = {"file": fh, "line": 0, "error": f"Spec: file not found ({kind} {name})", "fix_hint": f"Create {fh}"}
            diagnostics.append(d)
            continue
        with open(fh) as sf:
            pat = rf"^(?:class|def)\s+{re.escape(name)}\b"
            if not re.search(pat, sf.read(), re.MULTILINE):
                msg = f"Spec: {kind} '{name}' not implemented"
                d = {"file": fh, "line": 0, "error": msg, "fix_hint": f"Implement {kind} {name} in {fh}"}
                diagnostics.append(d)

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
                    if re.search(rf"^def\s+{re.escape(test_name)}\b", tf.read(), re.MULTILINE):
                        found = True
                        break
            if found:
                break
        if not found:
            d = {"file": "", "line": 0, "error": f"Spec: missing test '{test_name}'", "fix_hint": f"Write {test_name}"}
            diagnostics.append(d)

    for w in contract.get("wiring", []):
        wf: str = w.get("file", "")
        anchor: str = w.get("anchor", "")
        if not wf or not anchor:
            continue
        if not os.path.exists(wf):
            d = {"file": wf, "line": 0, "error": "Spec: wiring target not found", "fix_hint": f"Create {wf}"}
            diagnostics.append(d)
            continue
        with open(wf) as f:
            if anchor not in f.read():
                hint = f"Add ref to {anchor} in {wf}"
                d = {"file": wf, "line": 0, "error": f"Spec: missing anchor '{anchor}'", "fix_hint": hint}
                diagnostics.append(d)

    return (1 if diagnostics else 0, diagnostics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lean Check with JSON diagnostics.")
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--spec", default=None, help="Path to spec contract JSON for compliance verification")
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
                f"tests/{cat}/{'/'.join(parts[1:-1])}/{test_name}"
                if parts[1:-1] else f"tests/{cat}/{test_name}"
            )
            for cat in ["unit", "integration", "e2e"]
        )
        if not found:
            d = {"file": pf, "line": 0, "error": f"No matching test for {pf}", "fix_hint": ""}
            _fail_exit("co-modification", f"FAIL | {pf}: test file missing", d)

    # 2. print() Detection (scripts/ excluded)
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
    ruff_res = run_cmd(["uv", "run", "ruff", "check", *py_files, "--quiet"])
    if ruff_res.returncode != 0:
        out = (ruff_res.stdout or ruff_res.stderr).strip()
        d = {"file": py_files[0] if py_files else "", "line": 0, "error": out, "fix_hint": "Resolve ruff errors"}
        _fail_exit("ruff", "FAIL | Ruff Lint Failed", d)

    # 4. Mypy
    mypy_res = run_cmd(["uv", "run", "mypy", *py_files, "--ignore-missing-imports"])
    if mypy_res.returncode != 0:
        out = (mypy_res.stdout or mypy_res.stderr).strip()
        d = {"file": py_files[0] if py_files else "", "line": 0, "error": out, "fix_hint": "Resolve type errors"}
        _fail_exit("mypy", "FAIL | Mypy Type Check Failed", d)

    # 5. Single pytest with coverage
    test_files = _find_test_files(py_files)
    source_files = _get_source_files(py_files)

    if not test_files:
        print("PASS | Lint & Type check passed (no tests to run)")
        print(_emit_json("PASS", "all", [], None), file=sys.stderr)
        return

    cov_args = (
        [f"--cov={sf.replace('.py', '').replace('/', '.')}" for sf in source_files]
        if source_files else ["--cov=src"]
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
        for sf in source_files:
            mkey = sf.replace(".py", "")
            for line in pt_res.stdout.splitlines():
                if mkey in line or sf in line:
                    parts = line.split()
                    if len(parts) >= 5 and "%" in parts[-2]:
                        missing_infos.append(f"{sf.split('/')[-1]}:{parts[-1]}")
        suffix = f", Missing: [{', '.join(missing_infos)}]" if missing_infos else ""
        cov_s = f"{cov_val}%" if cov_val is not None else "N/A"
        print(f"PASS | All checks passed (Cov {cov_s}{suffix})")
        print(_emit_json("PASS", "all", [], cov_val), file=sys.stderr)
    else:
        last_err = [
            line for line in pt_res.stdout.splitlines()
            if any(x in line for x in ("FAIL", "Error", "AssertionError"))
        ]
        cause = last_err[-1] if last_err else "Check pytest output."
        d = {"file": "", "line": 0, "error": cause, "fix_hint": "Fix failing assertions in tests"}
        _fail_exit("pytest", f"FAIL | Pytest Failed: {cause}", d)


if __name__ == "__main__":
    main()
