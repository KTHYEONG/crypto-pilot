import argparse
import os
import re
import subprocess
import sys


def run_cmd(cmd: list[str], timeout: int = 60) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(  # noqa: S603
            cmd, capture_output=True, text=True, shell=False, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=124,
            stdout="",
            stderr=f"Error: Process timed out after {timeout} seconds.",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Lean Check Helper with advanced rules check.")
    parser.add_argument("--files", nargs="+", required=True, help="List of modified files to check.")
    args = parser.parse_args()

    files: list[str] = args.files
    py_files = [f for f in files if f.endswith(".py")]

    if not py_files:
        print("🟢 PASS | No python files to check.")
        sys.exit(0)

    # 1. 1:1 Co-modification Mapping Verification (testing.md §2.4)
    # Check if for every src/*.py there is a corresponding test_*.py
    for pf in py_files:
        if pf.startswith("src/") and not pf.endswith("__init__.py"):
            # e.g., src/domain/futures/strategy/tiered_workflow/signal_selection.py
            # -> tests/unit/domain/futures/strategy/tiered_workflow/test_signal_selection.py
            parts = pf.split("/")
            module_name = parts[-1]
            test_module_name = f"test_{module_name}"
            # Check unit, integration, or e2e folders
            matched_test_found = False
            for category in ["unit", "integration", "e2e"]:
                sub_path = "/".join(parts[1:-1])
                test_dir = f"tests/{category}/{sub_path}" if sub_path else f"tests/{category}"
                test_path = f"{test_dir}/{test_module_name}"
                if os.path.exists(test_path):
                    matched_test_found = True
                    break
            if not matched_test_found:
                print("🔴 FAIL | 1:1 Test File Missing")
                print(f"- 🔍 Cause: {pf} has no matching test file in tests/[category]/...")
                print("- 🛠️ Fix: Create a corresponding test file following Co-modification Mapping.")
                sys.exit(1)

    # 2. Strict print() Usage Check (logging.md §Core & testing.md §1)
    # Prohibit raw print statements in code
    print_pattern = re.compile(r"(?<!#)\bprint\s*\(")
    for pf in py_files:
        with open(pf, encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                if print_pattern.search(line):
                    print("🔴 FAIL | Unsanctioned print() Detected")
                    print(f"- 🔍 Cause: {pf}:{idx} - Found print() call.")
                    print("- 🛠️ Fix: Replace print() with standard logger call.")
                    sys.exit(1)

    # 3. Ruff
    ruff_res = run_cmd(["uv", "run", "ruff", "check", *py_files, "--quiet"])
    if ruff_res.returncode != 0:
        print("🔴 FAIL | Ruff Lint Failed")
        print(f"- 🔍 Cause: {ruff_res.stdout.strip() or ruff_res.stderr.strip()}")
        print("- 🛠️ Fix: Resolve ruff lint errors in modified files.")
        sys.exit(1)

    # 4. Mypy
    mypy_res = run_cmd([
        "uv", "run", "mypy", *py_files,
        "--ignore-missing-imports"
    ])
    if mypy_res.returncode != 0:
        print("🔴 FAIL | Mypy Type Check Failed")
        print(f"- 🔍 Cause: {mypy_res.stdout.strip() or mypy_res.stderr.strip()}")
        print("- 🛠️ Fix: Resolve static typing errors.")
        sys.exit(1)

    # 5. Test & Coverage (find test files among py_files or corresponding tests)
    test_files = [f for f in py_files if f.startswith("tests/") or "test_" in f]
    source_files = [f for f in py_files if not (f.startswith("tests/") or "test_" in f)]

    if test_files:
        # Run pytest quiet on tests
        pytest_res = run_cmd(["uv", "run", "pytest", *test_files, "-q", "--tb=line"])
        if pytest_res.returncode != 0:
            last_err = [
                line for line in pytest_res.stdout.splitlines()
                if any(x in line for x in ("FAIL", "Error", "AssertionError"))
            ]
            cause = last_err[-1] if last_err else "Check pytest traceback."
            print("🔴 FAIL | Pytest Regression Failed")
            print(f"- 🔍 Cause: {cause}")
            print("- 🛠️ Fix: Fix failing assertions or errors in tests.")
            sys.exit(1)

        # Run coverage with multi-module target mapping
        cov_args: list[str] = []
        if source_files:
            for sf in source_files:
                cov_mod = sf.replace(".py", "").replace("/", ".")
                cov_args.append(f"--cov={cov_mod}")
        else:
            cov_args.append("--cov=src")

        cov_res = run_cmd([
            "uv", "run", "pytest", *cov_args,
            *test_files, "--cov-report=term-missing"
        ])
        cov_val = "N/A"
        missing_infos: list[str] = []
        warnings_found: list[str] = []

        # Parse output for coverage, missing lines, and Runtime/JIT warnings
        for line in cov_res.stdout.splitlines():
            if "TOTAL" in line:
                match = re.search(r"(\d+)%", line)
                if match:
                    cov_val = match.group(1)
            elif "Warning" in line or "RuntimeWarning" in line:
                warnings_found.append(line.strip())
            else:
                for sf in source_files:
                    mod_path_key = sf.replace(".py", "")
                    if mod_path_key in line or sf in line:
                        parts = line.split()
                        if len(parts) >= 5 and "%" in parts[-2]:
                            missing_infos.append(f"{sf.split('/')[-1]}:{parts[-1]}")

        missing_suffix = f", Missing: [{', '.join(missing_infos)}]" if missing_infos else ""
        warning_suffix = f" (Warnings: {len(warnings_found)} detected)" if warnings_found else ""
        print(f"🟢 PASS | All checks passed (Cov {cov_val}%{missing_suffix}){warning_suffix}")
    else:
        print("🟢 PASS | Lint & Type check passed. (No tests to run)")


if __name__ == "__main__":
    main()
