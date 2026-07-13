import argparse
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
    parser = argparse.ArgumentParser(description="Lean Check Helper to save tokens.")
    parser.add_argument("--files", nargs="+", required=True, help="List of modified files to check.")
    args = parser.parse_args()

    files: list[str] = args.files
    py_files = [f for f in files if f.endswith(".py")]

    if not py_files:
        print("🟢 PASS | No python files to check.")
        sys.exit(0)

    # 1. Ruff
    ruff_res = run_cmd(["uv", "run", "ruff", "check", *py_files, "--quiet"])
    if ruff_res.returncode != 0:
        print("🔴 FAIL | Ruff Lint Failed")
        print(f"- 🔍 Cause: {ruff_res.stdout.strip() or ruff_res.stderr.strip()}")
        print("- 🛠️ Fix: Resolve ruff lint errors in modified files.")
        sys.exit(1)

    # 2. Mypy
    mypy_res = run_cmd([
        "uv", "run", "mypy", *py_files,
        "--ignore-missing-imports"
    ])
    if mypy_res.returncode != 0:
        print("🔴 FAIL | Mypy Type Check Failed")
        print(f"- 🔍 Cause: {mypy_res.stdout.strip() or mypy_res.stderr.strip()}")
        print("- 🛠️ Fix: Resolve static typing errors.")
        sys.exit(1)

    # 3. Test & Coverage (find test files among py_files or corresponding tests)
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
        for line in cov_res.stdout.splitlines():
            if "TOTAL" in line:
                match = re.search(r"(\d+)%", line)
                if match:
                    cov_val = match.group(1)
            else:
                for sf in source_files:
                    mod_path_key = sf.replace(".py", "")
                    if mod_path_key in line or sf in line:
                        parts = line.split()
                        if len(parts) >= 5 and "%" in parts[-2]:
                            missing_infos.append(f"{sf.split('/')[-1]}:{parts[-1]}")

        missing_suffix = f", Missing: [{', '.join(missing_infos)}]" if missing_infos else ""
        print(f"🟢 PASS | All checks passed (Cov {cov_val}%{missing_suffix})")
    else:
        print("🟢 PASS | Lint & Type check passed. (No tests to run)")


if __name__ == "__main__":
    main()
