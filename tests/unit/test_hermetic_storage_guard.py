"""Invariant scenarios for the hermetic storage-root guard."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.fixtures.hermetic import assert_storage_roots_hermetic


def test_contained_roots_pass(tmp_path: Path) -> None:
    assert assert_storage_roots_hermetic(
        {"BACKTESTS_DIR": tmp_path / "backtests", "LOG_DIR": tmp_path / "logs", "ROOT": tmp_path},
        tmp_path,
        ("src", "src.common.paths"),
    ) is None


def test_escaped_root_fails_loudly_with_diagnosis(tmp_path: Path) -> None:
    offending = tmp_path / "data" / "backtests"
    temp_root = tmp_path / "proc"
    temp_root.mkdir()
    with pytest.raises(pytest.UsageError) as excinfo:
        assert_storage_roots_hermetic(
            {"BACKTESTS_DIR": offending, "LOG_DIR": temp_root / "logs"},
            temp_root,
            ("src.common.paths",),
        )
    message = str(excinfo.value)
    assert "BACKTESTS_DIR" in message
    assert str(offending.resolve()) in message
    assert str(temp_root.resolve()) in message
    assert "src.common.paths" in message
    assert "LOG_DIR=" not in message


def test_all_offenders_reported_together(tmp_path: Path) -> None:
    temp_root = tmp_path / "proc"
    temp_root.mkdir()
    with pytest.raises(pytest.UsageError) as excinfo:
        assert_storage_roots_hermetic(
            {
                "BACKTESTS_DIR": tmp_path / "data" / "backtests",
                "LOG_DIR": tmp_path / "other" / "logs",
            },
            temp_root,
            (),
        )
    message = str(excinfo.value)
    assert "BACKTESTS_DIR" in message
    assert "LOG_DIR=" in message
    assert message.index("BACKTESTS_DIR=") < message.index("LOG_DIR=")
    assert "CRYPTO_PILOT_BACKTESTS_DIR/CRYPTO_PILOT_LOG_DIR" in message
    assert "preimported src modules: none" in message


def test_preimported_module_diagnosis_is_bounded(tmp_path: Path) -> None:
    modules = tuple(f"src.module_{index:02d}" for index in range(23))
    with pytest.raises(pytest.UsageError) as excinfo:
        assert_storage_roots_hermetic({"BACKTESTS_DIR": tmp_path / "outside"}, tmp_path / "proc", modules)
    message = str(excinfo.value)
    assert ", ".join(modules[:20]) in message
    assert "23 module(s)" in message
    assert "(+3 more)" in message
    assert all(name not in message for name in modules[20:])


def test_symlinked_root_resolves_before_containment(tmp_path: Path) -> None:
    temp_root = tmp_path / "proc"
    temp_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "link"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(pytest.UsageError):
        assert_storage_roots_hermetic(
            {"BACKTESTS_DIR": link},
            temp_root,
            (),
        )


def test_live_session_roots_are_partitioned() -> None:
    from src.common import logging as _src_logging
    from src.common import paths as _src_paths

    temp_root = Path(os.environ["PYTEST_DEBUG_TEMPROOT"]).resolve()
    for root in (_src_paths.BACKTESTS_DIR, _src_paths.STRATEGY_BACKTESTS_DIR, _src_logging.LOG_DIR):
        assert root.resolve().is_relative_to(temp_root)


def test_importing_research_surface_writes_nothing(tmp_path: Path) -> None:
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    backtests_dir = tmp_path / "bt"
    log_dir = tmp_path / "logs"
    probe = (
        "import src.common.paths as p, src.common.logging as l, src.mhs.telemetry as t,"
        " src.mhs.report.persist, src.mhs.preregistration,"
        " src.mhs.pipeline.orchestrator, src.cli.main;"
        " tel = t.StageTelemetry();"
        " print(str(p.BACKTESTS_DIR)); print(str(l.LOG_DIR))"
    )
    env = {
        **os.environ,
        "CRYPTO_PILOT_BACKTESTS_DIR": str(backtests_dir),
        "CRYPTO_PILOT_LOG_DIR": str(log_dir),
    }
    result = subprocess.run([sys.executable, "-c", probe], cwd=repo_root, capture_output=True, text=True, env=env, timeout=120)  # noqa: S603
    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert lines[-2] == str(backtests_dir)
    assert lines[-1] == str(log_dir)
    assert not backtests_dir.exists()
    assert not log_dir.exists()


def test_no_module_level_filesystem_call_in_src() -> None:
    import ast

    repo_root = Path(__file__).resolve().parents[2]
    watched = {"mkdir", "makedirs", "open", "write_text", "write_bytes", "touch", "FileHandler", "RotatingFileHandler", "basicConfig", "addHandler", "setup_logger", "connect", "to_parquet"}
    offenders: list[str] = []
    for path in sorted((repo_root / "src").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            if isinstance(node, ast.If):
                test = node.test
                if (
                    isinstance(test, ast.Compare)
                    and isinstance(test.left, ast.Name)
                    and test.left.id == "__name__"
                    and len(test.ops) == 1
                    and isinstance(test.ops[0], ast.Eq)
                    and len(test.comparators) == 1
                    and isinstance(test.comparators[0], ast.Constant)
                    and test.comparators[0].value == "__main__"
                ):
                    continue
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    func = child.func
                    name = func.attr if isinstance(func, ast.Attribute) else (func.id if isinstance(func, ast.Name) else None)
                    if name in watched:
                        offenders.append(f"{path.relative_to(repo_root)}:{child.lineno}:{name}")
    assert offenders == []


def test_escaped_file_handler_is_reported(tmp_path: Path) -> None:
    import contextlib
    import logging

    from tests.fixtures.hermetic import escaped_file_handlers

    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    name_a = "hermetic_probe_outside_logger"
    name_b = "hermetic_probe_inside_logger"
    logger_a = logging.getLogger(name_a)
    logger_b = logging.getLogger(name_b)
    handler_a = logging.FileHandler(outside / "a.log", encoding="utf-8")
    handler_b = logging.FileHandler(proc_root / "b.log", encoding="utf-8")
    stream_handler = logging.StreamHandler()
    logger_a.addHandler(handler_a)
    logger_b.addHandler(handler_b)
    logger_b.addHandler(stream_handler)
    try:
        probes = (name_a, name_b)
        offenders = [item for item in escaped_file_handlers(proc_root) if item[0] in probes]
        assert offenders == [(name_a, (outside / "a.log").resolve())]
        widened = [item for item in escaped_file_handlers(proc_root, outside) if item[0] in probes]
        assert widened == []
    finally:
        for logger, handler in ((logger_a, handler_a), (logger_b, handler_b), (logger_b, stream_handler)):
            with contextlib.suppress(Exception):
                logger.removeHandler(handler)
            with contextlib.suppress(Exception):
                handler.close()


def test_live_session_has_no_escaped_file_handlers(tmp_path_factory: pytest.TempPathFactory) -> None:
    from tests.fixtures.hermetic import escaped_file_handlers

    temp_root = Path(os.environ["PYTEST_DEBUG_TEMPROOT"])
    assert escaped_file_handlers(temp_root, tmp_path_factory.getbasetemp()) == []


def test_subprocess_child_without_destination_cannot_write(tmp_path: Path) -> None:
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    child_bt = tmp_path / "child_bt"
    child_logs = tmp_path / "child_logs"
    out = tmp_path / "out"
    probe = (
        "import pandas as pd\n"
        "from pathlib import Path as _P\n"
        f"out=_P(r'{out}')\n"
        "from src.mhs.run_history import append_run_history_record\n"
        "from src.mhs.report.persist import persist_mhs_report\n"
        "from src.mhs.preregistration import ProcedureRegistration, record_forward_evaluation, register_procedure\n"
        "from src.mhs.contracts import MhsDiagnosticRequest\n"
        "now=pd.Timestamp('2026-09-17', tz='UTC')\n"
        "reg=ProcedureRegistration('d'*32, now, pd.Timestamp('2026-06-30 23:59:59+00:00', tz='UTC'), {})\n"
        "end=pd.Timestamp('2026-12-31', tz='UTC')\n"
        "rec={'run_id':'x'}\n"
        "calls=[\n"
        "lambda: append_run_history_record(rec),\n"
        "lambda: append_run_history_record(rec, None),\n"
        "lambda: persist_mhs_report(object(), out/'r.json'),\n"
        "lambda: persist_mhs_report(object(), out/'r.json', history_dir=None),\n"
        "lambda: record_forward_evaluation(reg, end, now=now),\n"
        "lambda: register_procedure(MhsDiagnosticRequest(), now=now),\n"
        "]\n"
        "for _fn in calls:\n"
        " try:\n"
        "  _fn()\n"
        " except Exception as _e:\n"
        "  print(type(_e).__name__)\n"
    )
    env = {
        **os.environ,
        "CRYPTO_PILOT_BACKTESTS_DIR": str(child_bt),
        "CRYPTO_PILOT_LOG_DIR": str(child_logs),
    }
    result = subprocess.run([sys.executable, "-c", probe], cwd=repo_root, capture_output=True, text=True, env=env, timeout=120)  # noqa: S603
    assert result.returncode == 0, result.stderr
    lines = [line for line in result.stdout.strip().splitlines() if line]
    assert lines == ["TypeError"] * 6
    assert not child_bt.exists()
    assert not child_logs.exists()
    assert not out.exists()


def test_cli_default_resolves_operator_path() -> None:
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    probe = (
        "import src.common.paths as p, src.common.logging as l, src.mhs.preregistration as pr;"
        " print(str(p.BACKTESTS_DIR)); print(str(l.LOG_DIR)); print(str(pr.PROCEDURE_REGISTRY_PATH))"
    )
    env = {k: v for k, v in os.environ.items() if k not in ("CRYPTO_PILOT_BACKTESTS_DIR", "CRYPTO_PILOT_LOG_DIR")}
    result = subprocess.run([sys.executable, "-c", probe], cwd=repo_root, capture_output=True, text=True, env=env, timeout=120)  # noqa: S603
    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert lines[0] == str(repo_root / "data" / "backtests")
    assert lines[1] == str(repo_root / "logs")
    assert lines[2] == str(repo_root / "data" / "backtests" / "procedure_registry.jsonl")
