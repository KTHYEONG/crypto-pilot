"""McCabe complexity ratchet for production code."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Final

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

COMPLEXITY_CEILINGS: Final[dict[str, int]] = {
    "src/application/ops/gdrive_cleanup.py::build_cleanup_plan": 30,
    "src/backtests/migration.py::_migrate_history_source": 16,
    "src/backtests/migration.py::_migrate_run_source": 16,
    "src/backtests/migration.py::verify_legacy_history_migration": 26,
    "src/capture/config.py::__post_init__": 16,
    "src/live/data_refresh.py::refresh_live_market_data": 34,
    "src/live/execution_quality.py::summarize_execution_quality": 19,
    "src/live/executor.py::_poll_active": 16,
    "src/live/executor.py::_poll_or_post": 41,
    "src/live/executor.py::execute_intents": 20,
    "src/live/frozen_book.py::unit_proxy_returns": 22,
    "src/live/frozen_signal.py::run_frozen_signal_step": 37,
    "src/live/funding_backfill.py::compute_funding_backfill": 20,
    "src/live/ledger.py::load_ledger": 25,
    "src/live/microstructure.py::fetch_book_quotes": 16,
    "src/live/portfolio_state.py::summarize_portfolio_state": 19,
    "src/live/preflight.py::run_preflight": 25,
    "src/live/rest.py::_request_get": 24,
    "src/live/runner.py::run_shadow_cycle": 65,
    # spec 17: mainnet refuse-to-start gate (fail loud before any venue call).
    "src/live/scheduler.py::run_daemon": 64,
    "src/live/tax_ledger.py::_collect_income": 20,
    # spec 17: coverage/genesis watermark persistence validates every new key fail-closed.
    "src/live/tax_ledger.py::load_tax_watermark": 18,
    "src/market_data/binance/futures.py::fetch_funding_rate_history": 22,
    "src/market_data/binance/futures.py::fetch_ohlcv_with_taker": 27,
    "src/market_data/services/futures_collection.py::ensure_funding_data": 21,
    "src/market_data/services/futures_collection.py::ensure_ohlcv_data": 22,
    "src/market_data/services/source_gap_audit.py::audit_source_gap_registry": 19,
    "src/market_data/services/source_gap_audit.py::write_audited_registry": 18,
    "src/market_data/services/spot_collection.py::ensure_spot_ohlcv": 17,
    "src/market_data/streams/liquidations.py::parse_liquidation": 32,
    "src/market_data/streams/recorder_health.py::_capture_findings": 18,
    "src/market_data/streams/retention.py::prune_backed_up": 19,
    "src/market_data/streams/snapshots.py::_premium_row_reject": 17,
    "src/mhs/account_ledger.py::replay_account": 42,
    "src/mhs/backtest/certification.py::assess_process_validation": 31,
    "src/mhs/backtest/certification.py::inventory_daily_evidence": 19,
    "src/mhs/backtest/inventory.py::_validate_replay_window": 30,
    "src/mhs/backtest/inventory.py::evaluate_process_inventory_backtest": 28,
    "src/mhs/backtest/journal.py::__post_init__": 20,
    "src/mhs/backtest/labels.py::build_proxy_member_returns": 16,
    "src/mhs/backtest/paths.py::evaluate_process_backtest": 18,
    "src/mhs/backtest/paths.py::run_process_paths": 31,
    "src/mhs/backtest/selection.py::choose_refit_policy": 21,
    "src/mhs/deploy_gate.py::evaluate_continuous_growth_survival": 17,
    "src/mhs/discovery.py::select_horizon_by_discovery_qualification": 28,
    "src/mhs/evaluation/windows.py::_book_outcome": 25,
    "src/mhs/execution/accumulator.py::_consume_append_ledger": 26,
    "src/mhs/execution/batch.py::replay_execution_window_batch_isolated": 17,
    "src/mhs/execution/batch.py::replay_execution_windows_coupled": 18,
    "src/mhs/execution/ledger.py::simulated_inventory_ledger": 24,
    "src/mhs/execution/microstructure.py::peg_chase_partial_schedule": 19,
    "src/mhs/execution/window_stream.py::_iter_mhs_execution_windows": 38,
    "src/mhs/frozen_research_candidate.py::build_frozen_mhs_candidate": 18,
    "src/mhs/frozen_research_run.py::__post_init__": 16,
    "src/mhs/frozen_research_windows.py::validated_frozen_research_windows": 21,
    "src/mhs/growth_exposure.py::solve_log_growth_exposure": 18,
    "src/mhs/panel.py::load_base_panel": 21,
    "src/mhs/validation.py::validate_request": 45,
}
"""Frozen McCabe ceilings for pre-existing functions above the project threshold.

Keys are ``<repo-relative path>::<function name>`` exactly as ruff reports them.
A function not listed must stay at or below ``[tool.ruff.lint.mccabe]
max-complexity``; a listed function may never exceed its frozen value. Entries
are deleted (never raised) when a function is simplified, renamed or removed.
"""

_C901_PATTERN = re.compile(r"`(.+?)` is too complex \((\d+) > (\d+)\)")


def _complexity_threshold() -> int:
    """Project McCabe threshold read from pyproject.toml (single source of truth).

    Raises AssertionError when the ``[tool.ruff.lint.mccabe] max-complexity`` key
    is missing or not a positive int, so the ratchet can never silently run with
    an implicit default.
    """
    pyproject = _REPO_ROOT / "pyproject.toml"
    with pyproject.open("rb") as fh:
        config = tomllib.load(fh)
    try:
        value = config["tool"]["ruff"]["lint"]["mccabe"]["max-complexity"]
    except KeyError as exc:
        raise AssertionError("missing [tool.ruff.lint.mccabe] max-complexity in pyproject.toml") from exc
    assert isinstance(value, int), f"invalid max-complexity: {value!r}"
    assert value > 0, f"invalid max-complexity: {value!r}"
    return value


def _measured_complexity(scan_root: Path | None = None) -> dict[str, int]:
    """McCabe complexity of every src/ function above the threshold, via ruff C901.

    Runs ``python -m ruff check src --select C901 --ignore-noqa --exit-zero
    --no-cache --output-format json`` from the repository root. ``--ignore-noqa``
    makes this map the only exemption mechanism: a ``# noqa`` (blanket or C901)
    cannot hide a function from the ratchet. ``--no-cache`` keeps the test from
    writing ``.ruff_cache``. Same-named functions in one file collapse to their
    maximum. Raises AssertionError on a non-zero exit, unparsable JSON, or a C901
    message that does not match the expected
    ```name` is too complex (N > M)`` shape, or whose reported limit M differs
    from ``_complexity_threshold()`` (fail closed on tool-format drift).
    """
    threshold = _complexity_threshold()
    target = "src" if scan_root is None else str(scan_root)
    proc = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            target,
            "--select",
            "C901",
            "--ignore-noqa",
            "--exit-zero",
            "--no-cache",
            "--output-format",
            "json",
        ],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"ruff C901 scan failed: {proc.stderr[:500]}"
    try:
        findings = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise AssertionError(f"unparsable ruff JSON: {exc}") from exc
    measured: dict[str, int] = {}
    for finding in findings:
        message = finding.get("message", "")
        match = _C901_PATTERN.search(message)
        assert match is not None, f"unexpected C901 message shape: {message!r}"
        name, complexity_raw, limit_raw = match.groups()
        assert int(limit_raw) == threshold, f"C901 limit drift: {message!r}"
        filename = finding.get("filename", "")
        try:
            rel = Path(filename).relative_to(_REPO_ROOT).as_posix()
        except ValueError as exc:
            rel = Path(filename).as_posix()
            if rel.startswith("/"):
                raise AssertionError(f"C901 filename outside repo: {filename!r}") from exc
        key = f"{rel}::{name}"
        measured[key] = max(measured.get(key, 0), int(complexity_raw))
    return measured


def test_no_function_exceeds_complexity_ceiling() -> None:
    """Standing ratchet: no new function above the threshold, no frozen growth."""
    threshold = _complexity_threshold()
    measured = _measured_complexity()
    offenders = {
        key: value
        for key, value in measured.items()
        if value > COMPLEXITY_CEILINGS.get(key, threshold)
    }
    assert offenders == {}, (
        "functions exceed complexity ceiling: "
        + ", ".join(
            f"{key} measured {measured[key]} > ceiling {COMPLEXITY_CEILINGS.get(key, threshold)}"
            for key in sorted(offenders)
        )
    )


def test_complexity_ceilings_are_live() -> None:
    """Every frozen entry still exists in src/ and still exceeds the threshold.

    A missing entry means the function was renamed, deleted or simplified to
    <= threshold; in every case the entry must be removed so the map stays the
    exact inventory of tolerated debt.
    """
    threshold = _complexity_threshold()
    measured = _measured_complexity()
    stale = sorted(
        key
        for key, ceiling in COMPLEXITY_CEILINGS.items()
        if key not in measured or ceiling <= threshold
    )
    assert stale == [], f"stale complexity ceilings (remove them): {stale}"


def test_noqa_cannot_hide_complexity(tmp_path: Path) -> None:
    """A noqa comment cannot hide an over-threshold function from the ratchet."""
    import shutil

    shutil.copy(_REPO_ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    nested = "\n".join(f"{'    ' * (i + 1)}if x{i}:  # branch {i}" for i in range(16))
    closing = "\n".join(f"{'    ' * (16 - i)}    pass" for i in range(16))
    (tmp_path / "noisy.py").write_text(
        f"def over_threshold(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15):  # noqa: C901\n{nested}\n{closing}\n    return x0\n",
        encoding="utf-8",
    )
    measured = _measured_complexity(tmp_path / "noisy.py")
    threshold = _complexity_threshold()
    assert measured, "ruff reported nothing for the noisy fixture"
    assert max(measured.values()) > threshold
