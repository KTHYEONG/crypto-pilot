"""CLI surface contract after the part-4 lab isolation.

The ``research`` group is removed and ``backtest`` keeps only
``strategy|account|exposure``; exploratory commands live under ``lab``.
``lab horizon-diagnostic`` parses into the lab handler with unchanged
arguments while ``research …``, ``backtest mhs`` and the migrated ``ops``
leaves raise ``SystemExit``.
"""

from __future__ import annotations

import pytest

from src.cli.main import build_root_parser


def _parse(argv: list[str]):
    return build_root_parser(argv).parse_args(argv)


def test_root_parser_exposes_lab_instead_of_research() -> None:
    args = _parse(["lab", "horizon-diagnostic"])
    assert args.group == "lab"
    args = _parse(["lab", "process-backtest"])
    assert args.group == "lab"


def test_research_group_removed() -> None:
    with pytest.raises(SystemExit):
        _parse(["research", "run", "portfolio", "mhs-horizon-diagnostic"])


@pytest.mark.parametrize(
    "argv",
    [
        ["backtest", "mhs"],
        ["ops", "backtests-migrate", "--registry-path", "x"],
        ["ops", "backtests-verify-history-migration", "--registry-path", "x", "--history-directory", "y"],
        ["ops", "procedure-registry-migrate", "--legacy-path", "a", "--target-path", "b"],
    ],
)
def test_migrated_leaves_raise_system_exit(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        _parse(argv)


def test_lab_horizon_diagnostic_parses_into_lab_handler(monkeypatch) -> None:
    args = _parse([
        "lab", "horizon-diagnostic",
        "--start", "2021-01-01", "--end", "2021-01-02",
    ])
    assert args.lab_command == "horizon-diagnostic"

    from src.cli.commands.lab import _run_horizon_diagnostic

    captured: list[object] = []

    class _Report:
        status = "COMPLETE"
        blend = None

        def __init__(self) -> None:
            self.books: dict[str, object] = {}

    monkeypatch.setattr(
        "src.lab.mhs.pipeline.orchestrator.run_mhs_diagnostic",
        lambda config, **kwargs: captured.append(config) or _Report(),
    )
    monkeypatch.setattr(
        "src.lab.mhs.report.persist.persist_mhs_horizon_diagnostic_report",
        lambda report, path, tier, **kwargs: path,
    )
    assert args.handler is _run_horizon_diagnostic
    _run_horizon_diagnostic(args)
    assert len(captured) == 1
    assert "2021-01-01" in str(captured[0])


def test_cli_contract_groups() -> None:
    """The deployed daemon's entry point must survive the move."""
    from src.cli.main import build_root_parser

    parser = build_root_parser()
    groups = parser._subparsers._group_actions[0].choices  # type: ignore[union-attr]
    assert set(groups) == {"backtest", "data", "live", "ops", "lab"}
