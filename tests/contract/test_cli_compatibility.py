"""CLI surface contract after the legacy isolation refactor.

SCENARIO_MHS_REFACTOR_09: ``build_root_parser`` exposes exactly the ``data`` and
``research`` groups, and ``research run portfolio mhs-horizon-diagnostic`` parses
into the MHS handler while the removed ``single``/``expert``/``blend``/``growth``
subcommands and the ``provenance`` group raise ``SystemExit``.
"""

from __future__ import annotations

import pytest

from src.cli.main import build_root_parser


def test_root_parser_exposes_only_data_and_research_groups() -> None:
    parser = build_root_parser()
    args = parser.parse_args(["data", "collect", "funding", "BTCUSDT", "--end", "2025-01-01"])
    assert args.group == "data"
    args = parser.parse_args(["research", "run", "portfolio", "mhs-horizon-diagnostic"])
    assert args.group == "research"


def test_provenance_group_removed() -> None:
    with pytest.raises(SystemExit):
        build_root_parser().parse_args(["provenance", "compare-runs"])


@pytest.mark.parametrize(
    "argv",
    [
        ["research", "run", "single", "baseline"],
        ["research", "run", "single", "technical"],
        ["research", "run", "single", "carry"],
        ["research", "run", "single", "oi"],
        ["research", "run", "portfolio", "multi"],
        ["research", "run", "portfolio", "blend"],
        ["research", "run", "portfolio", "growth"],
        ["research", "run", "expert", "eval"],
    ],
)
def test_removed_subcommands_raise_system_exit(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        build_root_parser().parse_args(argv)


def test_mhs_horizon_diagnostic_parses_into_mhs_handler(monkeypatch) -> None:
    args = build_root_parser().parse_args([
        "research", "run", "portfolio", "mhs-horizon-diagnostic",
        "--no-log-run", "--start", "2021-01-01", "--end", "2021-01-02",
    ])
    assert args.portfolio_command == "mhs-horizon-diagnostic"

    from src.cli.commands.research.mhs import _run_mhs_horizon_diagnostic

    captured: list[object] = []

    class _Report:
        status = "COMPLETE"
        blend = None

        def __init__(self) -> None:
            self.books: dict[str, object] = {}

    monkeypatch.setattr(
        "src.application.research.mhs.evaluation.run_mhs_horizon_diagnostic",
        lambda request: captured.append(request) or _Report(),
    )
    monkeypatch.setattr(
        "src.application.research.mhs.evaluation.persist_mhs_horizon_diagnostic_report",
        lambda report, path, tier, **kwargs: path,
    )
    _run_mhs_horizon_diagnostic(args)
    assert len(captured) == 1
    assert captured[0].log_run is False
    assert "2021-01-01" in str(captured[0].start)
