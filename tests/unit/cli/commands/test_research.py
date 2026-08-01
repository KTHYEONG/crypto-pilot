from __future__ import annotations

from src.cli.main import build_root_parser
from src.research.contracts import OIDeleveragingEvaluationRequest


def test_oi_deleveraging_cli_dispatches_fixed_request(monkeypatch) -> None:
    calls: list[OIDeleveragingEvaluationRequest] = []
    monkeypatch.setattr(
        "src.cli.commands.research.run_oi_deleveraging_evaluation", calls.append,
    )
    args = build_root_parser().parse_args([
        "research", "run", "oi-deleveraging",
        "--symbol", "ETHUSDT", "--start", "2022-04-01", "--no-log-run",
    ])
    args.handler(args)

    assert calls == [OIDeleveragingEvaluationRequest(
        symbol="ETHUSDT", start="2022-04-01", log_run=False,
    )]
