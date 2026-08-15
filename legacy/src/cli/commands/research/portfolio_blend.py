from __future__ import annotations

import argparse

from src.application.research.blend import evaluation as evaluation_module
from src.research.contracts import SleeveBlendEvaluationRequest

_SLEEVE_UNIVERSE_CHOICES = ("core5_v1",)
_SLEEVE_CANDIDATE_KINDS = (
    "fixed_long_only_v1",
    "funding_signed_directional_v1",
    "core5_causal_tournament_v1",
)
# Source-controlled tournament profiles: discovery end and the untouched
# qualification window that follows it. A new profile is a new source id with a
# fresh pre-registration record, never a runtime argument.
_TOURNAMENT_PROFILES: dict[str, dict[str, str]] = {
    "pbgt_discovery_v1": {
        "discovery_end": "2024-12-31 23:59:59+00:00",
        "qualification_interval": "365D",
    },
}


def _run_sleeve_blend(args: argparse.Namespace) -> None:
    profile = _TOURNAMENT_PROFILES[args.tournament_profile]
    request = SleeveBlendEvaluationRequest(
        universe_id=args.universe_id,
        mdd_budget_fraction=args.mdd_budget_fraction,
        candidate_kind=args.candidate_kind,
        discovery_end=profile["discovery_end"],
        qualification_interval=profile["qualification_interval"],
        start=args.start,
        end=args.end,
        initial_equity=args.initial_equity,
        unseal_holdout=args.unseal_holdout,
        log_run=not args.no_log_run,
    )
    evaluation_module.run_sleeve_blend_evaluation(request)


def add_portfolio_blend_commands(run_sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Attach the ``research run portfolio blend`` subcommand.

    Production blend execution is locked to a source-controlled universe id and
    source-controlled candidate/tournament profile ids; free-form ``--symbols``
    is no longer accepted.
    """
    sleeve = run_sub.add_parser("blend", help="Run a sleeve-blend evaluation")
    sleeve.add_argument(
        "--universe-id", default="core5_v1", choices=_SLEEVE_UNIVERSE_CHOICES,
    )
    sleeve.add_argument("--mdd-budget-fraction", type=float, default=0.85)
    sleeve.add_argument(
        "--candidate-kind", default="fixed_long_only_v1",
        choices=_SLEEVE_CANDIDATE_KINDS,
    )
    sleeve.add_argument(
        "--tournament-profile", default="pbgt_discovery_v1",
        choices=sorted(_TOURNAMENT_PROFILES),
        help="Source-controlled discovery/qualification window profile",
    )
    sleeve.add_argument("--start", default=None)
    sleeve.add_argument("--end", default=None)
    sleeve.add_argument("--initial-equity", type=float, default=10_000.0)
    sleeve.add_argument("--unseal-holdout", action="store_true", default=False)
    sleeve.add_argument("--no-log-run", action="store_true", default=False)
    sleeve.set_defaults(handler=_run_sleeve_blend)
