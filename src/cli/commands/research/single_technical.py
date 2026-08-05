from __future__ import annotations

import argparse
import logging

from src.application.research.technical import evaluation as evaluation_module
from src.research.contracts import TechnicalExpertEvaluationRequest

_logger = logging.getLogger("XsScreen")


def _run_technical_expert(args: argparse.Namespace) -> None:
    request = TechnicalExpertEvaluationRequest(
        candidate_id=args.candidate_id,
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        initial_equity=args.initial_equity,
        unseal_holdout=args.unseal_holdout,
        log_run=not args.no_log_run,
    )
    evaluation_module.run_technical_expert_evaluation(request)


def _run_trend_screen(args: argparse.Namespace) -> None:
    from src.application.research.technical.trend_screen import (
        TREND_SCREEN_PROFILE_ID,
        persist_trend_screen_report,
        run_trend_screen,
        trend_screen_report_path,
    )

    profile = args.profile or TREND_SCREEN_PROFILE_ID
    if profile != TREND_SCREEN_PROFILE_ID:
        raise ValueError(
            f"unknown trend-screen profile '{profile}'; the source-controlled "
            f"profile is '{TREND_SCREEN_PROFILE_ID}'"
        )
    report = run_trend_screen(start=args.start, end=args.end)
    persist_trend_screen_report(report, trend_screen_report_path())


def _run_xs_trend_screen(args: argparse.Namespace) -> None:
    from src.application.research.technical.xs_trend_screen import (
        XS_ALPHA_PROFILE_ID,
        XS_CONTEXTUAL_ALPHA_PROFILE_ID,
        XS_DUAL_FAMILY_ALPHA_PROFILE_ID,
        XS_NEUTRAL_PROFILE_ID,
        XS_POSITIONING_ALPHA_PROFILE_ID,
        XS_SCORE_ROUTED_ALPHA_PROFILE_ID,
        XS_VOL_WEIGHTED_ALPHA_PROFILE_ID,
        persist_xs_screen_report,
        run_xs_trend_screen,
        xs_screen_report_path,
    )

    profile = args.profile or XS_VOL_WEIGHTED_ALPHA_PROFILE_ID
    if profile not in (
        XS_NEUTRAL_PROFILE_ID,
        XS_ALPHA_PROFILE_ID,
        XS_CONTEXTUAL_ALPHA_PROFILE_ID,
        XS_SCORE_ROUTED_ALPHA_PROFILE_ID,
        XS_DUAL_FAMILY_ALPHA_PROFILE_ID,
        XS_VOL_WEIGHTED_ALPHA_PROFILE_ID,
        XS_POSITIONING_ALPHA_PROFILE_ID,
    ):
        raise ValueError(
            f"unknown xs screen profile '{profile}'; the source-controlled "
            f"profiles are '{XS_NEUTRAL_PROFILE_ID}', '{XS_ALPHA_PROFILE_ID}', "
            f"'{XS_CONTEXTUAL_ALPHA_PROFILE_ID}', "
            f"'{XS_SCORE_ROUTED_ALPHA_PROFILE_ID}', "
            f"'{XS_DUAL_FAMILY_ALPHA_PROFILE_ID}', "
            f"'{XS_VOL_WEIGHTED_ALPHA_PROFILE_ID}', and "
            f"'{XS_POSITIONING_ALPHA_PROFILE_ID}'"
        )
    report = run_xs_trend_screen(
        start=args.start, end=args.end, unseal_holdout=args.unseal_holdout,
        profile=profile,
    )
    persist_xs_screen_report(report, xs_screen_report_path(profile))
    _logger.info(
        "xs-screen profile=%s discovery_admitted=%s qualification_admitted=%s "
        "qualification_sharpe=%.4f binding_constraint=%s",
        report.profile,
        report.discovery.admitted,
        report.qualification.admitted,
        report.qualification.sharpe,
        report.qualification.binding_constraint,
    )

    # v6's adopted deployment scale (ADR_20260805_xs_alpha_growth_vol_targeting):
    # the raw screen above answers "is there edge" (unscaled); this answers "how
    # much of it to run" and is reported by default alongside it for v6.
    if profile == XS_VOL_WEIGHTED_ALPHA_PROFILE_ID and not args.no_growth_sizing:
        from src.application.research.technical.xs_alpha_growth_sizing import (
            persist_xs_growth_sizing_report,
            run_xs_alpha_growth_sizing,
            xs_growth_sizing_report_path,
        )

        sizing_report = run_xs_alpha_growth_sizing(
            profile=profile, unseal_holdout=args.unseal_holdout,
        )
        persist_xs_growth_sizing_report(
            sizing_report, xs_growth_sizing_report_path(profile),
        )
        _logger.info(
            "xs-screen profile=%s adopted_sizing selected_risk=%s "
            "vol_target_window=%s qualification_admitted=%s "
            "qualification_sharpe=%.4f",
            sizing_report.profile,
            sizing_report.sizing.selected_risk,
            sizing_report.vol_target_window,
            sizing_report.qualification.admitted,
            sizing_report.qualification.sharpe,
        )


def _run_xs_alpha_growth_sizing(args: argparse.Namespace) -> None:
    from src.application.research.technical.xs_alpha_growth_sizing import (
        persist_xs_growth_sizing_report,
        run_xs_alpha_growth_sizing,
        xs_growth_sizing_report_path,
    )
    from src.application.research.technical.xs_trend_screen import (
        XS_ALPHA_PROFILE_ID,
        XS_VOL_WEIGHTED_ALPHA_PROFILE_ID,
    )

    profile = args.profile or XS_VOL_WEIGHTED_ALPHA_PROFILE_ID
    if profile not in (XS_ALPHA_PROFILE_ID, XS_VOL_WEIGHTED_ALPHA_PROFILE_ID):
        raise ValueError(
            f"unknown growth-sizing profile '{profile}'; sizing is restricted "
            f"to the end-to-end admitted profiles '{XS_ALPHA_PROFILE_ID}' and "
            f"'{XS_VOL_WEIGHTED_ALPHA_PROFILE_ID}'"
        )
    # --no-vol-target (disable) and --vol-target-window (fixed int) are the
    # only explicit overrides; when neither is given the kwarg is omitted so
    # run_xs_alpha_growth_sizing's own default (the grid search) applies.
    sizing_kwargs: dict[str, int | None] = {}
    if args.no_vol_target:
        sizing_kwargs["vol_target_window"] = None
    elif args.vol_target_window is not None:
        sizing_kwargs["vol_target_window"] = args.vol_target_window
    report = run_xs_alpha_growth_sizing(
        profile=profile,
        unseal_holdout=args.unseal_holdout,
        **sizing_kwargs,
    )
    persist_xs_growth_sizing_report(report, xs_growth_sizing_report_path(profile))
    _logger.info(
        "xs-growth-sizing profile=%s selected_risk=%s "
        "pre_qualification_admitted=%s post_qualification_admitted=%s "
        "post_qualification_sharpe=%.4f binding_constraint=%s",
        report.profile,
        report.sizing.selected_risk,
        report.pre_scaling_qualification.admitted,
        report.qualification.admitted,
        report.qualification.sharpe,
        report.qualification.binding_constraint,
    )


def _run_xs_alpha_baseline_blend(args: argparse.Namespace) -> None:
    from src.application.research.technical.xs_alpha_baseline_blend import (
        persist_xs_alpha_baseline_blend_report,
        run_xs_alpha_baseline_blend,
        xs_baseline_blend_report_path,
    )
    from src.application.research.technical.xs_trend_screen import (
        XS_VOL_WEIGHTED_ALPHA_PROFILE_ID,
    )

    profile = args.profile or XS_VOL_WEIGHTED_ALPHA_PROFILE_ID
    if profile != XS_VOL_WEIGHTED_ALPHA_PROFILE_ID:
        raise ValueError(
            f"unknown baseline-blend profile '{profile}'; blending is restricted "
            f"to '{XS_VOL_WEIGHTED_ALPHA_PROFILE_ID}'"
        )
    report = run_xs_alpha_baseline_blend(
        profile=profile, unseal_holdout=args.unseal_holdout,
    )
    persist_xs_alpha_baseline_blend_report(report, xs_baseline_blend_report_path())
    _logger.info(
        "xs-baseline-blend profile=%s blend_weight=%.4f "
        "pre_qualification_admitted=%s post_qualification_admitted=%s "
        "post_qualification_sharpe=%.4f binding_constraint=%s",
        report.profile,
        report.blend_weight,
        report.pre_blend_qualification.admitted,
        report.qualification.admitted,
        report.qualification.sharpe,
        report.qualification.binding_constraint,
    )


def _run_xs_alpha_baseline_blend_sized(args: argparse.Namespace) -> None:
    from src.application.research.technical.xs_alpha_baseline_blend import (
        persist_xs_alpha_baseline_blend_sized_report,
        run_xs_alpha_baseline_blend_sized,
        xs_baseline_blend_sized_report_path,
    )
    from src.application.research.technical.xs_trend_screen import (
        XS_VOL_WEIGHTED_ALPHA_PROFILE_ID,
    )

    profile = args.profile or XS_VOL_WEIGHTED_ALPHA_PROFILE_ID
    if profile != XS_VOL_WEIGHTED_ALPHA_PROFILE_ID:
        raise ValueError(
            f"unknown baseline-blend profile '{profile}'; blending is restricted "
            f"to '{XS_VOL_WEIGHTED_ALPHA_PROFILE_ID}'"
        )
    report = run_xs_alpha_baseline_blend_sized(unseal_holdout=args.unseal_holdout)
    persist_xs_alpha_baseline_blend_sized_report(
        report, xs_baseline_blend_sized_report_path(),
    )
    _logger.info(
        "xs-baseline-blend-sized profile=%s blend_weight=%.4f selected_risk=%s "
        "pre_qualification_admitted=%s post_qualification_admitted=%s "
        "post_qualification_sharpe=%.4f binding_constraint=%s",
        report.profile,
        report.blend_weight,
        report.sizing.selected_risk,
        report.pre_blend_qualification.admitted,
        report.qualification.admitted,
        report.qualification.sharpe,
        report.sizing.binding_constraint,
    )

def _run_xs_alpha_baseline_blend_joint(args: argparse.Namespace) -> None:
    from src.application.research.technical.xs_alpha_baseline_blend import (
        persist_xs_alpha_baseline_blend_joint_report,
        run_xs_alpha_baseline_blend_joint,
        xs_baseline_blend_joint_report_path,
    )
    from src.application.research.technical.xs_trend_screen import (
        XS_VOL_WEIGHTED_ALPHA_PROFILE_ID,
    )

    profile = args.profile or XS_VOL_WEIGHTED_ALPHA_PROFILE_ID
    if profile != XS_VOL_WEIGHTED_ALPHA_PROFILE_ID:
        raise ValueError(
            f"unknown baseline-blend profile '{profile}'; blending is restricted "
            f"to '{XS_VOL_WEIGHTED_ALPHA_PROFILE_ID}'"
        )
    report = run_xs_alpha_baseline_blend_joint(
        xs_alpha_weight=args.xs_alpha_weight,
        leverage_scale=args.leverage_scale,
        unseal_holdout=args.unseal_holdout,
    )
    persist_xs_alpha_baseline_blend_joint_report(
        report, xs_baseline_blend_joint_report_path(),
    )
    _logger.info(
        "xs-baseline-blend-joint profile=%s xs_alpha_weight=%.4f leverage_scale=%.4f "
        "pre_qualification_admitted=%s post_qualification_admitted=%s "
        "post_qualification_sharpe=%.4f binding_constraint=%s",
        report.profile,
        report.xs_alpha_weight,
        report.leverage_scale,
        report.pre_blend_qualification.admitted,
        report.qualification.admitted,
        report.qualification.sharpe,
        report.qualification.binding_constraint,
    )


def add_single_technical_commands(run_sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Attach the ``research run single technical`` subcommands."""
    technical = run_sub.add_parser(
        "technical", help="Run one sealed technical-expert candidate screen",
    )
    technical.add_argument("--candidate-id", required=True)
    technical.add_argument("--symbol", default="BTCUSDT")
    technical.add_argument("--start", default=None)
    technical.add_argument("--end", default=None)
    technical.add_argument("--initial-equity", type=float, default=10_000.0)
    technical.add_argument("--unseal-holdout", action="store_true", default=False)
    technical.add_argument("--no-log-run", action="store_true", default=False)
    technical.set_defaults(handler=_run_technical_expert)

    trend_screen = run_sub.add_parser(
        "trend-screen", help="Run the pre-registered 450-cell baseline-gate trend screen",
    )
    trend_screen.add_argument("--profile", default=None)
    trend_screen.add_argument("--start", default=None)
    trend_screen.add_argument("--end", default=None)
    trend_screen.add_argument("--no-log-run", action="store_true", default=False)
    trend_screen.set_defaults(handler=_run_trend_screen)

    xs_screen = run_sub.add_parser(
        "xs-screen",
        help="Run the XS dollar-neutral composite trend screen (xs_neutral_composite_v1)",
    )
    xs_screen.add_argument("--profile", default=None)
    xs_screen.add_argument("--start", default=None)
    xs_screen.add_argument("--end", default=None)
    xs_screen.add_argument("--unseal-holdout", action="store_true", default=False)
    xs_screen.add_argument("--no-log-run", action="store_true", default=False)
    xs_screen.add_argument("--no-growth-sizing", action="store_true", default=False)
    xs_screen.set_defaults(handler=_run_xs_trend_screen)

    xs_growth_sizing = run_sub.add_parser(
        "xs-growth-sizing",
        help="Select a growth-optimal gross-leverage overlay for an admitted XS alpha profile",
    )
    xs_growth_sizing.add_argument("--profile", default=None)
    xs_growth_sizing.add_argument("--start", default=None)
    xs_growth_sizing.add_argument("--end", default=None)
    xs_growth_sizing.add_argument("--unseal-holdout", action="store_true", default=False)
    xs_growth_sizing.add_argument("--no-log-run", action="store_true", default=False)
    xs_growth_sizing.add_argument("--vol-target-window", type=int, default=None)
    xs_growth_sizing.add_argument("--no-vol-target", action="store_true", default=False)
    xs_growth_sizing.set_defaults(handler=_run_xs_alpha_growth_sizing)

    xs_baseline_blend = run_sub.add_parser(
        "xs-baseline-blend",
        help="Blend xs_alpha_vol_weighted_v6 with the frozen Donchian baseline",
    )
    xs_baseline_blend.add_argument("--profile", default=None)
    xs_baseline_blend.add_argument("--start", default=None)
    xs_baseline_blend.add_argument("--end", default=None)
    xs_baseline_blend.add_argument("--unseal-holdout", action="store_true", default=False)
    xs_baseline_blend.add_argument("--no-log-run", action="store_true", default=False)
    xs_baseline_blend.set_defaults(handler=_run_xs_alpha_baseline_blend)

    xs_baseline_blend_sized = run_sub.add_parser(
        "xs-baseline-blend-sized",
        help=(
            "Blend xs_alpha_vol_weighted_v6 with the frozen Donchian baseline "
            "(worst-year-robust weight) and apply growth-optimal gross leverage"
        ),
    )
    xs_baseline_blend_sized.add_argument("--profile", default=None)
    xs_baseline_blend_sized.add_argument("--start", default=None)
    xs_baseline_blend_sized.add_argument("--end", default=None)
    xs_baseline_blend_sized.add_argument("--unseal-holdout", action="store_true", default=False)
    xs_baseline_blend_sized.add_argument("--no-log-run", action="store_true", default=False)
    xs_baseline_blend_sized.set_defaults(handler=_run_xs_alpha_baseline_blend_sized)

    from src.application.research.technical.xs_alpha_baseline_blend import (
        _JOINT_LEVERAGE_SCALE,
        _JOINT_XS_ALPHA_WEIGHT,
    )

    xs_baseline_blend_joint = run_sub.add_parser(
        "xs-baseline-blend-joint",
        help=(
            "Blend xs_alpha_vol_weighted_v6 with the frozen Donchian baseline "
            "at the jointly-searched sleeve weight and gross leverage"
        ),
    )
    xs_baseline_blend_joint.add_argument("--profile", default=None)
    xs_baseline_blend_joint.add_argument("--start", default=None)
    xs_baseline_blend_joint.add_argument("--end", default=None)
    xs_baseline_blend_joint.add_argument("--unseal-holdout", action="store_true", default=False)
    xs_baseline_blend_joint.add_argument("--no-log-run", action="store_true", default=False)
    xs_baseline_blend_joint.add_argument(
        "--xs-alpha-weight", type=float, default=_JOINT_XS_ALPHA_WEIGHT,
    )
    xs_baseline_blend_joint.add_argument(
        "--leverage-scale", type=float, default=_JOINT_LEVERAGE_SCALE,
    )
    xs_baseline_blend_joint.set_defaults(handler=_run_xs_alpha_baseline_blend_joint)
