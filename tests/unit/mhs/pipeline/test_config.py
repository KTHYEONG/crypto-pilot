"""Tests for MhsRunConfig: D1 fix verification."""

from __future__ import annotations

import dataclasses

from src.mhs.pipeline.config import MhsRunConfig, MemberSet


def test_config_defaults_match_cli_derived():
    """SCENARIO_ANALYSIS_ARCHITECTURE_08: MhsRunConfig() defaults must match
    what the CLI handler currently derives at lines 28-52.

    Concretely: committee_capital=True, committee_regime_adaptive_tranche=True,
    funding_carry_sleeve=True, committee_member_set=MemberSet.FLOW_MOMENTUM,
    committee_target_gross=0.92.
    """
    config = MhsRunConfig()
    d = dataclasses.asdict(config)
    assert d["committee_capital"] is True
    assert d["committee_regime_adaptive_tranche"] is True
    assert d["funding_carry_sleeve"] is True
    assert d["committee_member_set"] == MemberSet.FLOW_MOMENTUM
    assert d["committee_target_gross"] == 0.92
    assert d["funding_carry_weight"] == 0.3


def test_member_set_values():
    """MemberSet members have no _v<N> suffix (I_NOVERSION)."""
    assert MemberSet.RISK_PREMIA == "risk_premia"
    assert MemberSet.FLOW_MOMENTUM == "flow_momentum"
    assert len(MemberSet) == 2


def test_from_namespace_no_arg_cli_matches_bare_config():
    """SCENARIO_ANALYSIS_ARCHITECTURE_08: dataclasses.asdict(MhsRunConfig())
    == dataclasses.asdict(MhsRunConfig.from_namespace(<no-arg CLI parse>)).

    Pins the exact CLI defaults from src/cli/commands/research/mhs.py
    (committee_capital=True, committee_regime_adaptive_tranche=True,
    funding_carry_sleeve=True, committee_member_set=flow_momentum,
    committee_target_gross=0.92, pnl_vol_target_mode=exante_target) as the
    single source of truth the dataclass must reproduce (D1).
    """
    from src.cli.main import build_root_parser

    args = build_root_parser().parse_args(
        ["research", "run", "portfolio", "mhs-horizon-diagnostic"],
    )
    from_cli = dataclasses.asdict(MhsRunConfig.from_namespace(args))
    bare = dataclasses.asdict(MhsRunConfig())
    assert from_cli == bare


def test_from_namespace_respects_negate_flags():
    """--no-committee-capital cascades to regime-adaptive-tranche and funding-carry-sleeve."""
    from src.cli.main import build_root_parser

    args = build_root_parser().parse_args(
        [
            "research", "run", "portfolio", "mhs-horizon-diagnostic",
            "--no-committee-capital",
        ],
    )
    config = MhsRunConfig.from_namespace(args)
    assert config.committee_capital is False
    assert config.committee_regime_adaptive_tranche is False
    assert config.funding_carry_sleeve is False
    assert config.funding_carry_weight == 0.0


def test_from_namespace_fold_safe_horizon_flag_maps_to_selection_field():
    """--fold-safe-horizon maps to fold_safe_horizon_selection (name divergence, D1)."""
    from src.cli.main import build_root_parser

    args = build_root_parser().parse_args(
        [
            "research", "run", "portfolio", "mhs-horizon-diagnostic",
            "--fold-safe-horizon",
        ],
    )
    config = MhsRunConfig.from_namespace(args)
    assert config.fold_safe_horizon_selection is True
