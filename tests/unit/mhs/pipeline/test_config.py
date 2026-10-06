"""Tests for MhsDiagnosticRequest: D1 fix verification."""

from __future__ import annotations

import pytest

import dataclasses

from src.cli.dataclass_args import explicit_field_values
from src.mhs.contracts import MhsDiagnosticRequest
from src.mhs.pipeline.config import resolve_cli_request

BASE = ["research", "run", "portfolio", "mhs-horizon-diagnostic"]


def _parse(extra: list[str]):
    """Parse BASE + extra via the production root parser."""
    from src.cli.main import build_root_parser

    return build_root_parser().parse_args([*BASE, *extra])


def _explicit(extra: list[str]) -> dict:
    """Explicitly stated request field values for BASE + extra."""
    args = _parse(extra)
    return explicit_field_values(MhsDiagnosticRequest, args)


def _resolve(extra: list[str]) -> MhsDiagnosticRequest:
    """Resolve BASE + extra to the effective request."""
    args = _parse(extra)
    return resolve_cli_request(explicit_field_values(MhsDiagnosticRequest, args))


def test_config_defaults_match_cli_derived():
    """SCENARIO_ANALYSIS_ARCHITECTURE_08: MhsDiagnosticRequest() defaults must match
    what the CLI handler currently derives at lines 28-52.

    Concretely: committee_capital=True, committee_regime_adaptive_tranche=True,
    funding_carry_sleeve=True, committee_member_set="flow_momentum",
    committee_target_gross=0.92.
    """
    config = MhsDiagnosticRequest()
    d = dataclasses.asdict(config)
    assert d["committee_capital"] is True
    assert d["committee_regime_adaptive_tranche"] is True
    assert d["funding_carry_sleeve"] is True
    assert d["committee_member_set"] == "flow_momentum"
    assert d["committee_target_gross"] == 0.92
    assert d["funding_carry_weight"] == 0.3


def test_member_set_values():
    """Registered committee member sets have no _v<N> suffix (I_NOVERSION)."""
    from src.mhs.params import COMMITTEE_MEMBER_SETS

    assert set(COMMITTEE_MEMBER_SETS) == {"risk_premia", "flow_momentum"}


def test_from_namespace_no_arg_cli_matches_bare_config():
    """SCENARIO_ANALYSIS_ARCHITECTURE_08: dataclasses.asdict(MhsDiagnosticRequest())
    == dataclasses.asdict(resolve_cli_request(explicit_field_values(<no-arg CLI parse>))).

    Pins the exact CLI defaults from src/cli/commands/research/mhs.py
    (committee_capital=True, committee_regime_adaptive_tranche=True,
    funding_carry_sleeve=True, committee_member_set=flow_momentum,
    committee_target_gross=0.92, pnl_vol_target_mode=exante_target) as the
    single source of truth the dataclass must reproduce (D1).
    """
    explicit = _explicit([])
    assert explicit == {}
    from_cli = dataclasses.asdict(_resolve([]))
    bare = dataclasses.asdict(MhsDiagnosticRequest())
    assert from_cli == bare


def test_from_namespace_respects_negate_flags():
    """--no-committee-capital cascades to regime-adaptive-tranche and funding-carry-sleeve."""
    config = _resolve(["--no-committee-capital"])
    assert config.committee_capital is False
    assert config.committee_regime_adaptive_tranche is False
    assert config.funding_carry_sleeve is False
    assert config.funding_carry_weight == 0.0


def test_from_namespace_fold_safe_horizon_flag_maps_to_selection_field():
    """--fold-safe-horizon maps to fold_safe_horizon_selection (name divergence, D1)."""
    explicit = _explicit(["--fold-safe-horizon"])
    assert explicit == {"fold_safe_horizon_selection": True}
    config = _resolve(["--fold-safe-horizon"])
    assert config.fold_safe_horizon_selection is True


# SCENARIO_GROWTH_ENVELOPE_GOLDEN_IDENTITY_PRESERVED
def test_config_defaults_growth_envelope_and_attribution():
    """growth_envelope defaults to growth_extreme_budgeted: the budgeted twin
    of the 2026-08-23 main-logic rung (ADR_20260823_MHS_KELLY_TWO_SIDED_SIZING)
    with the identical leverage_ceiling, so the deployed exposure is unchanged;
    attribution stays False."""
    config = MhsDiagnosticRequest()
    d = dataclasses.asdict(config)
    assert d["growth_envelope"] == "growth_extreme_budgeted"
    assert d["committee_member_attribution"] is False


def test_config_defaults_growth_budget_main_logic():
    """Main-logic default (2026-08-22): growth_budget mode + evidence weighting on."""
    config = MhsDiagnosticRequest()
    d = dataclasses.asdict(config)
    assert d["pnl_vol_target_mode"] == "growth_budget"
    assert d["committee_evidence_weighting"] is True


def test_from_namespace_committee_evidence_weighting_cascades_with_capital():
    """--no-committee-capital also disables evidence weighting (gated, like funding_carry_sleeve)."""
    assert _resolve([]).committee_evidence_weighting is True

    assert _resolve(["--no-committee-evidence-weighting"]).committee_evidence_weighting is False

    assert _resolve(["--no-committee-capital"]).committee_evidence_weighting is False


def test_from_namespace_growth_envelope_flag():
    """--growth-envelope maps to growth_envelope field."""
    explicit = _explicit(["--growth-envelope", "balanced"])
    assert explicit == {"growth_envelope": "balanced"}
    config = _resolve(["--growth-envelope", "balanced"])
    assert config.growth_envelope == "balanced"


def test_from_namespace_committee_member_attribution_flag():
    """--committee-member-attribution maps to committee_member_attribution field."""
    explicit = _explicit(["--committee-member-attribution"])
    assert explicit == {"committee_member_attribution": True}
    config = _resolve(["--committee-member-attribution"])
    assert config.committee_member_attribution is True


def test_golden_identity_preserved():
    """I-CONFIG: MhsDiagnosticRequest() and a no-arg CLI invocation produce identical dicts."""
    from_cli = dataclasses.asdict(_resolve([]))
    bare = dataclasses.asdict(MhsDiagnosticRequest())
    # New fields must be identical
    assert from_cli["growth_envelope"] == bare["growth_envelope"]
    assert from_cli["committee_member_attribution"] == bare["committee_member_attribution"]


# SCENARIO_MHS_EXPOSURE_CEILING_06
def test_scenario_mhs_exposure_ceiling_06_two_sided_default_flipped_universe_wired():
    """exposure_scale_two_sided flips to True at the MhsDiagnosticRequest (CLI
    effective-default owner) layer only; --no-exposure-scale-two-sided opts
    back out; --execution-universe-size exposes the roster breadth field."""
    assert MhsDiagnosticRequest().exposure_scale_two_sided is True
    assert dataclasses.asdict(_resolve([])) == dataclasses.asdict(MhsDiagnosticRequest())
    assert _resolve(["--no-exposure-scale-two-sided"]).exposure_scale_two_sided is False
    assert _resolve(["--execution-universe-size", "60"]).execution_universe_size == 60
    assert _resolve([]).execution_universe_size == 60


# SCENARIO_MHS_KELLY_TWO_SIDED_06
def test_scenario_mhs_kelly_two_sided_06_universe_default_promotion() -> None:
    """Breadth default is 60 on the single request type, matching the CLI;
    an explicit --execution-universe-size still overrides."""
    from src.mhs.params import CLI_EXECUTION_UNIVERSE_SIZE_DEFAULT

    assert MhsDiagnosticRequest().execution_universe_size == 60
    assert MhsDiagnosticRequest().execution_universe_size == CLI_EXECUTION_UNIVERSE_SIZE_DEFAULT

    args = _parse([])
    assert not hasattr(args, "execution_universe_size")
    assert "execution_universe_size" not in _explicit([])
    assert _resolve([]).execution_universe_size == 60
    narrow = _parse(["--execution-universe-size", "30"])
    assert narrow.execution_universe_size == 30
    assert _explicit(["--execution-universe-size", "30"]) == {"execution_universe_size": 30}
    assert _resolve(["--execution-universe-size", "30"]).execution_universe_size == 30


# SCENARIO_MHS_CONSTANT_RISK_CLI_AND_CONFIG_PARITY
def test_constant_risk_cli_and_config_parity():
    """--pnl-vol-target-mode constant_risk parses through and keeps the
    two-sided scaling default ON; the no-arg effective default stays
    growth_budget (A2: opt-in mode, defaults unchanged)."""
    config = _resolve(["--pnl-vol-target-mode", "constant_risk"])
    assert config.pnl_vol_target_mode == "constant_risk"
    assert config.exposure_scale_two_sided is True
    assert MhsDiagnosticRequest().pnl_vol_target_mode == "growth_budget"


# SCENARIO_MHS_SELECTION_EXEC_DEFAULT_UNCHANGED_01
def test_scenario_mhs_selection_exec_default_unchanged_01() -> None:
    """MhsDiagnosticRequest() and a no-arg CLI parse stay identical, both resolving
    final_oos_2026h1=False and data_policy='zombie_mask_v1' as the ONLY keys added
    to the pre-spec field set, plus the sealed committee_tranche_count field."""
    pre_spec_fields = frozenset({
        "start", "end", "partition", "data_root",
        "execution_timeframe", "execution_universe_size", "max_rss_bytes",
        "log_run", "touch_diagnostic", "ladder_diagnostic",
        "peg_chase_diagnostic", "liquidity_cost_model",
        "passive_timeout_minutes", "discovery_gate",
        "discovery_gate_adjusted_net_t", "discovery_gate_regime_scaled_net_t",
        "fold_safe_horizon_selection", "crash_regime_tilt_alpha",
        "slow_book_mode", "fast_book_mode", "rebalance_filter",
        "beta_neutralize", "ensemble_signal", "trend_efficiency_overlay",
        "pnl_vol_target", "pnl_vol_target_mode", "trend_sleeve",
        "trend_sleeve_gross", "multi_feature_book", "committee_book",
        "committee_kelly_sizing", "committee_growth_diagnostic",
        "committee_capital", "committee_member_set",
        "committee_tranche_smoothing", "committee_regime_adaptive_tranche",
        "committee_target_gross", "committee_evidence_weighting",
        "funding_carry_sleeve", "funding_carry_weight",
        "execution_coverage_gate",
        "exposure_scale_two_sided", "exposure_drawdown_brake", "ram_guard",
        "growth_envelope", "committee_member_attribution",
    })
    assert _explicit([]) == {}
    bare = dataclasses.asdict(MhsDiagnosticRequest())
    from_cli = dataclasses.asdict(_resolve([]))
    assert from_cli == bare
    assert bare["final_oos_2026h1"] is False
    assert bare["data_policy"] == "zombie_mask_v1"
    assert bare["input_manifest_path"] is None
    assert bare["forward_execution_quality_dir"] is None
    assert bare["forward_strategy_digest"] is None
    assert bare["name_drift_trim"] is False
    assert set(bare) == pre_spec_fields | {"final_oos_2026h1", "data_policy", "input_manifest_path", "forward_execution_quality_dir", "forward_strategy_digest", "forward_registration_digest", "name_drift_trim", "committee_tranche_count", "placebo_diagnostic", "phase_diagnostic", "signal_48h_diagnostic", "bootstrap_ci_diagnostic", "reference_books_diagnostic", "patient_reference_diagnostic"}
    assert bare["placebo_diagnostic"] is False
    assert bare["phase_diagnostic"] is False
    assert bare["signal_48h_diagnostic"] is False
    assert bare["bootstrap_ci_diagnostic"] is False
    assert bare["reference_books_diagnostic"] is False
    assert bare["patient_reference_diagnostic"] is False


def test_mhs_run_config_data_policy_defaults_legacy_and_cli_flag() -> None:
    import dataclasses

    from src.mhs.contracts import MhsDiagnosticRequest

    # Given: 기본 설정과 --data-policy 지정 CLI
    assert MhsDiagnosticRequest().data_policy == "zombie_mask_v1"

    # When
    explicit = _explicit(["--data-policy", "legacy"])
    assert explicit == {"data_policy": "legacy"}
    config = _resolve(["--data-policy", "legacy"])
    request = MhsDiagnosticRequest(**dataclasses.asdict(config))

    # Then
    assert config.data_policy == "legacy"
    assert request.data_policy == "legacy"
    assert MhsDiagnosticRequest().data_policy == "zombie_mask_v1"


def test_cli_uses_shared_data_policy_and_manifest_flags() -> None:
    import argparse
    from src.cli.commands.research.mhs import add_mhs_commands
    parser = argparse.ArgumentParser()
    root = parser.add_subparsers(dest='root')
    research = root.add_parser('research')
    portfolio = research.add_subparsers(dest='portfolio')
    add_mhs_commands(portfolio)
    args = parser.parse_args(['research', 'mhs-horizon-diagnostic', '--input-manifest-path', 'inputs.json', '--forward-execution-quality-dir', 'quality', '--forward-strategy-digest', 'abc'])
    explicit = explicit_field_values(MhsDiagnosticRequest, args)
    config = resolve_cli_request(explicit)
    assert config.data_policy == 'zombie_mask_v1'
    assert config.input_manifest_path == 'inputs.json'
    assert config.forward_execution_quality_dir == 'quality'
    assert config.forward_strategy_digest == 'abc'

def test_name_drift_trim_cli_flag_maps_to_config_and_request() -> None:
    import dataclasses

    from src.mhs.contracts import MhsDiagnosticRequest

    default_config = _resolve([])
    assert default_config.name_drift_trim is False
    assert MhsDiagnosticRequest().name_drift_trim is False

    config = _resolve(["--name-drift-trim"])
    assert config.name_drift_trim is True
    assert MhsDiagnosticRequest(**dataclasses.asdict(config)).name_drift_trim is True

    field = next(f for f in dataclasses.fields(MhsDiagnosticRequest) if f.name == "name_drift_trim")
    assert field.default is False
    assert field.metadata["flag"] == "--name-drift-trim"



def test_config_forward_registration_default_none():

    assert MhsDiagnosticRequest().forward_registration_digest is None


def test_from_namespace_is_pure_and_idempotent() -> None:
    import copy

    from src.cli.main import build_root_parser

    base = ["research", "run", "portfolio", "mhs-horizon-diagnostic"]
    args = build_root_parser().parse_args([*base, "--pnl-vol-target-mode", "median_relative"])
    before = copy.deepcopy(vars(args))
    explicit = explicit_field_values(MhsDiagnosticRequest, args)
    explicit_before = copy.deepcopy(explicit)
    first = resolve_cli_request(explicit)
    second = resolve_cli_request(explicit)
    assert vars(args) == before
    assert explicit == explicit_before
    assert first == second
    assert first.exposure_scale_two_sided is False


def test_from_namespace_no_arg_parity_preserved() -> None:
    import dataclasses

    assert dataclasses.asdict(_resolve([])) == dataclasses.asdict(MhsDiagnosticRequest())


def test_from_namespace_rejects_explicit_inert_flags() -> None:
    import re

    import pytest

    from src.cli.main import build_root_parser

    base = ["research", "run", "portfolio", "mhs-horizon-diagnostic"]
    cases = [
        (["--no-committee-capital", "--committee-member-set", "risk_premia"], "--committee-member-set is inert unless committee_capital=True"),
        (["--no-committee-capital", "--funding-carry-weight", "0.25"], "--funding-carry-weight is inert unless funding_carry_sleeve=True"),
        (["--no-funding-carry-sleeve", "--funding-carry-weight", "0.3"], "--funding-carry-weight is inert unless funding_carry_sleeve=True"),
        (["--no-committee-capital", "--committee-target-gross", "1.2"], "--committee-target-gross is inert unless committee_capital=True"),
        (["--no-committee-regime-adaptive-tranche", "--committee-tranche-count", "5"], "--committee-tranche-count is inert unless committee_tranche_smoothing or committee_regime_adaptive_tranche"),
        (["--trend-sleeve-gross", "0.0"], "--trend-sleeve-gross is inert unless trend_sleeve=True"),
    ]
    for extra, message in cases:
        args = build_root_parser().parse_args([*base, *extra])
        explicit = explicit_field_values(MhsDiagnosticRequest, args)
        with pytest.raises(ValueError, match=re.escape(message)):
            resolve_cli_request(explicit)


def test_mutually_exclusive_gross_flags() -> None:
    import pytest

    from src.cli.main import build_root_parser

    base = ["research", "run", "portfolio", "mhs-horizon-diagnostic"]
    with pytest.raises(SystemExit):
        build_root_parser().parse_args([*base, "--committee-target-gross", "1.2", "--no-committee-target-gross"])


def test_capital_opt_out_yields_canonical_dependents() -> None:
    from src.mhs.params import COMMITTEE_TRANCHE_COUNT

    config = _resolve(["--no-committee-capital"])
    assert config.committee_member_set == "risk_premia"
    assert config.committee_tranche_count == COMMITTEE_TRANCHE_COUNT
    assert config.committee_target_gross is None
    assert config.funding_carry_weight == 0.0
    assert config.trend_sleeve_gross == 0.0
    assert config.committee_kelly_sizing is False
    assert config.committee_evidence_weighting is False
    assert config.committee_regime_adaptive_tranche is False
    assert config.funding_carry_sleeve is False


def test_active_explicit_values_pass_through() -> None:
    cfg = _resolve(["--committee-tranche-smoothing", "--committee-tranche-count", "7"])
    assert cfg.committee_tranche_count == 7
    cfg = _resolve(["--trend-sleeve", "--trend-sleeve-gross", "0.3"])
    assert cfg.trend_sleeve_gross == 0.3
    cfg = _resolve(["--funding-carry-weight", "0.25"])
    assert cfg.funding_carry_weight == 0.25
    cfg = _resolve(["--committee-member-set", "risk_premia"])
    assert cfg.committee_member_set == "risk_premia"
    cfg = _resolve(["--committee-target-gross", "1.2"])
    assert cfg.committee_target_gross == 1.2
    explicit = _explicit(["--no-committee-target-gross"])
    assert explicit == {"committee_target_gross": None}
    cfg = _resolve(["--no-committee-target-gross", "--no-funding-carry-sleeve"])
    assert cfg.committee_target_gross is None


@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_resolver_rejects_non_boolean_values(value: object) -> None:
    with pytest.raises(ValueError, match="beta_neutralize must be a bool"):
        resolve_cli_request({"beta_neutralize": value})


def test_resolver_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match=r"unknown CLI request fields.*execution_universe_szie"):
        resolve_cli_request({"execution_universe_szie": 8})


def test_gross_opt_out_rejects_explicit_carry_weight() -> None:
    with pytest.raises(ValueError, match="--funding-carry-weight is inert unless funding_carry_sleeve=True"):
        resolve_cli_request({"committee_target_gross": None, "funding_carry_weight": 0.25})


def test_resolver_preserves_context_fields_and_validates_partition() -> None:
    request = resolve_cli_request({"partition": "all", "data_root": "/research"})
    assert request.partition == "all"
    assert request.data_root == "/research"
    with pytest.raises(ValueError, match="unknown partition"):
        resolve_cli_request({"partition": "invalid"})
