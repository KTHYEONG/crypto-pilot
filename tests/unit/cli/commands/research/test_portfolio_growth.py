from __future__ import annotations

import argparse

from src.cli.commands.research.portfolio_growth import add_portfolio_growth_commands


def test_portfolio_growth_defaults_to_rebalance_bars_3_and_zero_trade_band() -> None:
    # GEV3-01-REBALANCE-DEFAULT: the single-axis sweep (growth_engine_v3.md section 1)
    # shows net Sharpe improves monotonically from rebalance_bars=1 through 12 with
    # no sharp collapse -- a wide plateau, not a spike -- and bar=1 is the uniquely
    # inferior point, so the default moves to 3. no_trade_band stays 0.0 (opt-in).
    sub = argparse.ArgumentParser().add_subparsers()
    add_portfolio_growth_commands(sub)
    parser = sub.choices["growth"]
    defaults = {action.dest: action.default for action in parser._actions}
    assert defaults["rebalance_bars"] == 3
    assert defaults["no_trade_band"] == 0.0


def test_portfolio_growth_rebalance_bars_flag_remains_overridable() -> None:
    # GEV3-01-REBALANCE-DEFAULT: only the default changes; the flag stays overridable.
    sub = argparse.ArgumentParser().add_subparsers()
    add_portfolio_growth_commands(sub)
    parser = sub.choices["growth"]
    args = parser.parse_args(["--rebalance-bars", "6", "--no-trade-band", "0.05"])
    assert args.rebalance_bars == 6
    assert args.no_trade_band == 0.05
