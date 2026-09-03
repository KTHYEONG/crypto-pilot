"""One-shot joint (discovery-only, Optuna) search over the blend's free parameters.

Dev tool (``tools/research/``, not runtime ``src/``): runs
``run_structural_search`` -- the already-built, already-tested joint TPE search
from ``tools/research/structural_tuner.py`` -- over the two parameters this
project-cycle established are free (``xs_alpha_weight`` in ``[0, 1]`` and
``leverage_scale`` in ``[1, 4]``), scored by
:func:`src.quant.technical_experts.xs_alpha_baseline_blend.discovery_reliability_score`
(LCB90 in % CAGR terms on the discovery window only -- an absolute metric, so it
genuinely responds to the leverage axis, unlike the scale-invariant
Sharpe/t_stat objectives the two prior cycles had to use sequentially).

Mirrors ``structural_tuner.py``'s own placement precedent: it lives here, not
in ``src/``, because ``optuna`` must never become a core runtime dependency of
the CLI (nothing in ``src/`` imports this module either). ``optuna`` is only
imported transitively through ``run_structural_search``'s own lazy import, so
this module stays importable -- for the contract's signature assertion and the
integration test -- without the optional ``tuning`` extra installed.

This script's only job is to produce a candidate point and its plateau audit.
It never touches qualification/holdout data and never calls
``evaluate_xs_admission``. Run it exactly once per cycle and record the printed
result; a failing plateau gate means no point from this search may be adopted.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Runnable as a plain script (``python tools/research/xs_alpha_blend_joint_search.py``):
# only ``src*`` is packaged/installed, so ``tools`` is not importable from the
# script's own directory -- put the repo root on ``sys.path`` first. Under pytest
# (``pythonpath = ["."]``) ``__package__`` is set and this block is skipped.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.quant.technical_experts.trend_screen import _load_symbol_data
from src.quant.technical_experts.xs_alpha_baseline_blend import (
    _baseline_realized_position,
)
from src.quant.technical_experts.xs_trend_screen import (
    XS_DISCOVERY_START,
    _bar_funding_series,
    _common_index,
)
from src.common.errors import DataIntegrityError
from src.quant.baseline.backtest import run_backtest
from src.quant.contracts import CostModel, StrategySpec
from src.quant.evaluation.policy import resolve_evaluation_end
from src.quant.evaluation.reliability import compute_turnover_fold_upper_bound
from src.quant.technical_experts.cross_sectional import XsAdmissionConfig
from src.quant.technical_experts.xs_alpha_baseline_blend import apply_fixed_gross_leverage, build_blended_ledger
from src.quant.technical_experts.cross_sectional import (
    XsAlphaCompositeSpec,
    XsCompositeSpec,
    build_xs_alpha_vol_weighted_weights,
    run_xs_composite_ledger,
)
from src.quant.technical_experts.trend_screen_catalog import (
    DISCOVERY_END,
    TREND_SCREEN_SYMBOLS,
)
from src.quant.technical_experts.xs_alpha_baseline_blend import (
    discovery_reliability_score,
)
from tools.research.structural_tuner import (
    StructuralSearchConfig,
    StructuralSearchResult,
    run_structural_search,
)

# Frozen 4h-calendar bars-per-year invariant, identical to
# ``xs_alpha_baseline_blend._BARS_PER_YEAR`` /
# ``GrowthSizingConfig.bars_per_year`` in ``cross_sectional``.
_BARS_PER_YEAR_CONST = 2190

# Joint search space over the blend's two free parameters. ``xs_alpha_weight``
# is the sleeve weight in ``[0, 1]`` (the grid the v8 blend already searched
# sequentially); ``leverage_scale`` is the pure-linear gross-leverage multiple
# in ``[1, 4]`` (the grid ``v8_sized`` already searched with a mismatched,
# sequential objective).
_SEARCH_SPACE: dict[str, tuple[float, float]] = {
    "xs_alpha_weight": (0.0, 1.0),
    "leverage_scale": (1.0, 4.0),
}


def _load_net_returns() -> tuple[pd.Series, pd.DataFrame, pd.Series, pd.Series]:
    """Replay v6's XS book and the frozen baseline, returning the two net-return
    legs and the two realized-weight paths on the common grid.

    Identical loader calls to ``run_xs_alpha_baseline_blend`` -- no new data
    plumbing. The discovery window is sealed here: ``unseal_holdout=False`` is
    fixed because the search itself may never read beyond discovery anyway.
    """
    end = resolve_evaluation_end(None, unseal_holdout=False)
    execution_spec = XsCompositeSpec()
    alpha_spec = XsAlphaCompositeSpec()

    data: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    for symbol in TREND_SCREEN_SYMBOLS:
        frame, funding, _fingerprint, _coverage = _load_symbol_data(
            symbol, XS_DISCOVERY_START, end,
        )
        data[symbol] = (frame, funding)

    common = _common_index([frame.index for frame, _funding in data.values()])
    if len(common) < 2:
        raise DataIntegrityError(
            "xs alpha blend joint search requires at least 2 common bars",
        )

    opens = pd.DataFrame(
        {symbol: frame["open"].reindex(common) for symbol, (frame, _funding) in data.items()},
    )
    bar_funding = pd.DataFrame(
        {
            symbol: _bar_funding_series(funding, frame.index).reindex(common)
            for symbol, (frame, funding) in data.items()
        },
    )
    closes = pd.DataFrame(
        {symbol: frame["close"].reindex(common) for symbol, (frame, _funding) in data.items()},
    )
    taker = pd.DataFrame(
        {
            symbol: frame["taker_buy_ratio"].reindex(common)
            for symbol, (frame, _funding) in data.items()
        },
    )

    weights = build_xs_alpha_vol_weighted_weights(
        closes, taker, bar_funding, opens, alpha_spec, execution_spec,
    )
    xs_equity, _xs_turnover = run_xs_composite_ledger(
        weights, opens, bar_funding, execution_spec,
    )
    xs_alpha_net = xs_equity.pct_change().fillna(0.0).rename("xs_alpha_net")
    xs_realized_weights = weights.shift(
        1 + execution_spec.execution_delay_bars,
    ).fillna(0.0)

    btc_frame, btc_funding = data["BTCUSDT"]
    baseline_result = run_backtest(
        btc_frame, StrategySpec(), CostModel(), funding_rates=btc_funding,
    )
    baseline_equity = baseline_result.equity.reindex(common).rename("baseline_equity")
    baseline_net = baseline_equity.pct_change().fillna(0.0).rename("baseline_net")
    baseline_realized_weight = _baseline_realized_position(
        baseline_result.trades, btc_frame.index, common,
    )

    return xs_alpha_net, xs_realized_weights, baseline_net, baseline_realized_weight


def run(*, max_trials: int = 30, seed: int = 0) -> StructuralSearchResult:
    """Run the joint discovery-only search once and print its plateau audit.

    Builds the objective as a closure over ``discovery_reliability_score``
    restricted to ``[XS_DISCOVERY_START, DISCOVERY_END]`` and delegates to
    ``run_structural_search`` unchanged (TPE sampler, ``max_trials`` grid-
    comparable budget cap, and the mandatory plateau-stability gate). Prints
    ``best_params``, ``best_is_score``, ``plateau_neighbor_ratio``, and
    ``plateau_passed``, then -- strictly after the search returns -- an
    informational turnover diagnostic (``discovery_worst_fold_turnover`` vs
    ``XsAdmissionConfig().turnover_max``) for the researcher to weigh; it never
    influences the search or re-enters optuna. Never calls
    ``evaluate_xs_admission`` and never touches qualification/holdout data.
    """
    xs_alpha_net, xs_realized_weights, baseline_net, baseline_realized_weight = (
        _load_net_returns()
    )

    def objective(params: dict[str, float]) -> float:
        return discovery_reliability_score(
            xs_alpha_net, xs_realized_weights, baseline_net, baseline_realized_weight,
            XS_DISCOVERY_START, DISCOVERY_END,
            params["xs_alpha_weight"], params["leverage_scale"],
            XsAdmissionConfig().round_trip_cost_rate,
        )

    result = run_structural_search(
        objective, _SEARCH_SPACE,
        StructuralSearchConfig(max_trials=max_trials, seed=seed),
    )
    print(f"best_params={result.best_params}")
    print(f"best_is_score={result.best_is_score:.6f}")
    print(f"plateau_neighbor_ratio={result.plateau_neighbor_ratio:.4f}")
    print(f"plateau_passed={result.plateau_passed}")

    # Post-search turnover diagnostic -- informational only, never a rejection
    # criterion. The real accept/reject decision stays exactly where it already
    # was: evaluate_xs_admission, called honestly on real qualification-window
    # data by the CLI orchestrator. turnover_max's own calibration could not be
    # traced to any derivation in this repo (see the contract's 'why'), so the
    # search reproduces the winning point's scaled realized-weight ledger and
    # reports the worst-fold discovery-window turnover for the researcher to
    # weigh -- not for the code to decide on.
    best = result.best_params
    xs_disc = xs_alpha_net[
        (xs_alpha_net.index >= XS_DISCOVERY_START) & (xs_alpha_net.index <= DISCOVERY_END)
    ]
    xs_w_disc = xs_realized_weights[
        (xs_realized_weights.index >= XS_DISCOVERY_START)
        & (xs_realized_weights.index <= DISCOVERY_END)
    ]
    bl_disc = baseline_net[
        (baseline_net.index >= XS_DISCOVERY_START) & (baseline_net.index <= DISCOVERY_END)
    ]
    bl_w_disc = baseline_realized_weight[
        (baseline_realized_weight.index >= XS_DISCOVERY_START)
        & (baseline_realized_weight.index <= DISCOVERY_END)
    ]
    common = (
        xs_disc.index.intersection(bl_disc.index)
        .intersection(xs_w_disc.index)
        .intersection(bl_w_disc.index)
    )
    blended_equity, blended_weights = build_blended_ledger(
        xs_disc.reindex(common).astype(np.float64),
        xs_w_disc.reindex(common),
        bl_disc.reindex(common).astype(np.float64),
        bl_w_disc.reindex(common),
        best["xs_alpha_weight"],
    )
    _scaled_net, scaled_weights = apply_fixed_gross_leverage(
        blended_equity.pct_change().fillna(0.0).rename("blended_net"),
        blended_weights,
        best["leverage_scale"],
    )
    per_bar_turnover = pd.Series(
        np.abs(np.diff(scaled_weights.to_numpy(dtype=np.float64), axis=0)).sum(axis=1),
        index=scaled_weights.index[1:],
        name="per_bar_turnover",
    )
    worst_fold_turnover = compute_turnover_fold_upper_bound(
        per_bar_turnover, _BARS_PER_YEAR_CONST,
    )
    print(
        f"discovery_worst_fold_turnover={worst_fold_turnover:.2f} "
        f"turnover_max={XsAdmissionConfig().turnover_max:.1f} "
        "(diagnostic only, not a rejection criterion -- calibration unverified "
        "in this repo, see contract 'why')"
    )
    return result


if __name__ == "__main__":
    run()
