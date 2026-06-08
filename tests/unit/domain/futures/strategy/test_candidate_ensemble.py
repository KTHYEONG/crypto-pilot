from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.domain.futures.strategy.candidate_ensemble import (
    fit_regime_conditional_ensemble,
    predict_regime_conditional_ensemble,
)
from src.domain.futures.strategy.config import CandidateStrategyConfig


def test_regime_conditional_ensemble_shrinks_small_cells_to_global() -> None:
    train_events = pd.DataFrame(
        {
            "archetype": ["trend_continuation", "trend_continuation", "mean_reversion"],
            "entry_regime_code": [0, 0, 4],
            "net_return_bps": [100.0, 100.0, -100.0],
        }
    )
    cfg = CandidateStrategyConfig(ensemble_shrinkage_k=50.0)

    model = fit_regime_conditional_ensemble(train_events=train_events, cfg=cfg)

    global_mu = (100.0 + 100.0 - 100.0) / 3.0
    shrunk_small_cell = model.cell_mu_bps[("mean_reversion", 4)]

    assert model.global_mu_bps == pytest.approx(global_mu)
    assert -100.0 < shrunk_small_cell < global_mu


def test_ensemble_predict_lookup_matches_cell_estimate() -> None:
    train_events = pd.DataFrame(
        {
            "archetype": ["trend_continuation", "trend_continuation", "mean_reversion"],
            "entry_regime_code": [0, 0, 4],
            "net_return_bps": [40.0, 60.0, 10.0],
        }
    )
    cfg = CandidateStrategyConfig(ensemble_shrinkage_k=1.0)
    model = fit_regime_conditional_ensemble(train_events=train_events, cfg=cfg)
    oos_events = pd.DataFrame(
        {
            "archetype": ["trend_continuation", "mean_reversion", "unseen"],
            "entry_regime_code": [0, 4, 9],
        }
    )

    out = predict_regime_conditional_ensemble(model=model, oos_events=oos_events)

    assert out.expected_net_bps[0] == pytest.approx(model.cell_mu_bps[("trend_continuation", 0)])
    assert out.expected_net_bps[1] == pytest.approx(model.cell_mu_bps[("mean_reversion", 4)])
    assert out.expected_net_bps[2] == pytest.approx(model.global_mu_bps)
    assert np.all(out.p_pass == 1.0)


def test_ensemble_fit_uses_train_window_only() -> None:
    train_events = pd.DataFrame(
        {
            "archetype": ["trend_continuation", "trend_continuation"],
            "entry_regime_code": [0, 0],
            "net_return_bps": [10.0, 20.0],
        }
    )
    cfg = CandidateStrategyConfig(ensemble_shrinkage_k=1.0)

    model = fit_regime_conditional_ensemble(train_events=train_events, cfg=cfg)
    out = predict_regime_conditional_ensemble(
        model=model,
        oos_events=pd.DataFrame({"archetype": ["trend_continuation"], "entry_regime_code": [7]}),
    )

    assert ("trend_continuation", 7) not in model.cell_mu_bps
    assert out.expected_net_bps[0] == pytest.approx(model.global_mu_bps)
