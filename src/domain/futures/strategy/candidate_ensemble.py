from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.domain.futures.strategy.candidate_contracts import CandidateModelOutput, EdgeSource
from src.domain.futures.strategy.config import CandidateStrategyConfig


@dataclass(frozen=True, slots=True)
class RegimeConditionalEnsemble:
    """Shrunk archetype-regime edge estimates fit on train-window events only."""

    cell_mu_bps: dict[tuple[str, int], float]
    cell_q10_bps: dict[tuple[str, int], float]
    global_mu_bps: float
    global_q10_bps: float


def _require_ensemble_columns(events: pd.DataFrame) -> None:
    required = {"archetype", "entry_regime_code", "net_return_bps"}
    missing = sorted(required.difference(events.columns))
    if missing:
        raise ValueError(f"missing required ensemble columns: {missing}")


def fit_regime_conditional_ensemble(
    *,
    train_events: pd.DataFrame,
    cfg: CandidateStrategyConfig,
) -> RegimeConditionalEnsemble:
    """Fit per-cell shrinkage estimates from train-window events."""
    if train_events.empty:
        return RegimeConditionalEnsemble({}, {}, 0.0, 0.0)

    _require_ensemble_columns(train_events)
    frame = train_events.loc[:, ["archetype", "entry_regime_code", "net_return_bps"]].copy()
    frame["archetype"] = frame["archetype"].astype(str)
    frame["entry_regime_code"] = pd.to_numeric(frame["entry_regime_code"], errors="coerce")
    frame["net_return_bps"] = pd.to_numeric(frame["net_return_bps"], errors="coerce")
    frame = frame.loc[
        frame["archetype"].ne("")
        & frame["entry_regime_code"].notna()
        & frame["net_return_bps"].notna()
    ].copy()
    if frame.empty:
        return RegimeConditionalEnsemble({}, {}, 0.0, 0.0)

    frame["entry_regime_code"] = frame["entry_regime_code"].astype(int)
    global_edge = frame["net_return_bps"].to_numpy(dtype=np.float64, copy=False)
    global_mu_bps = float(np.mean(global_edge))
    global_q10_bps = float(np.percentile(global_edge, 10))
    shrinkage_k = float(cfg.ensemble_shrinkage_k)
    cell_mu_bps: dict[tuple[str, int], float] = {}
    cell_q10_bps: dict[tuple[str, int], float] = {}
    for (archetype, regime_code), group in frame.groupby(["archetype", "entry_regime_code"], sort=False):
        values = group["net_return_bps"].to_numpy(dtype=np.float64, copy=False)
        obs = float(values.shape[0])
        mean_bps = float(np.mean(values))
        q10_bps = float(np.percentile(values, 10))
        weight = obs / (obs + shrinkage_k)
        key = (str(archetype), int(regime_code))
        cell_mu_bps[key] = (weight * mean_bps) + ((1.0 - weight) * global_mu_bps)
        cell_q10_bps[key] = (weight * q10_bps) + ((1.0 - weight) * global_q10_bps)
    return RegimeConditionalEnsemble(
        cell_mu_bps=cell_mu_bps,
        cell_q10_bps=cell_q10_bps,
        global_mu_bps=global_mu_bps,
        global_q10_bps=global_q10_bps,
    )


def predict_regime_conditional_ensemble(
    *,
    model: RegimeConditionalEnsemble,
    oos_events: pd.DataFrame,
) -> CandidateModelOutput:
    """Lookup shrunk edge estimates for OOS events."""
    if oos_events.empty:
        return CandidateModelOutput(
            events=oos_events.copy(),
            p_pass=np.zeros((0,), dtype=np.float64),
            gate_enabled=False,
            gate_threshold=0.0,
            edge_source=EdgeSource.PRIOR_ONLY,
            expected_net_bps=np.zeros((0,), dtype=np.float64),
            q10_net_bps=np.zeros((0,), dtype=np.float64),
            q90_net_bps=np.zeros((0,), dtype=np.float64),
            selection_score=np.zeros((0,), dtype=np.float64),
            validation_diagnostics={"allocation_backend": "ensemble_b0"},
        )

    required = {"archetype", "entry_regime_code"}
    missing = sorted(required.difference(oos_events.columns))
    if missing:
        raise ValueError(f"missing required OOS ensemble columns: {missing}")

    event_frame = oos_events.reset_index(drop=True).copy()
    mu_net_decision_bps = np.empty(len(event_frame), dtype=np.float64)
    q10_net_bps = np.empty(len(event_frame), dtype=np.float64)
    for idx, row in enumerate(event_frame.itertuples(index=False), start=0):
        key = (str(getattr(row, "archetype", "")), int(getattr(row, "entry_regime_code", 0)))
        mu_net_decision_bps[idx] = model.cell_mu_bps.get(key, model.global_mu_bps)
        q10_net_bps[idx] = model.cell_q10_bps.get(key, model.global_q10_bps)

    p_pass = np.ones(len(event_frame), dtype=np.float64)
    return CandidateModelOutput(
        events=event_frame,
        p_pass=p_pass,
        gate_enabled=False,
        gate_threshold=0.0,
        edge_source=EdgeSource.PRIOR_ONLY,
        expected_net_bps=mu_net_decision_bps,
        q10_net_bps=q10_net_bps,
        q90_net_bps=mu_net_decision_bps,
        selection_score=mu_net_decision_bps.copy(),
        validation_diagnostics={"allocation_backend": "ensemble_b0"},
    )
