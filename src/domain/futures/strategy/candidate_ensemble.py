from __future__ import annotations

from dataclasses import dataclass, field

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
    conditioning: str = "archetype_regime"
    archetype_mu_bps: dict[str, float] = field(default_factory=dict)
    archetype_q10_bps: dict[str, float] = field(default_factory=dict)
    validation_rank_ic: float = 0.0


def _require_ensemble_columns(events: pd.DataFrame) -> None:
    required = {"archetype", "entry_regime_code", "net_return_bps"}
    missing = sorted(required.difference(events.columns))
    if missing:
        raise ValueError(f"missing required ensemble columns: {missing}")


def _rank_ic_local(pred: np.ndarray, target: np.ndarray) -> float:
    """Spearman rank correlation between pred and target arrays."""
    from scipy.stats import spearmanr

    if pred.size < 2 or target.size < 2:
        return 0.0
    finite = np.isfinite(pred) & np.isfinite(target)
    if finite.sum() < 2:
        return 0.0
    rho, _ = spearmanr(pred[finite], target[finite])
    return float(rho) if np.isfinite(rho) else 0.0


def _fit_cell_means(
    frame: pd.DataFrame,
    *,
    shrinkage_k: float,
    axis: str,
) -> tuple[
    dict[tuple[str, int], float],
    dict[tuple[str, int], float],
    dict[str, float],
    dict[str, float],
    float,
    float,
]:
    """Compute shrunk cell means for given axis ('regime' or 'archetype').

    Returns cell_mu, cell_q10, arch_mu, arch_q10, global_mu, global_q10.
    'archetype' axis still fills cell_* as empty dicts.
    """
    edge = frame["net_return_bps"].to_numpy(dtype=np.float64, copy=False)
    global_mu = float(np.mean(edge))
    global_q10 = float(np.percentile(edge, 10))

    # archetype-only shrinkage
    arch_mu: dict[str, float] = {}
    arch_q10: dict[str, float] = {}
    for archetype, grp in frame.groupby("archetype", sort=False):
        vals = grp["net_return_bps"].to_numpy(dtype=np.float64, copy=False)
        n = float(vals.shape[0])
        w = n / (n + shrinkage_k)
        a = str(archetype)
        arch_mu[a] = w * float(np.mean(vals)) + (1.0 - w) * global_mu
        arch_q10[a] = w * float(np.percentile(vals, 10)) + (1.0 - w) * global_q10

    cell_mu: dict[tuple[str, int], float] = {}
    cell_q10: dict[tuple[str, int], float] = {}
    if axis == "archetype_regime":
        for (archetype, regime_code), grp in frame.groupby(
            ["archetype", "entry_regime_code"], sort=False
        ):
            vals = grp["net_return_bps"].to_numpy(dtype=np.float64, copy=False)
            n = float(vals.shape[0])
            w = n / (n + shrinkage_k)
            key = (str(archetype), int(regime_code))
            cell_mu[key] = w * float(np.mean(vals)) + (1.0 - w) * global_mu
            cell_q10[key] = w * float(np.percentile(vals, 10)) + (1.0 - w) * global_q10

    return cell_mu, cell_q10, arch_mu, arch_q10, global_mu, global_q10


def _internal_validation_rank_ic(
    frame: pd.DataFrame,
    *,
    shrinkage_k: float,
    val_fraction: float,
    axis: str,
) -> float:
    """In-fold purged validation Rank IC for the given axis.

    Splits frame by entry_idx time-order: last val_fraction is val,
    remainder minus a 1-bar purge gap is sub-fit.
    """
    if "entry_idx" not in frame.columns or frame.shape[0] < 10:
        return 0.0

    sorted_frame = frame.sort_values("entry_idx")
    n = len(sorted_frame)
    val_start = int(n * (1.0 - val_fraction))
    if val_start < 4 or (n - val_start) < 4:
        return 0.0

    val_idx_cutoff = int(sorted_frame.iloc[val_start]["entry_idx"])
    # Purge: remove 1 bar worth of boundary samples
    sub_fit = sorted_frame[sorted_frame["entry_idx"] < val_idx_cutoff - 1]
    val_set = sorted_frame[sorted_frame["entry_idx"] >= val_idx_cutoff]

    if sub_fit.shape[0] < 4 or val_set.shape[0] < 4:
        return 0.0

    _, _, arch_mu, _, global_mu, _ = _fit_cell_means(
        sub_fit, shrinkage_k=shrinkage_k, axis="archetype_only"
    )

    if axis == "archetype_regime":
        cell_mu, _, _, _, _, _ = _fit_cell_means(
            sub_fit, shrinkage_k=shrinkage_k, axis="archetype_regime"
        )

        def _predict_regime(row: pd.Series) -> float:
            key = (str(row["archetype"]), int(row["entry_regime_code"]))
            return cell_mu.get(key, arch_mu.get(str(row["archetype"]), global_mu))

        pred = val_set.apply(_predict_regime, axis=1).to_numpy(dtype=np.float64)
    else:

        def _predict_arch(row: pd.Series) -> float:
            return arch_mu.get(str(row["archetype"]), global_mu)

        pred = val_set.apply(_predict_arch, axis=1).to_numpy(dtype=np.float64)

    realized = val_set["net_return_bps"].to_numpy(dtype=np.float64, copy=False)
    return _rank_ic_local(pred, realized)


def fit_regime_conditional_ensemble(
    *,
    train_events: pd.DataFrame,
    cfg: CandidateStrategyConfig,
) -> RegimeConditionalEnsemble:
    """Fit per-cell shrinkage estimates from train-window events."""
    if train_events.empty:
        return RegimeConditionalEnsemble(
            cell_mu_bps={},
            cell_q10_bps={},
            global_mu_bps=0.0,
            global_q10_bps=0.0,
            conditioning="archetype_only",
            archetype_mu_bps={},
            archetype_q10_bps={},
            validation_rank_ic=0.0,
        )

    _require_ensemble_columns(train_events)
    frame = train_events.loc[:, ["archetype", "entry_regime_code", "net_return_bps"]].copy()
    if "entry_idx" in train_events.columns:
        frame["entry_idx"] = train_events["entry_idx"].values
    frame["archetype"] = frame["archetype"].astype(str)
    frame["entry_regime_code"] = pd.to_numeric(frame["entry_regime_code"], errors="coerce")
    frame["net_return_bps"] = pd.to_numeric(frame["net_return_bps"], errors="coerce")
    frame = frame.loc[
        frame["archetype"].ne("")
        & frame["entry_regime_code"].notna()
        & frame["net_return_bps"].notna()
    ].copy()
    if frame.empty:
        return RegimeConditionalEnsemble(
            cell_mu_bps={},
            cell_q10_bps={},
            global_mu_bps=0.0,
            global_q10_bps=0.0,
            conditioning="archetype_only",
            archetype_mu_bps={},
            archetype_q10_bps={},
            validation_rank_ic=0.0,
        )

    frame["entry_regime_code"] = frame["entry_regime_code"].astype(int)
    shrinkage_k = float(cfg.ensemble_shrinkage_k)
    val_fraction = float(cfg.ensemble_internal_val_fraction)
    conditioning_cfg = cfg.ensemble_conditioning

    # Compute archetype-only (always needed for fallback/auto)
    _, _, arch_mu, arch_q10, global_mu, global_q10 = _fit_cell_means(
        frame, shrinkage_k=shrinkage_k, axis="archetype_only"
    )

    if conditioning_cfg == "auto":
        ic_arch = _internal_validation_rank_ic(
            frame, shrinkage_k=shrinkage_k, val_fraction=val_fraction, axis="archetype_only"
        )
        ic_regime = _internal_validation_rank_ic(
            frame, shrinkage_k=shrinkage_k, val_fraction=val_fraction, axis="archetype_regime"
        )
        if ic_regime - ic_arch >= cfg.ensemble_min_conditioning_ic_gain:
            chosen = "archetype_regime"
            val_ic = ic_regime
        else:
            chosen = "archetype_only"
            val_ic = ic_arch
    elif conditioning_cfg == "archetype_regime":
        chosen = "archetype_regime"
        val_ic = _internal_validation_rank_ic(
            frame, shrinkage_k=shrinkage_k, val_fraction=val_fraction, axis="archetype_regime"
        )
    else:
        chosen = "archetype_only"
        val_ic = _internal_validation_rank_ic(
            frame, shrinkage_k=shrinkage_k, val_fraction=val_fraction, axis="archetype_only"
        )

    if chosen == "archetype_regime":
        cell_mu, cell_q10, _, _, _, _ = _fit_cell_means(
            frame, shrinkage_k=shrinkage_k, axis="archetype_regime"
        )
    else:
        cell_mu = {}
        cell_q10 = {}

    return RegimeConditionalEnsemble(
        cell_mu_bps=cell_mu,
        cell_q10_bps=cell_q10,
        global_mu_bps=global_mu,
        global_q10_bps=global_q10,
        conditioning=chosen,
        archetype_mu_bps=arch_mu,
        archetype_q10_bps=arch_q10,
        validation_rank_ic=float(val_ic),
    )


def predict_regime_conditional_ensemble(
    *,
    model: RegimeConditionalEnsemble,
    oos_events: pd.DataFrame,
    cfg: CandidateStrategyConfig | None = None,
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

    use_archetype_only = model.conditioning == "archetype_only"

    for idx, row in enumerate(event_frame.itertuples(index=False), start=0):
        arch = str(getattr(row, "archetype", ""))
        if use_archetype_only:
            mu_net_decision_bps[idx] = model.archetype_mu_bps.get(arch, model.global_mu_bps)
            q10_net_bps[idx] = model.archetype_q10_bps.get(arch, model.global_q10_bps)
        else:
            key = (arch, int(getattr(row, "entry_regime_code", 0)))
            # fallback: archetype → global
            mu_net_decision_bps[idx] = model.cell_mu_bps.get(
                key, model.archetype_mu_bps.get(arch, model.global_mu_bps)
            )
            q10_net_bps[idx] = model.cell_q10_bps.get(
                key, model.archetype_q10_bps.get(arch, model.global_q10_bps)
            )

    # mu-quality shrinkage: attenuate conviction proportional to in-fold val IC
    mu_shrinkage_lambda = 1.0
    if cfg is not None and cfg.mu_quality_shrinkage_enabled and mu_net_decision_bps.size > 0:
        lam = float(
            np.clip(model.validation_rank_ic / cfg.mu_quality_ic_full_scale, 0.0, 1.0)
        )
        mu_shrinkage_lambda = lam
        cs_mean = float(np.mean(mu_net_decision_bps))
        mu_net_decision_bps = lam * mu_net_decision_bps + (1.0 - lam) * cs_mean

    p_pass = np.ones(len(event_frame), dtype=np.float64)
    return CandidateModelOutput(
        events=oos_events.copy(),
        p_pass=p_pass,
        gate_enabled=False,
        gate_threshold=0.0,
        edge_source=EdgeSource.PRIOR_ONLY,
        expected_net_bps=mu_net_decision_bps,
        q10_net_bps=q10_net_bps,
        q90_net_bps=mu_net_decision_bps.copy(),
        selection_score=mu_net_decision_bps.copy(),
        validation_diagnostics={
            "allocation_backend": "ensemble_b0",
            "prediction_mode": "ensemble_b0",
            "conditioning": model.conditioning,
            "validation_rank_ic": model.validation_rank_ic,
            "mu_shrinkage_lambda": mu_shrinkage_lambda,
        },
    )
