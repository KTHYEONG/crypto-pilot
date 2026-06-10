from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.domain.futures.strategy.candidate_contracts import CandidateModelOutput, EdgeSource
from src.domain.futures.strategy.config import CandidateStrategyConfig
from src.domain.futures.strategy.regime_evaluation import (
    RegimeLiftProofResult,
    evaluate_regime_lift_proof,
)

_logger = logging.getLogger(__name__)


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
    # Variant-level shrunk means (3-level hierarchical prior).
    # key = "family:variant", value = shrunk mu_bps toward mode archetype-regime cell anchor.
    # Empty dict → backward compat (no variant prior).
    variant_mu_bps: dict[str, float] = field(default_factory=dict)
    # P0: Regime Lift Proof Gate fields
    conditioning_path: str = "pooled_fallback"  # "regime_conditioned" | "pooled_fallback" | "no_oos_evidence_failsafe"
    lift_proof: RegimeLiftProofResult | None = None  # None = proof 미실행 (backward compat)
    regime_oos_stability_rho: float | None = None  # 진단 전용 (C4 rho 주입, 게이팅 아님)


def _variant_key(family: str, variant: str) -> str:
    """Canonical variant identity key: 'family:variant'."""
    return f"{family}:{variant}"


def _fit_variant_means(
    frame: pd.DataFrame,
    *,
    cell_mu: dict[tuple[str, int], float],
    arch_mu: dict[str, float],
    global_mu: float,
    k_variant: float,
    min_obs: int,
    freq_n_cap: int = 0,
) -> dict[str, float]:
    """Fit variant-level shrunk means toward archetype-regime cell anchor.

    vmean_v = w_v * raw_v + (1 - w_v) * anchor_v
    anchor_v = cell_mu[(mode_arch, mode_regime)] → arch_mu[arch] → global_mu
    w_v = n_eff / (n_eff + k_variant), n_v < min_obs → w_v ≈ 0 (anchor fallback).

    Requires frame to have 'family', 'variant', 'archetype', 'entry_regime_code',
    'net_return_bps' columns. Returns empty dict if columns absent.
    """
    required = {"family", "variant", "archetype", "entry_regime_code", "net_return_bps"}
    if not required.issubset(frame.columns):
        return {}

    def _effective_n(raw_n: int) -> float:
        if freq_n_cap > 0:
            return float(min(raw_n, freq_n_cap))
        return float(raw_n)

    variant_mu: dict[str, float] = {}
    fam_col = frame["family"].astype(str)
    var_col = frame["variant"].astype(str)
    vkeys = (fam_col + ":" + var_col).values
    arch_col = frame["archetype"].astype(str).values
    regime_col = pd.to_numeric(frame["entry_regime_code"], errors="coerce").fillna(0).astype(int).values
    edge_col = pd.to_numeric(frame["net_return_bps"], errors="coerce").values

    for vkey in np.unique(vkeys):
        mask = vkeys == vkey
        n_v = int(mask.sum())
        edges = edge_col[mask]
        finite_mask = np.isfinite(edges)
        edges = edges[finite_mask]
        if len(edges) == 0:
            continue

        # mode archetype and regime for this variant
        archs = arch_col[mask][finite_mask]
        regimes = regime_col[mask][finite_mask]
        unique_archs, arch_counts = np.unique(archs, return_counts=True)
        mode_arch = str(unique_archs[np.argmax(arch_counts)])
        unique_regs, reg_counts = np.unique(regimes, return_counts=True)
        mode_regime = int(unique_regs[np.argmax(reg_counts)])

        anchor = cell_mu.get((mode_arch, mode_regime), arch_mu.get(mode_arch, global_mu))

        if n_v < min_obs:
            variant_mu[vkey] = anchor
        else:
            n_eff = _effective_n(n_v)
            w_v = n_eff / (n_eff + k_variant)
            raw_mean = float(np.mean(edges))
            variant_mu[vkey] = w_v * raw_mean + (1.0 - w_v) * anchor

    return variant_mu


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


def _predict_mu_by_event(
    events: pd.DataFrame,
    *,
    cell_mu: dict[tuple[str, int], float],
    arch_mu: dict[str, float],
    global_mu: float,
    use_archetype_only: bool,
) -> NDArray[np.float64]:
    """Project ensemble means onto an event frame for proof/prediction paths."""
    mu = np.empty(len(events), dtype=np.float64)
    for idx, row in enumerate(events.itertuples(index=False), start=0):
        arch = str(getattr(row, "archetype", ""))
        if use_archetype_only:
            mu[idx] = arch_mu.get(arch, global_mu)
            continue
        key = (arch, int(getattr(row, "entry_regime_code", 0)))
        mu[idx] = cell_mu.get(key, arch_mu.get(arch, global_mu))
    return mu


def _log_ensemble_diagnostics(
    *,
    frame: pd.DataFrame,
    global_mu: float,
    arch_mu: dict[str, float],
    val_ic: float,
    chosen: str,
    adaptive_shrinkage: bool,
    k_used: float,
) -> None:
    """Emit a table-format diagnostic log for IC sign audit.

    Helps identify anti-predictive archetypes (shrunk mean < 0) that may be
    diluting ensemble IC toward negative territory.
    """
    n_total = len(frame)
    ic_sign = "✅ POSITIVE" if val_ic > 0 else "❌ NEGATIVE"

    header = (
        "\n[ENSEMBLE DIAGNOSTICS] ─────────────────────────────────────────────\n"
        f"| {'Metric':<28} | {'Value':<30} |\n"
        f"| {'─'*28} | {'─'*30} |\n"
        f"| {'N events (train)':<28} | {n_total:<30} |\n"
        f"| {'Global mu (bps)':<28} | {global_mu:<30.3f} |\n"
        f"| {'Validation Rank IC':<28} | {val_ic:<30.4f} |\n"
        f"| {'IC sign':<28} | {ic_sign:<30} |\n"
        f"| {'Conditioning chosen':<28} | {chosen:<30} |\n"
        f"| {'Adaptive shrinkage':<28} | {adaptive_shrinkage!s:<30} |\n"
        f"| {'k_used (max/fixed)':<28} | {k_used:<30.1f} |\n"
        "├─────────────────────────────────────────────────────────────────────\n"
        f"| {'Archetype':<30} | {'Shrunk mu (bps)':<16} | {'Sign':<8} | {'N':<6} |\n"
        f"| {'─'*30} | {'─'*16} | {'─'*8} | {'─'*6} |\n"
    )
    rows = []
    for arch, mu_val in sorted(arch_mu.items()):
        n_arch = int((frame["archetype"] == arch).sum())
        sign_flag = "✅" if mu_val >= 0.0 else "❌ ANTI"
        rows.append(f"| {arch:<30} | {mu_val:<16.3f} | {sign_flag:<8} | {n_arch:<6} |")
    footer = "─────────────────────────────────────────────────────────────────────"

    _logger.info("%s%s\n%s", header, "\n".join(rows), footer)


def _compute_eb_shrinkage_k(
    cell_means: list[float],
    cell_vars: list[float],
    k_max: float,
) -> float:
    """Estimate k_eff via Empirical-Bayes James-Stein principle.

    k_eff = mean_within_var / between_var.
    - Large between_var (cells are genuinely distinct) → small k → trust cell means.
    - Small between_var (cells are homogeneous) → large k → shrink toward global.
    Clipped to [0, k_max].
    """
    if len(cell_means) < 2:
        return k_max
    mean_within = float(np.mean(cell_vars)) if cell_vars else 0.0
    grand = float(np.mean(cell_means))
    between = float(np.mean([(m - grand) ** 2 for m in cell_means]))
    if between < 1e-12:
        return k_max
    k_eff = mean_within / between
    return float(np.clip(k_eff, 0.0, k_max))


def _fit_cell_means(
    frame: pd.DataFrame,
    *,
    shrinkage_k: float,
    axis: str,
    adaptive_shrinkage: bool = False,
    shrinkage_k_max: float = 50.0,
    freq_n_cap: int = 0,
    min_cell_edge_floor_bps: float = 0.0,
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

    Args:
        adaptive_shrinkage: If True, derives k_eff from between/within cell variance
            (James-Stein EB) instead of using fixed shrinkage_k.
        shrinkage_k_max: Upper bound for adaptive k_eff.
        freq_n_cap: Clip n to this before computing w=n/(n+k); 0=disabled.
            Prevents high-frequency noise cells from dominating global pull.
        min_cell_edge_floor_bps: Cell means below this floor are set to 0.0
            (no-prediction rather than negative allocation).
    """
    edge = frame["net_return_bps"].to_numpy(dtype=np.float64, copy=False)
    global_mu = float(np.mean(edge))
    global_q10 = float(np.percentile(edge, 10))

    def _effective_n(raw_n: float) -> float:
        if freq_n_cap > 0:
            return float(min(raw_n, freq_n_cap))
        return raw_n

    # archetype-only shrinkage
    arch_mu: dict[str, float] = {}
    arch_q10: dict[str, float] = {}

    # Collect raw stats for EB estimation if adaptive
    arch_raw_means: list[float] = []
    arch_raw_vars: list[float] = []
    arch_groups: list[tuple[str, np.ndarray]] = []
    for archetype, grp in frame.groupby("archetype", sort=False):
        vals = grp["net_return_bps"].to_numpy(dtype=np.float64, copy=False)
        a = str(archetype)
        arch_raw_means.append(float(np.mean(vals)))
        arch_raw_vars.append(float(np.var(vals)) if len(vals) > 1 else 0.0)
        arch_groups.append((a, vals))

    k_arch = (
        _compute_eb_shrinkage_k(arch_raw_means, arch_raw_vars, shrinkage_k_max)
        if adaptive_shrinkage
        else shrinkage_k
    )
    for a, vals in arch_groups:
        n_eff = _effective_n(float(vals.shape[0]))
        w = n_eff / (n_eff + k_arch)
        raw_mean = float(np.mean(vals))
        mu_val = w * raw_mean + (1.0 - w) * global_mu
        if min_cell_edge_floor_bps > 0.0 and mu_val < min_cell_edge_floor_bps:
            mu_val = 0.0
        arch_mu[a] = mu_val
        arch_q10[a] = w * float(np.percentile(vals, 10)) + (1.0 - w) * global_q10

    cell_mu: dict[tuple[str, int], float] = {}
    cell_q10: dict[tuple[str, int], float] = {}
    if axis == "archetype_regime":
        cell_raw_means: list[float] = []
        cell_raw_vars: list[float] = []
        cell_groups: list[tuple[tuple[str, int], np.ndarray]] = []
        for (archetype, regime_code), grp in frame.groupby(
            ["archetype", "entry_regime_code"], sort=False
        ):
            vals = grp["net_return_bps"].to_numpy(dtype=np.float64, copy=False)
            key = (str(archetype), int(regime_code))
            cell_raw_means.append(float(np.mean(vals)))
            cell_raw_vars.append(float(np.var(vals)) if len(vals) > 1 else 0.0)
            cell_groups.append((key, vals))

        k_cell = (
            _compute_eb_shrinkage_k(cell_raw_means, cell_raw_vars, shrinkage_k_max)
            if adaptive_shrinkage
            else shrinkage_k
        )
        for key, vals in cell_groups:
            n_eff = _effective_n(float(vals.shape[0]))
            w = n_eff / (n_eff + k_cell)
            raw_mean = float(np.mean(vals))
            mu_val = w * raw_mean + (1.0 - w) * global_mu
            if min_cell_edge_floor_bps > 0.0 and mu_val < min_cell_edge_floor_bps:
                mu_val = 0.0
            cell_mu[key] = mu_val
            cell_q10[key] = w * float(np.percentile(vals, 10)) + (1.0 - w) * global_q10

    return cell_mu, cell_q10, arch_mu, arch_q10, global_mu, global_q10


def _internal_validation_rank_ic(
    frame: pd.DataFrame,
    *,
    shrinkage_k: float,
    val_fraction: float,
    axis: str,
    adaptive_shrinkage: bool = False,
    shrinkage_k_max: float = 50.0,
    freq_n_cap: int = 0,
    min_cell_edge_floor_bps: float = 0.0,
    variant_prior_enabled: bool = False,
    variant_shrinkage_k: float = 30.0,
    variant_min_obs: int = 40,
) -> float:
    """In-fold purged validation Rank IC for the given axis.

    Splits frame by entry_idx time-order: last val_fraction is val,
    remainder minus a 1-bar purge gap is sub-fit.
    When variant_prior_enabled=True, predictions use 3-level fallback:
    variant_mu → cell_mu → arch_mu → global_mu.
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

    fit_kwargs: dict[str, object] = {
        "adaptive_shrinkage": adaptive_shrinkage,
        "shrinkage_k_max": shrinkage_k_max,
        "freq_n_cap": freq_n_cap,
        "min_cell_edge_floor_bps": min_cell_edge_floor_bps,
    }
    _, _, arch_mu, _, global_mu, _ = _fit_cell_means(
        sub_fit, shrinkage_k=shrinkage_k, axis="archetype_only", **fit_kwargs  # type: ignore[arg-type]
    )

    if axis == "archetype_regime":
        cell_mu, _, _, _, _, _ = _fit_cell_means(
            sub_fit, shrinkage_k=shrinkage_k, axis="archetype_regime", **fit_kwargs  # type: ignore[arg-type]
        )
    else:
        cell_mu = {}

    # Fit variant prior on sub_fit only (IS-only, no OOS leakage)
    v_mu: dict[str, float] = {}
    if variant_prior_enabled:
        v_mu = _fit_variant_means(
            sub_fit,
            cell_mu=cell_mu,
            arch_mu=arch_mu,
            global_mu=global_mu,
            k_variant=variant_shrinkage_k,
            min_obs=variant_min_obs,
            freq_n_cap=freq_n_cap,
        )

    has_family_variant = variant_prior_enabled and bool(v_mu)

    if axis == "archetype_regime":
        def _predict_regime(row: pd.Series) -> float:
            if has_family_variant:
                fam = str(row.get("family", ""))
                var = str(row.get("variant", ""))
                vkey = _variant_key(fam, var)
                if vkey in v_mu:
                    return v_mu[vkey]
            key = (str(row["archetype"]), int(row["entry_regime_code"]))
            return cell_mu.get(key, arch_mu.get(str(row["archetype"]), global_mu))

        pred = val_set.apply(_predict_regime, axis=1).to_numpy(dtype=np.float64)
    else:
        def _predict_arch(row: pd.Series) -> float:
            if has_family_variant:
                fam = str(row.get("family", ""))
                var = str(row.get("variant", ""))
                vkey = _variant_key(fam, var)
                if vkey in v_mu:
                    return v_mu[vkey]
            return arch_mu.get(str(row["archetype"]), global_mu)

        pred = val_set.apply(_predict_arch, axis=1).to_numpy(dtype=np.float64)

    realized = val_set["net_return_bps"].to_numpy(dtype=np.float64, copy=False)
    return _rank_ic_local(pred, realized)


def fit_regime_conditional_ensemble(
    *,
    train_events: pd.DataFrame,
    cfg: CandidateStrategyConfig,
    oos_proof_events: pd.DataFrame | None = None,
    fold_ids: NDArray[np.int32] | None = None,
    regime_oos_stability_rho: float | None = None,
) -> RegimeConditionalEnsemble:
    """Fit per-cell shrinkage estimates from train-window events.

    Args:
        train_events: Training window events DataFrame.
        cfg: Candidate strategy configuration.
        oos_proof_events: Optional OOS events for P0 Regime Lift Proof Gate.
        fold_ids: Purged walk-forward fold IDs aligned with oos_proof_events.

    Returns:
        Fitted RegimeConditionalEnsemble with optional lift_proof diagnostics.
    """
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
    _variant_cols = [c for c in ("family", "variant") if c in train_events.columns]
    _base_cols = ["archetype", "entry_regime_code", "net_return_bps", *_variant_cols]
    frame = train_events.loc[:, _base_cols].copy()
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

    # EB adaptive shrinkage parameters (IS-only; no OOS leakage)
    adaptive_shrinkage: bool = bool(getattr(cfg, "ensemble_adaptive_shrinkage", True))
    shrinkage_k_max: float = float(getattr(cfg, "ensemble_shrinkage_k_max", 50.0))
    freq_n_cap: int = int(getattr(cfg, "ensemble_freq_n_cap", 200))
    min_cell_edge_floor_bps: float = float(getattr(cfg, "ensemble_min_cell_edge_floor_bps", 0.0))

    # Variant-edge hierarchical prior parameters
    variant_prior_enabled: bool = bool(getattr(cfg, "ensemble_variant_prior_enabled", True))
    variant_shrinkage_k: float = float(getattr(cfg, "ensemble_variant_shrinkage_k", 30.0))
    variant_min_obs: int = int(getattr(cfg, "ensemble_variant_min_obs", 40))

    eb_fit_kwargs: dict[str, object] = {
        "adaptive_shrinkage": adaptive_shrinkage,
        "shrinkage_k_max": shrinkage_k_max,
        "freq_n_cap": freq_n_cap,
        "min_cell_edge_floor_bps": min_cell_edge_floor_bps,
    }
    variant_ic_kwargs: dict[str, object] = {
        "variant_prior_enabled": variant_prior_enabled,
        "variant_shrinkage_k": variant_shrinkage_k,
        "variant_min_obs": variant_min_obs,
    }

    # Compute archetype-only (always needed for fallback/auto)
    _, _, arch_mu, arch_q10, global_mu, global_q10 = _fit_cell_means(
        frame, shrinkage_k=shrinkage_k, axis="archetype_only", **eb_fit_kwargs  # type: ignore[arg-type]
    )

    if conditioning_cfg == "auto":
        ic_arch = _internal_validation_rank_ic(
            frame,
            shrinkage_k=shrinkage_k,
            val_fraction=val_fraction,
            axis="archetype_only",
            **{**eb_fit_kwargs, **variant_ic_kwargs},  # type: ignore[arg-type]
        )
        ic_regime = _internal_validation_rank_ic(
            frame,
            shrinkage_k=shrinkage_k,
            val_fraction=val_fraction,
            axis="archetype_regime",
            **{**eb_fit_kwargs, **variant_ic_kwargs},  # type: ignore[arg-type]
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
            frame,
            shrinkage_k=shrinkage_k,
            val_fraction=val_fraction,
            axis="archetype_regime",
            **{**eb_fit_kwargs, **variant_ic_kwargs},  # type: ignore[arg-type]
        )
    else:
        chosen = "archetype_only"
        val_ic = _internal_validation_rank_ic(
            frame,
            shrinkage_k=shrinkage_k,
            val_fraction=val_fraction,
            axis="archetype_only",
            **{**eb_fit_kwargs, **variant_ic_kwargs},  # type: ignore[arg-type]
        )

    if chosen == "archetype_regime":
        cell_mu, cell_q10, _, _, _, _ = _fit_cell_means(
            frame, shrinkage_k=shrinkage_k, axis="archetype_regime", **eb_fit_kwargs  # type: ignore[arg-type]
        )
    else:
        cell_mu = {}
        cell_q10 = {}

    # ── Variant-edge hierarchical prior (IS-only) ────────────────────────────
    variant_mu: dict[str, float] = {}
    if variant_prior_enabled:
        variant_mu = _fit_variant_means(
            frame,
            cell_mu=cell_mu,
            arch_mu=arch_mu,
            global_mu=global_mu,
            k_variant=variant_shrinkage_k,
            min_obs=variant_min_obs,
            freq_n_cap=freq_n_cap,
        )
        _logger.info("[ENSEMBLE] variant_prior: %d variants fitted", len(variant_mu))

    # ── Diagnostic table (IC sign audit) ─────────────────────────────────────
    # Logs archetype-level mean edge vs global to identify anti-predictive variants.
    _log_ensemble_diagnostics(
        frame=frame,
        global_mu=global_mu,
        arch_mu=arch_mu,
        val_ic=float(val_ic),
        chosen=chosen,
        adaptive_shrinkage=adaptive_shrinkage,
        k_used=shrinkage_k if not adaptive_shrinkage else shrinkage_k_max,
    )

    # P0: Regime Lift Proof Gate
    lift_proof: RegimeLiftProofResult | None = None
    conditioning_path = "pooled_fallback"

    if (
        oos_proof_events is not None
        and fold_ids is not None
        and chosen == "archetype_regime"
        and not oos_proof_events.empty
        and "net_return_bps" in oos_proof_events.columns
    ):
        # Regime Lift Proof: compare realized edge captured by the two prediction
        # paths on an out-of-fit proof window. `evaluate_regime_lift_proof()`
        # converts each prediction path to realized signed edge and tests the
        # per-event lift versus the pooled baseline.
        proof_events = oos_proof_events.loc[
            :, ["archetype", "entry_regime_code", "net_return_bps"]
        ].copy()
        proof_events["archetype"] = proof_events["archetype"].astype(str)
        proof_events["entry_regime_code"] = pd.to_numeric(
            proof_events["entry_regime_code"], errors="coerce"
        )
        proof_events["net_return_bps"] = pd.to_numeric(
            proof_events["net_return_bps"], errors="coerce"
        )
        proof_events = proof_events.loc[
            proof_events["archetype"].ne("")
            & proof_events["entry_regime_code"].notna()
            & proof_events["net_return_bps"].notna()
        ].copy()
        proof_events["entry_regime_code"] = proof_events["entry_regime_code"].astype(int)

        proof_fold_ids = np.asarray(fold_ids, dtype=np.int32)
        if proof_events.shape[0] != proof_fold_ids.shape[0]:
            raise ValueError("oos_proof_events and fold_ids must align")

        realized = proof_events["net_return_bps"].to_numpy(dtype=np.float64, copy=False)
        mu_regime_arr = _predict_mu_by_event(
            proof_events,
            cell_mu=cell_mu,
            arch_mu=arch_mu,
            global_mu=global_mu,
            use_archetype_only=False,
        )
        mu_pooled_arr = _predict_mu_by_event(
            proof_events,
            cell_mu={},
            arch_mu=arch_mu,
            global_mu=global_mu,
            use_archetype_only=True,
        )

        regime_cfg = getattr(cfg, "regime", None)
        proof_enabled: bool = getattr(regime_cfg, "regime_lift_proof_enabled", True)
        nw_threshold: float = getattr(regime_cfg, "regime_lift_nw_tstat_threshold", 1.5)
        fold_ratio: float = getattr(regime_cfg, "regime_lift_fold_pass_ratio", 0.60)
        max_bars: int = getattr(regime_cfg, "regime_lift_max_holding_bars", 6)

        lift_proof = evaluate_regime_lift_proof(
            regime_cond_edges=mu_regime_arr,
            pooled_edges=mu_pooled_arr,
            realized_edges=realized,
            fold_ids=proof_fold_ids,
            nw_tstat_threshold=nw_threshold,
            fold_pass_ratio_threshold=fold_ratio,
            max_holding_bars=max_bars,
            proof_enabled=proof_enabled,
        )
        conditioning_path = lift_proof.conditioning_path
        _logger.info(
            "Regime lift proof: passed=%s nw_tstat=%.3f fold_pass_ratio=%.2f",
            lift_proof.proof_passed,
            lift_proof.nw_tstat,
            lift_proof.fold_pass_ratio,
        )

        # Proof 실패 시 conditioning을 archetype_only로 강제 downgrade
        if not lift_proof.proof_passed:
            chosen = "archetype_only"
            cell_mu = {}
            cell_q10 = {}

    elif chosen == "archetype_regime":
        # OOS 증거 없음 → fail-SAFE: 복잡한 경로를 증거 없이 선택하지 않음
        chosen = "archetype_only"
        cell_mu = {}
        cell_q10 = {}
        conditioning_path = "no_oos_evidence_failsafe"
        _logger.info("regime conditioning downgraded: no oos proof window → archetype_only (fail-safe)")

    return RegimeConditionalEnsemble(
        cell_mu_bps=cell_mu,
        cell_q10_bps=cell_q10,
        global_mu_bps=global_mu,
        global_q10_bps=global_q10,
        conditioning=chosen,
        archetype_mu_bps=arch_mu,
        archetype_q10_bps=arch_q10,
        validation_rank_ic=float(val_ic),
        variant_mu_bps=variant_mu,
        conditioning_path=conditioning_path,
        lift_proof=lift_proof,
        regime_oos_stability_rho=regime_oos_stability_rho,
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
    has_variant_prior = bool(model.variant_mu_bps)

    for idx, row in enumerate(event_frame.itertuples(index=False), start=0):
        arch = str(getattr(row, "archetype", ""))

        # 3-level fallback: variant → cell → archetype → global
        if has_variant_prior:
            fam = str(getattr(row, "family", ""))
            var = str(getattr(row, "variant", ""))
            vkey = _variant_key(fam, var)
            if vkey in model.variant_mu_bps:
                mu_net_decision_bps[idx] = model.variant_mu_bps[vkey]
                q10_net_bps[idx] = model.cell_q10_bps.get(
                    (arch, int(getattr(row, "entry_regime_code", 0))),
                    model.archetype_q10_bps.get(arch, model.global_q10_bps),
                )
                continue

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
            "conditioning_path": model.conditioning_path,
            "validation_rank_ic": model.validation_rank_ic,
            "mu_shrinkage_lambda": mu_shrinkage_lambda,
            "lift_proof_passed": (
                int(model.lift_proof.proof_passed) if model.lift_proof is not None else -1
            ),
            "lift_nw_tstat": (
                model.lift_proof.nw_tstat if model.lift_proof is not None else float("nan")
            ),
        },
    )
