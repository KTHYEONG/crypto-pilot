from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

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
    # Variant-level shrunk offsets (deviation from mode cell anchor)
    variant_offset_bps: dict[str, float] = field(default_factory=dict)
    # P0: Regime Lift Proof Gate fields
    conditioning_path: str = "pooled_fallback"  # "regime_conditioned" | "pooled_fallback" | "no_oos_evidence_failsafe"
    lift_proof: RegimeLiftProofResult | None = None  # None = proof 미실행 (backward compat)
    regime_oos_stability_rho: float | None = None  # 진단 전용 (C4 rho 주입, 게이팅 아님)
    # Direction B: q90 실제 산출 (Kelly sizing용 sigma_r 추정)
    # cell/arch/global q90은 _fit_cell_means에서 계산되며, predict 시 실제 q90으로 사용됨
    cell_q90_bps: dict[tuple[str, int], float] = field(default_factory=dict)
    archetype_q90_bps: dict[str, float] = field(default_factory=dict)
    global_q90_bps: float = 0.0
    # Direction A: regime-conditional score calibration (score_z slope fitting)
    # β > 0 regime만 score_calibration_valid=True; β ≤ 0 → fallback to cell lookup
    regime_score_slope: dict[int, float] = field(default_factory=dict)
    regime_score_intercept: dict[int, float] = field(default_factory=dict)
    score_calibration_valid: dict[int, bool] = field(default_factory=dict)
    # Diagnostics collection for summary table
    ensemble_diagnostics: dict[str, Any] = field(default_factory=dict)


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
    allowed_families: tuple[str, ...] = (),
) -> tuple[dict[str, float], dict[str, float]]:
    """Fit variant-level shrunk means toward archetype-regime cell anchor.

    vmean_v = w_v * raw_v + (1 - w_v) * anchor_v
    anchor_v = cell_mu[(mode_arch, mode_regime)] → arch_mu[arch] → global_mu
    w_v = n_eff / (n_eff + k_variant), n_v < min_obs → w_v ≈ 0 (anchor fallback).

    Requires frame to have 'family', 'variant', 'archetype', 'entry_regime_code',
    'net_return_bps' columns. Returns empty dicts if columns absent.
    """
    required = {"family", "variant", "archetype", "entry_regime_code", "net_return_bps"}
    if not required.issubset(frame.columns):
        return {}, {}

    def _effective_n(raw_n: int) -> float:
        if freq_n_cap > 0:
            return float(min(raw_n, freq_n_cap))
        return float(raw_n)

    variant_mu: dict[str, float] = {}
    variant_offset: dict[str, float] = {}
    fam_col = frame["family"].astype(str)
    var_col = frame["variant"].astype(str)
    vkeys = (fam_col + ":" + var_col).values
    arch_col = frame["archetype"].astype(str).values
    regime_col = pd.to_numeric(frame["entry_regime_code"], errors="coerce").fillna(0).astype(int).values
    edge_col = pd.to_numeric(frame["net_return_bps"], errors="coerce").values

    for vkey in np.unique(vkeys):
        family = vkey.split(":")[0]
        if allowed_families and family not in allowed_families:
            continue

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
            variant_offset[vkey] = 0.0
        else:
            n_eff = _effective_n(n_v)
            w_v = n_eff / (n_eff + k_variant)
            raw_mean = float(np.mean(edges))
            shrunk_mean = w_v * raw_mean + (1.0 - w_v) * anchor
            variant_mu[vkey] = shrunk_mean
            variant_offset[vkey] = shrunk_mean - anchor

    return variant_mu, variant_offset


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
    num_valid_regimes: int = 0,
) -> dict[str, Any]:
    """Emit a concise diagnostic log and return data for aggregation.

    Compresses the previous multi-line table into 2 lines to reduce noise.
    """
    n_total = len(frame)
    ic_sign = "✅" if val_ic > 0 else "❌"
    if "symbol" in frame.columns and not frame.empty:
        n_syms = int(frame["symbol"].nunique())
        symbol_name = f"POOL({n_syms})"
    else:
        symbol_name = "POOL(0)"

    summary = (
        f"[ENSEMBLE] {symbol_name} | N: {n_total} | IC: {val_ic:.4f} ({ic_sign}) | "
        f"Mu: {global_mu:.3f} | {chosen} | k: {k_used:.1f}"
    )
    
    arch_items = []
    for arch, mu_val in sorted(arch_mu.items()):
        sign = "✅" if mu_val >= 0.0 else "❌"
        # Extract short label (e.g., ts_mom from time_series_momentum)
        label = arch.replace("_reversion", "").replace("_continuation", "").replace("time_series_", "ts_")
        arch_items.append(f"{label}: {mu_val:.1f} ({sign})")
    
    detail = f"└─ mu_bps: [{', '.join(arch_items)}] | score_cal: {num_valid_regimes} valid"
    _logger.info("%s\n%s", summary, detail)

    return {
        "symbol": symbol_name,
        "n_events": n_total,
        "val_ic": val_ic,
        "global_mu": global_mu,
        "arch_mu": arch_mu,
        "chosen": chosen,
        "num_valid_regimes": num_valid_regimes,
    }


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
    dict[tuple[str, int], float],
    dict[str, float],
    float,
]:
    """Compute shrunk cell means for given axis ('regime' or 'archetype').

    Returns cell_mu, cell_q10, arch_mu, arch_q10, global_mu, global_q10,
    cell_q90, arch_q90, global_q90.
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
    global_q90 = float(np.percentile(edge, 90))

    def _effective_n(raw_n: float) -> float:
        if freq_n_cap > 0:
            return float(min(raw_n, freq_n_cap))
        return raw_n

    # archetype-only shrinkage
    arch_mu: dict[str, float] = {}
    arch_q10: dict[str, float] = {}
    arch_q90: dict[str, float] = {}

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
        arch_q90[a] = w * float(np.percentile(vals, 90)) + (1.0 - w) * global_q90

    cell_mu: dict[tuple[str, int], float] = {}
    cell_q10: dict[tuple[str, int], float] = {}
    cell_q90: dict[tuple[str, int], float] = {}
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
            cell_q90[key] = w * float(np.percentile(vals, 90)) + (1.0 - w) * global_q90

    return cell_mu, cell_q10, arch_mu, arch_q10, global_mu, global_q10, cell_q90, arch_q90, global_q90


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
    allowed_families: tuple[str, ...] = (),
    score_calibration_enabled: bool = False,
    score_z_clip: float = 3.0,
    score_calibration_min_obs: int = 60,
    score_slope_k: float = 100.0,
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
    _, _, arch_mu, _, global_mu, _, _, _, _ = _fit_cell_means(
        sub_fit, shrinkage_k=shrinkage_k, axis="archetype_only", **fit_kwargs  # type: ignore[arg-type]
    )

    if axis == "archetype_regime":
        cell_mu, _, _, _, _, _, _, _, _ = _fit_cell_means(
            sub_fit, shrinkage_k=shrinkage_k, axis="archetype_regime", **fit_kwargs  # type: ignore[arg-type]
        )
    else:
        cell_mu = {}

    # Fit variant prior on sub_fit only (IS-only, no OOS leakage)
    _v_mu: dict[str, float] = {}
    v_offset: dict[str, float] = {}
    if variant_prior_enabled:
        _v_mu, v_offset = _fit_variant_means(
            sub_fit,
            cell_mu=cell_mu,
            arch_mu=arch_mu,
            global_mu=global_mu,
            k_variant=variant_shrinkage_k,
            min_obs=variant_min_obs,
            freq_n_cap=freq_n_cap,
            allowed_families=allowed_families,
        )

    has_family_variant = variant_prior_enabled and bool(v_offset)

    # Direction A (Fix 2): fit score slope on sub_fit; measure IC with score path on val_set
    _sub_slope: dict[int, float] = {}
    _sub_intercept: dict[int, float] = {}
    _sub_valid: dict[int, bool] = {}
    if score_calibration_enabled and "score_z" in sub_fit.columns and "entry_regime_code" in sub_fit.columns:
        _sz_col = pd.to_numeric(sub_fit["score_z"], errors="coerce").clip(-score_z_clip, score_z_clip)
        _sub_z = sub_fit.copy()
        _sub_z["_sz_cal"] = _sz_col
        for _rc, _grp in _sub_z.groupby("entry_regime_code", sort=False):
            _gs = int(_rc)
            _vm = _grp["_sz_cal"].notna() & _grp["net_return_bps"].notna()
            _nv = int(_vm.sum())
            if _nv < score_calibration_min_obs:
                continue
            _zs = _grp.loc[_vm, "_sz_cal"].to_numpy(dtype=np.float64)
            _ys = _grp.loc[_vm, "net_return_bps"].to_numpy(dtype=np.float64)
            _rho = float(np.corrcoef(_zs, _ys)[0, 1])
            if not np.isfinite(_rho):
                continue
            _sz_std = float(np.std(_zs)) + 1e-12
            _sy_std = float(np.std(_ys)) + 1e-12
            _beta_r = _rho * (_sy_std / _sz_std)
            _ws = _nv / (_nv + score_slope_k)
            _beta_sh = _ws * _beta_r
            if _beta_sh <= 0.0:
                continue
            _sub_slope[_gs] = _beta_sh
            _sub_intercept[_gs] = float(np.mean(_ys)) - _beta_sh * float(np.mean(_zs))
            _sub_valid[_gs] = True

    # Prepare val_set with score_z column when score path is active
    val_set_p = val_set
    if _sub_valid and "score_z" in val_set.columns:
        val_set_p = val_set.copy()
        val_set_p["_sz_cal"] = pd.to_numeric(val_set["score_z"], errors="coerce").clip(
            -score_z_clip, score_z_clip
        )

    if axis == "archetype_regime":
        def _predict_regime(row: pd.Series) -> float:
            key = (str(row["archetype"]), int(row["entry_regime_code"]))
            base_val = cell_mu.get(key, arch_mu.get(str(row["archetype"]), global_mu))
            if has_family_variant:
                fam = str(row.get("family", ""))
                var = str(row.get("variant", ""))
                vkey = _variant_key(fam, var)
                if vkey in v_offset:
                    base_val = base_val + v_offset[vkey]
            if _sub_valid:
                _gr = int(row["entry_regime_code"])
                if _sub_valid.get(_gr, False):
                    try:
                        _zf = float(row.get("_sz_cal", float("nan")))
                    except (TypeError, ValueError):
                        _zf = float("nan")
                    if np.isfinite(_zf):
                        return _sub_intercept[_gr] + _sub_slope[_gr] * _zf
            return base_val

        pred = val_set_p.apply(_predict_regime, axis=1).to_numpy(dtype=np.float64)
    else:
        def _predict_arch(row: pd.Series) -> float:
            base_val = arch_mu.get(str(row["archetype"]), global_mu)
            if has_family_variant:
                fam = str(row.get("family", ""))
                var = str(row.get("variant", ""))
                vkey = _variant_key(fam, var)
                if vkey in v_offset:
                    base_val = base_val + v_offset[vkey]
            if _sub_valid:
                _gr = int(row["entry_regime_code"])
                if _sub_valid.get(_gr, False):
                    try:
                        _zf = float(row.get("_sz_cal", float("nan")))
                    except (TypeError, ValueError):
                        _zf = float("nan")
                    if np.isfinite(_zf):
                        return _sub_intercept[_gr] + _sub_slope[_gr] * _zf
            return base_val

        pred = val_set_p.apply(_predict_arch, axis=1).to_numpy(dtype=np.float64)

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
    _score_cols = [c for c in ("score_z",) if c in train_events.columns]
    _base_cols = ["archetype", "entry_regime_code", "net_return_bps", *_variant_cols, *_score_cols]
    if "symbol" in train_events.columns:
        _base_cols.append("symbol")
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

    # Pass allowed_families from cfg
    allowed_families = getattr(cfg, "ensemble_variant_prior_families", ())

    # Score calibration params — read early so val_ic measurement reflects Direction A
    score_calibration_enabled: bool = bool(getattr(cfg, "ensemble_score_calibration_enabled", False))
    score_z_clip: float = float(getattr(cfg, "ensemble_score_z_clip", 3.0))
    score_calibration_min_obs: int = int(getattr(cfg, "ensemble_score_calibration_min_obs", 60))
    score_slope_k: float = float(getattr(cfg, "ensemble_score_slope_k", 100.0))

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
        "allowed_families": allowed_families,
    }
    score_ic_kwargs: dict[str, object] = {
        "score_calibration_enabled": score_calibration_enabled,
        "score_z_clip": score_z_clip,
        "score_calibration_min_obs": score_calibration_min_obs,
        "score_slope_k": score_slope_k,
    }

    # Compute archetype-only (always needed for fallback/auto)
    _, _, arch_mu, arch_q10, global_mu, global_q10, _, arch_q90, global_q90 = _fit_cell_means(
        frame, shrinkage_k=shrinkage_k, axis="archetype_only", **eb_fit_kwargs  # type: ignore[arg-type]
    )

    if conditioning_cfg == "auto":
        ic_arch = _internal_validation_rank_ic(
            frame,
            shrinkage_k=shrinkage_k,
            val_fraction=val_fraction,
            axis="archetype_only",
            **{**eb_fit_kwargs, **variant_ic_kwargs, **score_ic_kwargs},  # type: ignore[arg-type]
        )
        ic_regime = _internal_validation_rank_ic(
            frame,
            shrinkage_k=shrinkage_k,
            val_fraction=val_fraction,
            axis="archetype_regime",
            **{**eb_fit_kwargs, **variant_ic_kwargs, **score_ic_kwargs},  # type: ignore[arg-type]
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
            **{**eb_fit_kwargs, **variant_ic_kwargs, **score_ic_kwargs},  # type: ignore[arg-type]
        )
    else:
        chosen = "archetype_only"
        val_ic = _internal_validation_rank_ic(
            frame,
            shrinkage_k=shrinkage_k,
            val_fraction=val_fraction,
            axis="archetype_only",
            **{**eb_fit_kwargs, **variant_ic_kwargs, **score_ic_kwargs},  # type: ignore[arg-type]
        )

    if chosen == "archetype_regime":
        cell_mu, cell_q10, _, _, _, _, cell_q90, _, _ = _fit_cell_means(
            frame, shrinkage_k=shrinkage_k, axis="archetype_regime", **eb_fit_kwargs  # type: ignore[arg-type]
        )
    else:
        cell_mu = {}
        cell_q10 = {}
        cell_q90 = {}

    # ── Variant-edge hierarchical prior (IS-only) ────────────────────────────
    variant_mu: dict[str, float] = {}
    variant_offset: dict[str, float] = {}
    if variant_prior_enabled:
        allowed_families = getattr(cfg, "ensemble_variant_prior_families", ())
        variant_mu, variant_offset = _fit_variant_means(
            frame,
            cell_mu=cell_mu,
            arch_mu=arch_mu,
            global_mu=global_mu,
            k_variant=variant_shrinkage_k,
            min_obs=variant_min_obs,
            freq_n_cap=freq_n_cap,
            allowed_families=allowed_families,
        )

    # Initialize direction A variables so they exist before the diagnostic log
    regime_score_slope: dict[int, float] = {}
    regime_score_intercept: dict[int, float] = {}
    score_calibration_valid: dict[int, bool] = {}

    # ── Direction A: regime-conditional score calibration ──────────────────
    if score_calibration_enabled and "score_z" in frame.columns:
        score_z_col = pd.to_numeric(frame["score_z"], errors="coerce")
        frame_with_z = frame.copy()
        frame_with_z["_score_z"] = score_z_col.clip(-score_z_clip, score_z_clip)

        for regime_code, grp in frame_with_z.groupby("entry_regime_code", sort=False):
            g = int(regime_code)
            valid_mask = grp["_score_z"].notna() & grp["net_return_bps"].notna()
            n_valid = int(valid_mask.sum())
            if n_valid < score_calibration_min_obs:
                score_calibration_valid[g] = False
                _logger.debug(
                    "[SCORE-CAL-DIAG] regime=%d REJECT obs_too_low n=%d min=%d",
                    g, n_valid, score_calibration_min_obs,
                )
                continue

            grp_ordered = (
                grp[valid_mask].sort_values("entry_idx")
                if "entry_idx" in grp.columns
                else grp[valid_mask]
            )

            z_arr = grp_ordered["_score_z"].to_numpy(dtype=np.float64)
            y_arr = grp_ordered["net_return_bps"].to_numpy(dtype=np.float64)

            rho = float(np.corrcoef(z_arr, y_arr)[0, 1])
            if not np.isfinite(rho):
                score_calibration_valid[g] = False
                _logger.debug(
                    "[SCORE-CAL-DIAG] regime=%d REJECT invalid_rho n=%d", g, n_valid,
                )
                continue

            sigma_z = float(np.std(z_arr)) + 1e-12
            sigma_y = float(np.std(y_arr)) + 1e-12
            beta_raw = rho * (sigma_y / sigma_z)

            w_slope = n_valid / (n_valid + score_slope_k)
            beta_shrunk = w_slope * beta_raw
            alpha = float(np.mean(y_arr)) - beta_shrunk * float(np.mean(z_arr))

            regime_score_slope[g] = beta_shrunk
            regime_score_intercept[g] = alpha

            probe_start = int(n_valid * (1.0 - val_fraction))
            oos_sign_ok = False
            if probe_start >= 10 and (n_valid - probe_start) >= 10:
                rho_probe = float(np.corrcoef(z_arr[probe_start:], y_arr[probe_start:])[0, 1])
                oos_sign_ok = np.isfinite(rho_probe) and rho_probe > 0.0

            if probe_start < 10 or (n_valid - probe_start) < 10:
                _valid = beta_shrunk > 0.0
                score_calibration_valid[g] = _valid
                _reason = "ACCEPT(short_probe)" if _valid else "REJECT negative_slope(short_probe)"
                _logger.debug(
                    "[SCORE-CAL-DIAG] regime=%d %s beta=%.4f n=%d probe_start=%d",
                    g, _reason, beta_shrunk, n_valid, probe_start,
                )
            else:
                _valid = (beta_shrunk > 0.0) and oos_sign_ok
                score_calibration_valid[g] = _valid
                if _valid:
                    _logger.debug(
                        "[SCORE-CAL-DIAG] regime=%d ACCEPT beta=%.4f oos_sign_ok=%s n=%d",
                        g, beta_shrunk, oos_sign_ok, n_valid,
                    )
                elif beta_shrunk <= 0.0:
                    _logger.debug(
                        "[SCORE-CAL-DIAG] regime=%d REJECT negative_slope beta=%.4f n=%d",
                        g, beta_shrunk, n_valid,
                    )
                else:
                    _logger.debug(
                        "[SCORE-CAL-DIAG] regime=%d REJECT oos_sign_fail beta=%.4f rho_probe=%.4f n=%d",
                        g, beta_shrunk, float(np.corrcoef(z_arr[probe_start:], y_arr[probe_start:])[0, 1]), n_valid,
                    )

    # ── Score-Cal 요약 (INFO, C1 진단) ──────────────────────────────────────
    if score_calibration_enabled and score_calibration_valid:
        _n_valid_sc = sum(score_calibration_valid.values())
        _n_total_sc = len(score_calibration_valid)
        # obs_too_low: regimes not in score_calibration_valid (skipped via continue)
        if "score_z" in frame.columns and "net_return_bps" in frame.columns:
            _all_regimes = set(frame["entry_regime_code"].dropna().astype(int).unique())
            _n_obs_low = len(_all_regimes - set(score_calibration_valid.keys()))
        else:
            _n_obs_low = 0
        _logger.info(
            "[SCORE-CAL-DIAG] valid=%d/%d obs_too_low=%d neg_slope_or_oos_fail=%d (min_obs=%d)",
            _n_valid_sc, _n_total_sc, _n_obs_low,
            sum(1 for v in score_calibration_valid.values() if not v),
            score_calibration_min_obs,
        )

    # ── Diagnostic table (IC sign audit) ─────────────────────────────────────
    ensemble_diag = _log_ensemble_diagnostics(
        frame=frame,
        global_mu=global_mu,
        arch_mu=arch_mu,
        val_ic=float(val_ic),
        chosen=chosen,
        adaptive_shrinkage=adaptive_shrinkage,
        k_used=shrinkage_k if not adaptive_shrinkage else shrinkage_k_max,
        num_valid_regimes=sum(score_calibration_valid.values()),
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

        if not lift_proof.proof_passed:
            chosen = "archetype_only"
            cell_mu = {}
            cell_q10 = {}
            cell_q90 = {}

    elif chosen == "archetype_regime":
        chosen = "archetype_only"
        cell_mu = {}
        cell_q10 = {}
        cell_q90 = {}
        conditioning_path = "no_oos_evidence_failsafe"

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
        variant_offset_bps=variant_offset,
        conditioning_path=conditioning_path,
        lift_proof=lift_proof,
        regime_oos_stability_rho=regime_oos_stability_rho,
        cell_q90_bps=cell_q90,
        archetype_q90_bps=arch_q90,
        global_q90_bps=global_q90,
        regime_score_slope=regime_score_slope,
        regime_score_intercept=regime_score_intercept,
        score_calibration_valid=score_calibration_valid,
        ensemble_diagnostics=ensemble_diag,
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
    q90_net_bps = np.empty(len(event_frame), dtype=np.float64)

    use_archetype_only = model.conditioning == "archetype_only"
    has_variant_prior = bool(model.variant_mu_bps)
    has_score_cal = bool(model.regime_score_slope)

    for idx, row in enumerate(event_frame.itertuples(index=False), start=0):
        arch = str(getattr(row, "archetype", ""))
        regime = int(getattr(row, "entry_regime_code", 0))
        key = (arch, regime)

        if use_archetype_only:
            cell_val = model.archetype_mu_bps.get(arch, model.global_mu_bps)
        else:
            cell_val = model.cell_mu_bps.get(
                key, model.archetype_mu_bps.get(arch, model.global_mu_bps)
            )

        # Direction B: q90 실제 lookup
        q90_val = model.cell_q90_bps.get(
            key, model.archetype_q90_bps.get(arch, model.global_q90_bps)
        )

        # Direction A: score-conditioned mu (calibration_valid=True인 regime만 적용)
        use_score_cal = (
            has_score_cal
            and bool(model.score_calibration_valid.get(regime, False))
        )

        # 3-level fallback: variant → cell → archetype → global
        if has_variant_prior:
            fam = str(getattr(row, "family", ""))
            var = str(getattr(row, "variant", ""))
            vkey = _variant_key(fam, var)
            if vkey in model.variant_mu_bps:
                try:
                    offset = model.variant_offset_bps.get(vkey, 0.0)
                except AttributeError:
                    offset = 0.0
                if use_score_cal:
                    z_raw = float(getattr(row, "score_z", 0.0))
                    beta = model.regime_score_slope.get(regime, 0.0)
                    alpha = model.regime_score_intercept.get(regime, cell_val)
                    mu_net_decision_bps[idx] = alpha + beta * z_raw
                else:
                    mu_net_decision_bps[idx] = cell_val + offset
                q10_net_bps[idx] = model.cell_q10_bps.get(
                    key,
                    model.archetype_q10_bps.get(arch, model.global_q10_bps),
                )
                q90_net_bps[idx] = q90_val
                continue

        if use_score_cal:
            z_raw = float(getattr(row, "score_z", 0.0))
            beta = model.regime_score_slope.get(regime, 0.0)
            alpha = model.regime_score_intercept.get(regime, cell_val)
            mu_net_decision_bps[idx] = alpha + beta * z_raw
        else:
            mu_net_decision_bps[idx] = cell_val
        q10_net_bps[idx] = model.cell_q10_bps.get(
            key,
            model.archetype_q10_bps.get(arch, model.global_q10_bps),
        )
        q90_net_bps[idx] = q90_val

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
        q90_net_bps=q90_net_bps,
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
