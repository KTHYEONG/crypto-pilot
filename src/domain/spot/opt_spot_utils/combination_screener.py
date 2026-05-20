"""Stage-1 screening: (signal × regime × sizing) combinations with fast CPCV."""

from __future__ import annotations

import itertools
import logging
import math
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.storages import InMemoryStorage
from optuna.trial import TrialState

from config.opt_config import OPT_SPOT_CONFIG, SPOT_EXCLUDED_SIZING_METHODS
from src.domain.spot.opt_spot_utils.cv_utils import build_cpcv_test_paths_with_fallback
from src.domain.spot.opt_spot_utils.objective import EMBARGO_BARS, objective_spot
from src.domain.spot.opt_spot_utils.opt_params import build_combined_param_space

_logger = logging.getLogger("combination_screener")


def _p25_path_consistency_score(metrics: dict[str, float]) -> float:
    """Stage-1 growth rank: p25 log-TWR scaled by inverse path CV (p25 / (1 + path_cv))."""
    p25 = float(metrics.get("cpcv_p25_log_tw", -1e9))
    if p25 <= -1e8:
        return p25
    cv = float(metrics.get("cpcv_path_cv", 10.0))
    return float(p25 * (1.0 / (1.0 + max(0.0, cv))))


@dataclass
class CombinationScore:
    signal: str
    regime: str
    sizing: str
    p10_gmgr: float
    mean_signal_rate: float
    disqualified: bool = False
    reason: str = ""


def _combo_search_dim(space: dict[str, Any]) -> int:
    dim = 0
    for k, spec in space.items():
        if k in ("SIGNAL_TYPE", "REGIME_TYPE", "SIZING_METHOD"):
            continue
        if not isinstance(spec, dict) or "type" not in spec:
            continue
        dim += 1
    return dim


def _auto_combo_trials(
    viable_count: int,
    median_dim: float,
    cfg: dict[str, Any],
) -> tuple[int, int]:
    """Practical auto-sizing for Stage-1 quick CPCV.
    Keep current config as floors, then scale mildly by viable combo count and
    effective combo dimension so runtime stays bounded as the search space grows.
    """
    v = max(1, int(viable_count))
    d = max(1.0, float(median_dim))
    p1_floor = int(cfg.get("SPOT_COMBO_QUICK_TRIALS_PHASE1", 10))
    p2_floor = int(cfg.get("SPOT_COMBO_QUICK_TRIALS", 28))
    p1_cap = max(p1_floor, int(cfg.get("SPOT_COMBO_QUICK_TRIALS_PHASE1_MAX", 18)))
    p2_cap = max(p2_floor, int(cfg.get("SPOT_COMBO_QUICK_TRIALS_MAX", 40)))

    phase1 = int(round(7.0 + 0.75 * math.log2(v + 1.0) + 0.20 * math.sqrt(d)))
    phase2 = int(round(19.0 + 1.25 * math.log2(v + 1.0) + 0.45 * math.sqrt(d)))

    phase1 = int(np.clip(phase1, p1_floor, p1_cap))
    phase2 = int(np.clip(phase2, p2_floor, p2_cap))
    if phase2 < phase1 + 12:
        phase2 = min(p2_cap, phase1 + 12)
    return phase1, phase2


def _ambiguity_phase2_boost(
    phase1_scores: Sequence[float],
    top_k: int,
    cfg: dict[str, Any],
) -> int:
    """If the prune boundary is crowded, add a small Phase-2 budget bump.
    This preserves fast default behavior while spending a bit more only when
    combo ranking is genuinely ambiguous.
    """
    arr = np.asarray([float(x) for x in phase1_scores if np.isfinite(float(x))], dtype=np.float64)
    if arr.size < max(6, top_k + 3):
        return 0
    arr.sort()
    std = float(np.std(arr))
    if std <= 1e-12:
        return int(cfg.get("SPOT_COMBO_PHASE2_AMBIGUITY_BOOST", 4))
    edge_idx = max(1, min(arr.size - 1, arr.size - top_k))
    gap = float(arr[edge_idx] - arr[edge_idx - 1])
    ratio = abs(gap) / std
    thr = float(cfg.get("SPOT_COMBO_AMBIGUITY_STD_RATIO", 0.15))
    if ratio < thr:
        return int(cfg.get("SPOT_COMBO_PHASE2_AMBIGUITY_BOOST", 4))
    return 0


def _mid_value_from_spec(spec: dict[str, Any]) -> int | float:
    t = spec["type"]
    if t == "int":
        lo = int(spec["low"])
        hi = int(spec["high"])
        step = int(spec.get("step", 1))
        mid = (lo + hi) // 2
        if step > 1:
            mid = lo + ((mid - lo) // step) * step
        return int(np.clip(mid, lo, hi))
    if t == "float":
        return float((float(spec["low"]) + float(spec["high"])) / 2.0)
    choices = spec.get("choices", ())
    if not choices:
        raise ValueError("categorical spec missing choices")
    return choices[len(choices) // 2]


def params_disqualified_against_space(params: dict[str, Any], space: dict[str, Any]) -> str:
    """Return reason string if params violate declared bounds (e.g. legacy KC_MULT=3.0)."""
    if "KC_MULT" in params and "KC_MULT" in space:
        spec = space["KC_MULT"]
        if isinstance(spec, dict) and spec.get("type") == "float":
            hi = float(spec["high"])
            if float(params["KC_MULT"]) > hi + 1e-9:
                return "kc_mult_oob"
    return ""


def build_probe_params(combo: CombinationScore, tf: str) -> dict[str, Any]:
    """Return search-space midpoint params for a Phase-B combo (signal-agnostic for Phase C)."""
    return _build_probe_params(combo.signal, combo.regime, combo.sizing, tf)


def _build_probe_params(sig: str, reg: str, siz: str, tf: str) -> dict[str, Any]:
    combo_space = build_combined_param_space(sig, reg, siz)
    probe_params: dict[str, Any] = {
        "SIGNAL_TYPE": sig,
        "REGIME_TYPE": reg,
        "SIZING_METHOD": siz,
        "TIMEFRAME": tf,
    }
    for k, spec in combo_space.items():
        if k in ("SIGNAL_TYPE", "REGIME_TYPE", "SIZING_METHOD"):
            continue
        if not isinstance(spec, dict) or "type" not in spec:
            continue
        if spec["type"] == "categorical":
            probe_params[k] = spec["choices"][0]
        else:
            probe_params[k] = _mid_value_from_spec(spec)
    return probe_params


def measure_signal_rate(
    data_maps: dict[str, dict[str, Any]],
    symbols: Sequence[str],
    tf: str,
    signal_name: str,
    *,
    mid_params: dict[str, Any],
) -> float:
    from src.domain.spot.signals import SIGNAL_REGISTRY

    ref_sym = symbols[0]
    target_df = data_maps[ref_sym][tf]
    is_off = int(data_maps[ref_sym].get(f"is_start_idx_{tf}", 0))
    df = target_df.iloc[is_off:].copy()
    if df.empty:
        return 0.0
    p = {**mid_params, "SIGNAL_TYPE": signal_name, "TIMEFRAME": tf}
    out = SIGNAL_REGISTRY[signal_name].compute(df, p)
    return float(np.mean(out.entry_signal.astype(np.float64)))


def _metrics_from_best_study(study: optuna.Study) -> dict[str, float]:
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None]
    if not completed:
        return {
            "objective_final": -1e9,
            "cpcv_mean_path_return_pct": -1e9,
            "cpcv_p25_log_tw": -1e9,
            "cpcv_path_cv": 10.0,
            "psr_paths": -1.0,
            "dsr_paths": -1.0,
            "p10_gmgr": -1e9,
        }
    completed.sort(key=lambda tr: float(tr.value or -1e9), reverse=True)
    best = completed[0]
    ua = best.user_attrs
    return {
        "objective_final": float(best.value or -1e9),
        "cpcv_mean_path_return_pct": float(ua.get("cpcv_mean_path_return_pct", -1e9)),
        "cpcv_p25_log_tw": float(ua.get("cpcv_p25_log_tw", -1e9)),
        "cpcv_path_cv": float(ua.get("cpcv_path_cv", 10.0)),
        "psr_paths": float(ua.get("psr_paths", 0.0)),
        "dsr_paths": float(ua.get("dsr_paths", -1.0)),
        "p10_gmgr": float(ua.get("p10_gmgr", -1e9)),
    }


def run_quick_cpcv_for_combo(
    combo: tuple[str, str, str],
    *,
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    n_trials: int,
    project_root: str,
    signal_cache_dir: str,
) -> float:
    sig, reg, siz = combo
    space = build_combined_param_space(sig, reg, siz)
    ref_sym = symbols[0]
    is_off = int(data_maps[ref_sym].get(f"is_start_idx_{tf}", 0))
    ref_df = data_maps[ref_sym][tf]
    prebuilt = None
    if ref_df is not None and not ref_df.empty:
        ref_len = len(ref_df) - is_off
        if ref_len >= 200:
            prebuilt = build_cpcv_test_paths_with_fallback(
                ref_len, embargo=int(EMBARGO_BARS.get(tf, 0))
            )
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(n_startup_trials=min(15, max(5, n_trials // 3)), seed=42),
        storage=InMemoryStorage(),
    )
    cache_root = signal_cache_dir if signal_cache_dir else None

    def _obj(trial: optuna.Trial) -> float:
        return objective_spot(
            trial,
            data_maps,
            list(symbols),
            tf,
            space=space,
            mode="multi",
            project_root=project_root,
            prebuilt_cpcv_bundle=prebuilt,
            signal_disk_cache_root=Path(cache_root) if cache_root else None,
        )

    study.optimize(_obj, n_trials=max(5, int(n_trials)), n_jobs=1, show_progress_bar=False)
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None]
    if not completed:
        return -1e9
    completed.sort(key=lambda tr: float(tr.value or -1e9), reverse=True)
    ktop = max(1, len(completed) // 5)
    top_trials = completed[:ktop]
    mean_score = float(
        np.mean([float(t.user_attrs.get("cpcv_mean_path_return_pct", -1e9)) for t in top_trials])
    )
    return mean_score


def run_quick_cpcv_for_combo_metrics(
    combo: tuple[str, str, str],
    *,
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    n_trials: int,
    project_root: str,
    signal_cache_dir: str,
) -> dict[str, float]:
    sig, reg, siz = combo
    space = build_combined_param_space(sig, reg, siz)
    ref_sym = symbols[0]
    is_off = int(data_maps[ref_sym].get(f"is_start_idx_{tf}", 0))
    ref_df = data_maps[ref_sym][tf]
    prebuilt = None
    if ref_df is not None and not ref_df.empty:
        ref_len = len(ref_df) - is_off
        if ref_len >= 200:
            prebuilt = build_cpcv_test_paths_with_fallback(
                ref_len, embargo=int(EMBARGO_BARS.get(tf, 0))
            )
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(n_startup_trials=min(15, max(5, n_trials // 3)), seed=42),
        storage=InMemoryStorage(),
    )
    cache_root = signal_cache_dir if signal_cache_dir else None

    def _obj(trial: optuna.Trial) -> float:
        return objective_spot(
            trial,
            data_maps,
            list(symbols),
            tf,
            space=space,
            mode="multi",
            project_root=project_root,
            prebuilt_cpcv_bundle=prebuilt,
            signal_disk_cache_root=Path(cache_root) if cache_root else None,
        )

    study.optimize(_obj, n_trials=max(5, int(n_trials)), n_jobs=1, show_progress_bar=False)
    return _metrics_from_best_study(study)


def _phase1_worker(
    combo: tuple[str, str, str],
    *,
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    n_trials: int,
    project_root: str,
    signal_cache_dir: str,
) -> float:
    """Quick Phase-1 CPCV; with fork, data_maps is shared read-only via CoW."""
    return run_quick_cpcv_for_combo(
        combo,
        data_maps=data_maps,
        symbols=symbols,
        tf=tf,
        n_trials=n_trials,
        project_root=project_root,
        signal_cache_dir=signal_cache_dir,
    )


def _phase2_worker(
    combo: tuple[str, str, str],
    *,
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    n_trials: int,
    project_root: str,
    signal_cache_dir: str,
) -> dict[str, float]:
    """Full-trial CPCV screen for surviving combinations."""
    return run_quick_cpcv_for_combo_metrics(
        combo,
        data_maps=data_maps,
        symbols=symbols,
        tf=tf,
        n_trials=n_trials,
        project_root=project_root,
        signal_cache_dir=signal_cache_dir,
    )


def _warmup_numba(
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    project_root: str = "",
    signal_cache_dir: str = "",
) -> None:
    """Trigger Numba JIT in the parent before fork so children inherit compiled code."""
    from src.domain.spot.regimes import REGIME_REGISTRY
    from src.domain.spot.signals import SIGNAL_REGISTRY
    from src.domain.spot.sizing import SIZING_REGISTRY

    sig = sorted(SIGNAL_REGISTRY.keys())[0]
    reg = sorted(REGIME_REGISTRY.keys())[0]
    siz_choices = sorted(k for k in SIZING_REGISTRY.keys() if k not in SPOT_EXCLUDED_SIZING_METHODS)
    siz = siz_choices[0] if siz_choices else sorted(SIZING_REGISTRY.keys())[0]
    space = build_combined_param_space(sig, reg, siz)
    try:
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(n_startup_trials=1, seed=0),
            storage=InMemoryStorage(),
        )

        def _warm_obj(trial: optuna.Trial) -> float:
            return objective_spot(
                trial,
                data_maps,
                list(symbols),
                tf,
                space=space,
                mode="multi",
                project_root=project_root,
                prebuilt_cpcv_bundle=None,
                signal_disk_cache_root=Path(signal_cache_dir) if signal_cache_dir else None,
            )

        study.optimize(
            _warm_obj,
            n_trials=1,
            n_jobs=1,
            show_progress_bar=False,
        )
    except Exception:
        ...


def _process_pool_context():
    """Prefer fork on Linux (WSL2); fall back to spawn where fork is unavailable."""
    import multiprocessing as mp

    if sys.platform == "win32":
        return mp.get_context("spawn")
    try:
        return mp.get_context("fork")
    except ValueError:
        return mp.get_context("spawn")


def run_combination_screening(
    *,
    data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
    project_root: str,
    signal_cache_dir: str = "",
) -> list[CombinationScore]:
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial

    from tqdm import tqdm

    from src.domain.spot.regimes import REGIME_REGISTRY
    from src.domain.spot.signals import SIGNAL_REGISTRY
    from src.domain.spot.sizing import SIZING_REGISTRY

    min_rate = float(OPT_SPOT_CONFIG.get("SPOT_COMBO_MIN_SIGNAL_RATE", 0.005))
    n_quick_floor = int(OPT_SPOT_CONFIG.get("SPOT_COMBO_QUICK_TRIALS", 40))
    n_phase1_floor = int(OPT_SPOT_CONFIG.get("SPOT_COMBO_QUICK_TRIALS_PHASE1", 10))
    prune_thr = float(OPT_SPOT_CONFIG.get("SPOT_COMBO_PRUNE_THRESHOLD", -0.5))
    top_k = int(OPT_SPOT_CONFIG.get("SPOT_COMBO_TOP_K", 3))
    bucket_each = int(OPT_SPOT_CONFIG.get("SPOT_BUCKET_TOP_EACH", 2))
    n_workers_cfg = int(OPT_SPOT_CONFIG.get("SPOT_COMBO_N_WORKERS", 0))
    worker_cap = int(os.getenv("OPT_SPOT_MAX_WORKERS", "3"))
    worker_cap = max(1, worker_cap)
    n_workers_base = n_workers_cfg if n_workers_cfg > 0 else max(1, (os.cpu_count() or 2) - 1)
    n_workers = max(1, min(n_workers_base, worker_cap))

    sizing_for_screen = sorted(
        k for k in SIZING_REGISTRY.keys() if k not in SPOT_EXCLUDED_SIZING_METHODS
    )
    combinations = list(
        itertools.product(
            sorted(SIGNAL_REGISTRY.keys()),
            sorted(REGIME_REGISTRY.keys()),
            sizing_for_screen,
        )
    )

    scores: list[CombinationScore] = []
    viable: list[tuple[str, str, str]] = []
    viable_dims: list[int] = []

    for sig, reg, siz in combinations:
        combo_space = build_combined_param_space(sig, reg, siz)
        probe_params = _build_probe_params(sig, reg, siz, tf)
        reason_bounds = params_disqualified_against_space(probe_params, combo_space)
        if reason_bounds:
            scores.append(
                CombinationScore(
                    signal=sig,
                    regime=reg,
                    sizing=siz,
                    p10_gmgr=-1e9,
                    mean_signal_rate=0.0,
                    disqualified=True,
                    reason=reason_bounds,
                )
            )
            continue

        sig_rate = measure_signal_rate(data_maps, symbols, tf, sig, mid_params=probe_params)
        if sig_rate < min_rate:
            scores.append(
                CombinationScore(
                    signal=sig,
                    regime=reg,
                    sizing=siz,
                    p10_gmgr=-1e9,
                    mean_signal_rate=sig_rate,
                    disqualified=True,
                    reason="signal_sparse",
                )
            )
            continue

        viable.append((sig, reg, siz))
        viable_dims.append(_combo_search_dim(combo_space))

    _logger.info(
        "Phase 0 complete: %d viable / %d total combinations",
        len(viable),
        len(combinations),
    )

    if not viable:
        _logger.error("Stage1: all combinations disqualified after Phase 0.")
        return []

    median_dim = float(np.median(np.asarray(viable_dims, dtype=np.float64))) if viable_dims else 1.0
    n_phase1, n_quick = _auto_combo_trials(len(viable), median_dim, OPT_SPOT_CONFIG)
    _logger.info(
        "Stage1 auto trials: phase1=%d (floor=%d), phase2=%d (floor=%d), viable=%d, median_dim=%.1f",
        n_phase1,
        n_phase1_floor,
        n_quick,
        n_quick_floor,
        len(viable),
        median_dim,
    )

    _logger.info("Warming up Numba JIT before process pool...")
    _warmup_numba(
        data_maps, symbols, tf, project_root=project_root, signal_cache_dir=signal_cache_dir
    )

    mp_ctx = _process_pool_context()

    phase1_fn = partial(
        _phase1_worker,
        data_maps=data_maps,
        symbols=symbols,
        tf=tf,
        n_trials=n_phase1,
        project_root=project_root,
        signal_cache_dir=signal_cache_dir,
    )

    _logger.info(
        "Phase 1: quick screen (%d trials × %d combos, %d workers)...",
        n_phase1,
        len(viable),
        n_workers,
    )
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_ctx) as pool:
        phase1_results = list(
            tqdm(
                pool.map(phase1_fn, viable, chunksize=2),
                total=len(viable),
                desc="[Combo Screen Phase-1]",
                unit="combo",
            )
        )

    surviving: list[tuple[str, str, str]] = []
    for combo, score in zip(viable, phase1_results, strict=True):
        sig, reg, siz = combo
        if score > prune_thr:
            surviving.append(combo)
        else:
            scores.append(
                CombinationScore(
                    signal=sig,
                    regime=reg,
                    sizing=siz,
                    p10_gmgr=score,
                    mean_signal_rate=0.0,
                    disqualified=True,
                    reason="phase1_pruned",
                )
            )

    _logger.info("Phase 1 complete: %d surviving / %d viable", len(surviving), len(viable))

    if not surviving:
        _logger.warning("Phase 1: all pruned. Relaxing threshold, keeping top-5.")
        surviving = [
            c for _, c in sorted(zip(phase1_results, viable, strict=True), reverse=True)[:5]
        ]

    phase2_boost = _ambiguity_phase2_boost(phase1_results, top_k=max(1, top_k), cfg=OPT_SPOT_CONFIG)
    if phase2_boost > 0:
        n_quick = min(
            int(OPT_SPOT_CONFIG.get("SPOT_COMBO_QUICK_TRIALS_MAX", 40)),
            n_quick + phase2_boost,
        )
        _logger.info(
            "Phase 1 boundary ambiguous: boosting phase2 trials by +%d -> %d",
            phase2_boost,
            n_quick,
        )

    phase2_fn = partial(
        _phase2_worker,
        data_maps=data_maps,
        symbols=symbols,
        tf=tf,
        n_trials=n_quick,
        project_root=project_root,
        signal_cache_dir=signal_cache_dir,
    )

    _logger.info(
        "Phase 2: full screen (%d trials × %d combos, %d workers)...",
        n_quick,
        len(surviving),
        n_workers,
    )
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_ctx) as pool:
        phase2_results = list(
            tqdm(
                pool.map(phase2_fn, surviving, chunksize=1),
                total=len(surviving),
                desc="[Combo Screen Phase-2]",
                unit="combo",
            )
        )

    scored_rows: list[tuple[tuple[str, str, str], dict[str, float]]] = list(
        zip(surviving, phase2_results, strict=True)
    )

    def _robust_key(m: dict[str, float]) -> float:
        return float(m["psr_paths"]) + float(max(0.0, m["dsr_paths"]))

    by_growth = sorted(
        scored_rows,
        key=lambda x: _p25_path_consistency_score(x[1]),
        reverse=True,
    )[: max(1, bucket_each)]
    by_robust = sorted(scored_rows, key=lambda x: _robust_key(x[1]), reverse=True)[
        : max(1, bucket_each)
    ]
    by_balance = sorted(scored_rows, key=lambda x: x[1]["objective_final"], reverse=True)[
        : max(1, bucket_each)
    ]

    seen: set[tuple[str, str, str]] = set()
    bucket_order: list[tuple[str, str, str]] = []
    for bucket in (by_growth, by_robust, by_balance):
        for combo, _m in bucket:
            if combo not in seen:
                seen.add(combo)
                bucket_order.append(combo)

    for combo in bucket_order:
        sig, reg, siz = combo
        metrics = next(m for c, m in scored_rows if c == combo)
        score = _p25_path_consistency_score(metrics)
        probe = _build_probe_params(sig, reg, siz, tf)
        sig_rate = measure_signal_rate(data_maps, symbols, tf, sig, mid_params=probe)
        scores.append(
            CombinationScore(
                signal=sig,
                regime=reg,
                sizing=siz,
                p10_gmgr=score,
                mean_signal_rate=sig_rate,
                disqualified=False,
                reason="",
            )
        )

    qualified = [s for s in scores if not s.disqualified]
    qualified.sort(key=lambda x: x.p10_gmgr, reverse=True)

    lines = [
        "",
        "=" * 71,
        " [STAGE 1. STRATEGY COMBINATION SCREENING REPORT]",
        "=" * 71,
        "▶ Qualified Combinations (growth sort: p25_log_tw / (1 + path_cv))",
        "  | Signal         | Regime         | Sizing Method  | Rate  | w_p25 |",
        "  |----------------|----------------|----------------|-------|-------|",
    ]

    for s in qualified[:10]:
        ret_str = f"{s.p10_gmgr:>8.4f}" if s.p10_gmgr > -1e6 else "  FAIL "
        lines.append(
            f"  | {s.signal:<14} | {s.regime:<14} | {s.sizing:<14} | "
            f"{s.mean_signal_rate:>5.1%} | {ret_str} |"
        )

    if not qualified:
        lines.append("  | (No qualified combinations found)                                |")

    sparse_count = sum(1 for s in scores if s.reason == "signal_sparse")
    bound_count = sum(1 for s in scores if s.reason == "kc_mult_oob")
    pruned_count = sum(1 for s in scores if s.reason == "phase1_pruned")
    negative_count = sum(1 for s in qualified if s.p10_gmgr <= 0)

    lines.extend(
        [
            "",
            "▶ Screening Summary (Exclusion Stats)",
            f"  - Signal Sparse (Rate < {min_rate:.1%}) : {sparse_count} combos",
            f"  - Phase-1 Pruned (mean path ≤ {prune_thr:.2f}%) : {pruned_count} combos",
            f"  - Negative Edge (Return <= 0%) : {negative_count} combos",
            f"  - Bound Violation (Legacy)     : {bound_count} combos",
        ]
    )

    if qualified:
        best = qualified[0]
        lines.extend(
            [
                "",
                f"※ Winning Combo for Stage 2: {best.signal} | {best.regime} | {best.sizing}",
            ]
        )

    lines.append("=" * 71)
    _logger.info("\n".join(lines))

    min_screen = float(OPT_SPOT_CONFIG.get("SPOT_COMBO_MIN_SCREEN_SCORE", 0.0))

    positive = [s for s in qualified if s.p10_gmgr > min_screen]
    out_cap = max(1, top_k, len(bucket_order))
    if not positive:
        _logger.warning(
            "No combo cleared min_screen_score. Returning bucket-union best %d.", out_cap
        )
        return qualified[:out_cap]
    return positive[:out_cap]
