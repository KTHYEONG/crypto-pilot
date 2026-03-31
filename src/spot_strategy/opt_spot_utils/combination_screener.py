"""Stage-1 screening: (signal × regime × sizing) combinations with fast CPCV."""
from __future__ import annotations

import itertools
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import optuna
from optuna.storages import InMemoryStorage
from optuna.samplers import TPESampler
from optuna.trial import TrialState

from config.opt_config import OPT_SPOT_CONFIG
from src.spot_strategy.opt_spot_utils.cv_utils import build_cpcv_test_paths_with_fallback
from src.spot_strategy.opt_spot_utils.evaluator import EMBARGO_BARS, objective_spot
from src.spot_strategy.opt_spot_utils.opt_params import build_combined_param_space

_logger = logging.getLogger("combination_screener")


@dataclass
class CombinationScore:
    signal: str
    regime: str
    sizing: str
    p10_gmgr: float
    mean_signal_rate: float
    disqualified: bool = False
    reason: str = ""


def _mid_value_from_spec(spec: Dict[str, Any]) -> int | float:
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


def params_disqualified_against_space(params: Dict[str, Any], space: Dict[str, Any]) -> str:
    """Return reason string if params violate declared bounds (e.g. legacy KC_MULT=3.0)."""
    if "KC_MULT" in params and "KC_MULT" in space:
        spec = space["KC_MULT"]
        if isinstance(spec, dict) and spec.get("type") == "float":
            hi = float(spec["high"])
            if float(params["KC_MULT"]) > hi + 1e-9:
                return "kc_mult_oob"
    return ""


def _build_probe_params(sig: str, reg: str, siz: str, tf: str) -> Dict[str, Any]:
    combo_space = build_combined_param_space(sig, reg, siz)
    probe_params: Dict[str, Any] = {
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
    data_maps: Dict[str, Dict[str, Any]],
    symbols: Sequence[str],
    tf: str,
    signal_name: str,
    *,
    mid_params: Dict[str, Any],
) -> float:
    from src.spot_strategy.signals import SIGNAL_REGISTRY

    ref_sym = symbols[0]
    target_df = data_maps[ref_sym][tf]
    is_off = int(data_maps[ref_sym].get(f"is_start_idx_{tf}", 0))
    df = target_df.iloc[is_off:].copy()
    if df.empty:
        return 0.0
    p = {**mid_params, "SIGNAL_TYPE": signal_name, "TIMEFRAME": tf}
    out = SIGNAL_REGISTRY[signal_name].compute(df, p)
    return float(np.mean(out.entry_signal.astype(np.float64)))


def run_quick_cpcv_for_combo(
    combo: tuple[str, str, str],
    *,
    data_maps: Dict[str, Dict[str, Any]],
    symbols: List[str],
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
        np.mean(
            [
                float(t.user_attrs.get("cpcv_mean_path_return_pct", -1e9))
                for t in top_trials
            ]
        )
    )
    return mean_score


def _phase1_worker(
    combo: tuple[str, str, str],
    *,
    data_maps: Dict[str, Dict[str, Any]],
    symbols: List[str],
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
    data_maps: Dict[str, Dict[str, Any]],
    symbols: List[str],
    tf: str,
    n_trials: int,
    project_root: str,
    signal_cache_dir: str,
) -> float:
    """Full-trial CPCV screen for surviving combinations."""
    return run_quick_cpcv_for_combo(
        combo,
        data_maps=data_maps,
        symbols=symbols,
        tf=tf,
        n_trials=n_trials,
        project_root=project_root,
        signal_cache_dir=signal_cache_dir,
    )


def _warmup_numba(
    data_maps: Dict[str, Dict[str, Any]],
    symbols: List[str],
    tf: str,
    project_root: str = "",
    signal_cache_dir: str = "",
) -> None:
    """Trigger Numba JIT in the parent before fork so children inherit compiled code."""
    from src.spot_strategy.regimes import REGIME_REGISTRY
    from src.spot_strategy.signals import SIGNAL_REGISTRY
    from src.spot_strategy.sizing import SIZING_REGISTRY

    sig = sorted(SIGNAL_REGISTRY.keys())[0]
    reg = sorted(REGIME_REGISTRY.keys())[0]
    siz = sorted(SIZING_REGISTRY.keys())[0]
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
        pass


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
    data_maps: Dict[str, Dict[str, Any]],
    symbols: List[str],
    tf: str,
    project_root: str,
    signal_cache_dir: str = "",
) -> List[CombinationScore]:
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial

    from src.spot_strategy.regimes import REGIME_REGISTRY
    from src.spot_strategy.signals import SIGNAL_REGISTRY
    from src.spot_strategy.sizing import SIZING_REGISTRY
    from tqdm import tqdm

    min_rate = float(OPT_SPOT_CONFIG.get("SPOT_COMBO_MIN_SIGNAL_RATE", 0.005))
    n_quick = int(OPT_SPOT_CONFIG.get("SPOT_COMBO_QUICK_TRIALS", 40))
    n_phase1 = int(OPT_SPOT_CONFIG.get("SPOT_COMBO_QUICK_TRIALS_PHASE1", 10))
    prune_thr = float(OPT_SPOT_CONFIG.get("SPOT_COMBO_PRUNE_THRESHOLD", -0.5))
    top_k = int(OPT_SPOT_CONFIG.get("SPOT_COMBO_TOP_K", 3))
    n_workers_cfg = int(OPT_SPOT_CONFIG.get("SPOT_COMBO_N_WORKERS", 0))
    n_workers = n_workers_cfg if n_workers_cfg > 0 else max(1, (os.cpu_count() or 2) - 1)

    combinations = list(
        itertools.product(
            sorted(SIGNAL_REGISTRY.keys()),
            sorted(REGIME_REGISTRY.keys()),
            sorted(SIZING_REGISTRY.keys()),
        )
    )

    scores: List[CombinationScore] = []
    viable: List[tuple[str, str, str]] = []

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

    _logger.info(
        "Phase 0 complete: %d viable / %d total combinations",
        len(viable),
        len(combinations),
    )

    if not viable:
        _logger.error("Stage1: all combinations disqualified after Phase 0.")
        return []

    _logger.info("Warming up Numba JIT before process pool...")
    _warmup_numba(data_maps, symbols, tf, project_root=project_root, signal_cache_dir=signal_cache_dir)

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

    surviving: List[tuple[str, str, str]] = []
    for combo, score in zip(viable, phase1_results):
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
        surviving = [c for _, c in sorted(zip(phase1_results, viable), reverse=True)[:5]]

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

    for combo, score in zip(surviving, phase2_results):
        sig, reg, siz = combo
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
        "▶ Qualified Combinations (Sorted by CPCV Return)",
        "  | Signal         | Regime         | Sizing Method  | Rate  | Return |",
        "  |----------------|----------------|----------------|-------|--------|",
    ]

    for s in qualified[:10]:
        ret_str = f"{s.p10_gmgr:>6.2f}%" if s.p10_gmgr > -1e6 else "  FAIL "
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
    if not positive:
        _logger.warning("No combo cleared min_screen_score. Returning best %d.", top_k)
        return qualified[: max(1, top_k)]
    return positive[: max(1, top_k)]
