from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Literal, get_args, get_origin, get_type_hints

import numpy as np
import optuna

from src.core.settings import (
    FUTURES_CACHE_DIR,
    FUTURES_INITIAL_BALANCE,
)
from src.domain.futures.optimization.candidate_selector import (
    sanitize_metric_map,
)
from src.domain.futures.optimization.common import EMBARGO_BARS
from src.domain.futures.optimization.evaluator import (
    calc_mdd_duration,
    calc_mdd_from_equity,
    calc_net_alpha_with_friction,
    calc_sortino_ratio,
    calc_time_to_target_wealth,
    perform_online_capital_allocation,
    run_oos_margin_shared_portfolio,
)
from src.domain.futures.optimization.ml_context import MLPhaseDContext
from src.domain.futures.optimization.observability.dashboard import (
    log_oos_alpha_attribution,
    log_oos_regime_attribution,
    print_dual_audit_dashboard,
    print_mechanical_dashboard,
    safe_float,
)
from src.domain.futures.optimization.opt_config import (
    OPT_FUTURES_CONFIG,
    default_ev_hurdle_bps,
)
from src.domain.futures.optimization.opt_data_utils import (
    compute_oos_regime_attribution,
    compute_regime_drift,
)
from src.domain.futures.optimization.validation import resolve_adjusted_gates
from src.domain.futures.portfolio.portfolio_optimizer import (
    finalize_strategy_portfolio_params,
    load_portfolio_policy_config,
)
from src.domain.futures.validation.champion_registry import (
    ChampionMetrics,
    resolve_champion_record_path,
    run_champion_promotion_guard,
)
from src.domain.futures.validation.gates import (
    GATE_CODE_DESCRIPTIONS,
    FuturesResearchGateInput,
    evaluate_research_gates,
)
from src.domain.futures.validation.walk_forward import (
    WalkForwardConfig,
    mirror_walk_forward_result_from_awf_user_attrs,
)

_logger: logging.Logger = logging.getLogger("final_evaluator")


def _validate_strategy_ml_member_param(field_name: str, value: Any, expected_type: Any) -> None:
    """Validate strict type/literal for StrategyMLConfig member override values."""
    origin = get_origin(expected_type)
    if origin is Literal:
        literal_values = get_args(expected_type)
        if value not in literal_values:
            raise ValueError(
                f"invalid StrategyMLConfig override for '{field_name}': "
                f"expected one of {literal_values}, got {value!r}"
            )
        return

    if expected_type is bool:
        if not isinstance(value, bool):
            raise ValueError(
                f"invalid StrategyMLConfig override for '{field_name}': "
                f"expected bool, got {type(value).__name__}"
            )
        return

    if expected_type is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"invalid StrategyMLConfig override for '{field_name}': "
                f"expected int, got {type(value).__name__}"
            )
        return

    if expected_type is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"invalid StrategyMLConfig override for '{field_name}': "
                f"expected float, got {type(value).__name__}"
            )
        return

    if expected_type is str:
        if not isinstance(value, str):
            raise ValueError(
                f"invalid StrategyMLConfig override for '{field_name}': "
                f"expected str, got {type(value).__name__}"
            )
        return


def _rebuild_member_strategy_config(
    strategy_cfg: Any,
    member_params: dict[str, Any],
) -> Any:
    """Rebuild per-member strategy config without mutating frozen dataclasses."""
    from src.domain.futures.strategy import StrategyConfig, StrategyMLConfig

    base_cfg = strategy_cfg if isinstance(strategy_cfg, StrategyConfig) else None
    base_ml_cfg = base_cfg.ml if base_cfg is not None else StrategyMLConfig()
    ml_type_hints = get_type_hints(StrategyMLConfig)
    ml_fields = {field.name: field for field in fields(StrategyMLConfig)}
    ml_updates: dict[str, Any] = {}
    unknown_keys: list[str] = []
    for key, value in member_params.items():
        field = ml_fields.get(key)
        if field is None:
            unknown_keys.append(key)
            continue
        expected_type = ml_type_hints.get(key, field.type)
        _validate_strategy_ml_member_param(key, value, expected_type)
        ml_updates[key] = value

    if unknown_keys:
        _logger.warning(
            "Ignoring unknown StrategyMLConfig override key(s): %s",
            sorted(unknown_keys),
        )

    member_ml_cfg = replace(base_ml_cfg, **ml_updates) if ml_updates else base_ml_cfg

    if base_cfg is not None:
        return replace(base_cfg, ml=member_ml_cfg)
    return StrategyConfig(name="ml_lambdamart_v1", ml=member_ml_cfg)


def _compute_expectancy_retention_pct(oos_expectancy: float, is_expectancy: float) -> float:
    if abs(float(is_expectancy)) <= 1e-9:
        return 0.0
    return float(oos_expectancy) / float(is_expectancy) * 100.0


def _infer_split_cost_source(
    split_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
) -> str:
    for sym in symbols:
        frame = split_maps.get(sym, {}).get(tf)
        if frame is None:
            continue
        if "execution_cost_bps" in frame.columns:
            return "per_symbol"
    return "fallback_global"


def _build_top5_ensemble_results(ensemble_results: list[Any]) -> list[Any]:
    if not ensemble_results:
        return []
    # Explicit v4.3 path: evaluate top-5 candidates as ensemble members.
    ranked = sorted(
        ensemble_results,
        key=lambda r: float(getattr(r.get("trial"), "value", -1e18) or -1e18),
        reverse=True,
    )
    return ranked[:5]


def _build_ensemble_evaluation_summary(
    selected_ensemble_results: list[Any],
    ensemble_ports: list[dict[str, Any]],
    meta_port: dict[str, Any],
) -> dict[str, Any]:
    members: list[dict[str, Any]] = []
    for idx, (res, port) in enumerate(
        zip(selected_ensemble_results, ensemble_ports, strict=True),
        start=1,
    ):
        members.append(
            {
                "rank": int(idx),
                "trial_number": int(getattr(res.get("trial"), "number", -1)),
                "cagr_pct": float(port.get("cagr_pct", 0.0)),
                "mdd_pct": float(port.get("mdd_pct", 0.0)),
                "terminal_wealth_ratio": float(port.get("terminal_wealth_ratio", 1.0)),
                "avg_trade_pnl_pct": float(port.get("avg_trade_pnl_pct", 0.0)),
            }
        )
    return {
        "selected_count": len(selected_ensemble_results),
        "members": members,
        "ensemble_meta": {
            "cagr_pct": float(meta_port.get("cagr_pct", 0.0)),
            "mdd_pct": float(meta_port.get("mdd_pct", 0.0)),
            "terminal_wealth_ratio": float(meta_port.get("terminal_wealth_ratio", 1.0)),
            "avg_trade_pnl_pct": float(meta_port.get("avg_trade_pnl_pct", 0.0)),
        },
    }


def _passes_champion_swap_4conditions(
    gate_ok: bool,
    new_m: dict[str, float],
    champ_m: dict[str, float],
    cand_ev_cost_ratio: float,
    champ_ev_cost_ratio: float,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if not gate_ok:
        failures.append("CHAMP_SWAP_GATE_NOT_PASSED")

    # Core-metric parity: higher-is-better for CAGR/Calmar/Sortino/TW; lower-is-better for MDD.
    core_parity_ok = (
        float(new_m.get("cagr", 0.0)) >= float(champ_m.get("cagr", 0.0))
        and float(new_m.get("calmar", 0.0)) >= float(champ_m.get("calmar", 0.0))
        and float(new_m.get("sortino", 0.0)) >= float(champ_m.get("sortino", 0.0))
        and float(new_m.get("tw", 0.0)) >= float(champ_m.get("tw", 0.0))
        and float(new_m.get("oos_retention_expectancy_pct", 0.0))
        >= float(champ_m.get("oos_retention_expectancy_pct", 0.0))
        and abs(float(new_m.get("mdd", 1e9))) <= abs(float(champ_m.get("mdd", 1e9)))
    )
    if not core_parity_ok:
        failures.append("CHAMP_SWAP_CORE_METRIC_PARITY_FAIL")

    # PBO superiority: lower overfitting probability is better.
    if float(new_m.get("pbo", 100.0)) >= float(champ_m.get("pbo", 100.0)):
        failures.append("CHAMP_SWAP_PBO_NOT_SUPERIOR")

    # EV/Cost superiority: higher expectancy with no worse EV-cost burden.
    cand_avg_pnl = float(new_m.get("avg_pnl", 0.0))
    champ_avg_pnl = float(champ_m.get("avg_pnl", 0.0))
    if cand_avg_pnl < champ_avg_pnl or float(cand_ev_cost_ratio) > float(champ_ev_cost_ratio):
        failures.append("CHAMP_SWAP_EV_COST_NOT_SUPERIOR")

    return len(failures) == 0, failures


def _safe_strategy_returns_from_port(oos_port: dict[str, Any]) -> np.ndarray:
    """Extract finite OOS strategy returns from equity curve."""
    eq = np.asarray(oos_port.get("equity_curve", []), dtype=np.float64)
    if eq.size <= 1:
        return np.zeros(0, dtype=np.float64)
    denom = np.maximum(eq[:-1], 1e-12)
    ret = np.diff(eq) / denom
    if ret.size == 0:
        return np.zeros(0, dtype=np.float64)
    return np.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)


def _build_oos_alpha_attribution_report(
    oos_port: dict[str, Any],
    oos_data_maps: dict[str, dict[str, Any]],
    symbols: list[str],
    tf: str,
) -> dict[str, Any]:
    """Build diagnostics-only OOS PnL attribution with finite fallbacks."""
    strategy_ret = _safe_strategy_returns_from_port(oos_port)
    n = int(strategy_ret.size)
    if n == 0:
        return {
            "n_obs": 0,
            "status": "fallback_empty",
            "pnl_pct": {"total": 0.0, "market": 0.0, "factor_proxy": 0.0, "residual": 0.0},
            "share_pct": {"market": 0.0, "factor_proxy": 0.0, "residual": 0.0},
        }

    market_stack: list[np.ndarray] = []
    for sym in symbols:
        df = oos_data_maps.get(sym, {}).get(tf)
        close = getattr(df, "get", lambda _k, _d=None: None)("close")
        if close is None:
            continue
        arr = np.asarray(close, dtype=np.float64)
        if arr.size <= 1:
            continue
        sym_ret = np.diff(arr) / np.maximum(arr[:-1], 1e-12)
        market_stack.append(np.nan_to_num(sym_ret, nan=0.0, posinf=0.0, neginf=0.0))

    if market_stack:
        min_len = min(len(x) for x in market_stack)
        market_mat = np.vstack([x[-min_len:] for x in market_stack])
        market_ret_full = np.nanmean(market_mat, axis=0)
    else:
        market_ret_full = np.zeros(n, dtype=np.float64)

    if market_ret_full.size == 0:
        market_ret_full = np.zeros(n, dtype=np.float64)
    if market_ret_full.size < n:
        pad = np.zeros(n - market_ret_full.size, dtype=np.float64)
        market_ret = np.concatenate([pad, market_ret_full])
    else:
        market_ret = market_ret_full[-n:]
    market_ret = np.nan_to_num(market_ret, nan=0.0, posinf=0.0, neginf=0.0)

    # Simple factor proxies: momentum(3) and reversal(1-lag sign-flip).
    mom = (market_ret + np.roll(market_ret, 1) + np.roll(market_ret, 2)) / 3.0
    mom[:2] = 0.0
    rev = -np.roll(market_ret, 1)
    rev[0] = 0.0

    x = np.column_stack([market_ret, mom, rev])
    y = strategy_ret
    valid = np.isfinite(y) & np.isfinite(x).all(axis=1)

    beta = np.zeros(3, dtype=np.float64)
    if int(np.count_nonzero(valid)) >= 5:
        try:
            beta, *_ = np.linalg.lstsq(x[valid], y[valid], rcond=None)
            beta = np.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0)
        except np.linalg.LinAlgError:
            beta = np.zeros(3, dtype=np.float64)

    market_component = beta[0] * market_ret
    factor_component = beta[1] * mom + beta[2] * rev
    residual_component = y - market_component - factor_component

    total = float(np.nansum(y) * 100.0)
    market_pnl = float(np.nansum(market_component) * 100.0)
    factor_pnl = float(np.nansum(factor_component) * 100.0)
    residual_pnl = float(np.nansum(residual_component) * 100.0)
    denom = abs(total) if abs(total) > 1e-9 else 1.0
    return {
        "n_obs": n,
        "status": "ok",
        "factor_proxy": "momentum3_reversal1",
        "beta": {
            "market": float(beta[0]),
            "momentum3": float(beta[1]),
            "reversal1": float(beta[2]),
        },
        "pnl_pct": {
            "total": safe_float(total),
            "market": safe_float(market_pnl),
            "factor_proxy": safe_float(factor_pnl),
            "residual": safe_float(residual_pnl),
        },
        "share_pct": {
            "market": safe_float(market_pnl / denom * 100.0),
            "factor_proxy": safe_float(factor_pnl / denom * 100.0),
            "residual": safe_float(residual_pnl / denom * 100.0),
        },
    }


def run_final_oos_evaluation(
    ensemble_results: list[Any],
    oos_data_maps: dict[str, dict[str, Any]],
    data_maps: dict[str, dict[str, Any]],
    valid_symbols: list[str],
    champion_awf_diag: dict[str, Any],
    args: Any,
    project_root: str | Path,
    study_ml: optuna.Study,
    run_id: str,
    ai_telemetry_payloads: list[dict[str, Any]],
    selection_summary: dict[str, Any],
    run_summary_extras: dict[str, Any],
    ml_ctx: MLPhaseDContext,
    n_ml_trials: int,
    target_seeds: list[int],
    selected_ops_profile: str,
    pbo_gate: float,
    dsr_gate: float,
    pbo_obs: float,
    dsr_obs: float,
    best_trial: optuna.trial.FrozenTrial,
    params: dict[str, Any],
    champ_stab_cv: float | None,
    stab_tmp_layer3_awf_fail: bool,
    cv_max: float,
    phase_c_diagnostics: dict[str, Any] | None = None,
) -> None:
    """Execute Step 4: Final OOS evaluation, research gates, and persistence."""
    policy_cfg = load_portfolio_policy_config(OPT_FUTURES_CONFIG)
    params = dict(params)
    params["STRATEGY_MODE"] = True

    _logger.debug("  [ENSEMBLE EVALUATION]")
    selected_ensemble_results = _build_top5_ensemble_results(ensemble_results)
    if not selected_ensemble_results:
        _logger.warning(" [ENSEMBLE] No members provided; skipping final OOS evaluation.")
        return
    if len(selected_ensemble_results) != len(ensemble_results):
        _logger.info(
            " [ENSEMBLE] Top-5 path enabled: selected %d/%d members",
            len(selected_ensemble_results),
            len(ensemble_results),
        )

    from src.domain.futures.optimization.samplers import build_ml_phase_d_params
    from src.domain.futures.strategy.builder import build_strategy_alpha
    from src.domain.futures.strategy_runtime.bridge import (
        MLPipelineOutput,
        copy_data_maps_tf_clone,
        merge_ml_output_into_data_maps,
    )

    ensemble_curves = []
    ensemble_ports = []
    ensemble_total_t0 = time.perf_counter()
    total_build_alpha_sec = 0.0
    total_merge_sec = 0.0
    total_oos_eval_sec = 0.0
    cache_hits = 0
    member_eval_cache: dict[str, dict[str, Any]] = {}
    alpha_build_signatures: set[str] = set()
    alpha_build_count = 0

    def _clone_member_port(port: dict[str, Any]) -> dict[str, Any]:
        cloned = dict(port)
        eq = cloned.get("equity_curve")
        if isinstance(eq, np.ndarray):
            cloned["equity_curve"] = eq.copy()
        return cloned

    for i, res in enumerate(selected_ensemble_results):
        member_t0 = time.perf_counter()
        m_params = build_ml_phase_d_params(dict(res["params"]), args.tf)
        m_params["STRATEGY_MODE"] = True
        m_params = finalize_strategy_portfolio_params(m_params, policy_cfg)
        param_sig = json.dumps(
            m_params,
            sort_keys=True,
            ensure_ascii=True,
            default=str,
            separators=(",", ":"),
        )
        cache_entry = member_eval_cache.get(param_sig)
        alpha_sig = json.dumps(
            {
                "tf": str(args.tf),
                "symbols": list(valid_symbols),
                "strategy_params": dict(res.get("params", {})),
            },
            sort_keys=True,
            ensure_ascii=True,
            default=str,
            separators=(",", ":"),
        )

        if cache_entry is not None:
            cache_hits += 1
            build_alpha_sec = 0.0
            merge_sec = 0.0
            oos_eval_sec = 0.0
            m_port = _clone_member_port(cache_entry["port"])
            ensemble_ports.append(m_port)
            eq_curve = m_port.get("equity_curve")
            if isinstance(eq_curve, np.ndarray):
                ensemble_curves.append(eq_curve)
            else:
                ensemble_curves.append(np.asarray(eq_curve, dtype=np.float64))
            m_cagr = m_port.get("cagr_pct", 0.0)
            m_mdd = m_port.get("mdd_pct", 0.0)
            m_trades = m_port.get("total_trades", m_port.get("trade_count", 0))
            _logger.info(
                "[ENSEMBLE_MEMBER] member=%d/%d cagr_pct=%.2f mdd_pct=%.2f "
                "trades=%d trial=%d status=cached",
                i + 1,
                len(selected_ensemble_results),
                m_cagr,
                m_mdd,
                m_trades,
                res["trial"].number,
            )
            _logger.info(
                "[ENSEMBLE_PROF] member=%d/%d status=cache_hit total_s=%.2f",
                i + 1,
                len(selected_ensemble_results),
                time.perf_counter() - member_t0,
            )
            continue

        # 1) Clone OOS maps to avoid cross-symbol / cross-member reference sharing
        m_oos_maps = copy_data_maps_tf_clone(oos_data_maps, valid_symbols, args.tf)

        # 2) Dynamically reconstruct completed alpha panel containing Virtual Refit Fold
        strategy_cfg = ml_ctx.strategy_cfg if ml_ctx is not None else None
        m_strat_cfg = _rebuild_member_strategy_config(strategy_cfg, dict(res.get("params", {})))

        # Build the completed alpha panel (including OOS Virtual Refit filling!)
        t_build_alpha = time.perf_counter()
        m_alpha_panel = build_strategy_alpha(
            data_maps=oos_data_maps,  # Pass full data maps containing complete history (IS + OOS)
            symbols=valid_symbols,
            tf=args.tf,
            cfg=m_strat_cfg,
        )
        build_alpha_sec = time.perf_counter() - t_build_alpha
        alpha_build_count += 1
        alpha_build_signatures.add(alpha_sig)
        total_build_alpha_sec += build_alpha_sec

        # 3) Merge the newly completed alpha panel into the cloned OOS maps
        t_merge = time.perf_counter()
        m_ml_out = MLPipelineOutput(alpha_panel=m_alpha_panel)
        merge_ml_output_into_data_maps(
            m_ml_out,
            m_oos_maps,
            valid_symbols,
            args.tf,
            log_tag=f"ensemble_m{i + 1}",
        )
        merge_sec = time.perf_counter() - t_merge
        total_merge_sec += merge_sec

        t_oos_eval = time.perf_counter()
        m_port = run_oos_margin_shared_portfolio(
            valid_symbols,
            args.tf,
            m_params,
            m_oos_maps,
            cache_root=FUTURES_CACHE_DIR,
            return_signal_dfs=(i == 0),
        )
        oos_eval_sec = time.perf_counter() - t_oos_eval
        total_oos_eval_sec += oos_eval_sec
        ensemble_ports.append(m_port)
        ensemble_curves.append(m_port["equity_curve"])
        member_eval_cache[param_sig] = {"port": _clone_member_port(m_port)}
        m_cagr = m_port.get("cagr_pct", 0.0)
        m_mdd = m_port.get("mdd_pct", 0.0)
        m_trades = m_port.get("total_trades", m_port.get("trade_count", 0))
        _logger.info(
            "[ENSEMBLE_MEMBER] member=%d/%d cagr_pct=%.2f mdd_pct=%.2f "
            "trades=%d trial=%d status=fresh",
            i + 1,
            len(selected_ensemble_results),
            m_cagr,
            m_mdd,
            m_trades,
            res["trial"].number,
        )
        _logger.info(
            "[ENSEMBLE_PROF] member=%d/%d build_alpha_s=%.2f merge_s=%.2f "
            "oos_eval_s=%.2f total_s=%.2f",
            i + 1,
            len(selected_ensemble_results),
            build_alpha_sec,
            merge_sec,
            oos_eval_sec,
            time.perf_counter() - member_t0,
        )

    ensemble_total_sec = time.perf_counter() - ensemble_total_t0
    member_count = max(1, len(selected_ensemble_results))
    _logger.info(
        "[ENSEMBLE_PROF_SUMMARY] members=%d total_s=%.2f avg_member_s=%.2f "
        "alpha_build_total_s=%.2f merge_total_s=%.2f oos_eval_total_s=%.2f "
        "cache_hits=%d unique_evals=%d",
        len(selected_ensemble_results),
        ensemble_total_sec,
        ensemble_total_sec / float(member_count),
        total_build_alpha_sec,
        total_merge_sec,
        total_oos_eval_sec,
        cache_hits,
        len(member_eval_cache),
    )

    # Online Capital Allocation (Meta-Strategy)
    meta_window = int(OPT_FUTURES_CONFIG.get("FUTURES_META_ALLOC_WINDOW", 24))
    meta_eta = float(OPT_FUTURES_CONFIG.get("FUTURES_META_ALLOC_ETA", 0.1))
    meta_equity, weight_history = perform_online_capital_allocation(
        ensemble_curves, float(FUTURES_INITIAL_BALANCE), window_size=meta_window, eta=meta_eta
    )

    # Log final weights
    final_weights = weight_history[-1]
    _logger.debug("  [META-ALLOCATION] EG Update Complete.")
    for i, w_val in enumerate(final_weights):
        _logger.debug(f"    Member {i + 1} Final Weight: {w_val:.4f}")

    # Calculate Meta-Strategy Metrics
    meta_final_bal = meta_equity[-1]
    meta_moic = meta_final_bal / float(FUTURES_INITIAL_BALANCE)
    meta_mdd = calc_mdd_from_equity(meta_equity)

    hours_per_bar = int(args.tf.replace("h", "")) if args.tf.endswith("h") else 4
    n_days = (len(meta_equity) * hours_per_bar) / 24.0

    try:
        exponent = 365.0 / max(n_days, 1e-3)
        log_meta_moic = math.log(max(meta_moic, 1e-9))
        meta_cagr = (math.exp(log_meta_moic * min(exponent, 100.0)) - 1.0) * 100.0
    except (ValueError, OverflowError):
        meta_cagr = 0.0

    meta_port = ensemble_ports[0].copy()
    meta_port["equity_curve"] = meta_equity
    meta_port["cagr_pct"] = meta_cagr
    meta_port["mdd_pct"] = meta_mdd
    meta_port["terminal_wealth_ratio"] = meta_moic

    oos_port = meta_port

    vcfg_block = OPT_FUTURES_CONFIG.get("FUTURES_VALIDATION_CONFIG", {})
    _emb_cfg = int(vcfg_block.get("wf_anchored_embargo_bars", -1))
    _emb = _emb_cfg if _emb_cfg >= 0 else int(EMBARGO_BARS.get(args.tf, 12))
    wf_cfg = WalkForwardConfig(
        n_legs=int(vcfg_block.get("wf_n_legs", 10)),
        purge_bars=int(vcfg_block.get("wf_purge_bars", 24)),
        min_positive_leg_ratio=float(vcfg_block.get("wf_min_positive_leg_ratio", 0.70)),
        worst_leg_tw_floor=float(vcfg_block.get("wf_worst_leg_tw_floor", 0.95)),
        mean_leg_tw_floor=float(vcfg_block.get("wf_mean_leg_tw_floor", 1.00)),
        ergodicity_guideline_pct=float(vcfg_block.get("wf_ergodicity_guideline_pct", 15.0)),
        ergodicity_hard_gate_enabled=bool(vcfg_block.get("wf_ergodicity_hard_gate_enabled", True)),
        use_anchored_awf_geometry=bool(vcfg_block.get("use_anchored_awf_geometry", False)),
        anchored_is_pool_frac=float(vcfg_block.get("wf_anchored_is_pool_frac", 0.70)),
        anchored_embargo_bars=_emb,
    )
    _erg_dev_val = 0.0
    wf_result = None
    if valid_symbols and wf_cfg.n_legs > 1:
        wf_result = mirror_walk_forward_result_from_awf_user_attrs(champion_awf_diag, wf_cfg)
        _erg_dev_val = float(wf_result.ergodicity_dev_pct)
        _logger.debug(
            " [WF] legs=%d pos_ratio=%.2f worst_tw=%.4f mean_tw=%.4f erg_dev=%.2f%%",
            len(wf_result.tw_legs),
            wf_result.positive_leg_ratio,
            wf_result.worst_leg_tw,
            wf_result.mean_tw_legs if hasattr(wf_result, "mean_tw_legs") else wf_result.mean_leg_tw,
            wf_result.ergodicity_dev_pct,
        )
        ai_telemetry_payloads.append(
            {
                "stage": "wf_ergodicity",
                "erg_dev": float(wf_result.ergodicity_dev_pct),
                "guideline": float(wf_cfg.ergodicity_guideline_pct),
                "tw_legs": [float(t) for t in wf_result.tw_legs],
                "positive_leg_ratio": float(wf_result.positive_leg_ratio),
                "worst_leg_tw": float(wf_result.worst_leg_tw),
                "leg_adaptation_logs": [dict(r) for r in wf_result.leg_adaptation_logs],
            }
        )

    # IS & Hold-out Evaluation
    is_data_maps: dict[str, dict[str, Any]] = {}
    ho_data_maps: dict[str, dict[str, Any]] = {}

    mai = ml_ctx.multi_alignment_info or {}
    alignment_offsets = mai.get("alignment_offsets", {})
    eff_len = mai.get("eff_ref_len", 0)
    ho_ratio = 0.20
    aligned_main_is_bars = max(200, int(eff_len * (1.0 - ho_ratio)))

    for sym in valid_symbols:
        sym_is_start = data_maps[sym].get(f"is_start_idx_{args.tf}", 0)
        aligned_is_start = alignment_offsets.get(sym, sym_is_start)

        is_dm = data_maps[sym].copy()
        is_dm[f"oos_start_idx_{args.tf}"] = aligned_is_start
        is_data_maps[sym] = is_dm

        ho_dm = data_maps[sym].copy()
        ho_start = min(aligned_is_start + aligned_main_is_bars, len(ho_dm[args.tf]) - 2)
        ho_dm[f"oos_start_idx_{args.tf}"] = max(aligned_is_start, ho_start)
        ho_data_maps[sym] = ho_dm

    base_strategy_cfg = ml_ctx.strategy_cfg if ml_ctx is not None else None
    split_strategy_cfg = _rebuild_member_strategy_config(
        base_strategy_cfg,
        dict(best_trial.params),
    )

    def _build_merge_eval_split(
        *,
        split_name: str,
        split_maps: dict[str, dict[str, Any]],
        split_params: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        split_maps_clone = copy_data_maps_tf_clone(split_maps, valid_symbols, args.tf)
        alpha_panel = build_strategy_alpha(
            data_maps=split_maps,
            symbols=valid_symbols,
            tf=args.tf,
            cfg=split_strategy_cfg,
        )
        merge_ml_output_into_data_maps(
            MLPipelineOutput(alpha_panel=alpha_panel),
            split_maps_clone,
            valid_symbols,
            args.tf,
            log_tag=split_name,
        )
        artifact_meta = {
            "strategy_name": str(alpha_panel.attrs.get("strategy_name", "")),
            "config_hash": str(alpha_panel.attrs.get("config_hash", "")),
            "selected_horizon": int(alpha_panel.attrs.get("selected_horizon", -1)),
            "model_family": str(alpha_panel.attrs.get("model_family", "")),
            "cost_source": _infer_split_cost_source(split_maps_clone, valid_symbols, args.tf),
            "alpha_artifact_structural_hash": str(
                alpha_panel.attrs.get("alpha_artifact_structural_hash", "")
            ),
        }
        port = run_oos_margin_shared_portfolio(
            valid_symbols,
            args.tf,
            split_params,
            split_maps_clone,
            cache_root=FUTURES_CACHE_DIR,
        )
        return port, artifact_meta

    is_port, is_meta = _build_merge_eval_split(
        split_name="is",
        split_maps=is_data_maps,
        split_params=params,
    )
    ho_port, ho_meta = _build_merge_eval_split(
        split_name="ho",
        split_maps=ho_data_maps,
        split_params=params,
    )
    oos_alpha_panel = build_strategy_alpha(
        data_maps=oos_data_maps,
        symbols=valid_symbols,
        tf=args.tf,
        cfg=split_strategy_cfg,
    )
    oos_meta = {
        "strategy_name": str(oos_alpha_panel.attrs.get("strategy_name", "")),
        "config_hash": str(oos_alpha_panel.attrs.get("config_hash", "")),
        "selected_horizon": int(oos_alpha_panel.attrs.get("selected_horizon", -1)),
        "model_family": str(oos_alpha_panel.attrs.get("model_family", "")),
        "cost_source": _infer_split_cost_source(oos_data_maps, valid_symbols, args.tf),
        "alpha_artifact_structural_hash": str(
            oos_alpha_panel.attrs.get("alpha_artifact_structural_hash", "")
        ),
    }
    split_meta_ref = is_meta
    split_meta_mismatch: list[str] = []
    for split_name, split_meta in (("ho", ho_meta), ("oos", oos_meta)):
        split_meta_mismatch.extend(
            [
                f"{split_name}:{key}"
                for key in (
                    "strategy_name",
                    "config_hash",
                    "selected_horizon",
                    "model_family",
                    "cost_source",
                    "alpha_artifact_structural_hash",
                )
                if split_meta.get(key) != split_meta_ref.get(key)
            ]
        )
    if split_meta_mismatch:
        raise RuntimeError(
            "split artifact metadata mismatch: "
            + ", ".join(split_meta_mismatch)
        )

    def _augment_port_metrics(p: dict[str, Any], hrs_tf: int) -> None:
        eq = p.get("equity_curve", np.array([float(FUTURES_INITIAL_BALANCE)]))
        ann_f = (365 * 24) / hrs_tf

        p["mdd_pct"] = calc_mdd_from_equity(eq)
        p["mdd_duration_days"] = (calc_mdd_duration(eq) * hrs_tf) / 24.0
        p["sortino"] = calc_sortino_ratio(eq, ann_f)

        cagr = float(p.get("cagr_pct", 0.0))
        mdd = abs(p["mdd_pct"])
        p["calmar"] = (cagr / mdd) if mdd > 1e-6 else (cagr / 0.1)

    hrs_tf = int(args.tf.replace("h", "")) if args.tf.endswith("h") else 4
    _augment_port_metrics(is_port, hrs_tf)
    _augment_port_metrics(ho_port, hrs_tf)
    _augment_port_metrics(oos_port, hrs_tf)
    if meta_port:
        _augment_port_metrics(meta_port, hrs_tf)
    run_summary_extras["ensemble_evaluation"] = _build_ensemble_evaluation_summary(
        selected_ensemble_results=selected_ensemble_results,
        ensemble_ports=ensemble_ports,
        meta_port=meta_port,
    )

    # Add OOS specific robustness
    oos_port["pbo_reliability"] = float(pbo_obs) * 100.0

    # [STEP 5.2/5] Final Performance Reports
    _btc_benchmark_oos: float | None = None
    _btc_benchmark_is: float | None = None
    _btc_sym = next((s for s in valid_symbols if "BTC" in s.upper()), None)
    if _btc_sym and _btc_sym in oos_data_maps:
        _hrs_tf = int(args.tf.replace("h", "")) if args.tf.endswith("h") else 4
        _btc_tf_df = oos_data_maps[_btc_sym][args.tf]
        _btc_o0 = int(oos_data_maps[_btc_sym][f"oos_start_idx_{args.tf}"])
        _slices = [
            (
                "OOS",
                _btc_tf_df["close"].iloc[_btc_o0:].to_numpy(dtype=np.float64),
                "_btc_benchmark_oos",
            ),
            (
                "IS",
                _btc_tf_df["close"].iloc[:_btc_o0].to_numpy(dtype=np.float64),
                "_btc_benchmark_is",
            ),
        ]
        for _label, _arr_slice, _out_var in _slices:
            if _arr_slice.size > 1 and _arr_slice[0] > 0:
                _years = _arr_slice.size * _hrs_tf / (365.0 * 24.0)
                if _years > 0.01:
                    _cagr = (float(_arr_slice[-1]) / float(_arr_slice[0])) ** (1.0 / _years) - 1.0
                    if _out_var == "_btc_benchmark_oos":
                        _btc_benchmark_oos = _cagr * 100.0
                    else:
                        _btc_benchmark_is = _cagr * 100.0

    is_port["symbol_names"] = valid_symbols
    ho_port["symbol_names"] = valid_symbols
    oos_port["symbol_names"] = valid_symbols
    regime_attr = compute_oos_regime_attribution(
        oos_port=oos_port, oos_data_maps=oos_data_maps, symbols=valid_symbols, tf=args.tf
    )
    oos_port["regime_attribution"] = regime_attr
    run_summary_extras["oos_regime_attribution"] = regime_attr
    log_oos_regime_attribution(regime_attr)
    alpha_attr = _build_oos_alpha_attribution_report(
        oos_port=oos_port,
        oos_data_maps=oos_data_maps,
        symbols=valid_symbols,
        tf=args.tf,
    )
    oos_port["alpha_attribution_diag"] = alpha_attr
    run_summary_extras["oos_alpha_attribution_diag"] = alpha_attr
    log_oos_alpha_attribution(alpha_attr)

    # S3: Regime distribution drift detection (IS vs OOS KL divergence)
    regime_drift_info = compute_regime_drift(
        data_maps=data_maps,
        oos_data_maps=oos_data_maps,
        symbols=valid_symbols,
        tf=args.tf,
    )
    run_summary_extras["regime_drift"] = regime_drift_info
    if phase_c_diagnostics is not None:
        run_summary_extras["phase_c_diagnostics"] = dict(phase_c_diagnostics)

    is_cagr_v = float(is_port.get("cagr_pct", is_port.get("cagr", 0.0)))
    oos_cagr_v = float(oos_port.get("cagr_pct", oos_port.get("cagr", 0.0)))
    oos_retention_cagr_aux = (oos_cagr_v / is_cagr_v * 100.0) if abs(is_cagr_v) > 1e-6 else 0.0
    is_expectancy_v = float(is_port.get("avg_trade_pnl_pct", 0.0))
    oos_expectancy_v = float(oos_port.get("avg_trade_pnl_pct", 0.0))
    oos_retention_expectancy = _compute_expectancy_retention_pct(
        oos_expectancy=oos_expectancy_v, is_expectancy=is_expectancy_v
    )
    # S1: Cap IS BTC benchmark to prevent unfair hurdle during bull-market IS periods.
    # 2023-2025 IS = crypto bull run (BTC CAGR ≈ 150%); a market-neutral strategy cannot
    # beat raw BTC buy-and-hold. Cap anchors benchmark to long-run sustainable BTC return.
    _btc_is_cap = float(OPT_FUTURES_CONFIG.get("FUTURES_IS_ALPHA_BTC_CAP_PCT", 999.0))
    if _btc_benchmark_is is not None and _btc_benchmark_is > _btc_is_cap:
        _logger.info("\n ⚖️ [BENCHMARK ADJUSTMENT]")
        _logger.info(
            "   > BTC Hurdle Capped: %.1f%% ➔ %.1f%% (FUTURES_IS_ALPHA_BTC_CAP_PCT 적용)",
            _btc_benchmark_is,
            _btc_is_cap,
        )
        _btc_benchmark_is = _btc_is_cap
    is_net_alpha_v = is_cagr_v - (_btc_benchmark_is if _btc_benchmark_is is not None else 0.0)

    rets_is = np.diff(is_port.get("equity_curve", np.array([FUTURES_INITIAL_BALANCE])))
    ann_f = (365 * 24) / hours_per_bar
    is_sharpe_v = 0.0
    if rets_is.size > 0 and np.std(rets_is) > 1e-9:
        is_sharpe_v = float(np.mean(rets_is) / np.std(rets_is)) * np.sqrt(ann_f)

    oos_eq = oos_port.get("equity_curve", np.array([FUTURES_INITIAL_BALANCE]))
    oos_rets = np.diff(oos_eq) / np.maximum(oos_eq[:-1], 1e-9)
    oos_sharpe_v = 0.0
    if oos_rets.size > 0 and np.std(oos_rets) > 1e-9:
        oos_sharpe_v = float(np.mean(oos_rets) / np.std(oos_rets)) * np.sqrt(ann_f)

    _lpf_oos = float(oos_port.get("long_pf", oos_port.get("long_profit_factor", 1.0)))
    _spf_oos = float(oos_port.get("short_pf", oos_port.get("short_profit_factor", 1.0)))
    worst_leg = float(best_trial.user_attrs.get("awf_worst_leg_log_tw", -10.0))
    p10_floor = float(OPT_FUTURES_CONFIG.get("FUTURES_AWF_P10_LOG_TW_MIN", -0.10))

    wf_fail_t = tuple(wf_result.failures) if wf_result is not None else ()

    _gate_inp = FuturesResearchGateInput(
        phase3_enabled=bool(OPT_FUTURES_CONFIG.get("FUTURES_PHASE3_HARD_GATE", True)),
        pbo_max=float(pbo_gate),
        dsr_min=float(dsr_gate),
        is_precision=0.55,
        oos_port=oos_port,
        pbo_obs=float(pbo_obs),
        dsr_obs=float(dsr_obs),
        wf_failures=wf_fail_t,
        min_is_net_alpha_pct=float(policy_cfg.min_is_net_alpha_pct),
        is_net_alpha_pct=float(is_net_alpha_v),
        min_long_pf=float(policy_cfg.min_long_pf),
        min_short_pf=float(policy_cfg.min_short_pf),
        oos_long_pf=float(_lpf_oos),
        oos_short_pf=float(_spf_oos),
        is_cagr_pct=float(is_cagr_v),
        is_sharpe=float(is_sharpe_v),
        is_survival_min_cagr=float(
            OPT_FUTURES_CONFIG.get("FUTURES_IS_SURVIVAL_MIN_CAGR_PCT", 15.0)
        ),
        is_survival_min_sharpe=float(OPT_FUTURES_CONFIG.get("FUTURES_IS_SURVIVAL_MIN_SHARPE", 1.5)),
        worst_leg_log_tw=float(worst_leg),
        awf_p10_log_tw_floor=float(p10_floor),
        oos_mdd_duration=float(oos_port.get("mdd_duration_days", 0.0)),
        max_mdd_duration=180.0,
        oos_expectancy=float(oos_expectancy_v),
        min_expectancy=float(default_ev_hurdle_bps(OPT_FUTURES_CONFIG)) / 100.0,
        is_expectancy=float(is_expectancy_v),
        min_oos_retention_expectancy_pct=float(
            OPT_FUTURES_CONFIG.get("FUTURES_OOS_RETENTION_EXPECTANCY_MIN_PCT", 50.0)
        ),
        oos_cagr_pct=float(oos_cagr_v),
        is_cagr_ref_pct=float(is_cagr_v),
    )
    gate_ok, _gf_codes = evaluate_research_gates(_gate_inp)
    gate_failures = list(_gf_codes)

    if bool(OPT_FUTURES_CONFIG.get("FUTURES_STEP4_DEPLOYABILITY_ENABLED", False)):
        _step4_chop_trade_max = float(
            OPT_FUTURES_CONFIG.get("FUTURES_STEP4_CHOP_TRADE_SHARE_MAX", 0.70)
        )
        _step4_turnover_max = float(
            OPT_FUTURES_CONFIG.get("FUTURES_STEP4_TURNOVER_COST_RATIO_MAX", 0.35)
        )
        _step4_chop_pf_floor = float(OPT_FUTURES_CONFIG.get("FUTURES_STEP4_CHOP_PF_FLOOR", 0.95))
        _chop_trade = float(champion_awf_diag.get("awf_chop_trade_share", 0.0))
        _turnover_cost_ratio = float(champion_awf_diag.get("awf_turnover_cost_ratio", 0.0))
        if _chop_trade > _step4_chop_trade_max:
            gate_failures.append("STEP4_CHOP_HEAVY_TRADE")
            gate_ok = False
        if _turnover_cost_ratio > _step4_turnover_max:
            gate_failures.append("STEP4_HIGH_TURNOVER_COST")
            gate_ok = False
        _chop_pf_raw = champion_awf_diag.get("awf_chop_pf")
        if _chop_trade > _step4_chop_trade_max and _chop_pf_raw is not None:
            _chop_pf = safe_float(_chop_pf_raw, 0.0)
            if _chop_pf < _step4_chop_pf_floor:
                gate_failures.append("STEP4_CHOP_PF_TOO_LOW")
                gate_ok = False

    if (
        bool(OPT_FUTURES_CONFIG.get("FUTURES_TMP_LAYER3_HARD_GATE", False))
        and stab_tmp_layer3_awf_fail
    ):
        gate_failures.append("TMP_LAYER3_STABILITY_LAYER1")
        gate_ok = False

    if (
        champ_stab_cv is not None
        and champ_stab_cv > cv_max
        and bool(OPT_FUTURES_CONFIG.get("FUTURES_CHAMP_STABILITY_HARD_GATE", False))
    ):
        gate_failures.append("CHAMP_STAB_CV")
        gate_ok = False

    if phase_c_diagnostics:
        phase_c_cv = safe_float(phase_c_diagnostics.get("stability_cv"), 1.0)
        phase_c_rob = safe_float(phase_c_diagnostics.get("robustness_score"), 0.0)
        _phase_c_cv_max = float(OPT_FUTURES_CONFIG.get("FUTURES_PHASE_C_STABILITY_CV_MAX", 0.35))
        _phase_c_rob_min = float(OPT_FUTURES_CONFIG.get("FUTURES_PHASE_C_ROBUSTNESS_MIN", 0.45))
        if phase_c_cv > _phase_c_cv_max:
            gate_failures.append("PHASE_C_STABILITY_CV")
            gate_ok = False
        if phase_c_rob < _phase_c_rob_min:
            gate_failures.append("PHASE_C_LOW_ROBUSTNESS")
            gate_ok = False

    for _code in gate_failures:
        _logger.debug(" [GATE] %s — %s", _code, GATE_CODE_DESCRIPTIONS.get(_code, ""))

    # [Dual-Audit Dashboard]
    logs_dir = Path(project_root) / "logs"
    champ_path = resolve_champion_record_path(logs_dir)
    champ_m = {
        "pbo": 50.0,
        "p10": 0.0,
        "dsr": 0.0,
        "tw": 1.0,
        "cagr": 0.0,
        "mdd": 0.0,
        "calmar": 0.0,
        "sortino": 0.0,
        "mdd_duration": 0.0,
        "time_2x": 999.0,
        "cvar": 0.0,
        "net_alpha": 0.0,
        "avg_pnl": 0.0,
        "pf": 1.0,
        "oos_retention_expectancy_pct": 0.0,
    }
    if champ_path and champ_path.exists():
        try:
            with open(champ_path, encoding="utf-8") as _cf:
                _c = json.load(_cf)
            _met = _c.get("metrics", {})
            champ_m = {
                "pbo": safe_float(_met.get("pbo_paired", _met.get("pbo", 0.5)), 0.5, 1.0) * 100.0,
                "p10": safe_float(
                    _met.get("awf_worst_leg_log_tw", _met.get("cpcv_p10_log_tw", 0.0)), 0.0, 100.0
                ),
                "dsr": safe_float(_met.get("dsr", 0.0), 0.0, 1e3),
                "tw": safe_float(_met.get("oos_terminal_wealth", 1.0), 1.0, 1e6),
                "cagr": safe_float(_met.get("oos_cagr_pct", 0.0), 0.0, 1e5),
                "mdd": safe_float(_met.get("oos_mdd_pct", 0.0), 0.0, 1e3),
                "calmar": safe_float(_met.get("oos_calmar", 0.0), 0.0, 1e3),
                "sortino": safe_float(_met.get("oos_sortino", 0.0), 0.0, 1e3),
                "mdd_duration": safe_float(_met.get("oos_mdd_duration_days", 0.0), 0.0, 1e5),
                "time_2x": safe_float(_met.get("oos_time_to_2x", 999.0), 999.0, 1e6),
                "cvar": safe_float(_met.get("oos_cvar_pct", 0.0), 0.0, 1e3),
                "net_alpha": safe_float(_met.get("oos_net_alpha_pct", 0.0), 0.0, 1e5),
                "avg_pnl": safe_float(_met.get("oos_avg_trade_pnl_pct", 0.0), 0.0, 1e5),
                "pf": safe_float(_met.get("oos_profit_factor", 1.0), 1.0, 1e3),
                "ev_cost_ratio": safe_float(_met.get("ev_cost_ratio", 1e3), 1e3, 1e6),
                "oos_retention_expectancy_pct": safe_float(
                    _met.get("oos_retention_expectancy_pct", 0.0), 0.0, 1e6
                ),
            }
        except Exception as _ce:
            _logger.debug("Champion metrics parse failed: %s", _ce)

    eq_arr = np.asarray(meta_port.get("equity_curve", []), dtype=np.float64)
    bpy = (24.0 / hours_per_bar) * 365.0
    if eq_arr.size > 1:
        step_log = np.log(np.clip(eq_arr[1:] / eq_arr[:-1], 1e-9, None))
        t2x_n, _ = calc_time_to_target_wealth(step_log, 2.0, bpy)
        nalpha_n = calc_net_alpha_with_friction(eq_arr, 0.0, bpy)
    else:
        t2x_n, nalpha_n = 999.0, 0.0

    new_m = sanitize_metric_map(
        {
            "pbo": float(pbo_obs) * 100.0,
            "p10": float(champion_awf_diag.get("worst_leg", 0.0)),
            "dsr": float(champion_awf_diag.get("dsr_awf", 0.0)),
            "tw": float(meta_port.get("terminal_wealth_ratio", 1.0)),
            "cagr": float(meta_port.get("cagr_pct", 0.0)),
            "mdd": float(meta_port.get("mdd_pct", 0.0)),
            "calmar": float(meta_port.get("calmar", 0.0)),
            "sortino": float(meta_port.get("sortino", 0.0)),
            "mdd_duration": float(meta_port.get("mdd_duration_days", 0.0)),
            "time_2x": float(t2x_n),
            "cvar": float(meta_port.get("cvar_pct", 0.0)),
            "net_alpha": float(nalpha_n * 100.0),
            "avg_pnl": float(oos_port.get("avg_trade_pnl_pct", 0.0)),
            "pf": float(meta_port.get("profit_factor", 1.0)),
            "is_cagr": float(is_port.get("cagr_pct", 0.0)),
            "ho_cagr": float(ho_port.get("cagr_pct", 0.0)),
            "awf_pos_frac": float(champion_awf_diag.get("awf_pos_frac", 0.0)),
            "mu_awf": float(champion_awf_diag.get("mu_log", 0.0)),
            "sig_awf": float(champion_awf_diag.get("sig_awf_diag", 0.0)),
            "plgd": float(champion_awf_diag.get("robust_val", 0.0)),
            "erg_dev": float(_erg_dev_val),
            "oos_long_pf": float(meta_port.get("long_pf", 1.0)),
            "oos_short_pf": float(meta_port.get("short_pf", 1.0)),
            # v4.3 primary retention is expectancy-based; CAGR retention kept as auxiliary.
            "oos_retention_pct": float(oos_retention_expectancy),
            "oos_retention_expectancy_pct": float(oos_retention_expectancy),
            "oos_retention_cagr_pct_aux": float(oos_retention_cagr_aux),
            "is_alpha": float(is_net_alpha_v),
            "awf_chop_loss_share": float(champion_awf_diag.get("awf_chop_loss_share", 0.0)),
            "awf_chop_trade_share": float(champion_awf_diag.get("awf_chop_trade_share", 0.0)),
            "awf_flip_rate_proxy": float(champion_awf_diag.get("awf_flip_rate_proxy", 0.0)),
            "awf_turnover_cost_ratio": float(champion_awf_diag.get("awf_turnover_cost_ratio", 0.0)),
        }
    )

    params["ensemble_members"] = [res["params"] for res in selected_ensemble_results]
    params["meta_allocation"] = {
        "window_size": meta_window,
        "eta": meta_eta,
        "final_weights": final_weights.tolist(),
    }
    params["is_ensemble"] = True

    gate_ok_before_champ = gate_ok
    pbo_champ_eff = float(resolve_adjusted_gates(OPT_FUTURES_CONFIG, n_ml_trials)[2])

    cand_ev_cost_ratio = safe_float(best_trial.user_attrs.get("ev_cost_ratio"), 1e3, 1e6)
    champ_ev_cost_ratio = safe_float(champ_m.get("ev_cost_ratio", 1e3), 1e3, 1e6)
    swap_ok, swap_failures = _passes_champion_swap_4conditions(
        gate_ok=gate_ok,
        new_m=new_m,
        champ_m=champ_m,
        cand_ev_cost_ratio=cand_ev_cost_ratio,
        champ_ev_cost_ratio=champ_ev_cost_ratio,
    )
    if not swap_ok:
        gate_ok = False
        gate_failures.extend(swap_failures)

    if gate_ok:
        cand_metrics = ChampionMetrics(
            cagr=float(new_m.get("cagr", 0.0)),
            mdd=abs(float(new_m.get("mdd", 100.0))),
            net_alpha=float(new_m.get("net_alpha", 0.0)),
            sharpe=float(oos_sharpe_v),
            pbo=float(new_m.get("pbo", 1.0)),
        )
        allow, reason = run_champion_promotion_guard(
            Path(project_root) / "logs",
            Path(project_root),
            cand_metrics,
            pbo_champ_eff,
            bool(args.bypass_champion_guard),
        )
        if not allow:
            gate_ok = False
            _logger.warning(" [CHAMPION GUARD] HOLD reason=%s", reason)

    _verdict = (
        "PROMOTE ✅"
        if gate_ok
        else ("HOLD (CHAMPION_BLOCKED) 🛡️" if gate_ok_before_champ else "HOLD (GATE_FAIL) ⚠️")
    )

    # Final result logging & saving logic...
    print_mechanical_dashboard(
        oos_port=meta_port if meta_port is not None else oos_port,
        gate_status=_verdict,
        pbo_obs=pbo_obs,
        dsr_obs=dsr_obs,
    )
    print_dual_audit_dashboard(new_m, champ_m, _verdict)

    # Persistence logic...
    res_dir = Path(project_root) / "results" / "futures"
    res_dir.mkdir(parents=True, exist_ok=True)

    if gate_ok:
        prod_json = res_dir / f"best_futures_{args.tf}.json"
        with open(prod_json, "w") as f:
            json.dump(params, f, indent=4)
        _logger.info(f" [PRODUCTION] New Champion deployed to {prod_json}")
