from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import optuna

from config.opt_config import OPT_FUTURES_CONFIG
from config.settings import (
    FUTURES_CACHE_DIR,
    FUTURES_INITIAL_BALANCE,
)
from src.domain.futures.optimization.candidate_selector import (
    sanitize_metric_map,
)
from src.domain.futures.optimization.dashboard import (
    log_oos_regime_attribution,
    print_dual_audit_dashboard,
    print_human_dashboard,
    safe_float,
)
from src.domain.futures.optimization.evaluator import (
    calc_mdd_from_equity,
    calc_net_alpha_with_friction,
    calc_time_to_target_wealth,
    perform_online_capital_allocation,
    run_oos_margin_shared_portfolio,
)
from src.domain.futures.optimization.opt_data_utils import (
    compute_oos_regime_attribution,
    compute_regime_drift,
)
from src.domain.futures.optimization.optimizer import (
    EMBARGO_BARS,
    MLPhaseDContext,
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
from src.domain.futures.validation.tmp_md_champion import (
    collect_tmp_md_champion_gate_failures,
)
from src.domain.futures.validation.unified_gates import (
    GATE_CODE_DESCRIPTIONS,
    FuturesResearchGateInput,
    evaluate_research_gates,
)
from src.domain.futures.validation.walk_forward import (
    WalkForwardConfig,
    mirror_walk_forward_result_from_awf_user_attrs,
)

_logger: logging.Logger = logging.getLogger("final_evaluator")


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
) -> None:
    """Execute Step 5: Final OOS evaluation, research gates, and persistence."""
    _logger.info("\n" + "═" * 85)
    _logger.info(" [STEP 5/5] FINAL OOS EVALUATION & WF ADAPTATION")
    _logger.info("═" * 85)

    policy_cfg = load_portfolio_policy_config(OPT_FUTURES_CONFIG)

    _logger.debug("  [ENSEMBLE EVALUATION]")
    ensemble_curves = []
    ensemble_ports = []
    for i, res in enumerate(ensemble_results):
        m_params = finalize_strategy_portfolio_params(res["params"], policy_cfg)
        m_port = run_oos_margin_shared_portfolio(
            valid_symbols,
            args.tf,
            m_params,
            oos_data_maps,
            cache_root=FUTURES_CACHE_DIR,
            return_signal_dfs=(i == 0),
        )
        ensemble_ports.append(m_port)
        ensemble_curves.append(m_port["equity_curve"])
        m_cagr = m_port.get("cagr_pct", 0.0)
        m_mdd = m_port.get("mdd_pct", 0.0)
        _logger.debug(
            f"    Member {i+1}/{len(ensemble_results)}: CAGR={m_cagr:7.2f}% | "
            f"MDD={m_mdd:6.2f}% | Trial={res['trial'].number}"
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
        _logger.debug(f"    Member {i+1} Final Weight: {w_val:.4f}")

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
            wf_result.mean_tw_legs if hasattr(wf_result, 'mean_tw_legs') else wf_result.mean_leg_tw,
            wf_result.ergodicity_dev_pct,
        )
        ai_telemetry_payloads.append({
            "stage": "wf_ergodicity",
            "erg_dev": float(wf_result.ergodicity_dev_pct),
            "guideline": float(wf_cfg.ergodicity_guideline_pct),
            "tw_legs": [float(t) for t in wf_result.tw_legs],
            "positive_leg_ratio": float(wf_result.positive_leg_ratio),
            "worst_leg_tw": float(wf_result.worst_leg_tw),
            "leg_adaptation_logs": [dict(r) for r in wf_result.leg_adaptation_logs],
        })

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

    is_port = run_oos_margin_shared_portfolio(
        valid_symbols, args.tf, params, is_data_maps, cache_root=FUTURES_CACHE_DIR
    )
    ho_port = run_oos_margin_shared_portfolio(
        valid_symbols, args.tf, params, ho_data_maps, cache_root=FUTURES_CACHE_DIR
    )

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
                "_btc_benchmark_oos"
            ),
            (
                "IS", 
                _btc_tf_df["close"].iloc[:_btc_o0].to_numpy(dtype=np.float64), 
                "_btc_benchmark_is"
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

    # S3: Regime distribution drift detection (IS vs OOS KL divergence)
    regime_drift_info = compute_regime_drift(
        data_maps=data_maps, oos_data_maps=oos_data_maps,
        symbols=valid_symbols, tf=args.tf,
    )
    run_summary_extras["regime_drift"] = regime_drift_info

    is_cagr_v = float(is_port.get("cagr_pct", is_port.get("cagr", 0.0)))
    oos_cagr_v = float(oos_port.get("cagr_pct", oos_port.get("cagr", 0.0)))
    oos_retention = (oos_cagr_v / is_cagr_v * 100.0) if abs(is_cagr_v) > 1e-6 else 0.0
    # S1: Cap IS BTC benchmark to prevent unfair hurdle during bull-market IS periods.
    # 2023-2025 IS = crypto bull run (BTC CAGR ≈ 150%); a market-neutral strategy cannot
    # beat raw BTC buy-and-hold. Cap anchors benchmark to long-run sustainable BTC return.
    _btc_is_cap = float(OPT_FUTURES_CONFIG.get("FUTURES_IS_ALPHA_BTC_CAP_PCT", 999.0))
    if _btc_benchmark_is is not None and _btc_benchmark_is > _btc_is_cap:
        _logger.info(
            " [S1] IS BTC benchmark capped: %.1f%% → %.1f%% (FUTURES_IS_ALPHA_BTC_CAP_PCT)",
            _btc_benchmark_is, _btc_is_cap,
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
        is_survival_min_sharpe=float(
            OPT_FUTURES_CONFIG.get("FUTURES_IS_SURVIVAL_MIN_SHARPE", 1.5)
        ),
        worst_leg_log_tw=float(worst_leg),
        awf_p10_log_tw_floor=float(p10_floor),
    )
    gate_ok, _gf_codes = evaluate_research_gates(_gate_inp)
    gate_failures = list(_gf_codes)
    
    # Custom Gates
    if bool(OPT_FUTURES_CONFIG.get("FUTURES_STEP2_REGIME_DEPLOY_ENABLED", False)):
        _chop_loss = float(champion_awf_diag.get("awf_chop_loss_share", 0.0))
        _chop_trade = float(champion_awf_diag.get("awf_chop_trade_share", 0.0))
        _flip_proxy = float(champion_awf_diag.get("awf_flip_rate_proxy", 0.0))
        if _chop_loss > float(OPT_FUTURES_CONFIG.get("FUTURES_STEP2_CHOP_LOSS_SHARE_MAX", 0.60)):
            gate_failures.append("STEP2_CHOP_HEAVY_LOSS")
            gate_ok = False
        if _chop_trade > float(OPT_FUTURES_CONFIG.get("FUTURES_STEP2_CHOP_TRADE_SHARE_MAX", 0.70)):
            gate_failures.append("STEP2_CHOP_HEAVY_TRADE")
            gate_ok = False
        if _flip_proxy > float(OPT_FUTURES_CONFIG.get("FUTURES_STEP2_FLIP_RATE_PROXY_MAX", 0.75)):
            gate_failures.append("STEP2_HIGH_FLIP_PROXY")
            gate_ok = False

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

    if bool(OPT_FUTURES_CONFIG.get("FUTURES_TMP_MD_CHAMPION_GATES_ENABLED", True)):
        tmp_gf = collect_tmp_md_champion_gate_failures(
            dict(best_trial.user_attrs),
            oos_bar_rets=oos_rets,
            ann_factor=float(ann_f),
            cfg=OPT_FUTURES_CONFIG,
        )
        if tmp_gf:
            gate_failures.extend(tmp_gf)
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

    for _code in gate_failures:
        _logger.debug(" [GATE] %s — %s", _code, GATE_CODE_DESCRIPTIONS.get(_code, ""))

    # [Dual-Audit Dashboard]
    logs_dir = Path(project_root) / "logs"
    champ_path = resolve_champion_record_path(logs_dir)
    champ_m = {
        "pbo": 0.5, "p10": 0.0, "dsr": 0.0, "tw": 1.0, "cagr": 0.0, "mdd": 0.0,
        "time_2x": 999.0, "cvar": 0.0, "net_alpha": 0.0, "avg_pnl": 0.0, "pf": 1.0
    }
    if champ_path and champ_path.exists():
        try:
            with open(champ_path, encoding="utf-8") as _cf:
                _c = json.load(_cf)
            _met = _c.get("metrics", {})
            champ_m = {
                "pbo": safe_float(_met.get("pbo_paired", _met.get("pbo", 0.5)), 0.5, 1e3),
                "p10": safe_float(
                    _met.get("awf_worst_leg_log_tw", _met.get("cpcv_p10_log_tw", 0.0)), 0.0, 100.0
                ),
                "dsr": safe_float(_met.get("dsr", 0.0), 0.0, 1e3),
                "tw": safe_float(_met.get("oos_terminal_wealth", 1.0), 1.0, 1e6),
                "cagr": safe_float(_met.get("oos_cagr_pct", 0.0), 0.0, 1e5),
                "mdd": safe_float(_met.get("oos_mdd_pct", 0.0), 0.0, 1e3),
                "time_2x": safe_float(_met.get("oos_time_to_2x", 999.0), 999.0, 1e6),
                "cvar": safe_float(_met.get("oos_cvar_pct", 0.0), 0.0, 1e3),
                "net_alpha": safe_float(_met.get("oos_net_alpha_pct", 0.0), 0.0, 1e5),
                "avg_pnl": safe_float(_met.get("oos_avg_trade_pnl_pct", 0.0), 0.0, 1e5),
                "pf": safe_float(_met.get("oos_profit_factor", 1.0), 1.0, 1e3),
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

    new_m = sanitize_metric_map({
        "pbo": float(pbo_obs),
        "p10": float(champion_awf_diag.get("worst_leg", 0.0)),
        "dsr": float(champion_awf_diag.get("dsr_awf", 0.0)),
        "tw": float(meta_port.get("terminal_wealth_ratio", 1.0)),
        "cagr": float(meta_port.get("cagr_pct", 0.0)),
        "mdd": float(meta_port.get("mdd_pct", 0.0)),
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
        "oos_retention_pct": float(oos_retention),
        "is_alpha": float(is_net_alpha_v),
        "awf_chop_loss_share": float(champion_awf_diag.get("awf_chop_loss_share", 0.0)),
        "awf_chop_trade_share": float(champion_awf_diag.get("awf_chop_trade_share", 0.0)),
        "awf_flip_rate_proxy": float(champion_awf_diag.get("awf_flip_rate_proxy", 0.0)),
        "awf_turnover_cost_ratio": float(champion_awf_diag.get("awf_turnover_cost_ratio", 0.0)),
    })

    params["ensemble_members"] = [res["params"] for res in ensemble_results]
    params["meta_allocation"] = {
        "window_size": meta_window, 
        "eta": meta_eta, 
        "final_weights": final_weights.tolist()
    }
    params["is_ensemble"] = True

    gate_ok_before_champ = gate_ok
    pbo_champ_eff = float(resolve_adjusted_gates(OPT_FUTURES_CONFIG, n_ml_trials)[2])
    
    if gate_ok:
        cand_metrics = ChampionMetrics(
            cagr=float(new_m.get("cagr", 0.0)), 
            mdd=abs(float(new_m.get("mdd", 100.0))), 
            net_alpha=float(new_m.get("net_alpha", 0.0)), 
            sharpe=float(oos_sharpe_v), 
            pbo=float(new_m.get("pbo", 1.0))
        )
        allow, reason = run_champion_promotion_guard(
            Path(project_root) / "logs", 
            Path(project_root), 
            cand_metrics, 
            pbo_champ_eff, 
            bool(args.bypass_champion_guard)
        )
        if not allow:
            gate_ok = False
            _logger.warning(" [CHAMPION GUARD] HOLD reason=%s", reason)

    _verdict = (
        "PROMOTE ✅" if gate_ok 
        else ("HOLD (CHAMPION_BLOCKED) 🛡️" if gate_ok_before_champ else "HOLD (GATE_FAIL) ⚠️")
    )

    # Final result logging & saving logic...
    print_human_dashboard(
        is_port, ho_port, oos_port, _verdict, 
        benchmark_is=_btc_benchmark_is, 
        benchmark_oos=_btc_benchmark_oos, 
        meta_port=meta_port
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
