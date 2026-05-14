from __future__ import annotations

import logging
import math
import re
from typing import Any

import numpy as np
import pandas as pd

from config.settings import FUTURES_INITIAL_BALANCE

_logger = logging.getLogger(__name__)

REGIME_NAMES: tuple[str, ...] = ("bull", "bear", "chop", "crisis")


def safe_float(val: Any, default: float = 0.0, clip: float | None = None) -> float:
    """Safe float conversion with optional clipping and finite check.

    Args:
        val: Value to convert.
        default: Default value if conversion fails.
        clip: Optional absolute clip limit.

    Returns:
        Converted and potentially clipped float value.

    """
    try:
        out = float(val)
    except (TypeError, ValueError):
        out = default
    if not np.isfinite(out):
        out = default
    if clip is not None:
        out = float(np.clip(out, -abs(float(clip)), abs(float(clip))))
    return out


def get_vis_width(s: str) -> int:
    """Calculate visual width of string, stripping ANSI and handling wide chars.

    Args:
        s: String to measure.

    Returns:
        Visual width in characters.

    """
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    plain = ansi_escape.sub("", s)
    w = 0
    i = 0
    while i < len(plain):
        c = plain[i]
        if c in ("\u200d", "\ufe0f"):
            i += 1
            continue
        # Heuristic for wide characters (emojis, CJK)
        if ord(c) > 0x1100 and (0x2E80 <= ord(c) <= 0x1F9FF):  # Broad range for CJK and Emojis
            w += 2
        else:
            w += 1
        i += 1
    return w


def fmt_row(text: str, width: int = 90, align: str = "left", char: str = " ") -> str:
    """Format a row with borders and correct visual padding.

    Args:
        text: Text to format.
        width: Target visual width.
        align: Text alignment ('left', 'right', 'center').
        char: Padding character.

    Returns:
        Formatted row string with borders.

    """
    v = get_vis_width(text)
    pad = max(0, width - v)
    if align == "left":
        return f"║ {text}{char * pad} ║"
    elif align == "right":
        return f"║ {char * pad}{text} ║"
    else:  # center
        l_pad = pad // 2
        r_pad = pad - l_pad
        return f"║ {char * l_pad}{text}{char * r_pad} ║"


def print_performance_report(
    title: str,
    port: dict[str, Any],
    dsr: float | None = None,
    pbo: float | None = None,
    tf: str = "1h",
    benchmark_cagr: float | None = None,
    meta_port: dict[str, Any] | None = None,
) -> None:
    """Print a detailed performance report for a portfolio.

    Args:
        title: Report title.
        port: Primary portfolio metrics dictionary.
        dsr: Deflated Sharpe Ratio.
        pbo: Probability of Backtest Overfitting.
        tf: Timeframe string.
        benchmark_cagr: Benchmark CAGR for alpha calculation.
        meta_port: Optional ensemble portfolio metrics.

    """
    # Use meta_port if available as primary, otherwise base port
    active = meta_port if meta_port else port
    eq = active.get("equity_curve", np.array([FUTURES_INITIAL_BALANCE]))
    trades = active.get("trades_df", pd.DataFrame())

    # 1. Volatility & Sharpe/Sortino
    rets = np.diff(eq) / np.maximum(eq[:-1], 1e-9)
    hrs = int(tf.replace("h", "")) if tf.endswith("h") else 4
    ann_factor = (365 * 24) / hrs

    ann_vol = np.std(rets) * np.sqrt(ann_factor) * 100.0 if rets.size > 0 else 0.0
    sharpe = (
        (np.mean(rets) / np.std(rets)) * np.sqrt(ann_factor)
        if rets.size > 0 and np.std(rets) > 1e-9
        else 0.0
    )

    # 2. PSR (Probabilistic Sharpe Ratio)
    psr = 0.5
    if rets.size >= 4:
        _sr_hat = float(np.mean(rets)) / (float(np.std(rets, ddof=1)) + 1e-12)
        _sk = float(np.nan_to_num(pd.Series(rets).skew()))
        _ex_k = float(np.nan_to_num(pd.Series(rets).kurt()))
        _denom = max(1e-12, 1.0 - _sk * _sr_hat + ((_ex_k + 2.0) / 4.0) * _sr_hat**2)
        _se_sr = math.sqrt(_denom / max(int(rets.size) - 1, 1))
        psr = float(0.5 * (1.0 + math.erf((_sr_hat / (_se_sr + 1e-12)) / math.sqrt(2.0))))

    # 3. t-stat of Avg Trade
    t_stat = 0.0
    if not trades.empty:
        pnl_arr = trades["pnl"].to_numpy()
        mu_pnl, std_pnl = np.mean(pnl_arr), np.std(pnl_arr, ddof=1)
        if std_pnl > 1e-9:
            t_stat = mu_pnl / (std_pnl / np.sqrt(len(pnl_arr)))

    # 4. Market Exposure (%)
    exposure = 0.0
    n_syms = max(1, len(active.get("symbol_names", [])))
    if not trades.empty and len(eq) > 1:
        exposure = (trades["exit_idx"] - trades["entry_idx"]).sum() / (len(eq) * n_syms)

    _logger.info("\n [%s]", title.upper())
    _logger.info(" ────────────────────────────────────────────────────────────────────────────")

    # Wealth Row
    w_row = (
        f"  Wealth  : CAGR {active['cagr_pct']:>+.1f}%  |  "
        f"MDD {active['mdd_pct']:>4.1f}%  |  "
        f"Profit Factor {active['profit_factor']:>5.2f}"
    )
    _logger.info(w_row)

    # Risk Row
    r_row = (
        f"  Risk    : Sharpe {sharpe:>5.2f} |  "
        f"Vol {ann_vol:>5.2f}% |  "
        f"Ulcer Index {active['ulcer_index']:>5.2f}"
    )
    _logger.info(r_row)

    # Stats Row
    s_row = (
        f"  Stats   : Win {active['win_rate_pct']:>4.1f}%  |  "
        f"Trades {active['total_trades']:>4d}  |  "
        f"PnL/Cost {active.get('ev_cost_ratio', 0.0):>5.2f}"
    )
    _logger.info(s_row)

    _logger.info(" ────────────────────────────────────────────────────────────────────────────")

    # Meta-Improvement (Only if ensemble)
    if meta_port:
        diff = meta_port["cagr_pct"] - port["cagr_pct"]
        trend = "(No Change)" if abs(diff) < 0.01 else (f"({diff:>+0.1f}%)")
        _logger.info(
            f"  > Meta-Improvement: CAGR {port['cagr_pct']:>4.1f}%% ➔ "
            f"{meta_port['cagr_pct']:>4.1f}%% {trend}"
        )

    # Market Coverage
    benchmark_str = (
        f" | Alpha {active['cagr_pct'] - benchmark_cagr:>+5.1f}%"
        if benchmark_cagr is not None
        else ""
    )
    _logger.info(
        f"  > Market Coverage : {exposure * 100.0:>6.2f}% Exposure | "
        f"PSR {psr:>5.4f} | t-stat {t_stat:>5.2f}{benchmark_str}"
    )
    _logger.info(" ────────────────────────────────────────────────────────────────────────────\n")


def print_human_dashboard(
    is_port: dict[str, Any],
    ho_port: dict[str, Any],
    oos_port: dict[str, Any],
    gate_status: str,
    benchmark_is: float = 0.0,
    benchmark_oos: float = 0.0,
    meta_port: dict[str, Any] | None = None,
) -> None:
    """Unified Human Dashboard for strategy performance summary (Compact V2)."""
    c_grn, c_red, c_rst, c_bld = "\033[92m", "\033[91m", "\033[0m", "\033[1m"

    _logger.info("\n 🧑💻 [HUMAN DASHBOARD: PERFORMANCE SUMMARY]")
    _logger.info(" ────────────────────────────────────────────────────────────────────────────")
    _logger.info("  METRIC          │      IS      │   Hold-Out   │    OOS (Single ➔ Meta)")
    _logger.info(" ─────────────────┼──────────────┼──────────────┼────────────────────────────")

    def get_v(p: dict[str, Any], k: str) -> float:
        return float(p.get(k, 0.0))

    metrics = [
        ("CAGR (%)", "cagr_pct", True),
        ("Net Alpha (%)", "net_alpha_pct", True),
        ("Max Drawdown (%)", "mdd_pct", True),
        ("Profit Factor", "profit_factor", False),
    ]

    for label, key, is_pct in metrics:
        suffix = "%" if is_pct else ""
        is_v, ho_v = get_v(is_port, key), get_v(ho_port, key)
        oos_v = get_v(oos_port, key)

        if key == "net_alpha_pct":
            is_v = get_v(is_port, "cagr_pct") - benchmark_is
            oos_v = get_v(oos_port, "cagr_pct") - benchmark_oos

        val_str = f"{oos_v:>8.1f}{suffix}"
        if meta_port:
            m_v = (
                get_v(meta_port, "cagr_pct") - benchmark_oos
                if key == "net_alpha_pct"
                else get_v(meta_port, key)
            )
            val_str = f"{oos_v:>6.1f}{suffix}  ➔  {m_v:>5.1f}{suffix}"

        _logger.info(f"  {label:<15} │ {is_v:>10.2f}{suffix} │ {ho_v:>10.2f}{suffix} │ {val_str}")

    _logger.info(" ────────────────────────────────────────────────────────────────────────────")

    is_cagr = get_v(is_port, "cagr_pct")
    oos_cagr = get_v(oos_port, "cagr_pct")
    retention = (oos_cagr / is_cagr * 100.0) if is_cagr > 1e-6 else 0.0

    ret_info = f"{retention:.1f}% of IS Performance"
    if meta_port:
        meta_cagr = get_v(meta_port, "cagr_pct")
        meta_ret = (meta_cagr / is_cagr * 100.0) if is_cagr > 1e-6 else 0.0
        ret_info = (
            f"{meta_ret:.1f}% of IS Performance (Meta-Gain: {meta_ret - retention:>+0.1f}%)"
        )

    v_color = c_grn if "PROMOTE" in gate_status else c_red
    persisted = "" if "PROMOTE" in gate_status else " - Parameters NOT persisted"

    _logger.info(f"  > OOS Retention : {ret_info}")
    _logger.info(f"  > FINAL VERDICT : {v_color}{c_bld}{gate_status}{c_rst}{persisted}")
    _logger.info(" ────────────────────────────────────────────────────────────────────────────\n")


def print_dual_audit_dashboard(
    new_m: dict[str, Any],
    champ_m: dict[str, Any],
    gate_status: str,
) -> None:
    """SOTA Dashboard for Strategy Promotion Audit (Compact V2)."""
    c_grn, c_red, c_rst, c_bld = "\033[92m", "\033[91m", "\033[0m", "\033[1m"

    _logger.info("\n 🛡️ [STRATEGY AUDIT: CANDIDATE vs CHAMPION]")
    _logger.info(" ────────────────────────────────────────────────────────────────────────────")
    _logger.info("  CATEGORY (OOS)  │  CHAMPION    │  CANDIDATE   │  DELTA (Δ)")
    _logger.info(" ─────────────────┼──────────────┼──────────────┼────────────────────────────")

    def log_row(cat, met, c_val, n_val, is_pct=False, low_better=False):
        c_v = safe_float(c_val)
        n_v = safe_float(n_val)
        diff = n_v - c_v
        suffix = "%" if is_pct else ""

        good = (diff < 0) if low_better else (diff > 0)
        mark = f"{c_grn}▲{c_rst}" if good else f"{c_red}▼{c_rst}"
        if abs(diff) < 1e-7:
            mark = "─"

        c_str = f"{c_v:.2f}{suffix}" if is_pct else f"{c_v:.4f}"
        n_str = f"{n_v:.2f}{suffix}" if is_pct else f"{n_v:.4f}"
        d_str = f"{diff:+.2f}{suffix}" if is_pct else f"{diff:+.4f}"

        _logger.info(f"  {met:<15} │ {c_str:>10}   │ {n_str:>10}   │ {d_str:>8} ({mark})")

    log_row(
        "REL",
        "PBO (Reliability)",
        champ_m.get("pbo", 0.5),
        new_m.get("pbo", 0.5),
        low_better=True,
    )
    log_row("COM", "CAGR (%)", champ_m.get("cagr", 0.0), new_m.get("cagr", 0.0), is_pct=True)
    log_row(
        "COM",
        "Max Drawdown (%)",
        champ_m.get("mdd", 0.0),
        new_m.get("mdd", 0.0),
        is_pct=True,
        low_better=True,
    )
    log_row("COM", "Profit Factor", champ_m.get("pf", 1.0), new_m.get("pf", 1.0))
    log_row(
        "COM",
        "Net Alpha (%)",
        champ_m.get("net_alpha", 0.0),
        new_m.get("net_alpha", 0.0),
        is_pct=True,
    )

    _logger.info(" ────────────────────────────────────────────────────────────────────────────")

    is_alpha = new_m.get("is_alpha", 0.0)
    ho_cagr = new_m.get("ho_cagr", 0.0)
    is_cagr = new_m.get("is_cagr", 0.0)
    retention = (new_m.get("cagr", 0.0) / is_cagr * 100.0) if abs(is_cagr) > 1e-6 else 0.0

    s_tag = f"{c_grn}PASS{c_rst}" if is_alpha >= 0 else f"{c_red}FAIL{c_rst}"
    _logger.info(
        f"  > Sanity Check  : IS Alpha {is_alpha:.1f}% ({s_tag}) | "
        f"HO CAGR {ho_cagr:.1f}% | OOS Ret {retention:.1f}%"
    )

    v_color = c_grn if "PROMOTE" in gate_status else c_red
    _logger.info(f"  > FINAL VERDICT : {v_color}{c_bld}{gate_status}{c_rst}")
    _logger.info(" ────────────────────────────────────────────────────────────────────────────\n")


def log_oos_regime_attribution(attr: dict[str, Any]) -> None:
    """Log out-of-sample regime performance attribution.

    Args:
        attr: Dictionary containing regime metrics and chop diagnostics.

    """
    _logger.info(" [OOS REGIME ATTRIBUTION]")
    regime_metrics = attr.get("regime_metrics", {})
    for rn in REGIME_NAMES:
        m = regime_metrics.get(rn, {})
        _logger.info(
            "   %-6s time=%6.2f%% trades=%4d win=%6.2f%% pf=%5.2f avg_pnl=% .6f",
            rn.upper(),
            float(m.get("time_pct", 0.0)),
            int(m.get("trade_count", 0)),
            float(m.get("win_rate", 0.0)),
            float(m.get("profit_factor", 1.0)),
            float(m.get("avg_pnl", 0.0)),
        )
    _logger.info(
        "   CHOP diagnostics: loss_share=%.2f%% trade_share=%.2f%% flip_proxy=%.3f (%s) "
        "coverage=%.2f%%",
        float(attr.get("chop_loss_share", 0.0)) * 100.0,
        float(attr.get("chop_trade_share", 0.0)) * 100.0,
        float(attr.get("chop_flip_proxy", 0.0)),
        str(attr.get("chop_flip_proxy_label", "proxy")),
        float(attr.get("trade_regime_coverage_pct", 0.0)),
    )


def log_hmm_report_summary(h_rep: dict[str, Any]) -> None:
    """Log a summary of HMM regime probabilities and stats.

    Args:
        h_rep: Dictionary with HMM report data.

    """
    bull_calm = float(h_rep.get("hmm_prob_bull_calm", 0.0))
    bull_vol_up = float(h_rep.get("hmm_prob_bull_vol_up", 0.0))
    bear = float(h_rep.get("hmm_prob_bear_trend", 0.0))
    chop = float(h_rep.get("hmm_prob_chop", 0.0))
    crisis = float(h_rep.get("hmm_prob_crisis", 0.0))
    tail_capture = float(h_rep.get("hmm_tail_capture", 0.0))
    avg_duration = float(h_rep.get("hmm_avg_duration", 0.0))
    switches = round(float(h_rep.get("hmm_switches", 0.0)))

    _logger.info(
        " [HMM SUMMARY] 🐂Bull:%.1f%% 🚀Rally:%.1f%% 🐻Bear:%.1f%% 🎢Chop:%.1f%% 💀Crisis:%.1f%%",
        bull_calm,
        bull_vol_up,
        bear,
        chop,
        crisis,
    )
    _logger.info(
        " [HMM SUMMARY] Tail-Capture: %.1f%% | Avg-Duration: %.1f bars | Switches: %d",
        tail_capture,
        avg_duration,
        switches,
    )


def feature_slice_stats(series: pd.Series) -> tuple[float, float, float]:
    """Calculate statistics for a feature slice.

    Args:
        series: Pandas Series to analyze.

    Returns:
        Tuple of (std_dev, nan_percentage, zero_percentage).

    """
    arr = pd.to_numeric(series, errors="coerce")
    n = max(int(arr.shape[0]), 1)
    nan_pct = float(arr.isna().mean()) * 100.0
    zero_pct = float((arr.notna() & (arr == 0.0)).mean()) * 100.0
    std_v = float(arr.std(ddof=0)) if n > 0 else 0.0
    return std_v, nan_pct, zero_pct


def log_ml_merge_feature_stats(
    oos_data_maps: dict[str, dict[str, Any]],
    valid_symbols: list[str],
    tf: str,
) -> None:
    """Log feature statistics for ML data merging.

    Args:
        oos_data_maps: Map of symbol to data dictionaries.
        valid_symbols: List of symbols to log.
        tf: Timeframe string.

    """
    cols = ("ml_alpha_00", "xs_score_long", "hmm_modulator_long")
    for col in cols:
        for sym in valid_symbols[: min(8, len(valid_symbols))]:
            df = oos_data_maps[sym][tf]
            if col not in df.columns:
                _logger.warning("[ML_MERGE] %s missing column %s", sym, col)
                continue
            o0 = int(oos_data_maps[sym][f"oos_start_idx_{tf}"])
            is_ser, oos_ser = df[col].iloc[:o0], df[col].iloc[o0:]
            is_std, is_nan, is_z = feature_slice_stats(is_ser)
            oos_std, oos_nan, oos_z = feature_slice_stats(oos_ser)
            _logger.debug(
                "[ML_MERGE] %s %s IS std=%.6f nan%%=%.2f zero%%=%.2f | "
                "OOS std=%.6f nan%%=%.2f zero%%=%.2f",
                sym,
                col,
                is_std,
                is_nan,
                is_z,
                oos_std,
                oos_nan,
                oos_z,
            )


def log_alpha_component_summary(alpha_panel: pd.DataFrame) -> None:
    """Log a summary of ML alpha component extraction and filtering results.

    Args:
        alpha_panel: Dataframe containing alpha components with 'attrs' metadata.

    """
    meta = alpha_panel.attrs.get("alpha_component_filter", {})
    if not meta:
        _logger.warning(" [WARN] No alpha component metadata found for summary.")
        return

    n_surv = int(meta.get("n_surviving", 0))
    n_total = int(meta.get("n_components", 0))
    surv_rate = (n_surv / n_total * 100.0) if n_total > 0 else 0.0

    ic_map = meta.get("ic_by_slot", {})
    primary_ic = float(ic_map.get("ml_alpha_00", 0.0))
    
    # Calculate pool avg IC for surviving components
    surviving_cols = [
        c for c in alpha_panel.columns if c.startswith("ml_alpha_") and c[-2:].isdigit()
    ]
    surviving_ics = [
        float(ic_map.get(c, 0.0)) for c in surviving_cols if alpha_panel[c].std() > 1e-9
    ]
    pool_avg_ic = float(np.mean(surviving_ics)) if surviving_ics else 0.0
    pos_ic_ratio = (
        (sum(1 for ic in surviving_ics if ic > 0) / len(surviving_ics) * 100.0)
        if surviving_ics
        else 0.0
    )

    _logger.info("\n [ALPHA COMPONENT SUMMARY]")
    _logger.info(" ────────────────────────────────────────────────────────────────────────────")
    _logger.info(
        "  🧬 Survival   : %d / %d (%.1f%%)",
        n_surv,
        n_total,
        surv_rate,
    )
    _logger.info(
        "  📊 Primary IC : %.4f",
        primary_ic,
    )
    _logger.info(
        "  📈 Pool Avg IC: %.4f (IC > 0: %.1f%%)",
        pool_avg_ic,
        pos_ic_ratio,
    )
    
    # Filter failure breakdown
    fail_parts = []
    for k, label in [
        ("fail_fdr", "FDR"),
        ("fail_dsr", "DSR"),
        ("fail_sym_bal", "Bal"),
        ("fail_regime", "Regime"),
        ("fail_tail", "Tail"),
    ]:
        val = int(meta.get(k, 0))
        if val > 0:
            fail_parts.append(f"{label}: {val}")
    
    if fail_parts:
        _logger.info("  🛡️ Filter Fail: %s", " | ".join(fail_parts))
    
    _logger.info(" ────────────────────────────────────────────────────────────────────────────")
