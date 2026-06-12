"""Tiered pipeline pipe-table log formatters (§9 Logging Contract).

All functions are pure (no side-effects). They return formatted pipe-table
strings intended to be passed to logging.info() by callers.

Time Complexity: O(n) where n is the length of optional detail lists.
Space Complexity: O(n) for output string construction.
"""
from __future__ import annotations

from typing import Any

from src.domain.futures.optimization.opt_config import LayeredWindow

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gate(passed: bool) -> str:
    """Return 'PASS' or 'BLOCKED' based on gate status."""
    return "PASS" if passed else "BLOCKED"


def _pct(v: float) -> str:
    """Format float as percentage string (e.g. 0.85 → '85.0%')."""
    return f"{v:.1%}"


def _fmt_date(d: Any) -> str:
    """Format a date-like object as ISO string."""
    return str(d)


# ---------------------------------------------------------------------------
# §9.1 Window table
# ---------------------------------------------------------------------------

def format_window_table(w: LayeredWindow) -> str:
    """Format LayeredWindow as a pipe-table string (§9.1).

    Args:
        w: LayeredWindow instance containing all segment boundary dates.

    Returns:
        Multi-line pipe-table string with segment start/end/duration info.

    Time Complexity: O(1).
    Space Complexity: O(1).
    """
    lines: list[str] = [
        "[WINDOW: TIERED] ------------------------------------",
        "| Segment      | Start      | End        | Duration  |",
        "| ------------ | ---------- | ---------- | --------- |",
        f"| Regime Floor | {_fmt_date(w.regime_floor):<10} | {'—':<10} | (hard LB) |",
        f"| L1 (CPCV)   | {_fmt_date(w.l1_start):<10} | {_fmt_date(w.l2_start):<10} | 18 months |",
        f"| L2 (AWF)    | {_fmt_date(w.l2_start):<10} | {_fmt_date(w.holdout_start):<10} | 12 months |",
        f"| Holdout     | {_fmt_date(w.holdout_start):<10} | {_fmt_date(w.holdout_end):<10} | 6 months  |",
        "------------------------------------------------------",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# §9.2 Layer 1 table
# ---------------------------------------------------------------------------

def format_layer1_table(
    r: Any,
    *,
    fold_details: list[dict[str, Any]] | None = None,
    per_symbol_top10: list[dict[str, Any]] | None = None,
) -> str:
    """Format Layer 1 (CPCV) result as pipe-table string (§9.2).

    Args:
        r: Layer1Result-compatible object with fields:
            mean_ic, ic_tstat, breadth, valid_coverage,
            n_valid, n_total, fold_pass_ratio, gate_passed.
        fold_details: Optional list of dicts with keys:
            fold, ic, breadth, n_valid, n_events, pass.
        per_symbol_top10: Optional list of dicts with keys:
            symbol, raw_mu, vol, t_stat, ic, valid.

    Returns:
        Multi-line pipe-table string for Layer 1 diagnostics.

    Time Complexity: O(n) where n = len(fold_details) + len(per_symbol_top10).
    Space Complexity: O(n).
    """
    n_valid: int = getattr(r, "n_valid", 0)
    n_total: int = getattr(r, "n_total", 0)
    n_trade_scope: int = getattr(r, "n_trade_scope", n_total)
    gate_str: str = _gate(r.gate_passed)

    lines: list[str] = [
        "[LAYER 1: CPCV SIGNAL VALIDATION] -------------------",
        "| Metric               | Value   | Gate  | Status      |",
        "| -------------------- | ------- | ----- | ----------- |",
        f"| Mean IC (HAC)        | {r.mean_ic:.3f}   | >0.03 | {gate_str:<11} |",
        f"| IC t-stat            | {r.ic_tstat:.2f}    | >1.96 | {'—':<11} |",
        f"| Symbol Breadth       | {r.breadth:.3f}   | >0.3  | {'—':<11} |",
        f"| Valid Coverage       | {_pct(r.valid_coverage):<7} | >80%  | {'—':<11} |",
        f"| Valid Symbols/N      | {n_valid}/{n_trade_scope:<4}   | —     | {'—':<11} |",
        f"| CPCV Fold Pass Ratio | {r.fold_pass_ratio:.3f}   | >0.6  | {'—':<11} |",
        f"| L1 Gate              | {'—':<7} | —     | {gate_str:<11} |",
        "------------------------------------------------------",
    ]

    if fold_details:
        lines.append("")
        lines.append("[CPCV FOLD DETAILS] ---------------------------------")
        lines.append("| Fold | IC     | Breadth | N Valid | N Events | Pass  |")
        lines.append("| ---- | ------ | ------- | ------- | -------- | ----- |")
        for fd in fold_details:
            pass_str: str = "PASS" if fd.get("pass") else "FAIL"
            lines.append(
                f"| {fd['fold']:<4} | {fd['ic']:.3f}  | {fd['breadth']:.3f}    | "
                f"{fd['n_valid']:<7} | {fd['n_events']:<8} | {pass_str:<5} |"
            )
        lines.append("------------------------------------------------------")

    if per_symbol_top10:
        lines.append("")
        lines.append("[PER-SYMBOL AGGREGATE] ------------------------------")
        lines.append("| Symbol | Raw Mu  | Vol     | t-stat | IC(avg) | Valid |")
        lines.append("| ------ | ------- | ------- | ------ | ------- | ----- |")
        for ps in per_symbol_top10:
            valid_str: str = "Y" if ps.get("valid") else "N"
            lines.append(
                f"| {ps['symbol']!s:<6} | {ps['raw_mu']:.3f}   | {ps['vol']:.3f}    | "
                f"{ps['t_stat']:.2f}   | {ps['ic']:.3f}  | {valid_str:<5} |"
            )
        lines.append("------------------------------------------------------")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# §9.3 Layer 2 table
# ---------------------------------------------------------------------------

def format_layer2_table(
    r: Any,
    *,
    awf_folds: list[dict[str, Any]] | None = None,
    topk_selection: list[dict[str, Any]] | None = None,
) -> str:
    """Format Layer 2 (AWF) result as pipe-table string (§9.3).

    Args:
        r: Layer2Result-compatible object with fields:
            top_k, friction_pass_pct, sharpe_hybrid, sharpe_1n,
            mdd_hybrid, mdd_1n, avg_active_positions, turnover,
            gate_passed.
        awf_folds: Optional list of dicts with keys:
            fold, sharpe, mdd, active_pos, pass.
        topk_selection: Optional list of dicts with keys:
            rank, symbol, score, selected.

    Returns:
        Multi-line pipe-table string for Layer 2 diagnostics.

    Time Complexity: O(n) where n = len(awf_folds) + len(topk_selection).
    Space Complexity: O(n).
    """
    gate_str: str = _gate(r.gate_passed)
    sharpe_vs: float = getattr(r, "sharpe_hybrid", 0.0) - getattr(r, "sharpe_1n", 0.0)
    mdd_reduced: bool = getattr(r, "mdd_hybrid", 1.0) < getattr(r, "mdd_1n", 1.0)
    mdd_reduced_str: str = "Y" if mdd_reduced else "N"

    lines: list[str] = [
        "[LAYER 2: AWF PORTFOLIO VALIDATION] -----------------",
        "| Metric                 | Value   | Gate  | Status      |",
        "| ---------------------- | ------- | ----- | ----------- |",
        f"| Top-K                  | {getattr(r, 'top_k', 0):<7} | —     | {'—':<11} |",
        f"| Friction Filter Pass%  | {_pct(getattr(r, 'friction_pass_pct', 0.0)):<7} | >50%  | {'—':<11} |",
        f"| Sharpe (Hybrid)        | {getattr(r, 'sharpe_hybrid', 0.0):.2f}    | >1.0  | {'—':<11} |",
        f"| Sharpe (1/N)           | {getattr(r, 'sharpe_1n', 0.0):.2f}    | —     | {'—':<11} |",
        f"| Sharpe vs 1/N          | {sharpe_vs:+.2f}   | >0    | {'—':<11} |",
        f"| MDD (Hybrid)           | {_pct(getattr(r, 'mdd_hybrid', 0.0)):<7} | <20%  | {'—':<11} |",
        f"| MDD (1/N)              | {_pct(getattr(r, 'mdd_1n', 0.0)):<7} | —     | {'—':<11} |",
        f"| MDD Reduced            | {mdd_reduced_str:<7} | Y     | {'—':<11} |",
        f"| Avg Active Positions   | {getattr(r, 'avg_active_positions', 0.0):.2f}    | —     | {'—':<11} |",
        f"| Turnover/rebal         | {getattr(r, 'turnover', 0.0):.3f}   | —     | {'—':<11} |",
        f"| L2 Gate                | {'—':<7} | —     | {gate_str:<11} |",
        "------------------------------------------------------",
    ]

    if awf_folds:
        lines.append("")
        lines.append("[AWF FOLD DETAILS] ----------------------------------")
        lines.append("| Fold | Sharpe | MDD     | Active Pos | Pass  |")
        lines.append("| ---- | ------ | ------- | ---------- | ----- |")
        for af in awf_folds:
            pass_str = "PASS" if af.get("pass") else "FAIL"
            lines.append(
                f"| {af['fold']:<4} | {af['sharpe']:.2f}   | {_pct(af['mdd']):<7} | "
                f"{af['active_pos']:.2f}       | {pass_str:<5} |"
            )
        lines.append("------------------------------------------------------")

    if topk_selection:
        lines.append("")
        lines.append("[TOP-K SELECTION] -----------------------------------")
        lines.append("| Rank | Symbol | Score   | Selected |")
        lines.append("| ---- | ------ | ------- | -------- |")
        for ts in topk_selection:
            sel_str: str = "Y" if ts.get("selected") else "N"
            lines.append(
                f"| {ts['rank']:<4} | {ts['symbol']!s:<6} | {ts['score']:.3f}   | {sel_str:<8} |"
            )
        lines.append("------------------------------------------------------")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# §9.4 Layer 3 table
# ---------------------------------------------------------------------------

def format_layer3_table(
    r: Any,
    *,
    ho_start: str,
    ho_end: str,
) -> str:
    """Format Layer 3 (Hold-out Backtest) result as pipe-table string (§9.4).

    Args:
        r: Layer3Result-compatible object with fields:
            cagr_hybrid, mdd_hybrid, sharpe_hybrid, mar_hybrid,
            cagr_1n, mdd_1n, sharpe_1n, mar_1n,
            cagr_vs, mdd_vs, sharpe_vs, mar_vs,
            gate_passed.
        ho_start: Hold-out start date string (ISO format).
        ho_end: Hold-out end date string (ISO format).

    Returns:
        Multi-line pipe-table string for Layer 3 hold-out diagnostics.

    Time Complexity: O(1).
    Space Complexity: O(1).
    """
    gate_str: str = _gate(r.gate_passed)

    cagr_h: float = getattr(r, "cagr_hybrid", 0.0)
    mdd_h: float = getattr(r, "mdd_hybrid", 0.0)
    sharpe_h: float = getattr(r, "sharpe_hybrid", 0.0)
    mar_h: float = getattr(r, "mar_hybrid", 0.0)

    cagr_1n: float = getattr(r, "cagr_1n", 0.0)
    mdd_1n: float = getattr(r, "mdd_1n", 0.0)
    sharpe_1n: float = getattr(r, "sharpe_1n", 0.0)
    mar_1n: float = getattr(r, "mar_1n", 0.0)

    cagr_vs: float = getattr(r, "cagr_vs", cagr_h - cagr_1n)
    mdd_vs: float = getattr(r, "mdd_vs", mdd_h - mdd_1n)
    sharpe_vs: float = getattr(r, "sharpe_vs", sharpe_h - sharpe_1n)
    mar_vs: float = getattr(r, "mar_vs", mar_h - mar_1n)

    lines: list[str] = [
        f"[LAYER 3: HOLD-OUT BACKTEST ({ho_start} ~ {ho_end})] --------",
        "| Model         |   CAGR  |  MaxDD  |  Sharpe |   MAR   | Pass        |",
        "| ------------- | ------- | ------- | ------- | ------- | ----------- |",
        f"| L1+L2 Hybrid  | {_pct(cagr_h):<7} | {_pct(mdd_h):<7}"
        f" | {sharpe_h:.2f}    | {mar_h:.2f}    | {gate_str:<11} |",
        f"| 1/N Baseline  | {_pct(cagr_1n):<7} | {_pct(mdd_1n):<7}"
        f" | {sharpe_1n:.2f}    | {mar_1n:.2f}    | {'—':<11} |",
        f"| vs Baseline   | {_pct(cagr_vs):<7} | {_pct(mdd_vs):<7}"
        f" | {sharpe_vs:.2f}    | {mar_vs:.2f}    | {gate_str:<11} |",
        f"| L3 Gate       | {'—':<7} | {'—':<7} | {'—':<7} | {'—':<7} | {gate_str:<11} |",
        "------------------------------------------------------",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# §9.5 System status table
# ---------------------------------------------------------------------------

def format_system_status(
    l1: Any,
    l2: Any | None,
    l3: Any | None,
) -> str:
    """Format overall pipeline system status as pipe-table string (§9.5).

    Args:
        l1: Layer1Result-compatible object with ``gate_passed`` attribute.
        l2: Layer2Result-compatible object or None (None → SKIP).
        l3: Layer3Result-compatible object or None (None → SKIP).

    Returns:
        Multi-line pipe-table string showing per-layer status.

    Time Complexity: O(1).
    Space Complexity: O(1).
    """
    def _layer_status(r: Any | None, *, skip_if_none: bool = True) -> tuple[str, str]:
        """Return (status, blocker) tuple for a layer result."""
        if r is None and skip_if_none:
            return "SKIP", "—"
        if r is None:
            return "SKIP", "—"
        passed: bool = getattr(r, "gate_passed", False)
        if passed:
            return "PASS", "—"
        blocker: str = getattr(r, "blocker_reason", "gate_passed=False")
        return "BLOCKED", blocker

    l1_status, l1_blocker = _layer_status(l1, skip_if_none=False)
    l2_status, l2_blocker = _layer_status(l2)
    l3_status, l3_blocker = _layer_status(l3)

    lines: list[str] = [
        "[SYSTEM STATUS] ------------------------------------",
        "| Layer   | Status  | Blocker (if any)            |",
        "| ------- | ------- | --------------------------- |",
        f"| Layer 1 | {l1_status:<7} | {l1_blocker:<27} |",
        f"| Layer 2 | {l2_status:<7} | {l2_blocker:<27} |",
        f"| Layer 3 | {l3_status:<7} | {l3_blocker:<27} |",
        "-----------------------------------------------------",
    ]
    return "\n".join(lines)
