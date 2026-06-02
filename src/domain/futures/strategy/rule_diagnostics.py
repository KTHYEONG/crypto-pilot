from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.domain.futures.strategy.candidate_labels import label_candidate_events
from src.domain.futures.strategy.common.alignment import AlignedMarketData
from src.domain.futures.strategy.config import CandidateStrategyConfig

_logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class RuleDiagnosticsResult:
    """Rule alpha diagnostics tables."""

    by_family: pd.DataFrame
    by_variant: pd.DataFrame
    by_family_side: pd.DataFrame
    side_flip: pd.DataFrame
    decision: dict[str, float | int | str]


def _safe_spearman(signal_scores: pd.Series, target_returns: pd.Series) -> float:
    """Return Spearman IC with finite fallbacks."""
    signal = pd.to_numeric(signal_scores, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    target = pd.to_numeric(target_returns, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    if signal.shape[0] != target.shape[0]:
        return 0.0

    mask = np.isfinite(signal) & np.isfinite(target)
    if int(mask.sum()) < 2:
        return 0.0
    if np.unique(signal[mask]).size < 2 or np.unique(target[mask]).size < 2:
        return 0.0

    ic = float(pd.Series(signal[mask]).corr(pd.Series(target[mask]), method="spearman"))
    return ic if np.isfinite(ic) else 0.0


def _safe_payoff_ratio(mean_mfe_bps: float, mean_mae_bps: float) -> float:
    """Return a stable favorable/adverse excursion ratio."""
    if not np.isfinite(mean_mfe_bps) or not np.isfinite(mean_mae_bps):
        return 0.0
    denom = abs(mean_mae_bps)
    if denom <= 1e-12:
        return 0.0
    return float(mean_mfe_bps / denom)


def _group_key(view: str, row: pd.Series) -> str:
    """Build a stable diagnostic key for a view/group row."""
    if view == "family":
        return f"family={row['family']}"
    if view == "variant":
        return f"variant={row['family']}:{row['variant']}"
    if view == "family_side":
        return f"family_side={row['family']}:{int(row['side'])}"
    raise ValueError(f"unsupported view: {view}")


def _candidate_action(
    *,
    n: int,
    min_obs: int,
    mean_edge_bps: float,
    pct_edge_pos: float,
    flip_delta_bps: float | None = None,
    flip_mean_edge_bps: float | None = None,
) -> str:
    """Classify a diagnostic group."""
    if n < min_obs:
        return "INSUFFICIENT_OBS"
    if np.isfinite(mean_edge_bps) and np.isfinite(pct_edge_pos):
        if mean_edge_bps > 0.0 and pct_edge_pos >= 0.50:
            return "KEEP_CANDIDATE"
        if (
            flip_delta_bps is not None
            and flip_mean_edge_bps is not None
            and mean_edge_bps < 0.0
            and flip_delta_bps >= 25.0
            and flip_mean_edge_bps > mean_edge_bps
        ):
            return "SIDE_FLIP_CANDIDATE"
    return "DROP_OR_REWORK"


def _summarize_view(
    *,
    events: pd.DataFrame,
    view: str,
    min_obs: int,
) -> pd.DataFrame:
    """Summarize a view of rule events."""
    group_cols: list[str]
    if view == "family":
        group_cols = ["family"]
    elif view == "variant":
        group_cols = ["family", "variant"]
    elif view == "family_side":
        group_cols = ["family", "side"]
    else:
        raise ValueError(f"unsupported view: {view}")

    records: list[dict[str, float | int | str]] = []
    grouped = events.groupby(group_cols, sort=False, dropna=False)
    for _, group in grouped:
        row = group.iloc[0]
        key = _group_key(view, row)
        edge = pd.to_numeric(group["edge_after_hurdle_bps"], errors="coerce").to_numpy(
            dtype=np.float64,
            copy=False,
        )
        mae = pd.to_numeric(group["mae_bps"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        mfe = pd.to_numeric(group["mfe_bps"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        raw_score = pd.to_numeric(group["raw_score"], errors="coerce")
        score_abs = raw_score.abs()
        side = pd.to_numeric(group["side"], errors="coerce").to_numpy(dtype=np.float64, copy=False)

        finite_edge = edge[np.isfinite(edge)]
        mean_edge = float(np.mean(finite_edge)) if finite_edge.size > 0 else float("nan")
        median_edge = float(np.median(finite_edge)) if finite_edge.size > 0 else float("nan")
        pct_edge_pos = float((finite_edge > 0.0).mean()) if finite_edge.size > 0 else 0.0
        p10_edge = float(np.percentile(finite_edge, 10)) if finite_edge.size > 0 else float("nan")
        p90_edge = float(np.percentile(finite_edge, 90)) if finite_edge.size > 0 else float("nan")
        mean_mae = float(np.mean(mae[np.isfinite(mae)])) if np.isfinite(mae).any() else float("nan")
        mean_mfe = float(np.mean(mfe[np.isfinite(mfe)])) if np.isfinite(mfe).any() else float("nan")

        records.append(
            {
                "group": key,
                "n": len(group),
                "long_n": int(np.sum(side > 0.0)),
                "short_n": int(np.sum(side < 0.0)),
                "mean_edge_bps": mean_edge,
                "median_edge_bps": median_edge,
                "pct_edge_pos": pct_edge_pos,
                "p10_edge_bps": p10_edge,
                "p90_edge_bps": p90_edge,
                "mean_mae_bps": mean_mae,
                "mean_mfe_bps": mean_mfe,
                "payoff_ratio": _safe_payoff_ratio(mean_mfe, mean_mae),
                "spearman_score_edge": _safe_spearman(raw_score, group["edge_after_hurdle_bps"]),
                "spearman_abs_score_edge": _safe_spearman(score_abs, group["edge_after_hurdle_bps"]),
                "candidate_action": _candidate_action(
                    n=len(group),
                    min_obs=min_obs,
                    mean_edge_bps=mean_edge,
                    pct_edge_pos=pct_edge_pos,
                ),
            }
        )

    columns = [
        "group",
        "n",
        "long_n",
        "short_n",
        "mean_edge_bps",
        "median_edge_bps",
        "pct_edge_pos",
        "p10_edge_bps",
        "p90_edge_bps",
        "mean_mae_bps",
        "mean_mfe_bps",
        "payoff_ratio",
        "spearman_score_edge",
        "spearman_abs_score_edge",
        "candidate_action",
    ]
    if not records:
        return pd.DataFrame(columns=columns)

    out = pd.DataFrame.from_records(records, columns=columns)
    return out.sort_values(["mean_edge_bps", "group"], ascending=[False, True]).reset_index(drop=True)


def _summarize_side_flip(
    *,
    original: pd.DataFrame,
    flipped: pd.DataFrame,
    view: str,
    min_obs: int,
) -> pd.DataFrame:
    """Compare original and side-flipped diagnostics for a view."""
    orig_summary = _summarize_view(events=original, view=view, min_obs=min_obs)
    flip_summary = _summarize_view(events=flipped, view=view, min_obs=min_obs)

    flip_map = flip_summary.set_index("group") if not flip_summary.empty else pd.DataFrame()
    records: list[dict[str, float | int | str]] = []
    for row in orig_summary.itertuples(index=False):
        flip_row: pd.Series | None
        if isinstance(flip_map, pd.DataFrame) and row.group in flip_map.index:
            flip_row = flip_map.loc[row.group]
        else:
            flip_row = None

        if flip_row is None or isinstance(flip_row, pd.DataFrame):
            flip_mean = float("nan")
            flip_pct = float("nan")
        else:
            flip_mean = float(flip_row["mean_edge_bps"])
            flip_pct = float(flip_row["pct_edge_pos"])

        delta = flip_mean - float(row.mean_edge_bps) if np.isfinite(flip_mean) else float("nan")
        action = _candidate_action(
            n=int(row.n),
            min_obs=min_obs,
            mean_edge_bps=float(row.mean_edge_bps),
            pct_edge_pos=float(row.pct_edge_pos),
            flip_delta_bps=delta,
            flip_mean_edge_bps=flip_mean,
        )
        records.append(
            {
                "group": row.group,
                "n": int(row.n),
                "orig_mean_edge_bps": float(row.mean_edge_bps),
                "flip_mean_edge_bps": flip_mean,
                "delta_mean_edge_bps": delta,
                "orig_pct_edge_pos": float(row.pct_edge_pos),
                "flip_pct_edge_pos": flip_pct,
                "candidate_action": action,
            }
        )

    columns = [
        "group",
        "n",
        "orig_mean_edge_bps",
        "flip_mean_edge_bps",
        "delta_mean_edge_bps",
        "orig_pct_edge_pos",
        "flip_pct_edge_pos",
        "candidate_action",
    ]
    if not records:
        return pd.DataFrame(columns=columns)

    out = pd.DataFrame.from_records(records, columns=columns)
    return out.sort_values(["delta_mean_edge_bps", "group"], ascending=[False, True]).reset_index(drop=True)


def _log_summary_block(
    *,
    summary: pd.DataFrame,
    view: str,
) -> None:
    """Emit a compact diagnostic log for a summary table."""
    for row in summary.itertuples(index=False):
        if view == "family":
            family = str(row.group).removeprefix("family=")
            _logger.info(
                "[DIAG][RULE_FAMILY] family=%s n=%d mean_edge=%.1f pct_pos=%.3f ic=%.4f action=%s",
                family,
                int(row.n),
                float(row.mean_edge_bps),
                float(row.pct_edge_pos),
                float(row.spearman_score_edge),
                str(row.candidate_action),
            )
        elif view == "variant":
            payload = str(row.group).removeprefix("variant=")
            family, variant = payload.split(":", 1)
            _logger.info(
                "[DIAG][RULE_VARIANT] family=%s variant=%s n=%d mean_edge=%.1f pct_pos=%.3f ic=%.4f action=%s",
                family,
                variant,
                int(row.n),
                float(row.mean_edge_bps),
                float(row.pct_edge_pos),
                float(row.spearman_score_edge),
                str(row.candidate_action),
            )
        elif view == "family_side":
            payload = str(row.group).removeprefix("family_side=")
            family, side = payload.split(":", 1)
            _logger.info(
                "[DIAG][RULE_FAMILY_SIDE] family=%s side=%s n=%d mean_edge=%.1f pct_pos=%.3f ic=%.4f action=%s",
                family,
                side,
                int(row.n),
                float(row.mean_edge_bps),
                float(row.pct_edge_pos),
                float(row.spearman_score_edge),
                str(row.candidate_action),
            )
        else:
            raise ValueError(f"unsupported view: {view}")


def _log_side_flip_block(side_flip: pd.DataFrame) -> None:
    """Emit a compact diagnostic log for side-flip comparisons."""
    for row in side_flip.itertuples(index=False):
        _logger.info(
            "[DIAG][RULE_SIDE_FLIP] group=%s n=%d orig_mean=%.1f flip_mean=%.1f delta=%.1f action=%s",
            str(row.group),
            int(row.n),
            float(row.orig_mean_edge_bps),
            float(row.flip_mean_edge_bps),
            float(row.delta_mean_edge_bps),
            str(row.candidate_action),
        )


def _log_decision_block(decision: dict[str, float | int | str]) -> None:
    """Emit a compact decision summary."""
    _logger.info(
        "[DIAG][RULE_DECISION] keep=%d flip=%d drop=%d insufficient=%d best_group=%s best_mean_edge=%.1f",
        int(decision.get("keep", 0)),
        int(decision.get("flip", 0)),
        int(decision.get("drop", 0)),
        int(decision.get("insufficient", 0)),
        str(decision.get("best_group", "")),
        float(decision.get("best_mean_edge", float("nan"))),
    )


def compute_rule_diagnostics(
    *,
    labeled_events: pd.DataFrame,
    aligned: AlignedMarketData,
    cfg: CandidateStrategyConfig,
    min_obs: int = 100,
) -> RuleDiagnosticsResult:
    """Compute family/variant/side diagnostics for rule alpha events."""
    if labeled_events.empty:
        empty = pd.DataFrame(
            columns=[
                "group",
                "n",
                "long_n",
                "short_n",
                "mean_edge_bps",
                "median_edge_bps",
                "pct_edge_pos",
                "p10_edge_bps",
                "p90_edge_bps",
                "mean_mae_bps",
                "mean_mfe_bps",
                "payoff_ratio",
                "spearman_score_edge",
                "spearman_abs_score_edge",
                "candidate_action",
            ]
        )
        empty_side = pd.DataFrame(
            columns=[
                "group",
                "n",
                "orig_mean_edge_bps",
                "flip_mean_edge_bps",
                "delta_mean_edge_bps",
                "orig_pct_edge_pos",
                "flip_pct_edge_pos",
                "candidate_action",
            ]
        )
        empty_decision: dict[str, float | int | str] = {
            "keep": 0,
            "flip": 0,
            "drop": 0,
            "insufficient": 0,
            "best_group": "",
            "best_mean_edge": float("nan"),
        }
        return RuleDiagnosticsResult(empty, empty, empty, empty_side, empty_decision)

    required = {
        "family",
        "variant",
        "side",
        "raw_score",
        "score_z",
        "entry_idx",
        "expected_holding_bars",
        "stop_atr_mult",
        "take_profit_atr_mult",
        "edge_after_hurdle_bps",
        "profitable_after_hurdle_label",
        "mae_bps",
        "mfe_bps",
    }
    missing = required.difference(labeled_events.columns)
    if missing:
        raise ValueError(f"missing required diagnostic columns: {sorted(missing)}")

    by_family = _summarize_view(events=labeled_events, view="family", min_obs=min_obs)
    by_variant = _summarize_view(events=labeled_events, view="variant", min_obs=min_obs)
    by_family_side = _summarize_view(events=labeled_events, view="family_side", min_obs=min_obs)

    flipped = labeled_events.copy()
    flipped["side"] = -flipped["side"]
    flipped_labeled = label_candidate_events(events=flipped, aligned=aligned, cfg=cfg)

    side_flip_frames = [
        _summarize_side_flip(original=labeled_events, flipped=flipped_labeled, view="family", min_obs=min_obs),
        _summarize_side_flip(original=labeled_events, flipped=flipped_labeled, view="variant", min_obs=min_obs),
        _summarize_side_flip(
            original=labeled_events,
            flipped=flipped_labeled,
            view="family_side",
            min_obs=min_obs,
        ),
    ]
    side_flip = pd.concat(side_flip_frames, axis=0, ignore_index=True) if side_flip_frames else pd.DataFrame()
    side_flip_lookup = {
        str(row.group): row
        for row in side_flip.itertuples(index=False)
    } if not side_flip.empty else {}

    def _apply_action(table: pd.DataFrame) -> pd.DataFrame:
        if table.empty:
            return table
        actions: list[str] = []
        for row in table.itertuples(index=False):
            flip_row = side_flip_lookup.get(str(row.group))
            action = _candidate_action(
                n=int(row.n),
                min_obs=min_obs,
                mean_edge_bps=float(row.mean_edge_bps),
                pct_edge_pos=float(row.pct_edge_pos),
                flip_delta_bps=float(flip_row.delta_mean_edge_bps) if flip_row is not None else None,
                flip_mean_edge_bps=float(flip_row.flip_mean_edge_bps) if flip_row is not None else None,
            )
            actions.append(action)
        updated = table.copy()
        updated["candidate_action"] = actions
        return updated

    by_family = _apply_action(by_family)
    by_variant = _apply_action(by_variant)
    by_family_side = _apply_action(by_family_side)

    decision_counts = by_variant["candidate_action"].value_counts() if not by_variant.empty else pd.Series(dtype=int)
    best_row = by_variant.iloc[0] if not by_variant.empty else None
    decision: dict[str, float | int | str] = {
        "keep": int(decision_counts.get("KEEP_CANDIDATE", 0)),
        "flip": int(decision_counts.get("SIDE_FLIP_CANDIDATE", 0)),
        "drop": int(decision_counts.get("DROP_OR_REWORK", 0)),
        "insufficient": int(decision_counts.get("INSUFFICIENT_OBS", 0)),
        "best_group": str(best_row.group) if best_row is not None else "",
        "best_mean_edge": float(best_row.mean_edge_bps) if best_row is not None else float("nan"),
    }

    _log_summary_block(summary=by_family, view="family")
    _log_summary_block(summary=by_variant, view="variant")
    _log_side_flip_block(side_flip=side_flip)
    _log_decision_block(decision)

    return RuleDiagnosticsResult(
        by_family=by_family,
        by_variant=by_variant,
        by_family_side=by_family_side,
        side_flip=side_flip,
        decision=decision,
    )
