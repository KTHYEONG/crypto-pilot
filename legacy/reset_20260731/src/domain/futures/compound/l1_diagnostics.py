from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger(__name__)


class L1AdmissionRecorder:
    def __init__(self, path: Path | None = None) -> None:
        self._enabled = os.environ.get("L1_DEBUG") == "1"
        if path is not None:
            self._path = path
        else:
            self._path = Path("logs/l1_admission.jsonl")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _append_jsonl(self, record: dict[str, Any]) -> None:
        if not self._enabled:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError:
            _LOGGER.warning("L1AdmissionRecorder: failed to write %s, disabling", self._path)
            self._enabled = False

    def record_sleeve(
        self, *,
        signal_id: str, fold: int, cluster: int,
        beta: float, se_hac: float, se_ols_ratio: float,
        prob: float, n_obs: int, n_blocks: int, admitted: bool,
    ) -> None:
        if not self._enabled:
            return
        _LOGGER.debug(
            "[ALGO] signal_id=%s fold=%d cluster=%d beta=%.3f se_hac=%.3f "
            "se_ols_ratio=%.3f prob=%.3f n_obs=%d n_blocks=%d admitted=%s",
            signal_id, fold, cluster, beta, se_hac, se_ols_ratio,
            prob, n_obs, n_blocks, admitted,
        )
        self._append_jsonl({
            "tag": "ALGO", "signal_id": signal_id, "fold": fold,
            "cluster": cluster, "beta": round(beta, 3),
            "se_hac": round(se_hac, 3), "se_ols_ratio": round(se_ols_ratio, 3),
            "prob": round(prob, 3), "n_obs": n_obs, "n_blocks": n_blocks,
            "admitted": admitted,
        })

    def record_gate(
        self, *,
        admitted_sleeves: int, distinct_series: int, oos_bars: int,
        ann_growth: float, ann_lcb90: float, pw_block: float,
        turnover: float, cost_drag: float,
        positive_folds: int = 0, fold_growths: tuple[float, ...] = (),
        mean_abs_net: float = 0.0, admitted: bool,
    ) -> None:
        if not self._enabled:
            return
        _LOGGER.info(
            "[EVAL] admitted_sleeves=%d distinct_series=%d oos_bars=%d "
            "ann_growth=%.4f ann_lcb90=%.4f pw_block=%.2f "
            "turnover=%.4f cost_drag=%.6f positive_folds=%d mean_abs_net=%.4f admitted=%s",
            admitted_sleeves, distinct_series, oos_bars,
            ann_growth, ann_lcb90, pw_block,
            turnover, cost_drag, positive_folds, mean_abs_net, admitted,
        )
        self._append_jsonl({
            "tag": "EVAL", "admitted_sleeves": admitted_sleeves,
            "distinct_series": distinct_series, "oos_bars": oos_bars,
            "ann_growth": round(ann_growth, 4),
            "ann_lcb90": round(ann_lcb90, 4),
            "pw_block": round(pw_block, 2),
            "turnover": round(turnover, 4),
            "cost_drag": round(cost_drag, 6),
            "positive_folds": positive_folds,
            "fold_growths": [round(g, 6) for g in fold_growths],
            "mean_abs_net": round(mean_abs_net, 4),
            "admitted": admitted,
        })

    def record_regime_evidence(
        self, *,
        signal_id: str, outer_fold_id: int, regime_code: int,
        effective_blocks: int, posterior_probability: float,
        growth_lcb90: float, growth_2x_cost: float,
        robust_inner_growth: float, positive_inner_folds: int,
        scale: float, admitted: bool, reasons: tuple[str, ...],
        turnover: float = 0.0, cost_drag: float = 0.0,
        n_evidence_bars: int = 0, regime_mean_net: float = 0.0,
        carry_applied: bool = False,
    ) -> None:
        if not self._enabled:
            return
        _LOGGER.debug(
            "[REGIME] signal_id=%s fold=%d regime=%d eff_blocks=%d prob=%.3f "
            "lcb90=%.4f g2x=%.4f robust_g=%.4f pos_inner=%d scale=%.3f admitted=%s "
            "turnover=%.4f cost_drag=%.6f n_evidence_bars=%d regime_mean_net=%.6f carry=%s",
            signal_id, outer_fold_id, regime_code, effective_blocks,
            posterior_probability, growth_lcb90, growth_2x_cost,
            robust_inner_growth, positive_inner_folds, scale, admitted,
            turnover, cost_drag, n_evidence_bars, regime_mean_net, carry_applied,
        )
        self._append_jsonl({
            "tag": "REGIME",
            "signal_id": signal_id,
            "outer_fold_id": outer_fold_id,
            "regime_code": regime_code,
            "effective_blocks": effective_blocks,
            "posterior_probability": round(posterior_probability, 3),
            "growth_lcb90": round(growth_lcb90, 4),
            "growth_2x_cost": round(growth_2x_cost, 4),
            "robust_inner_growth": round(robust_inner_growth, 4),
            "positive_inner_folds": positive_inner_folds,
            "scale": round(scale, 3),
            "admitted": admitted,
            "reasons": list(reasons),
            "turnover": round(turnover, 4),
            "cost_drag": round(cost_drag, 6),
            "n_evidence_bars": n_evidence_bars,
            "regime_mean_net": round(regime_mean_net, 6),
            "carry_applied": carry_applied,
        })

    def record_leg(
        self, *,
        concept_id: str, mode: str,
        alpha_ann: float, beta_market: float,
        alpha_sharpe: float, t_alpha: float,
        breakeven_cost_bps: float, mean_turnover_per_bar: float,
        positive_folds: int, n_folds: int,
        posterior_positive: float, evidence_weight: float,
        reasons: tuple[str, ...],
        net_alpha_ann: float = 0.0, net_alpha_sharpe: float = 0.0,
        t_net_alpha: float = 0.0, critical_t: float = 0.0,
        n_tested_hypotheses: int = 1,
    ) -> None:
        if not self._enabled:
            return
        _LOGGER.info(
            "[LEG] concept_id=%s mode=%s alpha_ann=%.4f net_alpha_ann=%.4f "
            "t_alpha=%.3f t_net_alpha=%.3f critical_t=%.3f K=%d "
            "be_bps=%.1f pos_folds=%d/%d weight=%.4f reasons=%s",
            concept_id, mode, alpha_ann, net_alpha_ann,
            t_alpha, t_net_alpha, critical_t, n_tested_hypotheses,
            breakeven_cost_bps, positive_folds, n_folds, evidence_weight, reasons,
        )
        self._append_jsonl({
            "tag": "LEG",
            "concept_id": concept_id,
            "mode": mode,
            "alpha_ann": round(alpha_ann, 4),
            "net_alpha_ann": round(net_alpha_ann, 4),
            "alpha_sharpe": round(alpha_sharpe, 3),
            "net_alpha_sharpe": round(net_alpha_sharpe, 3),
            "t_alpha": round(t_alpha, 3),
            "t_net_alpha": round(t_net_alpha, 3),
            "critical_t": round(critical_t, 3),
            "n_tested_hypotheses": n_tested_hypotheses,
            "breakeven_cost_bps": round(breakeven_cost_bps, 1),
            "mean_turnover_per_bar": round(mean_turnover_per_bar, 6),
            "positive_folds": positive_folds,
            "n_folds": n_folds,
            "posterior_positive": round(posterior_positive, 3),
            "evidence_weight": round(evidence_weight, 4),
            "reasons": list(reasons),
        })

    def record_family_screen(
        self, *,
        family: str, n_signals: int, n_ic_bars: int,
        mean_ic: float, t_newey_west: float, sidak_alpha: float,
        declared_orientation: int, admitted: bool,
        reasons: tuple[str, ...] = (),
        intrinsic_turnover_per_bar: float = 0.0,
        net_growth_ann: float = 0.0,
        net_growth_probability: float = 0.0,
        edge_per_turnover_bps: float = 0.0,
        effective_horizon_hours: int = 0,
        effective_orientation: int = 0,
        effective_horizon_t_stat: float = 0.0,
    ) -> None:
        if not self._enabled:
            return
        _LOGGER.info(
            "[ALGO] tag=SCREEN family=%s n_signals=%d n_ic_bars=%d "
            "mean_ic=%.4f t_nw=%.3f sidak_alpha=%.4f "
            "declared_orientation=%d admitted=%s reasons=%s "
            "turn=%.6f net_ann=%.6f net_prob=%.4f edge_turn=%.4f "
            "eh=%d eo=%d et=%.3f",
            family, n_signals, n_ic_bars,
            mean_ic, t_newey_west, sidak_alpha,
            declared_orientation, admitted, reasons,
            intrinsic_turnover_per_bar, net_growth_ann, net_growth_probability,
            edge_per_turnover_bps,
            effective_horizon_hours, effective_orientation, effective_horizon_t_stat,
        )
        self._append_jsonl({
            "tag": "SCREEN",
            "family": family,
            "n_signals": n_signals,
            "n_ic_bars": n_ic_bars,
            "mean_ic": round(mean_ic, 4),
            "t_newey_west": round(t_newey_west, 3),
            "sidak_alpha": round(sidak_alpha, 4),
            "declared_orientation": declared_orientation,
            "admitted": admitted,
            "reasons": list(reasons),
            "intrinsic_turnover_per_bar": round(intrinsic_turnover_per_bar, 6),
            "net_growth_ann": round(net_growth_ann, 6),
            "net_growth_probability": round(net_growth_probability, 4),
            "edge_per_turnover_bps": round(edge_per_turnover_bps, 4),
            "effective_horizon_hours": effective_horizon_hours,
            "effective_orientation": effective_orientation,
            "effective_horizon_t_stat": round(effective_horizon_t_stat, 3),
        })


    def record_attribution(
        self, *,
        bottleneck_code: str,
        economic_candidate_count: int,
        capital_candidate_count: int,
        production_admitted: bool,
        shadow_admitted: bool,
        production_net_alpha_ann: float,
        shadow_net_alpha_ann: float,
        shadow_available: bool,
    ) -> None:
        if not self._enabled:
            return
        _LOGGER.info(
            "[ATTR] bottleneck=%s economic=%d capital=%d "
            "prod_admitted=%s shadow_admitted=%s "
            "prod_net_ann=%.4f shadow_net_ann=%.4f shadow_avail=%s",
            bottleneck_code, economic_candidate_count, capital_candidate_count,
            production_admitted, shadow_admitted,
            production_net_alpha_ann, shadow_net_alpha_ann, shadow_available,
        )
        self._append_jsonl({
            "tag": "ATTR",
            "bottleneck_code": bottleneck_code,
            "economic_candidate_count": economic_candidate_count,
            "capital_candidate_count": capital_candidate_count,
            "production_admitted": production_admitted,
            "shadow_admitted": shadow_admitted,
            "production_net_alpha_ann": round(production_net_alpha_ann, 4),
            "shadow_net_alpha_ann": round(shadow_net_alpha_ann, 4),
            "shadow_available": shadow_available,
        })
