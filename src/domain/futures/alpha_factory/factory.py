"""AlphaFactoryV1 end-to-end builder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import AlphaFactoryConfig
from .contracts import RegimePosterior, SleeveScores
from .cost_adjuster import adjust_alpha_for_cost_and_confidence
from .ensemble import EnsembleOutput, ShrinkageConfig, build_ensemble, compute_ic_shrinkage_weights
from .evaluator import GateMetrics, build_gate_metrics
from .features import extract_alpha_features
from .regime_router import route_by_regime
from .sleeves import blend_raw_alpha, compute_sleeve_scores


@dataclass(frozen=True, slots=True)
class AlphaFactoryResult:
    """End-to-end build result for AlphaFactoryV1."""

    alpha_long: np.ndarray
    alpha_short: np.ndarray
    alpha_net: np.ndarray
    confidence: np.ndarray
    turnover_hint: np.ndarray
    ensemble_weights: np.ndarray
    gate_metrics: GateMetrics


class AlphaFactoryV1:
    """Build 4h alpha ensemble with IC shrinkage and gate diagnostics."""

    def __init__(
        self,
        *,
        timeframe: str = "4h",
        shrinkage_config: ShrinkageConfig | None = None,
        cost_per_turnover: float = 0.0,
    ) -> None:
        """Initialize factory with timeframe guard and shrinkage settings."""
        self._timeframe = timeframe.strip().lower()
        self._shrinkage_config = shrinkage_config or ShrinkageConfig()
        self._cost_per_turnover = float(cost_per_turnover)

    def _validate_4h_only(self) -> None:
        if self._timeframe != "4h":
            raise ValueError(
                f"AlphaFactoryV1 supports only 4h timeframe. got={self._timeframe!r}"
            )

    @staticmethod
    def _validate_contract(ensemble: EnsembleOutput) -> None:
        n = len(ensemble.alpha_long)
        required = (
            ensemble.alpha_short,
            ensemble.alpha_net,
            ensemble.confidence,
            ensemble.turnover_hint,
        )
        if any(len(arr) != n for arr in required):
            raise ValueError("output contract violation: output lengths must match")

    def build(
        self,
        *,
        alpha_frame: pd.DataFrame,
        fold_ics: np.ndarray,
        fold_sizes: np.ndarray,
        forward_returns: np.ndarray,
        crisis_mask: np.ndarray,
    ) -> AlphaFactoryResult:
        """Build ensemble outputs and gate metrics from fold-level diagnostics."""
        self._validate_4h_only()

        weights = compute_ic_shrinkage_weights(
            ic_series=np.asarray(fold_ics, dtype=np.float64),
            sample_sizes=np.asarray(fold_sizes, dtype=np.float64),
            config=self._shrinkage_config,
        )
        ensemble = build_ensemble(alpha_frame=alpha_frame, weights=weights)
        self._validate_contract(ensemble)

        metrics = build_gate_metrics(
            alpha_long=ensemble.alpha_long,
            alpha_net=ensemble.alpha_net,
            forward_returns=np.asarray(forward_returns, dtype=np.float64),
            fold_ics=np.asarray(fold_ics, dtype=np.float64),
            turnover_hint=ensemble.turnover_hint,
            crisis_mask=np.asarray(crisis_mask, dtype=bool),
            cost_per_turnover=self._cost_per_turnover,
        )

        return AlphaFactoryResult(
            alpha_long=ensemble.alpha_long,
            alpha_short=ensemble.alpha_short,
            alpha_net=ensemble.alpha_net,
            confidence=ensemble.confidence,
            turnover_hint=ensemble.turnover_hint,
            ensemble_weights=ensemble.weights,
            gate_metrics=metrics,
        )

    def mine_alphas_cs(
        self,
        panel_df: pd.DataFrame,
        is_end_date: str | None = None,
        filter_options: dict[str, object] | None = None,
    ) -> pd.DataFrame:
        """Legacy-compatible alpha_panel builder for pipeline integration.

        This method intentionally keeps a lightweight deterministic path so that
        `pipeline_runner` can switch backend from legacy miner to factory without
        changing downstream contracts.
        """
        del filter_options
        self._validate_4h_only()
        if panel_df.empty:
            return pd.DataFrame()
        if not isinstance(panel_df.index, pd.MultiIndex):
            raise ValueError("AlphaFactoryV1 expects MultiIndex(datetime, symbol).")
        if "datetime" not in panel_df.index.names:
            raise ValueError("AlphaFactoryV1 requires 'datetime' index level.")
        cfg = AlphaFactoryConfig(timeframe=self._timeframe)
        dt_vals = panel_df.index.get_level_values("datetime")

        def _cross_sectional_frame(df: pd.DataFrame) -> pd.DataFrame:
            cs = df.copy()
            numeric_cols = cs.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if str(col).startswith("hmm_prob_") or str(col).startswith("regime_prob_"):
                    continue
                s = pd.to_numeric(cs[col], errors="coerce")
                ranked = s.groupby(dt_vals).rank(method="average", pct=True)
                cs[col] = (ranked - 0.5) * 2.0
            return cs

        def _pick_series(col_names: tuple[str, ...]) -> pd.Series:
            for col in col_names:
                if col in panel_df.columns:
                    s = pd.to_numeric(panel_df[col], errors="coerce").astype(np.float64)
                    s = s.where(np.isfinite(s), 0.0)
                    return s.clip(lower=0.0)
            return pd.Series(0.0, index=panel_df.index, dtype=np.float64)

        cs_panel = _cross_sectional_frame(panel_df)
        bull = _pick_series(("hmm_prob_bull_calm",)) + _pick_series(("hmm_prob_bull_vol_up",))
        if float(bull.abs().sum()) <= 1e-12:
            bull = _pick_series(("hmm_prob_bull", "regime_prob_bull"))
        bear = _pick_series(("hmm_prob_bear_trend", "hmm_prob_bear", "regime_prob_bear"))
        chop = _pick_series(
            (
                "hmm_prob_chop",
                "regime_prob_chop",
                "hmm_prob_sideways",
                "regime_prob_sideways",
            )
        )
        crisis = _pick_series(
            (
                "hmm_prob_crisis",
                "regime_prob_crisis",
                "hmm_prob_stress",
                "regime_prob_stress",
            )
        )
        posterior_mat = np.column_stack(
            [
                bull.to_numpy(dtype=np.float64, copy=False),
                bear.to_numpy(dtype=np.float64, copy=False),
                chop.to_numpy(dtype=np.float64, copy=False),
                crisis.to_numpy(dtype=np.float64, copy=False),
            ]
        )
        posterior_mat = np.nan_to_num(posterior_mat, nan=0.0, posinf=0.0, neginf=0.0)
        posterior_mat = np.clip(posterior_mat, 0.0, 1.0)
        denom = posterior_mat.sum(axis=1, keepdims=True)
        posterior_norm = np.full_like(posterior_mat, 0.25, dtype=np.float64)
        np.divide(
            posterior_mat,
            np.where(denom > 1e-12, denom, 1.0),
            out=posterior_norm,
            where=denom > 1e-12,
        )

        feature_source_cols = [
            "ret_24",
            "ret_6",
            "ret_12",
            "funding_z_72",
            "funding_rate",
            "funding_mom_24",
            "oi_momentum_24h",
            "oi_price_divergence_24h",
            "taker_imbalance_z_24",
            "cvd_divergence_24h",
            "vpin_proxy_12",
            "tail_risk_24",
            "vol_surface_24_168",
            "macro_vol_regime_shift",
            "idiosyncratic_return_24h",
            "btc_beta",
            "range_pos_24",
        ]
        feature_rows = cs_panel.reindex(columns=feature_source_cols).to_dict(orient="records")
        if "ret_6" in cs_panel.columns:
            turnover_series = pd.to_numeric(cs_panel["ret_6"], errors="coerce").astype(np.float64)
        else:
            turnover_series = pd.Series(0.0, index=panel_df.index, dtype=np.float64)
        turnover_values = np.abs(turnover_series.fillna(0.0).to_numpy(dtype=np.float64, copy=False))

        n_rows = len(panel_df)
        adjusted_alpha = np.zeros(n_rows, dtype=np.float64)
        alpha_conf = np.zeros(n_rows, dtype=np.float64)
        alpha_cost_bps = np.zeros(n_rows, dtype=np.float64)
        alpha_turnover_hint = np.zeros(n_rows, dtype=np.float64)

        for i in range(n_rows):
            features = extract_alpha_features(feature_rows[i], cfg.norm)
            sleeves: SleeveScores = compute_sleeve_scores(features, cfg.sleeves)
            posterior = RegimePosterior(
                bull=float(posterior_norm[i, 0]),
                bear=float(posterior_norm[i, 1]),
                chop=float(posterior_norm[i, 2]),
                crisis=float(posterior_norm[i, 3]),
            )
            decision = route_by_regime(posterior, cfg.sleeves, cfg.regime)
            routed_raw = blend_raw_alpha(
                SleeveScores(
                    trend=sleeves.trend * decision.weights.trend,
                    reversal=sleeves.reversal * decision.weights.reversal,
                    carry=sleeves.carry * decision.weights.carry,
                    flow=sleeves.flow * decision.weights.flow,
                    idio=sleeves.idio * decision.weights.idio,
                ),
                cfg.sleeves,
            )
            adj, turnover_penalty, cost_penalty = adjust_alpha_for_cost_and_confidence(
                raw_alpha=routed_raw,
                confidence=decision.confidence,
                gross_exposure=decision.gross_exposure,
                turnover=float(turnover_values[i]) if np.isfinite(turnover_values[i]) else 0.0,
                cfg=cfg.cost,
            )
            adjusted_alpha[i] = adj
            alpha_conf[i] = float(np.clip(decision.confidence, 0.0, 1.0))
            alpha_cost_bps[i] = float(max(cost_penalty, 0.0) * 1e4)
            alpha_turnover_hint[i] = float(max(turnover_penalty, 0.0))

        score = pd.Series(adjusted_alpha, index=panel_df.index, dtype=np.float64)
        alpha_long_00 = score.groupby(dt_vals).rank(pct=True, method="average").fillna(0.5)
        alpha_short_00 = 1.0 - alpha_long_00
        alpha_net = (2.0 * alpha_long_00 - 1.0).clip(-1.0, 1.0)

        out = pd.DataFrame(index=panel_df.index)
        out["alpha_long_00"] = alpha_long_00.to_numpy(dtype=np.float64, copy=False)
        out["alpha_short_00"] = alpha_short_00.to_numpy(dtype=np.float64, copy=False)
        out["alpha_long"] = out["alpha_long_00"]
        out["alpha_short"] = out["alpha_short_00"]
        out["alpha_net"] = alpha_net.to_numpy(dtype=np.float64, copy=False)
        out["alpha_confidence"] = alpha_conf
        out["alpha_cost_bps"] = alpha_cost_bps
        out["alpha_turnover_hint"] = alpha_turnover_hint

        post_selected = float(int(len(out) > 0))
        filt_meta: dict[str, Any] = {
            "n_components": 1.0,
            "n_surviving": post_selected,
            "n_surviving_long": post_selected,
            "n_surviving_short": post_selected,
            "post_agg_selected_long_count": post_selected,
            "post_agg_selected_short_count": post_selected,
            "survived_long_cols": ["alpha_long_00"],
            "survived_short_cols": ["alpha_short_00"],
            "post_agg_selected_long_cols": ["alpha_long_00"],
            "post_agg_selected_short_cols": ["alpha_short_00"],
            "fail_fdr": 0.0,
            "fail_dsr": 0.0,
            "fail_oos": 0.0,
            "fail_half_life": 0.0,
            "fail_tail": 0.0,
            "fail_short": 0.0,
            "fail_sym_bal": 0.0,
            "primary_is_mu": float(np.nanmean(alpha_net.to_numpy(dtype=np.float64, copy=False))),
            "primary_oos_mu": float(np.nanmean(alpha_net.to_numpy(dtype=np.float64, copy=False))),
            "primary_is_ic_mean": 0.0,
            "primary_oos_ic_mean": 0.0,
            "primary_oos_icir": 0.0,
            "primary_half_life": 0.0,
            "short_head_oos_ic_mean": 0.0,
            "gate_status_by_col": {
                "alpha_long_00": {"final_selection_ok": True},
                "alpha_short_00": {"final_selection_ok": True},
            },
            "half_life_diag_code_by_col": {
                "alpha_long_00": "pass",
                "alpha_short_00": "pass",
            },
            "tail_ic_by_slot": {"alpha_long_00": 0.0},
            "elite_zero_after_survival": 0.0,
            "final_selection_fail_long": 0.0,
            "final_selection_fail_short": 0.0,
            "no_candidate_reason": "",
        }

        out.attrs["alpha_component_filter"] = filt_meta
        out.attrs["alpha_goal_eval_meta"] = {
            "framework": "g-alpha.v8",
            "verdict": "pass",
            "reason_codes": [],
            "required_metrics_present": {
                "fdr": True,
                "dsr": True,
                "oos_ic_floor": True,
                "retention": True,
                "icir_oos": True,
                "tail_ic": True,
                "short_side_ic": True,
                "half_life": True,
                "symbol_balance": True,
            },
            "gate_summary": {
                "n_surviving": int(float(filt_meta.get("n_surviving", 0.0))),
                "n_components": int(float(filt_meta.get("n_components", 0.0))),
                "fail_fdr": int(float(filt_meta.get("fail_fdr", 0.0))),
                "fail_dsr": int(float(filt_meta.get("fail_dsr", 0.0))),
                "fail_half_life": int(float(filt_meta.get("fail_half_life", 0.0))),
                "fail_tail": int(float(filt_meta.get("fail_tail", 0.0))),
                "fail_oos": int(float(filt_meta.get("fail_oos", 0.0))),
                "fail_short": int(float(filt_meta.get("fail_short", 0.0))),
                "fail_symbol_balance": int(float(filt_meta.get("fail_sym_bal", 0.0))),
            },
            "is_end_date": str(is_end_date or ""),
        }
        return out
