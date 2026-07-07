"""Alpha Foundry recipe catalog builder.

[ADR_20260706_ALPHA_FOUNDRY_L0_L1_HANDOFF_GUARD]
[ADR_20260706_ALPHA_FOUNDRY_SYNC][ADR_20260706_ALPHA_FOUNDRY_L0_SIGNAL_RIGOR]
"""

from __future__ import annotations

import hashlib
from typing import cast

from src.domain.futures.alpha_foundry.contracts import AlphaArchetype, AlphaRecipe


def _rd(
    variant: str,
    params: dict[str, float | int | str],
    causal_lag: int,
    required: tuple[str, ...],
) -> dict[str, object]:
    return {
        "variant": variant,
        "params": params,
        "causal_lag": causal_lag,
        "required": required,
    }


RECIPE_DEFINITIONS: dict[str, tuple[dict[str, object], ...]] = {
    "trend_ma": (
        _rd("ema_12_72", {"fast": 12, "slow": 72}, 1, ("close",)),
        _rd("ema_18_108", {"fast": 18, "slow": 108}, 1, ("close",)),
    ),
    "ema_trend": (
        _rd("ema_12_24", {"fast": 12, "slow": 24}, 1, ("close",)),
        _rd("ema_12_48", {"fast": 12, "slow": 48}, 2, ("close",)),
        _rd("ema_12_72", {"fast": 12, "slow": 72}, 3, ("close",)),
    ),
    "hma_trend": (
        _rd("hma_55", {"period": 55}, 2, ("close",)),
        _rd("hma_100", {"period": 100}, 3, ("close",)),
    ),
    "kama_trend": (
        _rd("kama_30_5", {"er_period": 30, "fast_sc": 2, "slow_sc": 30}, 2, ("close",)),
        _rd("kama_60_10", {"er_period": 60, "fast_sc": 5, "slow_sc": 40}, 3, ("close",)),
    ),
    "macd_trend": (
        _rd("macd_12_26_9", {"fast": 12, "slow": 26, "signal": 9}, 1, ("close",)),
        _rd("macd_24_52_18", {"fast": 24, "slow": 52, "signal": 18}, 2, ("close",)),
    ),
    "ichimoku_trend": (
        _rd(
            "ichimoku_9_26_52",
            {"tenkan": 9, "kijun": 26, "senkou": 52},
            26,
            ("high", "low", "close"),
        ),
        _rd("ichi_9_26", {"tenkan": 9, "kijun": 26}, 2, ("high", "low", "close")),
    ),
    "rsi_mean_reversion": (
        _rd("rsi_14_30_70", {"period": 14, "oversold": 30, "overbought": 70}, 1, ("close",)),
        _rd("rsi_7_25_75", {"period": 7, "oversold": 25, "overbought": 75}, 1, ("close",)),
    ),
    "stoch_rsi_mean_reversion": (
        _rd(
            "stoch_rsi_14_3_3",
            {"rsi_period": 14, "stoch_k": 3, "stoch_d": 3},
            1,
            ("close",),
        ),
    ),
    "bollinger_mean_reversion": (
        _rd("bb_20_2", {"period": 20, "std": 2.0}, 1, ("close",)),
        _rd("bb_40_2.5", {"period": 40, "std": 2.5}, 2, ("close",)),
    ),
    "keltner_mean_reversion": (_rd("kc_20_1.5", {"period": 20, "atr_mult": 1.5}, 1, ("close", "high", "low")),),
    "funding_slope_carry": (
        _rd("funding_slope_12", {"slope_window": 12}, 2, ("close", "funding")),
        _rd("funding_slope_24", {"slope_window": 24}, 3, ("close", "funding")),
    ),
    "oi_buildup_flow": (
        _rd("oi_zscore_12", {"z_window": 12, "oi_type": "oi"}, 2, ("close", "oi")),
        _rd("oi_zscore_24", {"z_window": 24, "oi_type": "oi"}, 3, ("close", "oi")),
    ),
    "lsr_skew_flow": (
        _rd("lsr_zscore_12", {"z_window": 12}, 2, ("close", "lsr")),
        _rd("lsr_zscore_24", {"z_window": 24}, 3, ("close", "lsr")),
    ),
    "taker_flow_imbalance": (
        _rd("taker_imbalance_6", {"window": 6}, 1, ("close", "taker_buy")),
        _rd("taker_imbalance_24", {"window": 24}, 2, ("close", "taker_buy")),
    ),
    "xs_momentum": (
        _rd("xs_mom_12", {"rank_window": 12}, 1, ("close",)),
        _rd("xs_mom_48", {"rank_window": 48}, 2, ("close",)),
        _rd("xs_mom_rank_12", {"rank_window": 12}, 2, ("close",)),
        _rd("xs_mom_rank_24", {"rank_window": 24}, 3, ("close",)),
    ),
    "macd_4h": (_rd("macd_12_26_9", {"fast": 12, "slow": 26, "signal": 9}, 1, ("close",)),),
    "sparse_breakout_retest_v2": (
        _rd("bor_v2_20", {"channel": 20, "retest": 3}, 1, ("close", "high", "low")),
        _rd("bor_v2_40", {"channel": 40, "retest": 5}, 2, ("close", "high", "low")),
    ),
    "trend_pullback_quality_v2": (
        _rd("tpq_v2_20_100", {"fast": 20, "slow": 100, "rsi_lo": 40, "rsi_hi": 65}, 1, ("close", "high", "low")),
        _rd("tpq_v2_50_200", {"fast": 50, "slow": 200, "rsi_lo": 35, "rsi_hi": 70}, 2, ("close", "high", "low")),
    ),
    "residual_momentum_xs": (
        _rd("rm_xs_12", {"lookback": 12, "btc_beta_cap": 0.80}, 1, ("close",)),
        _rd("rm_xs_24", {"lookback": 24, "btc_beta_cap": 0.80}, 2, ("close",)),
    ),
    "funding_contra_carry_sparse": (
        _rd("fccs_96", {"funding_window": 96, "funding_z_threshold": 1.5}, 2, ("close", "funding")),
    ),
    "oi_price_divergence_unwind": (
        _rd("opdu_42", {"oi_window": 42, "price_window": 24}, 2, ("close", "oi")),
    ),
    "taker_flow_exhaustion": (
        _rd("tfe_12", {"imbalance_window": 12, "threshold": 0.8}, 1, ("close", "taker_buy")),
    ),
    "liquidity_vacuum_breakout": (
        _rd("lvb_20", {"bb_window": 20, "vol_window": 120}, 1, ("close", "high", "low")),
    ),
    "volatility_contraction_expansion": (
        _rd("vce_20", {"window": 20, "expansion_ratio": 1.5}, 1, ("close", "high", "low")),
    ),
    "btc_regime_relative_strength": (
        _rd("brrs_50", {"window": 50}, 2, ("close",)),
    ),
    "mean_reversion_after_liquidation_proxy": (
        _rd("mralp_24", {"window": 24, "z_entry": 2.0}, 1, ("close", "high", "low")),
    ),
}

FAMILY_ARCHETYPE: dict[str, AlphaArchetype] = {
    "trend_ma": "trend",
    "ema_trend": "trend",
    "hma_trend": "trend",
    "kama_trend": "trend",
    "macd_trend": "trend",
    "ichimoku_trend": "trend",
    "rsi_mean_reversion": "mean_reversion",
    "stoch_rsi_mean_reversion": "mean_reversion",
    "bollinger_mean_reversion": "mean_reversion",
    "keltner_mean_reversion": "mean_reversion",
    "funding_slope_carry": "carry",
    "oi_buildup_flow": "flow",
    "lsr_skew_flow": "flow",
    "taker_flow_imbalance": "flow",
    "xs_momentum": "cross_sectional",
    "macd_4h": "trend",
    "sparse_breakout_retest_v2": "trend",
    "trend_pullback_quality_v2": "trend",
    "residual_momentum_xs": "cross_sectional",
    "funding_contra_carry_sparse": "carry",
    "oi_price_divergence_unwind": "flow",
    "taker_flow_exhaustion": "flow",
    "liquidity_vacuum_breakout": "trend",
    "volatility_contraction_expansion": "mean_reversion",
    "btc_regime_relative_strength": "trend",
    "mean_reversion_after_liquidation_proxy": "mean_reversion",
}

FAMILY_SIDE_RULE: dict[str, str] = {
    "trend_ma": "trend_follow",
    "ema_trend": "trend_follow",
    "hma_trend": "trend_follow",
    "kama_trend": "trend_follow",
    "macd_trend": "trend_follow",
    "ichimoku_trend": "trend_follow",
    "rsi_mean_reversion": "mean_reversion",
    "stoch_rsi_mean_reversion": "mean_reversion",
    "bollinger_mean_reversion": "mean_reversion",
    "keltner_mean_reversion": "mean_reversion",
    "funding_slope_carry": "carry_mean_rev",
    "oi_buildup_flow": "flow_reversal",
    "lsr_skew_flow": "flow_reversal",
    "taker_flow_imbalance": "flow_reversal",
    "xs_momentum": "xs_momentum",
    "macd_4h": "trend_follow",
    "sparse_breakout_retest_v2": "breakout_retest_sparse",
    "trend_pullback_quality_v2": "trend_pullback_quality",
    "residual_momentum_xs": "xs_residual_momentum",
    "funding_contra_carry_sparse": "carry_mean_rev",
    "oi_price_divergence_unwind": "flow_reversal",
    "taker_flow_exhaustion": "flow_exhaustion",
    "liquidity_vacuum_breakout": "atr_trail_2",
    "volatility_contraction_expansion": "tp_sl_1.5_3",
    "btc_regime_relative_strength": "trend_follow",
    "mean_reversion_after_liquidation_proxy": "tp_sl_1.5_3",
}

FAMILY_EXIT_POLICY: dict[str, str] = {
    "trend_ma": "atr_trail_2",
    "ema_trend": "atr_trail_2",
    "hma_trend": "atr_trail_2",
    "kama_trend": "atr_trail_2",
    "macd_trend": "atr_trail_2",
    "ichimoku_trend": "atr_trail_3",
    "rsi_mean_reversion": "tp_sl_1.5_3",
    "stoch_rsi_mean_reversion": "tp_sl_1.5_3",
    "bollinger_mean_reversion": "tp_sl_1.5_3",
    "keltner_mean_reversion": "tp_sl_1.5_3",
    "funding_slope_carry": "tp_sl_1_2",
    "oi_buildup_flow": "tp_sl_1_2",
    "lsr_skew_flow": "tp_sl_1_2",
    "taker_flow_imbalance": "tp_sl_1_2",
    "xs_momentum": "atr_trail_2",
    "macd_4h": "atr_trail_2",
    "sparse_breakout_retest_v2": "atr_trail_2",
    "trend_pullback_quality_v2": "atr_trail_2",
    "residual_momentum_xs": "atr_trail_2",
    "funding_contra_carry_sparse": "tp_sl_1_2",
    "oi_price_divergence_unwind": "tp_sl_1_2",
    "taker_flow_exhaustion": "tp_sl_1_2",
    "liquidity_vacuum_breakout": "atr_trail_2",
    "volatility_contraction_expansion": "tp_sl_1.5_3",
    "btc_regime_relative_strength": "atr_trail_2",
    "mean_reversion_after_liquidation_proxy": "tp_sl_1.5_3",
}

FAMILY_MAX_TURNOVER: dict[str, float] = {
    "trend_ma": 365.0,
    "ema_trend": 365.0,
    "hma_trend": 365.0,
    "kama_trend": 365.0,
    "macd_trend": 365.0,
    "ichimoku_trend": 52.0,
    "rsi_mean_reversion": 730.0,
    "stoch_rsi_mean_reversion": 730.0,
    "bollinger_mean_reversion": 365.0,
    "keltner_mean_reversion": 365.0,
    "funding_slope_carry": 365.0,
    "oi_buildup_flow": 365.0,
    "lsr_skew_flow": 365.0,
    "taker_flow_imbalance": 730.0,
    "xs_momentum": 365.0,
    "macd_4h": 365.0,
    "sparse_breakout_retest_v2": 180.0,
    "trend_pullback_quality_v2": 180.0,
    "residual_momentum_xs": 365.0,
    "funding_contra_carry_sparse": 365.0,
    "oi_price_divergence_unwind": 365.0,
    "taker_flow_exhaustion": 730.0,
    "liquidity_vacuum_breakout": 180.0,
    "volatility_contraction_expansion": 730.0,
    "btc_regime_relative_strength": 365.0,
    "mean_reversion_after_liquidation_proxy": 730.0,
}


def _make_recipe_id(family: str, variant: str, timeframe: str, params: dict[str, float | int | str]) -> str:
    raw = f"{family}:{variant}:{timeframe}:{sorted(params.items())}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return f"{family}:{variant}:{timeframe}:{digest}"


_SIGNAL_TO_ALPHA: dict[str, AlphaArchetype] = {
    "trend": "trend",
    "ts_mom": "trend",
    "mean_rev": "mean_reversion",
    "flow_rev": "flow",
    "unwind": "flow",
    "carry_rev": "carry",
    "beta_neut": "hedge",
    "xs_alpha": "cross_sectional",
}


def map_signal_archetype_to_alpha_archetype(signal_archetype: str) -> AlphaArchetype:
    return _SIGNAL_TO_ALPHA.get(signal_archetype, "trend")


def build_alpha_recipe_catalog(
    *,
    timeframe: str,
    include_families: tuple[str, ...] = (),
    exclude_families: tuple[str, ...] = (),
    max_recipes_per_family: int = 64,
) -> tuple[AlphaRecipe, ...]:
    families = tuple(RECIPE_DEFINITIONS.keys())
    if include_families:
        families = tuple(f for f in families if f in include_families)
    if exclude_families:
        families = tuple(f for f in families if f not in exclude_families)

    result: list[AlphaRecipe] = []
    seen_ids: set[str] = set()

    for family in families:
        count = 0
        for defn in RECIPE_DEFINITIONS[family]:
            if count >= max_recipes_per_family:
                break
            variant = cast(str, defn["variant"])
            params = cast(dict[str, float | int | str], defn["params"])
            causal_lag = cast(int, defn["causal_lag"])
            required = cast(tuple[str, ...], defn["required"])
            recipe = AlphaRecipe(
                recipe_id=_make_recipe_id(family, variant, timeframe, params),
                family=family,
                variant=variant,
                timeframe=timeframe,
                archetype=FAMILY_ARCHETYPE[family],
                indicator_params=params,
                side_rule_id=FAMILY_SIDE_RULE[family],
                exit_policy_id=FAMILY_EXIT_POLICY[family],
                required_fields=required,
                causal_lag_bars=causal_lag,
                max_turnover_per_year=FAMILY_MAX_TURNOVER[family],
            )
            if recipe.recipe_id in seen_ids:
                continue
            seen_ids.add(recipe.recipe_id)
            result.append(recipe)
            count += 1

    return tuple(result)
