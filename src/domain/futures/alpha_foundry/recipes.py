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
    "ichimoku_trend": (
        _rd(
            "ichimoku_9_26_52",
            {"tenkan": 9, "kijun": 26, "senkou": 52},
            26,
            ("high", "low", "close"),
        ),
        _rd("ichi_9_26", {"tenkan": 9, "kijun": 26}, 2, ("high", "low", "close")),
    ),
    "xs_momentum": (
        _rd("xs_mom_12", {"rank_window": 12}, 1, ("close",)),
        _rd("xs_mom_48", {"rank_window": 48}, 2, ("close",)),
        _rd("xs_mom_rank_12", {"rank_window": 12}, 2, ("close",)),
        _rd("xs_mom_rank_24", {"rank_window": 24}, 3, ("close",)),
    ),
    "macd_4h": (_rd("macd_12_26_9", {"fast": 12, "slow": 26, "signal": 9}, 1, ("close",)),),
    "trend_pullback_quality_v2": (
        _rd("tpq_v2_20_100", {"fast": 20, "slow": 100, "rsi_lo": 40, "rsi_hi": 65}, 1, ("close", "high", "low")),
        _rd("tpq_v2_50_200", {"fast": 50, "slow": 200, "rsi_lo": 35, "rsi_hi": 70}, 2, ("close", "high", "low")),
    ),
    "residual_momentum_xs": (
        _rd("rm_xs_12", {"lookback": 12, "btc_beta_cap": 0.80}, 1, ("close",)),
        _rd("rm_xs_24", {"lookback": 24, "btc_beta_cap": 0.80}, 2, ("close",)),
    ),
    "funding_flow_exhaustion_sparse": (
        _rd("ffes_96", {"funding_window": 96, "funding_z": 1.5, "imbalance_window": 12}, 2, ("close", "funding")),
    ),
    "oi_lsr_unwind": (_rd("oiu_42", {"oi_window": 42, "lsr_window": 21, "z_exit": 0.5}, 3, ("close", "oi", "lsr")),),
    "vol_contraction_breakout": (
        _rd("vcb_20_120", {"bb_window": 20, "vol_window": 120, "expansion_ratio": 1.5}, 1, ("close", "high", "low")),
    ),
    "xs_residual_rebalance": (_rd("xsrr_12", {"rank_window": 12, "bucket_threshold": 0.20}, 1, ("close",)),),
    "carry_net_of_funding": (
        _rd("cnf_96", {"funding_window": 96, "z_threshold": 0.5, "carry_window": 24}, 2, ("close", "funding")),
    ),
    "funding_session_orb_flow": (
        _rd(
            "fs_orb_15m",
            {"ltf": "15m", "opening_minutes": 15, "volume_z": 1.5, "cvd_min": 0.15},
            1,
            ("close", "taker_buy"),
        ),
        _rd(
            "fs_orb_30m",
            {"ltf": "30m", "opening_minutes": 30, "volume_z": 1.5, "cvd_min": 0.15},
            1,
            ("close", "taker_buy"),
        ),
    ),
    "liquidity_sweep_reclaim": (
        _rd("lsr_5m_36", {"ltf": "5m", "sweep_window": 36, "atr_mult": 0.25}, 1, ("close", "high", "low", "taker_buy")),
    ),
    "volume_participation_breakout": (
        _rd("vpb_15m_48", {"ltf": "15m", "channel": 48, "volume_z": 2.0}, 1, ("close", "high", "low", "taker_buy")),
        _rd("vpb_30m_48", {"ltf": "30m", "channel": 48, "volume_z": 2.0}, 1, ("close", "high", "low", "taker_buy")),
    ),
    "liquidity_participation_breakout": (
        _rd("lpb_40", {"channel": 40}, 1, ("close", "high", "low", "volume")),
        _rd("lpb_60", {"channel": 60}, 1, ("close", "high", "low", "volume")),
    ),
    "btc_neutral_residual_reversal": (
        _rd("bnrr_24", {"lookback": 24, "tail_fraction": 0.20}, 1, ("close",)),
        _rd("bnrr_48", {"lookback": 48, "tail_fraction": 0.20}, 1, ("close",)),
    ),
    "price_band_reversion": (
        _rd("pbr_std_20", {"band_period": 20, "band_std": 2.0}, 1, ("close",)),
        _rd("pbr_atr_20", {"band_period": 20, "atr_mult": 2.0}, 1, ("close", "high", "low")),
    ),
}

FAMILY_ARCHETYPE: dict[str, AlphaArchetype] = {
    "trend_ma": "trend",
    "ichimoku_trend": "trend",
    "xs_momentum": "cross_sectional",
    "macd_4h": "trend",
    "trend_pullback_quality_v2": "trend",
    "residual_momentum_xs": "cross_sectional",
    "funding_flow_exhaustion_sparse": "flow",
    "oi_lsr_unwind": "flow",
    "vol_contraction_breakout": "trend",
    "xs_residual_rebalance": "cross_sectional",
    "carry_net_of_funding": "carry",
    "funding_session_orb_flow": "trend",
    "liquidity_sweep_reclaim": "mean_reversion",
    "volume_participation_breakout": "trend",
    "liquidity_participation_breakout": "trend",
    "btc_neutral_residual_reversal": "cross_sectional",
    "price_band_reversion": "mean_reversion",
}

FAMILY_SIDE_RULE: dict[str, str] = {
    "trend_ma": "trend_follow",
    "ichimoku_trend": "trend_follow",
    "xs_momentum": "xs_momentum",
    "macd_4h": "trend_follow",
    "trend_pullback_quality_v2": "trend_pullback_quality",
    "residual_momentum_xs": "xs_residual_momentum",
    "funding_flow_exhaustion_sparse": "flow_exhaustion",
    "oi_lsr_unwind": "flow_reversal",
    "vol_contraction_breakout": "vol_breakout",
    "xs_residual_rebalance": "xs_residual_momentum",
    "carry_net_of_funding": "carry_mean_rev",
    "funding_session_orb_flow": "breakout_retest",
    "liquidity_sweep_reclaim": "flow_reversal",
    "volume_participation_breakout": "breakout_retest",
    "liquidity_participation_breakout": "breakout_retest",
    "btc_neutral_residual_reversal": "xs_neutral",
    "price_band_reversion": "mean_reversion",
}

FAMILY_EXIT_POLICY: dict[str, str] = {
    "trend_ma": "atr_trail_2",
    "ichimoku_trend": "atr_trail_3",
    "xs_momentum": "atr_trail_2",
    "macd_4h": "atr_trail_2",
    "trend_pullback_quality_v2": "atr_trail_2",
    "residual_momentum_xs": "atr_trail_2",
    "funding_flow_exhaustion_sparse": "tp_sl_1_2",
    "oi_lsr_unwind": "tp_sl_1_2",
    "vol_contraction_breakout": "tp_sl_1.5_3",
    "xs_residual_rebalance": "atr_trail_2",
    "carry_net_of_funding": "tp_sl_1_2",
    "funding_session_orb_flow": "atr_trail_2",
    "liquidity_sweep_reclaim": "tp_sl_1_2",
    "volume_participation_breakout": "atr_trail_2",
    "liquidity_participation_breakout": "atr_trail_2",
    "btc_neutral_residual_reversal": "atr_trail_2",
    "price_band_reversion": "tp_sl_1.5_3",
}

FAMILY_MAX_TURNOVER: dict[str, float] = {
    "trend_ma": 365.0,
    "ichimoku_trend": 52.0,
    "xs_momentum": 365.0,
    "macd_4h": 365.0,
    "trend_pullback_quality_v2": 180.0,
    "residual_momentum_xs": 365.0,
    "funding_flow_exhaustion_sparse": 120.0,
    "oi_lsr_unwind": 120.0,
    "vol_contraction_breakout": 180.0,
    "xs_residual_rebalance": 240.0,
    "carry_net_of_funding": 180.0,
    "funding_session_orb_flow": 240.0,
    "liquidity_sweep_reclaim": 365.0,
    "volume_participation_breakout": 240.0,
    "liquidity_participation_breakout": 180.0,
    "btc_neutral_residual_reversal": 365.0,
    "price_band_reversion": 365.0,
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
    from src.domain.futures.strategy.family_lifecycle import resolve_retired_families_for_tf

    families = tuple(RECIPE_DEFINITIONS.keys())
    if include_families:
        families = tuple(f for f in families if f in include_families)
    if exclude_families:
        families = tuple(f for f in families if f not in exclude_families)
    # Apply FAMILY_TF_RETIREMENT — retired (family, tf) pairs are excluded
    retired = frozenset(resolve_retired_families_for_tf(timeframe))
    families = tuple(f for f in families if f not in retired)

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
