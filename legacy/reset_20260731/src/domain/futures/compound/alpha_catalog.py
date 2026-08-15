from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from src.domain.futures.compound.contracts import (
    AlphaCandidateState,
    AlphaDefinition,
    MarketFeatureCube,
    MultiscaleAlphaDefinition,
    RawAlphaTape,
)

_logger = logging.getLogger(__name__)

_EXPANDED_FAMILIES: tuple[str, ...] = (
    "time_series_trend",
    "breakout",
    "cross_sectional_momentum",
    "short_term_reversal",
    "carry_basis",
    "flow_positioning",
    "volatility_squeeze_keltner",
    "funding_carry_reversion",
    "flow_imbalance_taker",
    "open_interest_confirmation",
)

_EXPANDED_HORIZONS: tuple[int, ...] = (4, 8, 12, 24, 48, 96)


def build_canonical_alpha_catalog() -> tuple[AlphaDefinition, ...]:
    catalog: list[AlphaDefinition] = []
    for family in _EXPANDED_FAMILIES:
        for h in _EXPANDED_HORIZONS:
            recipe_id = f"{family}_h{h}"
            if family in ("flow_positioning",):
                required: tuple[str, ...] = ("taker_buy_quote", "quote_volume", "funding", "premium")
                data_tier: Literal["core", "conditional"] = "conditional"
            elif family in ("carry_basis", "funding_carry_reversion"):
                required = ("funding", "premium")
                data_tier = "core"
            elif family == "flow_imbalance_taker":
                required = ("taker_buy_quote", "quote_volume")
                data_tier = "conditional"
            elif family == "open_interest_confirmation":
                required = ("open_interest", "quote_volume")
                data_tier = "conditional"
            else:
                required = ("open", "high", "low", "close", "quote_volume")
                data_tier = "core"
            lookback = 12 * h
            catalog.append(
                AlphaDefinition(
                    recipe_id=recipe_id,
                    family=family,
                    horizon_bars=h,
                    lookback_bars=lookback,
                    required_fields=required,
                    data_tier=data_tier,
                    causal_lag_bars=1,
                )
            )
    return tuple(catalog)


_MULTISCALE_RECIPES: tuple[dict[str, Any], ...] = (
    {"recipe_id": "ts_trend_4h_h24", "family": "trend", "native_timeframe": "4h", "lookback_hours": (72, 168), "horizon_hours": 24, "required_fields": ("open", "high", "low", "close", "quote_volume"), "initial_state": AlphaCandidateState.CORE_CANDIDATE, "max_half_life_hours": 12.0},
    {"recipe_id": "ts_trend_12h_h72", "family": "trend", "native_timeframe": "12h", "lookback_hours": (168, 504), "horizon_hours": 72, "required_fields": ("open", "high", "low", "close", "quote_volume"), "initial_state": AlphaCandidateState.CORE_CANDIDATE, "max_half_life_hours": 36.0},
    {"recipe_id": "ts_trend_1d_h168", "family": "trend", "native_timeframe": "1d", "lookback_hours": (336, 1008), "horizon_hours": 168, "required_fields": ("open", "high", "low", "close", "quote_volume"), "initial_state": AlphaCandidateState.CORE_CANDIDATE, "max_half_life_hours": 84.0},
    {"recipe_id": "xs_resmom_4h_h24", "family": "residual_momentum", "native_timeframe": "4h", "lookback_hours": (72,), "horizon_hours": 24, "required_fields": ("open", "high", "low", "close", "quote_volume"), "initial_state": AlphaCandidateState.CORE_CANDIDATE, "max_half_life_hours": 12.0},
    {"recipe_id": "xs_resmom_12h_h72", "family": "residual_momentum", "native_timeframe": "12h", "lookback_hours": (168,), "horizon_hours": 72, "required_fields": ("open", "high", "low", "close", "quote_volume"), "initial_state": AlphaCandidateState.CORE_CANDIDATE, "max_half_life_hours": 36.0},
    {"recipe_id": "breakout_4h_h24", "family": "breakout", "native_timeframe": "4h", "lookback_hours": (72,), "horizon_hours": 24, "required_fields": ("open", "high", "low", "close", "quote_volume"), "initial_state": AlphaCandidateState.CONDITIONAL_CANDIDATE, "max_half_life_hours": 12.0},
    {"recipe_id": "breakout_12h_h72", "family": "breakout", "native_timeframe": "12h", "lookback_hours": (168,), "horizon_hours": 72, "required_fields": ("open", "high", "low", "close", "quote_volume"), "initial_state": AlphaCandidateState.CONDITIONAL_CANDIDATE, "max_half_life_hours": 36.0},
    {"recipe_id": "carry_funding_event_h8", "family": "carry", "native_timeframe": "funding_event", "lookback_hours": (168,), "horizon_hours": 8, "required_fields": ("funding", "premium"), "initial_state": AlphaCandidateState.CONDITIONAL_CANDIDATE, "max_half_life_hours": 4.0},
    {"recipe_id": "basis_reversion_1h_h8", "family": "basis_reversion", "native_timeframe": "1h", "lookback_hours": (24, 72), "horizon_hours": 8, "required_fields": ("mark", "index"), "initial_state": AlphaCandidateState.SHADOW_RESEARCH, "max_half_life_hours": 4.0},
    {"recipe_id": "flow_imbalance_15m_h1", "family": "taker_flow", "native_timeframe": "15m", "lookback_hours": (4, 12), "horizon_hours": 1, "required_fields": ("taker_buy_quote", "quote_volume"), "initial_state": AlphaCandidateState.SHADOW_RESEARCH, "max_half_life_hours": 0.5},
    {"recipe_id": "flow_oi_confirm_1h_h4", "family": "flow_oi", "native_timeframe": "1h", "lookback_hours": (24, 72), "horizon_hours": 4, "required_fields": ("taker_buy_quote", "quote_volume", "open_interest"), "initial_state": AlphaCandidateState.CONDITIONAL_CANDIDATE, "max_half_life_hours": 2.0},
    {"recipe_id": "liquidity_exhaustion_15m_h1", "family": "reversal", "native_timeframe": "15m", "lookback_hours": (2, 6), "horizon_hours": 1, "required_fields": ("open", "high", "low", "close", "quote_volume"), "initial_state": AlphaCandidateState.SHADOW_RESEARCH, "max_half_life_hours": 0.5},
)


def build_multiscale_alpha_catalog() -> tuple[MultiscaleAlphaDefinition, ...]:
    seen_ids: set[str] = set()
    supported_tfs = {"15m", "1h", "4h", "12h", "1d", "funding_event"}

    definitions: list[MultiscaleAlphaDefinition] = []
    for recipe in _MULTISCALE_RECIPES:
        rid: str = recipe["recipe_id"]
        if rid in seen_ids:
            msg = f"duplicate recipe_id: {rid}"
            raise ValueError(msg)
        seen_ids.add(rid)

        tf: str = recipe["native_timeframe"]
        if tf not in supported_tfs:
            msg = f"unsupported timeframe {tf!r} for {rid}, supported: {supported_tfs}"
            raise ValueError(msg)

        for field in recipe["required_fields"]:
            if field not in ("open", "high", "low", "close", "quote_volume", "funding", "premium", "mark", "index", "taker_buy_quote", "open_interest"):
                msg = f"undeclared field {field!r} in {rid}"
                raise ValueError(msg)

        definitions.append(
            MultiscaleAlphaDefinition(
                recipe_id=rid,
                family=recipe["family"],
                native_timeframe=recipe["native_timeframe"],
                lookback_hours=recipe["lookback_hours"],
                horizon_hours=recipe["horizon_hours"],
                required_fields=recipe["required_fields"],
                initial_state=recipe["initial_state"],
                max_half_life_hours=recipe["max_half_life_hours"],
            )
        )

    result = tuple(definitions)
    _logger.info("built multiscale alpha catalog: %d recipes", len(result))
    return result


def _ewm(arr: NDArray[Any], span: int) -> NDArray[Any]:
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(arr)
    out[:] = np.nan
    if arr.shape[0] == 0:
        return out
    out[0] = arr[0]
    for t in range(1, arr.shape[0]):
        out[t] = alpha * arr[t] + (1 - alpha) * out[t - 1]
    return out


def _robust_z_score(x: NDArray[Any]) -> NDArray[Any]:
    med = np.nanmedian(x, axis=0, keepdims=True)
    mad = np.nanmedian(np.abs(x - med), axis=0, keepdims=True)
    mad = np.where(mad < 1e-12, 1e-12, mad)
    result: NDArray[Any] = (x - med) / (1.4826 * mad)
    return result


def _atr(high: NDArray[Any], low: NDArray[Any], close: NDArray[Any], span: int) -> NDArray[Any]:
    prev_close = np.roll(close, 1, axis=0)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return _ewm(tr, span)


def _check_required_fields(
    recipe: AlphaDefinition,
    cube: MarketFeatureCube,
) -> bool:
    return all(field in cube.fields_2d for field in recipe.required_fields)


def compute_raw_alpha_tape(
    *, cube: MarketFeatureCube, catalog: Sequence[AlphaDefinition]
) -> RawAlphaTape:
    n_bars = cube.timestamps_ns.size
    n_syms = len(cube.symbols)
    n_recipes = len(catalog)
    scores = np.full((n_bars, n_syms, n_recipes), np.nan, dtype=np.float32)
    valid = np.zeros((n_bars, n_syms, n_recipes), dtype=np.bool_)
    horizon_bars = np.array([a.horizon_bars for a in catalog], dtype=np.int16)
    recipe_ids = tuple(a.recipe_id for a in catalog)

    high = cube.fields_2d.get("high", None)
    low = cube.fields_2d.get("low", None)
    close_f64 = cube.fields_2d.get("close", None)
    close = close_f64.astype(np.float32) if close_f64 is not None else None
    volume = cube.fields_2d.get("quote_volume", None)
    funding = cube.fields_2d.get("funding", None)
    premium = cube.fields_2d.get("premium", None)
    taker_buy = cube.fields_2d.get("taker_buy_quote", None)

    log_return = None
    if close is not None:
        log_return = np.full_like(close, np.nan, dtype=np.float32)
        mask = (close[:-1] > 0) & (close[1:] > 0)
        log_return[1:] = np.where(mask, np.log(close[1:] / close[:-1]), np.nan)
        log_return[0] = 0.0

    for k, recipe in enumerate(catalog):
        if not _check_required_fields(recipe, cube):
            continue

        h = recipe.horizon_bars
        span_6h = 6 * h
        span_12h = 12 * h

        if recipe.family == "time_series_trend":
            if close is None or log_return is None:
                continue
            ret = _ewm(log_return, span_6h)
            vol = _ewm(np.abs(log_return), span_6h)
            vol = np.where(vol < 1e-12, 1e-12, vol)
            scores[:, :, k] = ret / vol
            valid[:, :, k] = cube.eligible_2d & np.isfinite(scores[:, :, k])

        elif recipe.family == "breakout":
            if high is None or low is None or close is None:
                continue
            mid = (high + low) / 2.0
            roll_mid = _ewm(mid, span_12h)
            atr_val = _atr(high, low, close, span_6h)
            atr_val = np.where(atr_val < 1e-12, 1e-12, atr_val)
            scores[:, :, k] = (close - roll_mid) / atr_val
            valid[:, :, k] = cube.eligible_2d & np.isfinite(scores[:, :, k])

        elif recipe.family == "cross_sectional_momentum":
            if close is None:
                continue
            ret_6h = np.full_like(close, np.nan, dtype=np.float32)
            lookback = min(span_6h, n_bars - 1)
            ret_6h[lookback:] = np.where(
                (close[:-lookback] > 0) & (close[lookback:] > 0),
                np.log(close[lookback:] / close[:-lookback]),
                np.nan,
            )
            eligible_breadth = np.sum(cube.eligible_2d, axis=1, keepdims=True)
            min_breadth = 5
            breadth_ok = eligible_breadth >= min_breadth
            z = _robust_z_score(ret_6h)
            scores[:, :, k] = np.where(breadth_ok, z, np.nan)
            valid[:, :, k] = cube.eligible_2d & np.isfinite(scores[:, :, k]) & breadth_ok

        elif recipe.family == "short_term_reversal":
            if close is None:
                continue
            residual = np.full_like(close, np.nan, dtype=np.float32)
            lookback = min(h, n_bars - 1)
            residual[lookback:] = np.where(
                (close[:-lookback] > 0) & (close[lookback:] > 0),
                np.log(close[lookback:] / close[:-lookback]),
                np.nan,
            )
            z = _robust_z_score(residual)
            scores[:, :, k] = -z
            valid[:, :, k] = cube.eligible_2d & np.isfinite(scores[:, :, k])

        elif recipe.family == "carry_basis":
            if funding is None or premium is None:
                continue
            combined = funding + premium
            ewm_combined = _ewm(combined, span_6h)
            z = _robust_z_score(ewm_combined)
            scores[:, :, k] = -z
            valid[:, :, k] = cube.eligible_2d & np.isfinite(scores[:, :, k])

        elif recipe.family == "flow_positioning":
            if taker_buy is None or volume is None or funding is None:
                continue
            vol_safe = np.where(volume > 0, volume, np.float32(1.0))
            taker_buy_f32 = taker_buy.astype(np.float32)
            taker_imbalance = ((taker_buy_f32 * 2.0 - volume) / vol_safe).astype(np.float32)
            taker_z = _robust_z_score(taker_imbalance)
            oi_delta_z = _robust_z_score(funding)
            lsr = cube.fields_2d.get("lsr", None)
            lsr_z = _robust_z_score(lsr) if lsr is not None else np.zeros_like(taker_z)
            scores[:, :, k] = taker_z * np.sign(oi_delta_z) - 0.25 * lsr_z
            valid[:, :, k] = cube.eligible_2d & np.isfinite(scores[:, :, k])

        elif recipe.family == "volatility_squeeze_keltner":
            high_arr = cube.fields_2d["high"]
            low_arr = cube.fields_2d["low"]
            close_arr = cube.fields_2d["close"]
            mid = (high_arr + low_arr) / 2.0
            roll_mid = _ewm(mid, span_12h)
            atr_val = _atr(high_arr, low_arr, close_arr, span_6h)
            atr_val = np.where(atr_val < 1e-12, 1e-12, atr_val)
            scores[:, :, k] = (close_arr.astype(np.float32) - roll_mid) / atr_val
            valid[:, :, k] = cube.eligible_2d & np.isfinite(scores[:, :, k])

        elif recipe.family == "funding_carry_reversion":
            funding_arr = cube.fields_2d["funding"]
            premium_arr = cube.fields_2d["premium"]
            combined = funding_arr + premium_arr
            ewm_combined = _ewm(combined, span_6h)
            z = _robust_z_score(ewm_combined)
            scores[:, :, k] = -z
            valid[:, :, k] = cube.eligible_2d & np.isfinite(scores[:, :, k])

        elif recipe.family == "flow_imbalance_taker":
            taker_arr = cube.fields_2d["taker_buy_quote"]
            vol_arr = cube.fields_2d["quote_volume"]
            vol_safe = np.where(vol_arr > 0, vol_arr, np.float32(1.0))
            taker_buy_f32 = taker_arr.astype(np.float32)
            imbalance = ((taker_buy_f32 * 2.0 - vol_arr) / vol_safe).astype(np.float32)
            z = _robust_z_score(imbalance)
            scores[:, :, k] = z
            valid[:, :, k] = cube.eligible_2d & np.isfinite(scores[:, :, k])

        elif recipe.family == "open_interest_confirmation":
            oi_arr = cube.fields_2d["open_interest"]
            vol_arr = cube.fields_2d["quote_volume"]
            oi_change = np.zeros_like(oi_arr)
            oi_change[1:] = np.diff(oi_arr, axis=0)
            vol_safe = np.where(vol_arr > 0, vol_arr, np.float32(1.0))
            oi_norm = (oi_change / vol_safe).astype(np.float32)
            z = _robust_z_score(oi_norm)
            scores[:, :, k] = z
            valid[:, :, k] = cube.eligible_2d & np.isfinite(scores[:, :, k])

    return RawAlphaTape(
        timestamps_ns=cube.timestamps_ns,
        symbols=cube.symbols,
        recipe_ids=recipe_ids,
        scores_3d=scores,
        valid_3d=valid,
        horizon_bars_1d=horizon_bars,
    )
