from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import talib
from numba import njit

CacheMode = str


def _safe_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _normalize_cache_mode(cache_mode: str) -> CacheMode:
    mode = str(cache_mode).strip().lower()
    if mode in {"none", "off", "false"}:
        return "disabled"
    if mode in {"signature", "sig"}:
        return "signature"
    if mode in {"id", "object_id"}:
        return "id"
    return "disabled"


def _data_signature(data_obj: Any) -> tuple[Any, ...]:
    if isinstance(data_obj, pd.DataFrame):
        n = len(data_obj)
        if n == 0:
            return ("df", 0)
        idx0 = _safe_scalar(data_obj.index[0])
        idxn = _safe_scalar(data_obj.index[-1])
        if "timestamp" in data_obj.columns:
            t0 = int(data_obj["timestamp"].iloc[0])
            tn = int(data_obj["timestamp"].iloc[-1])
        elif "datetime" in data_obj.columns:
            t0 = _safe_scalar(data_obj["datetime"].iloc[0])
            tn = _safe_scalar(data_obj["datetime"].iloc[-1])
        else:
            t0 = idx0
            tn = idxn
        c0 = float(data_obj["close"].iloc[0]) if "close" in data_obj.columns else 0.0
        cn = float(data_obj["close"].iloc[-1]) if "close" in data_obj.columns else 0.0
        return ("df", n, idx0, idxn, t0, tn, c0, cn)

    if isinstance(data_obj, pd.Series):
        n = len(data_obj)
        if n == 0:
            return ("sr", 0)
        idx0 = _safe_scalar(data_obj.index[0])
        idxn = _safe_scalar(data_obj.index[-1])
        v0 = _safe_scalar(data_obj.iloc[0])
        vn = _safe_scalar(data_obj.iloc[-1])
        return ("sr", n, idx0, idxn, v0, vn)

    n = len(data_obj) if hasattr(data_obj, "__len__") else None
    return (type(data_obj).__name__, id(data_obj), n)


@njit
def _hurst_numba_logic(data: np.ndarray, window: int) -> np.ndarray:
    n = len(data)
    out = np.empty(n)
    out[:] = np.nan

    for i in range(window - 1, n):
        ts = data[i - window + 1 : i + 1]
        mean_val = np.mean(ts)
        ts_adj = ts - mean_val
        y = np.cumsum(ts_adj)
        r = np.max(y) - np.min(y)
        s = np.std(ts)

        if s > 0 and r > 0:
            h = np.log(r / s) / np.log(window)
            if h < 0:
                h = 0.0
            if h > 1:
                h = 1.0
            out[i] = h
        else:
            out[i] = 0.5

    return out


@njit
def _aroon_numba(high: np.ndarray, low: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Aroon Up/Down: 100 * (period - periods_since_high/low) / period. Causal."""
    n = len(high)
    aroon_up = np.empty(n)
    aroon_down = np.empty(n)
    aroon_up[:] = np.nan
    aroon_down[:] = np.nan
    for i in range(window - 1, n):
        win_high = high[i - window + 1 : i + 1]
        win_low = low[i - window + 1 : i + 1]
        idx_max = np.argmax(win_high)
        idx_min = np.argmin(win_low)
        aroon_up[i] = 100.0 * (idx_max + 1) / window
        aroon_down[i] = 100.0 * (idx_min + 1) / window
    return aroon_up, aroon_down


class IndicatorEngine:
    def __init__(self, cache_mode: str = "disabled") -> None:
        self.cache_mode: CacheMode = _normalize_cache_mode(cache_mode)
        self._indicator_cache: dict[tuple[Any, ...], Any] = {}

    def clear_indicator_cache(self) -> None:
        self._indicator_cache = {}

    def _cache_key(
        self,
        func_name: str,
        data_obj: Any,
        params: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[Any, ...] | None:
        if self.cache_mode == "disabled":
            return None
        data_key = id(data_obj) if self.cache_mode == "id" else _data_signature(data_obj)
        return (func_name, data_key, params, tuple(sorted(kwargs.items())))

    def _cached_call(
        self,
        func_name: str,
        data_obj: Any,
        params: tuple[Any, ...],
        kwargs: dict[str, Any],
        compute_fn: Callable[[], Any],
    ) -> Any:
        key = self._cache_key(func_name, data_obj, params, kwargs)
        if key is None:
            return compute_fn()
        cached = self._indicator_cache.get(key)
        if cached is not None:
            return cached
        result = compute_fn()
        self._indicator_cache[key] = result
        return result

    def calculate_sma(self, series: pd.Series, window: int) -> pd.Series:
        return self._cached_call(
            "calculate_sma",
            series,
            (int(window),),
            {},
            lambda: pd.Series(talib.SMA(series.values.astype(np.float64), timeperiod=window), index=series.index),
        )

    def calculate_ema(self, series: pd.Series, window: int) -> pd.Series:
        return self._cached_call(
            "calculate_ema",
            series,
            (int(window),),
            {},
            lambda: pd.Series(talib.EMA(series.values.astype(np.float64), timeperiod=window), index=series.index),
        )

    def calculate_wma(self, series: pd.Series, window: int) -> pd.Series:
        return self._cached_call(
            "calculate_wma",
            series,
            (int(window),),
            {},
            lambda: pd.Series(talib.WMA(series.values.astype(np.float64), timeperiod=window), index=series.index),
        )

    def calculate_hma(self, series: pd.Series, window: int) -> pd.Series:
        return self._cached_call(
            "calculate_hma",
            series,
            (int(window),),
            {},
            lambda: self.calculate_wma(
                2 * self.calculate_wma(series, window // 2) - self.calculate_wma(series, window),
                int(np.sqrt(window)),
            ),
        )

    def calculate_dema(self, series: pd.Series, window: int) -> pd.Series:
        return self._cached_call(
            "calculate_dema",
            series,
            (int(window),),
            {},
            lambda: pd.Series(talib.DEMA(series.values.astype(np.float64), timeperiod=window), index=series.index),
        )

    def calculate_tema(self, series: pd.Series, window: int) -> pd.Series:
        return self._cached_call(
            "calculate_tema",
            series,
            (int(window),),
            {},
            lambda: pd.Series(talib.TEMA(series.values.astype(np.float64), timeperiod=window), index=series.index),
        )

    def calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        return self._cached_call(
            "calculate_atr",
            df,
            (int(window),),
            {},
            lambda: pd.Series(
                talib.ATR(
                    df["high"].values.astype(np.float64),
                    df["low"].values.astype(np.float64),
                    df["close"].values.astype(np.float64),
                    timeperiod=window,
                ),
                index=df.index,
            ),
        )

    def calculate_bollinger_bands(
        self,
        df: pd.DataFrame,
        window: int = 20,
        num_std: float = 2.0,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        def _compute() -> tuple[pd.Series, pd.Series, pd.Series]:
            upper, middle, lower = talib.BBANDS(
                df["close"].values.astype(np.float64),
                timeperiod=window,
                nbdevup=num_std,
                nbdevdn=num_std,
                matype=0,
            )
            return (
                pd.Series(upper, index=df.index),
                pd.Series(middle, index=df.index),
                pd.Series(lower, index=df.index),
            )

        return self._cached_call(
            "calculate_bollinger_bands",
            df,
            (int(window), float(num_std)),
            {},
            _compute,
        )

    def calculate_keltner_channels(
        self,
        df: pd.DataFrame,
        window: int = 20,
        atr_mult: float = 1.5,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        def _compute() -> tuple[pd.Series, pd.Series, pd.Series]:
            mid = self.calculate_ema(df["close"], window)
            atr = self.calculate_atr(df, window=window)
            upper = mid + (atr * atr_mult)
            lower = mid - (atr * atr_mult)
            return upper, mid, lower

        return self._cached_call(
            "calculate_keltner_channels",
            df,
            (int(window), float(atr_mult)),
            {},
            _compute,
        )

    def calculate_keltner_channel(
        self,
        df: pd.DataFrame,
        window: int = 20,
        atr_mult: float = 1.5,
    ) -> tuple[pd.Series, pd.Series]:
        return self._cached_call(
            "calculate_keltner_channel",
            df,
            (int(window), float(atr_mult)),
            {},
            lambda: (
                self.calculate_ema(df["close"], window) + (self.calculate_atr(df, window=window) * atr_mult),
                self.calculate_ema(df["close"], window) - (self.calculate_atr(df, window=window) * atr_mult),
            ),
        )

    def calculate_supertrend(
        self,
        df: pd.DataFrame,
        period: int = 10,
        multiplier: float = 3.0,
    ) -> pd.Series:
        def _compute() -> pd.Series:
            atr = self.calculate_atr(df, window=period)
            hl2 = (df["high"] + df["low"]) / 2
            basic_upper = hl2 + (multiplier * atr)
            basic_lower = hl2 - (multiplier * atr)

            close = df["close"].values
            bu = basic_upper.values
            bl = basic_lower.values
            final_upper = np.zeros(len(df))
            final_lower = np.zeros(len(df))
            trend = np.zeros(len(df), dtype=int)

            upper = bu[0]
            lower = bl[0]
            curr_trend = 1
            final_upper[0] = upper
            final_lower[0] = lower
            trend[0] = curr_trend

            for i in range(1, len(df)):
                c = close[i]
                c_prev = close[i - 1]
                if bu[i] < upper or c_prev > upper:
                    upper = bu[i]
                if bl[i] > lower or c_prev < lower:
                    lower = bl[i]
                curr_trend = (-1 if c < lower else 1) if curr_trend == 1 else 1 if c > upper else -1
                final_upper[i] = upper
                final_lower[i] = lower
                trend[i] = curr_trend

            return pd.Series(trend, index=df.index)

        return self._cached_call(
            "calculate_supertrend",
            df,
            (int(period), float(multiplier)),
            {},
            _compute,
        )

    def calculate_vhf(self, series: pd.Series, window: int = 28) -> pd.Series:
        return self._cached_call(
            "calculate_vhf",
            series,
            (int(window),),
            {},
            lambda: (
                (series.rolling(window).max() - series.rolling(window).min()).abs()
                / series.diff().abs().rolling(window).sum()
            ),
        )

    def calculate_adx(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        return self._cached_call(
            "calculate_adx",
            df,
            (int(window),),
            {},
            lambda: pd.Series(
                talib.ADX(
                    df["high"].values.astype(np.float64),
                    df["low"].values.astype(np.float64),
                    df["close"].values.astype(np.float64),
                    timeperiod=window,
                ),
                index=df.index,
            ),
        )

    def calculate_dmi(self, df: pd.DataFrame, window: int = 14) -> tuple[pd.Series, pd.Series]:
        """Directional Movement: +DI and -DI. Causal (Wilder smoothing uses past only)."""

        def _compute() -> tuple[pd.Series, pd.Series]:
            plus_di = talib.PLUS_DI(
                df["high"].values.astype(np.float64),
                df["low"].values.astype(np.float64),
                df["close"].values.astype(np.float64),
                timeperiod=window,
            )
            minus_di = talib.MINUS_DI(
                df["high"].values.astype(np.float64),
                df["low"].values.astype(np.float64),
                df["close"].values.astype(np.float64),
                timeperiod=window,
            )
            return (
                pd.Series(plus_di, index=df.index),
                pd.Series(minus_di, index=df.index),
            )

        return self._cached_call("calculate_dmi", df, (int(window),), {}, _compute)

    def calculate_aroon(self, df: pd.DataFrame, window: int = 14) -> tuple[pd.Series, pd.Series]:
        """Aroon Up/Down. Causal (rolling window uses only past and current bar)."""

        def _compute() -> tuple[pd.Series, pd.Series]:
            aup, adown = _aroon_numba(
                df["high"].values.astype(np.float64),
                df["low"].values.astype(np.float64),
                window,
            )
            return (
                pd.Series(aup, index=df.index),
                pd.Series(adown, index=df.index),
            )

        return self._cached_call("calculate_aroon", df, (int(window),), {}, _compute)

    def calculate_rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        return self._cached_call(
            "calculate_rsi",
            series,
            (int(window),),
            {},
            lambda: pd.Series(talib.RSI(series.values.astype(np.float64), timeperiod=window), index=series.index),
        )

    def calculate_stochastic(
        self,
        df: pd.DataFrame,
        window: int = 14,
        smooth_k: int = 3,
        smooth_d: int = 3,
    ) -> tuple[pd.Series, pd.Series]:
        def _compute() -> tuple[pd.Series, pd.Series]:
            stoch_k, stoch_d = talib.STOCH(
                df["high"].values.astype(np.float64),
                df["low"].values.astype(np.float64),
                df["close"].values.astype(np.float64),
                fastk_period=window,
                slowk_period=smooth_k,
                slowk_matype=0,
                slowd_period=smooth_d,
                slowd_matype=0,
            )
            return pd.Series(stoch_k, index=df.index), pd.Series(stoch_d, index=df.index)

        return self._cached_call(
            "calculate_stochastic",
            df,
            (int(window), int(smooth_k), int(smooth_d)),
            {},
            _compute,
        )

    def calculate_stoch_rsi(
        self,
        series: pd.Series,
        window: int = 14,
        smooth_k: int = 3,
        smooth_d: int = 3,
    ) -> tuple[pd.Series, pd.Series]:
        def _compute() -> tuple[pd.Series, pd.Series]:
            stoch_k, stoch_d = talib.STOCHRSI(
                series.values.astype(np.float64),
                timeperiod=window,
                fastk_period=smooth_k,
                fastd_period=smooth_d,
                fastd_matype=0,
            )
            return pd.Series(stoch_k, index=series.index), pd.Series(stoch_d, index=series.index)

        return self._cached_call(
            "calculate_stoch_rsi",
            series,
            (int(window), int(smooth_k), int(smooth_d)),
            {},
            _compute,
        )

    def calculate_macd(
        self,
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        def _compute() -> tuple[pd.Series, pd.Series, pd.Series]:
            macd, macd_signal, macd_hist = talib.MACD(
                df["close"].values.astype(np.float64),
                fastperiod=fast,
                slowperiod=slow,
                signalperiod=signal,
            )
            return (
                pd.Series(macd, index=df.index),
                pd.Series(macd_signal, index=df.index),
                pd.Series(macd_hist, index=df.index),
            )

        return self._cached_call(
            "calculate_macd",
            df,
            (int(fast), int(slow), int(signal)),
            {},
            _compute,
        )

    def calculate_ichimoku(
        self,
        df: pd.DataFrame,
        tenkan_window: int = 9,
        kijun_window: int = 26,
        senkou_span_b_window: int = 52,
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        def _compute() -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
            high = df["high"].values.astype(np.float64)
            low = df["low"].values.astype(np.float64)
            tenkan_sen = (talib.MAX(high, timeperiod=tenkan_window) + talib.MIN(low, timeperiod=tenkan_window)) / 2
            kijun_sen = (talib.MAX(high, timeperiod=kijun_window) + talib.MIN(low, timeperiod=kijun_window)) / 2
            senkou_span_a = (tenkan_sen + kijun_sen) / 2
            sb_high = talib.MAX(high, timeperiod=senkou_span_b_window)
            sb_low = talib.MIN(low, timeperiod=senkou_span_b_window)
            senkou_span_b = (sb_high + sb_low) / 2
            return (
                pd.Series(tenkan_sen, index=df.index),
                pd.Series(kijun_sen, index=df.index),
                pd.Series(senkou_span_a, index=df.index).shift(kijun_window),
                pd.Series(senkou_span_b, index=df.index).shift(kijun_window),
            )

        return self._cached_call(
            "calculate_ichimoku",
            df,
            (int(tenkan_window), int(kijun_window), int(senkou_span_b_window)),
            {},
            _compute,
        )

    def calculate_cci(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        return self._cached_call(
            "calculate_cci",
            df,
            (int(window),),
            {},
            lambda: pd.Series(
                talib.CCI(
                    df["high"].values.astype(np.float64),
                    df["low"].values.astype(np.float64),
                    df["close"].values.astype(np.float64),
                    timeperiod=window,
                ),
                index=df.index,
            ),
        )

    def calculate_mfi(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        return self._cached_call(
            "calculate_mfi",
            df,
            (int(window),),
            {},
            lambda: pd.Series(
                talib.MFI(
                    df["high"].values.astype(np.float64),
                    df["low"].values.astype(np.float64),
                    df["close"].values.astype(np.float64),
                    df["volume"].values.astype(np.float64),
                    timeperiod=window,
                ),
                index=df.index,
            ),
        )

    def calculate_parabolic_sar(
        self,
        df: pd.DataFrame,
        step: float = 0.02,
        max_step: float = 0.2,
    ) -> tuple[pd.Series, None]:
        return self._cached_call(
            "calculate_parabolic_sar",
            df,
            (float(step), float(max_step)),
            {},
            lambda: (
                pd.Series(
                    talib.SAR(
                        df["high"].values.astype(np.float64),
                        df["low"].values.astype(np.float64),
                        acceleration=step,
                        maximum=max_step,
                    ),
                    index=df.index,
                ),
                None,
            ),
        )

    def calculate_vwap(
        self,
        df: pd.DataFrame,
        window: int | None = None,
        std_mult: float = 1.5,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        def _compute() -> tuple[pd.Series, pd.Series, pd.Series]:
            typical_price = (df["high"] + df["low"] + df["close"]) / 3
            tp_volume = typical_price * df["volume"]
            if window is None:
                vwap = tp_volume.cumsum() / df["volume"].cumsum()
                vwap_std = typical_price.expanding().std()
            else:
                vwap = tp_volume.rolling(window=window).sum() / df["volume"].rolling(window=window).sum()
                vwap_std = typical_price.rolling(window=window).std()
            vwap_upper = vwap + (vwap_std * std_mult)
            vwap_lower = vwap - (vwap_std * std_mult)
            return vwap, vwap_upper, vwap_lower

        return self._cached_call(
            "calculate_vwap",
            df,
            (None if window is None else int(window), float(std_mult)),
            {},
            _compute,
        )

    def calculate_cmf(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        def _compute() -> pd.Series:
            mf_multiplier = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"])
            mf_multiplier = mf_multiplier.replace([np.inf, -np.inf], 0).fillna(0)
            mf_volume = mf_multiplier * df["volume"]
            return mf_volume.rolling(window=window).sum() / df["volume"].rolling(window=window).sum()

        return self._cached_call(
            "calculate_cmf",
            df,
            (int(window),),
            {},
            _compute,
        )

    def calculate_hurst_exponent(self, series: pd.Series, window: int = 100) -> pd.Series:
        return self._cached_call(
            "calculate_hurst_exponent",
            series,
            (int(window),),
            {},
            lambda: pd.Series(_hurst_numba_logic(series.values.astype(np.float64), window), index=series.index),
        )

    def calculate_efficiency_ratio(self, series: pd.Series, window: int = 10) -> pd.Series:
        return self._cached_call(
            "calculate_efficiency_ratio",
            series,
            (int(window),),
            {},
            lambda: (
                (series - series.shift(window)).abs()
                / (series - series.shift(1)).abs().rolling(window=window).sum().replace(0, 0.001)
            ),
        )

    def calculate_natr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        return self._cached_call(
            "calculate_natr",
            df,
            (int(window),),
            {},
            lambda: (self.calculate_atr(df, window=window) / df["close"]) * 100,
        )

    def calculate_garman_klass_vol(self, df: pd.DataFrame, window: int = 30) -> pd.Series:
        def _compute() -> pd.Series:
            log_hl = np.log(df["high"] / df["low"])
            log_co = np.log(df["close"] / df["open"])
            gk = 0.5 * (log_hl**2) - (2 * np.log(2) - 1) * (log_co**2)
            return np.sqrt(gk.rolling(window=window).mean())

        return self._cached_call(
            "calculate_garman_klass_vol",
            df,
            (int(window),),
            {},
            _compute,
        )

    def calculate_force_index(self, df: pd.DataFrame, smooth_period: int = 2) -> pd.Series:
        """Force Index: (close - prev_close) * volume, then EMA smoothed. Causal."""

        def _compute() -> pd.Series:
            raw = df["close"].diff() * df["volume"]
            raw = raw.fillna(0.0)
            smoothed = pd.Series(
                talib.EMA(raw.values.astype(np.float64), timeperiod=smooth_period),
                index=df.index,
            )
            return smoothed

        return self._cached_call("calculate_force_index", df, (int(smooth_period),), {}, _compute)

    def calculate_williams_r(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Williams %%R: -100 * (HH - Close) / (HH - LL). Range [-100, 0]. Causal. Div-by-zero -> -50."""

        def _compute() -> pd.Series:
            out = pd.Series(
                talib.WILLR(
                    df["high"].values.astype(np.float64),
                    df["low"].values.astype(np.float64),
                    df["close"].values.astype(np.float64),
                    timeperiod=window,
                ),
                index=df.index,
            )
            out = out.replace([np.inf, -np.inf], np.nan).fillna(-50.0)
            return out.clip(-100.0, 0.0)

        return self._cached_call("calculate_williams_r", df, (int(window),), {}, _compute)

    def calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """On Balance Volume: cumulative signed volume by close direction. Causal, no division."""

        def _compute() -> pd.Series:
            delta = np.sign(df["close"].diff())
            delta = delta.fillna(0.0)
            return (delta * df["volume"]).cumsum()

        return self._cached_call("calculate_obv", df, (), {}, _compute)

    def calculate_roc(self, series: pd.Series, window: int = 10) -> pd.Series:
        """Rate of Change: (price / prev_price - 1) * 100. Pure momentum without smoothing."""
        return self._cached_call(
            "calculate_roc",
            series,
            (int(window),),
            {},
            lambda: pd.Series(talib.ROC(series.values.astype(np.float64), timeperiod=window), index=series.index),
        )

    def calculate_vwma(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Volume Weighted Moving Average: sum(price * volume) / sum(volume) over rolling window."""

        def _compute() -> pd.Series:
            pv = df["close"] * df["volume"]
            return pv.rolling(window=window).sum() / df["volume"].rolling(window=window).sum()

        return self._cached_call("calculate_vwma", df, (int(window),), {}, _compute)


def get_indicator_engine(domain: str = "generic") -> IndicatorEngine:
    """도메인별 환경 변수를 참조하여 IndicatorEngine 인스턴스를 생성하는 팩토리 함수."""
    env_suffix = domain.upper()
    cache_mode = os.getenv(f"{env_suffix}_INDICATOR_CACHE_MODE", os.getenv("INDICATOR_CACHE_MODE", "signature"))
    return IndicatorEngine(cache_mode=cache_mode)


# 기본 전역 인스턴스 (범용 사용)
global_ind = get_indicator_engine()
