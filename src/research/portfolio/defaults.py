from __future__ import annotations

# Pre-declared data-complete USDT-perpetual majors. The trailing quote-volume
# ranking used for dynamic symbol selection lives in the rolling path
# (select_symbols_for_window), not here.
DEFAULT_SYMBOLS: tuple[str, ...] = (
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT",
    "ADAUSDT", "DOGEUSDT", "LTCUSDT", "LINKUSDT", "AVAXUSDT",
)

STRESS_FEE_MULT = 1.5
STRESS_SLIPPAGE_MULT = 2.0
