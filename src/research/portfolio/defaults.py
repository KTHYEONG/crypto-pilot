from __future__ import annotations

# Pre-declared data-complete USDT-perpetual majors. The daily liquidity
# selection still chooses the top five by trailing quote volume among these.
DEFAULT_SYMBOLS: tuple[str, ...] = (
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT",
    "ADAUSDT", "DOGEUSDT", "LTCUSDT", "LINKUSDT", "AVAXUSDT",
)

STRESS_FEE_MULT = 1.5
STRESS_SLIPPAGE_MULT = 2.0
