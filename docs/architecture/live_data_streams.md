# Live Data Streams

| stream | path | partition scheme | time column | schema (key cols) | retention |
|---|---|---|---|---|---|
| ohlcv/1h | `data/futures/ohlcv/1h/*.parquet` | per-symbol parquet | `timestamp` (epoch ms, int64) | `timestamp, open, high, low, close, volume, quote_vol` | `data_retention_days` (default 220) |
| markPriceKlines/1h | `data/futures/markPriceKlines/1h/*.parquet` | per-symbol parquet | `timestamp` (epoch ms, int64) | `timestamp, open, high, low, close` | `data_retention_days` (default 220) |
| funding | `data/futures/funding/*.parquet` | per-symbol parquet | `timestamp` (epoch ms, int64) | `timestamp, funding_rate` | `data_retention_days` (default 220) |
| live_fills | `data/state/live_fills/*.parquet` | monthly shards | `decision_time` (UTC), `timestamp` (fill time) | `decision_time, timestamp, symbol, side, qty, price` | kept |
| live_execution_quality | `data/state/live_execution_quality/*.parquet` | monthly shards | `decision_time` (UTC) | `decision_time, symbol, slippage_bps` | kept |
| live_microstructure | `data/state/live_microstructure/*.parquet` | monthly shards | `decision_time` (UTC) | `decision_time, symbol, spread_bps` | kept |
| live_portfolio_state | `data/state/live_portfolio_state/*.parquet` | rotating shards | `decision_time` (UTC) | `decision_time, equity, positions` | kept |
| live_orderbook | `data/state/live_orderbook/live_orderbook_YYYYMMDD.parquet` | daily parquet | `captured_at` (UTC) | `captured_at, decision_time, symbol, best_bid, best_ask` | `orderbook_retention_days` (default 365) |
| live_tax_ledger | `data/state/live_tax_ledger/*.jsonl` | monthly JSONL | `decision_time` (UTC) | `decision_time, realized_pnl` | kept forever |

All parquet files use zstd compression.
`decision_time` is the canonical event column; `live_fills` additionally carries `timestamp` as fill time.
Load via the module `load_*` helpers which union active and archived shards.
