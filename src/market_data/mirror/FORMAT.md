# Mirror object format (version 1)

This describes the Parquet objects written by `src/market_data/mirror`. It is
self-contained: a reader needs only this file and a Parquet library.

## Purpose and promise

Each object stores one or more original Binance Vision USD-M CSV files. Every
original CSV can be rebuilt byte for byte from the object. Files that cannot be
stored this way are kept as the untouched official zip (`verbatim` objects).

## Object layout

One Parquet file per dataset bucket, such as one month of one dataset. Rows of
several source CSVs may be packed into the same object. Each source is
identified by `__source`.

Fixed leading columns:

| column | type | meaning |
|---|---|---|
| `__source` | int32 | index of the source CSV within the object |
| `__block` | int32 | index of the block within the object; a block is one contiguous run of source rows |
| `__symbol` | string | symbol the source belongs to |

Then, for every CSV column `c` in the dataset's header order:

- Text columns (for example `symbol`, `create_time`, `is_buyer_maker`): one string
  column `c` holding the cell exactly as written in the CSV.
- Numeric columns: four columns.
  - `c` (int64, nullable): unscaled integer value.
  - `c__dp` (int8, nullable): number of decimal places printed in the source cell.
  - `c__scale` (int8): decimal scale of the block. Constant within a block.
  - `c__txt` (string, nullable): the original cell text, present only when the
    integer form cannot reproduce the text.

For `aggTrades`, `first_trade_id` and `last_trade_id` are stored as
`first_trade_gap` and `trade_span` (same four-column shape) so that a plain reader
cannot mistake them for trade ids. See the residual decode below.

## Value formula

For a numeric cell in a block with scale `s`:

- If `c__txt` is not null, the cell text is exactly `c__txt`.
- Otherwise the value is `c / 10**s` and the text is that value written with
  exactly `c__dp` decimal places, with no sign for zero.

Decimal arithmetic must be exact: never round through floating point.

## Writing rules

- A cell is integer-coded only if its text matches `-?(0|[1-9][0-9]*)(\.[0-9]+)?`,
  has at most 18 decimal places, and its digits at the block scale fit in 18
  characters. The integer form must then reproduce the text exactly; otherwise the
  text goes to `c__txt`. For example `-0.00000000`, `007`, `0E-8` and `""` are
  stored as text.
- Block scale is the largest number of significant decimals (trailing zeros
  removed) among the block's integer-codable cells.
- Row order is the source order.

## aggTrades residual ids

Within each block, with the previous last id taken as -1 at the block start:

- `first_trade_gap = first_trade_id - previous_last_trade_id - 1`
- `trade_span = last_trade_id - first_trade_id`

Decode, for the rows of one block in order:

- `last_trade_id = cumsum(first_trade_gap + 1 + trade_span) - 1`
- `first_trade_id = last_trade_id - trade_span`

## Rebuilding a source CSV

1. Emit the header line and a newline, if the source had a header.
2. For each block of the source in order, emit the block's rows: each row is its
   cells joined by commas. Rows are separated by newlines. Emit a newline between
   blocks.
3. Emit a final newline if the source ended with one.

The result must match the sha256 and byte count recorded for the source. A
mismatch means the object is corrupt.

## Sidecar index

Part 2 of the mirror stores one index record per source alongside the objects.
It records the source key, the upstream ETag, the official zip sha256, the
source index, the storage kind (`columnar` or `verbatim`), the header line, the
trailing newline flag, the CSV sha256 and the CSV byte count. Those fields are
needed to rebuild and verify each source without reading unrelated sources.

## Verbatim objects

A `verbatim` object is the untouched official zip. It is used when a CSV cannot
be stored in this format without loss, for example CRLF line endings, ragged rows,
or a changed header.

## Parquet settings

- zstd compression at level 19; no column statistics.
- `BYTE_STREAM_SPLIT` on int64 value columns, except that aggTrades uses
  `DELTA_BINARY_PACKED` on `agg_trade_id`, `transact_time` and `price`, and
  `BYTE_STREAM_SPLIT` on `quantity`.
- All other columns use dictionary encoding.
- Key-value metadata: `mirror_format_version`, `mirror_dataset`.
