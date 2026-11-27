from __future__ import annotations

import hashlib
import itertools
import os
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from src.common.errors import DataIntegrityError
from src.market_data.mirror.datasets import MirrorDataset

MIRROR_FORMAT_VERSION: int = 1
MIRROR_ZSTD_LEVEL: int = 19
MIRROR_ROW_GROUP_ROWS: int = 1 << 20
MIRROR_BLOCK_BYTES: int = 1 << 22

_MAX_DIGITS = 18
_HASH_CHUNK_BYTES = 1 << 20
_NUMERIC_PATTERN = r"^(?P<sign>-?)(?P<int>0|[1-9][0-9]*)(?:\.(?P<frac>[0-9]+))?$"
_ZERO_PAD = "0" * _MAX_DIGITS
_RESIDUAL_STORED_NAMES = {"first_trade_id": "first_trade_gap", "last_trade_id": "trade_span"}
_RESIDUAL_EXACT_IDS = ("agg_trade_id", "first_trade_id", "last_trade_id", "transact_time")
_META_VERSION_KEY = b"mirror_format_version"
_META_DATASET_KEY = b"mirror_dataset"


class UnencodableSourceError(Exception):
    """This CSV cannot be stored columnar byte-exactly; the caller keeps the official zip instead."""


@dataclass(frozen=True, slots=True)
class SourceLayout:
    header_line: str | None
    trailing_newline: bool
    csv_sha256: str
    csv_bytes: int


@dataclass(frozen=True, slots=True)
class _EncodedNumeric:
    value: pa.Array
    dp: pa.Array
    txt: pa.Array
    scale: int
    text_cells: int


def _stored(dataset: MirrorDataset, column: str) -> str:
    return _RESIDUAL_STORED_NAMES.get(column, column) if dataset.id_residual else column


def _numeric_fields(stored: str) -> list[pa.Field]:
    return [
        pa.field(stored, pa.int64()),
        pa.field(f"{stored}__dp", pa.int8()),
        pa.field(f"{stored}__scale", pa.int8()),
        pa.field(f"{stored}__txt", pa.string()),
    ]


def mirror_schema(dataset: MirrorDataset) -> pa.Schema:
    fields: list[pa.Field] = [
        pa.field("__source", pa.int32()),
        pa.field("__block", pa.int32()),
        pa.field("__symbol", pa.string()),
    ]
    for column in dataset.columns:
        stored = _stored(dataset, column)
        if column in dataset.text_columns:
            fields.append(pa.field(stored, pa.string()))
        else:
            fields.extend(_numeric_fields(stored))
    metadata = {_META_VERSION_KEY: str(MIRROR_FORMAT_VERSION).encode(), _META_DATASET_KEY: dataset.name.encode()}
    return pa.schema(fields, metadata=metadata)


def inspect_csv(open_stream: Callable[[], IO[bytes]], dataset: MirrorDataset) -> SourceLayout:
    digest = hashlib.sha256()
    total = 0
    head = bytearray()
    newline_seen = False
    header_end = 0
    has_cr = False
    last_byte = b""
    with open_stream() as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
            if not newline_seen:
                cut = chunk.find(b"\n")
                head += chunk if cut < 0 else chunk[:cut]
                if cut >= 0:
                    newline_seen = True
                    header_end = total + cut + 1
            has_cr = has_cr or b"\r" in chunk
            total += len(chunk)
            last_byte = chunk[-1:]
    if total == 0 or has_cr:
        raise UnencodableSourceError("source is empty or contains carriage returns")
    header_line: str | None = None
    if head[:1].isalpha():
        try:
            header_line = head.decode("ascii")
        except UnicodeDecodeError as exc:
            raise UnencodableSourceError("header is not ASCII") from exc
        if header_line != ",".join(dataset.columns):
            raise UnencodableSourceError("header differs from the dataset contract")
        if not newline_seen:
            header_end = total
        if total <= header_end:
            raise UnencodableSourceError("source has a header but no rows")
    return SourceLayout(
        header_line=header_line,
        trailing_newline=last_byte == b"\n",
        csv_sha256=digest.hexdigest(),
        csv_bytes=total,
    )


def _numeric_regex_child(matched: pa.Array, parts: pa.Array, name: str, empty: str) -> pa.Array:
    return pc.if_else(matched, pc.fill_null(parts.field(name), empty), empty)


def _render(value: pa.Array, dp: pa.Array, scale: int) -> pa.Array:
    magnitude = pc.cast(pc.abs(value), pa.string())
    if scale:
        magnitude = pc.utf8_lpad(magnitude, width=scale + 1, padding="0")
        integer = pc.utf8_slice_codeunits(magnitude, start=0, stop=-scale)
        fraction = pc.utf8_slice_codeunits(magnitude, start=-scale)
    else:
        integer = magnitude
        fraction = pa.repeat(pa.scalar("", pa.string()), len(value))
    signed_integer = pc.binary_join_element_wise(pc.if_else(pc.less(value, 0), "-", ""), integer, "")
    padded = pc.binary_join_element_wise(fraction, _ZERO_PAD, "")
    rendered = signed_integer
    for places in pc.unique(dp).to_pylist():
        if places == 0:
            continue
        piece = pc.binary_join_element_wise(signed_integer, pc.utf8_slice_codeunits(padded, start=0, stop=places), ".")
        rendered = pc.if_else(pc.equal(dp, places), piece, rendered)
    return rendered


def _encode_numeric(column: pa.Array) -> _EncodedNumeric:
    matched = pc.match_substring_regex(column, _NUMERIC_PATTERN)
    parts = pc.extract_regex(column, _NUMERIC_PATTERN)
    sign = _numeric_regex_child(matched, parts, "sign", "")
    integer = _numeric_regex_child(matched, parts, "int", "0")
    frac = _numeric_regex_child(matched, parts, "frac", "")
    shape = pc.and_(matched, pc.less_equal(pc.utf8_length(frac), _MAX_DIGITS))
    integer = pc.if_else(shape, integer, "0")
    frac = pc.if_else(shape, frac, "")
    sign = pc.if_else(shape, sign, "")
    scale = int(pc.max(pc.utf8_length(pc.utf8_rtrim(frac, "0"))).as_py() or 0)
    frac_at_scale = pc.utf8_slice_codeunits(pc.utf8_rpad(frac, width=scale, padding="0"), start=0, stop=scale)
    magnitude = pc.utf8_ltrim(pc.binary_join_element_wise(integer, frac_at_scale, ""), "0")
    shape = pc.and_(shape, pc.less_equal(pc.utf8_length(magnitude), _MAX_DIGITS))
    magnitude = pc.if_else(
        shape,
        pc.if_else(pc.equal(pc.utf8_length(magnitude), 0), "0", magnitude),
        "0",
    )
    absolute = pc.cast(magnitude, pa.int64())
    value = pc.if_else(pc.equal(sign, "-"), pc.negate(absolute), absolute)
    dp = pc.cast(pc.utf8_length(frac), pa.int8())
    good = pc.and_(shape, pc.equal(_render(value, dp, scale), column))
    return _EncodedNumeric(
        value=pc.if_else(good, value, pa.scalar(None, pa.int64())),
        dp=pc.if_else(good, dp, pa.scalar(None, pa.int8())),
        txt=pc.if_else(good, pa.scalar(None, pa.string()), column),
        scale=scale,
        text_cells=int(pc.count(pc.if_else(good, pa.scalar(None, pa.string()), column)).as_py()),
    )


def _require_plain_integer(encoded: _EncodedNumeric) -> None:
    max_dp = pc.max(encoded.dp).as_py()
    if encoded.text_cells or encoded.scale or max_dp not in (None, 0):
        raise UnencodableSourceError("aggTrades id column is not a canonical integer")


def _text_rows(columns: list[pa.Array]) -> bytes:
    lines = pc.binary_join_element_wise(*columns, ",")
    offsets = pa.array([0, len(lines)], pa.int32())
    joined: str = pc.binary_join(pa.ListArray.from_arrays(offsets, lines), "\n")[0].as_py()
    return joined.encode("utf-8")


def _encode_block(
    batch: pa.RecordBatch,
    dataset: MirrorDataset,
    *,
    source_index: int,
    block_index: int,
    symbol: str,
    schema: pa.Schema,
) -> pa.Table:
    rows = batch.num_rows
    arrays: dict[str, pa.Array] = {
        "__source": pa.array(np.full(rows, source_index, dtype=np.int32)),
        "__block": pa.array(np.full(rows, block_index, dtype=np.int32)),
        "__symbol": pa.repeat(pa.scalar(symbol, pa.string()), rows),
    }
    encoded: dict[str, _EncodedNumeric] = {}
    for index, column in enumerate(dataset.columns):
        stored = _stored(dataset, column)
        if column in dataset.text_columns:
            arrays[stored] = batch.column(index)
            continue
        encoded[column] = _encode_numeric(batch.column(index))
        if dataset.id_residual and column in _RESIDUAL_EXACT_IDS:
            _require_plain_integer(encoded[column])
    for column, item in encoded.items():
        if dataset.id_residual and column in _RESIDUAL_STORED_NAMES:
            continue
        stored = _stored(dataset, column)
        arrays[stored] = item.value
        arrays[f"{stored}__dp"] = item.dp
        arrays[f"{stored}__scale"] = pa.array(np.full(rows, item.scale, dtype=np.int8))
        arrays[f"{stored}__txt"] = item.txt
    if dataset.id_residual:
        first = encoded["first_trade_id"].value
        last = encoded["last_trade_id"].value
        prev = pa.concat_arrays([pa.array([-1], pa.int64()), last.slice(0, rows - 1)])
        gap = pc.subtract(pc.subtract(first, prev), 1)
        span = pc.subtract(last, first)
        for stored, residual in (("first_trade_gap", gap), ("trade_span", span)):
            arrays[stored] = residual
            arrays[f"{stored}__dp"] = pa.array(np.zeros(rows, dtype=np.int8))
            arrays[f"{stored}__scale"] = pa.array(np.zeros(rows, dtype=np.int8))
            arrays[f"{stored}__txt"] = pa.nulls(rows, pa.string())
    return pa.Table.from_arrays([arrays[name] for name in schema.names], schema=schema)


def _block_scale(table: pa.Table, stored: str) -> int:
    scales = pc.unique(table.column(f"{stored}__scale").combine_chunks()).to_pylist()
    if len(scales) != 1 or scales[0] is None:
        raise DataIntegrityError(f"column {stored} has inconsistent scale within a block")
    return int(scales[0])


def _decode_numeric(table: pa.Table, stored: str) -> pa.Array:
    raw_value = table.column(stored).combine_chunks()
    raw_dp = table.column(f"{stored}__dp").combine_chunks()
    txt = table.column(f"{stored}__txt").combine_chunks()
    lost = pc.and_(pc.or_(pc.is_null(raw_value), pc.is_null(raw_dp)), pc.is_null(txt))
    if pc.any(lost).as_py():
        raise DataIntegrityError(f"column {stored} has a cell with neither integer nor verbatim text")
    rendered = _render(pc.fill_null(raw_value, 0), pc.fill_null(raw_dp, 0), _block_scale(table, stored))
    return pc.coalesce(txt, rendered)


def decode_block(table: pa.Table, dataset: MirrorDataset) -> bytes:
    columns: list[pa.Array] = []
    decoded: dict[str, pa.Array] = {}
    if dataset.id_residual:
        gap = table.column("first_trade_gap").combine_chunks()
        span = table.column("trade_span").combine_chunks()
        if gap.null_count or span.null_count:
            raise DataIntegrityError("residual id columns contain nulls")
        gap_values = gap.to_numpy()
        span_values = span.to_numpy()
        last_values = np.cumsum(gap_values + 1 + span_values) - 1
        zeros = pa.array(np.zeros(table.num_rows, dtype=np.int8))
        scale_zero = 0
        decoded["first_trade_id"] = _render(pa.array(last_values - span_values), zeros, scale_zero)
        decoded["last_trade_id"] = _render(pa.array(last_values), zeros, scale_zero)
    for column in dataset.columns:
        stored = _stored(dataset, column)
        if column in dataset.text_columns:
            columns.append(table.column(stored).combine_chunks())
        elif column in decoded:
            columns.append(decoded[column])
        else:
            columns.append(_decode_numeric(table, stored))
    return _text_rows(columns)


def _text_batches(
    stream: IO[bytes], dataset: MirrorDataset, *, skip_header: bool, block_bytes: int
) -> Iterator[pa.RecordBatch]:
    names = list(dataset.columns)
    try:
        reader = pacsv.open_csv(
            stream,
            read_options=pacsv.ReadOptions(column_names=names, block_size=block_bytes),
            parse_options=pacsv.ParseOptions(quote_char=False),
            convert_options=pacsv.ConvertOptions(
                column_types={name: pa.string() for name in names},
                strings_can_be_null=False,
            ),
        )
        for batch in reader:
            if skip_header:
                batch = batch.slice(1)
                skip_header = False
            if batch.num_rows:
                yield batch
    except pa.ArrowInvalid as exc:
        raise UnencodableSourceError(f"CSV cannot be parsed losslessly: {exc}") from exc


def encode_source(
    open_stream: Callable[[], IO[bytes]],
    dataset: MirrorDataset,
    layout: SourceLayout,
    *,
    symbol: str,
    source_index: int,
    first_block: int,
    block_bytes: int = MIRROR_BLOCK_BYTES,
) -> Iterator[pa.Table]:
    schema = mirror_schema(dataset)
    digest = hashlib.sha256()
    if layout.header_line is not None:
        digest.update(layout.header_line.encode("ascii") + b"\n")
    emitted = False
    block_index = first_block
    with open_stream() as stream:
        for batch in _text_batches(
            stream, dataset, skip_header=layout.header_line is not None, block_bytes=block_bytes
        ):
            table = _encode_block(
                batch,
                dataset,
                source_index=source_index,
                block_index=block_index,
                symbol=symbol,
                schema=schema,
            )
            rows = _text_rows(list(batch.columns))
            if decode_block(table, dataset) != rows:
                raise UnencodableSourceError("block does not decode to its source rows")
            if emitted:
                digest.update(b"\n")
            digest.update(rows)
            emitted = True
            block_index += 1
            yield table
    if not emitted:
        raise UnencodableSourceError("source has no data rows")
    if layout.trailing_newline:
        digest.update(b"\n")
    if digest.hexdigest() != layout.csv_sha256:
        raise UnencodableSourceError("rebuilt CSV does not match the source checksum")


class MirrorObjectWriter:
    def __init__(self, path: Path, dataset: MirrorDataset, *, row_group_rows: int = MIRROR_ROW_GROUP_ROWS) -> None:
        self._path = path
        self._tmp_path = path.with_name(path.name + ".tmp")
        self._dataset = dataset
        self._schema = mirror_schema(dataset)
        self._row_group_rows = row_group_rows
        self._pending: list[pa.Table] = []
        self._pending_rows = 0
        self._rows_appended = 0
        self._writer: pq.ParquetWriter | None = None
        self._finished = False

    def append(self, table: pa.Table) -> None:
        if self._finished:
            raise RuntimeError("writer is already closed or aborted")
        if not table.schema.equals(self._schema, check_metadata=True):
            raise ValueError("table schema does not match the dataset mirror schema")
        self._pending.append(table)
        self._pending_rows += table.num_rows
        self._rows_appended += table.num_rows
        if self._pending_rows >= self._row_group_rows:
            self._flush()

    def close(self) -> None:
        if self._finished:
            raise RuntimeError("writer is already closed or aborted")
        if self._rows_appended == 0:
            self.abort()
            raise ValueError("no rows were appended")
        self._flush()
        assert self._writer is not None
        self._writer.close()
        self._writer = None
        fd = os.open(self._tmp_path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(self._tmp_path, self._path)
        self._finished = True

    def abort(self) -> None:
        self._finished = True
        self._pending = []
        writer, self._writer = self._writer, None
        try:
            if writer is not None:
                writer.close()
        finally:
            self._tmp_path.unlink(missing_ok=True)

    def _flush(self) -> None:
        if not self._pending:
            return
        if self._writer is None:
            self._writer = pq.ParquetWriter(
                self._tmp_path,
                self._schema,
                compression="zstd",
                compression_level=MIRROR_ZSTD_LEVEL,
                write_statistics=False,
                use_dictionary=[name for name in self._schema.names if name not in _column_encodings(self._dataset)],
                column_encoding=_column_encodings(self._dataset),
            )
        self._writer.write_table(pa.concat_tables(self._pending), row_group_size=self._row_group_rows)
        self._pending = []
        self._pending_rows = 0


def _column_encodings(dataset: MirrorDataset) -> dict[str, str]:
    if dataset.id_residual:
        encodings = dict.fromkeys(("agg_trade_id", "transact_time", "price"), "DELTA_BINARY_PACKED")
        encodings["quantity"] = "BYTE_STREAM_SPLIT"
        return encodings
    return {column: "BYTE_STREAM_SPLIT" for column in dataset.columns if column not in dataset.text_columns}


def _source_blocks(
    path: Path, dataset: MirrorDataset, source_index: int | None = None
) -> Iterator[tuple[int, bytes]]:
    """(source index, rebuilt block rows) in file order from a single pass over the row groups."""
    pending: list[pa.Table] = []
    pending_key: tuple[int, int] | None = None
    with pq.ParquetFile(path) as parquet:
        for group in range(parquet.num_row_groups):
            table = parquet.read_row_group(group)
            if source_index is not None:
                table = table.filter(pc.equal(table.column("__source"), source_index))
            if table.num_rows == 0:
                continue
            sources = table.column("__source").to_numpy()
            blocks = table.column("__block").to_numpy()
            changed = (np.diff(sources) != 0) | (np.diff(blocks) != 0)
            bounds = [0, *(np.flatnonzero(changed) + 1).tolist(), table.num_rows]
            for start, end in itertools.pairwise(bounds):
                key = (int(sources[start]), int(blocks[start]))
                if pending_key is not None and key != pending_key:
                    yield pending_key[0], decode_block(pa.concat_tables(pending), dataset)
                    pending = []
                pending.append(table.slice(start, end - start))
                pending_key = key
    if pending_key is not None:
        yield pending_key[0], decode_block(pa.concat_tables(pending), dataset)


def iter_source_csv(path: Path, dataset: MirrorDataset, source_index: int, layout: SourceLayout) -> Iterator[bytes]:
    if layout.header_line is not None:
        yield layout.header_line.encode("ascii") + b"\n"
    emitted = False
    for _, block in _source_blocks(path, dataset, source_index):
        if emitted:
            yield b"\n"
        yield block
        emitted = True
    if layout.trailing_newline and emitted:
        yield b"\n"


def verify_object(path: Path, dataset: MirrorDataset, layouts: Mapping[int, SourceLayout]) -> None:
    with pq.ParquetFile(path) as parquet:
        metadata = parquet.schema_arrow.metadata or {}
    if (
        metadata.get(_META_DATASET_KEY) != dataset.name.encode()
        or metadata.get(_META_VERSION_KEY) != str(MIRROR_FORMAT_VERSION).encode()
    ):
        raise DataIntegrityError(f"{path} metadata does not match dataset {dataset.name}")
    digests: dict[int, tuple[hashlib._Hash, int]] = {}
    for index, block in _source_blocks(path, dataset):
        layout = layouts.get(index)
        if layout is None:
            raise DataIntegrityError(f"{path} contains unexpected source index {index}")
        if index in digests:
            digest, size = digests[index]
            piece = b"\n" + block
        else:
            digest, size = hashlib.sha256(), 0
            header = b"" if layout.header_line is None else layout.header_line.encode("ascii") + b"\n"
            piece = header + block
        digest.update(piece)
        digests[index] = (digest, size + len(piece))
    if set(layouts) - set(digests):
        raise DataIntegrityError(f"{path} is missing source indexes")
    for index, layout in layouts.items():
        digest, size = digests[index]
        if layout.trailing_newline:
            digest.update(b"\n")
            size += 1
        if size != layout.csv_bytes or digest.hexdigest() != layout.csv_sha256:
            raise DataIntegrityError(f"{path} source {index} does not rebuild to its recorded CSV")
