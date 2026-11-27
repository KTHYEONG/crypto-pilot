from __future__ import annotations

import io
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.common.errors import DataIntegrityError
import src.market_data.mirror.codec as codec_module
from src.market_data.mirror.codec import (
    MIRROR_ROW_GROUP_ROWS,
    MirrorObjectWriter,
    SourceLayout,
    UnencodableSourceError,
    decode_block,
    encode_source,
    inspect_csv,
    iter_source_csv,
    mirror_schema,
    verify_object,
)
from src.market_data.mirror.datasets import MIRROR_DATASET_SPECS, MirrorDataset

_KLINE_ROWS = (
    "1704067200000,42000.10,42010.5,41990,42005.25,12.345,1704067259999,518000.123,100,6.1,256000.5,0\n"
    "1704067260000,42005.25,42020,42000,42010,7,1704067319999,293000,55,3.5,146000.25,0\n"
)
_METRICS_ROWS = (
    "2024-01-01 00:05:00,BTCUSDT,30000.12345678,1500000.5,1.2,0.9,1.5,0.7\n"
    "2024-01-01 00:00:00,BTCUSDT,30001,1500100,1.1,0.8,1.4,0.6\n"
)
_FUNDING_ROWS = "1704067200000,8,0.0001\n1704096000000,8,-0.00005\n"
_BOOK_ROWS = "2024-01-01T00:00:00,-5,1234.5,9999.99\n2024-01-01T00:00:00,-4,1300.25,10000\n"
_TRADE_ROWS = "100,42000.10,1.500,200,201,1704067200000,true\n103,42000.20,0.001,203,203,1704067200100,false\n"
_SAMPLES: dict[str, str] = {
    "metrics": _METRICS_ROWS,
    "fundingRate": _FUNDING_ROWS,
    "klines": _KLINE_ROWS,
    "markPriceKlines": _KLINE_ROWS,
    "indexPriceKlines": _KLINE_ROWS,
    "premiumIndexKlines": _KLINE_ROWS,
    "bookDepth": _BOOK_ROWS,
    "aggTrades": _TRADE_ROWS,
}


def _with_header(dataset: MirrorDataset, rows: str) -> str:
    return ",".join(dataset.columns) + "\n" + rows


def _open(raw: bytes) -> Callable[[], io.BytesIO]:
    def open_stream() -> io.BytesIO:
        return io.BytesIO(raw)

    return open_stream


def _write(
    tmp_path: Path,
    dataset: MirrorDataset,
    raw: bytes,
    *,
    block_bytes: int = MIRROR_ROW_GROUP_ROWS,
    row_group_rows: int = MIRROR_ROW_GROUP_ROWS,
) -> tuple[Path, SourceLayout]:
    layout = inspect_csv(_open(raw), dataset)
    path = tmp_path / "object.parquet"
    writer = MirrorObjectWriter(path, dataset, row_group_rows=row_group_rows)
    try:
        for table in encode_source(
            _open(raw), dataset, layout, symbol="BTCUSDT", source_index=0, first_block=0, block_bytes=block_bytes
        ):
            writer.append(table)
        writer.close()
    except BaseException:
        writer.abort()
        raise
    return path, layout


def _rebuild(path: Path, dataset: MirrorDataset, layout: SourceLayout, index: int = 0) -> bytes:
    return b"".join(iter_source_csv(path, dataset, index, layout))


def _column(path: Path, name: str) -> list[object]:
    return pq.read_table(path, columns=[name]).column(name).to_pylist()


@pytest.mark.parametrize("name", sorted(MIRROR_DATASET_SPECS))
def test_roundtrip_each_dataset(tmp_path: Path, name: str) -> None:
    dataset = MIRROR_DATASET_SPECS[name]
    raw = _with_header(dataset, _SAMPLES[name]).encode()
    path, layout = _write(tmp_path, dataset, raw)
    assert _rebuild(path, dataset, layout) == raw
    verify_object(path, dataset, {0: layout})


def test_headerless_and_no_trailing_newline(tmp_path: Path) -> None:
    dataset = MIRROR_DATASET_SPECS["aggTrades"]
    raw = _TRADE_ROWS.rstrip("\n").encode()
    path, layout = _write(tmp_path, dataset, raw)
    assert layout.header_line is None
    assert layout.trailing_newline is False
    assert _rebuild(path, dataset, layout) == raw


def test_awkward_cells_stay_exact(tmp_path: Path) -> None:
    dataset = MIRROR_DATASET_SPECS["metrics"]
    rows = (
        "2024-01-01 00:00:00,BTCUSDT,,0E-8,-1.4E-7,-0.00000000,007,0.1234567890123456\n"
        "2024-01-01 00:05:00,BTCUSDT,123456789012345678901,84231,1.5,0.00000001,2,0E-8\n"
    )
    raw = _with_header(dataset, rows).encode()
    path, layout = _write(tmp_path, dataset, raw)
    assert _rebuild(path, dataset, layout) == raw
    txt = _column(path, "sum_open_interest__txt")
    assert txt == ["", "123456789012345678901"]
    assert _column(path, "sum_open_interest_value__txt") == ["0E-8", None]
    assert _column(path, "count_toptrader_long_short_ratio__txt") == ["-1.4E-7", None]
    assert _column(path, "sum_toptrader_long_short_ratio__txt") == ["-0.00000000", None]
    assert _column(path, "count_long_short_ratio__txt") == ["007", None]
    assert _column(path, "sum_taker_long_short_vol_ratio__txt") == [None, "0E-8"]


def test_canonical_cells_use_ints(tmp_path: Path) -> None:
    dataset = MIRROR_DATASET_SPECS["klines"]
    raw = _with_header(dataset, _KLINE_ROWS).encode()
    path, _ = _write(tmp_path, dataset, raw)
    for column in dataset.columns:
        if column not in dataset.text_columns:
            assert set(_column(path, f"{column}__txt")) == {None}


def test_int_columns_are_exact_values(tmp_path: Path) -> None:
    dataset = MIRROR_DATASET_SPECS["klines"]
    raw = _with_header(dataset, _KLINE_ROWS).encode()
    path, _ = _write(tmp_path, dataset, raw)
    table = pq.read_table(path)
    source_rows = [line.split(",") for line in _KLINE_ROWS.splitlines()]
    for index, column in enumerate(dataset.columns):
        scale = table.column(f"{column}__scale").to_pylist()[0]
        values = table.column(column).to_pylist()
        for row, value in enumerate(values):
            assert Decimal(value).scaleb(-scale) == Decimal(source_rows[row][index])


def test_row_order_preserved(tmp_path: Path) -> None:
    dataset = MIRROR_DATASET_SPECS["metrics"]
    raw = _with_header(dataset, _METRICS_ROWS).encode()
    path, layout = _write(tmp_path, dataset, raw)
    assert _column(path, "create_time") == ["2024-01-01 00:05:00", "2024-01-01 00:00:00"]
    assert _rebuild(path, dataset, layout) == raw


def _agg_rows(count: int) -> str:
    rows = []
    for i in range(count):
        first = 200 + i * 7
        last = first + i % 4
        rows.append(
            f"{100 + i * 3},{42000 + i % 7}.{i % 10}0,{i % 9 + 1}.{i % 1000:03d},"
            f"{first},{last},{1704067200000 + i * 17},{'true' if i % 2 else 'false'}"
        )
    return "\n".join(rows) + "\n"


def test_residual_ids_across_blocks_and_row_groups(tmp_path: Path) -> None:
    dataset = MIRROR_DATASET_SPECS["aggTrades"]
    raw = _agg_rows(40).encode()
    path, layout = _write(tmp_path, dataset, raw, block_bytes=120, row_group_rows=2)
    assert _rebuild(path, dataset, layout) == raw
    assert len(set(_column(path, "__block"))) > 1
    verify_object(path, dataset, {0: layout})
    wrong = SourceLayout(layout.header_line, layout.trailing_newline, "0" * 64, layout.csv_bytes)
    with pytest.raises(DataIntegrityError):
        verify_object(path, dataset, {0: wrong})


def test_residual_rejects_noncanonical_id(tmp_path: Path) -> None:
    dataset = MIRROR_DATASET_SPECS["aggTrades"]
    raw = b"100.0,42000.10,1.5,200,201,1704067200000,true\n"
    with pytest.raises(UnencodableSourceError):
        _write(tmp_path, dataset, raw)


@pytest.mark.parametrize(
    ("name", "raw"),
    [
        ("metrics", b"create_time,symbol\r\n1,2\r\n"),
        ("fundingRate", b""),
        ("fundingRate", b"calc_time,funding_interval_hours,last_funding_rate\n"),
        ("fundingRate", b"calc_time,funding_interval_hours,last_funding_rate"),
        ("fundingRate", b"calc_time,funding_interval_hours,changed\n1,8,0.1\n"),
        ("fundingRate", b"calc_time,funding_interval_hours,last_funding_rate\n1,8\n"),
    ],
)
def test_unencodable_inputs(tmp_path: Path, name: str, raw: bytes) -> None:
    dataset = MIRROR_DATASET_SPECS[name]
    with pytest.raises(UnencodableSourceError):
        _write(tmp_path, dataset, raw)


def test_multi_source_pack(tmp_path: Path) -> None:
    dataset = MIRROR_DATASET_SPECS["klines"]
    raw_by_symbol = {
        "BTCUSDT": _with_header(dataset, _KLINE_ROWS).encode(),
        "ETHUSDT": _with_header(dataset, _KLINE_ROWS.replace("42000.10", "2200.5")).encode(),
    }
    path = tmp_path / "packed.parquet"
    writer = MirrorObjectWriter(path, dataset)
    layouts: dict[int, SourceLayout] = {}
    next_block = 0
    for index, (symbol, raw) in enumerate(raw_by_symbol.items()):
        layout = inspect_csv(_open(raw), dataset)
        layouts[index] = layout
        for table in encode_source(
            _open(raw), dataset, layout, symbol=symbol, source_index=index, first_block=next_block
        ):
            writer.append(table)
            next_block += 1
    writer.close()
    for index, (symbol, raw) in enumerate(raw_by_symbol.items()):
        assert _rebuild(path, dataset, layouts[index], index) == raw
        symbols = {
            value
            for value, source in zip(_column(path, "__symbol"), _column(path, "__source"), strict=True)
            if source == index
        }
        assert symbols == {symbol}
    verify_object(path, dataset, layouts)


def test_verify_detects_corruption(tmp_path: Path) -> None:
    dataset = MIRROR_DATASET_SPECS["fundingRate"]
    raw = _with_header(dataset, _FUNDING_ROWS).encode()
    path, layout = _write(tmp_path, dataset, raw)
    wrong = SourceLayout(layout.header_line, layout.trailing_newline, "0" * 64, layout.csv_bytes)
    with pytest.raises(DataIntegrityError):
        verify_object(path, dataset, {0: wrong})
    with pytest.raises(DataIntegrityError):
        verify_object(path, dataset, {0: layout, 1: layout})


def test_writer_atomic_and_abort(tmp_path: Path) -> None:
    dataset = MIRROR_DATASET_SPECS["fundingRate"]
    raw = _with_header(dataset, _FUNDING_ROWS).encode()
    layout = inspect_csv(_open(raw), dataset)
    tables = list(encode_source(_open(raw), dataset, layout, symbol="BTCUSDT", source_index=0, first_block=0))
    path = tmp_path / "object.parquet"

    writer = MirrorObjectWriter(path, dataset)
    writer.append(tables[0])
    assert not path.exists()
    writer.abort()
    assert not path.exists()
    assert not (tmp_path / "object.parquet.tmp").exists()
    writer.abort()

    empty = MirrorObjectWriter(path, dataset)
    with pytest.raises(ValueError, match="no rows"):
        empty.close()
    assert not path.exists()
    assert not (tmp_path / "object.parquet.tmp").exists()


def test_schema_is_stable(tmp_path: Path) -> None:
    dataset = MIRROR_DATASET_SPECS["metrics"]
    schema = mirror_schema(dataset)
    plain = list(
        encode_source(
            _open(_with_header(dataset, _METRICS_ROWS).encode()),
            dataset,
            inspect_csv(_open(_with_header(dataset, _METRICS_ROWS).encode()), dataset),
            symbol="BTCUSDT",
            source_index=0,
            first_block=0,
        )
    )
    raw_awkward = _with_header(dataset, "2024-01-01 00:00:00,BTCUSDT,,0E-8,1,2,3,4\n").encode()
    awkward = list(
        encode_source(
            _open(raw_awkward),
            dataset,
            inspect_csv(_open(raw_awkward), dataset),
            symbol="BTCUSDT",
            source_index=0,
            first_block=0,
        )
    )
    assert plain[0].schema.equals(schema)
    assert awkward[0].schema.equals(schema)
    assert decode_block(awkward[0], dataset) == b"2024-01-01 00:00:00,BTCUSDT,,0E-8,1,2,3,4"


def _first_table(dataset: MirrorDataset, raw: bytes) -> pa.Table:
    layout = inspect_csv(_open(raw), dataset)
    return next(encode_source(_open(raw), dataset, layout, symbol="BTCUSDT", source_index=0, first_block=0))


def test_header_not_ascii_is_unencodable(tmp_path: Path) -> None:
    dataset = MIRROR_DATASET_SPECS["fundingRate"]
    raw = b"calc_time,funding_interval_hours,last_funding_rate\xc3\xa9\n1,8,0.1\n"
    with pytest.raises(UnencodableSourceError):
        _write(tmp_path, dataset, raw)


def test_blank_lines_only_are_unencodable(tmp_path: Path) -> None:
    dataset = MIRROR_DATASET_SPECS["fundingRate"]
    with pytest.raises(UnencodableSourceError, match="no data rows"):
        _write(tmp_path, dataset, b"\n\n")


def test_blank_line_inside_source_breaks_checksum(tmp_path: Path) -> None:
    dataset = MIRROR_DATASET_SPECS["fundingRate"]
    raw = b"1704067200000,8,0.0001\n\n1704096000000,8,-0.00005\n"
    with pytest.raises(UnencodableSourceError, match="checksum"):
        _write(tmp_path, dataset, raw)


def test_block_that_fails_decode_is_unencodable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = MIRROR_DATASET_SPECS["fundingRate"]
    raw = _with_header(dataset, _FUNDING_ROWS).encode()
    layout = inspect_csv(_open(raw), dataset)
    monkeypatch.setattr(codec_module, "decode_block", lambda table, dataset: b"not the rows")
    with pytest.raises(UnencodableSourceError, match="decode to its source rows"):
        list(encode_source(_open(raw), dataset, layout, symbol="BTCUSDT", source_index=0, first_block=0))


def test_writer_rejects_foreign_schema_and_reuse(tmp_path: Path) -> None:
    dataset = MIRROR_DATASET_SPECS["fundingRate"]
    table = _first_table(dataset, _with_header(dataset, _FUNDING_ROWS).encode())
    writer = MirrorObjectWriter(tmp_path / "object.parquet", dataset)
    with pytest.raises(ValueError, match="schema"):
        writer.append(pa.table({"other": [1]}))
    writer.append(table)
    writer.close()
    with pytest.raises(RuntimeError):
        writer.append(table)
    with pytest.raises(RuntimeError):
        writer.close()


def test_decode_rejects_inconsistent_block_scale(tmp_path: Path) -> None:
    dataset = MIRROR_DATASET_SPECS["fundingRate"]
    table = _first_table(dataset, _with_header(dataset, _FUNDING_ROWS).encode())
    index = table.schema.get_field_index("last_funding_rate__scale")
    corrupt = table.set_column(index, "last_funding_rate__scale", pa.array([0, 1], pa.int8()))
    with pytest.raises(DataIntegrityError, match="scale"):
        decode_block(corrupt, dataset)


def test_decode_rejects_null_residual_gap() -> None:
    dataset = MIRROR_DATASET_SPECS["aggTrades"]
    table = _first_table(dataset, _TRADE_ROWS.encode())
    index = table.schema.get_field_index("first_trade_gap")
    corrupt = table.set_column(index, "first_trade_gap", pa.array([None, 1], pa.int64()))
    with pytest.raises(DataIntegrityError, match="nulls"):
        decode_block(corrupt, dataset)


def test_abort_after_flush_closes_open_writer(tmp_path: Path) -> None:
    dataset = MIRROR_DATASET_SPECS["fundingRate"]
    table = _first_table(dataset, _with_header(dataset, _FUNDING_ROWS).encode())
    writer = MirrorObjectWriter(tmp_path / "object.parquet", dataset, row_group_rows=1)
    writer.append(table)
    writer.abort()
    assert not (tmp_path / "object.parquet").exists()
    assert not (tmp_path / "object.parquet.tmp").exists()


def test_sources_read_across_row_groups_of_other_sources(tmp_path: Path) -> None:
    dataset = MIRROR_DATASET_SPECS["klines"]
    raw_by_index = [
        _with_header(dataset, _KLINE_ROWS).encode(),
        _with_header(dataset, _KLINE_ROWS.replace("42000.10", "2200.5")).encode(),
    ]
    path = tmp_path / "packed.parquet"
    writer = MirrorObjectWriter(path, dataset, row_group_rows=1)
    layouts: dict[int, SourceLayout] = {}
    next_block = 0
    for index, raw in enumerate(raw_by_index):
        layout = inspect_csv(_open(raw), dataset)
        layouts[index] = layout
        for table in encode_source(
            _open(raw), dataset, layout, symbol=f"S{index}", source_index=index, first_block=next_block
        ):
            writer.append(table)
            next_block += 1
    writer.close()
    for index, raw in enumerate(raw_by_index):
        assert _rebuild(path, dataset, layouts[index], index) == raw
    verify_object(path, dataset, layouts)


def test_verify_rejects_foreign_dataset_and_unexpected_source(tmp_path: Path) -> None:
    klines = MIRROR_DATASET_SPECS["klines"]
    raw = _with_header(klines, _KLINE_ROWS).encode()
    path, layout = _write(tmp_path, klines, raw)
    with pytest.raises(DataIntegrityError, match="metadata"):
        verify_object(path, MIRROR_DATASET_SPECS["markPriceKlines"], {0: layout})
    with pytest.raises(DataIntegrityError, match="unexpected"):
        verify_object(path, klines, {})


def test_decode_rejects_cell_without_integer_or_text() -> None:
    dataset = MIRROR_DATASET_SPECS["fundingRate"]
    table = _first_table(dataset, _with_header(dataset, _FUNDING_ROWS).encode())
    index = table.schema.get_field_index("last_funding_rate")
    lost = table.set_column(index, table.schema.field(index), pa.nulls(table.num_rows, pa.int64()))
    lost_txt = lost.schema.get_field_index("last_funding_rate__txt")
    lost = lost.set_column(lost_txt, lost.schema.field(lost_txt), pa.nulls(table.num_rows, pa.string()))
    with pytest.raises(DataIntegrityError):
        decode_block(lost, dataset)


def test_writer_rejects_table_of_same_shaped_dataset(tmp_path: Path) -> None:
    mark = MIRROR_DATASET_SPECS["markPriceKlines"]
    table = _first_table(mark, _with_header(mark, _KLINE_ROWS).encode())
    writer = MirrorObjectWriter(tmp_path / "object.parquet", MIRROR_DATASET_SPECS["klines"])
    with pytest.raises(ValueError, match="schema"):
        writer.append(table)
    writer.abort()


def test_abort_removes_tmp_when_close_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = MIRROR_DATASET_SPECS["fundingRate"]
    table = _first_table(dataset, _with_header(dataset, _FUNDING_ROWS).encode())
    path = tmp_path / "object.parquet"
    writer = MirrorObjectWriter(path, dataset, row_group_rows=1)
    writer.append(table)

    real_close = pq.ParquetWriter.close
    failures = [OSError("disk full")]

    def _fail_once(self: pq.ParquetWriter) -> None:
        real_close(self)
        if failures:
            raise failures.pop()

    monkeypatch.setattr(pq.ParquetWriter, "close", _fail_once)
    with pytest.raises(OSError, match="disk full"):
        writer.abort()
    assert list(tmp_path.iterdir()) == []
