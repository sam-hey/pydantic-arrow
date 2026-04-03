"""Tests for lazy data sources (_sources.py)."""

from __future__ import annotations

import pyarrow as pa
import pytest

from pydantic_arrow._sources import (
    BatchReaderSource,
    ConcatSource,
    ParquetSource,
    RowSource,
    TableSource,
)

SCHEMA = pa.schema([pa.field("x", pa.int64()), pa.field("y", pa.utf8())])
ROWS = [{"x": i, "y": f"val_{i}"} for i in range(10)]


# ---------------------------------------------------------------------------
# RowSource
# ---------------------------------------------------------------------------


class TestRowSource:
    def test_schema(self):
        source = RowSource(ROWS, SCHEMA)
        assert source.schema == SCHEMA

    def test_yields_batches(self):
        source = RowSource(ROWS, SCHEMA, batch_size=4)
        batches = list(source.iter_batches())
        assert len(batches) == 3  # 4 + 4 + 2
        assert all(isinstance(b, pa.RecordBatch) for b in batches)

    def test_total_rows(self):
        source = RowSource(ROWS, SCHEMA, batch_size=4)
        total = sum(b.num_rows for b in source.iter_batches())
        assert total == len(ROWS)

    def test_replayable(self):
        source = RowSource(ROWS, SCHEMA, batch_size=4)
        first = list(source.iter_batches())
        second = list(source.iter_batches())
        assert len(first) == len(second)

    def test_empty_rows(self):
        source = RowSource([], SCHEMA)
        assert list(source.iter_batches()) == []

    def test_batch_size_larger_than_data(self):
        source = RowSource(ROWS, SCHEMA, batch_size=1000)
        batches = list(source.iter_batches())
        assert len(batches) == 1
        assert batches[0].num_rows == len(ROWS)


# ---------------------------------------------------------------------------
# TableSource
# ---------------------------------------------------------------------------


class TestTableSource:
    @pytest.fixture
    def table(self):
        return pa.Table.from_pylist(ROWS, schema=SCHEMA)

    def test_schema(self, table):
        source = TableSource(table)
        assert source.schema == SCHEMA

    def test_yields_batches(self, table):
        source = TableSource(table, batch_size=3)
        batches = list(source.iter_batches())
        assert all(isinstance(b, pa.RecordBatch) for b in batches)
        assert sum(b.num_rows for b in batches) == len(ROWS)

    def test_replayable(self, table):
        source = TableSource(table, batch_size=3)
        first = list(source.iter_batches())
        second = list(source.iter_batches())
        assert len(first) == len(second)


# ---------------------------------------------------------------------------
# BatchReaderSource
# ---------------------------------------------------------------------------


class TestBatchReaderSource:
    @pytest.fixture
    def reader(self):
        batches = [
            pa.RecordBatch.from_pylist(ROWS[:5], schema=SCHEMA),
            pa.RecordBatch.from_pylist(ROWS[5:], schema=SCHEMA),
        ]
        return pa.RecordBatchReader.from_batches(SCHEMA, iter(batches))

    def test_schema(self, reader):
        source = BatchReaderSource(reader)
        assert source.schema == SCHEMA

    def test_yields_batches(self, reader):
        source = BatchReaderSource(reader)
        batches = list(source.iter_batches())
        total = sum(b.num_rows for b in batches)
        assert total == len(ROWS)

    def test_one_shot_exhaustion(self, reader):
        source = BatchReaderSource(reader)
        list(source.iter_batches())  # consume
        second = list(source.iter_batches())
        assert second == []  # exhausted, no error


# ---------------------------------------------------------------------------
# ParquetSource
# ---------------------------------------------------------------------------


class TestParquetSource:
    def test_iter_batches(self, parquet_file):
        source = ParquetSource(parquet_file, batch_size=2)
        batches = list(source.iter_batches())
        total = sum(b.num_rows for b in batches)
        assert total == 3  # 3 rows in fixture

    def test_replayable(self, parquet_file):
        source = ParquetSource(parquet_file, batch_size=2)
        first = list(source.iter_batches())
        second = list(source.iter_batches())
        assert len(first) == len(second)

    def test_large_file_lazy(self, large_parquet_file):
        path, _num_rows = large_parquet_file
        source = ParquetSource(path, batch_size=1000)
        # Consume only the first batch -- should not load everything
        it = source.iter_batches()
        first_batch = next(it)
        assert first_batch.num_rows <= 1000
        # Don't consume the rest -- verify no error on early exit
        del it

    def test_arrow_schema_returns_full_unfiltered_schema(self, parquet_file):
        """arrow_schema always returns the raw file schema, ignoring column filters."""
        source = ParquetSource(parquet_file, columns=["name"])
        # .schema filters to the selected columns; .arrow_schema returns everything
        assert len(source.schema.names) == 1
        assert len(source.arrow_schema.names) > 1


# ---------------------------------------------------------------------------
# ConcatSource
# ---------------------------------------------------------------------------


class TestConcatSource:
    @pytest.fixture
    def two_sources(self):
        s1 = RowSource(ROWS[:5], SCHEMA)
        s2 = RowSource(ROWS[5:], SCHEMA)
        return ConcatSource([s1, s2])

    def test_schema_from_first_source(self, two_sources):
        assert two_sources.schema == SCHEMA

    def test_yields_all_rows(self, two_sources):
        batches = list(two_sources.iter_batches())
        total = sum(b.num_rows for b in batches)
        assert total == len(ROWS)

    def test_empty_sources_raises(self):
        """ConcatSource with an empty list must raise ValueError immediately."""
        with pytest.raises(ValueError, match="at least one source"):
            ConcatSource([])
