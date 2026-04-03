"""Tests for RowSource."""

from __future__ import annotations

import pyarrow as pa

from pydantic_arrow._sources import RowSource

from .conftest import ROWS, SCHEMA


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
