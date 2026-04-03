"""Tests for TableSource."""

from __future__ import annotations

import pyarrow as pa
import pytest

from pydantic_arrow._sources import TableSource

from .conftest import ROWS, SCHEMA


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
