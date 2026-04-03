"""Tests for BatchReaderSource."""

from __future__ import annotations

import pyarrow as pa
import pytest

from pydantic_arrow._sources import BatchReaderSource

from .conftest import ROWS, SCHEMA


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

    def test_yields_all_rows(self, reader):
        source = BatchReaderSource(reader)
        total = sum(b.num_rows for b in source.iter_batches())
        assert total == len(ROWS)

    def test_one_shot_exhaustion(self, reader):
        """A second iteration after exhaustion yields nothing without raising."""
        source = BatchReaderSource(reader)
        list(source.iter_batches())
        second = list(source.iter_batches())
        assert second == []
