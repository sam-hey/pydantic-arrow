"""Tests for ConcatSource."""

from __future__ import annotations

import pytest

from pydantic_arrow._sources import ConcatSource, RowSource

from .conftest import ROWS, SCHEMA


class TestConcatSource:
    @pytest.fixture
    def two_sources(self):
        return ConcatSource([RowSource(ROWS[:5], SCHEMA), RowSource(ROWS[5:], SCHEMA)])

    def test_schema_taken_from_first_source(self, two_sources):
        assert two_sources.schema == SCHEMA

    def test_yields_all_rows_from_all_sources(self, two_sources):
        total = sum(b.num_rows for b in two_sources.iter_batches())
        assert total == len(ROWS)

    def test_empty_sources_list_raises(self):
        """Constructing ConcatSource with no sources must fail immediately."""
        with pytest.raises(ValueError, match="at least one source"):
            ConcatSource([])
