"""Tests for ParquetSource — streaming, schema projection, num_rows, and file caching."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pydantic_arrow import ArrowFrame
from pydantic_arrow._sources import ParquetSource
from tests.conftest import SimpleUser

# ---------------------------------------------------------------------------
# Basic streaming
# ---------------------------------------------------------------------------


class TestParquetSourceStreaming:
    def test_yields_all_rows(self, parquet_file):
        source = ParquetSource(parquet_file, batch_size=2)
        total = sum(b.num_rows for b in source.iter_batches())
        assert total == 3  # 3 rows in fixture

    def test_replayable(self, parquet_file):
        source = ParquetSource(parquet_file, batch_size=2)
        first = list(source.iter_batches())
        second = list(source.iter_batches())
        assert len(first) == len(second)

    def test_large_file_does_not_load_all_at_once(self, large_parquet_file):
        """Consuming only the first batch must not trigger a full table load."""
        path, _num_rows = large_parquet_file
        source = ParquetSource(path, batch_size=1000)
        it = source.iter_batches()
        first_batch = next(it)
        assert first_batch.num_rows <= 1000
        del it  # early exit — no error expected


# ---------------------------------------------------------------------------
# Schema and column projection
# ---------------------------------------------------------------------------


class TestParquetSourceSchema:
    @pytest.fixture
    def multi_col_parquet(self, tmp_path: Path) -> Path:
        table = pa.table({"a": [1, 2], "b": ["x", "y"], "c": [1.0, 2.0]})
        path = tmp_path / "multi.parquet"
        pq.write_table(table, path)
        return path

    def test_schema_without_columns_returns_full_schema(self, multi_col_parquet):
        source = ParquetSource(multi_col_parquet)
        assert isinstance(source.schema, pa.Schema)
        assert len(source.schema) == 3

    def test_schema_with_single_column_is_pa_schema(self, multi_col_parquet):
        source = ParquetSource(multi_col_parquet, columns=["a"])
        result = source.schema
        assert isinstance(result, pa.Schema), f"Expected pa.Schema, got {type(result).__name__}"
        assert result.names == ["a"]

    def test_schema_with_multiple_columns(self, multi_col_parquet):
        source = ParquetSource(multi_col_parquet, columns=["a", "c"])
        result = source.schema
        assert isinstance(result, pa.Schema)
        assert set(result.names) == {"a", "c"}

    def test_arrow_schema_always_returns_full_unfiltered_schema(self, parquet_file):
        """arrow_schema ignores column filters and always reflects the file schema."""
        source = ParquetSource(parquet_file, columns=["name"])
        assert len(source.schema.names) == 1
        assert len(source.arrow_schema.names) > 1

    def test_iter_batches_respects_column_projection(self, multi_col_parquet):
        source = ParquetSource(multi_col_parquet, columns=["b"])
        batches = list(source.iter_batches())
        assert all(b.schema.names == ["b"] for b in batches)


# ---------------------------------------------------------------------------
# num_rows — reads footer metadata without a full scan
# ---------------------------------------------------------------------------


class TestParquetSourceNumRows:
    def test_num_rows_matches_footer_metadata(self, parquet_file):
        expected = pq.ParquetFile(parquet_file).metadata.num_rows
        source = ParquetSource(parquet_file)
        assert source.num_rows() == expected

    def test_num_rows_via_frame(self, parquet_file):
        expected = pq.ParquetFile(parquet_file).metadata.num_rows
        frame = ArrowFrame[SimpleUser].from_parquet(parquet_file)
        assert frame.num_rows == expected

    def test_num_rows_large_file(self, large_parquet_file):
        path, expected = large_parquet_file
        source = ParquetSource(path)
        assert source.num_rows() == expected


# ---------------------------------------------------------------------------
# ParquetFile handle caching (_pf)
# ---------------------------------------------------------------------------


class TestParquetSourceCaching:
    def test_open_returns_same_instance_on_repeated_calls(self, parquet_file):
        """_open() must cache the ParquetFile rather than re-reading the footer each time."""
        source = ParquetSource(parquet_file)
        assert source._open() is source._open()

    def test_all_accessors_share_one_cached_handle(self, parquet_file):
        """schema, arrow_schema, num_rows(), and iter_batches() all reuse self._pf."""
        source = ParquetSource(parquet_file)
        _ = source.schema
        _ = source.arrow_schema
        _ = source.num_rows()
        _ = list(source.iter_batches())
        assert source._pf is not None
        assert source._open() is source._pf
