"""Tests for extended sources: GeneratorSource, ParquetSource.schema fix, num_rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import BaseModel, ValidationError

from pydantic_arrow import ArrowFrame
from pydantic_arrow._sources import GeneratorSource, ParquetSource

SCHEMA = pa.schema([pa.field("x", pa.int64()), pa.field("y", pa.utf8())])


class XY(BaseModel):
    x: int
    y: str


# ---------------------------------------------------------------------------
# ParquetSource.schema bug fix
# ---------------------------------------------------------------------------


class TestParquetSourceSchemaWithColumns:
    """ParquetSource.schema must return pa.Schema, not pa.DataType, when columns= is set."""

    @pytest.fixture
    def multi_col_parquet(self, tmp_path: Path) -> Path:
        table = pa.table({"a": [1, 2], "b": ["x", "y"], "c": [1.0, 2.0]})
        path = tmp_path / "multi.parquet"
        pq.write_table(table, path)
        return path

    def test_schema_without_columns_is_pa_schema(self, multi_col_parquet):
        source = ParquetSource(multi_col_parquet)
        assert isinstance(source.schema, pa.Schema)
        assert len(source.schema) == 3

    def test_schema_with_single_column_is_pa_schema(self, multi_col_parquet):
        source = ParquetSource(multi_col_parquet, columns=["a"])
        result = source.schema
        assert isinstance(result, pa.Schema), f"Expected pa.Schema, got {type(result).__name__}: {result!r}"
        assert result.names == ["a"]

    def test_schema_with_multiple_columns_is_pa_schema(self, multi_col_parquet):
        source = ParquetSource(multi_col_parquet, columns=["a", "c"])
        result = source.schema
        assert isinstance(result, pa.Schema)
        assert set(result.names) == {"a", "c"}

    def test_iter_batches_respects_column_projection(self, multi_col_parquet):
        source = ParquetSource(multi_col_parquet, columns=["b"])
        batches = list(source.iter_batches())
        assert all(b.schema.names == ["b"] for b in batches)


# ---------------------------------------------------------------------------
# GeneratorSource
# ---------------------------------------------------------------------------


class TestGeneratorSource:
    def _make_rows(self, n: int = 5) -> list[dict[str, Any]]:
        return [{"x": i, "y": f"val_{i}"} for i in range(n)]

    def test_schema(self):
        source = GeneratorSource(iter(self._make_rows()), XY, SCHEMA)
        assert source.schema == SCHEMA

    def test_yields_correct_rows(self):
        rows = self._make_rows(10)
        source = GeneratorSource(iter(rows), XY, SCHEMA, batch_size=4)
        batches = list(source.iter_batches())
        total = sum(b.num_rows for b in batches)
        assert total == 10

    def test_chunking_into_batches(self):
        rows = self._make_rows(10)
        source = GeneratorSource(iter(rows), XY, SCHEMA, batch_size=3)
        batches = list(source.iter_batches())
        assert len(batches) == 4  # 3+3+3+1

    def test_one_shot_exhaustion(self):
        rows = self._make_rows(5)
        source = GeneratorSource(iter(rows), XY, SCHEMA)
        list(source.iter_batches())
        second = list(source.iter_batches())
        assert second == []

    def test_empty_iterable(self):
        source = GeneratorSource(iter([]), XY, SCHEMA)
        assert list(source.iter_batches()) == []

    def test_validation_error_surfaces_at_iteration(self):
        bad_rows = [{"x": "not-an-int", "y": "hello"}]
        source = GeneratorSource(iter(bad_rows), XY, SCHEMA)
        with pytest.raises(ValidationError):
            list(source.iter_batches())

    def test_validation_happens_inside_source(self):
        """model_validate is called by GeneratorSource, not by the caller."""
        rows = [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}]
        source = GeneratorSource(iter(rows), XY, SCHEMA)
        batches = list(source.iter_batches())
        data = batches[0].to_pylist()
        assert data[0]["x"] == 1
        assert data[1]["y"] == "b"


# ---------------------------------------------------------------------------
# ArrowFrame.from_iterable
# ---------------------------------------------------------------------------


class TestFromIterable:
    def test_basic_round_trip(self):
        rows = [{"x": i, "y": f"v{i}"} for i in range(5)]
        frame = ArrowFrame[XY].from_iterable(iter(rows))
        result = list(frame)
        assert len(result) == 5
        assert all(isinstance(r, XY) for r in result)
        assert result[0].x == 0
        assert result[4].y == "v4"

    def test_generator_expression(self):
        frame = ArrowFrame[XY].from_iterable({"x": i, "y": str(i)} for i in range(3))
        assert len(list(frame)) == 3

    def test_num_rows_raises_for_generator(self):
        frame = ArrowFrame[XY].from_iterable(iter([{"x": 1, "y": "a"}]))
        with pytest.raises(TypeError):
            _ = frame.num_rows

    def test_one_shot_not_replayable(self):
        rows = [{"x": 1, "y": "a"}]
        frame = ArrowFrame[XY].from_iterable(iter(rows))
        first = list(frame)
        second = list(frame)
        assert len(first) == 1
        assert second == []

    def test_validation_error_surfaces_lazily(self):
        bad = [{"x": "bad", "y": "ok"}]
        frame = ArrowFrame[XY].from_iterable(iter(bad))
        with pytest.raises(ValidationError):
            list(frame)


# ---------------------------------------------------------------------------
# num_rows for ParquetSource
# ---------------------------------------------------------------------------


class TestParquetSourceNumRows:
    def test_num_rows_from_metadata(self, parquet_file):
        """num_rows reads the Parquet footer metadata — no full scan needed."""
        expected = pq.ParquetFile(parquet_file).metadata.num_rows
        source = ParquetSource(parquet_file)
        assert source.num_rows() == expected

    def test_num_rows_via_frame(self, parquet_file):
        """ArrowFrame.num_rows works for Parquet-backed frames."""
        from tests.conftest import SimpleUser

        expected = pq.ParquetFile(parquet_file).metadata.num_rows
        frame = ArrowFrame[SimpleUser].from_parquet(parquet_file)
        assert frame.num_rows == expected

    def test_num_rows_large_parquet(self, large_parquet_file):
        path, expected = large_parquet_file
        source = ParquetSource(path)
        assert source.num_rows() == expected
