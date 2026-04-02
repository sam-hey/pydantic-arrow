"""Tests for ArrowFrame (_frame.py)."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from dirty_equals import IsInstance
from inline_snapshot import snapshot

from pydantic_arrow import ArrowFrame, model_to_schema
from tests.conftest import SimpleUser

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestFromRows:
    def test_basic(self, simple_rows):
        frame = ArrowFrame[SimpleUser].from_rows(simple_rows)
        assert frame == IsInstance(ArrowFrame)

    def test_repr(self, simple_rows):
        frame = ArrowFrame[SimpleUser].from_rows(simple_rows)
        assert repr(frame) == snapshot("ArrowFrame[SimpleUser](RowSource)")

    def test_schema_matches_model(self, simple_rows):
        frame = ArrowFrame[SimpleUser].from_rows(simple_rows)
        expected = model_to_schema(SimpleUser)
        assert frame.schema == expected

    def test_from_model_instances(self, simple_models):
        frame = ArrowFrame[SimpleUser].from_rows(simple_models)
        collected = frame.collect()
        assert len(collected) == len(simple_models)

    def test_empty_rows(self):
        frame = ArrowFrame[SimpleUser].from_rows([])
        assert frame.collect() == []

    def test_unparameterised_raises(self, simple_rows):
        with pytest.raises(TypeError, match="must be parameterised"):
            ArrowFrame.from_rows(simple_rows)


class TestFromParquet:
    def test_basic(self, parquet_file):
        frame = ArrowFrame[SimpleUser].from_parquet(parquet_file)
        assert frame == IsInstance(ArrowFrame)

    def test_collect_matches_original(self, simple_rows, parquet_file):
        frame = ArrowFrame[SimpleUser].from_parquet(parquet_file)
        collected = frame.collect()
        assert len(collected) == len(simple_rows)
        assert all(u == IsInstance(SimpleUser) for u in collected)

    def test_repr(self, parquet_file):
        frame = ArrowFrame[SimpleUser].from_parquet(parquet_file)
        assert repr(frame) == snapshot("ArrowFrame[SimpleUser](ParquetSource)")


class TestFromArrow:
    def test_from_table(self, simple_rows):
        schema = model_to_schema(SimpleUser)
        table = pa.Table.from_pylist(simple_rows, schema=schema)
        frame = ArrowFrame[SimpleUser].from_arrow(table)
        assert frame.collect() == IsInstance(list)

    def test_from_record_batch(self, simple_rows):
        schema = model_to_schema(SimpleUser)
        batch = pa.RecordBatch.from_pylist(simple_rows, schema=schema)
        frame = ArrowFrame[SimpleUser].from_arrow(batch)
        assert len(frame.collect()) == len(simple_rows)

    def test_from_record_batch_reader(self, simple_rows):
        schema = model_to_schema(SimpleUser)
        batch = pa.RecordBatch.from_pylist(simple_rows, schema=schema)
        reader = pa.RecordBatchReader.from_batches(schema, [batch])
        frame = ArrowFrame[SimpleUser].from_arrow(reader)
        assert len(frame.collect()) == len(simple_rows)


# ---------------------------------------------------------------------------
# Iteration
# ---------------------------------------------------------------------------


class TestIteration:
    def test_iter_yields_model_instances(self, simple_rows):
        frame = ArrowFrame[SimpleUser].from_rows(simple_rows)
        for user in frame:
            assert user == IsInstance(SimpleUser)

    def test_iter_count(self, simple_rows):
        frame = ArrowFrame[SimpleUser].from_rows(simple_rows)
        assert sum(1 for _ in frame) == len(simple_rows)

    def test_iter_batches_yields_record_batches(self, simple_rows):
        frame = ArrowFrame[SimpleUser].from_rows(simple_rows, batch_size=2)
        batches = list(frame.iter_batches())
        assert all(isinstance(b, pa.RecordBatch) for b in batches)
        assert sum(b.num_rows for b in batches) == len(simple_rows)

    def test_replayable_row_source(self, simple_rows):
        frame = ArrowFrame[SimpleUser].from_rows(simple_rows)
        first = frame.collect()
        second = frame.collect()
        assert first == second

    def test_lazy_not_loaded_until_iterated(self, simple_rows):
        """Creating a frame should not consume or load any data."""
        consumed = []

        def tracking_rows():
            for row in simple_rows:
                consumed.append(row)
                yield row

        # RowSource takes a list, not a generator -- use a generator-based
        # manual test via RecordBatchReader
        schema = model_to_schema(SimpleUser)

        def batch_gen():
            yield pa.RecordBatch.from_pylist(list(tracking_rows()), schema=schema)

        reader = pa.RecordBatchReader.from_batches(schema, batch_gen())
        frame = ArrowFrame[SimpleUser].from_arrow(reader)
        assert consumed == []  # not touched yet
        frame.collect()
        assert len(consumed) == len(simple_rows)


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


class TestIndexing:
    def test_single_index(self, simple_rows):
        frame = ArrowFrame[SimpleUser].from_rows(simple_rows)
        user = frame[0]
        assert user == IsInstance(SimpleUser)
        assert user.name == simple_rows[0]["name"]

    def test_negative_index_via_materialise(self, simple_rows):
        frame = ArrowFrame[SimpleUser].from_rows(simple_rows)
        frame.to_arrow()
        last = frame[len(simple_rows) - 1]
        assert last.name == simple_rows[-1]["name"]

    def test_slice_returns_frame(self, simple_rows):
        frame = ArrowFrame[SimpleUser].from_rows(simple_rows)
        sliced = frame[0:2]
        assert sliced == IsInstance(ArrowFrame)
        assert len(sliced.collect()) == 2

    def test_slice_with_step_raises(self, simple_rows):
        frame = ArrowFrame[SimpleUser].from_rows(simple_rows)
        with pytest.raises(ValueError, match="step != 1"):
            frame[::2]


# ---------------------------------------------------------------------------
# num_rows
# ---------------------------------------------------------------------------


class TestNumRows:
    def test_row_source(self, simple_rows):
        frame = ArrowFrame[SimpleUser].from_rows(simple_rows)
        assert frame.num_rows == len(simple_rows)

    def test_table_source(self, simple_rows):
        schema = model_to_schema(SimpleUser)
        table = pa.Table.from_pylist(simple_rows, schema=schema)
        frame = ArrowFrame[SimpleUser].from_arrow(table)
        assert frame.num_rows == len(simple_rows)

    def test_parquet_source_raises(self, parquet_file):
        frame = ArrowFrame[SimpleUser].from_parquet(parquet_file)
        with pytest.raises(TypeError, match="not available for streaming"):
            _ = frame.num_rows

    def test_len_matches_num_rows(self, simple_rows):
        frame = ArrowFrame[SimpleUser].from_rows(simple_rows)
        assert len(frame) == frame.num_rows


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


class TestExport:
    def test_to_arrow(self, simple_rows):
        frame = ArrowFrame[SimpleUser].from_rows(simple_rows)
        table = frame.to_arrow()
        assert isinstance(table, pa.Table)
        assert table.num_rows == len(simple_rows)

    def test_to_parquet_and_read_back(self, simple_rows, tmp_path):
        out = tmp_path / "out.parquet"
        frame = ArrowFrame[SimpleUser].from_rows(simple_rows)
        frame.to_parquet(out)
        assert out.exists()

        frame2 = ArrowFrame[SimpleUser].from_parquet(out)
        collected = frame2.collect()
        assert len(collected) == len(simple_rows)

    def test_to_parquet_streaming(self, large_parquet_file, tmp_path):
        """Write large file streaming without materialising in memory."""
        path, num_rows = large_parquet_file
        frame = ArrowFrame[SimpleUser].from_parquet(path, batch_size=500)
        out = tmp_path / "copy.parquet"
        frame.to_parquet(out)  # should not OOM on large files
        result = pq.read_table(out)
        assert result.num_rows == num_rows

    def test_collect_returns_model_list(self, simple_rows):
        frame = ArrowFrame[SimpleUser].from_rows(simple_rows)
        result = frame.collect()
        assert result == snapshot(
            [
                SimpleUser(name="Alice", age=30, score=9.5, active=True, email=None),
                SimpleUser(name="Bob", age=25, score=7.2, active=False, email="bob@test.com"),
                SimpleUser(name="Carol", age=35, score=8.8, active=True, email=None),
            ]
        )


# ---------------------------------------------------------------------------
# Batching behaviour
# ---------------------------------------------------------------------------


class TestBatching:
    def test_small_batch_size(self, simple_rows):
        frame = ArrowFrame[SimpleUser].from_rows(simple_rows, batch_size=1)
        batches = list(frame.iter_batches())
        assert len(batches) == len(simple_rows)
        assert all(b.num_rows == 1 for b in batches)

    def test_collect_same_regardless_of_batch_size(self, simple_rows):
        f1 = ArrowFrame[SimpleUser].from_rows(simple_rows, batch_size=1)
        f2 = ArrowFrame[SimpleUser].from_rows(simple_rows, batch_size=1000)
        assert f1.collect() == f2.collect()


# ---------------------------------------------------------------------------
# Nested model
# ---------------------------------------------------------------------------


class TestNestedModel:
    def test_nested_roundtrip(self):
        from tests.conftest import Address, PersonWithNested

        rows = [
            {"name": "Alice", "address": {"street": "Main St", "city": "Springfield", "zip_code": "12345"}},
            {"name": "Bob", "address": {"street": "Oak Ave", "city": "Shelbyville", "zip_code": "67890"}},
        ]
        frame = ArrowFrame[PersonWithNested].from_rows(rows)
        collected = frame.collect()
        assert collected[0].address == IsInstance(Address)
        assert collected[0].address.city == "Springfield"
