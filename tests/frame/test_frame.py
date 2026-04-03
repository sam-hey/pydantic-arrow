"""Tests for ArrowFrame (_frame.py)."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from dirty_equals import IsInstance
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_arrow import ArrowFrame, model_to_schema, rows_to_batches
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

    def test_out_of_range_index_raises(self, simple_rows):
        frame = ArrowFrame[SimpleUser].from_rows(simple_rows)
        with pytest.raises(IndexError, match="out of range"):
            frame[999]


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

    def test_parquet_source_returns_row_count(self, parquet_file):
        frame = ArrowFrame[SimpleUser].from_parquet(parquet_file)
        assert frame.num_rows == 3  # parquet_file fixture has 3 rows

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


class TestAppend:
    """Tests for ArrowFrame.append()."""

    @pytest.mark.parametrize(
        "new_entry",
        [
            pytest.param(
                {"name": "Dave", "age": 28, "score": 8.0, "active": True},
                id="dict",
            ),
            pytest.param(
                SimpleUser(name="Dave", age=28, score=8.0, active=True),
                id="model",
            ),
        ],
    )
    def test_append_single_entry(self, simple_rows, new_entry):
        """A single dict or model instance can be appended."""
        frame = ArrowFrame[SimpleUser].from_rows(simple_rows)
        appended = frame.append(new_entry)
        collected = appended.collect()
        assert len(collected) == len(simple_rows) + 1
        assert collected[-1].name == "Dave"
        assert collected[-1].age == 28

    @pytest.mark.parametrize(
        "extra",
        [
            pytest.param(
                [
                    {"name": "Frank", "age": 40, "score": 6.0, "active": True},
                    {"name": "Grace", "age": 31, "score": 7.5, "active": False},
                ],
                id="dicts",
            ),
            pytest.param(
                [
                    SimpleUser(name="Frank", age=40, score=6.0, active=True),
                    SimpleUser(name="Grace", age=31, score=7.5, active=False),
                ],
                id="models",
            ),
        ],
    )
    def test_append_multiple_entries(self, simple_rows, extra):
        """A list of dicts or model instances can be appended at once."""
        frame = ArrowFrame[SimpleUser].from_rows(simple_rows)
        appended = frame.append(extra)
        collected = appended.collect()
        assert len(collected) == len(simple_rows) + 2
        assert collected[-2].name == "Frank"
        assert collected[-1].name == "Grace"

    def test_append_preserves_original(self, simple_rows):
        """append() must return a new frame; the original must be unchanged."""
        frame = ArrowFrame[SimpleUser].from_rows(simple_rows)
        _ = frame.append({"name": "X", "age": 1, "score": 0.0, "active": True})
        assert frame.num_rows == len(simple_rows)

    def test_append_to_parquet_frame(self, parquet_file):
        """Appending to a ParquetSource-backed frame must work lazily."""
        frame = ArrowFrame[SimpleUser].from_parquet(parquet_file)
        appended = frame.append({"name": "Zara", "age": 19, "score": 10.0, "active": True})
        assert appended.collect()[-1].name == "Zara"

    def test_append_schema_preserved(self, simple_rows):
        frame = ArrowFrame[SimpleUser].from_rows(simple_rows)
        appended = frame.append({"name": "H", "age": 5, "score": 1.0, "active": True})
        assert appended.schema == frame.schema

    def test_append_write_to_parquet(self, parquet_file, tmp_path):
        """Appended frame can be streamed to a new Parquet file."""
        frame = ArrowFrame[SimpleUser].from_parquet(parquet_file)
        original_count = sum(b.num_rows for b in frame.iter_batches())
        out = tmp_path / "appended.parquet"
        frame.append({"name": "Zara", "age": 19, "score": 10.0, "active": True}).to_parquet(out)
        assert pq.read_metadata(out).num_rows == original_count + 1


class TestExtend:
    """Tests for ArrowFrame.extend() -- in-place list-style extension."""

    @pytest.mark.parametrize(
        "extra",
        [
            pytest.param(
                [
                    {"name": "Dave", "age": 28, "score": 8.0, "active": True},
                    {"name": "Eve", "age": 22, "score": 9.9, "active": False},
                ],
                id="dicts",
            ),
            pytest.param(
                [
                    SimpleUser(name="Dave", age=28, score=8.0, active=True),
                    SimpleUser(name="Eve", age=22, score=9.9, active=False),
                ],
                id="models",
            ),
        ],
    )
    def test_extend_rows(self, simple_rows, extra):
        """extend() appends dicts or model instances in place."""
        frame = ArrowFrame[SimpleUser].from_rows(simple_rows)
        frame.extend(extra)
        collected = frame.collect()
        assert len(collected) == len(simple_rows) + 2
        assert collected[-2].name == "Dave"
        assert collected[-1].name == "Eve"

    def test_extend_modifies_in_place(self, simple_rows):
        """extend() must mutate the frame object itself, returning None."""
        frame = ArrowFrame[SimpleUser].from_rows(simple_rows)
        result = frame.extend([{"name": "X", "age": 1, "score": 0.0, "active": True}])
        assert result is None
        assert frame.num_rows == len(simple_rows) + 1

    def test_extend_multiple_times(self, simple_rows):
        frame = ArrowFrame[SimpleUser].from_rows(simple_rows)
        frame.extend([{"name": "A", "age": 1, "score": 1.0, "active": True}])
        frame.extend([{"name": "B", "age": 2, "score": 2.0, "active": False}])
        collected = frame.collect()
        assert len(collected) == len(simple_rows) + 2
        assert collected[-2].name == "A"
        assert collected[-1].name == "B"

    def test_extend_schema_unchanged(self, simple_rows):
        frame = ArrowFrame[SimpleUser].from_rows(simple_rows)
        original_schema = frame.schema
        frame.extend([{"name": "Z", "age": 9, "score": 0.5, "active": True}])
        assert frame.schema == original_schema


class TestConcatOperator:
    """Tests for ArrowFrame + and += operators."""

    @pytest.mark.parametrize(
        "rhs",
        [
            pytest.param(
                [
                    {"name": "H", "age": 10, "score": 1.1, "active": True},
                    {"name": "I", "age": 11, "score": 2.2, "active": False},
                ],
                id="list",
            ),
            pytest.param(
                ArrowFrame[SimpleUser].from_rows(
                    [
                        {"name": "H", "age": 10, "score": 1.1, "active": True},
                        {"name": "I", "age": 11, "score": 2.2, "active": False},
                    ]
                ),
                id="frame",
            ),
        ],
    )
    def test_add(self, simple_rows, rhs):
        """frame + list and frame + frame both produce a new concatenated frame."""
        frame = ArrowFrame[SimpleUser].from_rows(simple_rows)
        result = frame + rhs
        assert result is not frame
        collected = result.collect()
        assert len(collected) == len(simple_rows) + 2
        assert collected[-2].name == "H"
        assert collected[-1].name == "I"

    def test_add_preserves_original(self, simple_rows):
        """frame + rhs must not mutate the original frame."""
        frame = ArrowFrame[SimpleUser].from_rows(simple_rows)
        _ = frame + [{"name": "Dave", "age": 28, "score": 8.0, "active": True}]  # noqa: RUF005
        assert frame.num_rows == len(simple_rows)

    @pytest.mark.parametrize(
        "rhs",
        [
            pytest.param(
                [
                    {"name": "K", "age": 20, "score": 3.0, "active": True},
                    {"name": "L", "age": 21, "score": 4.0, "active": False},
                ],
                id="list",
            ),
            pytest.param(
                ArrowFrame[SimpleUser].from_rows(
                    [
                        {"name": "K", "age": 20, "score": 3.0, "active": True},
                        {"name": "L", "age": 21, "score": 4.0, "active": False},
                    ]
                ),
                id="frame",
            ),
        ],
    )
    def test_iadd(self, simple_rows, rhs):
        """frame += list and frame += frame both extend the frame in place."""
        frame = ArrowFrame[SimpleUser].from_rows(simple_rows)
        original_id = id(frame)
        frame += rhs
        assert id(frame) == original_id  # same object
        collected = frame.collect()
        assert len(collected) == len(simple_rows) + 2
        assert collected[-2].name == "K"
        assert collected[-1].name == "L"

    def test_chained_operations(self, simple_rows):
        """Combining extend, +, and += in sequence produces correct results."""
        frame = ArrowFrame[SimpleUser].from_rows(simple_rows)
        frame.extend([{"name": "N", "age": 1, "score": 1.0, "active": True}])
        frame2 = frame + [{"name": "O", "age": 2, "score": 2.0, "active": False}]  # noqa: RUF005
        frame2 += [{"name": "P", "age": 3, "score": 3.0, "active": True}]

        assert frame.num_rows == len(simple_rows) + 1  # extend only
        names = [u.name for u in frame2]
        assert names[-3] == "N"
        assert names[-2] == "O"
        assert names[-1] == "P"


class TestPlainEnumRoundtrip:
    """Regression tests for plain enum.Enum (non-str, non-int subclasses).

    Previously, model_dump() returned the raw enum *object* instead of its
    .value, causing Arrow to raise ArrowTypeError when building the batch.
    """

    def test_str_enum_roundtrip(self):
        import enum

        class Status(enum.Enum):
            ACTIVE = "active"
            INACTIVE = "inactive"

        class Doc(BaseModel):
            status: Status

        frame = ArrowFrame[Doc].from_rows([Doc(status=Status.ACTIVE)])
        restored = frame.collect()
        assert restored[0].status == Status.ACTIVE

    def test_int_enum_roundtrip(self):
        import enum

        class Priority(enum.IntEnum):
            LOW = 1
            HIGH = 3

        class Doc(BaseModel):
            priority: Priority

        frame = ArrowFrame[Doc].from_rows([Doc(priority=Priority.HIGH)])
        restored = frame.collect()
        assert restored[0].priority == Priority.HIGH

    def test_mixed_enum_model_roundtrip(self):
        """Full scenario from the bug report: schema, Arrow table, and restore."""
        import enum

        class Status(enum.Enum):
            ACTIVE = "active"
            INACTIVE = "inactive"

        class Priority(enum.IntEnum):
            LOW = 1
            HIGH = 3

        class M(BaseModel):
            status: Status
            priority: Priority
            opt: str | None = None

        frame = ArrowFrame[M].from_rows([M(status=Status.ACTIVE, priority=Priority.HIGH)])
        table = frame.to_arrow()
        assert table.num_rows == 1
        restored = ArrowFrame[M].from_arrow(table).collect()
        assert restored[0].status == Status.ACTIVE
        assert restored[0].priority == Priority.HIGH
        assert restored[0].opt is None

    def test_enum_via_append(self):
        """append() and extend() must also coerce plain enum values."""
        import enum

        class Color(enum.Enum):
            RED = "red"
            BLUE = "blue"

        class Paint(BaseModel):
            color: Color

        frame = ArrowFrame[Paint].from_rows([Paint(color=Color.RED)])
        frame.extend([Paint(color=Color.BLUE)])
        colors = [p.color for p in frame]
        assert colors == [Color.RED, Color.BLUE]


class TestRowsToBatches:
    """Tests for the public rows_to_batches() utility."""

    def test_single_batch_when_rows_fit(self):
        schema = pa.schema([pa.field("x", pa.int64())])
        rows = [{"x": i} for i in range(5)]
        batches = list(rows_to_batches(rows, schema, batch_size=100))
        assert len(batches) == 1
        assert batches[0].num_rows == 5

    def test_multiple_batches_when_chunked(self):
        schema = pa.schema([pa.field("x", pa.int64())])
        rows = [{"x": i} for i in range(10)]
        batches = list(rows_to_batches(rows, schema, batch_size=3))
        # 10 rows in chunks of 3 → 3+3+3+1 = 4 batches
        assert len(batches) == 4
        assert sum(b.num_rows for b in batches) == 10

    def test_empty_rows_yields_nothing(self):
        schema = pa.schema([pa.field("x", pa.int64())])
        batches = list(rows_to_batches([], schema))
        assert batches == []


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
