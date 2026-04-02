from __future__ import annotations

import enum
import uuid
from typing import Any

import pyarrow as pa
from pydantic import BaseModel

from pydantic_arrow import ArrowFrame, model_to_schema
from pydantic_arrow._convert import batch_to_models, models_to_batch

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


class Color(enum.Enum):
    """StrEnum: values are strings."""

    RED = "red"
    BLUE = "blue"
    GREEN = "green"


class Priority(enum.IntEnum):
    """IntEnum: values are ints — works today, kept for contrast."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3


class ModelWithStrEnum(BaseModel):
    name: str
    color: Color


class ModelWithIntEnum(BaseModel):
    name: str
    priority: Priority


class ModelWithUUID(BaseModel):
    id: uuid.UUID
    label: str


class ModelWithDict(BaseModel):
    tags: dict[str, int]


class ModelWithDictOfStr(BaseModel):
    labels: dict[str, str]


class ModelWithOptionalDict(BaseModel):
    metadata: dict[str, str] | None = None


class ModelWithDictOfList(BaseModel):
    groups: dict[str, list[int]]


class TestFromRowsStrEnum:
    def test_single_instance_round_trips(self):
        """ArrowFrame[M].from_rows([m]).collect() should return [m]."""
        m = ModelWithStrEnum(name="Alice", color=Color.RED)
        frame = ArrowFrame[ModelWithStrEnum].from_rows([m])
        result = frame.collect()
        assert result == [m]

    def test_multiple_instances_round_trip(self):
        instances = [
            ModelWithStrEnum(name="Alice", color=Color.RED),
            ModelWithStrEnum(name="Bob", color=Color.BLUE),
            ModelWithStrEnum(name="Carol", color=Color.GREEN),
        ]
        frame = ArrowFrame[ModelWithStrEnum].from_rows(instances)
        assert frame.collect() == instances

    def test_to_arrow_then_collect(self):
        """from_rows().to_arrow() should not raise."""
        m = ModelWithStrEnum(name="Alice", color=Color.RED)
        table = ArrowFrame[ModelWithStrEnum].from_rows([m]).to_arrow()
        assert isinstance(table, pa.Table)
        assert table.num_rows == 1

    def test_schema_uses_dictionary_type(self):
        """StrEnum → dictionary<int32, utf8> schema should be generated."""
        schema = model_to_schema(ModelWithStrEnum)
        color_field = schema.field("color")
        assert pa.types.is_dictionary(color_field.type), f"Expected dictionary type for StrEnum, got {color_field.type}"

    def test_dict_of_rows_also_works(self):
        """from_rows with plain dicts should still work as a control case."""
        rows = [{"name": "Alice", "color": "red"}]
        frame = ArrowFrame[ModelWithStrEnum].from_rows(rows)
        result = frame.collect()
        assert result[0].color == Color.RED


class TestFromRowsUUID:
    def test_single_uuid_round_trips(self):
        """UUID field: from_rows([m]).collect() should return [m]."""
        m = ModelWithUUID(id=uuid.UUID("12345678-1234-5678-1234-567812345678"), label="x")
        result = ArrowFrame[ModelWithUUID].from_rows([m]).collect()
        assert result == [m]

    def test_uuid_stored_as_string_in_arrow(self):
        """UUID should be stored as utf8 in Arrow (canonical hex string)."""
        m = ModelWithUUID(id=uuid.UUID("12345678-1234-5678-1234-567812345678"), label="x")
        table = ArrowFrame[ModelWithUUID].from_rows([m]).to_arrow()
        assert pa.types.is_string(table.schema.field("id").type)
        assert table["id"][0].as_py() == "12345678-1234-5678-1234-567812345678"

    def test_multiple_uuids_round_trip(self):
        instances = [ModelWithUUID(id=uuid.uuid4(), label=str(i)) for i in range(5)]
        result = ArrowFrame[ModelWithUUID].from_rows(instances).collect()
        assert result == instances

    def test_parquet_round_trip(self, tmp_path):
        """UUID survives a Parquet write/read cycle via from_rows."""
        m = ModelWithUUID(id=uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"), label="y")
        path = tmp_path / "uuids.parquet"
        ArrowFrame[ModelWithUUID].from_rows([m]).to_parquet(path)
        result = ArrowFrame[ModelWithUUID].from_parquet(path).collect()
        assert result == [m]


# ---------------------------------------------------------------------------
# Control: IntEnum works today (regression guard)
# ---------------------------------------------------------------------------


class TestFromRowsIntEnum:
    """IntEnum already works — these tests must remain green after the fix."""

    def test_int_enum_round_trips(self):
        m = ModelWithIntEnum(name="Dev", priority=Priority.HIGH)
        result = ArrowFrame[ModelWithIntEnum].from_rows([m]).collect()
        assert result == [m]

    def test_int_enum_stored_as_int64(self):
        schema = model_to_schema(ModelWithIntEnum)
        assert schema.field("priority").type == pa.int64()


class TestBatchToModelsDict:
    def test_dict_str_int_round_trips(self):
        """dict[str, int] field should survive batch_to_models."""
        m = ModelWithDict(tags={"a": 1, "b": 2, "c": 3})
        schema = model_to_schema(ModelWithDict)
        batch = models_to_batch([m], schema)
        result = list(batch_to_models(batch, ModelWithDict))
        assert result == [m]

    def test_dict_str_str_round_trips(self):
        m = ModelWithDictOfStr(labels={"env": "prod", "region": "eu-central"})
        schema = model_to_schema(ModelWithDictOfStr)
        batch = models_to_batch([m], schema)
        result = list(batch_to_models(batch, ModelWithDictOfStr))
        assert result == [m]

    def test_empty_dict_round_trips(self):
        m = ModelWithDict(tags={})
        schema = model_to_schema(ModelWithDict)
        batch = models_to_batch([m], schema)
        result = list(batch_to_models(batch, ModelWithDict))
        assert result == [m]

    def test_optional_dict_none_round_trips(self):
        m = ModelWithOptionalDict(metadata=None)
        schema = model_to_schema(ModelWithOptionalDict)
        batch = models_to_batch([m], schema)
        result = list(batch_to_models(batch, ModelWithOptionalDict))
        assert result == [m]

    def test_optional_dict_present_round_trips(self):
        m = ModelWithOptionalDict(metadata={"key": "value"})
        schema = model_to_schema(ModelWithOptionalDict)
        batch = models_to_batch([m], schema)
        result = list(batch_to_models(batch, ModelWithOptionalDict))
        assert result == [m]

    def test_multiple_dict_rows_round_trip(self):
        instances = [
            ModelWithDict(tags={"x": 10}),
            ModelWithDict(tags={"y": 20, "z": 30}),
        ]
        schema = model_to_schema(ModelWithDict)
        batch = models_to_batch(instances, schema)
        result = list(batch_to_models(batch, ModelWithDict))
        assert result == instances


# ---------------------------------------------------------------------------
# BUG-2 via full ArrowFrame pipeline
# ---------------------------------------------------------------------------


class TestArrowFrameDictRoundTrip:
    """End-to-end round-trips that combine BUG-1 and BUG-2 fixes."""

    def test_frame_from_rows_dict_collect(self):
        """from_rows → to_arrow → from_arrow → collect for dict field."""
        m = ModelWithDict(tags={"a": 1, "b": 2})
        schema = model_to_schema(ModelWithDict)
        batch = models_to_batch([m], schema)
        table = pa.Table.from_batches([batch], schema=schema)
        frame = ArrowFrame[ModelWithDict].from_arrow(table)
        result = frame.collect()
        assert result == [m]

    def test_frame_parquet_dict_round_trip(self, tmp_path):
        """dict[str,int] survives Parquet write/read via ArrowFrame."""
        instances = [
            ModelWithDict(tags={"alpha": 1}),
            ModelWithDict(tags={"beta": 2, "gamma": 3}),
        ]
        schema = model_to_schema(ModelWithDict)
        batch = models_to_batch(instances, schema)
        table = pa.Table.from_batches([batch], schema=schema)
        path = tmp_path / "dicts.parquet"
        ArrowFrame[ModelWithDict].from_arrow(table).to_parquet(path)
        result = ArrowFrame[ModelWithDict].from_parquet(path).collect()
        assert result == instances


# ---------------------------------------------------------------------------
# Combined: model with both StrEnum and dict — tests full fix integration
# ---------------------------------------------------------------------------


class ComplexModel(BaseModel):
    name: str
    color: Color
    scores: dict[str, int]
    uid: uuid.UUID


class TestComplexModelFullRoundTrip:
    def test_from_rows_collect(self):
        """Model with StrEnum + UUID + dict[str,int] should round-trip via from_rows."""
        m = ComplexModel(
            name="Alice",
            color=Color.RED,
            scores={"math": 95, "art": 80},
            uid=uuid.UUID("12345678-1234-5678-1234-567812345678"),
        )
        result = ArrowFrame[ComplexModel].from_rows([m]).collect()
        assert result == [m]

    def test_models_to_batch_then_batch_to_models(self):
        """models_to_batch + batch_to_models should round-trip ComplexModel."""
        instances = [
            ComplexModel(
                name="Alice",
                color=Color.RED,
                scores={"math": 95},
                uid=uuid.UUID("11111111-1111-1111-1111-111111111111"),
            ),
            ComplexModel(
                name="Bob",
                color=Color.BLUE,
                scores={"science": 88, "history": 72},
                uid=uuid.UUID("22222222-2222-2222-2222-222222222222"),
            ),
        ]
        schema = model_to_schema(ComplexModel)
        batch = models_to_batch(instances, schema)
        result = list(batch_to_models(batch, ComplexModel))
        assert result == instances


class TestDiagnostics:
    def test_str_enum_model_dump_returns_object(self):
        """model_dump() returns enum objects (Python mode) — this is the bug source."""
        m = ModelWithStrEnum(name="x", color=Color.RED)
        dumped = m.model_dump()
        # In Python mode, Pydantic preserves the enum object
        assert isinstance(dumped["color"], Color), (
            "After fix: model_dump() still returns Color here — fix must be in from_rows, not model_dump"
        )

    def test_str_enum_model_dump_json_returns_str(self):
        """model_dump(mode='json') returns the string value — the correct form for Arrow."""
        m = ModelWithStrEnum(name="x", color=Color.RED)
        dumped = m.model_dump(mode="json")
        assert dumped["color"] == "red"  # this is what Arrow expects

    def test_uuid_model_dump_returns_object(self):
        """model_dump() returns UUID object — this is the bug source for UUIDs."""
        uid = uuid.UUID("12345678-1234-5678-1234-567812345678")
        m = ModelWithUUID(id=uid, label="x")
        dumped = m.model_dump()
        assert isinstance(dumped["id"], uuid.UUID), (
            "After fix: model_dump() still returns UUID here — fix must be in from_rows, not model_dump"
        )

    def test_map_column_to_pylist_returns_tuples(self):
        """Arrow map<> column returns list-of-tuples from to_pylist() — the bug source."""
        schema = model_to_schema(ModelWithDict)
        batch = models_to_batch([ModelWithDict(tags={"a": 1})], schema)
        row = batch.to_pylist()[0]
        # Currently a list of tuples — should become a dict after the fix
        assert isinstance(row["tags"], list), "map columns give list-of-tuples from to_pylist()"
        assert row["tags"] == [("a", 1)]

    def test_map_column_fix_is_dict_conversion(self):
        """Manually applying dict() to the list-of-tuples produces correct input for Pydantic."""
        schema = model_to_schema(ModelWithDict)
        batch = models_to_batch([ModelWithDict(tags={"a": 1, "b": 2})], schema)
        row = batch.to_pylist()[0]
        # The fix: convert list-of-tuples → dict
        fixed: dict[str, Any] = {**row, "tags": dict(row["tags"])}
        m = ModelWithDict.model_validate(fixed)
        assert m.tags == {"a": 1, "b": 2}
