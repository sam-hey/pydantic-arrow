"""Tests for type coercion between Pydantic models and Apache Arrow."""

from __future__ import annotations

import uuid
from typing import Any

import pyarrow as pa
import pytest
from pydantic import ValidationError

from pydantic_arrow import ArrowFrame, model_to_schema
from pydantic_arrow._convert import batch_to_models, models_to_batch
from tests.conftest import (
    Color,
    ComplexModel,
    ModelWithDict,
    ModelWithDictOfList,
    ModelWithDictOfStr,
    ModelWithIntEnum,
    ModelWithOptionalDict,
    ModelWithStrEnum,
    ModelWithUUID,
    ReaderMissingRequiredField,
    ReaderOptionalExtra,
    ReaderSubset,
    ReaderTotallyDifferent,
    ReaderWrongType,
    Severity,
)

# ---------------------------------------------------------------------------
# Parametrize helpers
# ---------------------------------------------------------------------------

_FIXED_UUID = uuid.UUID("12345678-1234-5678-1234-567812345678")
_FIVE_UUIDS = [uuid.UUID(int=i + 1) for i in range(5)]


def _round_trip_params():
    """(id, list-of-instances) pairs covering every coercion-sensitive model."""
    return [
        pytest.param(
            [ModelWithStrEnum(name="Alice", color=Color.RED)],
            id="str-enum-single",
        ),
        pytest.param(
            [
                ModelWithStrEnum(name="Alice", color=Color.RED),
                ModelWithStrEnum(name="Bob", color=Color.BLUE),
                ModelWithStrEnum(name="Carol", color=Color.GREEN),
            ],
            id="str-enum-multiple",
        ),
        pytest.param(
            [ModelWithIntEnum(name="Dev", severity=Severity.HIGH)],
            id="int-enum",
        ),
        pytest.param(
            [ModelWithUUID(id=_FIXED_UUID, label="x")],
            id="uuid-single",
        ),
        pytest.param(
            [ModelWithUUID(id=uid, label=str(i)) for i, uid in enumerate(_FIVE_UUIDS)],
            id="uuid-multiple",
        ),
        pytest.param(
            [ModelWithDict(tags={"a": 1, "b": 2})],
            id="dict-str-int",
        ),
        pytest.param(
            [ModelWithDictOfStr(labels={"env": "prod", "region": "eu-west"})],
            id="dict-str-str",
        ),
        pytest.param(
            [ModelWithDictOfList(groups={"odds": [1, 3, 5], "evens": [2, 4]})],
            id="dict-of-list",
        ),
        pytest.param(
            [
                ComplexModel(
                    name="Alice",
                    color=Color.RED,
                    scores={"math": 95, "art": 80},
                    uid=uuid.UUID("11111111-1111-1111-1111-111111111111"),
                ),
                ComplexModel(
                    name="Bob",
                    color=Color.BLUE,
                    scores={"science": 88},
                    uid=uuid.UUID("22222222-2222-2222-2222-222222222222"),
                ),
            ],
            id="complex",
        ),
    ]


# ---------------------------------------------------------------------------
# TestRoundTrip — from_rows / to_parquet / from_parquet
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Round-trip every coercion-sensitive model through all major code paths."""

    @pytest.mark.parametrize("instances", _round_trip_params())
    def test_from_rows_collect(self, instances):
        """from_rows([…]).collect() returns the original instances."""
        Model = type(instances[0])
        assert ArrowFrame[Model].from_rows(instances).collect() == instances

    @pytest.mark.parametrize("instances", _round_trip_params())
    def test_to_arrow_then_from_arrow(self, instances):
        """from_rows → to_arrow → from_arrow → collect preserves every row."""
        Model = type(instances[0])
        table = ArrowFrame[Model].from_rows(instances).to_arrow()
        assert isinstance(table, pa.Table)
        assert table.num_rows == len(instances)
        assert ArrowFrame[Model].from_arrow(table).collect() == instances

    @pytest.mark.parametrize("instances", _round_trip_params())
    def test_parquet_round_trip(self, tmp_path, instances):
        """from_rows → to_parquet → from_parquet → collect survives disk."""
        Model = type(instances[0])
        path = tmp_path / "round_trip.parquet"
        ArrowFrame[Model].from_rows(instances).to_parquet(path)
        assert ArrowFrame[Model].from_parquet(path).collect() == instances


# ---------------------------------------------------------------------------
# TestSchemaTypes — verify Arrow column types for coercion-sensitive fields
# ---------------------------------------------------------------------------


class TestSchemaTypes:
    """Arrow schema must map each field to the correct column type."""

    @pytest.mark.parametrize(
        "Model,field_name,type_check",
        [
            pytest.param(
                ModelWithStrEnum,
                "color",
                pa.types.is_dictionary,
                id="str-enum-dictionary",
            ),
            pytest.param(
                ModelWithIntEnum,
                "severity",
                lambda t: t == pa.int64(),
                id="int-enum-int64",
            ),
            pytest.param(
                ModelWithUUID,
                "id",
                pa.types.is_string,
                id="uuid-string",
            ),
            pytest.param(
                ModelWithDict,
                "tags",
                pa.types.is_map,
                id="dict-map",
            ),
            pytest.param(
                ModelWithDictOfList,
                "groups",
                pa.types.is_map,
                id="dict-of-list-map",
            ),
        ],
    )
    def test_field_arrow_type(self, Model, field_name, type_check):
        schema = model_to_schema(Model)
        assert type_check(schema.field(field_name).type), (
            f"{Model.__name__}.{field_name}: unexpected type {schema.field(field_name).type}"
        )


# ---------------------------------------------------------------------------
# TestDictMapCoercion — batch-level coercion for map<K,V> columns
# ---------------------------------------------------------------------------


class TestDictMapCoercion:
    """Arrow map<K,V> must survive the models_to_batch → batch_to_models round-trip."""

    @pytest.mark.parametrize(
        "instance",
        [
            pytest.param(ModelWithDict(tags={"a": 1, "b": 2, "c": 3}), id="str-int"),
            pytest.param(ModelWithDictOfStr(labels={"env": "prod", "region": "eu"}), id="str-str"),
            pytest.param(ModelWithDict(tags={}), id="empty"),
            pytest.param(ModelWithOptionalDict(metadata=None), id="optional-none"),
            pytest.param(ModelWithOptionalDict(metadata={"key": "value"}), id="optional-present"),
            pytest.param(
                ModelWithDictOfList(groups={"a": [1, 2], "b": [3]}),
                id="dict-of-list",
            ),
        ],
    )
    def test_batch_round_trip(self, instance):
        """models_to_batch + batch_to_models returns the original instance."""
        Model = type(instance)
        schema = model_to_schema(Model)
        batch = models_to_batch([instance], schema)
        assert list(batch_to_models(batch, Model)) == [instance]

    def test_multiple_rows_batch_round_trip(self):
        """Multiple rows with varying dict sizes all survive the batch cycle."""
        instances = [
            ModelWithDict(tags={"x": 10}),
            ModelWithDict(tags={"y": 20, "z": 30}),
        ]
        schema = model_to_schema(ModelWithDict)
        batch = models_to_batch(instances, schema)
        assert list(batch_to_models(batch, ModelWithDict)) == instances

    def test_map_column_raw_format(self):
        """Document: Arrow map<> deserialises as list-of-tuples from to_pylist()."""
        schema = model_to_schema(ModelWithDict)
        batch = models_to_batch([ModelWithDict(tags={"a": 1})], schema)
        row = batch.to_pylist()[0]
        assert isinstance(row["tags"], list)
        assert row["tags"] == [("a", 1)]

    def test_tuple_to_dict_conversion(self):
        """Document: dict() applied to the list-of-tuples gives Pydantic-valid input."""
        schema = model_to_schema(ModelWithDict)
        batch = models_to_batch([ModelWithDict(tags={"a": 1, "b": 2})], schema)
        row: dict[str, Any] = batch.to_pylist()[0]
        fixed = {**row, "tags": dict(row["tags"])}
        assert ModelWithDict.model_validate(fixed).tags == {"a": 1, "b": 2}


# ---------------------------------------------------------------------------
# TestParquetSchemaMismatch — reading a file with a different model
# ---------------------------------------------------------------------------


class TestParquetSchemaMismatch:
    """Behaviour when reading a Parquet file with a mismatched Pydantic model.

    Compatible readers succeed; incompatible readers raise ``ValidationError``.
    """

    @pytest.mark.parametrize(
        "ReaderModel,check",
        [
            pytest.param(
                ReaderSubset,
                lambda rows: [r.name for r in rows] == ["Alice", "Bob"],
                id="subset",
            ),
            pytest.param(
                ReaderOptionalExtra,
                lambda rows: all(r.country is None for r in rows),
                id="optional-extra",
            ),
        ],
    )
    def test_compatible_reader(self, writer_parquet, ReaderModel, check):
        """Compatible readers return correct data without error."""
        result = ArrowFrame[ReaderModel].from_parquet(writer_parquet).collect()
        assert len(result) == 2
        assert check(result)

    @pytest.mark.parametrize(
        "ReaderModel,match",
        [
            pytest.param(ReaderMissingRequiredField, "country", id="missing-required"),
            pytest.param(ReaderWrongType, "age", id="type-mismatch"),
            pytest.param(ReaderTotallyDifferent, "product", id="totally-different"),
        ],
    )
    def test_incompatible_reader_raises(self, writer_parquet, ReaderModel, match):
        """Incompatible readers raise ValidationError naming the problem field."""
        with pytest.raises(ValidationError, match=match):
            ArrowFrame[ReaderModel].from_parquet(writer_parquet).collect()

    def test_explicit_column_projection(self, writer_parquet):
        """columns= projection lets a subset reader safely ignore extra file columns."""
        result = ArrowFrame[ReaderSubset].from_parquet(writer_parquet, columns=["name"]).collect()
        assert [r.name for r in result] == ["Alice", "Bob"]


# ---------------------------------------------------------------------------
# TestDiagnostics — document Arrow / Pydantic internals (no assertions that
# the lib should change; these explain the root causes of BUG-1 and BUG-2)
# ---------------------------------------------------------------------------


class TestDiagnostics:
    """Non-regression tests that pin the observed behaviour of underlying libraries.

    They explain *why* the coercion fixes are needed and ensure the assumptions
    they rely on have not changed in a new library version.
    """

    @pytest.mark.parametrize(
        "instance,field,expected_type",
        [
            pytest.param(
                ModelWithStrEnum(name="x", color=Color.RED),
                "color",
                Color,
                id="str-enum",
            ),
            pytest.param(
                ModelWithUUID(id=_FIXED_UUID, label="x"),
                "id",
                uuid.UUID,
                id="uuid",
            ),
        ],
    )
    def test_model_dump_python_preserves_rich_type(self, instance, field, expected_type):
        """model_dump() (Python mode) returns the rich object, not a primitive.

        This is expected Pydantic behaviour; our fix is in _frame.py/_convert.py,
        NOT in how model_dump() is called.
        """
        dumped = instance.model_dump()
        assert isinstance(dumped[field], expected_type)

    @pytest.mark.parametrize(
        "instance,field,expected_value",
        [
            pytest.param(
                ModelWithStrEnum(name="x", color=Color.RED),
                "color",
                "red",
                id="str-enum",
            ),
            pytest.param(
                ModelWithUUID(id=_FIXED_UUID, label="x"),
                "id",
                "12345678-1234-5678-1234-567812345678",
                id="uuid",
            ),
        ],
    )
    def test_model_dump_json_serializes_to_primitive(self, instance, field, expected_value):
        """model_dump(mode='json') returns primitives — the correct form for Arrow."""
        assert instance.model_dump(mode="json")[field] == expected_value
