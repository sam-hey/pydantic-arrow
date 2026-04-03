"""Tests for extended type mappings: timedelta, Annotated, frozenset."""

from __future__ import annotations

from datetime import timedelta
from typing import Annotated, get_origin

import pyarrow as pa
from pydantic import BaseModel
from pydantic.functional_validators import BeforeValidator

from pydantic_arrow import ArrowFrame
from pydantic_arrow._schema import model_to_schema
from pydantic_arrow._types import python_type_to_arrow

# ---------------------------------------------------------------------------
# timedelta → pa.duration("us")
# ---------------------------------------------------------------------------


def test_timedelta():
    dtype, nullable = python_type_to_arrow(timedelta)
    assert dtype == pa.duration("us")
    assert nullable is False


def test_optional_timedelta():
    dtype, nullable = python_type_to_arrow(timedelta | None)
    assert dtype == pa.duration("us")
    assert nullable is True


def test_list_of_timedelta():
    dtype, nullable = python_type_to_arrow(list[timedelta])
    assert pa.types.is_list(dtype)
    assert dtype.value_type == pa.duration("us")
    assert nullable is False


class ModelWithTimedelta(BaseModel):
    name: str
    elapsed: timedelta
    lag: timedelta | None = None


def test_timedelta_schema():
    schema = model_to_schema(ModelWithTimedelta)
    assert schema.field("elapsed").type == pa.duration("us")
    assert schema.field("lag").type == pa.duration("us")
    assert schema.field("lag").nullable is True


def test_timedelta_round_trip():
    rows = [
        {"name": "a", "elapsed": timedelta(seconds=10)},
        {"name": "b", "elapsed": timedelta(minutes=2), "lag": timedelta(milliseconds=500)},
    ]
    frame = ArrowFrame[ModelWithTimedelta].from_rows(rows)
    result = list(frame)
    assert result[0].elapsed == timedelta(seconds=10)
    assert result[1].lag == timedelta(milliseconds=500)


# ---------------------------------------------------------------------------
# Annotated[T, pa.DataType()] → pins Arrow width
# ---------------------------------------------------------------------------


def test_annotated_int32():
    dtype, _ = python_type_to_arrow(Annotated[int, pa.int32()])
    assert dtype == pa.int32()


def test_annotated_int16():
    dtype, _ = python_type_to_arrow(Annotated[int, pa.int16()])
    assert dtype == pa.int16()


def test_annotated_float32():
    dtype, _ = python_type_to_arrow(Annotated[float, pa.float32()])
    assert dtype == pa.float32()


def test_annotated_large_utf8():
    dtype, _ = python_type_to_arrow(Annotated[str, pa.large_utf8()])
    assert dtype == pa.large_utf8()


def test_annotated_optional():
    dtype, nullable = python_type_to_arrow(Annotated[int, pa.int8()] | None)
    assert dtype == pa.int8()
    assert nullable is True


def test_annotated_non_arrow_metadata_falls_through():
    """Annotated metadata that is not a pa.DataType should be ignored; base type used."""
    dtype, _ = python_type_to_arrow(Annotated[int, "some_doc_string"])
    assert dtype == pa.int64()


# ---------------------------------------------------------------------------
# Regression tests for Annotated detection (get_origin(ann) is Annotated)
#
# Bug: the original code used typing_objects.is_annotated(get_origin(ann)),
# relying on an opaque double-indirection.  The fix uses the explicit
# `get_origin(annotation) is Annotated` form.  These tests pin the detection
# mechanism so any future regression is caught immediately.
# ---------------------------------------------------------------------------


def test_get_origin_of_annotated_is_annotated_class():
    """Prerequisite: get_origin(Annotated[T, ...]) must be the bare Annotated class.

    This is the invariant the detection in _types._resolve() depends on.
    If this fails, the entire Annotated branch will silently stop working.
    """
    assert get_origin(Annotated[int, pa.int32()]) is Annotated


def test_annotated_detection_with_multiple_metadata_items():
    """Arrow DataType override is found even when it is not the first metadata item."""
    dtype, _ = python_type_to_arrow(Annotated[int, "description", pa.int8()])
    assert dtype == pa.int8()


def test_annotated_detection_first_arrow_type_wins():
    """When two pa.DataType values appear, the first one is used."""
    dtype, _ = python_type_to_arrow(Annotated[int, pa.int8(), pa.int16()])
    assert dtype == pa.int8()


def test_annotated_with_pydantic_validator_falls_through_to_base():
    """Annotated[str, BeforeValidator(...)] carries no pa.DataType; base type used."""
    dtype, _ = python_type_to_arrow(Annotated[str, BeforeValidator(str.strip)])
    assert dtype == pa.utf8()


def test_annotated_nullable_preserves_override():
    """Nullability is preserved when Annotated wraps an Optional."""
    dtype, nullable = python_type_to_arrow(Annotated[int, pa.int8()] | None)
    assert dtype == pa.int8()
    assert nullable is True


def test_annotated_non_nullable_by_default():
    """Annotated override without Optional is non-nullable."""
    _, nullable = python_type_to_arrow(Annotated[int, pa.int8()])
    assert nullable is False


class SensorReading(BaseModel):
    sensor_id: Annotated[int, pa.int16()]
    temperature: Annotated[float, pa.float32()]
    label: Annotated[str, pa.large_utf8()]


def test_annotated_model_schema():
    schema = model_to_schema(SensorReading)
    assert schema.field("sensor_id").type == pa.int16()
    assert schema.field("temperature").type == pa.float32()
    assert schema.field("label").type == pa.large_utf8()


def test_annotated_round_trip():
    rows = [{"sensor_id": 1, "temperature": 23.5, "label": "outside"}]
    frame = ArrowFrame[SensorReading].from_rows(rows)
    result = list(frame)
    assert result[0].sensor_id == 1
    assert abs(result[0].temperature - 23.5) < 0.01


def test_annotated_round_trip_via_from_iterable():
    """Annotated DataType overrides work end-to-end through GeneratorSource."""
    frame = ArrowFrame[SensorReading].from_iterable(iter([{"sensor_id": 42, "temperature": 0.5, "label": "test"}]))
    schema = frame.schema
    assert schema.field("sensor_id").type == pa.int16()
    assert schema.field("temperature").type == pa.float32()
    result = frame.collect()
    assert result[0].sensor_id == 42


# ---------------------------------------------------------------------------
# frozenset[T] → pa.list_(T)
# ---------------------------------------------------------------------------


def test_frozenset_of_str():
    dtype, nullable = python_type_to_arrow(frozenset[str])
    assert pa.types.is_list(dtype)
    assert dtype.value_type == pa.utf8()
    assert nullable is False


def test_frozenset_of_int():
    dtype, nullable = python_type_to_arrow(frozenset[int])
    assert pa.types.is_list(dtype)
    assert dtype.value_type == pa.int64()


def test_optional_frozenset():
    dtype, nullable = python_type_to_arrow(frozenset[str] | None)
    assert pa.types.is_list(dtype)
    assert nullable is True


class ModelWithFrozenset(BaseModel):
    tags: frozenset[str]
    codes: frozenset[int] | None = None


def test_frozenset_schema():
    schema = model_to_schema(ModelWithFrozenset)
    assert pa.types.is_list(schema.field("tags").type)
    assert schema.field("tags").type.value_type == pa.utf8()


def test_frozenset_round_trip():
    rows = [
        {"tags": frozenset(["a", "b"])},
        {"tags": frozenset(["c"]), "codes": frozenset([1, 2])},
    ]
    frame = ArrowFrame[ModelWithFrozenset].from_rows(rows)
    result = list(frame)
    assert isinstance(result[0].tags, frozenset)
    assert "a" in result[0].tags
    assert result[1].codes == frozenset({1, 2})
