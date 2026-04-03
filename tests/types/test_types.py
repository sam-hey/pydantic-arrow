"""Tests for Python type -> Arrow type mapping (_types.py)."""

from __future__ import annotations

from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from typing import Literal
from uuid import UUID

import pyarrow as pa
import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel, Field

from pydantic_arrow._types import python_type_to_arrow


def arrow_str(annotation, **kwargs) -> str:
    """Helper: return the string representation of the mapped Arrow type."""
    dtype, nullable = python_type_to_arrow(annotation, **kwargs)
    return str(dtype), nullable


# ---------------------------------------------------------------------------
# Scalar types
# ---------------------------------------------------------------------------


def test_str():
    assert arrow_str(str) == snapshot(("string", False))


def test_int():
    assert arrow_str(int) == snapshot(("int64", False))


def test_float():
    assert arrow_str(float) == snapshot(("double", False))


def test_bool():
    assert arrow_str(bool) == snapshot(("bool", False))


def test_bytes():
    assert arrow_str(bytes) == snapshot(("binary", False))


def test_datetime():
    assert arrow_str(datetime) == snapshot(("timestamp[us]", False))


def test_date():
    assert arrow_str(date) == snapshot(("date32[day]", False))


def test_time():
    assert arrow_str(time) == snapshot(("time64[us]", False))


def test_uuid_stored_as_string():
    dtype, nullable = python_type_to_arrow(UUID)
    assert dtype == pa.utf8()
    assert nullable is False


# ---------------------------------------------------------------------------
# Decimal -- requires FieldInfo constraints
# ---------------------------------------------------------------------------


def test_decimal_with_constraints():
    from pydantic.fields import FieldInfo

    fi = FieldInfo.from_annotated_attribute(Decimal, Field(max_digits=10, decimal_places=2))
    dtype, nullable = python_type_to_arrow(Decimal, field_info=fi)
    assert dtype == pa.decimal128(10, 2)
    assert nullable is False


def test_decimal_fallback():
    dtype, _nullable = python_type_to_arrow(Decimal)
    assert dtype == pa.decimal128(38, 18)


# ---------------------------------------------------------------------------
# Optional / nullable
# ---------------------------------------------------------------------------


def test_optional_str():
    dtype, nullable = python_type_to_arrow(str | None)
    assert dtype == pa.utf8()
    assert nullable is True


def test_optional_via_typing():
    # Test the Union[X, None] form explicitly (equivalent to X | None)
    dtype, nullable = python_type_to_arrow(int | None)
    assert dtype == pa.int64()
    assert nullable is True


def test_optional_nested_model():
    class Inner(BaseModel):
        x: int

    dtype, nullable = python_type_to_arrow(Inner | None)
    assert pa.types.is_struct(dtype)
    assert nullable is True


# ---------------------------------------------------------------------------
# Generics
# ---------------------------------------------------------------------------


def test_list_of_str():
    dtype, nullable = python_type_to_arrow(list[str])
    assert pa.types.is_list(dtype)
    assert dtype.value_type == pa.utf8()
    assert nullable is False


def test_dict_str_int():
    dtype, nullable = python_type_to_arrow(dict[str, int])
    assert pa.types.is_map(dtype)
    assert nullable is False


# ---------------------------------------------------------------------------
# Literal
# ---------------------------------------------------------------------------


def test_literal_strings():
    dtype, _nullable = python_type_to_arrow(Literal["a", "b", "c"])
    assert pa.types.is_dictionary(dtype)
    assert dtype.value_type == pa.utf8()


def test_literal_ints():
    dtype, _nullable = python_type_to_arrow(Literal[1, 2, 3])
    assert dtype == pa.int64()


def test_literal_mixed_raises():
    with pytest.raises(TypeError, match="Literal values must all be str or all be int"):
        python_type_to_arrow(Literal["a", 1])


def test_any_maps_to_large_binary():
    import typing

    dtype, nullable = python_type_to_arrow(typing.Any)
    assert dtype == pa.large_binary()
    assert nullable is False


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------


class StrEnum(str, Enum):
    a = "alpha"
    b = "beta"


class IntEnum(int, Enum):
    x = 1
    y = 2


def test_str_enum():
    dtype, _nullable = python_type_to_arrow(StrEnum)
    assert pa.types.is_dictionary(dtype)


def test_int_enum():
    dtype, _nullable = python_type_to_arrow(IntEnum)
    assert dtype == pa.int64()


# ---------------------------------------------------------------------------
# Nested BaseModel -> struct
# ---------------------------------------------------------------------------


def test_nested_model_struct():
    class Inner(BaseModel):
        x: int
        y: str

    dtype, _nullable = python_type_to_arrow(Inner)
    assert pa.types.is_struct(dtype)
    field_names = {dtype.field(i).name for i in range(dtype.num_fields)}
    assert field_names == {"x", "y"}


# ---------------------------------------------------------------------------
# Unsupported type raises
# ---------------------------------------------------------------------------


def test_union_multiple_non_none_raises():
    with pytest.raises(TypeError, match="Union of multiple non-None types"):
        python_type_to_arrow(int | str)


def test_unknown_type_raises():
    with pytest.raises(TypeError):
        python_type_to_arrow(object)
