"""Mapping from Python/Pydantic type annotations to PyArrow data types."""

from __future__ import annotations

import typing
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any, get_args, get_origin
from uuid import UUID

import pyarrow as pa
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from typing_inspection import typing_objects
from typing_inspection.introspection import is_union_origin

__all__ = ["python_type_to_arrow"]


def python_type_to_arrow(annotation: Any, field_info: FieldInfo | None = None) -> tuple[pa.DataType, bool]:
    """Convert a Python type annotation to a (PyArrow DataType, nullable) pair.

    Args:
        annotation: The type annotation to convert.
        field_info: Optional Pydantic FieldInfo for extracting constraints (e.g. Decimal precision).

    Returns:
        A tuple of (pa.DataType, nullable: bool).

    Raises:
        TypeError: When the annotation cannot be mapped to an Arrow type.
    """
    return _resolve(annotation, field_info=field_info, nullable=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve(annotation: Any, *, field_info: FieldInfo | None, nullable: bool) -> tuple[pa.DataType, bool]:
    """Recursively resolve a type annotation to (pa.DataType, nullable)."""

    # --- Annotated[T, ...] — check for a pa.DataType override first ---------
    # get_origin(Annotated[T, ...]) returns the bare `Annotated` class, so
    # comparing with `is Annotated` is the clearest way to detect this form.
    if get_origin(annotation) is Annotated:
        base, *metadata = get_args(annotation)
        for meta in metadata:
            if isinstance(meta, pa.DataType):
                return meta, nullable
        # No pa.DataType found in metadata; fall through with the base type
        return _resolve(base, field_info=field_info, nullable=nullable)

    # --- Union / Optional ---------------------------------------------------
    origin = get_origin(annotation)
    if origin is not None and is_union_origin(origin):
        args = get_args(annotation)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            # Optional[T] or T | None
            dtype, _ = _resolve(non_none[0], field_info=field_info, nullable=True)
            return dtype, True
        # Union of multiple non-None types -- store as large_binary (opaque bytes)
        raise TypeError(
            f"Union of multiple non-None types is not supported: {annotation}. "
            "Use Optional[T] (i.e. T | None) or a nested BaseModel."
        )

    # --- Literal ------------------------------------------------------------
    if origin is not None and typing_objects.is_literal(origin):
        values = get_args(annotation)
        if not values:
            raise TypeError(f"Literal with no arguments: {annotation}")
        if all(isinstance(v, str) for v in values):
            return pa.dictionary(pa.int32(), pa.utf8()), nullable
        if all(isinstance(v, int) for v in values):
            return pa.int64(), nullable
        raise TypeError(f"Literal values must all be str or all be int, got: {annotation}")

    # --- list[T] ------------------------------------------------------------
    if origin is list:
        (inner_args,) = get_args(annotation) or (Any,)
        inner_dtype, _ = _resolve(inner_args, field_info=None, nullable=False)
        return pa.list_(inner_dtype), nullable

    # --- frozenset[T] → pa.list_(T) — Arrow has no native set type ----------
    if origin is frozenset:
        (inner_args,) = get_args(annotation) or (Any,)
        inner_dtype, _ = _resolve(inner_args, field_info=None, nullable=False)
        return pa.list_(inner_dtype), nullable

    # --- dict[K, V] ---------------------------------------------------------
    if origin is dict:
        kv = get_args(annotation)
        k_type, v_type = (kv[0], kv[1]) if len(kv) == 2 else (Any, Any)
        k_dtype, _ = _resolve(k_type, field_info=None, nullable=False)
        v_dtype, _ = _resolve(v_type, field_info=None, nullable=False)
        return pa.map_(k_dtype, v_dtype), nullable

    # --- Bare / scalar types ------------------------------------------------
    return _scalar_to_arrow(annotation, field_info=field_info, nullable=nullable)


def _scalar_to_arrow(tp: Any, *, field_info: FieldInfo | None, nullable: bool) -> tuple[pa.DataType, bool]:
    """Map a concrete (non-generic) type to a PyArrow DataType."""

    if tp is str:
        return pa.utf8(), nullable
    if tp is int:
        return pa.int64(), nullable
    if tp is float:
        return pa.float64(), nullable
    if tp is bool:
        return pa.bool_(), nullable
    if tp is bytes:
        return pa.binary(), nullable
    if tp is datetime:
        return pa.timestamp("us"), nullable
    if tp is date:
        return pa.date32(), nullable
    if tp is time:
        return pa.time64("us"), nullable
    if tp is timedelta:
        return pa.duration("us"), nullable
    if tp is UUID:
        return pa.utf8(), nullable  # stored as canonical UUID string
    if tp is Decimal:
        precision, scale = _decimal_constraints(field_info)
        return pa.decimal128(precision, scale), nullable

    # --- Enum ---------------------------------------------------------------
    if isinstance(tp, type) and issubclass(tp, Enum):
        # Determine the Enum's underlying type from the first member
        members = list(tp)
        if members and isinstance(members[0].value, int):
            return pa.int64(), nullable
        return pa.dictionary(pa.int32(), pa.utf8()), nullable

    # --- Pydantic BaseModel (nested struct) ---------------------------------
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        from pydantic_arrow._schema import _fields_to_arrow_fields

        fields = _fields_to_arrow_fields(tp)
        return pa.struct(fields), nullable

    # --- typing.Any ---------------------------------------------------------
    if tp is typing.Any:
        return pa.large_binary(), nullable

    raise TypeError(f"Cannot map Python type {tp!r} to an Arrow type. Unsupported annotation.")


def _decimal_constraints(field_info: FieldInfo | None) -> tuple[int, int]:
    """Extract (precision, scale) from Pydantic FieldInfo metadata for Decimal fields."""
    if field_info is not None:
        for meta in field_info.metadata:
            # pydantic annotated_types constraints carry max_digits / decimal_places
            max_digits = getattr(meta, "max_digits", None)
            decimal_places = getattr(meta, "decimal_places", None)
            if max_digits is not None and decimal_places is not None:
                return int(max_digits), int(decimal_places)
    return 38, 18  # sensible fallback
