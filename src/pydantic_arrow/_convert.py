"""Conversion utilities between PyArrow RecordBatches and Pydantic models."""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from decimal import Decimal
from enum import Enum
from typing import Any, TypeVar
from uuid import UUID

import pyarrow as pa
from pydantic import BaseModel

__all__ = ["batch_to_models", "models_to_batch", "rows_to_batches"]

T = TypeVar("T", bound=BaseModel)

DEFAULT_BATCH_SIZE = 65_536


def batch_to_models(batch: pa.RecordBatch, model: type[T]) -> Iterator[T]:
    """Yield validated *model* instances from a single :class:`pyarrow.RecordBatch`.

    Each row is converted to a Python dict via ``batch.to_pylist()`` and then
    validated with ``model.model_validate(row)``.  Arrow ``map<K,V>`` columns
    deserialise as ``[(key, value), ...]``; these are converted back to plain
    ``dict`` objects before validation.

    Args:
        batch: A PyArrow record batch.
        model: The Pydantic model class to validate each row against.

    Yields:
        Validated model instances, one per row.
    """
    schema = batch.schema
    for row in batch.to_pylist():
        yield model.model_validate(_fix_arrow_row(row, schema))


def models_to_batch(
    models: list[BaseModel | dict[str, Any]],
    schema: pa.Schema,
) -> pa.RecordBatch:
    """Convert a list of Pydantic model instances (or plain dicts) to a RecordBatch.

    Model instances are serialised with ``model.model_dump()`` before conversion.
    Special types (``UUID``, ``Enum``, ``Decimal``) are coerced to their Arrow-
    compatible representations (str, value, and left as-is respectively).

    Args:
        models: A list of ``BaseModel`` instances or plain ``dict`` objects.
        schema: The target Arrow schema.

    Returns:
        A :class:`pyarrow.RecordBatch` with the given *schema*.
    """
    rows = [_to_dict(m) for m in models]
    return pa.RecordBatch.from_pylist(rows, schema=schema)


def rows_to_batches(
    rows: list[dict[str, Any]],
    schema: pa.Schema,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Iterator[pa.RecordBatch]:
    """Chunk a list of dicts into record batches of at most *batch_size* rows.

    Args:
        rows: A list of row dicts.
        schema: The Arrow schema to apply.
        batch_size: Maximum rows per batch.

    Yields:
        :class:`pyarrow.RecordBatch` objects.
    """
    it = iter(rows)
    while True:
        chunk = list(itertools.islice(it, batch_size))
        if not chunk:
            break
        yield pa.RecordBatch.from_pylist(chunk, schema=schema)


def _to_dict(value: BaseModel | dict[str, Any]) -> dict[str, Any]:
    """Serialise a model or dict to an Arrow-compatible plain dict."""
    raw = value.model_dump() if isinstance(value, BaseModel) else dict(value)
    return {k: _coerce(v) for k, v in raw.items()}


def _coerce(value: Any) -> Any:
    """Recursively coerce Python values to Arrow-friendly primitives."""
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Decimal):
        return value  # pyarrow handles Decimal natively with decimal128 schema
    if isinstance(value, frozenset):
        return [_coerce(v) for v in value]
    if isinstance(value, dict):
        return {k: _coerce(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_coerce(v) for v in value]
    return value


def _fix_arrow_row(row: dict[str, Any], schema: pa.Schema) -> dict[str, Any]:
    """Re-hydrate a row produced by ``RecordBatch.to_pylist()``.

    Arrow ``map<K,V>`` columns deserialise as ``[(key, value), ...]`` rather
    than ``{key: value}``.  This function walks the schema and converts any
    such values back to plain Python dicts so Pydantic can validate them.
    """
    return {field.name: _fix_arrow_value(row[field.name], field.type) for field in schema}


def _fix_arrow_value(value: Any, arrow_type: pa.DataType) -> Any:
    """Recursively convert Arrow-deserialized values to Python-native equivalents."""
    if value is None:
        return None
    if pa.types.is_map(arrow_type):
        # map<K,V> -> [(k, v), ...] in to_pylist(); convert to dict
        if isinstance(value, list):
            return {k: _fix_arrow_value(v, arrow_type.item_type) for k, v in value}
        return value
    if pa.types.is_struct(arrow_type):
        if isinstance(value, dict):
            return {
                arrow_type.field(i).name: _fix_arrow_value(
                    value[arrow_type.field(i).name],
                    arrow_type.field(i).type,
                )
                for i in range(arrow_type.num_fields)
            }
        return value
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
        if isinstance(value, list):
            return [_fix_arrow_value(item, arrow_type.value_type) for item in value]
        return value
    return value
