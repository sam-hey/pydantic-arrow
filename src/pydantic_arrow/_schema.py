"""Convert Pydantic BaseModel classes to PyArrow schemas."""

from __future__ import annotations

import typing

import pyarrow as pa
from pydantic import BaseModel

from pydantic_arrow._types import python_type_to_arrow

__all__ = ["model_to_schema"]


def model_to_schema(model: type[BaseModel]) -> pa.Schema:
    """Derive a :class:`pyarrow.Schema` from a Pydantic ``BaseModel`` class.

    Each model field becomes a schema field with the corresponding Arrow type.
    Nullable fields (``Optional[T]`` / ``T | None``) produce nullable Arrow fields;
    all others are non-nullable.

    Args:
        model: A Pydantic v2 ``BaseModel`` subclass.

    Returns:
        A :class:`pyarrow.Schema` matching the model's fields.
    """
    return pa.schema(_fields_to_arrow_fields(model))


def _fields_to_arrow_fields(model: type[BaseModel]) -> list[pa.Field]:
    """Return a list of :class:`pyarrow.Field` objects for *model*'s fields."""
    # get_type_hints with include_extras=True resolves string annotations (from
    # `from __future__ import annotations`) and preserves Annotated[T, ...] metadata.
    # Pydantic strips Annotated in field_info.annotation, so we always prefer
    # the resolved hint for accurate Arrow type mapping.
    try:
        resolved = typing.get_type_hints(model, include_extras=True)
    except Exception:
        resolved = {}

    arrow_fields: list[pa.Field] = []
    for name, field_info in model.model_fields.items():
        # Fields marked exclude=True are omitted from model_dump(), so there is
        # no data to store in Arrow.  Skip them to keep the schema consistent.
        if field_info.exclude:
            continue
        annotation = resolved.get(name, field_info.annotation)
        if annotation is None:
            raise TypeError(f"Field '{name}' on {model.__name__} has no type annotation.")

        dtype, nullable = python_type_to_arrow(annotation, field_info=field_info)
        # A field with a default value is implicitly optional in Pydantic,
        # but we only mark it nullable in Arrow when the type itself is Optional.
        arrow_fields.append(pa.field(name, dtype, nullable=nullable))

    return arrow_fields
