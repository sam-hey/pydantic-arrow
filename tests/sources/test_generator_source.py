"""Tests for GeneratorSource — batching, validation, and Arrow coercions.

GeneratorSource is the engine behind ArrowFrame.from_iterable().  It must:
  - validate each row with model_validate() before encoding it
  - apply Arrow coercions (_to_dict) so UUID/Enum/frozenset are converted
  - yield one batch at a time without consuming the whole iterator upfront
"""

from __future__ import annotations

import enum
from typing import Any
from uuid import UUID

import pyarrow as pa
import pytest
from pydantic import BaseModel, ValidationError

from pydantic_arrow._sources import GeneratorSource

from .conftest import SCHEMA, XY

# ---------------------------------------------------------------------------
# Batching and exhaustion
# ---------------------------------------------------------------------------


class TestGeneratorSourceBatching:
    def _rows(self, n: int = 5) -> list[dict[str, Any]]:
        return [{"x": i, "y": f"val_{i}"} for i in range(n)]

    def test_schema_property(self):
        source = GeneratorSource(iter(self._rows()), XY, SCHEMA)
        assert source.schema == SCHEMA

    def test_yields_correct_total_rows(self):
        source = GeneratorSource(iter(self._rows(10)), XY, SCHEMA, batch_size=4)
        total = sum(b.num_rows for b in source.iter_batches())
        assert total == 10

    def test_chunking_respects_batch_size(self):
        source = GeneratorSource(iter(self._rows(10)), XY, SCHEMA, batch_size=3)
        batches = list(source.iter_batches())
        assert len(batches) == 4  # 3 + 3 + 3 + 1

    def test_empty_iterable_yields_nothing(self):
        source = GeneratorSource(iter([]), XY, SCHEMA)
        assert list(source.iter_batches()) == []

    def test_one_shot_exhaustion(self):
        """A second call to iter_batches() after exhaustion yields nothing."""
        source = GeneratorSource(iter(self._rows()), XY, SCHEMA)
        list(source.iter_batches())
        assert list(source.iter_batches()) == []


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestGeneratorSourceValidation:
    def test_invalid_row_raises_validation_error(self):
        """model_validate errors surface at iteration time, not at construction."""
        source = GeneratorSource(iter([{"x": "not-an-int", "y": "hello"}]), XY, SCHEMA)
        with pytest.raises(ValidationError):
            list(source.iter_batches())

    def test_validation_happens_inside_source_not_at_call_site(self):
        """The caller never calls model_validate; GeneratorSource does it internally."""
        rows = [{"x": 1, "y": "a"}, {"x": 2, "y": "b"}]
        source = GeneratorSource(iter(rows), XY, SCHEMA)
        data = next(iter(source.iter_batches())).to_pylist()
        assert data[0]["x"] == 1
        assert data[1]["y"] == "b"


# ---------------------------------------------------------------------------
# Arrow coercions — regression for GeneratorSource skipping _to_dict()
# ---------------------------------------------------------------------------


class _Colour(enum.Enum):
    RED = "red"
    BLUE = "blue"


class _ModelWithUUID(BaseModel):
    uid: UUID
    label: str


class _ModelWithEnum(BaseModel):
    name: str
    colour: _Colour


class _ModelWithFrozenset(BaseModel):
    tags: frozenset[str]


class TestGeneratorSourceCoercions:
    """GeneratorSource must apply _to_dict() coercions, not raw model_dump().

    Before the fix, UUID objects, Enum members, and frozensets were passed raw
    to pa.RecordBatch.from_pylist() and caused ArrowTypeError at runtime.
    """

    def test_uuid_coerced_to_string(self):
        uid = UUID("12345678-1234-5678-1234-567812345678")
        schema = pa.schema([pa.field("uid", pa.utf8()), pa.field("label", pa.utf8())])
        source = GeneratorSource(iter([{"uid": str(uid), "label": "test"}]), _ModelWithUUID, schema)
        batch = next(iter(source.iter_batches()))
        assert batch.column("uid")[0].as_py() == str(uid)

    def test_enum_value_coerced_to_primitive(self):
        schema = pa.schema([pa.field("name", pa.utf8()), pa.field("colour", pa.dictionary(pa.int32(), pa.utf8()))])
        source = GeneratorSource(iter([{"name": "Alice", "colour": "red"}]), _ModelWithEnum, schema)
        batch = next(iter(source.iter_batches()))
        assert batch.column("colour")[0].as_py() == "red"

    def test_frozenset_coerced_to_list(self):
        schema = pa.schema([pa.field("tags", pa.list_(pa.utf8()))])
        source = GeneratorSource(iter([{"tags": ["a", "b"]}]), _ModelWithFrozenset, schema)
        batch = next(iter(source.iter_batches()))
        assert sorted(batch.column("tags")[0].as_py()) == ["a", "b"]
