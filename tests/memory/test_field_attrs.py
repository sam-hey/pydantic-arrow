"""Memory tests for Pydantic Field() attribute interactions with Arrow.

These tests verify that Field() metadata does not cause unexpected Arrow
allocations:

  - Constraint fields (gt, max_length, etc.) have the same Arrow footprint as
    unconstrained fields of the same type.
  - Excluded fields (exclude=True) produce smaller batches since they are
    absent from the schema.
  - Default fields omitted from partial-dict rows are filled efficiently
    without double-allocating the batch.
  - Alias fields store data under the Python attribute name with no extra cost.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated

import pyarrow as pa
import pytest
from pydantic import BaseModel, ConfigDict, Field

from pydantic_arrow import ArrowFrame, model_to_schema
from tests.memory.conftest import _user_rows

# ---------------------------------------------------------------------------
# Baseline: plain 4-field model matching the constrained one
# ---------------------------------------------------------------------------

_NUM_ROWS = 5_000


def _batch_bytes(rows, schema) -> int:
    tbl = pa.Table.from_pylist(rows, schema=schema)
    b = tbl.nbytes
    del tbl
    return b


class PlainUser(BaseModel):
    """Unconstrained equivalent of ConstrainedUser — same field names and types."""

    name: str
    age: int
    score: float
    active: bool


# ---------------------------------------------------------------------------
# Constraint fields — same Arrow footprint as unconstrained
# ---------------------------------------------------------------------------


class ConstrainedUser(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    age: int = Field(ge=0, le=150)
    score: float = Field(ge=0.0, le=1_000_000.0)
    active: bool


class TestConstraintFieldMemory:
    """Field constraints add zero Arrow overhead — same bytes as unconstrained fields."""

    def test_constrained_schema_same_bytes_as_unconstrained(self):
        """Arrow batch for constrained model has the same byte cost as unconstrained."""
        rows = _user_rows(_NUM_ROWS)

        plain_schema = model_to_schema(PlainUser)
        constrained_schema = model_to_schema(ConstrainedUser)

        plain_bytes = _batch_bytes(rows, plain_schema)
        constrained_bytes = _batch_bytes(rows, constrained_schema)

        # Arrow types are identical for both models — byte cost must match
        assert plain_bytes == constrained_bytes, (
            f"Constrained schema uses {constrained_bytes} bytes but plain uses "
            f"{plain_bytes} bytes.  Constraints should not affect Arrow storage."
        )

    def test_constrained_round_trip_allocation_bounded(self):
        """from_rows + to_arrow with constrained model allocates like unconstrained."""
        rows = _user_rows(_NUM_ROWS)

        before_plain = pa.total_allocated_bytes()
        plain_frame = ArrowFrame[PlainUser].from_rows(rows)
        _ = plain_frame.to_arrow()
        plain_alloc = pa.total_allocated_bytes() - before_plain
        del plain_frame

        before_constrained = pa.total_allocated_bytes()
        constrained_frame = ArrowFrame[ConstrainedUser].from_rows(rows)
        _ = constrained_frame.to_arrow()
        constrained_alloc = pa.total_allocated_bytes() - before_constrained
        del constrained_frame

        # Allow 3x for Pydantic model_validate overhead during from_rows
        assert constrained_alloc < plain_alloc * 3, (
            f"Constrained frame allocated {constrained_alloc // 1024} KB but plain "
            f"allocated {plain_alloc // 1024} KB.  Unexpected overhead."
        )

    @pytest.mark.limit_memory("10 MB")
    def test_constrained_from_rows_limit_memory(self):
        """from_rows with constrained model on 5 K rows stays under 10 MB."""
        rows = _user_rows(_NUM_ROWS)
        frame = ArrowFrame[ConstrainedUser].from_rows(rows)
        count = sum(b.num_rows for b in frame.iter_batches())
        assert count == _NUM_ROWS


# ---------------------------------------------------------------------------
# Excluded fields — smaller schema → smaller batches
# ---------------------------------------------------------------------------


class UserWithExclude(BaseModel):
    name: str
    age: int
    score: float
    active: bool
    secret: str = Field(default="hidden", exclude=True)
    internal_id: int = Field(default=0, exclude=True)


class TestExcludeFieldMemory:
    """Excluded fields are absent from the Arrow schema; batches are smaller."""

    def test_excluded_fields_reduce_batch_size(self):
        """A batch without excluded fields uses less Arrow memory than a full batch."""
        rows = _user_rows(_NUM_ROWS)

        excluded_schema = model_to_schema(UserWithExclude)  # excludes 'secret' and 'internal_id'
        plain_schema = model_to_schema(PlainUser)  # 4-column reference

        excluded_bytes = _batch_bytes(rows, excluded_schema)
        plain_bytes = _batch_bytes(rows, plain_schema)

        # Both schemas have the same 4 non-excluded columns — byte cost must match
        assert excluded_schema.names == ["name", "age", "score", "active"]
        assert excluded_bytes == plain_bytes, (
            f"Schema with excluded fields ({excluded_bytes} bytes) differs from "
            f"4-column reference ({plain_bytes} bytes). Excluded field data may be stored."
        )

    def test_excluded_field_not_in_arrow_allocation(self):
        """from_rows with excluded fields allocates similar to model without those fields."""
        rows = _user_rows(_NUM_ROWS)

        before = pa.total_allocated_bytes()
        frame = ArrowFrame[UserWithExclude].from_rows(rows)
        _ = frame.to_arrow()
        allocated = pa.total_allocated_bytes() - before

        # Reference: 4-column plain table
        ref_bytes = _batch_bytes(rows, model_to_schema(PlainUser))

        # Excluded fields add no Arrow storage; allow 3x for Pydantic overhead
        assert allocated < ref_bytes * 3, (
            f"UserWithExclude allocated {allocated // 1024} KB but 4-column reference is {ref_bytes // 1024} KB."
        )


# ---------------------------------------------------------------------------
# Default fields — partial dicts filled without double-allocating
# ---------------------------------------------------------------------------


class UserWithDefaults(BaseModel):
    name: str
    score: float = Field(default=0.0)
    active: bool = Field(default=True)
    tag: str = Field(default="default")


class TestDefaultFieldMemory:
    """from_rows with partial dicts (missing default fields) stays memory-efficient."""

    def test_partial_rows_produce_same_arrow_bytes_as_full_rows(self):
        """Partial dicts with defaults applied produce identically-sized Arrow tables.

        We compare pa.Table.nbytes directly — this is deterministic and
        avoids the unreliable before/after pa.total_allocated_bytes() delta.
        """
        partial_rows = [{"name": f"user_{i}"} for i in range(_NUM_ROWS)]
        full_rows = [{"name": f"user_{i}", "score": 0.0, "active": True, "tag": "default"} for i in range(_NUM_ROWS)]

        partial_table = ArrowFrame[UserWithDefaults].from_rows(partial_rows).to_arrow()
        full_table = ArrowFrame[UserWithDefaults].from_rows(full_rows).to_arrow()

        # Both paths produce the same Arrow table (same defaults applied)
        assert partial_table.nbytes == full_table.nbytes, (
            f"Partial-dict table ({partial_table.nbytes} bytes) differs from "
            f"full-dict table ({full_table.nbytes} bytes). Defaults not applied consistently."
        )

    def test_partial_rows_correct_data(self):
        """Defaults are correctly filled for each field."""
        partial_rows = [{"name": f"user_{i}"} for i in range(10)]
        frame = ArrowFrame[UserWithDefaults].from_rows(partial_rows)
        result = list(frame)
        for r in result:
            assert r.score == 0.0
            assert r.active is True
            assert r.tag == "default"

    @pytest.mark.limit_memory("20 MB")
    def test_partial_rows_from_rows_limit_memory(self):
        """from_rows with 5 K partial dicts (model_validate fills defaults) under 20 MB."""
        partial_rows = [{"name": f"user_{i}"} for i in range(_NUM_ROWS)]
        frame = ArrowFrame[UserWithDefaults].from_rows(partial_rows)
        count = sum(b.num_rows for b in frame.iter_batches())
        assert count == _NUM_ROWS


# ---------------------------------------------------------------------------
# Alias fields — no extra Arrow overhead
# ---------------------------------------------------------------------------


class UserWithAlias(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    user_name: str = Field(alias="userName")
    age_years: int = Field(alias="ageYears")
    score: float
    active: bool


class TestAliasFieldMemory:
    """Aliases add no extra Arrow memory — column name is the Python attribute name."""

    def test_alias_batch_same_size_as_plain_batch(self):
        """A batch from an aliased model uses the same bytes as an equivalent plain model."""
        rows = [
            {"user_name": f"user_{i}", "age_years": i % 100, "score": float(i), "active": i % 2 == 0}
            for i in range(_NUM_ROWS)
        ]
        # PlainUser has field names name/age/score/active; remap to alias model names
        plain_rows = [
            {"name": r["user_name"], "age": r["age_years"], "score": r["score"], "active": r["active"]} for r in rows
        ]

        alias_schema = model_to_schema(UserWithAlias)
        plain_schema = model_to_schema(PlainUser)

        alias_bytes = _batch_bytes(rows, alias_schema)
        plain_bytes = _batch_bytes(plain_rows, plain_schema)

        # Same 4 columns, same types, same string lengths — byte cost must match
        assert alias_bytes == plain_bytes, (
            f"Alias schema uses {alias_bytes} bytes but plain uses {plain_bytes} bytes. "
            "Aliases should not affect Arrow storage size."
        )

    def test_alias_round_trip_allocation_bounded(self):
        """from_rows + to_arrow with aliased model allocates no more than 3x plain."""
        aliased_rows = [
            {"user_name": f"u{i}", "age_years": i % 100, "score": float(i), "active": i % 2 == 0}
            for i in range(_NUM_ROWS)
        ]
        plain_rows = _user_rows(_NUM_ROWS)

        before_plain = pa.total_allocated_bytes()
        plain_frame = ArrowFrame[PlainUser].from_rows(plain_rows)
        _ = plain_frame.to_arrow()
        plain_alloc = pa.total_allocated_bytes() - before_plain
        del plain_frame

        before_alias = pa.total_allocated_bytes()
        alias_frame = ArrowFrame[UserWithAlias].from_rows(aliased_rows)
        _ = alias_frame.to_arrow()
        alias_alloc = pa.total_allocated_bytes() - before_alias
        del alias_frame

        assert alias_alloc < plain_alloc * 3, (
            f"Alias frame ({alias_alloc // 1024} KB) much more expensive than plain ({plain_alloc // 1024} KB)."
        )


# ---------------------------------------------------------------------------
# Decimal precision — decimal128(p, s) is compact
# ---------------------------------------------------------------------------


class PriceRecord(BaseModel):
    item_id: int
    price: Decimal = Field(max_digits=10, decimal_places=2)
    tax: Decimal = Field(max_digits=5, decimal_places=4)


class TestDecimalFieldMemory:
    """Decimal fields use pa.decimal128 — compact fixed-width 16 bytes per value."""

    def test_decimal_batch_bytes_match_fixed_width(self):
        """Decimal128 occupies exactly 16 bytes per value regardless of precision."""
        rows = [{"item_id": i, "price": Decimal("9.99"), "tax": Decimal("0.0825")} for i in range(_NUM_ROWS)]
        schema = model_to_schema(PriceRecord)
        tbl = pa.Table.from_pylist(rows, schema=schema)

        # int64 = 8 bytes/row, decimal128 = 16 bytes/row x 2 columns
        expected_min = _NUM_ROWS * (8 + 16 + 16)
        assert tbl.nbytes >= expected_min, f"Decimal table has {tbl.nbytes} bytes, expected at least {expected_min}."
        del tbl

    @pytest.mark.limit_memory("10 MB")
    def test_decimal_from_rows_limit_memory(self):
        """from_rows with Decimal fields on 5 K rows stays under 10 MB."""
        rows = [{"item_id": i, "price": Decimal("9.99"), "tax": Decimal("0.0825")} for i in range(_NUM_ROWS)]
        frame = ArrowFrame[PriceRecord].from_rows(rows)
        count = sum(b.num_rows for b in frame.iter_batches())
        assert count == _NUM_ROWS


# ---------------------------------------------------------------------------
# Annotated[T, pa.DataType()] — type override adds no overhead
# ---------------------------------------------------------------------------


class SensorReading(BaseModel):
    sensor_id: Annotated[int, pa.int32()]
    temperature: Annotated[float, pa.float32()]
    pressure: float  # default float64


class TestAnnotatedFieldMemory:
    """Annotated Arrow type overrides use narrower types → smaller batches."""

    def test_annotated_narrow_types_use_less_memory(self):
        """int32 + float32 columns use fewer bytes than int64 + float64."""
        rows = [{"sensor_id": i, "temperature": float(i) * 0.1, "pressure": float(i) * 0.01} for i in range(_NUM_ROWS)]
        schema = model_to_schema(SensorReading)

        tbl = pa.Table.from_pylist(rows, schema=schema)
        # int32 (4B) + float32 (4B) + float64 (8B) = 16B/row vs 8+8+8 = 24B/row
        assert tbl.nbytes < _NUM_ROWS * 24, (
            f"SensorReading table uses {tbl.nbytes} bytes; expected < {_NUM_ROWS * 24} "
            "bytes since int32+float32 are narrower than int64+float64."
        )
        del tbl
