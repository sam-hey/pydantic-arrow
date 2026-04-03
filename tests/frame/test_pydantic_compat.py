"""Tests verifying that Pydantic behaviour is never broken by pydantic-arrow.

Rule: Pydantic behaviour must never be broken. Any incompatibility must be
detected and handled within `pydantic-arrow`, not worked around by the caller.

Covers:
  1. @field_validator — transformations and validations run on Arrow read-back.
  2. @model_validator — model-level validators run on read-back.
  3. @computed_field — excluded from Arrow schema; round-trips without error.
  4. Model inheritance — subclass fields fully reflected in the schema.
  5. ConfigDict(frozen=True) — instances from Arrow are immutable.
  6. Annotated validators (BeforeValidator / AfterValidator) — respected on read.
"""

from __future__ import annotations

from typing import Annotated

import pyarrow as pa
import pytest
from pydantic import BaseModel, ConfigDict, ValidationError, computed_field, field_validator, model_validator
from pydantic.functional_validators import AfterValidator, BeforeValidator

from pydantic_arrow import ArrowFrame
from pydantic_arrow._convert import batch_to_models
from pydantic_arrow._schema import model_to_schema

# ===========================================================================
# 1. @field_validator — run on write AND on Arrow read-back
# ===========================================================================


class ModelUpperName(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def uppercase_name(cls, v: str) -> str:
        return v.upper()


class TestFieldValidator:
    def test_validator_normalises_model_instance_input(self):
        """Validator runs when a model instance is passed to from_rows."""
        m = ModelUpperName(name="alice")
        assert m.name == "ALICE"
        frame = ArrowFrame[ModelUpperName].from_rows([m])
        assert frame.collect()[0].name == "ALICE"

    def test_validator_runs_on_read_via_batch_to_models(self):
        """Validator is triggered inside batch_to_models even on pre-existing batches."""
        # Create a batch that stores lowercase directly (bypassing from_rows normalisation)
        schema = pa.schema([pa.field("name", pa.utf8())])
        batch = pa.RecordBatch.from_pylist([{"name": "bob"}], schema=schema)
        results = list(batch_to_models(batch, ModelUpperName))
        assert results[0].name == "BOB"

    def test_validator_enforced_on_collect(self):
        """Constraints enforced by @field_validator surface as ValidationError on collect."""

        class ModelPositiveAge(BaseModel):
            age: int

            @field_validator("age")
            @classmethod
            def must_be_positive(cls, v: int) -> int:
                if v <= 0:
                    raise ValueError("age must be positive")
                return v

        schema = pa.schema([pa.field("age", pa.int64())])
        batch = pa.RecordBatch.from_pylist([{"age": -5}], schema=schema)
        with pytest.raises(ValidationError, match="age must be positive"):
            list(batch_to_models(batch, ModelPositiveAge))


# ===========================================================================
# 2. @model_validator — model-level validators run on read-back
# ===========================================================================


class ModelCrossFieldCheck(BaseModel):
    low: int
    high: int

    @model_validator(mode="after")
    def ensure_low_lt_high(self) -> ModelCrossFieldCheck:
        if self.low >= self.high:
            raise ValueError("low must be less than high")
        return self


class TestModelValidator:
    def test_valid_data_passes(self):
        frame = ArrowFrame[ModelCrossFieldCheck].from_rows([{"low": 1, "high": 10}])
        results = frame.collect()
        assert results[0].low == 1
        assert results[0].high == 10

    def test_invalid_data_raises_on_collect(self):
        """Cross-field model_validator fires on read and raises ValidationError."""
        schema = pa.schema([pa.field("low", pa.int64()), pa.field("high", pa.int64())])
        batch = pa.RecordBatch.from_pylist([{"low": 5, "high": 2}], schema=schema)
        with pytest.raises(ValidationError, match="low must be less than high"):
            list(batch_to_models(batch, ModelCrossFieldCheck))

    def test_model_validator_after_computes_value(self):
        """model_validator(mode='after') can compute derived values on every read."""

        class ModelWithSum(BaseModel):
            x: int
            y: int
            total: int = 0

            @model_validator(mode="after")
            def set_total(self) -> ModelWithSum:
                object.__setattr__(self, "total", self.x + self.y)
                return self

        schema = pa.schema([pa.field("x", pa.int64()), pa.field("y", pa.int64()), pa.field("total", pa.int64())])
        # Store stale total=0; validator should recompute it to 7 on read
        batch = pa.RecordBatch.from_pylist([{"x": 3, "y": 4, "total": 0}], schema=schema)
        results = list(batch_to_models(batch, ModelWithSum))
        assert results[0].total == 7


# ===========================================================================
# 3. @computed_field — excluded from Arrow schema; round-trips without error
# ===========================================================================


class ModelComputedField(BaseModel):
    first_name: str
    last_name: str

    @computed_field
    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"


class TestComputedField:
    def test_computed_field_absent_from_schema(self):
        """computed_field properties are not in model_fields so they are not in the schema."""
        schema = model_to_schema(ModelComputedField)
        assert schema.names == ["first_name", "last_name"]

    def test_computed_field_roundtrip(self):
        """A model with @computed_field can be written and read back without errors."""
        frame = ArrowFrame[ModelComputedField].from_rows([ModelComputedField(first_name="Jane", last_name="Doe")])
        results = frame.collect()
        assert results[0].first_name == "Jane"
        assert results[0].last_name == "Doe"
        # The computed property is reconstructed from stored fields on read
        assert results[0].full_name == "Jane Doe"

    def test_computed_field_does_not_pollute_batch(self):
        """The Arrow batch must not contain a column for the computed_field."""
        frame = ArrowFrame[ModelComputedField].from_rows([ModelComputedField(first_name="A", last_name="B")])
        table = frame.to_arrow()
        assert "full_name" not in table.schema.names


# ===========================================================================
# 4. Model inheritance — all fields (own + inherited) appear in the schema
# ===========================================================================


class BaseEntity(BaseModel):
    entity_id: int
    entity_type: str


class ExtendedEntity(BaseEntity):
    label: str
    active: bool


class TestModelInheritance:
    def test_schema_contains_inherited_fields(self):
        """model_to_schema must include fields from parent models."""
        schema = model_to_schema(ExtendedEntity)
        assert set(schema.names) == {"entity_id", "entity_type", "label", "active"}

    def test_inherited_fields_correct_types(self):
        schema = model_to_schema(ExtendedEntity)
        assert schema.field("entity_id").type == pa.int64()
        assert schema.field("entity_type").type == pa.utf8()
        assert schema.field("label").type == pa.utf8()
        assert schema.field("active").type == pa.bool_()

    def test_inherited_model_roundtrip(self):
        frame = ArrowFrame[ExtendedEntity].from_rows(
            [{"entity_id": 1, "entity_type": "user", "label": "Alice", "active": True}]
        )
        results = frame.collect()
        assert results[0].entity_id == 1
        assert results[0].entity_type == "user"
        assert results[0].label == "Alice"
        assert results[0].active is True


# ===========================================================================
# 5. ConfigDict(frozen=True) — instances from Arrow are immutable
# ===========================================================================


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)
    value: int
    label: str


class TestFrozenModel:
    def test_schema_correct(self):
        schema = model_to_schema(FrozenModel)
        assert schema.field("value").type == pa.int64()
        assert schema.field("label").type == pa.utf8()

    def test_collected_instances_are_frozen(self):
        """Instances produced by collect() on a frozen model raise ValidationError on mutation."""
        frame = ArrowFrame[FrozenModel].from_rows([{"value": 42, "label": "test"}])
        instance = frame.collect()[0]
        with pytest.raises(ValidationError):
            instance.value = 99  # type: ignore[misc]

    def test_frozen_roundtrip(self):
        frame = ArrowFrame[FrozenModel].from_rows([{"value": 7, "label": "hello"}])
        results = frame.collect()
        assert results[0].value == 7
        assert results[0].label == "hello"


# ===========================================================================
# 6. Annotated validators (BeforeValidator / AfterValidator) respected on read
# ===========================================================================


def _strip_lower(v: str) -> str:
    return v.strip().lower()


class ModelAnnotatedValidators(BaseModel):
    email: Annotated[str, BeforeValidator(_strip_lower)]
    tag: Annotated[str, AfterValidator(str.upper)]


class TestAnnotatedValidators:
    def test_before_validator_normalises_on_read(self):
        """BeforeValidator in Annotated type runs when deserializing Arrow rows."""
        schema = pa.schema([pa.field("email", pa.utf8()), pa.field("tag", pa.utf8())])
        batch = pa.RecordBatch.from_pylist([{"email": "  Alice@Example.COM  ", "tag": "python"}], schema=schema)
        results = list(batch_to_models(batch, ModelAnnotatedValidators))
        assert results[0].email == "alice@example.com"

    def test_after_validator_runs_on_read(self):
        """AfterValidator in Annotated type runs when deserializing Arrow rows."""
        schema = pa.schema([pa.field("email", pa.utf8()), pa.field("tag", pa.utf8())])
        batch = pa.RecordBatch.from_pylist([{"email": "test@test.com", "tag": "python"}], schema=schema)
        results = list(batch_to_models(batch, ModelAnnotatedValidators))
        assert results[0].tag == "PYTHON"

    def test_annotated_validator_roundtrip(self):
        """Full from_rows → collect round-trip with Annotated validators."""
        frame = ArrowFrame[ModelAnnotatedValidators].from_rows([{"email": "  User@DOMAIN.org  ", "tag": "arrow"}])
        results = frame.collect()
        assert results[0].email == "user@domain.org"
        assert results[0].tag == "ARROW"
