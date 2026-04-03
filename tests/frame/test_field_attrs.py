"""Tests for Pydantic Field() attribute compatibility with pydantic-arrow.

Every Field() parameter is grouped by how it interacts with the Arrow layer:

  A. Numeric constraints (gt, ge, lt, le, multiple_of)
       Arrow type unchanged; constraint enforced by Pydantic on validation.

  B. String / bytes / list constraints (min_length, max_length, pattern)
       Arrow type unchanged; constraint enforced by Pydantic on validation.

  C. Decimal precision constraints (max_digits, decimal_places)
       Directly determines pa.decimal128(precision, scale).

  D. Default values (default, default_factory)
       Does NOT change nullability; fields with non-None defaults are still
       non-nullable in Arrow.  Missing rows get the default filled in.

  E. Aliases (alias, validation_alias, serialization_alias)
       Arrow column names always use the Python attribute name.
       Round-trips work when populate_by_name=True is set on the model.
       serialization_alias is transparent because _to_dict uses model_dump()
       without by_alias=True.

  F. Metadata (title, description, repr, frozen)
       Completely ignored by the Arrow layer; type and round-trip unaffected.

  G. exclude=True
       The field is omitted from Arrow schema and batches because model_dump()
       does not include excluded fields.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated

import pyarrow as pa
import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pydantic_arrow import ArrowFrame
from pydantic_arrow._schema import model_to_schema
from pydantic_arrow._types import python_type_to_arrow

# ===========================================================================
# A. Numeric constraints
# ===========================================================================


class ModelNumericConstraints(BaseModel):
    count: int = Field(gt=0, lt=1000)
    ratio: float = Field(ge=0.0, le=1.0)
    step: int = Field(multiple_of=5)
    score: float = Field(gt=0.0)


class TestNumericConstraints:
    def test_int_gt_arrow_type_is_int64(self):
        schema = model_to_schema(ModelNumericConstraints)
        assert schema.field("count").type == pa.int64()

    def test_float_ge_arrow_type_is_float64(self):
        schema = model_to_schema(ModelNumericConstraints)
        assert schema.field("ratio").type == pa.float64()

    def test_multiple_of_arrow_type_is_int64(self):
        schema = model_to_schema(ModelNumericConstraints)
        assert schema.field("step").type == pa.int64()

    def test_valid_data_round_trips(self):
        rows = [{"count": 5, "ratio": 0.5, "step": 10, "score": 1.5}]
        frame = ArrowFrame[ModelNumericConstraints].from_rows(rows)
        result = list(frame)
        assert result[0].count == 5
        assert result[0].ratio == 0.5
        assert result[0].step == 10

    def test_gt_constraint_enforced_on_read(self):
        """Reading an Arrow batch with count=0 violates gt=0 and raises ValidationError."""
        schema = model_to_schema(ModelNumericConstraints)
        batch = pa.RecordBatch.from_pylist(
            [{"count": 0, "ratio": 0.5, "step": 5, "score": 1.0}],
            schema=schema,
        )
        frame = ArrowFrame[ModelNumericConstraints].from_arrow(pa.Table.from_batches([batch]))
        with pytest.raises(ValidationError, match="greater_than"):
            list(frame)

    def test_ge_constraint_enforced_on_read(self):
        """ratio=-0.1 violates ge=0.0."""
        schema = model_to_schema(ModelNumericConstraints)
        batch = pa.RecordBatch.from_pylist(
            [{"count": 1, "ratio": -0.1, "step": 5, "score": 1.0}],
            schema=schema,
        )
        frame = ArrowFrame[ModelNumericConstraints].from_arrow(pa.Table.from_batches([batch]))
        with pytest.raises(ValidationError):
            list(frame)

    def test_lt_constraint_enforced_on_read(self):
        """count=1000 violates lt=1000."""
        schema = model_to_schema(ModelNumericConstraints)
        batch = pa.RecordBatch.from_pylist(
            [{"count": 1000, "ratio": 0.5, "step": 5, "score": 1.0}],
            schema=schema,
        )
        frame = ArrowFrame[ModelNumericConstraints].from_arrow(pa.Table.from_batches([batch]))
        with pytest.raises(ValidationError):
            list(frame)

    def test_multiple_of_constraint_enforced_on_read(self):
        """step=7 violates multiple_of=5."""
        schema = model_to_schema(ModelNumericConstraints)
        batch = pa.RecordBatch.from_pylist(
            [{"count": 1, "ratio": 0.5, "step": 7, "score": 1.0}],
            schema=schema,
        )
        frame = ArrowFrame[ModelNumericConstraints].from_arrow(pa.Table.from_batches([batch]))
        with pytest.raises(ValidationError):
            list(frame)

    def test_annotated_constraints_same_as_field(self):
        """Annotated[int, Field(gt=0)] produces the same Arrow type as int."""
        dtype, nullable = python_type_to_arrow(Annotated[int, Field(gt=0)])
        assert dtype == pa.int64()
        assert nullable is False


# ===========================================================================
# B. String / bytes / list constraints
# ===========================================================================


class ModelStringConstraints(BaseModel):
    name: str = Field(min_length=1, max_length=50)
    code: str = Field(pattern=r"^[A-Z]{3}$")
    tag: str = Field(min_length=2)


class TestStringConstraints:
    def test_str_max_length_arrow_type_is_utf8(self):
        schema = model_to_schema(ModelStringConstraints)
        assert schema.field("name").type == pa.utf8()

    def test_str_pattern_arrow_type_is_utf8(self):
        schema = model_to_schema(ModelStringConstraints)
        assert schema.field("code").type == pa.utf8()

    def test_valid_data_round_trips(self):
        rows = [{"name": "Alice", "code": "ABC", "tag": "ok"}]
        frame = ArrowFrame[ModelStringConstraints].from_rows(rows)
        result = list(frame)
        assert result[0].name == "Alice"
        assert result[0].code == "ABC"

    def test_min_length_constraint_enforced_on_read(self):
        """name="" violates min_length=1."""
        schema = model_to_schema(ModelStringConstraints)
        batch = pa.RecordBatch.from_pylist(
            [{"name": "", "code": "ABC", "tag": "ok"}],
            schema=schema,
        )
        frame = ArrowFrame[ModelStringConstraints].from_arrow(pa.Table.from_batches([batch]))
        with pytest.raises(ValidationError, match="string_too_short"):
            list(frame)

    def test_max_length_constraint_enforced_on_read(self):
        """name with 51 chars violates max_length=50."""
        long_name = "A" * 51
        schema = model_to_schema(ModelStringConstraints)
        batch = pa.RecordBatch.from_pylist(
            [{"name": long_name, "code": "ABC", "tag": "ok"}],
            schema=schema,
        )
        frame = ArrowFrame[ModelStringConstraints].from_arrow(pa.Table.from_batches([batch]))
        with pytest.raises(ValidationError, match="string_too_long"):
            list(frame)

    def test_pattern_constraint_enforced_on_read(self):
        """code="abc" violates pattern=^[A-Z]{3}$."""
        schema = model_to_schema(ModelStringConstraints)
        batch = pa.RecordBatch.from_pylist(
            [{"name": "Alice", "code": "abc", "tag": "ok"}],
            schema=schema,
        )
        frame = ArrowFrame[ModelStringConstraints].from_arrow(pa.Table.from_batches([batch]))
        with pytest.raises(ValidationError, match="string_pattern_mismatch"):
            list(frame)

    def test_list_max_length_constraint(self):
        """list field with max_length enforced."""

        class M(BaseModel):
            items: list[str] = Field(max_length=3)

        schema = model_to_schema(M)
        assert pa.types.is_list(schema.field("items").type)

        # Valid: 3 items
        frame = ArrowFrame[M].from_rows([{"items": ["a", "b", "c"]}])
        result = list(frame)
        assert result[0].items == ["a", "b", "c"]

        # Invalid: 4 items
        batch = pa.RecordBatch.from_pylist([{"items": ["a", "b", "c", "d"]}], schema=schema)
        with pytest.raises(ValidationError):
            list(ArrowFrame[M].from_arrow(pa.Table.from_batches([batch])))


# ===========================================================================
# C. Decimal precision constraints
# ===========================================================================


class ModelDecimal(BaseModel):
    price: Decimal = Field(max_digits=10, decimal_places=2)
    tax_rate: Decimal = Field(max_digits=5, decimal_places=4)
    amount: Decimal  # no constraints → fallback (38, 18)


class TestDecimalConstraints:
    def test_decimal_with_max_digits_and_decimal_places(self):
        schema = model_to_schema(ModelDecimal)
        assert schema.field("price").type == pa.decimal128(10, 2)

    def test_decimal_second_field_precision(self):
        schema = model_to_schema(ModelDecimal)
        assert schema.field("tax_rate").type == pa.decimal128(5, 4)

    def test_decimal_without_constraints_uses_fallback(self):
        schema = model_to_schema(ModelDecimal)
        assert schema.field("amount").type == pa.decimal128(38, 18)

    def test_decimal_round_trip(self):
        rows = [{"price": Decimal("9.99"), "tax_rate": Decimal("0.0825"), "amount": Decimal("100")}]
        frame = ArrowFrame[ModelDecimal].from_rows(rows)
        result = list(frame)
        assert result[0].price == Decimal("9.99")
        assert result[0].tax_rate == Decimal("0.0825")

    def test_annotated_decimal_constraints(self):
        """Annotated[Decimal, Field(max_digits=6, decimal_places=2)]."""
        from pydantic.fields import FieldInfo

        fi = FieldInfo.from_annotated_attribute(Decimal, Field(max_digits=6, decimal_places=2))
        dtype, _ = python_type_to_arrow(Decimal, field_info=fi)
        assert dtype == pa.decimal128(6, 2)


# ===========================================================================
# D. Default values
# ===========================================================================


class ModelWithDefaults(BaseModel):
    name: str
    score: float = Field(default=0.0)
    label: str = Field(default="unknown")
    tags: list[str] = Field(default_factory=list)
    priority: int = Field(default=1)


class ModelWithNullableDefault(BaseModel):
    name: str
    nickname: str | None = Field(default=None)
    extra: int | None = None


class TestDefaultValues:
    def test_non_none_default_field_is_non_nullable(self):
        """Fields with non-None defaults still produce non-nullable Arrow columns."""
        schema = model_to_schema(ModelWithDefaults)
        assert schema.field("score").nullable is False
        assert schema.field("label").nullable is False
        assert schema.field("priority").nullable is False

    def test_default_factory_field_is_non_nullable(self):
        schema = model_to_schema(ModelWithDefaults)
        assert schema.field("tags").nullable is False

    def test_nullable_default_none_is_nullable(self):
        """str | None = None produces a nullable Arrow column."""
        schema = model_to_schema(ModelWithNullableDefault)
        assert schema.field("nickname").nullable is True
        assert schema.field("extra").nullable is True

    def test_row_omitting_default_field_uses_default(self):
        """from_rows with a row that omits a default field uses the default."""
        rows = [{"name": "Alice"}]
        frame = ArrowFrame[ModelWithDefaults].from_rows(rows)
        result = list(frame)
        assert result[0].score == 0.0
        assert result[0].label == "unknown"
        assert result[0].tags == []
        assert result[0].priority == 1

    def test_row_omitting_nullable_default_uses_none(self):
        rows = [{"name": "Bob"}]
        frame = ArrowFrame[ModelWithNullableDefault].from_rows(rows)
        result = list(frame)
        assert result[0].nickname is None
        assert result[0].extra is None

    def test_default_factory_list_round_trip(self):
        rows = [{"name": "Alice", "tags": ["x", "y"]}, {"name": "Bob"}]
        frame = ArrowFrame[ModelWithDefaults].from_rows(rows)
        result = list(frame)
        assert result[0].tags == ["x", "y"]
        assert result[1].tags == []

    def test_explicit_value_overrides_default(self):
        rows = [{"name": "Alice", "score": 9.5, "label": "expert"}]
        frame = ArrowFrame[ModelWithDefaults].from_rows(rows)
        result = list(frame)
        assert result[0].score == 9.5
        assert result[0].label == "expert"


# ===========================================================================
# E. Aliases
# ===========================================================================


class ModelWithAlias(BaseModel):
    """Arrow column uses Python attribute name; alias used only for Pydantic validation."""

    model_config = ConfigDict(populate_by_name=True)

    user_name: str = Field(alias="userName")
    age_years: int = Field(alias="ageYears")


class ModelWithSerializationAlias(BaseModel):
    """serialization_alias does not affect Arrow column name because _to_dict uses
    model_dump() without by_alias=True."""

    user_name: str = Field(serialization_alias="userName")
    score: float = Field(serialization_alias="finalScore")


class ModelWithValidationAlias(BaseModel):
    """validation_alias only — Arrow column is the Python attribute name."""

    model_config = ConfigDict(populate_by_name=True)

    user_name: str = Field(validation_alias="userName")


class TestAliases:
    # -- alias --

    def test_alias_schema_uses_python_name(self):
        """Arrow schema field name = Python attribute, NOT alias."""
        schema = model_to_schema(ModelWithAlias)
        assert "user_name" in schema.names
        assert "userName" not in schema.names
        assert "age_years" in schema.names
        assert "ageYears" not in schema.names

    def test_alias_round_trip_with_python_names(self):
        """from_rows with Python attribute names works when populate_by_name=True."""
        rows = [{"user_name": "Alice", "age_years": 30}]
        frame = ArrowFrame[ModelWithAlias].from_rows(rows)
        result = list(frame)
        assert result[0].user_name == "Alice"
        assert result[0].age_years == 30

    def test_alias_round_trip_with_alias_names(self):
        """from_rows with alias keys also works (Pydantic validates by alias)."""
        rows = [ModelWithAlias(userName="Bob", ageYears=25)]
        frame = ArrowFrame[ModelWithAlias].from_rows(rows)
        result = list(frame)
        assert result[0].user_name == "Bob"

    def test_alias_arrow_column_contains_values(self):
        """The Arrow table has a column named 'user_name', not 'userName'."""
        rows = [{"user_name": "Alice", "age_years": 30}]
        frame = ArrowFrame[ModelWithAlias].from_rows(rows)
        table = frame.to_arrow()
        assert "user_name" in table.schema.names
        assert "userName" not in table.schema.names

    def test_alias_without_populate_by_name_fails_on_read(self):
        """Without populate_by_name=True, reading Arrow data back via model_validate
        fails because Pydantic requires the alias key, not the Python attribute name."""

        class StrictAliasModel(BaseModel):
            user_name: str = Field(alias="userName")

        rows = [StrictAliasModel(userName="Alice")]
        frame = ArrowFrame[StrictAliasModel].from_rows(rows)
        with pytest.raises(ValidationError):
            list(frame)

    # -- serialization_alias --

    def test_serialization_alias_schema_uses_python_name(self):
        """serialization_alias does not change Arrow column name."""
        schema = model_to_schema(ModelWithSerializationAlias)
        assert "user_name" in schema.names
        assert "userName" not in schema.names

    def test_serialization_alias_round_trip(self):
        """Serialization alias is transparent: model_dump() uses Python names by default."""
        rows = [{"user_name": "Alice", "score": 9.5}]
        frame = ArrowFrame[ModelWithSerializationAlias].from_rows(rows)
        result = list(frame)
        assert result[0].user_name == "Alice"
        assert result[0].score == 9.5

    def test_serialization_alias_arrow_column_name(self):
        rows = [{"user_name": "Alice", "score": 8.0}]
        table = ArrowFrame[ModelWithSerializationAlias].from_rows(rows).to_arrow()
        assert "user_name" in table.schema.names
        assert "userName" not in table.schema.names

    # -- validation_alias --

    def test_validation_alias_schema_uses_python_name(self):
        schema = model_to_schema(ModelWithValidationAlias)
        assert "user_name" in schema.names
        assert "userName" not in schema.names

    def test_validation_alias_round_trip(self):
        rows = [{"user_name": "Carol"}]
        frame = ArrowFrame[ModelWithValidationAlias].from_rows(rows)
        result = list(frame)
        assert result[0].user_name == "Carol"


# ===========================================================================
# F. Metadata: title, description, repr, frozen
# ===========================================================================


class ModelWithMetadata(BaseModel):
    name: str = Field(title="Full Name", description="The user's full name")
    age: int = Field(title="Age", description="Age in years", repr=False)
    salary: float = Field(description="Annual salary in USD")
    employee_id: int = Field(frozen=True, description="Immutable employee ID")


class TestMetadata:
    def test_title_does_not_change_arrow_type(self):
        schema = model_to_schema(ModelWithMetadata)
        assert schema.field("name").type == pa.utf8()
        assert schema.field("age").type == pa.int64()

    def test_description_does_not_change_arrow_type(self):
        schema = model_to_schema(ModelWithMetadata)
        assert schema.field("salary").type == pa.float64()

    def test_repr_false_does_not_change_arrow_type(self):
        schema = model_to_schema(ModelWithMetadata)
        assert schema.field("age").type == pa.int64()

    def test_frozen_does_not_change_arrow_type(self):
        schema = model_to_schema(ModelWithMetadata)
        assert schema.field("employee_id").type == pa.int64()

    def test_all_fields_in_schema(self):
        schema = model_to_schema(ModelWithMetadata)
        assert set(schema.names) == {"name", "age", "salary", "employee_id"}

    def test_metadata_fields_round_trip(self):
        rows = [{"name": "Alice", "age": 30, "salary": 75000.0, "employee_id": 1}]
        frame = ArrowFrame[ModelWithMetadata].from_rows(rows)
        result = list(frame)
        assert result[0].name == "Alice"
        assert result[0].age == 30
        assert result[0].salary == 75000.0
        assert result[0].employee_id == 1

    def test_frozen_field_value_round_trips(self):
        rows = [{"name": "Bob", "age": 25, "salary": 50000.0, "employee_id": 99}]
        frame = ArrowFrame[ModelWithMetadata].from_rows(rows)
        result = list(frame)
        assert result[0].employee_id == 99


# ===========================================================================
# G. exclude=True
# ===========================================================================


class ModelWithExclude(BaseModel):
    name: str
    age: int
    secret_token: str = Field(default="hidden", exclude=True)


class ModelWithMultipleExcludes(BaseModel):
    user_id: int
    display_name: str
    password_hash: str = Field(default="", exclude=True)
    internal_flag: bool = Field(default=False, exclude=True)


class TestExclude:
    def test_excluded_field_absent_from_schema(self):
        """Fields with exclude=True are omitted from the Arrow schema."""
        schema = model_to_schema(ModelWithExclude)
        assert "secret_token" not in schema.names
        assert "name" in schema.names
        assert "age" in schema.names

    def test_multiple_excluded_fields_absent_from_schema(self):
        schema = model_to_schema(ModelWithMultipleExcludes)
        assert "password_hash" not in schema.names
        assert "internal_flag" not in schema.names
        assert "user_id" in schema.names
        assert "display_name" in schema.names

    def test_excluded_field_schema_column_count(self):
        schema = model_to_schema(ModelWithExclude)
        assert len(schema) == 2  # only name and age

    def test_excluded_field_round_trip_without_secret(self):
        """The excluded field is not in Arrow data; the default is used on read."""
        rows = [{"name": "Alice", "age": 30}]
        frame = ArrowFrame[ModelWithExclude].from_rows(rows)
        result = list(frame)
        assert result[0].name == "Alice"
        assert result[0].age == 30
        assert result[0].secret_token == "hidden"  # default applied on read  # pragma: allowlist secret

    def test_excluded_field_not_in_arrow_table(self):
        rows = [{"name": "Bob", "age": 25}]
        table = ArrowFrame[ModelWithExclude].from_rows(rows).to_arrow()
        assert "secret_token" not in table.schema.names
        assert table.num_columns == 2

    def test_excluded_field_from_model_instances(self):
        """from_rows with full model instances: excluded field still absent from Arrow."""
        rows = [ModelWithExclude(name="Carol", age=35, secret_token="mysecret")]  # pragma: allowlist secret
        frame = ArrowFrame[ModelWithExclude].from_rows(rows)
        table = frame.to_arrow()
        assert "secret_token" not in table.schema.names

    def test_excluded_field_round_trip_multiple(self):
        rows = [
            {"user_id": 1, "display_name": "Alice"},
            {"user_id": 2, "display_name": "Bob"},
        ]
        frame = ArrowFrame[ModelWithMultipleExcludes].from_rows(rows)
        result = list(frame)
        assert result[0].user_id == 1
        assert result[0].display_name == "Alice"
        assert result[0].internal_flag is False  # default applied
