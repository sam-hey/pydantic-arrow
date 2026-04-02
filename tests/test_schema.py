"""Tests for Pydantic model -> Arrow schema conversion (_schema.py)."""

from __future__ import annotations

import pyarrow as pa
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_arrow._schema import model_to_schema


def test_simple_schema():
    class User(BaseModel):
        name: str
        age: int
        score: float
        active: bool

    schema = model_to_schema(User)
    assert str(schema) == snapshot(
        """\
name: string not null
age: int64 not null
score: double not null
active: bool not null"""
    )


def test_optional_fields_are_nullable():
    class Model(BaseModel):
        required: str
        optional: str | None = None

    schema = model_to_schema(Model)
    assert schema.field("required").nullable is False
    assert schema.field("optional").nullable is True


def test_nested_model_becomes_struct():
    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    schema = model_to_schema(Person)
    addr_field = schema.field("address")
    assert pa.types.is_struct(addr_field.type)
    assert addr_field.nullable is False

    inner_names = {addr_field.type.field(i).name for i in range(addr_field.type.num_fields)}
    assert inner_names == {"street", "city"}


def test_list_field():
    class Model(BaseModel):
        tags: list[str]

    schema = model_to_schema(Model)
    tags_field = schema.field("tags")
    assert pa.types.is_list(tags_field.type)
    assert tags_field.type.value_type == pa.utf8()


def test_dict_field():
    class Model(BaseModel):
        counts: dict[str, int]

    schema = model_to_schema(Model)
    counts_field = schema.field("counts")
    assert pa.types.is_map(counts_field.type)


def test_schema_field_count():
    class Model(BaseModel):
        a: str
        b: int
        c: float

    schema = model_to_schema(Model)
    assert len(schema) == 3


def test_full_complex_schema_snapshot():
    """Snapshot of a model with many types for regression testing."""
    from datetime import date, datetime

    class Inner(BaseModel):
        x: int

    class Complex(BaseModel):
        name: str
        count: int
        ratio: float
        flag: bool
        created: datetime
        birthday: date
        tags: list[str]
        meta: Inner
        alias: str | None = None

    schema = model_to_schema(Complex)
    assert str(schema) == snapshot(
        """\
name: string not null
count: int64 not null
ratio: double not null
flag: bool not null
created: timestamp[us] not null
birthday: date32[day] not null
tags: list<item: string> not null
  child 0, item: string
meta: struct<x: int64 not null> not null
  child 0, x: int64 not null
alias: string"""
    )
