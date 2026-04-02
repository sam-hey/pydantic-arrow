"""Shared fixtures for pydantic-arrow tests."""

from __future__ import annotations

from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum, StrEnum
from pathlib import Path
from typing import Literal
from uuid import UUID

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Fixture models
# ---------------------------------------------------------------------------


class SimpleUser(BaseModel):
    name: str
    age: int
    score: float
    active: bool
    email: str | None = None


class Address(BaseModel):
    street: str
    city: str
    zip_code: str


class PersonWithNested(BaseModel):
    name: str
    address: Address


class AllTypes(BaseModel):
    str_field: str
    int_field: int
    float_field: float
    bool_field: bool
    bytes_field: bytes
    dt_field: datetime
    date_field: date
    time_field: time
    decimal_field: Decimal = Field(decimal_places=4, max_digits=12)
    uuid_field: UUID
    list_field: list[str]
    dict_field: dict[str, int]
    opt_field: str | None = None


class StatusEnum(StrEnum):
    active = "active"
    inactive = "inactive"


class PriorityEnum(int, Enum):
    low = 1
    high = 2


class ModelWithEnum(BaseModel):
    name: str
    status: StatusEnum
    priority: PriorityEnum


class ModelWithLiteral(BaseModel):
    name: str
    role: Literal["admin", "user", "guest"]


# ---------------------------------------------------------------------------
# Data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_rows() -> list[dict]:
    return [
        {"name": "Alice", "age": 30, "score": 9.5, "active": True},
        {"name": "Bob", "age": 25, "score": 7.2, "active": False, "email": "bob@test.com"},
        {"name": "Carol", "age": 35, "score": 8.8, "active": True},
    ]


@pytest.fixture
def simple_models(simple_rows) -> list[SimpleUser]:
    return [SimpleUser(**r) for r in simple_rows]


@pytest.fixture
def parquet_file(simple_rows, tmp_path: Path) -> Path:
    """Write a small Parquet file and return its path."""
    from pydantic_arrow import model_to_schema

    schema = model_to_schema(SimpleUser)
    table = pa.Table.from_pylist(simple_rows, schema=schema)
    path = tmp_path / "test.parquet"
    pq.write_table(table, path)
    return path


@pytest.fixture
def large_parquet_file(tmp_path: Path) -> tuple[Path, int]:
    """Write a Parquet file with 10 000 rows, return (path, num_rows)."""
    from pydantic_arrow import model_to_schema

    num_rows = 10_000
    schema = model_to_schema(SimpleUser)
    rows = [{"name": f"user_{i}", "age": i % 100, "score": float(i), "active": i % 2 == 0} for i in range(num_rows)]
    table = pa.Table.from_pylist(rows, schema=schema)
    path = tmp_path / "large.parquet"
    pq.write_table(table, path)
    return path, num_rows


@pytest.fixture(scope="session")
def prebuilt_rows() -> list[dict]:
    """50 K pre-built row dicts.  Session-scoped so they are allocated once and
    not counted against any individual test's memray budget."""
    return [{"name": f"user_{i}", "age": i % 100, "score": float(i), "active": i % 2 == 0} for i in range(50_000)]


@pytest.fixture(scope="session")
def multigroup_parquet_file(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, int, int]:
    """Write a large Parquet file with *many small row groups* to demonstrate true lazy loading.

    Returns (path, total_rows, rows_per_group).  Using ``scope="session"`` avoids
    re-generating the 200 K row dataset for every test that needs it.
    """
    from pydantic_arrow import model_to_schema

    total_rows = 200_000
    rows_per_group = 10_000  # 20 row groups of 10 K rows each
    schema = model_to_schema(SimpleUser)
    rows = [{"name": f"user_{i}", "age": i % 100, "score": float(i), "active": i % 2 == 0} for i in range(total_rows)]
    table = pa.Table.from_pylist(rows, schema=schema)
    path = tmp_path_factory.mktemp("memray_data") / "multigroup.parquet"
    pq.write_table(table, path, row_group_size=rows_per_group)
    return path, total_rows, rows_per_group
