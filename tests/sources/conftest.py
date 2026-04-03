"""Shared fixtures and models for sources tests."""

from __future__ import annotations

import pyarrow as pa
import pytest
from pydantic import BaseModel

SCHEMA = pa.schema([pa.field("x", pa.int64()), pa.field("y", pa.utf8())])
ROWS = [{"x": i, "y": f"val_{i}"} for i in range(10)]


class XY(BaseModel):
    x: int
    y: str


@pytest.fixture
def schema() -> pa.Schema:
    return SCHEMA


@pytest.fixture
def rows() -> list[dict]:
    return list(ROWS)
