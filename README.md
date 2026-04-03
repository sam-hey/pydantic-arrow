# pydantic-arrow

[![Tests](https://github.com/sam-hey/pydantic-arrow/actions/workflows/tests.yml/badge.svg)](https://github.com/sam-hey/pydantic-arrow/actions/workflows/tests.yml)
[![Lint](https://github.com/sam-hey/pydantic-arrow/actions/workflows/lint.yml/badge.svg)](https://github.com/sam-hey/pydantic-arrow/actions/workflows/lint.yml)
[![Memory](https://github.com/sam-hey/pydantic-arrow/actions/workflows/memory.yml/badge.svg)](https://github.com/sam-hey/pydantic-arrow/actions/workflows/memory.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://github.com/sam-hey/pydantic-arrow)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A Pydantic-native lazy Apache Arrow DataFrame wrapper.

Define your data schema once as a Pydantic model and get a fully typed, lazily-evaluated
Arrow-backed frame -- streaming data batch-by-batch without ever loading the full dataset
into memory.

## Features

- **Pydantic-first** -- derive Arrow schemas directly from `BaseModel` field annotations.
- **Lazy evaluation** -- data flows as `RecordBatch` streams; nothing is materialised until you ask for it.
- **Multiple sources** -- rows/dicts, Parquet files, existing `pa.Table` / `pa.RecordBatchReader`.
- **Rich type mapping** -- covers `str`, `int`, `float`, `bool`, `bytes`, `datetime`, `date`, `time`, `Decimal`, `UUID`, `list[T]`, `dict[K,V]`, `Optional[T]`, `Literal`, `Enum`, and nested `BaseModel` structs.
- **Python 3.11+** with [`typing-inspection`](https://typing-inspection.pydantic.dev/) for safe introspection across Python versions.

## Installation

```bash
pip install pydantic-arrow
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv add pydantic-arrow
```

## Quick start

```python
from pydantic import BaseModel
from pydantic_arrow import ArrowFrame


class User(BaseModel):
    name: str
    age: int
    email: str | None = None


# Build from rows -- never loaded all at once
users = ArrowFrame[User].from_rows([
    {"name": "Alice", "age": 30},
    {"name": "Bob",   "age": 25, "email": "bob@example.com"},
])

# Iterate lazily -- yields validated User instances, batch-by-batch
for user in users:
    print(user.name, user.age)

# Collect everything (materialises into memory)
all_users: list[User] = users.collect()

# Single row / slice (only materialises what you ask for)
first: User           = users[0]
page:  ArrowFrame     = users[10:20]
```

## Reading Parquet files

```python
# Streams row-groups lazily -- the file is never fully loaded
users = ArrowFrame[User].from_parquet("users.parquet")

for user in users:           # validated User instances, one batch at a time
    process(user)

users.to_parquet("copy.parquet")  # incremental write, also lazy
```

## Wrapping existing Arrow data

```python
import pyarrow as pa

# from a pa.Table
table: pa.Table = ...
users = ArrowFrame[User].from_arrow(table)

# from a pa.RecordBatchReader (one-shot stream)
reader: pa.RecordBatchReader = ...
users = ArrowFrame[User].from_arrow(reader)
```

## Low-level batch access

```python
for batch in users.iter_batches():   # pa.RecordBatch
    ...
```

## Exporting

```python
table = users.to_arrow()           # materialise as pa.Table
rows  = users.collect()            # list[User]
users.to_parquet("out.parquet")    # streaming write, no full materialisation
```

## Schema

The Arrow schema is derived automatically from the Pydantic model:

```python
from pydantic_arrow import model_to_schema

print(model_to_schema(User))
# name: string not null
# age: int64 not null
# email: string
```

## Type mapping

| Python / Pydantic | Arrow |
|---|---|
| `str` | `pa.utf8()` |
| `int` | `pa.int64()` |
| `float` | `pa.float64()` |
| `bool` | `pa.bool_()` |
| `bytes` | `pa.binary()` |
| `datetime` | `pa.timestamp("us")` |
| `date` | `pa.date32()` |
| `time` | `pa.time64("us")` |
| `Decimal` (with `max_digits`/`decimal_places`) | `pa.decimal128(p, s)` |
| `UUID` | `pa.utf8()` (stored as canonical UUID string) |
| `list[X]` | `pa.list_(arrow(X))` |
| `dict[K, V]` | `pa.map_(arrow(K), arrow(V))` |
| `T \| None` / `Optional[T]` | same type, nullable |
| `Literal["a", "b"]` | `pa.dictionary(pa.int32(), pa.utf8())` |
| `Enum(str)` | `pa.dictionary(pa.int32(), pa.utf8())` |
| `Enum(int)` | `pa.int64()` |
| nested `BaseModel` | `pa.struct(...)` (recursive) |

## Development

```bash
# Install all dev dependencies
make install

# Run tests
make test

# Run tests in parallel
pytest -n auto

# Auto-fill/update inline snapshots
pytest --inline-snapshot=fix
```

## Memory guarantees

`ArrowFrame` is designed so that at most one `RecordBatch` (one Parquet row group) is
live in the Arrow memory pool at any time during iteration.  This is verified by two
complementary test strategies in `tests/test_memory.py`:

- **`pa.total_allocated_bytes()` assertions** (always active): prove that the Arrow
  C++ allocator peak stays well below the full table size.
- **`pytest-memray` `limit_memory` markers** (enforced with `--memray`): set a hard
  2 MB cap on tests that stream a 200 K-row / 5+ MB dataset.
