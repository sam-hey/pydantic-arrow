"""Shared helpers and fixtures for the tests/memory/ package.

All fixtures declared in the parent tests/conftest.py (e.g. multigroup_parquet_file,
parquet_file, tmp_path) are automatically available here via pytest's conftest
inheritance — they do not need to be re-declared.

Helper functions (_peak_pool_during, _user_rows, _user_generator, _SCHEMA)
are importable by individual memory test modules:

    from tests.memory.conftest import _peak_pool_during, _user_rows, _SCHEMA
"""

from __future__ import annotations

from collections.abc import Generator, Iterator
from typing import Any

import pyarrow as pa

from pydantic_arrow import model_to_schema
from tests.conftest import SimpleUser

# ---------------------------------------------------------------------------
# Module-level constants shared across all memory test files
# ---------------------------------------------------------------------------

_SCHEMA = model_to_schema(SimpleUser)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _peak_pool_during(gen: Iterator[pa.RecordBatch]) -> tuple[int, int]:
    """Consume *gen* and return (peak_arrow_bytes, total_rows).

    Measures the Arrow memory-pool high-watermark across all batches.
    Each batch is immediately deleted after measurement so the pool can reclaim
    its memory, proving only a single batch is live at a time.
    """
    peak = 0
    total = 0
    for batch in gen:
        current = pa.total_allocated_bytes()
        if current > peak:
            peak = current
        total += batch.num_rows
        del batch
    return peak, total


def _user_rows(n: int) -> list[dict[str, Any]]:
    """Return a list of *n* plain dicts matching SimpleUser."""
    return [{"name": f"u{i}", "age": i % 100, "score": float(i), "active": i % 2 == 0} for i in range(n)]


def _user_generator(n: int) -> Generator[dict[str, Any], None, None]:
    """Yield *n* SimpleUser-compatible dicts one at a time — never builds the full list."""
    for i in range(n):
        yield {"name": f"u{i}", "age": i % 100, "score": float(i), "active": i % 2 == 0}
