"""Memory tests for ArrowFrame.filter().

Classification: EAGER result, but LAZY per-batch during filtering.

filter() scans source batches one at a time via iter_batches(), applies the
predicate to each batch, and concatenates matching rows into a result table.
The Arrow pool during the scan is bounded to one input batch at a time.
The final pool allocation is proportional to the number of matching rows.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from pydantic_arrow import ArrowFrame
from tests.conftest import SimpleUser


class TestFilterArrowExpressionMemory:
    """filter(pc.Expression) scans batch-by-batch; result is proportional to matches."""

    def test_selective_filter_allocates_less_than_full_table(self, multigroup_parquet_file):
        """A highly selective filter (1% match rate) allocates far less than the full table."""
        path, _total_rows, rows_per_group = multigroup_parquet_file

        before_full = pa.total_allocated_bytes()
        full_table = ArrowFrame[SimpleUser].from_parquet(path).to_arrow()
        full_bytes = pa.total_allocated_bytes() - before_full
        del full_table
        assert full_bytes > 0

        frame = ArrowFrame[SimpleUser].from_parquet(path, batch_size=rows_per_group)
        before_filter = pa.total_allocated_bytes()
        filtered = list(frame.filter(pc.field("age") == 0))
        filter_alloc = pa.total_allocated_bytes() - before_filter

        assert len(filtered) > 0
        assert filter_alloc < full_bytes / 2, (
            f"filter result allocated {filter_alloc // 1024} KB but full table was "
            f"{full_bytes // 1024} KB.  Filter result should be much smaller."
        )

    def test_zero_match_filter_allocates_nothing(self, multigroup_parquet_file):
        """filter with zero matching rows allocates close to nothing."""
        path, _total_rows, rows_per_group = multigroup_parquet_file

        frame = ArrowFrame[SimpleUser].from_parquet(path, batch_size=rows_per_group)
        baseline = pa.total_allocated_bytes()
        result = list(frame.filter(pc.field("age") == 999))
        allocated = pa.total_allocated_bytes() - baseline

        assert result == []
        assert allocated < 512 * 1024, f"Zero-match filter allocated {allocated // 1024} KB, expected near zero."

    def test_full_match_filter_allocates_proportional_to_table(self, multigroup_parquet_file):
        """filter matching all rows allocates ~the full table size."""
        path, total_rows, rows_per_group = multigroup_parquet_file

        before_full = pa.total_allocated_bytes()
        full_table = ArrowFrame[SimpleUser].from_parquet(path).to_arrow()
        full_bytes = pa.total_allocated_bytes() - before_full
        del full_table

        frame = ArrowFrame[SimpleUser].from_parquet(path, batch_size=rows_per_group)
        before = pa.total_allocated_bytes()
        filtered = list(frame.filter(pc.field("age") >= 0))  # all rows
        allocated = pa.total_allocated_bytes() - before

        assert len(filtered) == total_rows
        # Full match → result ≈ full table; should be within 2x (Arrow internals)
        assert allocated < full_bytes * 2, (
            f"Full-match filter allocated {allocated // 1024} KB but full table is {full_bytes // 1024} KB."
        )


class TestFilterCallableMemory:
    """filter(callable) materialises model instances per batch; result proportional to matches."""

    def test_selective_callable_filter_allocates_less_than_full_table(self, multigroup_parquet_file):
        """Callable filter with 1% match rate allocates far less than the full table."""
        path, _total_rows, rows_per_group = multigroup_parquet_file

        before_full = pa.total_allocated_bytes()
        full_table = ArrowFrame[SimpleUser].from_parquet(path).to_arrow()
        full_bytes = pa.total_allocated_bytes() - before_full
        del full_table

        frame = ArrowFrame[SimpleUser].from_parquet(path, batch_size=rows_per_group)
        before_filter = pa.total_allocated_bytes()
        filtered = list(frame.filter(lambda u: u.age == 1))
        filter_alloc = pa.total_allocated_bytes() - before_filter

        assert len(filtered) > 0
        assert filter_alloc < full_bytes / 2, (
            f"callable filter allocated {filter_alloc // 1024} KB but full table was {full_bytes // 1024} KB."
        )

    def test_zero_match_callable_allocates_nothing(self, multigroup_parquet_file):
        """Callable filter returning False for all rows allocates close to nothing."""
        path, _total_rows, rows_per_group = multigroup_parquet_file

        frame = ArrowFrame[SimpleUser].from_parquet(path, batch_size=rows_per_group)
        baseline = pa.total_allocated_bytes()
        result = list(frame.filter(lambda u: u.age > 10_000))
        allocated = pa.total_allocated_bytes() - baseline

        assert result == []
        assert allocated < 512 * 1024, (
            f"Zero-match callable filter allocated {allocated // 1024} KB, expected near zero."
        )

    def test_arrow_expr_vs_callable_similar_allocation(self, multigroup_parquet_file):
        """Arrow expression and callable filters with the same predicate allocate similarly."""
        path, _total_rows, rows_per_group = multigroup_parquet_file

        frame_expr = ArrowFrame[SimpleUser].from_parquet(path, batch_size=rows_per_group)
        before_expr = pa.total_allocated_bytes()
        result_expr = list(frame_expr.filter(pc.field("age") == 5))
        alloc_expr = pa.total_allocated_bytes() - before_expr

        frame_callable = ArrowFrame[SimpleUser].from_parquet(path, batch_size=rows_per_group)
        before_callable = pa.total_allocated_bytes()
        result_callable = list(frame_callable.filter(lambda u: u.age == 5))
        alloc_callable = pa.total_allocated_bytes() - before_callable

        assert len(result_expr) == len(result_callable)
        # Both should produce allocations in the same order of magnitude (within 5x)
        if alloc_expr > 0 and alloc_callable > 0:
            ratio = max(alloc_expr, alloc_callable) / min(alloc_expr, alloc_callable)
            assert ratio < 5, (
                f"Arrow expr ({alloc_expr // 1024} KB) and callable ({alloc_callable // 1024} KB) "
                f"allocations differ by {ratio:.1f}x, expected similar."
            )

    @pytest.mark.limit_memory("5 MB")
    def test_selective_filter_limit_memory(self, multigroup_parquet_file):
        """Highly selective Arrow expression filter on 200 K rows stays under 5 MB."""
        path, _total_rows, rows_per_group = multigroup_parquet_file
        frame = ArrowFrame[SimpleUser].from_parquet(path, batch_size=rows_per_group)
        result = list(frame.filter(pc.field("age") == 0))
        assert len(result) > 0
