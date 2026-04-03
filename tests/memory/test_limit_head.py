"""Memory tests for ArrowFrame.limit(), head(), and the related tail().

Classification
--------------
limit(n) / head(n) — LAZY:
  Stop consuming batches as soon as n rows have been collected.
  Arrow pool is bounded to the batches read before the cutoff.

tail(n) — EAGER:
  Must scan the entire source to find the last n rows.
  Arrow pool peak reflects reading all batches, not just n.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from pydantic_arrow import ArrowFrame
from tests.conftest import SimpleUser

# ---------------------------------------------------------------------------
# limit / head — LAZY
# ---------------------------------------------------------------------------


class TestLimitMemory:
    """limit(n) is lazy: only reads batches needed to satisfy n rows."""

    def test_limit_does_not_scan_full_parquet(self, multigroup_parquet_file):
        """limit(1) on a 200 K-row Parquet file must not load the entire file."""
        path, _total_rows, rows_per_group = multigroup_parquet_file

        before_full = pa.total_allocated_bytes()
        full_table = ArrowFrame[SimpleUser].from_parquet(path).to_arrow()
        full_bytes = pa.total_allocated_bytes() - before_full
        del full_table

        frame = ArrowFrame[SimpleUser].from_parquet(path, batch_size=rows_per_group)
        before_limit = pa.total_allocated_bytes()
        result = list(frame.limit(1))
        limit_bytes = pa.total_allocated_bytes() - before_limit

        assert len(result) == 1
        assert limit_bytes < full_bytes / 5, (
            f"limit(1) allocated {limit_bytes // 1024} KB but full table is "
            f"{full_bytes // 1024} KB.  Full file was likely loaded."
        )

    def test_limit_zero_allocates_nothing(self, multigroup_parquet_file):
        """limit(0) must load no row data at all."""
        path, _total_rows, _rows_per_group = multigroup_parquet_file

        baseline = pa.total_allocated_bytes()
        frame = ArrowFrame[SimpleUser].from_parquet(path)
        result = list(frame.limit(0))
        allocated = pa.total_allocated_bytes() - baseline

        assert result == []
        assert allocated < 512 * 1024, f"limit(0) allocated {allocated // 1024} KB, expected near zero."

    def test_limit_exactly_one_batch(self, multigroup_parquet_file):
        """limit(rows_per_group) reads exactly one row group."""
        path, _total_rows, rows_per_group = multigroup_parquet_file

        before_full = pa.total_allocated_bytes()
        full_table = ArrowFrame[SimpleUser].from_parquet(path).to_arrow()
        full_bytes = pa.total_allocated_bytes() - before_full
        del full_table

        frame = ArrowFrame[SimpleUser].from_parquet(path, batch_size=rows_per_group)
        before = pa.total_allocated_bytes()
        result = list(frame.limit(rows_per_group))
        allocated = pa.total_allocated_bytes() - before

        assert len(result) == rows_per_group
        # One batch is 1/20th of the table; allow 1/5 as headroom
        assert allocated < full_bytes / 5, (
            f"limit(rows_per_group) allocated {allocated // 1024} KB but full table is {full_bytes // 1024} KB."
        )

    @pytest.mark.limit_memory("2 MB")
    def test_limit_single_row_on_large_parquet(self, multigroup_parquet_file):
        """limit(1) on 200 K rows must stay under 2 MB memray budget."""
        path, _total_rows, rows_per_group = multigroup_parquet_file
        frame = ArrowFrame[SimpleUser].from_parquet(path, batch_size=rows_per_group)
        result = list(frame.limit(1))
        assert len(result) == 1


class TestHeadMemory:
    """head(n) is lazy: identical behaviour to limit(n) for reading."""

    def test_head_does_not_scan_full_parquet(self, multigroup_parquet_file):
        """head(rows_per_group) on a large Parquet file reads only one batch."""
        path, _total_rows, rows_per_group = multigroup_parquet_file

        before_full = pa.total_allocated_bytes()
        full_table = ArrowFrame[SimpleUser].from_parquet(path).to_arrow()
        full_bytes = pa.total_allocated_bytes() - before_full
        del full_table

        frame = ArrowFrame[SimpleUser].from_parquet(path, batch_size=rows_per_group)
        before_head = pa.total_allocated_bytes()
        result = list(frame.head(rows_per_group))
        head_bytes = pa.total_allocated_bytes() - before_head

        assert len(result) == rows_per_group
        assert head_bytes < full_bytes / 5, (
            f"head({rows_per_group}) allocated {head_bytes // 1024} KB but full table is {full_bytes // 1024} KB."
        )

    @pytest.mark.limit_memory("2 MB")
    def test_head_one_batch_raw_batches(self, multigroup_parquet_file):
        """head(rows_per_group) raw-batch iteration stays under 2 MB memray budget.

        We iterate iter_batches() (not model instances) to avoid cumulative
        Python-object allocations from model_validate that memray counts
        even after the objects are freed.
        """
        path, _total_rows, rows_per_group = multigroup_parquet_file
        frame = ArrowFrame[SimpleUser].from_parquet(path, batch_size=rows_per_group)
        count = sum(b.num_rows for b in frame.head(rows_per_group).iter_batches())
        assert count == rows_per_group


# ---------------------------------------------------------------------------
# tail — EAGER
# ---------------------------------------------------------------------------


class TestTailMemory:
    """tail(n) materialises the full source before returning the last n rows.

    Because pa.total_allocated_bytes() measures net-live bytes (not peak),
    we cannot reliably prove tail is expensive via a simple before/after delta.
    Instead we verify eagerness structurally: tail must consume every batch.
    """

    def test_tail_scans_all_batches(self, multigroup_parquet_file):
        """tail(n) must consume every batch in the source — full scan."""
        path, total_rows, rows_per_group = multigroup_parquet_file

        batches_seen: list[int] = []
        frame = ArrowFrame[SimpleUser].from_parquet(path, batch_size=rows_per_group)
        original_iter = frame._source.iter_batches

        def _counting_iter():
            for batch in original_iter():
                batches_seen.append(batch.num_rows)
                yield batch

        frame._source.iter_batches = _counting_iter  # type: ignore[method-assign]
        result = list(frame.tail(1))

        assert len(result) == 1
        assert sum(batches_seen) == total_rows, (
            f"tail scanned {sum(batches_seen)} rows, expected {total_rows}. tail did not perform a full scan."
        )

    def test_tail_peak_exceeds_limit_peak(self, multigroup_parquet_file):
        """tail(n) Arrow pool peak is larger than limit(n) for the same n.

        limit stops after the first batch; tail scans the entire source.
        """
        path, _total_rows, rows_per_group = multigroup_parquet_file

        def _measure_peak(method: str, n: int) -> int:
            frame = ArrowFrame[SimpleUser].from_parquet(path, batch_size=rows_per_group)
            peak = [0]
            original = frame._source.iter_batches

            def _tracked():
                for batch in original():
                    current = pa.total_allocated_bytes()
                    if current > peak[0]:
                        peak[0] = current
                    yield batch

            frame._source.iter_batches = _tracked  # type: ignore[method-assign]
            if method == "tail":
                list(frame.tail(n))
            else:
                list(frame.limit(n))
            return peak[0]

        peak_tail = _measure_peak("tail", 1)
        peak_limit = _measure_peak("limit", 1)

        assert peak_tail > peak_limit, (
            f"tail peak ({peak_tail // 1024} KB) should exceed "
            f"limit peak ({peak_limit // 1024} KB) since tail does a full scan."
        )

    def test_tail_returns_correct_last_rows(self, multigroup_parquet_file):
        """Correctness check: tail(n) returns the last n model instances."""
        path, total_rows, rows_per_group = multigroup_parquet_file

        frame = ArrowFrame[SimpleUser].from_parquet(path, batch_size=rows_per_group)
        result = list(frame.tail(3))

        assert len(result) == 3
        # The fixture writes sequential rows; verify last rows are from the end
        all_rows = list(ArrowFrame[SimpleUser].from_parquet(path))
        assert result == all_rows[-3:]
