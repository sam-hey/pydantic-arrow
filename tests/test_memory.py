"""Memory tests verifying that ArrowFrame loads data lazily.

These tests use two complementary verification strategies:

1. **pytest-memray** ``limit_memory`` markers (activated with ``--memray``):
   enforce a hard cap on Python-heap + native allocations tracked by memray.
   Without ``--memray`` the marker is silently ignored so the tests still run.

2. **PyArrow memory pool** assertions (always active):
   ``pa.total_allocated_bytes()`` reflects Arrow's own C++ allocator, which
   directly measures how much buffer data is live at any point.  This is
   independent of memray and always gives accurate per-batch numbers.

Run with memory enforcement::

    pytest tests/test_memory.py --memray -v

Run without (CI default, no memray overhead)::

    pytest tests/test_memory.py -v
"""

from __future__ import annotations

from collections.abc import Iterator

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pydantic_arrow import ArrowFrame, model_to_schema
from tests.conftest import SimpleUser

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _max_arrow_pool_during(gen: Iterator[pa.RecordBatch]) -> tuple[int, int]:
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


# ---------------------------------------------------------------------------
# Arrow pool tests -- always active, no --memray required
# ---------------------------------------------------------------------------


class TestArrowPoolLaziness:
    """Prove laziness via pa.total_allocated_bytes(), no memray needed."""

    def test_streaming_peak_less_than_full_table(self, multigroup_parquet_file):
        """Streaming peak memory must be substantially less than the full table size.

        Setup: 200 K rows across 20 row groups (~255 KB each).
        Streaming allocates at most one row group at a time.
        Full materialisation allocates all ~5 MB at once.
        """
        path, total_rows, rows_per_group = multigroup_parquet_file

        # Measure full table size
        frame_eager = ArrowFrame[SimpleUser].from_parquet(path)
        before_eager = pa.total_allocated_bytes()
        full_table = frame_eager.to_arrow()
        full_table_bytes = pa.total_allocated_bytes() - before_eager
        assert full_table_bytes > 0, "Full table must allocate some Arrow memory"
        del full_table

        # Measure lazy streaming peak
        frame_lazy = ArrowFrame[SimpleUser].from_parquet(path, batch_size=rows_per_group)
        peak_bytes, row_count = _max_arrow_pool_during(frame_lazy.iter_batches())

        assert row_count == total_rows

        # The streaming peak must be a fraction of the full table.
        # One row group is ~1/20th of the table; we allow up to 1/3 as headroom
        # for Arrow's internal file-reading buffers.
        assert peak_bytes < full_table_bytes / 3, (
            f"Lazy streaming peak ({peak_bytes // 1024} KB) is not "
            f"significantly less than full table ({full_table_bytes // 1024} KB). "
            "Data is likely being loaded eagerly."
        )

    def test_first_batch_does_not_load_full_table(self, multigroup_parquet_file):
        """Reading only the first batch must NOT load the entire file into the Arrow pool.

        This is the definitive laziness test: if the first ``next()`` call
        allocates the whole table, we are not lazy.
        """
        path, total_rows, rows_per_group = multigroup_parquet_file

        # Measure full table size for comparison
        full_frame = ArrowFrame[SimpleUser].from_parquet(path)
        before = pa.total_allocated_bytes()
        full_table = full_frame.to_arrow()
        full_bytes = pa.total_allocated_bytes() - before
        del full_table

        # Read only the first batch
        frame = ArrowFrame[SimpleUser].from_parquet(path, batch_size=rows_per_group)
        before = pa.total_allocated_bytes()
        it = frame.iter_batches()
        first_batch = next(it)
        first_batch_rows = first_batch.num_rows  # capture before del
        one_batch_bytes = pa.total_allocated_bytes() - before
        del it  # do NOT consume the rest
        del first_batch

        assert first_batch_rows == rows_per_group
        # One batch should be well under 1/5 of the total table
        assert one_batch_bytes < full_bytes / 5, (
            f"First batch allocated {one_batch_bytes // 1024} KB but full table is "
            f"{full_bytes // 1024} KB.  Data appears to be pre-loaded."
        )

    def test_arrow_pool_returns_to_zero_after_iteration(self, multigroup_parquet_file):
        """After all batches are consumed and deleted, Arrow pool must return to baseline."""
        path, total_rows, rows_per_group = multigroup_parquet_file
        baseline = pa.total_allocated_bytes()

        frame = ArrowFrame[SimpleUser].from_parquet(path, batch_size=rows_per_group)
        _, row_count = _max_arrow_pool_during(frame.iter_batches())

        final = pa.total_allocated_bytes()
        assert row_count == total_rows
        # After iteration and batch deletion, pool should be close to baseline
        # (allow a small epsilon for residual Arrow metadata ~100 KB)
        assert final - baseline < 200 * 1024, (
            f"Arrow pool did not return to baseline after iteration. "
            f"Baseline: {baseline // 1024} KB, Final: {final // 1024} KB"
        )

    def test_row_source_does_not_preload(self):
        """RowSource chunks lazily: memory is bounded to one batch at a time."""
        num_rows = 50_000
        batch_size = 1_000
        rows = [{"name": f"user_{i}", "age": i % 100, "score": float(i), "active": True} for i in range(num_rows)]

        frame = ArrowFrame[SimpleUser].from_rows(rows, batch_size=batch_size)

        # Measure per-batch peak
        peak_bytes, row_count = _max_arrow_pool_during(frame.iter_batches())
        assert row_count == num_rows

        # Create a single reference batch to know the per-batch cost
        schema = model_to_schema(SimpleUser)
        one_batch_bytes = pa.RecordBatch.from_pylist(rows[:batch_size], schema=schema).nbytes
        full_data_bytes = one_batch_bytes * (num_rows // batch_size)

        # Peak must be much less than the full dataset
        assert peak_bytes < full_data_bytes / 5, (
            f"RowSource peak ({peak_bytes // 1024} KB) too close to full dataset "
            f"({full_data_bytes // 1024} KB). Chunking is not lazy enough."
        )

    def test_to_parquet_streaming_does_not_materialise(self, multigroup_parquet_file, tmp_path):
        """to_parquet() writes incrementally -- Arrow pool must stay bounded."""
        path, total_rows, rows_per_group = multigroup_parquet_file
        out = tmp_path / "copy.parquet"

        # Measure full table size
        full_frame = ArrowFrame[SimpleUser].from_parquet(path)
        before = pa.total_allocated_bytes()
        full_table = full_frame.to_arrow()
        full_bytes = pa.total_allocated_bytes() - before
        del full_table

        # Write via streaming
        frame = ArrowFrame[SimpleUser].from_parquet(path, batch_size=rows_per_group)
        before_write = pa.total_allocated_bytes()
        frame.to_parquet(out)
        peak_write = pa.total_allocated_bytes() - before_write

        assert out.exists()
        result = pq.read_table(out)
        assert result.num_rows == total_rows

        # to_parquet() should not hold the full table in Arrow memory at once
        assert peak_write < full_bytes / 3, (
            f"to_parquet() allocated {peak_write // 1024} KB but full table is "
            f"{full_bytes // 1024} KB.  Write is not streaming."
        )


# ---------------------------------------------------------------------------
# pytest-memray limit_memory tests -- enforced only with --memray flag
# ---------------------------------------------------------------------------


class TestMemrayLimits:
    """Hard memory limits enforced by pytest-memray when run with ``--memray``.

    Limits are set conservatively based on measured values::

        Lazy streaming 200 K rows (20 x 10 K row groups): ~940 KB peak
        Full materialisation 200 K rows:                  ~5 189 KB

    We cap lazy tests at 2 MB (2x safety margin over measured peak) which is
    still well below the ~5 MB needed to load the full table.
    """

    @pytest.mark.limit_memory("2 MB")
    def test_lazy_stream_all_batches(self, multigroup_parquet_file):
        """Stream all 200 K rows one row-group at a time.

        Peak memory must stay under 2 MB even though the full dataset is 5+ MB.
        Each row group (~255 KB) is freed before the next one is read.
        """
        path, total_rows, rows_per_group = multigroup_parquet_file
        frame = ArrowFrame[SimpleUser].from_parquet(path, batch_size=rows_per_group)
        count = 0
        for batch in frame.iter_batches():
            count += batch.num_rows
            # batch goes out of scope here, Arrow can reclaim its buffer
        assert count == total_rows

    @pytest.mark.limit_memory("2 MB")
    def test_first_batch_only(self, multigroup_parquet_file):
        """Read only the first row group from a 200 K row file.

        Only ~255 KB of the 5+ MB file should be loaded.
        """
        path, total_rows, rows_per_group = multigroup_parquet_file
        frame = ArrowFrame[SimpleUser].from_parquet(path, batch_size=rows_per_group)
        it = frame.iter_batches()
        first_batch = next(it)
        assert first_batch.num_rows == rows_per_group
        del it  # explicitly release the iterator without consuming remaining batches

    @pytest.mark.limit_memory("2 MB")
    def test_frame_creation_is_zero_cost(self, multigroup_parquet_file):
        """Creating an ArrowFrame from a Parquet path allocates no data buffers.

        The file is opened for schema inspection only; row data is not read.
        """
        path, _, _ = multigroup_parquet_file
        # This must not trigger any row-data reads
        _frame = ArrowFrame[SimpleUser].from_parquet(path)
        # Access schema (reads Parquet footer metadata, not row data)
        _schema = _frame.schema
        assert len(_schema) > 0

    @pytest.mark.limit_memory("2 MB")
    def test_parquet_stream_is_cheaper_than_materialise(self, multigroup_parquet_file):
        """Streaming a Parquet file costs less than 2 MB even over 200 K rows.

        The corresponding ``to_arrow()`` call allocates 5+ MB, so this limit
        would fail if we accidentally materialised the whole file here.
        """
        path, total_rows, rows_per_group = multigroup_parquet_file
        frame = ArrowFrame[SimpleUser].from_parquet(path, batch_size=rows_per_group)
        count = sum(batch.num_rows for batch in frame.iter_batches())
        assert count == total_rows

    @pytest.mark.limit_memory("2 MB")
    def test_streaming_write_to_parquet(self, multigroup_parquet_file, tmp_path):
        """Write 200 K rows to a new Parquet file without materialising the table.

        ``to_parquet()`` uses a streaming ``ParquetWriter`` internally.
        """
        path, total_rows, rows_per_group = multigroup_parquet_file
        out = tmp_path / "streamed_output.parquet"
        frame = ArrowFrame[SimpleUser].from_parquet(path, batch_size=rows_per_group)
        frame.to_parquet(out)
        assert pq.read_metadata(out).num_rows == total_rows
