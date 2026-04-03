"""Memory tests for ParquetSource — lazy streaming from Parquet files.

Strategy
--------
Arrow pool assertions (always active):
  pa.total_allocated_bytes() is measured before/after operations.  A peak well
  below the full-table size proves only one row group is live at a time.

pytest-memray limits (only enforced with --memray):
  Hard caps that confirm the lazy implementation stays far below the cost of
  materialising the entire file.

All tests use the ``multigroup_parquet_file`` fixture (200 K rows, 20 row groups
of 10 K rows each, ~5 MB total) declared in tests/conftest.py.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pydantic_arrow import ArrowFrame
from tests.conftest import SimpleUser
from tests.memory.conftest import _peak_pool_during

# ---------------------------------------------------------------------------
# Arrow pool assertions — always active
# ---------------------------------------------------------------------------


class TestParquetSourceLaziness:
    """Prove ParquetSource laziness via pa.total_allocated_bytes()."""

    def test_streaming_peak_less_than_full_table(self, multigroup_parquet_file):
        """Streaming peak must be substantially less than the full table.

        Setup: 200 K rows across 20 row groups (~255 KB each).
        Streaming allocates at most one row group at a time.
        Full materialisation allocates all ~5 MB at once.
        """
        path, total_rows, rows_per_group = multigroup_parquet_file

        before_eager = pa.total_allocated_bytes()
        full_table = ArrowFrame[SimpleUser].from_parquet(path).to_arrow()
        full_table_bytes = pa.total_allocated_bytes() - before_eager
        assert full_table_bytes > 0
        del full_table

        frame_lazy = ArrowFrame[SimpleUser].from_parquet(path, batch_size=rows_per_group)
        peak_bytes, row_count = _peak_pool_during(frame_lazy.iter_batches())

        assert row_count == total_rows
        assert peak_bytes < full_table_bytes / 3, (
            f"Lazy streaming peak ({peak_bytes // 1024} KB) is not "
            f"significantly less than full table ({full_table_bytes // 1024} KB). "
            "Data is likely being loaded eagerly."
        )

    def test_first_batch_does_not_load_full_table(self, multigroup_parquet_file):
        """Reading only the first batch must NOT load the entire file into the Arrow pool."""
        path, _total_rows, rows_per_group = multigroup_parquet_file

        before_full = pa.total_allocated_bytes()
        full_table = ArrowFrame[SimpleUser].from_parquet(path).to_arrow()
        full_bytes = pa.total_allocated_bytes() - before_full
        del full_table

        frame = ArrowFrame[SimpleUser].from_parquet(path, batch_size=rows_per_group)
        before = pa.total_allocated_bytes()
        it = frame.iter_batches()
        first_batch = next(it)
        first_batch_rows = first_batch.num_rows
        one_batch_bytes = pa.total_allocated_bytes() - before
        del it
        del first_batch

        assert first_batch_rows == rows_per_group
        assert one_batch_bytes < full_bytes / 5, (
            f"First batch allocated {one_batch_bytes // 1024} KB but full table is "
            f"{full_bytes // 1024} KB.  Data appears to be pre-loaded."
        )

    def test_arrow_pool_returns_to_baseline_after_full_iteration(self, multigroup_parquet_file):
        """After all batches are consumed and deleted, Arrow pool returns to baseline."""
        path, total_rows, rows_per_group = multigroup_parquet_file
        baseline = pa.total_allocated_bytes()

        frame = ArrowFrame[SimpleUser].from_parquet(path, batch_size=rows_per_group)
        _, row_count = _peak_pool_during(frame.iter_batches())

        final = pa.total_allocated_bytes()
        assert row_count == total_rows
        assert final - baseline < 200 * 1024, (
            f"Arrow pool did not return to baseline after iteration. "
            f"Baseline: {baseline // 1024} KB, Final: {final // 1024} KB"
        )

    def test_frame_creation_allocates_no_row_data(self, multigroup_parquet_file):
        """Creating an ArrowFrame from a Parquet path allocates no row-data buffers.

        Only the file footer (schema + metadata) is read on construction.
        """
        path, _, _ = multigroup_parquet_file

        baseline = pa.total_allocated_bytes()
        frame = ArrowFrame[SimpleUser].from_parquet(path)
        # Touch the schema — triggers a footer read, but no row data
        _ = frame.schema
        allocated = pa.total_allocated_bytes() - baseline

        assert len(frame.schema) > 0
        # Footer metadata is small; allow up to 512 KB
        assert allocated < 512 * 1024, (
            f"Frame creation allocated {allocated // 1024} KB.  Row data may have been pre-loaded."
        )

    def test_to_parquet_streaming_does_not_materialise(self, multigroup_parquet_file, tmp_path):
        """to_parquet() writes incrementally — Arrow pool must stay bounded."""
        path, total_rows, rows_per_group = multigroup_parquet_file
        out = tmp_path / "copy.parquet"

        before_full = pa.total_allocated_bytes()
        full_table = ArrowFrame[SimpleUser].from_parquet(path).to_arrow()
        full_bytes = pa.total_allocated_bytes() - before_full
        del full_table

        frame = ArrowFrame[SimpleUser].from_parquet(path, batch_size=rows_per_group)
        before_write = pa.total_allocated_bytes()
        frame.to_parquet(out)
        peak_write = pa.total_allocated_bytes() - before_write

        assert out.exists()
        assert pq.read_table(out).num_rows == total_rows
        assert peak_write < full_bytes / 3, (
            f"to_parquet() allocated {peak_write // 1024} KB but full table is "
            f"{full_bytes // 1024} KB.  Write is not streaming."
        )

    def test_num_rows_reads_only_metadata(self, multigroup_parquet_file):
        """ParquetSource.num_rows reads Parquet footer metadata — no row data loaded."""
        path, total_rows, _ = multigroup_parquet_file

        baseline = pa.total_allocated_bytes()
        frame = ArrowFrame[SimpleUser].from_parquet(path)
        n = frame.num_rows
        allocated = pa.total_allocated_bytes() - baseline

        assert n == total_rows
        # Metadata read should be negligible
        assert allocated < 512 * 1024, f"num_rows allocated {allocated // 1024} KB; expected near-zero (metadata only)."


# ---------------------------------------------------------------------------
# pytest-memray limits — enforced only with --memray
# ---------------------------------------------------------------------------


class TestParquetMemrayLimits:
    """Hard memory caps for ParquetSource operations.

    Limits are set conservatively based on measured values:
      Lazy streaming 200 K rows (20 x 10 K row groups): ~940 KB peak
      Full materialisation 200 K rows:                  ~5 189 KB
    We cap lazy tests at 2 MB (2x safety margin over measured peak).
    """

    @pytest.mark.limit_memory("2 MB")
    def test_lazy_stream_all_batches(self, multigroup_parquet_file):
        """Stream all 200 K rows one row-group at a time under a 2 MB cap."""
        path, total_rows, rows_per_group = multigroup_parquet_file
        frame = ArrowFrame[SimpleUser].from_parquet(path, batch_size=rows_per_group)
        count = sum(batch.num_rows for batch in frame.iter_batches())
        assert count == total_rows

    @pytest.mark.limit_memory("2 MB")
    def test_first_batch_only(self, multigroup_parquet_file):
        """Read only the first row group — well under 2 MB."""
        path, _total_rows, rows_per_group = multigroup_parquet_file
        frame = ArrowFrame[SimpleUser].from_parquet(path, batch_size=rows_per_group)
        it = frame.iter_batches()
        first_batch = next(it)
        assert first_batch.num_rows == rows_per_group
        del it

    @pytest.mark.limit_memory("2 MB")
    def test_frame_creation_is_zero_cost(self, multigroup_parquet_file):
        """Creating a frame + reading its schema allocates no row data."""
        path, _, _ = multigroup_parquet_file
        frame = ArrowFrame[SimpleUser].from_parquet(path)
        assert len(frame.schema) > 0

    @pytest.mark.limit_memory("2 MB")
    def test_streaming_cheaper_than_materialising(self, multigroup_parquet_file):
        """Streaming 200 K rows costs < 2 MB — loading all at once costs 5+ MB."""
        path, total_rows, rows_per_group = multigroup_parquet_file
        frame = ArrowFrame[SimpleUser].from_parquet(path, batch_size=rows_per_group)
        count = sum(batch.num_rows for batch in frame.iter_batches())
        assert count == total_rows

    @pytest.mark.limit_memory("2 MB")
    def test_streaming_write_to_parquet(self, multigroup_parquet_file, tmp_path):
        """Write 200 K rows to Parquet without materialising the full table."""
        path, total_rows, rows_per_group = multigroup_parquet_file
        out = tmp_path / "streamed.parquet"
        frame = ArrowFrame[SimpleUser].from_parquet(path, batch_size=rows_per_group)
        frame.to_parquet(out)
        assert pq.read_metadata(out).num_rows == total_rows
