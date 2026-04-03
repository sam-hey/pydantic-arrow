"""Memory tests for RowSource — chunked in-memory row lists.

RowSource is LAZY in the sense that it chunks its row list into batches of
``batch_size`` rows; only one batch is live in the Arrow pool at a time during
iteration.  The list itself is held in Python memory, so its total footprint is
bounded by the Python list size, not the Arrow buffer size.

Sections
--------
TestRowSourceLaziness  -- pool bounded to one batch during iter_batches()
TestWriteRows          -- to_parquet() streams RowSource without holding full table
TestAppend             -- appending one row and writing stays cheap
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pydantic_arrow import ArrowFrame, model_to_schema
from tests.conftest import SimpleUser
from tests.memory.conftest import _SCHEMA, _peak_pool_during, _user_rows

# ---------------------------------------------------------------------------
# RowSource laziness
# ---------------------------------------------------------------------------


class TestRowSourceLaziness:
    """RowSource chunks lazily: Arrow pool is bounded to one batch at a time."""

    _NUM_ROWS = 50_000
    _BATCH_SIZE = 1_000

    def test_peak_bounded_to_one_batch(self):
        """Arrow pool peak during RowSource iteration equals ~one batch."""
        rows = _user_rows(self._NUM_ROWS)
        frame = ArrowFrame[SimpleUser].from_rows(rows, batch_size=self._BATCH_SIZE)

        one_batch_bytes = pa.RecordBatch.from_pylist(rows[: self._BATCH_SIZE], schema=_SCHEMA).nbytes
        full_data_bytes = one_batch_bytes * (self._NUM_ROWS // self._BATCH_SIZE)

        peak_bytes, row_count = _peak_pool_during(frame.iter_batches())

        assert row_count == self._NUM_ROWS
        assert peak_bytes < full_data_bytes / 5, (
            f"RowSource peak ({peak_bytes // 1024} KB) too close to full dataset "
            f"({full_data_bytes // 1024} KB). Chunking is not lazy enough."
        )

    def test_pool_returns_to_baseline_after_iteration(self):
        """Arrow pool returns to near-baseline after all RowSource batches consumed."""
        rows = _user_rows(self._NUM_ROWS)
        baseline = pa.total_allocated_bytes()

        frame = ArrowFrame[SimpleUser].from_rows(rows, batch_size=self._BATCH_SIZE)
        _peak_pool_during(frame.iter_batches())

        final = pa.total_allocated_bytes()
        assert final - baseline < 200 * 1024, (
            f"Pool did not return to baseline. Baseline: {baseline // 1024} KB, Final: {final // 1024} KB"
        )

    def test_batch_size_controls_arrow_peak(self):
        """Halving batch_size halves the Arrow pool peak during iteration."""
        rows = _user_rows(self._NUM_ROWS)

        peak_large, _ = _peak_pool_during(ArrowFrame[SimpleUser].from_rows(rows, batch_size=10_000).iter_batches())
        peak_small, _ = _peak_pool_during(ArrowFrame[SimpleUser].from_rows(rows, batch_size=1_000).iter_batches())

        # Smaller batch → smaller peak; expect at least 3x difference
        assert peak_small * 3 < peak_large, (
            f"batch_size=1000 peak ({peak_small // 1024} KB) should be "
            f"significantly less than batch_size=10000 peak ({peak_large // 1024} KB)."
        )


# ---------------------------------------------------------------------------
# Writing rows via RowSource + to_parquet
# ---------------------------------------------------------------------------


class TestWriteRows:
    """to_parquet() streaming over RowSource — Arrow pool bounded to one batch."""

    _NUM_ROWS = 200_000
    _BATCH_SIZE = 10_000

    @pytest.fixture()
    def large_rows(self) -> list[dict]:
        return _user_rows(self._NUM_ROWS)

    def test_write_rows_arrow_pool_bounded(self, large_rows, tmp_path):
        """Writing 200 K rows to Parquet keeps Arrow pool under one full-table load."""
        out = tmp_path / "written.parquet"
        schema = model_to_schema(SimpleUser)

        before_full = pa.total_allocated_bytes()
        full_table = pa.Table.from_pylist(large_rows, schema=schema)
        full_table_bytes = pa.total_allocated_bytes() - before_full
        del full_table
        assert full_table_bytes > 0

        frame = ArrowFrame[SimpleUser].from_rows(large_rows, batch_size=self._BATCH_SIZE)
        peak_during_write = [0]
        original_iter = frame._source.iter_batches

        def _tracking_iter():
            for batch in original_iter():
                current = pa.total_allocated_bytes()
                if current > peak_during_write[0]:
                    peak_during_write[0] = current
                yield batch

        frame._source.iter_batches = _tracking_iter  # type: ignore[method-assign]
        frame.to_parquet(out)

        assert pq.read_metadata(out).num_rows == self._NUM_ROWS
        assert peak_during_write[0] < full_table_bytes, (
            f"Write peak ({peak_during_write[0] // 1024} KB) exceeded full table "
            f"({full_table_bytes // 1024} KB). Rows are being buffered entirely."
        )

    def test_write_peak_scales_with_batch_not_total(self, large_rows, tmp_path):
        """Peak Arrow memory when writing scales with batch_size, not total rows."""
        out_small = tmp_path / "small_batch.parquet"
        out_large = tmp_path / "large_batch.parquet"

        def _peak_write(batch_size: int, out) -> int:
            frame = ArrowFrame[SimpleUser].from_rows(large_rows, batch_size=batch_size)
            peak = [0]
            original = frame._source.iter_batches

            def _tracked():
                for batch in original():
                    current = pa.total_allocated_bytes()
                    if current > peak[0]:
                        peak[0] = current
                    yield batch

            frame._source.iter_batches = _tracked  # type: ignore[method-assign]
            frame.to_parquet(out)
            return peak[0]

        peak_small = _peak_write(1_000, out_small)
        peak_large = _peak_write(50_000, out_large)

        assert peak_small * 5 < peak_large, (
            f"peak with batch_size=1000 ({peak_small // 1024} KB) is not "
            f"significantly less than peak with batch_size=50000 ({peak_large // 1024} KB)."
        )
        assert pq.read_metadata(out_small).num_rows == self._NUM_ROWS
        assert pq.read_metadata(out_large).num_rows == self._NUM_ROWS


# ---------------------------------------------------------------------------
# Append + write
# ---------------------------------------------------------------------------


class TestAppend:
    """Appending one row to a large Parquet frame and writing stays cheap."""

    def test_append_single_entry_stays_bounded(self, multigroup_parquet_file, tmp_path):
        """Appending one row does not materialise the full Parquet source."""
        path, total_rows, rows_per_group = multigroup_parquet_file
        out = tmp_path / "appended.parquet"

        before_full = pa.total_allocated_bytes()
        full_table = ArrowFrame[SimpleUser].from_parquet(path).to_arrow()
        full_table_bytes = pa.total_allocated_bytes() - before_full
        del full_table
        assert full_table_bytes > 0

        new_entry = {"name": "new_user", "age": 42, "score": 99.9, "active": True}
        frame = ArrowFrame[SimpleUser].from_parquet(path, batch_size=rows_per_group)
        appended = frame.append(new_entry)

        before_write = pa.total_allocated_bytes()
        appended.to_parquet(out)
        peak_write = pa.total_allocated_bytes() - before_write

        assert pq.read_metadata(out).num_rows == total_rows + 1
        assert peak_write < full_table_bytes, (
            f"Append+write peak ({peak_write // 1024} KB) exceeded full table "
            f"({full_table_bytes // 1024} KB). Full table was materialised."
        )

    @pytest.mark.limit_memory("2 MB")
    def test_append_single_entry_limit_memory(self, multigroup_parquet_file, tmp_path):
        """Append + write to Parquet must stay under 2 MB memray budget."""
        path, total_rows, rows_per_group = multigroup_parquet_file
        out = tmp_path / "appended_memray.parquet"
        new_entry = {"name": "new_user", "age": 42, "score": 99.9, "active": True}
        frame = ArrowFrame[SimpleUser].from_parquet(path, batch_size=rows_per_group)
        frame.append(new_entry).to_parquet(out)
        assert pq.read_metadata(out).num_rows == total_rows + 1
