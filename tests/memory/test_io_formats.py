"""Memory tests for I/O format methods: CSV, JSON, Feather, and IPC.

Classification: EAGER — all four formats load the full table into a TableSource
backed by a pa.Table.  There is no row-group streaming available for these
formats; the full file is decoded at read time.

What we verify:
  1. The allocation scales sensibly with the actual data size (not wildly over-
     allocated due to double-buffering or serialisation artefacts).
  2. write methods (to_csv, to_feather, to_ipc) do not double-allocate — they
     should not need more than ~2x the table size in Arrow memory at once.
  3. pytest-memray limits confirm that small datasets stay within a tight budget.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from pydantic_arrow import ArrowFrame
from tests.conftest import SimpleUser
from tests.memory.conftest import _SCHEMA, _user_rows

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_NUM_ROWS = 5_000  # small: I/O tests are file-based, keep fast


@pytest.fixture()
def rows() -> list[dict[str, Any]]:
    return _user_rows(_NUM_ROWS)


@pytest.fixture()
def source_frame(rows) -> ArrowFrame:
    return ArrowFrame[SimpleUser].from_rows(rows)


@pytest.fixture()
def source_table_bytes(rows) -> int:
    """Bytes of a pa.Table holding the test dataset."""
    tbl = pa.Table.from_pylist(rows, schema=_SCHEMA)
    b = tbl.nbytes
    del tbl
    return b


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


class TestCsvMemory:
    """from_csv and to_csv memory behaviour."""

    def test_from_csv_allocation_bounded(self, source_frame, source_table_bytes, tmp_path: Path):
        """from_csv Arrow pool allocation must not exceed 3x the raw table bytes."""
        path = tmp_path / "data.csv"
        source_frame.to_csv(path)

        before = pa.total_allocated_bytes()
        frame = ArrowFrame[SimpleUser].from_csv(path)
        _ = frame.to_arrow()
        allocated = pa.total_allocated_bytes() - before

        # CSV text overhead means slightly more memory than binary; allow 3x
        assert allocated < source_table_bytes * 3, (
            f"from_csv allocated {allocated // 1024} KB; table is {source_table_bytes // 1024} KB."
        )

    def test_to_csv_does_not_double_allocate(self, source_frame, source_table_bytes, tmp_path: Path):
        """to_csv must not allocate more than 2x the table size."""
        path = tmp_path / "out.csv"
        before = pa.total_allocated_bytes()
        source_frame.to_csv(path)
        allocated = pa.total_allocated_bytes() - before

        assert path.exists()
        assert allocated < source_table_bytes * 2, (
            f"to_csv allocated {allocated // 1024} KB; table is {source_table_bytes // 1024} KB."
        )

    def test_csv_round_trip_row_count(self, source_frame, tmp_path: Path):
        """CSV round-trip preserves row count."""
        path = tmp_path / "rt.csv"
        source_frame.to_csv(path)
        count = sum(b.num_rows for b in ArrowFrame[SimpleUser].from_csv(path).iter_batches())
        assert count == _NUM_ROWS

    @pytest.mark.limit_memory("10 MB")
    def test_csv_round_trip_limit_memory(self, source_frame, tmp_path: Path):
        """CSV round-trip on 5 K rows stays under 10 MB memray budget."""
        path = tmp_path / "rt.csv"
        source_frame.to_csv(path)
        count = sum(b.num_rows for b in ArrowFrame[SimpleUser].from_csv(path).iter_batches())
        assert count == _NUM_ROWS


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


class TestJsonMemory:
    """from_json memory behaviour (no to_json — not supported by pydantic-arrow)."""

    def test_from_json_allocation_bounded(self, source_frame, source_table_bytes, tmp_path: Path):
        """from_json Arrow pool allocation must not exceed 3x the raw table bytes."""
        path = tmp_path / "data.json"
        rows_py = source_frame.to_arrow().to_pylist()
        path.write_text("\n".join(json.dumps(r) for r in rows_py))

        before = pa.total_allocated_bytes()
        frame = ArrowFrame[SimpleUser].from_json(path)
        _ = frame.to_arrow()
        allocated = pa.total_allocated_bytes() - before

        assert allocated < source_table_bytes * 3, (
            f"from_json allocated {allocated // 1024} KB; table is {source_table_bytes // 1024} KB."
        )

    def test_json_row_count(self, source_frame, tmp_path: Path):
        """from_json round-trip preserves row count."""
        path = tmp_path / "data.json"
        rows_py = source_frame.to_arrow().to_pylist()
        path.write_text("\n".join(json.dumps(r) for r in rows_py))
        count = sum(b.num_rows for b in ArrowFrame[SimpleUser].from_json(path).iter_batches())
        assert count == _NUM_ROWS

    @pytest.mark.limit_memory("10 MB")
    def test_from_json_limit_memory(self, source_frame, tmp_path: Path):
        """from_json on 5 K rows stays under 10 MB memray budget."""
        path = tmp_path / "data.json"
        rows_py = source_frame.to_arrow().to_pylist()
        path.write_text("\n".join(json.dumps(r) for r in rows_py))
        count = sum(b.num_rows for b in ArrowFrame[SimpleUser].from_json(path).iter_batches())
        assert count == _NUM_ROWS


# ---------------------------------------------------------------------------
# Feather
# ---------------------------------------------------------------------------


class TestFeatherMemory:
    """from_feather and to_feather memory behaviour."""

    def test_from_feather_allocation_bounded(self, source_frame, source_table_bytes, tmp_path: Path):
        """from_feather Arrow pool allocation must not exceed 2x the table bytes.

        Feather is a binary Arrow format; per-column decompression overhead is minimal.
        """
        path = tmp_path / "data.feather"
        source_frame.to_feather(path)

        before = pa.total_allocated_bytes()
        frame = ArrowFrame[SimpleUser].from_feather(path)
        _ = frame.to_arrow()
        allocated = pa.total_allocated_bytes() - before

        assert allocated < source_table_bytes * 2, (
            f"from_feather allocated {allocated // 1024} KB; table is {source_table_bytes // 1024} KB."
        )

    def test_to_feather_does_not_double_allocate(self, source_frame, source_table_bytes, tmp_path: Path):
        """to_feather must not allocate more than 2x the table size."""
        path = tmp_path / "out.feather"
        before = pa.total_allocated_bytes()
        source_frame.to_feather(path)
        allocated = pa.total_allocated_bytes() - before

        assert path.exists()
        assert allocated < source_table_bytes * 2, (
            f"to_feather allocated {allocated // 1024} KB; table is {source_table_bytes // 1024} KB."
        )

    def test_feather_round_trip_row_count(self, source_frame, tmp_path: Path):
        """Feather round-trip preserves row count."""
        path = tmp_path / "rt.feather"
        source_frame.to_feather(path)
        count = sum(b.num_rows for b in ArrowFrame[SimpleUser].from_feather(path).iter_batches())
        assert count == _NUM_ROWS

    @pytest.mark.limit_memory("5 MB")
    def test_feather_round_trip_limit_memory(self, source_frame, tmp_path: Path):
        """Feather round-trip on 5 K rows stays under 5 MB memray budget."""
        path = tmp_path / "rt.feather"
        source_frame.to_feather(path)
        count = sum(b.num_rows for b in ArrowFrame[SimpleUser].from_feather(path).iter_batches())
        assert count == _NUM_ROWS


# ---------------------------------------------------------------------------
# IPC (bytes and file)
# ---------------------------------------------------------------------------


class TestIpcMemory:
    """from_ipc and to_ipc memory behaviour for both bytes and file paths."""

    def test_to_ipc_bytes_does_not_double_allocate(self, source_frame, source_table_bytes):
        """to_ipc() → bytes must not allocate more than 2x the table size."""
        before = pa.total_allocated_bytes()
        buf = source_frame.to_ipc()
        allocated = pa.total_allocated_bytes() - before

        assert isinstance(buf, bytes)
        assert allocated < source_table_bytes * 2, (
            f"to_ipc (bytes) allocated {allocated // 1024} KB; table is {source_table_bytes // 1024} KB."
        )

    def test_from_ipc_bytes_allocation_bounded(self, source_frame, source_table_bytes):
        """from_ipc(bytes) allocation must not exceed 2x the table size."""
        buf = source_frame.to_ipc()

        before = pa.total_allocated_bytes()
        frame = ArrowFrame[SimpleUser].from_ipc(buf)
        _ = frame.to_arrow()
        allocated = pa.total_allocated_bytes() - before

        assert allocated < source_table_bytes * 2, (
            f"from_ipc (bytes) allocated {allocated // 1024} KB; table is {source_table_bytes // 1024} KB."
        )

    def test_to_ipc_file_does_not_double_allocate(self, source_frame, source_table_bytes, tmp_path: Path):
        """to_ipc(path) must not allocate more than 2x the table size."""
        path = tmp_path / "out.arrow"
        before = pa.total_allocated_bytes()
        source_frame.to_ipc(path)
        allocated = pa.total_allocated_bytes() - before

        assert path.exists()
        assert allocated < source_table_bytes * 2, (
            f"to_ipc (file) allocated {allocated // 1024} KB; table is {source_table_bytes // 1024} KB."
        )

    def test_from_ipc_file_allocation_bounded(self, source_frame, source_table_bytes, tmp_path: Path):
        """from_ipc(path) allocation must not exceed 2x the table size."""
        path = tmp_path / "data.arrow"
        source_frame.to_ipc(path)

        before = pa.total_allocated_bytes()
        frame = ArrowFrame[SimpleUser].from_ipc(path)
        _ = frame.to_arrow()
        allocated = pa.total_allocated_bytes() - before

        assert allocated < source_table_bytes * 2, (
            f"from_ipc (file) allocated {allocated // 1024} KB; table is {source_table_bytes // 1024} KB."
        )

    def test_ipc_round_trip_row_count_bytes(self, source_frame):
        """IPC bytes round-trip preserves row count."""
        buf = source_frame.to_ipc()
        count = sum(b.num_rows for b in ArrowFrame[SimpleUser].from_ipc(buf).iter_batches())
        assert count == _NUM_ROWS

    def test_ipc_round_trip_row_count_file(self, source_frame, tmp_path: Path):
        """IPC file round-trip preserves row count."""
        path = tmp_path / "data.arrow"
        source_frame.to_ipc(path)
        count = sum(b.num_rows for b in ArrowFrame[SimpleUser].from_ipc(path).iter_batches())
        assert count == _NUM_ROWS

    @pytest.mark.limit_memory("5 MB")
    def test_ipc_bytes_round_trip_limit_memory(self, source_frame):
        """IPC bytes round-trip on 5 K rows stays under 5 MB memray budget."""
        buf = source_frame.to_ipc()
        count = sum(b.num_rows for b in ArrowFrame[SimpleUser].from_ipc(buf).iter_batches())
        assert count == _NUM_ROWS
