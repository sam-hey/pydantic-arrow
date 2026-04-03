"""Memory tests for GeneratorSource and ArrowFrame.from_iterable.

Classification: LAZY — Arrow pool bounded to one batch at a time.

GeneratorSource pulls rows from the user's iterator on demand, building one
pa.RecordBatch per ``batch_size`` rows and immediately discarding it after the
caller processes it.  The peak Arrow pool therefore equals ~one batch regardless
of how many rows the generator produces in total.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from pydantic_arrow import ArrowFrame
from pydantic_arrow._sources import GeneratorSource
from tests.conftest import SimpleUser
from tests.memory.conftest import _SCHEMA, _peak_pool_during, _user_generator, _user_rows


class TestGeneratorSourceMemory:
    """GeneratorSource is lazy: Arrow pool bounded to one batch at a time."""

    _NUM_ROWS = 50_000
    _BATCH_SIZE = 1_000

    def test_peak_bounded_to_one_batch(self):
        """Peak Arrow memory during GeneratorSource iteration equals ~one batch."""
        source = GeneratorSource(
            _user_generator(self._NUM_ROWS),
            SimpleUser,
            _SCHEMA,
            batch_size=self._BATCH_SIZE,
        )

        one_batch_bytes = pa.RecordBatch.from_pylist(_user_rows(self._BATCH_SIZE), schema=_SCHEMA).nbytes
        full_estimate = one_batch_bytes * (self._NUM_ROWS // self._BATCH_SIZE)

        peak_bytes, row_count = _peak_pool_during(source.iter_batches())

        assert row_count == self._NUM_ROWS
        assert peak_bytes < full_estimate / 5, (
            f"GeneratorSource peak ({peak_bytes // 1024} KB) too close to "
            f"full dataset estimate ({full_estimate // 1024} KB)."
        )

    def test_from_iterable_peak_bounded(self):
        """ArrowFrame.from_iterable keeps Arrow pool bounded to one batch."""
        frame = ArrowFrame[SimpleUser].from_iterable(
            _user_generator(self._NUM_ROWS),
            batch_size=self._BATCH_SIZE,
        )

        one_batch_bytes = pa.RecordBatch.from_pylist(_user_rows(self._BATCH_SIZE), schema=_SCHEMA).nbytes
        full_estimate = one_batch_bytes * (self._NUM_ROWS // self._BATCH_SIZE)

        peak_bytes, row_count = _peak_pool_during(frame.iter_batches())

        assert row_count == self._NUM_ROWS
        assert peak_bytes < full_estimate / 5, (
            f"from_iterable peak ({peak_bytes // 1024} KB) exceeds 1/5 of full "
            f"dataset estimate ({full_estimate // 1024} KB)."
        )

    def test_generator_not_exhausted_at_construction(self):
        """GeneratorSource must NOT consume the iterator during __init__."""
        consumed: list[int] = []

        def tracking_gen():
            for row in _user_generator(100):
                consumed.append(1)
                yield row

        frame = ArrowFrame[SimpleUser].from_iterable(tracking_gen())
        assert len(consumed) == 0, f"from_iterable consumed {len(consumed)} rows before iteration."
        list(frame)
        assert len(consumed) == 100

    def test_pool_returns_to_baseline_after_iteration(self):
        """Arrow pool returns to near-baseline after all GeneratorSource batches consumed."""
        baseline = pa.total_allocated_bytes()

        source = GeneratorSource(
            _user_generator(self._NUM_ROWS),
            SimpleUser,
            _SCHEMA,
            batch_size=self._BATCH_SIZE,
        )
        _peak_pool_during(source.iter_batches())

        final = pa.total_allocated_bytes()
        assert final - baseline < 200 * 1024, (
            f"Arrow pool did not return to baseline after GeneratorSource. "
            f"Baseline: {baseline // 1024} KB, Final: {final // 1024} KB"
        )

    def test_batch_size_controls_generator_peak(self):
        """Smaller batch_size → smaller Arrow peak during GeneratorSource iteration."""
        peak_large, _ = _peak_pool_during(
            GeneratorSource(_user_generator(self._NUM_ROWS), SimpleUser, _SCHEMA, batch_size=10_000).iter_batches()
        )
        peak_small, _ = _peak_pool_during(
            GeneratorSource(_user_generator(self._NUM_ROWS), SimpleUser, _SCHEMA, batch_size=500).iter_batches()
        )

        assert peak_small * 5 < peak_large, (
            f"batch_size=500 peak ({peak_small // 1024} KB) should be "
            f"much less than batch_size=10000 peak ({peak_large // 1024} KB)."
        )

    @pytest.mark.limit_memory("2 MB")
    def test_from_iterable_limit_memory(self):
        """Stream 50 K rows through from_iterable under a 2 MB memray cap."""
        frame = ArrowFrame[SimpleUser].from_iterable(
            _user_generator(self._NUM_ROWS),
            batch_size=self._BATCH_SIZE,
        )
        count = sum(b.num_rows for b in frame.iter_batches())
        assert count == self._NUM_ROWS
