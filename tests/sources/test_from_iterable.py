"""Tests for ArrowFrame.from_iterable().

from_iterable() wraps a user-supplied iterator in a GeneratorSource.  These
tests cover the public contract: laziness, one-shot exhaustion, validation
timing, and UUID round-trip (to confirm coercions propagate end-to-end).
"""

from __future__ import annotations

from uuid import UUID

import pytest
from pydantic import ValidationError

from pydantic_arrow import ArrowFrame
from tests.conftest import SimpleUser

from .conftest import XY

# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------


class TestFromIterableBasic:
    def test_round_trip_yields_correct_models(self):
        rows = [{"x": i, "y": f"v{i}"} for i in range(5)]
        frame = ArrowFrame[XY].from_iterable(iter(rows))
        result = list(frame)
        assert len(result) == 5
        assert all(isinstance(r, XY) for r in result)
        assert result[0].x == 0
        assert result[4].y == "v4"

    def test_accepts_generator_expression(self):
        frame = ArrowFrame[XY].from_iterable({"x": i, "y": str(i)} for i in range(3))
        assert len(list(frame)) == 3

    def test_empty_iterable_produces_empty_frame(self):
        frame = ArrowFrame[XY].from_iterable(iter([]))
        assert list(frame) == []


# ---------------------------------------------------------------------------
# Laziness and one-shot exhaustion
# ---------------------------------------------------------------------------


class TestFromIterableLaziness:
    def test_num_rows_raises_because_generator_is_not_rewindable(self):
        frame = ArrowFrame[XY].from_iterable(iter([{"x": 1, "y": "a"}]))
        with pytest.raises(TypeError):
            _ = frame.num_rows

    def test_one_shot_not_replayable(self):
        """A second iteration after exhaustion silently yields nothing."""
        frame = ArrowFrame[XY].from_iterable(iter([{"x": 1, "y": "a"}]))
        assert len(list(frame)) == 1
        assert list(frame) == []

    def test_iterator_not_consumed_at_construction_time(self):
        """from_iterable() must not pull any rows until iteration begins."""
        consumed: list[int] = []

        def tracking_gen():
            for i in range(5):
                consumed.append(i)
                yield {"name": f"user_{i}", "age": i, "score": float(i), "active": True}

        frame = ArrowFrame[SimpleUser].from_iterable(tracking_gen())
        assert len(consumed) == 0, f"from_iterable consumed {len(consumed)} rows before iteration"
        list(frame)
        assert len(consumed) == 5


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestFromIterableValidation:
    def test_validation_error_surfaces_at_iteration_not_construction(self):
        bad = [{"x": "not-an-int", "y": "ok"}]
        frame = ArrowFrame[XY].from_iterable(iter(bad))
        with pytest.raises(ValidationError):
            list(frame)


# ---------------------------------------------------------------------------
# End-to-end coercion
# ---------------------------------------------------------------------------


class TestFromIterableCoercion:
    def test_uuid_survives_round_trip(self):
        """UUID fields are stored as strings in Arrow and restored as UUID on read."""
        from tests.conftest import ModelWithUUID

        uid = UUID("12345678-1234-5678-1234-567812345678")
        frame = ArrowFrame[ModelWithUUID].from_iterable(iter([{"id": str(uid), "label": "test"}]))
        result = frame.collect()
        assert result[0].id == uid
