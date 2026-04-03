"""Tests for frame compute operations: filter, limit, head, tail."""

from __future__ import annotations

import pyarrow.compute as pc
import pytest
from pydantic import BaseModel

from pydantic_arrow import ArrowFrame


class Item(BaseModel):
    name: str
    value: int
    active: bool


ROWS = [
    {"name": "a", "value": 10, "active": True},
    {"name": "b", "value": 20, "active": False},
    {"name": "c", "value": 30, "active": True},
    {"name": "d", "value": 40, "active": False},
    {"name": "e", "value": 50, "active": True},
]


@pytest.fixture
def frame() -> ArrowFrame:
    return ArrowFrame[Item].from_rows(ROWS)


# ---------------------------------------------------------------------------
# filter — Arrow expression
# ---------------------------------------------------------------------------


class TestFilterArrowExpr:
    def test_filter_gt(self, frame):
        result = list(frame.filter(pc.field("value") > 25))
        assert len(result) == 3
        assert all(r.value > 25 for r in result)

    def test_filter_eq(self, frame):
        result = list(frame.filter(pc.field("active") == True))  # noqa: E712
        assert len(result) == 3
        assert all(r.active for r in result)

    def test_filter_no_match(self, frame):
        result = list(frame.filter(pc.field("value") > 999))
        assert result == []

    def test_filter_all_match(self, frame):
        result = list(frame.filter(pc.field("value") >= 0))
        assert len(result) == 5

    def test_filter_compound(self, frame):
        expr = (pc.field("value") > 15) & (pc.field("active") == True)  # noqa: E712
        result = list(frame.filter(expr))
        assert all(r.value > 15 and r.active for r in result)

    def test_filter_returns_new_frame(self, frame):
        filtered = frame.filter(pc.field("value") > 10)
        assert isinstance(filtered, ArrowFrame)
        assert filtered is not frame

    def test_filter_original_unchanged(self, frame):
        frame.filter(pc.field("value") > 99)
        assert len(list(frame)) == 5


# ---------------------------------------------------------------------------
# filter — Python callable
# ---------------------------------------------------------------------------


class TestFilterCallable:
    def test_filter_lambda(self, frame):
        result = list(frame.filter(lambda item: item.value > 25))
        assert len(result) == 3
        assert all(r.value > 25 for r in result)

    def test_filter_callable_no_match(self, frame):
        result = list(frame.filter(lambda item: item.value > 999))
        assert result == []

    def test_filter_callable_all_active(self, frame):
        result = list(frame.filter(lambda item: item.active))
        assert len(result) == 3
        assert all(r.active for r in result)

    def test_filter_callable_returns_new_frame(self, frame):
        filtered = frame.filter(lambda item: item.value > 10)
        assert isinstance(filtered, ArrowFrame)

    def test_filter_callable_string_match(self, frame):
        result = list(frame.filter(lambda item: item.name in ("a", "c")))
        assert {r.name for r in result} == {"a", "c"}


# ---------------------------------------------------------------------------
# limit / head
# ---------------------------------------------------------------------------


class TestLimit:
    def test_limit_basic(self, frame):
        result = list(frame.limit(3))
        assert len(result) == 3

    def test_limit_zero(self, frame):
        """limit(0) must return an empty frame with the correct schema, not raise."""
        limited = frame.limit(0)
        assert list(limited) == []
        assert limited.schema == frame.schema

    def test_limit_larger_than_data(self, frame):
        result = list(frame.limit(100))
        assert len(result) == 5

    def test_limit_returns_new_frame(self, frame):
        limited = frame.limit(2)
        assert isinstance(limited, ArrowFrame)
        assert limited is not frame

    def test_limit_first_rows(self, frame):
        result = list(frame.limit(2))
        assert result[0].name == "a"
        assert result[1].name == "b"

    def test_head_alias(self, frame):
        assert list(frame.head(2)) == list(frame.limit(2))

    def test_head_basic(self, frame):
        result = list(frame.head(3))
        assert len(result) == 3
        assert result[0].value == 10


# ---------------------------------------------------------------------------
# tail
# ---------------------------------------------------------------------------


class TestTail:
    def test_tail_basic(self, frame):
        result = list(frame.tail(2))
        assert len(result) == 2

    def test_tail_last_values(self, frame):
        result = list(frame.tail(2))
        assert result[-1].name == "e"
        assert result[-1].value == 50

    def test_tail_zero(self, frame):
        """tail(0) must return an empty frame with the correct schema, not raise."""
        tailed = frame.tail(0)
        assert list(tailed) == []
        assert tailed.schema == frame.schema

    def test_tail_larger_than_data(self, frame):
        result = list(frame.tail(100))
        assert len(result) == 5

    def test_tail_returns_new_frame(self, frame):
        tailed = frame.tail(2)
        assert isinstance(tailed, ArrowFrame)

    def test_tail_correct_order(self, frame):
        result = list(frame.tail(3))
        names = [r.name for r in result]
        assert names == ["c", "d", "e"]


# ---------------------------------------------------------------------------
# chaining
# ---------------------------------------------------------------------------


class TestChaining:
    def test_filter_then_limit(self, frame):
        result = list(frame.filter(pc.field("active") == True).limit(2))  # noqa: E712
        assert len(result) == 2
        assert all(r.active for r in result)

    def test_limit_then_filter(self, frame):
        result = list(frame.limit(3).filter(pc.field("value") > 15))
        assert all(r.value > 15 for r in result)
        assert len(result) == 2  # b(20) and c(30) from first 3
