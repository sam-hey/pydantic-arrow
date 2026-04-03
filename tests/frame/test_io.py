"""Tests for new I/O formats: CSV, JSON, Feather, IPC stream."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from pydantic_arrow import ArrowFrame


class User(BaseModel):
    name: str
    age: int
    score: float


SAMPLE_ROWS = [
    {"name": "Alice", "age": 30, "score": 9.5},
    {"name": "Bob", "age": 25, "score": 7.2},
    {"name": "Carol", "age": 35, "score": 8.8},
]


@pytest.fixture
def user_frame() -> ArrowFrame:
    return ArrowFrame[User].from_rows(SAMPLE_ROWS)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


class TestCsv:
    def test_to_csv_creates_file(self, user_frame, tmp_path: Path):
        path = tmp_path / "out.csv"
        user_frame.to_csv(path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_from_csv_round_trip(self, user_frame, tmp_path: Path):
        path = tmp_path / "users.csv"
        user_frame.to_csv(path)
        frame2 = ArrowFrame[User].from_csv(path)
        result = list(frame2)
        assert len(result) == 3
        assert all(isinstance(r, User) for r in result)

    def test_from_csv_name_values(self, user_frame, tmp_path: Path):
        path = tmp_path / "users.csv"
        user_frame.to_csv(path)
        frame2 = ArrowFrame[User].from_csv(path)
        names = [r.name for r in frame2]
        assert names == ["Alice", "Bob", "Carol"]

    def test_from_csv_ages(self, user_frame, tmp_path: Path):
        path = tmp_path / "users.csv"
        user_frame.to_csv(path)
        frame2 = ArrowFrame[User].from_csv(path)
        ages = [r.age for r in frame2]
        assert ages == [30, 25, 35]

    def test_to_csv_contains_header(self, user_frame, tmp_path: Path):
        path = tmp_path / "users.csv"
        user_frame.to_csv(path)
        content = path.read_text()
        assert "name" in content
        assert "age" in content
        assert "score" in content

    def test_from_csv_is_lazy(self, user_frame, tmp_path: Path):
        path = tmp_path / "users.csv"
        user_frame.to_csv(path)
        # Construction should not raise even before iteration
        frame2 = ArrowFrame[User].from_csv(path)
        assert repr(frame2).startswith("ArrowFrame")


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


class TestJson:
    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        import json

        lines = "\n".join(json.dumps(r) for r in rows)
        path.write_text(lines)

    def test_from_json_basic(self, tmp_path: Path):
        path = tmp_path / "users.json"
        self._write_jsonl(path, SAMPLE_ROWS)
        frame = ArrowFrame[User].from_json(path)
        result = list(frame)
        assert len(result) == 3
        assert all(isinstance(r, User) for r in result)

    def test_from_json_values(self, tmp_path: Path):
        path = tmp_path / "users.json"
        self._write_jsonl(path, SAMPLE_ROWS)
        frame = ArrowFrame[User].from_json(path)
        names = [r.name for r in frame]
        assert names == ["Alice", "Bob", "Carol"]

    def test_from_json_ages(self, tmp_path: Path):
        path = tmp_path / "users.json"
        self._write_jsonl(path, SAMPLE_ROWS)
        frame = ArrowFrame[User].from_json(path)
        ages = [r.age for r in frame]
        assert ages == [30, 25, 35]


# ---------------------------------------------------------------------------
# Feather
# ---------------------------------------------------------------------------


class TestFeather:
    def test_to_feather_creates_file(self, user_frame, tmp_path: Path):
        path = tmp_path / "out.feather"
        user_frame.to_feather(path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_from_feather_round_trip(self, user_frame, tmp_path: Path):
        path = tmp_path / "users.feather"
        user_frame.to_feather(path)
        frame2 = ArrowFrame[User].from_feather(path)
        result = list(frame2)
        assert len(result) == 3
        assert all(isinstance(r, User) for r in result)

    def test_from_feather_values(self, user_frame, tmp_path: Path):
        path = tmp_path / "users.feather"
        user_frame.to_feather(path)
        frame2 = ArrowFrame[User].from_feather(path)
        names = [r.name for r in frame2]
        assert names == ["Alice", "Bob", "Carol"]

    def test_from_feather_ages(self, user_frame, tmp_path: Path):
        path = tmp_path / "users.feather"
        user_frame.to_feather(path)
        frame2 = ArrowFrame[User].from_feather(path)
        ages = [r.age for r in frame2]
        assert ages == [30, 25, 35]

    def test_feather_replayable(self, user_frame, tmp_path: Path):
        path = tmp_path / "users.feather"
        user_frame.to_feather(path)
        frame2 = ArrowFrame[User].from_feather(path)
        first = list(frame2)
        second = list(frame2)
        assert len(first) == len(second) == 3


# ---------------------------------------------------------------------------
# IPC stream
# ---------------------------------------------------------------------------


class TestIpc:
    def test_to_ipc_returns_bytes(self, user_frame):
        buf = user_frame.to_ipc()
        assert isinstance(buf, bytes)
        assert len(buf) > 0

    def test_from_ipc_round_trip(self, user_frame):
        buf = user_frame.to_ipc()
        frame2 = ArrowFrame[User].from_ipc(buf)
        result = list(frame2)
        assert len(result) == 3
        assert all(isinstance(r, User) for r in result)

    def test_from_ipc_values(self, user_frame):
        buf = user_frame.to_ipc()
        frame2 = ArrowFrame[User].from_ipc(buf)
        names = [r.name for r in frame2]
        assert names == ["Alice", "Bob", "Carol"]

    def test_from_ipc_ages(self, user_frame):
        buf = user_frame.to_ipc()
        frame2 = ArrowFrame[User].from_ipc(buf)
        ages = [r.age for r in frame2]
        assert ages == [30, 25, 35]

    def test_to_ipc_file(self, user_frame, tmp_path: Path):
        path = tmp_path / "data.arrow"
        user_frame.to_ipc(path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_from_ipc_file(self, user_frame, tmp_path: Path):
        path = tmp_path / "data.arrow"
        user_frame.to_ipc(path)
        frame2 = ArrowFrame[User].from_ipc(path)
        result = list(frame2)
        assert len(result) == 3

    def test_ipc_replayable(self, user_frame):
        buf = user_frame.to_ipc()
        frame2 = ArrowFrame[User].from_ipc(buf)
        first = list(frame2)
        second = list(frame2)
        assert len(first) == len(second) == 3
