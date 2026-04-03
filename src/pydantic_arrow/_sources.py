"""Lazy data sources that produce streams of :class:`pyarrow.RecordBatch` objects."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

__all__ = [
    "BatchReaderSource",
    "ConcatSource",
    "GeneratorSource",
    "LazySource",
    "ParquetSource",
    "RowSource",
    "TableSource",
]

DEFAULT_BATCH_SIZE = 65_536


@runtime_checkable
class LazySource(Protocol):
    """Protocol for all lazy batch sources.

    Implementations must produce :class:`pyarrow.RecordBatch` objects one at a
    time without materialising the full dataset.  Sources that hold replayable
    data (e.g. a list of rows or a Parquet file) must support multiple calls to
    :meth:`iter_batches`; one-shot streams (e.g. a ``RecordBatchReader``) may
    raise :class:`StopIteration` on subsequent calls.
    """

    @property
    def schema(self) -> pa.Schema:
        """The Arrow schema for batches produced by this source."""
        ...

    def iter_batches(self) -> Iterator[pa.RecordBatch]:
        """Yield record batches lazily."""
        ...


# ---------------------------------------------------------------------------
# RowSource
# ---------------------------------------------------------------------------


class RowSource:
    """Lazy source that chunks an iterable of plain dicts into record batches.

    The rows are stored in memory (as a list) to allow replay.  For truly
    unbounded / one-shot iterables, wrap the ``RecordBatchReader`` produced by
    :meth:`to_reader` in a :class:`BatchReaderSource` instead.

    Args:
        rows: An iterable of ``dict`` or ``BaseModel`` instances (as dicts via
              ``model.model_dump()``).
        schema: The Arrow schema to cast each batch to.
        batch_size: Maximum rows per :class:`pyarrow.RecordBatch`.
    """

    def __init__(
        self,
        rows: list[dict[str, Any]],
        schema: pa.Schema,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self._rows = rows
        self._schema = schema
        self._batch_size = batch_size

    @property
    def schema(self) -> pa.Schema:
        return self._schema

    def iter_batches(self) -> Iterator[pa.RecordBatch]:
        it = iter(self._rows)
        while True:
            chunk = list(itertools.islice(it, self._batch_size))
            if not chunk:
                break
            yield pa.RecordBatch.from_pylist(chunk, schema=self._schema)


# ---------------------------------------------------------------------------
# ParquetSource
# ---------------------------------------------------------------------------


class ParquetSource:
    """Lazy source that streams row groups from a Parquet file.

    Args:
        path: Path to the Parquet file.
        batch_size: Maximum rows per :class:`pyarrow.RecordBatch`.
        columns: Optional list of column names to read (projection push-down).
    """

    def __init__(
        self,
        path: str | Path,
        batch_size: int = DEFAULT_BATCH_SIZE,
        columns: list[str] | None = None,
    ) -> None:
        self._path = Path(path)
        self._batch_size = batch_size
        self._columns = columns
        self._pf: pq.ParquetFile | None = None

    def _open(self) -> pq.ParquetFile:
        return pq.ParquetFile(self._path)

    @property
    def schema(self) -> pa.Schema:
        pf = self._open()
        if self._columns:
            full = pf.schema_arrow
            return pa.schema([full.field(name) for name in self._columns])
        return pf.schema_arrow

    @property
    def arrow_schema(self) -> pa.Schema:
        pf = self._open()
        return pf.schema_arrow

    def num_rows(self) -> int:
        """Return the total row count from Parquet file footer metadata (O(1))."""
        return int(self._open().metadata.num_rows)

    def iter_batches(self) -> Iterator[pa.RecordBatch]:
        pf = self._open()
        yield from pf.iter_batches(batch_size=self._batch_size, columns=self._columns)


# ---------------------------------------------------------------------------
# TableSource
# ---------------------------------------------------------------------------


class TableSource:
    """Lazy source wrapping a materialised :class:`pyarrow.Table`.

    This is useful for interoperability with existing Arrow-based code.  The
    table is never copied; batches are produced by ``table.to_batches()``.

    Args:
        table: An existing :class:`pyarrow.Table`.
        batch_size: Maximum rows per yielded :class:`pyarrow.RecordBatch`.
    """

    def __init__(self, table: pa.Table, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        self._table = table
        self._batch_size = batch_size

    @property
    def schema(self) -> pa.Schema:
        return self._table.schema

    def iter_batches(self) -> Iterator[pa.RecordBatch]:
        yield from self._table.to_batches(max_chunksize=self._batch_size)


# ---------------------------------------------------------------------------
# ConcatSource
# ---------------------------------------------------------------------------


class ConcatSource:
    """Lazy source that concatenates multiple sources in sequence.

    Batches from each source are yielded one at a time without materialising
    any source in full.  All sources must share the same Arrow schema.

    Args:
        sources: Two or more :class:`LazySource` implementations to chain.
    """

    def __init__(self, sources: list[LazySource]) -> None:
        if not sources:
            raise ValueError("ConcatSource requires at least one source.")
        self._sources = sources

    @property
    def schema(self) -> pa.Schema:
        return self._sources[0].schema

    def iter_batches(self) -> Iterator[pa.RecordBatch]:
        for source in self._sources:
            yield from source.iter_batches()


# ---------------------------------------------------------------------------
# BatchReaderSource
# ---------------------------------------------------------------------------


class BatchReaderSource:
    """One-shot lazy source wrapping a :class:`pyarrow.RecordBatchReader`.

    .. warning::
        This source is **not** replayable.  Once the reader is exhausted, a
        second call to :meth:`iter_batches` will yield nothing.

    Args:
        reader: An existing :class:`pyarrow.RecordBatchReader`.
    """

    def __init__(self, reader: pa.RecordBatchReader) -> None:
        self._reader = reader
        self._exhausted = False

    @property
    def schema(self) -> pa.Schema:
        return self._reader.schema

    def iter_batches(self) -> Iterator[pa.RecordBatch]:
        if self._exhausted:
            return
        self._exhausted = True
        yield from self._reader


# ---------------------------------------------------------------------------
# GeneratorSource
# ---------------------------------------------------------------------------


class GeneratorSource:
    """One-shot lazy source that validates and batches an unbounded iterable of dicts.

    Accepts any :class:`~collections.abc.Iterable` of plain ``dict`` objects.
    :meth:`model.model_validate` is called on each dict inside
    :meth:`iter_batches` before the row is encoded into an Arrow batch — the
    caller never needs to perform validation manually.

    .. warning::
        This source is **not** replayable.  Once the iterable is exhausted a
        second call to :meth:`iter_batches` yields nothing (same contract as
        :class:`BatchReaderSource`).

    Args:
        rows: Any iterable of plain ``dict`` objects (e.g. a DB cursor generator).
        model: The Pydantic model class used to validate each row.
        schema: The Arrow schema to cast each batch to.
        batch_size: Maximum rows per :class:`pyarrow.RecordBatch`.
    """

    def __init__(
        self,
        rows: Iterable[dict[str, Any]],
        model: type[BaseModel],
        schema: pa.Schema,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self._iter = iter(rows)
        self._model = model
        self._schema = schema
        self._batch_size = batch_size
        self._exhausted = False

    @property
    def schema(self) -> pa.Schema:
        return self._schema

    def iter_batches(self) -> Iterator[pa.RecordBatch]:
        if self._exhausted:
            return
        self._exhausted = True
        while True:
            chunk = list(itertools.islice(self._iter, self._batch_size))
            if not chunk:
                break
            validated = [self._model.model_validate(row).model_dump() for row in chunk]
            yield pa.RecordBatch.from_pylist(validated, schema=self._schema)
