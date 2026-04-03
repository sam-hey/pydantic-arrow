"""ArrowFrame -- a lazy, Pydantic-typed Apache Arrow DataFrame."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar, overload

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from pydantic import BaseModel

from pydantic_arrow._convert import _to_dict, batch_to_models
from pydantic_arrow._schema import model_to_schema
from pydantic_arrow._sources import (
    BatchReaderSource,
    ConcatSource,
    GeneratorSource,
    LazySource,
    ParquetSource,
    RowSource,
    TableSource,
)

__all__ = ["ArrowFrame"]

T = TypeVar("T", bound=BaseModel)

_BATCH_SIZE = 65_536


def _count_source_rows(source: LazySource) -> int:
    """Recursively count rows for sources that support it.

    Supports :class:`TableSource`, :class:`RowSource`, :class:`ParquetSource`
    (via file footer metadata), and :class:`ConcatSource` (by summing
    sub-sources).  Raises :class:`TypeError` for one-shot streaming sources.
    """
    if isinstance(source, TableSource):
        return int(source._table.num_rows)
    if isinstance(source, RowSource):
        return len(source._rows)
    if isinstance(source, ParquetSource):
        return source.num_rows()
    if isinstance(source, ConcatSource):
        return sum(_count_source_rows(s) for s in source._sources)
    raise TypeError("num_rows is not available for streaming sources. Use collect() to materialise first.")


class ArrowFrame(Generic[T]):
    """A lazy, Pydantic-typed frame backed by Apache Arrow RecordBatches.

    ``ArrowFrame`` is parameterised by a Pydantic ``BaseModel`` class:

    .. code-block:: python

        class User(BaseModel):
            name: str
            age: int

        users = ArrowFrame[User].from_rows([{"name": "Alice", "age": 30}])

    Data is **never** fully loaded into memory.  Iteration streams
    :class:`pyarrow.RecordBatch` objects one at a time and yields validated
    model instances.

    Class attributes
    ----------------
    __model__
        The Pydantic model class attached when you subscript ``ArrowFrame[Model]``.
    """

    __model__: ClassVar[type[BaseModel]]

    def __class_getitem__(cls, model: Any) -> type[ArrowFrame[Any]]:
        """Bind a Pydantic model to this frame class.

        When *model* is a concrete ``BaseModel`` subclass we create a new
        anonymous subclass that carries ``__model__`` so that classmethods
        like :meth:`from_rows` can derive the Arrow schema automatically.

        For type-checker-only subscriptions (e.g. ``ArrowFrame[T]`` inside a
        function signature) we fall back to the standard Generic behaviour.
        """
        if isinstance(model, type) and issubclass(model, BaseModel):
            return type(
                f"ArrowFrame[{model.__name__}]",
                (cls,),
                {"__model__": model},
            )
        return super().__class_getitem__(model)  # type: ignore[misc, no-any-return]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, source: LazySource) -> None:
        self._source: LazySource = source

    @classmethod
    def from_rows(
        cls,
        rows: list[dict[str, Any] | BaseModel],
        *,
        batch_size: int = _BATCH_SIZE,
    ) -> ArrowFrame[Any]:
        """Create a frame from an iterable of dicts or Pydantic model instances.

        Rows are chunked into :class:`pyarrow.RecordBatch` objects of at most
        *batch_size* rows; the Arrow schema is derived from ``cls.__model__``.

        Args:
            rows: A list of dicts or ``BaseModel`` instances.
            batch_size: Maximum rows per record batch.

        Returns:
            A new :class:`ArrowFrame`.
        """
        model = cls._require_model()
        schema = model_to_schema(model)
        # Validate each row through the model before serialising.  This applies
        # Pydantic default values for fields that are absent from a partial dict,
        # ensuring non-nullable Arrow columns are never populated with null.
        normalised = [_to_dict(r if isinstance(r, BaseModel) else model.model_validate(r)) for r in rows]
        source = RowSource(normalised, schema, batch_size=batch_size)
        return cls(source)

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        *,
        batch_size: int = _BATCH_SIZE,
        columns: list[str] | None = None,
    ) -> ArrowFrame[Any]:
        """Create a lazy frame from a Parquet file.

        The file is **not** read until iteration begins.

        Args:
            path: Path to the Parquet file.
            batch_size: Maximum rows per batch (default 65 536).
            columns: Optional column projection.

        Returns:
            A new :class:`ArrowFrame`.
        """
        cls._require_model()
        source = ParquetSource(path, batch_size=batch_size, columns=columns)
        return cls(source)

    @classmethod
    def from_arrow(
        cls,
        data: pa.Table | pa.RecordBatch | pa.RecordBatchReader,
        *,
        batch_size: int = _BATCH_SIZE,
    ) -> ArrowFrame[Any]:
        """Wrap existing Arrow data in a lazy frame.

        Args:
            data: A :class:`pyarrow.Table`, :class:`pyarrow.RecordBatch`, or
                  :class:`pyarrow.RecordBatchReader`.
            batch_size: Maximum rows per batch when wrapping a Table.

        Returns:
            A new :class:`ArrowFrame`.
        """
        cls._require_model()
        if isinstance(data, pa.RecordBatchReader):
            source: LazySource = BatchReaderSource(data)
        elif isinstance(data, pa.RecordBatch):
            source = TableSource(pa.Table.from_batches([data]), batch_size=batch_size)
        else:
            source = TableSource(data, batch_size=batch_size)
        return cls(source)

    @classmethod
    def from_iterable(
        cls,
        rows: Iterable[dict[str, Any]],
        *,
        batch_size: int = _BATCH_SIZE,
    ) -> ArrowFrame[Any]:
        """Create a one-shot lazy frame from an unbounded iterable of plain dicts.

        Each dict is validated with ``model.model_validate(row)`` inside the
        source at iteration time — the caller does not need to perform
        validation.

        .. warning::
            This source is **not** replayable.  Once the iterable is exhausted
            a second iteration yields nothing.

        Args:
            rows: Any iterable of plain ``dict`` objects (e.g. a DB cursor
                  generator expression).
            batch_size: Maximum rows per record batch.

        Returns:
            A new :class:`ArrowFrame` backed by a :class:`GeneratorSource`.
        """
        model = cls._require_model()
        schema = model_to_schema(model)
        source = GeneratorSource(rows, model, schema, batch_size=batch_size)
        return cls(source)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        batch_size: int = _BATCH_SIZE,
    ) -> ArrowFrame[Any]:
        """Create a lazy frame from a CSV file.

        The CSV is read lazily; Arrow infers column types from the file content.

        Args:
            path: Path to the CSV file.
            batch_size: Maximum rows per batch.

        Returns:
            A new :class:`ArrowFrame`.
        """
        import pyarrow.csv as pa_csv

        cls._require_model()
        table = pa_csv.read_csv(str(path))
        source = TableSource(table, batch_size=batch_size)
        return cls(source)

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        batch_size: int = _BATCH_SIZE,
    ) -> ArrowFrame[Any]:
        """Create a lazy frame from a newline-delimited JSON file.

        Args:
            path: Path to the JSON Lines file.
            batch_size: Maximum rows per batch.

        Returns:
            A new :class:`ArrowFrame`.
        """
        import pyarrow.json as pa_json

        cls._require_model()
        table = pa_json.read_json(str(path))
        source = TableSource(table, batch_size=batch_size)
        return cls(source)

    @classmethod
    def from_feather(
        cls,
        path: str | Path,
        *,
        batch_size: int = _BATCH_SIZE,
    ) -> ArrowFrame[Any]:
        """Create a lazy frame from a Feather (Arrow IPC file) file.

        Args:
            path: Path to the ``.feather`` file.
            batch_size: Maximum rows per batch.

        Returns:
            A new :class:`ArrowFrame`.
        """
        cls._require_model()
        table = feather.read_table(str(path))
        source = TableSource(table, batch_size=batch_size)
        return cls(source)

    @classmethod
    def from_ipc(
        cls,
        data: bytes | str | Path,
        *,
        batch_size: int = _BATCH_SIZE,
    ) -> ArrowFrame[Any]:
        """Create a replayable frame from Arrow IPC stream bytes or a file path.

        Args:
            data: IPC stream as ``bytes`` (from :meth:`to_ipc`) or a path to an
                  ``.arrow`` IPC stream file.
            batch_size: Maximum rows per batch.

        Returns:
            A new :class:`ArrowFrame`.
        """
        cls._require_model()
        if isinstance(data, str | Path):
            with pa.memory_map(str(data), "rb") as source_file:
                table = ipc.open_stream(source_file).read_all()
        else:
            buf = pa.BufferReader(data)
            table = ipc.open_stream(buf).read_all()
        return cls(TableSource(table, batch_size=batch_size))

    def append(
        self,
        rows: list[dict[str, Any] | BaseModel] | dict[str, Any] | BaseModel,
        *,
        batch_size: int = _BATCH_SIZE,
    ) -> ArrowFrame[Any]:
        """Return a new frame with *rows* appended after the existing data.

        Neither the existing frame nor the new rows are loaded into memory
        until iteration begins.  The result is backed by a :class:`ConcatSource`
        that chains the original source with a :class:`RowSource` for *rows*.

        A single dict or model instance is accepted as well as a list.

        Args:
            rows: One or more dicts or ``BaseModel`` instances to append.
            batch_size: Batch size for the new :class:`RowSource`.

        Returns:
            A new :class:`ArrowFrame` of the same type.
        """
        if isinstance(rows, dict | BaseModel):
            rows = [rows]
        model = self._require_model()
        schema = model_to_schema(model)
        normalised = [_to_dict(r) for r in rows]
        extra = RowSource(normalised, schema, batch_size=batch_size)
        return type(self)(ConcatSource([self._source, extra]))

    def extend(
        self,
        rows: list[dict[str, Any] | BaseModel],
        *,
        batch_size: int = _BATCH_SIZE,
    ) -> None:
        """Extend this frame **in place** with additional rows.

        Equivalent to ``my_list.extend([...])``.  The existing source is
        chained with a new :class:`RowSource` via :class:`ConcatSource`; no
        data is loaded until the next iteration.

        Args:
            rows: A list of dicts or ``BaseModel`` instances to append.
            batch_size: Batch size for the new :class:`RowSource`.
        """
        model = self._require_model()
        schema = model_to_schema(model)
        normalised = [_to_dict(r) for r in rows]
        extra = RowSource(normalised, schema, batch_size=batch_size)
        self._source = ConcatSource([self._source, extra])

    def __add__(
        self,
        other: ArrowFrame[Any] | list[dict[str, Any] | BaseModel],
    ) -> ArrowFrame[Any]:
        """Return a new frame that is the concatenation of this frame and *other*.

        Equivalent to ``new_frame = frame + rows`` or ``new_frame = frame + other_frame``.

        Args:
            other: Another :class:`ArrowFrame` or a list of dicts / model instances.

        Returns:
            A new :class:`ArrowFrame` backed by a :class:`ConcatSource`.
        """
        if isinstance(other, ArrowFrame):
            return type(self)(ConcatSource([self._source, other._source]))
        return self.append(other)

    def __iadd__(
        self,
        other: ArrowFrame[Any] | list[dict[str, Any] | BaseModel],
    ) -> ArrowFrame[Any]:
        """Extend this frame **in place** and return ``self``.

        Equivalent to ``frame += rows`` or ``frame += other_frame``.

        Args:
            other: Another :class:`ArrowFrame` or a list of dicts / model instances.

        Returns:
            ``self`` after the in-place extension.
        """
        if isinstance(other, ArrowFrame):
            self._source = ConcatSource([self._source, other._source])
        else:
            self.extend(other)
        return self

    # ------------------------------------------------------------------
    # Schema / metadata
    # ------------------------------------------------------------------

    @property
    def schema(self) -> pa.Schema:
        """The Arrow schema for this frame's data."""
        return self._source.schema

    @property
    def num_rows(self) -> int:
        """Total row count.

        Supported for :class:`TableSource`, :class:`RowSource`, and any
        :class:`ConcatSource` whose sub-sources all support it.
        Raises :class:`TypeError` for one-shot streaming sources.
        """
        return _count_source_rows(self._source)

    # ------------------------------------------------------------------
    # Lazy iteration
    # ------------------------------------------------------------------

    def iter_batches(self) -> Iterator[pa.RecordBatch]:
        """Yield :class:`pyarrow.RecordBatch` objects lazily, one at a time."""
        yield from self._source.iter_batches()

    def __iter__(self) -> Iterator[Any]:
        """Yield validated Pydantic model instances, streaming batch-by-batch."""
        model = self._require_model()
        for batch in self._source.iter_batches():
            yield from batch_to_models(batch, model)

    def __len__(self) -> int:
        return self.num_rows

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    @overload
    def __getitem__(self, index: int) -> Any: ...

    @overload
    def __getitem__(self, index: slice) -> ArrowFrame[Any]: ...

    def __getitem__(self, index: int | slice) -> Any:
        """Index into the frame.

        - ``frame[0]``       -- returns a single validated model instance.
        - ``frame[10:20]``   -- returns a new :class:`ArrowFrame` backed by a
                                slice of the materialised table.

        .. note::
            Indexing **materialises** only the sliced region of the underlying
            data.  For large streaming sources this requires a full scan.
        """
        table = self.to_arrow()
        if isinstance(index, int):
            row = table.slice(index, 1).to_pylist()
            if not row:
                raise IndexError(f"Index {index} out of range.")
            return self._require_model().model_validate(row[0])
        # slice
        start, stop, step = index.indices(len(table))
        if step != 1:
            raise ValueError("ArrowFrame does not support step != 1 in slices.")
        sliced = table.slice(start, stop - start)
        return type(self)(TableSource(sliced))

    # ------------------------------------------------------------------
    # Materialisation / export
    # ------------------------------------------------------------------

    def collect(self) -> list[Any]:
        """Materialise the entire frame as a list of Pydantic model instances.

        .. warning::
            This loads all data into memory.  Use iteration (``for row in frame``)
            for large datasets.
        """
        return list(self)

    def to_arrow(self) -> pa.Table:
        """Materialise the frame as a :class:`pyarrow.Table`.

        .. warning::
            This loads all data into memory.
        """
        return pa.Table.from_batches(
            list(self._source.iter_batches()),
            schema=self._source.schema,
        )

    def to_parquet(self, path: str | Path, **kwargs: Any) -> None:
        """Write the frame to a Parquet file without fully loading it into memory.

        Batches are written incrementally using a
        :class:`pyarrow.parquet.ParquetWriter`.

        Args:
            path: Destination file path.
            **kwargs: Additional keyword arguments forwarded to
                      :class:`pyarrow.parquet.ParquetWriter`.
        """
        writer: pq.ParquetWriter | None = None
        try:
            for batch in self._source.iter_batches():
                if writer is None:
                    writer = pq.ParquetWriter(path, batch.schema, **kwargs)
                writer.write_batch(batch)
        finally:
            if writer is not None:
                writer.close()

    def to_csv(self, path: str | Path, **kwargs: Any) -> None:
        """Write the frame to a CSV file.

        Args:
            path: Destination file path.
            **kwargs: Additional keyword arguments forwarded to
                      :func:`pyarrow.csv.write_csv`.
        """
        import pyarrow.csv as pa_csv

        table = self.to_arrow()
        pa_csv.write_csv(table, str(path), **kwargs)

    def to_feather(self, path: str | Path, **kwargs: Any) -> None:
        """Write the frame to a Feather (Arrow IPC file) file.

        Args:
            path: Destination file path.
            **kwargs: Additional keyword arguments forwarded to
                      :func:`pyarrow.feather.write_feather`.
        """
        table = self.to_arrow()
        feather.write_feather(table, str(path), **kwargs)

    def to_ipc(self, path: str | Path | None = None) -> bytes | None:
        """Serialize the frame as an Arrow IPC stream.

        When *path* is ``None`` the serialized bytes are returned.
        When *path* is provided the data is written to the file and
        ``None`` is returned.

        Args:
            path: Optional destination file path.  If omitted, bytes are
                  returned instead.

        Returns:
            IPC stream as ``bytes`` when *path* is ``None``, else ``None``.
        """
        table = self.to_arrow()
        if path is not None:
            with pa.OSFile(str(path), "wb") as sink, ipc.new_stream(sink, table.schema) as writer:
                writer.write_table(table)
            return None
        sink = pa.BufferOutputStream()
        with ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        return sink.getvalue().to_pybytes()  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Compute operations
    # ------------------------------------------------------------------

    def filter(
        self,
        predicate: pc.Expression | Callable[[Any], bool],
    ) -> ArrowFrame[Any]:
        """Return a new frame containing only rows that satisfy *predicate*.

        Two forms are accepted:

        - **Arrow expression** (efficient, applied at the Arrow level per batch):

          .. code-block:: python

              import pyarrow.compute as pc
              frame.filter(pc.field("age") > 30)

        - **Python callable** (model-level predicate, applied after validation):

          .. code-block:: python

              frame.filter(lambda user: user.age > 30)

        Args:
            predicate: An :class:`pyarrow.compute.Expression` or a Python
                       callable that accepts a validated model instance and
                       returns ``bool``.

        Returns:
            A new :class:`ArrowFrame` with only matching rows.
        """
        if callable(predicate) and not isinstance(predicate, pc.Expression):
            return self._filter_callable(predicate)
        return self._filter_expr(predicate)

    def _filter_expr(self, expr: pc.Expression) -> ArrowFrame[Any]:
        batches = [batch.filter(expr) for batch in self._source.iter_batches()]
        table = pa.Table.from_batches(batches, schema=self._source.schema)
        return type(self)(TableSource(table))

    def _filter_callable(self, fn: Callable[[Any], bool]) -> ArrowFrame[Any]:
        model = self._require_model()
        kept: list[dict[str, Any]] = []
        schema = self._source.schema
        for batch in self._source.iter_batches():
            for instance in batch_to_models(batch, model):
                if fn(instance):
                    kept.append(_to_dict(instance))
        source = RowSource(kept, schema)
        return type(self)(source)

    def limit(self, n: int) -> ArrowFrame[Any]:
        """Return a new frame with at most *n* rows, taken from the start.

        The operation is lazy: batch iteration stops as soon as *n* rows have
        been collected.

        Args:
            n: Maximum number of rows to keep.

        Returns:
            A new :class:`ArrowFrame`.
        """
        if n == 0:
            return type(self)(TableSource(pa.table(self._source.schema.empty_table())))
        collected: list[pa.RecordBatch] = []
        remaining = n
        for batch in self._source.iter_batches():
            if batch.num_rows <= remaining:
                collected.append(batch)
                remaining -= batch.num_rows
            else:
                collected.append(batch.slice(0, remaining))
                remaining = 0
            if remaining == 0:
                break
        table = pa.Table.from_batches(collected, schema=self._source.schema)
        return type(self)(TableSource(table))

    def head(self, n: int) -> ArrowFrame[Any]:
        """Return a new frame with the first *n* rows.

        Alias for :meth:`limit`.

        Args:
            n: Number of rows to take from the start.

        Returns:
            A new :class:`ArrowFrame`.
        """
        return self.limit(n)

    def tail(self, n: int) -> ArrowFrame[Any]:
        """Return a new frame with the last *n* rows.

        .. note::
            This method materialises the full frame to determine the end.

        Args:
            n: Number of rows to take from the end.

        Returns:
            A new :class:`ArrowFrame`.
        """
        if n == 0:
            return type(self)(TableSource(pa.table(self._source.schema.empty_table())))
        table = self.to_arrow()
        sliced = table.slice(max(0, table.num_rows - n))
        return type(self)(TableSource(sliced))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _require_model(cls) -> type[BaseModel]:
        model: type[BaseModel] | None = getattr(cls, "__model__", None)
        if model is None:
            raise TypeError(
                "ArrowFrame must be parameterised with a Pydantic model. "
                "Use ArrowFrame[MyModel].from_rows(...) instead of ArrowFrame.from_rows(...)."
            )
        return model

    def __repr__(self) -> str:
        model = getattr(self.__class__, "__model__", None)
        model_name = model.__name__ if model else "?"
        source_type = type(self._source).__name__
        return f"ArrowFrame[{model_name}]({source_type})"
