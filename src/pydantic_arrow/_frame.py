"""ArrowFrame -- a lazy, Pydantic-typed Apache Arrow DataFrame."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar, overload

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from pydantic_arrow._convert import batch_to_models
from pydantic_arrow._schema import model_to_schema
from pydantic_arrow._sources import (
    BatchReaderSource,
    LazySource,
    ParquetSource,
    RowSource,
    TableSource,
)

__all__ = ["ArrowFrame"]

T = TypeVar("T", bound=BaseModel)

_BATCH_SIZE = 65_536


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
        normalised = [r.model_dump() if isinstance(r, BaseModel) else dict(r) for r in rows]
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

        Only available for :class:`TableSource` and :class:`RowSource`-backed
        frames.  Raises :class:`TypeError` for streaming sources.
        """
        if isinstance(self._source, TableSource):
            return int(self._source._table.num_rows)
        if isinstance(self._source, RowSource):
            return len(self._source._rows)
        raise TypeError("num_rows is not available for streaming sources. Use collect() to materialise first.")

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
