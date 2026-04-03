"""pydantic-arrow -- Pydantic-native lazy Apache Arrow DataFrame wrapper."""

from pydantic_arrow._convert import batch_to_models, models_to_batch, rows_to_batches
from pydantic_arrow._frame import ArrowFrame
from pydantic_arrow._schema import model_to_schema
from pydantic_arrow._sources import (
    BatchReaderSource,
    GeneratorSource,
    LazySource,
    ParquetSource,
    RowSource,
    TableSource,
)
from pydantic_arrow._types import python_type_to_arrow

__all__ = [
    "ArrowFrame",
    "BatchReaderSource",
    "GeneratorSource",
    # sources
    "LazySource",
    "ParquetSource",
    "RowSource",
    "TableSource",
    # convert
    "batch_to_models",
    # schema
    "model_to_schema",
    "models_to_batch",
    "python_type_to_arrow",
    "rows_to_batches",
]
