# pydantic-arrow — Agent / Contributor Memory

## Rules
- Pydantic behaviour must never be broken. Any incompatibility must be detected and handled within `pydantic-arrow`, not worked around by the caller.

## Memory testing conventions

Every new public function that touches data **must** have a memory test.
The project uses two complementary strategies — both live in
`tests/memory/**`

### Strategy 1 — Arrow pool assertions (always active)

`pa.total_allocated_bytes()` measures the Arrow C++ allocator.  Take
before/after snapshots and compare to a known reference.

```python
baseline = pa.total_allocated_bytes()
# ... do work ...
after = pa.total_allocated_bytes()
allocated = after - baseline
```

**Important pitfalls:**
- This measures **net live bytes**, not peak.  If you allocate a large
  intermediate and free it before taking the `after` snapshot, `allocated`
  will be zero even though the allocation happened.
- To measure peak, use the `_peak_pool_during` helper (see below) or
  intercept `iter_batches` with a tracking wrapper.

Helper for measuring peak during batch iteration:

```python
def _peak_pool_during(gen: Iterator[pa.RecordBatch]) -> tuple[int, int]:
    peak = 0
    total = 0
    for batch in gen:
        current = pa.total_allocated_bytes()
        if current > peak:
            peak = current
        total += batch.num_rows
        del batch  # free immediately so pool can reclaim
    return peak, total
```

To measure peak for operations that internally call `to_arrow()` (e.g. `tail`,
`filter`), intercept the source's `iter_batches`:

```python
peak = [0]
original = frame._source.iter_batches

def _tracked():
    for batch in original():
        current = pa.total_allocated_bytes()
        if current > peak[0]:
            peak[0] = current
        yield batch

frame._source.iter_batches = _tracked
list(frame.tail(1))
```


**Test recipe for lazy features:**
1. Build a 200 K-row Parquet file (`multigroup_parquet_file` fixture).
2. Measure full-table cost via `to_arrow()`.
3. Run the lazy operation and measure peak via the tracking iterator.
4. Assert `peak < full_bytes / 5` (allows 4× safety margin over one batch).
5. Add a `@pytest.mark.limit_memory("2 MB")` test using `iter_batches()`.
