"""Stream a CSV out of a .7z archive as Arrow tables, never extracting to disk.

py7zr hands decompressed bytes to a writer object as it produces them. Our
writer buffers ~chunk_bytes, cuts at the last newline, parses that slice with
pyarrow's CSV reader (re-prepending the header), and passes the table to a
callback. Memory stays bounded by one chunk regardless of file size.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path

import py7zr
import py7zr.io
import pyarrow as pa
import pyarrow.csv as pacsv

TableCallback = Callable[[pa.Table], None]
ProgressCallback = Callable[[int, int], None]  # (bytes_decompressed, total_bytes)


class _CsvChunkWriter(py7zr.io.Py7zIO):
    """Py7zIO sink that turns a decompressed byte stream into Arrow tables."""

    def __init__(
        self,
        on_table: TableCallback,
        column_types: dict[str, pa.DataType],
        chunk_bytes: int,
        on_progress: ProgressCallback,
        total_bytes: int,
    ) -> None:
        self._on_table = on_table
        self._convert = pacsv.ConvertOptions(column_types=column_types)
        self._chunk_bytes = chunk_bytes
        self._on_progress = on_progress
        self._total_bytes = total_bytes
        self._buf = bytearray()
        self._header: bytes | None = None
        self._seen = 0

    def write(self, s: bytes | bytearray) -> int:
        self._buf += s
        self._seen += len(s)
        # One write() can carry far more than a chunk, so drain in chunk-sized pieces.
        while len(self._buf) >= self._chunk_bytes:
            if not self._drain(final=False):
                break
        return len(s)

    def close(self) -> None:
        while self._buf:
            self._drain(final=True)

    def _drain(self, final: bool) -> bool:
        """Parse up to ~chunk_bytes of complete lines (everything up to a chunk when final).

        Returns False when no complete line is buffered yet.
        """
        if self._header is None:
            newline = self._buf.find(b"\n")
            if newline < 0:
                if not final:
                    return False
                self._buf.clear()
                return True
            self._header = bytes(self._buf[: newline + 1])
            del self._buf[: newline + 1]
        if final and len(self._buf) <= self._chunk_bytes:
            cut = len(self._buf)
        else:
            # Last newline inside the chunk; if a line is longer than a chunk, take through its end.
            cut = self._buf.rfind(b"\n", 0, self._chunk_bytes) + 1 or self._buf.find(b"\n") + 1
            if cut <= 0:
                if not final:
                    return False
                cut = len(self._buf)
        body = bytes(self._buf[:cut])
        del self._buf[:cut]
        if body.strip():
            table = pacsv.read_csv(io.BytesIO(self._header + body), convert_options=self._convert)
            self._on_table(table)
        self._on_progress(self._seen, self._total_bytes)
        return True

    # py7zr's interface also requires these; the stream is write-only.
    def read(self, size: int | None = None) -> bytes:
        return b""

    def seek(self, offset: int, whence: int = 0) -> int:
        return 0

    def flush(self) -> None:
        return None

    def size(self) -> int:
        return self._seen


class _SingleCsvFactory(py7zr.io.WriterFactory):
    """Routes the archive's CSV member to our chunk writer; ignores anything else."""

    def __init__(self, make_writer: Callable[[], _CsvChunkWriter]) -> None:
        self._make_writer = make_writer
        self.csv_members: list[str] = []

    def create(self, filename: str) -> py7zr.io.Py7zIO:
        if filename.lower().endswith(".csv"):
            self.csv_members.append(filename)
            return self._make_writer()
        return py7zr.io.NullIO()


def csv_uncompressed_size(archive: Path) -> int:
    """Total uncompressed bytes of the CSV member(s), for progress reporting."""
    with py7zr.SevenZipFile(archive, mode="r") as zf:
        return sum(
            info.uncompressed for info in zf.list() if info.filename.lower().endswith(".csv")
        )


def stream_csv_tables(
    archive: Path,
    column_types: dict[str, pa.DataType],
    on_table: TableCallback,
    chunk_bytes: int,
    on_progress: ProgressCallback,
) -> list[str]:
    """Decompress `archive` and call `on_table` for each parsed chunk. Returns CSV member names."""
    total = csv_uncompressed_size(archive)
    factory = _SingleCsvFactory(
        lambda: _CsvChunkWriter(on_table, column_types, chunk_bytes, on_progress, total)
    )
    with py7zr.SevenZipFile(archive, mode="r") as zf:
        zf.extractall(factory=factory)
    if len(factory.csv_members) != 1:
        raise ValueError(
            f"{archive.name}: expected exactly 1 CSV member, got {factory.csv_members}")
    return factory.csv_members
