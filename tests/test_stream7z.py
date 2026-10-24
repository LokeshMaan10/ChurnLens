"""Tests for streaming CSV parsing out of a .7z archive."""

from pathlib import Path

import py7zr
import pyarrow as pa

from churnlens.data.stream7z import stream_csv_tables

TYPES = {"msno": pa.string(), "date": pa.int32()}


def _make_archive(tmp_path: Path, n_rows: int) -> Path:
    csv = tmp_path / "logs.csv"
    csv.write_text("msno,date\n" + "".join(f"u{i},{20170101 + i % 28}\n" for i in range(n_rows)))
    archive = tmp_path / "logs.csv.7z"
    with py7zr.SevenZipFile(archive, "w") as zf:
        zf.write(csv, "logs.csv")
    csv.unlink()
    return archive


def test_small_chunks_preserve_every_row(tmp_path: Path) -> None:
    archive = _make_archive(tmp_path, 5000)
    tables: list[pa.Table] = []
    members = stream_csv_tables(archive, TYPES, tables.append, chunk_bytes=1000,
                                on_progress=lambda seen, total: None)
    combined = pa.concat_tables(tables)
    assert members == ["logs.csv"]
    assert len(tables) > 1, "chunking should have produced several tables"
    assert combined.num_rows == 5000
    assert combined["msno"].to_pylist() == [f"u{i}" for i in range(5000)]
    assert combined.schema == pa.schema(TYPES)


def test_nothing_is_extracted_to_disk(tmp_path: Path) -> None:
    archive = _make_archive(tmp_path, 100)
    stream_csv_tables(archive, TYPES, lambda t: None, chunk_bytes=64,
                      on_progress=lambda seen, total: None)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["logs.csv.7z"]
