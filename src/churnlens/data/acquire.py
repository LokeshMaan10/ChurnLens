"""Phase -1: turn downloaded KKBox .7z archives into sampled, validated raw Parquet.

Run with `python -m churnlens.data.acquire` (or `make data`). Idempotent: an
output whose checksum and sampling settings match data/raw/manifest.json is
skipped. Each output is written to a .tmp path and renamed only after
validation, so an interrupted run never leaves a half-written output behind;
re-running simply redoes the interrupted table.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from churnlens.config import PROJECT_ROOT, date_to_int, load_config, resolve_path
from churnlens.data.sampling import UserSampler
from churnlens.data.stream7z import stream_csv_tables

HASH_BLOCK_BYTES = 8 * 1024 * 1024

# Raw column types per file kind, from the competition's data description.
# Dates stay as KKBox's YYYYMMDD integers in the raw layer; parsing is Phase 1.
COLUMN_TYPES: dict[str, dict[str, pa.DataType]] = {
    "labels": {"msno": pa.string(), "is_churn": pa.int8()},
    "members": {
        "msno": pa.string(),
        "city": pa.int32(),
        "bd": pa.int32(),
        "gender": pa.string(),
        "registered_via": pa.int32(),
        "registration_init_time": pa.int32(),
    },
    "transactions": {
        "msno": pa.string(),
        "payment_method_id": pa.int32(),
        "payment_plan_days": pa.int32(),
        "plan_list_price": pa.int32(),
        "actual_amount_paid": pa.int32(),
        "is_auto_renew": pa.int8(),
        "transaction_date": pa.int32(),
        "membership_expire_date": pa.int32(),
        "is_cancel": pa.int8(),
    },
    "logs": {
        "msno": pa.string(),
        "date": pa.int32(),
        "num_25": pa.int32(),
        "num_50": pa.int32(),
        "num_75": pa.int32(),
        "num_985": pa.int32(),
        "num_100": pa.int32(),
        "num_unq": pa.int32(),
        "total_secs": pa.float64(),
    },
}


class MissingArchiveError(RuntimeError):
    """Raised when an archive is needed but absent and cannot be downloaded."""


@dataclass
class TableStats:
    """Counters accumulated while streaming one archive."""

    rows_read: int = 0
    rows_kept: int = 0
    rows_outside_window: int = 0
    label_rows_full: int = 0
    label_churn_full: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------- helpers


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, read in blocks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(HASH_BLOCK_BYTES):
            h.update(block)
    return h.hexdigest()


def sha256_output(path: Path) -> str:
    """SHA-256 of a file, or of the sorted (relative path, file hash) list for a directory."""
    if path.is_file():
        return sha256_file(path)
    h = hashlib.sha256()
    for f in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(f"{f.relative_to(path).as_posix()}:{sha256_file(f)}\n".encode())
    return h.hexdigest()


def disk_bytes(path: Path) -> int:
    """Size on disk of a file or directory tree."""
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def sampling_fingerprint(cfg: dict[str, Any]) -> dict[str, Any]:
    """The settings that, if changed, invalidate previously filtered outputs."""
    return {
        "mode": cfg["mode"],
        "user_sample_pct": cfg["user_sample_pct"],
        "sample_seed": cfg["sample_seed"],
        "log_window_start": str(cfg["log_window_start"]),
        "log_window_end": str(cfg["log_window_end"]),
    }


def effective_pct(cfg: dict[str, Any]) -> int:
    """Sample percentage actually applied: 100 in full mode."""
    return 100 if cfg["mode"] == "full" else int(cfg["user_sample_pct"])


def load_manifest(path: Path) -> dict[str, Any]:
    """Read manifest.json, or return an empty manifest."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"tables": {}}


def log(msg: str) -> None:
    """Timestamped progress line, flushed immediately."""
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


# ---------------------------------------------------------------- download


def try_cli_download(archive: Path, cfg: dict[str, Any]) -> bool:
    """Download one archive with the Kaggle CLI if a token file exists. Never reads the token."""
    token = resolve_path(cfg["paths"]["kaggle_token"], cfg)
    if not token.exists():
        return False
    for attempt in range(1, cfg["source"]["download_retries"] + 2):
        log(f"kaggle download {archive.name} (attempt {attempt})")
        result = subprocess.run(
            ["kaggle", "competitions", "download", "-c", cfg["source"]["competition"],
             "-f", archive.name, "-p", str(archive.parent)],
            check=False,
        )
        if result.returncode == 0 and archive.exists():
            return True
    return False


def ensure_archive(archive: Path, cfg: dict[str, Any]) -> None:
    """Make sure an archive is present, downloading via CLI if possible."""
    if archive.exists() or try_cli_download(archive, cfg):
        return
    raise MissingArchiveError(
        f"{archive.name} not found in {archive.parent}.\n"
        f"Download it manually from {cfg['source']['url']} (signed in, rules accepted) "
        f"and place it in {archive.parent}, or set up ~/.kaggle/kaggle.json for the CLI."
    )


# ---------------------------------------------------------------- filtering


class _Sink:
    """Collects filtered tables: one Parquet file, or month-partitioned Parquet files."""

    def __init__(self, tmp_out: Path, partition_by_month: bool, compression: str) -> None:
        self.tmp_out = tmp_out
        self.partition_by_month = partition_by_month
        self.compression = compression
        self._tables: list[pa.Table] = []
        self._writers: dict[str, pq.ParquetWriter] = {}

    def add(self, table: pa.Table) -> None:
        if table.num_rows == 0:
            return
        if not self.partition_by_month:
            self._tables.append(table)
            return
        months = pc.cast(pc.divide(table["date"], pa.scalar(100, pa.int32())), pa.int32())
        for month in pc.unique(months).to_pylist():
            self._writer(month).write_table(table.filter(pc.equal(months, month)))

    def _writer(self, yyyymm: int) -> pq.ParquetWriter:
        key = f"month={yyyymm // 100:04d}-{yyyymm % 100:02d}"
        if key not in self._writers:
            path = self.tmp_out / key / "part-0.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            schema = pa.schema(COLUMN_TYPES["logs"])
            self._writers[key] = pq.ParquetWriter(path, schema, compression=self.compression)
        return self._writers[key]

    def close(self, schema: pa.Schema) -> None:
        if self.partition_by_month:
            for w in self._writers.values():
                w.close()
            return
        table = pa.concat_tables(self._tables) if self._tables else schema.empty_table()
        self.tmp_out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, self.tmp_out, compression=self.compression)


def filter_archive(
    archive: Path, tmp_out: Path, spec: dict[str, Any], cfg: dict[str, Any]
) -> TableStats:
    """Stream one archive through the user sample (and log window) into tmp_out."""
    kind = spec["kind"]
    sampler = UserSampler(int(cfg["sample_seed"]), effective_pct(cfg))
    window = (date_to_int(cfg["log_window_start"]), date_to_int(cfg["log_window_end"]))
    apply_window = bool(spec.get("apply_log_window", False))
    sink = _Sink(tmp_out, bool(spec.get("partition_by_month", False)),
                 cfg["stream"]["parquet_compression"])
    stats = TableStats()
    every = int(cfg["stream"]["progress_every_chunks"])
    chunk_no = [0]
    started = time.time()

    def on_table(table: pa.Table) -> None:
        stats.rows_read += table.num_rows
        if kind == "labels":
            stats.label_rows_full += table.num_rows
            stats.label_churn_full += pc.sum(table["is_churn"]).as_py() or 0
        if apply_window:
            in_window = pc.and_(pc.greater_equal(table["date"], window[0]),
                                pc.less_equal(table["date"], window[1]))
            stats.rows_outside_window += table.num_rows - pc.sum(in_window).as_py()
            table = table.filter(in_window)
        table = table.filter(sampler.mask(table["msno"]))
        stats.rows_kept += table.num_rows
        sink.add(table)

    def on_progress(seen: int, total: int) -> None:
        chunk_no[0] += 1
        if chunk_no[0] % every == 0:
            pct = 100 * seen / total if total else 0
            log(f"  {archive.name}: {pct:5.1f}% of {total / 1e9:.2f} GB | "
                f"read {stats.rows_read:,} kept {stats.rows_kept:,} | "
                f"{time.time() - started:,.0f}s")

    stream_csv_tables(archive, COLUMN_TYPES[kind], on_table,
                      int(cfg["stream"]["chunk_bytes"]), on_progress)
    sink.close(pa.schema(COLUMN_TYPES[kind]))
    stats.extra["seconds"] = round(time.time() - started, 1)
    return stats


# ---------------------------------------------------------------- validation


def validate_output(path: Path, spec: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Check columns, non-emptiness, and date ranges. Returns profile facts; raises on failure."""
    dataset = ds.dataset(path, format="parquet", partitioning="hive")
    expected = list(COLUMN_TYPES[spec["kind"]])
    actual = [c for c in dataset.schema.names if c != "month"]
    if actual != expected:
        raise ValueError(f"{path.name}: columns {actual} != expected {expected}")
    table = dataset.to_table(columns=["msno", *spec.get("date_columns", [])])
    if table.num_rows == 0:
        raise ValueError(f"{path.name}: output is empty")
    facts: dict[str, Any] = {"rows": table.num_rows,
                             "unique_msno": pc.count_distinct(table["msno"]).as_py(),
                             "date_ranges": {}}
    for col in spec.get("date_columns", []):
        mm = pc.min_max(table[col])
        facts["date_ranges"][col] = [mm["min"].as_py(), mm["max"].as_py()]
    if spec.get("apply_log_window"):
        lo, hi = facts["date_ranges"]["date"]
        if lo < date_to_int(cfg["log_window_start"]) or hi > date_to_int(cfg["log_window_end"]):
            raise ValueError(f"{path.name}: dates {lo}..{hi} escape the configured log window")
    return facts


# ---------------------------------------------------------------- orchestration


def is_up_to_date(out: Path, entry: dict[str, Any] | None, fingerprint: dict[str, Any]) -> bool:
    """True if the output exists and matches the manifest checksum and sampling settings."""
    if entry is None or not out.exists() or entry.get("fingerprint") != fingerprint:
        return False
    return sha256_output(out) == entry["output"]["sha256"]


def process_table(name: str, spec: dict[str, Any], cfg: dict[str, Any],
                  manifest: dict[str, Any], allow_download: bool) -> None:
    """Filter, validate, and register one table in the manifest (skips if already current)."""
    downloads = resolve_path(cfg["paths"]["downloads"], cfg)
    raw_dir = resolve_path(cfg["paths"]["raw_dir"], cfg)
    out = raw_dir / spec["output"]
    archive = downloads / spec["archive"]
    fingerprint = sampling_fingerprint(cfg)

    if is_up_to_date(out, manifest["tables"].get(name), fingerprint):
        log(f"{name}: up to date, skipping")
        return
    if not archive.exists() and not allow_download:
        raise MissingArchiveError(
            f"{archive.name} missing; download it first ({cfg['source']['url']})")
    ensure_archive(archive, cfg)

    log(f"{name}: hashing archive")
    archive_info = {"file": archive.name, "bytes": archive.stat().st_size,
                    "sha256": sha256_file(archive),
                    "downloaded": dt.date.fromtimestamp(archive.stat().st_mtime).isoformat()}
    tmp_out = out.with_name(out.name + ".tmp")
    if tmp_out.is_dir():
        shutil.rmtree(tmp_out, ignore_errors=True)
    else:
        tmp_out.unlink(missing_ok=True)

    log(f"{name}: streaming {archive.name}")
    stats = filter_archive(archive, tmp_out, spec, cfg)
    facts = validate_output(tmp_out, spec, cfg)
    if out.exists():
        shutil.rmtree(out) if out.is_dir() else out.unlink()
    tmp_out.rename(out)

    entry: dict[str, Any] = {
        "fingerprint": fingerprint,
        "scoring_only": bool(spec.get("scoring_only", False)),
        "archive": archive_info,
        "output": {"path": out.relative_to(PROJECT_ROOT).as_posix(), "bytes": disk_bytes(out),
                   "sha256": sha256_output(out), **facts},
        "rows_read": stats.rows_read, "rows_kept": stats.rows_kept,
        "rows_outside_window": stats.rows_outside_window,
        "seconds": stats.extra["seconds"],
    }
    if spec["kind"] == "labels":
        churn_kept = pc.sum(pq.read_table(out, columns=["is_churn"])["is_churn"]).as_py()
        entry["churn_rate_full"] = stats.label_churn_full / stats.label_rows_full
        entry["churn_rate_sample"] = churn_kept / stats.rows_kept
    manifest["tables"][name] = entry
    log(f"{name}: kept {stats.rows_kept:,}/{stats.rows_read:,} rows -> {out.name} "
        f"({entry['output']['bytes'] / 1e6:.1f} MB)")


def write_manifest(manifest: dict[str, Any], cfg: dict[str, Any]) -> None:
    """Persist manifest.json and render the human-readable MANIFEST.md from it."""
    manifest["settings"] = sampling_fingerprint(cfg)
    manifest["source_url"] = cfg["source"]["url"]
    manifest["updated"] = dt.datetime.now().isoformat(timespec="seconds")
    resolve_path(cfg["paths"]["manifest_json"], cfg).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    resolve_path(cfg["paths"]["manifest_md"], cfg).write_text(render_manifest_md(manifest),
                                                              encoding="utf-8")


def render_manifest_md(manifest: dict[str, Any]) -> str:
    """Markdown view of manifest.json (generated; do not edit by hand)."""
    s = manifest["settings"]
    lines = [
        "# Raw data manifest",
        "",
        "_Generated by `python -m churnlens.data.acquire` from `manifest.json`. "
        "Do not edit by hand._",
        "",
        f"- Source: {manifest['source_url']}",
        f"- Mode: **{s['mode']}**, user sample: **{s['user_sample_pct']}%**, "
        f"seed: {s['sample_seed']}",
        f"- Log window (user_logs only): {s['log_window_start']} to {s['log_window_end']}",
        f"- Last updated: {manifest['updated']}",
        "",
        "## Outputs",
        "",
        "| Table | Output | Rows read | Rows kept | Unique msno | Date range(s) | "
        "Size (MB) | SHA-256 |",
        "|---|---|---:|---:|---:|---|---:|---|",
    ]
    for name, e in manifest["tables"].items():
        o = e["output"]
        ranges = "; ".join(f"{c}: {lo}–{hi}" for c, (lo, hi) in o["date_ranges"].items()) or "—"
        tag = " (scoring_only)" if e["scoring_only"] else ""
        lines.append(f"| {name}{tag} | `{o['path']}` | {e['rows_read']:,} | {o['rows']:,} | "
                     f"{o['unique_msno']:,} | {ranges} | {o['bytes'] / 1e6:.1f} | "
                     f"`{o['sha256'][:16]}…` |")
    lines += ["", "## Source archives", "",
              "| Archive | Size (MB) | Downloaded | SHA-256 |", "|---|---:|---|---|"]
    for e in manifest["tables"].values():
        a = e["archive"]
        lines.append(f"| `{a['file']}` | {a['bytes'] / 1e6:.2f} | {a['downloaded']} | "
                     f"`{a['sha256']}` |")
    labels = {n: e for n, e in manifest["tables"].items() if "churn_rate_full" in e}
    if labels:
        lines += ["", "## Sample representativeness (churn rate)", "",
                  "| Label file | Full file | Sample | Difference (pp) |", "|---|---:|---:|---:|"]
        for n, e in labels.items():
            diff = 100 * (e["churn_rate_sample"] - e["churn_rate_full"])
            lines.append(f"| {n} | {e['churn_rate_full']:.4%} | "
                         f"{e['churn_rate_sample']:.4%} | {diff:+.3f} |")
    return "\n".join(lines) + "\n"


def cleanup_archives(cfg: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Delete archives whose outputs are validated, unless keep_archives is set."""
    if cfg.get("keep_archives", False):
        log("keep_archives: true, archives retained")
        return
    downloads = resolve_path(cfg["paths"]["downloads"], cfg)
    for e in manifest["tables"].values():
        path = downloads / e["archive"]["file"]
        if path.exists():
            path.unlink()
            log(f"deleted {path.name}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", nargs="*", help="subset of table names (default: all)")
    parser.add_argument("--skip-missing", action="store_true",
                        help="skip tables whose archive is not downloaded yet instead of failing")
    args = parser.parse_args(argv)

    cfg = load_config("data")
    manifest = load_manifest(resolve_path(cfg["paths"]["manifest_json"], cfg))
    for name in args.tables or list(cfg["tables"]):
        try:
            process_table(name, cfg["tables"][name], cfg, manifest,
                          allow_download=not args.skip_missing)
        except MissingArchiveError:
            if not args.skip_missing:
                raise
            log(f"{name}: archive not downloaded yet, skipping")
        finally:
            write_manifest(manifest, cfg)  # persist progress even if a later table fails
    cleanup_archives(cfg, manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
