"""Tests for the generated-section writer behind reports/results.md."""

from pathlib import Path

from churnlens.reporting import fmt_pct, md_table, write_section


def test_write_section_creates_then_replaces_only_its_block(tmp_path: Path) -> None:
    path = tmp_path / "results.md"
    write_section(path, "audit", "first audit")
    write_section(path, "modeling", "model results")
    write_section(path, "audit", "second audit")
    text = path.read_text(encoding="utf-8")
    assert "second audit" in text and "first audit" not in text
    assert "model results" in text
    assert text.count("<!-- BEGIN:audit -->") == 1
    assert text.index("BEGIN:audit") < text.index("BEGIN:modeling"), "section order is kept"


def test_md_table_and_pct_helpers() -> None:
    table = md_table(["a", "b"], [[1, 2]], "lr")
    assert table.splitlines() == ["| a | b |", "|---|---:|", "| 1 | 2 |"]
    assert fmt_pct(1, 4) == "25.00%"
    assert fmt_pct(1, 0) == "—"
