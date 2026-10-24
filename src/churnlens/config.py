"""Config loading. Every tunable lives in configs/*.yaml, not in code."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_config(name: str) -> dict[str, Any]:
    """Load configs/<name>.yaml as a dict."""
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_path(template: str, cfg: dict[str, Any]) -> Path:
    """Resolve a config path template ({mode} placeholder, ~ expansion) against the project root."""
    path = Path(template.format(mode=cfg.get("mode", ""))).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def date_to_int(value: dt.date | str) -> int:
    """Convert a date (or ISO string) to KKBox's YYYYMMDD integer encoding."""
    if isinstance(value, str):
        value = dt.date.fromisoformat(value)
    return value.year * 10000 + value.month * 100 + value.day
