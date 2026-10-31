"""The Data Dragon champion table: ``champion.json.gz`` from ``setup`` if present, else the bundled copy.

Ids: Data Dragon names champions by a string id (``MonkeyKing``, ``KSante``); Riot's numeric ``key`` (``"62"``,
``"897"``) is what every other part of this tool and its output uses as ``champion_id``.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from . import paths

BUNDLED = Path(__file__).resolve().parent / "data" / "champion.json.gz"
FILE_NAME = "champion.json.gz"


def table_path() -> Path:
    downloaded = paths.champions_dir() / FILE_NAME
    return downloaded if downloaded.is_file() else BUNDLED


def load_payload(path: Path | None = None) -> dict[str, Any]:
    with gzip.open(path or table_path(), "rt", encoding="utf-8") as handle:
        return json.load(handle)


def id_to_key(path: Path | None = None) -> dict[str, str]:
    """Data Dragon id -> Riot numeric key, e.g. ``MonkeyKing`` -> ``"62"``."""
    return {name: str(entry["key"]) for name, entry in load_payload(path)["data"].items()}


def key_to_name(path: Path | None = None) -> dict[str, str]:
    """Riot numeric key -> display name, e.g. ``"62"`` -> ``"Wukong"``."""
    return {str(entry["key"]): str(entry["name"]) for entry in load_payload(path)["data"].values()
            if "key" in entry and "name" in entry}
