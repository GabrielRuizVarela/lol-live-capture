"""Where the tool keeps its files. ``$LIVE_CAPTURE_HOME`` is read at call time so tests can override it."""
from __future__ import annotations

import os
from pathlib import Path


def home_dir() -> Path:
    return Path(os.environ.get("LIVE_CAPTURE_HOME") or (Path.home() / ".local" / "share" / "lol-live-capture"))


def run_dir() -> Path:
    return home_dir() / "run"


def logs_dir() -> Path:
    return home_dir() / "logs"


def videos_dir() -> Path:
    """Local video files the API may read (saved broadcasts for replay)."""
    return home_dir() / "capture"


def refs_dir() -> Path:
    """Cache of the built champion reference bank."""
    return videos_dir() / "refs"


def champions_dir() -> Path:
    """Champion art (square tiles + ``loading/`` portraits) and ``champion.json.gz``, written by ``setup``."""
    return home_dir() / "champions"
