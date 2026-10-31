"""``setup``: download the champion art the recognizer matches against, from Riot's Data Dragon CDN.

Nothing from Riot is stored in this repository: ``lol-live-capture setup`` fetches, for the latest patch, the
champion table, one square tile per champion (ban slots) and one loading-screen portrait per champion (pick
slots) into ``$LIVE_CAPTURE_HOME/champions``. Files already present are kept unless ``force``.
"""
from __future__ import annotations

import gzip
import json
import logging
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from . import paths
from .champions import FILE_NAME

log = logging.getLogger(__name__)

CDN = "https://ddragon.leagueoflegends.com"
TIMEOUT_S = 30
WORKERS = 8
USER_AGENT = "lol-live-capture/0.1 (setup)"

Fetch = Callable[[str], bytes]


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:  # noqa: S310 - fixed https CDN
        return response.read()


def latest_version(fetch: Fetch = _fetch) -> str:
    return json.loads(fetch(f"{CDN}/api/versions.json"))[0]


def setup(out: Path | None = None, *, version: str | None = None, force: bool = False,
          fetch: Fetch = _fetch, workers: int = WORKERS) -> dict:
    """Download the table and art; returns ``{version, fetched, kept, failed}``."""
    out = out or paths.champions_dir()
    (out / "loading").mkdir(parents=True, exist_ok=True)
    version = version or latest_version(fetch)
    raw = fetch(f"{CDN}/cdn/{version}/data/en_US/champion.json")
    table = json.loads(raw)
    (out / FILE_NAME).write_bytes(gzip.compress(raw))
    jobs: list[tuple[str, Path]] = []
    kept = 0
    for ddragon_id in sorted(table["data"]):
        for url, dest in (
            (f"{CDN}/cdn/{version}/img/champion/{ddragon_id}.png", out / f"{ddragon_id}.png"),
            (f"{CDN}/cdn/img/champion/loading/{ddragon_id}_0.jpg", out / "loading" / f"{ddragon_id}.jpg"),
        ):
            if dest.exists() and not force:
                kept += 1
            else:
                jobs.append((url, dest))

    def download(job: tuple[str, Path]) -> str | None:
        url, dest = job
        try:
            body = fetch(url)
        except Exception as error:  # noqa: BLE001 - keep going, report at the end
            return f"{dest.name}: {error}"
        tmp = dest.with_name(dest.name + ".part")
        tmp.write_bytes(body)
        tmp.replace(dest)
        return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        failed = [error for error in pool.map(download, jobs) if error]
    fetched = len(jobs) - len(failed)
    (out / "version.txt").write_text(version + "\n", encoding="utf-8")
    return {"version": version, "fetched": fetched, "kept": kept, "failed": failed, "dir": str(out)}


def is_ready(out: Path | None = None) -> bool:
    out = out or paths.champions_dir()
    return (out / "loading").is_dir() and any((out / "loading").glob("*.jpg")) and any(out.glob("*.png"))
