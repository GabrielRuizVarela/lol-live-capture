"""``setup`` with a fake CDN: what it downloads, where, and that existing files are kept."""
from __future__ import annotations

import gzip
import json

from lol_live_capture import assets, champions

TABLE = {"data": {"Ahri": {"key": "103", "name": "Ahri"}, "MonkeyKing": {"key": "62", "name": "Wukong"}}}


class FakeCdn:
    def __init__(self, missing: str | None = None) -> None:
        self.urls: list[str] = []
        self.missing = missing

    def __call__(self, url: str) -> bytes:
        self.urls.append(url)
        if self.missing and self.missing in url:
            raise OSError("404")
        if url.endswith("versions.json"):
            return json.dumps(["16.99.1", "16.98.1"]).encode()
        if url.endswith("champion.json"):
            return json.dumps(TABLE).encode()
        return b"img:" + url.encode()


def test_setup_downloads_table_tiles_and_loading_art(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVE_CAPTURE_HOME", str(tmp_path))
    cdn = FakeCdn()
    result = assets.setup(fetch=cdn)
    out = tmp_path / "champions"
    assert result["version"] == "16.99.1" and result["fetched"] == 4 and result["failed"] == []
    assert (out / "Ahri.png").read_bytes().endswith(b"/cdn/16.99.1/img/champion/Ahri.png")
    assert (out / "loading" / "MonkeyKing.jpg").read_bytes().endswith(b"loading/MonkeyKing_0.jpg")
    assert json.loads(gzip.decompress((out / "champion.json.gz").read_bytes())) == TABLE
    assert (out / "version.txt").read_text().strip() == "16.99.1"
    assert not list(out.rglob("*.part"))
    assert assets.is_ready()
    # the downloaded table now wins over the bundled one
    assert champions.id_to_key() == {"Ahri": "103", "MonkeyKing": "62"}


def test_setup_keeps_existing_files_and_reports_failures(tmp_path):
    out = tmp_path / "c"
    assets.setup(out, version="1.0.0", fetch=FakeCdn())
    again = FakeCdn(missing="Ahri_0.jpg")
    (out / "loading" / "Ahri.jpg").unlink()
    result = assets.setup(out, version="1.0.0", fetch=again)
    assert result["kept"] == 3 and result["fetched"] == 0
    assert len(result["failed"]) == 1 and "Ahri.jpg" in result["failed"][0]
    assert not any("MonkeyKing" in url for url in again.urls)


def test_not_ready_without_art(tmp_path):
    assert not assets.is_ready(tmp_path)
