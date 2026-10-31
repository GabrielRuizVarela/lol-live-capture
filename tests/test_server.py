"""The app: health, champion table and art, and the SPA fallback."""
from __future__ import annotations

from fastapi.testclient import TestClient

from lol_live_capture import paths, server


def _client(monkeypatch, web=None):
    monkeypatch.setattr(server, "web_dir", lambda: web)
    return TestClient(server.create_app())


def test_health_reports_missing_art(monkeypatch):
    body = _client(monkeypatch).get("/api/health").json()
    assert body["status"] == "ok" and body["champion_art"] is False


def test_champion_list_uses_the_bundled_table(monkeypatch):
    body = _client(monkeypatch).get("/api/champions").json()
    by_id = {c["id"]: c for c in body["champions"]}
    assert by_id["62"]["name"] == "Wukong" and by_id["62"]["ddragon_id"] == "MonkeyKing"
    assert by_id["62"]["art"] is False


def test_champion_art_is_served_and_validated(monkeypatch):
    art = paths.champions_dir()
    (art / "loading").mkdir(parents=True)
    (art / "Ahri.png").write_bytes(b"png")
    (art / "loading" / "Ahri.jpg").write_bytes(b"jpg")
    client = _client(monkeypatch)
    assert client.get("/api/champions/Ahri/square").content == b"png"
    assert client.get("/api/champions/Ahri/loading").content == b"jpg"
    assert client.get("/api/champions/Ahri/other").status_code == 404
    assert client.get("/api/champions/Nobody/square").status_code == 404
    assert client.get("/api/champions/..%2F..%2Fx/square").status_code == 404


def test_spa_fallback_serves_index_but_not_for_api_paths(monkeypatch, tmp_path):
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<html>ui</html>")
    (web / "app.js").write_text("js")
    client = _client(monkeypatch, web)
    assert client.get("/").text == "<html>ui</html>"
    assert client.get("/some/route").text == "<html>ui</html>"
    assert client.get("/app.js").text == "js"
    assert client.get("/../secret").status_code in (200, 404)
    assert client.get("/api/nope").status_code == 404
