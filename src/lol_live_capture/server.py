"""The web app: the capture API under ``/api`` plus the built UI (``ui/`` -> ``web/``) when it is present."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

from . import api, assets, champions, paths

PACKAGE_WEB = Path(__file__).resolve().parent / "web"
REPO_WEB = Path(__file__).resolve().parents[2] / "ui" / "dist"


def web_dir() -> Path | None:
    for candidate in (PACKAGE_WEB, REPO_WEB):
        if (candidate / "index.html").is_file():
            return candidate
    return None


def create_app() -> FastAPI:
    app = FastAPI(title="lol-live-capture", docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.include_router(api.create_router(), prefix="/api")

    @app.get("/api/health")
    def health():
        return {"status": "ok", "champion_art": assets.is_ready(), "home": str(paths.home_dir())}

    @app.get("/api/champions")
    def champion_list():
        """Every champion: Riot key (``champion_id`` everywhere else), name, Data Dragon id, and whether art is on disk."""
        art = paths.champions_dir()
        payload = champions.load_payload()["data"]
        return {"version": champions.load_payload().get("version"), "champions": [
            {"id": str(e["key"]), "name": e["name"], "ddragon_id": ddragon_id,
             "art": (art / f"{ddragon_id}.png").is_file()} for ddragon_id, e in sorted(payload.items())]}

    @app.get("/api/champions/{ddragon_id}/{kind}")
    def champion_art(ddragon_id: str, kind: str):
        base = paths.champions_dir()
        if kind == "square":
            target = base / f"{ddragon_id}.png"
        elif kind == "loading":
            target = base / "loading" / f"{ddragon_id}.jpg"
        else:
            return JSONResponse(status_code=404, content={"error": {"code": "not_found", "message": kind}})
        if not ddragon_id.isalnum() or not target.is_file():
            return JSONResponse(status_code=404, content={"error": {"code": "not_found", "message": "no art"}})
        return FileResponse(target, headers={"Cache-Control": "public, max-age=86400"})

    web = web_dir()
    if web is not None:
        root = web.resolve()

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            if path.startswith("api/"):
                return JSONResponse(status_code=404, content={"error": {"code": "not_found", "message": path}})
            target = (root / path).resolve()
            if path and target.is_file() and target.is_relative_to(root):
                return FileResponse(target)
            return FileResponse(root / "index.html")

    return app


app = create_app()
