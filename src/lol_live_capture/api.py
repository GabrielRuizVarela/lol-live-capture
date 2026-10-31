"""HTTP control of the capture worker (docs/contract.md, "API").

``POST /capture/start`` validates the stream source, stops any previous
worker and launches ``lol_live_capture.run`` in its own process group
with the pid in ``$LIVE_CAPTURE_HOME/run/capture.pid``. ``/stop`` terminates that
group; ``/state`` reads the atomically written CaptureState file; ``/events``
streams it as SSE. ``$LIVE_CAPTURE_HOME`` is read at call time so tests can
override it via env.
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import paths
from .run import idle_state, write_state

POLL_INTERVAL_S = 0.5  # module constants so tests can make them tiny
KEEP_ALIVE_S = 15.0

ALLOWED_STREAM_HOSTS = frozenset({"youtube.com", "www.youtube.com", "m.youtube.com",
                                  "youtu.be", "twitch.tv", "www.twitch.tv"})
MAX_TEAM_FIELD = 80


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def _home() -> Path:
    return paths.home_dir()


def _state_path() -> Path:
    return _home() / "run" / "capture-state.json"


def _teams_path() -> Path:
    return _home() / "run" / "capture-teams.json"


def _pid_path() -> Path:
    return _home() / "run" / "capture.pid"


def _log_path() -> Path:
    return _home() / "logs" / "capture.log"


def _capture_dir() -> Path:
    return _home() / "capture"


def _validate_source(source: Any) -> tuple[str | None, JSONResponse | None]:
    """Allowed: YouTube/Twitch https URLs, or a local file resolving inside $LIVE_CAPTURE_HOME/capture."""
    if not isinstance(source, str) or not source.strip():
        return None, _error(422, "invalid_source", "source must be a non-empty stream URL or local video path")
    candidate = source.strip()
    parts = urlsplit(candidate)
    if parts.scheme in ("http", "https"):
        host = (parts.hostname or "").lower()
        if host in ALLOWED_STREAM_HOSTS:
            return candidate, None
        return None, _error(422, "invalid_source", f"source host must be one of {sorted(ALLOWED_STREAM_HOSTS)}")
    if parts.scheme and parts.scheme not in ("file",):
        return None, _error(422, "invalid_source", "source must be an https YouTube/Twitch URL or a local video path")
    try:
        resolved = Path(candidate).expanduser().resolve()
    except OSError as error:
        return None, _error(422, "invalid_source", f"source path cannot be resolved: {error}")
    base = _capture_dir().resolve()
    if not resolved.is_file():
        return None, _error(422, "invalid_source", "local source must be an existing file under the capture directory")
    if not resolved.is_relative_to(base):
        return None, _error(422, "invalid_source", "local source must resolve inside the capture directory")
    return str(resolved), None


def _valid_team_field(value: Any) -> bool:
    return (isinstance(value, str) and bool(value) and len(value) <= MAX_TEAM_FIELD
            and value.isprintable())


def _validate_teams(value: Any) -> tuple[dict[str, dict[str, str]] | None, JSONResponse | None]:
    """A ``{blue: {id, name}, red: {id, name}}`` body with short printable strings."""
    if value is None:
        return None, None
    if not isinstance(value, Mapping):
        return None, _error(422, "invalid_teams", "teams must be an object with blue and red teams")
    teams: dict[str, dict[str, str]] = {}
    for side in ("blue", "red"):
        team = value.get(side)
        if not isinstance(team, Mapping):
            return None, _error(422, "invalid_teams", f"teams.{side} must be an object with id and name")
        entry: dict[str, str] = {}
        for field in ("id", "name"):
            field_value = team.get(field)
            if not _valid_team_field(field_value):
                return None, _error(
                    422, "invalid_teams",
                    f"teams.{side}.{field} must be a printable string of at most {MAX_TEAM_FIELD} characters")
            entry[field] = field_value
        teams[side] = entry
    return teams, None


def _write_teams(teams: Mapping[str, Mapping[str, str]] | None) -> None:
    """Write (or clear) ``run/capture-teams.json``; the worker re-reads it every 5 s."""
    path = _teams_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if teams is None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(teams), encoding="utf-8")
    os.replace(tmp, path)


def _read_pid() -> int | None:
    try:
        return int(_pid_path().read_text().strip())
    except (OSError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stop_worker() -> None:
    """SIGTERM the worker's process group (the pid is the pgid: start_new_session)."""
    pid = _read_pid()
    if pid is not None:
        try:
            os.killpg(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        _pid_path().unlink(missing_ok=True)
    except OSError:
        pass


def _current_state() -> dict[str, Any]:
    path = _state_path()
    raw: Any = None
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        raw = None
    if isinstance(raw, dict):
        status = raw.get("status")
        pid = _read_pid()
        running = pid is not None and _pid_alive(pid)
        if status == "error" or running:
            return raw
    return idle_state()


def _launch_worker(src: str,
                   teams: Mapping[str, Mapping[str, str]] | None = None) -> int:
    state_file = _state_path()
    state_file.parent.mkdir(parents=True, exist_ok=True)
    _pid_path().parent.mkdir(parents=True, exist_ok=True)
    _log_path().parent.mkdir(parents=True, exist_ok=True)
    # The worker may run in its own venv (LIVE_CAPTURE_PYTHON); default: this interpreter.
    python = os.environ.get("LIVE_CAPTURE_PYTHON") or sys.executable
    if not Path(python).is_file():
        python = sys.executable
    command = [python, "-m", "lol_live_capture.run",
               "--source", src, "--state-file", str(state_file)]
    if teams is not None:
        command += ["--blue-team", teams["blue"]["name"], "--red-team", teams["red"]["name"]]
    if not src.startswith(("http://", "https://")):
        command.append("--realtime")  # a saved broadcast replays at 1x, like the live stream
    with open(_log_path(), "ab") as log:
        proc = subprocess.Popen(command, start_new_session=True,
                                stdout=log, stderr=log)
    _pid_path().write_text(f"{proc.pid}\n")
    return proc.pid


async def event_stream(request: Any, *, once: bool = False):
    """SSE body: state events on updated_at change, keep-alive comments every KEEP_ALIVE_S.

    Polls the state file every POLL_INTERVAL_S and stops when the client disconnects.
    Kept at module level so tests can drive it directly (the TestClient buffers whole
    responses and cannot observe an endless stream incrementally).
    """
    if once:
        yield f"event: state\ndata: {json.dumps(_current_state())}\n\n"
        return
    last_updated: Any = object()
    last_keep = time.monotonic()
    while True:
        if await request.is_disconnected():
            return
        data = _current_state()
        updated = data.get("updated_at")
        if updated != last_updated:
            last_updated = updated
            last_keep = time.monotonic()
            yield f"event: state\ndata: {json.dumps(data)}\n\n"
        elif time.monotonic() - last_keep >= KEEP_ALIVE_S:
            last_keep = time.monotonic()
            yield ": keep-alive\n\n"
        await asyncio.sleep(POLL_INTERVAL_S)


def create_router() -> APIRouter:
    router = APIRouter(prefix="/capture", tags=["live-capture"])

    @router.post("/start")
    def start(payload: dict[str, Any] | None = None):
        body = payload if isinstance(payload, Mapping) else {}
        src, rejected = _validate_source(body.get("source"))
        if rejected is not None:
            return rejected
        teams, rejected = _validate_teams(body.get("teams"))
        if rejected is not None:
            return rejected
        _stop_worker()  # at most one worker runs
        _write_teams(teams)  # no teams clears a previous session's file
        try:
            _launch_worker(src, teams)
        except OSError as error:
            return _error(503, "capture_start_failed", f"could not start the capture worker: {error}")
        return {"status": "connecting", "source": src}

    @router.post("/teams")
    def teams(payload: dict[str, Any] | None = None):
        body: Any = payload
        if isinstance(payload, Mapping) and isinstance(payload.get("teams"), Mapping):
            body = payload.get("teams")
        validated, rejected = _validate_teams(body)
        if rejected is not None:
            return rejected
        if validated is None:
            return _error(422, "invalid_teams", "teams must be an object with blue and red teams")
        _write_teams(validated)
        return {"status": "ok", "teams": validated}

    @router.post("/stop")
    def stop():
        _stop_worker()
        previous: Mapping[str, Any] = {}
        try:
            raw = json.loads(_state_path().read_text())
            if isinstance(raw, dict):
                previous = raw
        except (OSError, ValueError):
            previous = {}
        state = idle_state()
        state.update({k: previous[k] for k in state if k in previous})
        state["status"] = "idle"
        write_state(_state_path(), state)
        return {"status": "idle"}

    @router.get("/state")
    def state():
        return _current_state()

    @router.get("/events")
    async def events(request: Request, once: int = 0):
        return StreamingResponse(event_stream(request, once=bool(once)), media_type="text/event-stream")

    return router
