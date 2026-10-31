"""HTTP contract tests for /api/capture (docs/contract.md):
source validation, worker start/stop via a monkeypatched Popen/killpg, /state
fallbacks and the SSE /events stream (?once=1)."""
from __future__ import annotations

import json
import signal
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lol_live_capture import api as cap_api

API = "/api/capture"
CONTRACT_FIELDS = {"status", "source", "layout", "updated_at", "stream_t", "frame_age_ms",
                   "game_index", "teams", "actions", "hover", "warnings", "error"}


class FakeProc:
    pid = 424242


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVE_CAPTURE_HOME", str(tmp_path))
    (tmp_path / "capture").mkdir()
    (tmp_path / "run").mkdir()
    (tmp_path / "logs").mkdir()
    return tmp_path


@pytest.fixture
def popen(monkeypatch):
    calls = []

    def fake_popen(command, **kwargs):
        calls.append({"command": command, **kwargs})
        return FakeProc()

    monkeypatch.setattr(cap_api.subprocess, "Popen", fake_popen)
    return calls


@pytest.fixture
def killpg(monkeypatch):
    calls = []

    def fake_killpg(pid, sig):
        calls.append((pid, sig))

    monkeypatch.setattr(cap_api.os, "killpg", fake_killpg)
    return calls


def _client(root: Path) -> TestClient:
    app = FastAPI()
    app.include_router(cap_api.create_router(), prefix="/api")
    return TestClient(app)


@pytest.fixture
def client(home):
    return _client(home)


# ---- POST /start: validation ---------------------------------------------------------------------------
@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=abc",
    "https://youtube.com/watch?v=abc",
    "https://m.youtube.com/watch?v=abc",
    "https://youtu.be/abc",
    "https://www.twitch.tv/videos/123",
    "https://twitch.tv/somechannel",
])
def test_start_allows_youtube_and_twitch(client, popen, url):
    r = client.post(f"{API}/start", json={"source": url})
    assert r.status_code == 200, r.text
    assert r.json() == {"status": "connecting", "source": url}


@pytest.mark.parametrize("source", [
    "http://evil.com/watch?v=abc",
    "https://evil.com/watch?v=abc",
    "https://youtube.com.evil.com/watch",
    "javascript:alert(1)",
    "ftp://youtube.com/x",
    "",
    "   ",
    None,
    42,
])
def test_start_rejects_disallowed_sources(client, popen, source):
    r = client.post(f"{API}/start", json={"source": source})
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "invalid_source"
    assert popen == []


def test_start_rejects_local_file_outside_capture(home, client, popen, tmp_path):
    outside = home / "elsewhere" / "game.mp4"
    outside.parent.mkdir()
    outside.write_bytes(b"x")
    r = client.post(f"{API}/start", json={"source": str(outside)})
    assert r.status_code == 422 and "capture directory" in r.json()["error"]["message"]
    assert popen == []


def test_start_rejects_path_traversal_outside_home(client, popen):
    r = client.post(f"{API}/start", json={"source": "../../../../etc/passwd"})
    assert r.status_code == 422 and popen == []


def test_start_rejects_missing_local_file(client, popen):
    r = client.post(f"{API}/start", json={"source": "videos/nope.mp4"})
    assert r.status_code == 422 and popen == []


def test_start_rejects_symlink_escape(home, client, popen, tmp_path):
    target = tmp_path / "outside.mp4"
    target.write_bytes(b"x")
    link = home / "capture" / "sneaky.mp4"
    link.symlink_to(target)
    r = client.post(f"{API}/start", json={"source": str(link)})
    assert r.status_code == 422 and "capture directory" in r.json()["error"]["message"]
    assert popen == []


def test_start_accepts_local_file_inside_capture(home, client, popen):
    video = home / "capture" / "game.mp4"
    video.write_bytes(b"x")
    r = client.post(f"{API}/start", json={"source": str(video)})
    assert r.status_code == 200, r.text
    assert r.json()["source"] == str(video.resolve())


# ---- POST /start: worker lifecycle ----------------------------------------------------------------------
def test_start_writes_pid_file_and_launches_the_worker(home, client, popen):
    r = client.post(f"{API}/start", json={"source": "https://youtu.be/abc"})
    assert r.status_code == 200
    assert (home / "run" / "capture.pid").read_text().strip() == str(FakeProc.pid)
    call = popen[0]
    command = call["command"]
    assert command[1:4] == ["-m", "lol_live_capture.run", "--source"]
    assert command[4] == "https://youtu.be/abc"
    assert command[5] == "--state-file" and command[6] == str(home / "run" / "capture-state.json")
    assert call["start_new_session"] is True
    assert call["stdout"] is call["stderr"]


def test_start_passes_team_names_and_writes_the_teams_file(home, client, popen):
    teams = {"blue": {"id": "Karmine Corp", "name": "Karmine Corp"},
             "red": {"id": "MVK Esports", "name": "MVK Esports"}}
    r = client.post(f"{API}/start", json={"source": "https://youtu.be/abc", "teams": teams})
    assert r.status_code == 200, r.text
    command = popen[0]["command"]
    assert command[command.index("--blue-team") + 1] == "Karmine Corp"
    assert command[command.index("--red-team") + 1] == "MVK Esports"
    assert json.loads((home / "run" / "capture-teams.json").read_text()) == teams


def test_start_without_teams_clears_an_old_teams_file(home, client, popen):
    (home / "run" / "capture-teams.json").write_text(json.dumps(
        {"blue": {"id": "old", "name": "Old"}, "red": {"id": "old2", "name": "Old Too"}}))
    r = client.post(f"{API}/start", json={"source": "https://youtu.be/abc"})
    assert r.status_code == 200
    assert not (home / "run" / "capture-teams.json").exists()


@pytest.mark.parametrize("teams", [
    {"blue": {"id": "KC", "name": "Karmine Corp"}},
    {"blue": {"id": "KC", "name": "Karmine Corp"}, "red": {"id": "", "name": "MVK"}},
    {"blue": {"id": "KC", "name": "x" * 81}, "red": {"id": "MVK", "name": "MVK Esports"}},
    {"blue": {"id": "KC", "name": "bad\nname"}, "red": {"id": "MVK", "name": "MVK Esports"}},
    {"blue": {"id": 1, "name": "Karmine Corp"}, "red": {"id": "MVK", "name": "MVK Esports"}},
    "KC vs MVK",
])
def test_start_rejects_invalid_teams(client, popen, teams):
    r = client.post(f"{API}/start", json={"source": "https://youtu.be/abc", "teams": teams})
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "invalid_teams"
    assert popen == []


def test_teams_endpoint_updates_the_file_while_running(home, client):
    teams = {"blue": {"id": "MVK Esports", "name": "MVK Esports"},
             "red": {"id": "Karmine Corp", "name": "Karmine Corp"}}
    r = client.post(f"{API}/teams", json=teams)
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "ok"
    assert json.loads((home / "run" / "capture-teams.json").read_text()) == teams


def test_teams_endpoint_rejects_a_missing_body(home, client):
    r = client.post(f"{API}/teams", json={})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_teams"


def test_start_stops_a_previous_worker_first(home, client, popen, killpg):
    (home / "run" / "capture.pid").write_text(f"{FakeProc.pid}\n")
    r = client.post(f"{API}/start", json={"source": "https://twitch.tv/x"})
    assert r.status_code == 200
    assert killpg[0] == (FakeProc.pid, signal.SIGTERM)
    # the new worker's pid replaced the old file
    assert (home / "run" / "capture.pid").read_text().strip() == str(FakeProc.pid)


# ---- POST /stop ------------------------------------------------------------------------------------------
def test_stop_kills_the_group_removes_pid_and_writes_idle(home, client, popen, killpg):
    client.post(f"{API}/start", json={"source": "https://youtu.be/abc"})
    state_file = home / "run" / "capture-state.json"
    state_file.write_text(json.dumps({"status": "drafting", "source": "https://youtu.be/abc",
                                      "actions": [{"sequence": 1}], "updated_at": "2026-10-18T00:00:00+00:00"}))
    r = client.post(f"{API}/stop")
    assert r.status_code == 200 and r.json() == {"status": "idle"}
    assert (FakeProc.pid, signal.SIGTERM) in killpg
    assert not (home / "run" / "capture.pid").exists()
    state = json.loads(state_file.read_text())
    assert state["status"] == "idle" and CONTRACT_FIELDS <= set(state)
    assert state["actions"] == [{"sequence": 1}]  # last actions kept for the page


def test_stop_without_a_worker_still_succeeds(home, client, killpg):
    r = client.post(f"{API}/stop")
    assert r.status_code == 200 and r.json() == {"status": "idle"}
    state = json.loads((home / "run" / "capture-state.json").read_text())
    assert state["status"] == "idle" and CONTRACT_FIELDS <= set(state)


# ---- GET /state ------------------------------------------------------------------------------------------
def test_state_when_missing_is_a_full_idle_document(home, client):
    r = client.get(f"{API}/state")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "idle" and data["actions"] == []
    assert CONTRACT_FIELDS <= set(data)
    assert data["teams"] == {"blue": {"tricode": None}, "red": {"tricode": None}}


def test_state_reads_the_file_while_the_worker_runs(home, client, popen, monkeypatch):
    client.post(f"{API}/start", json={"source": "https://youtu.be/abc"})
    written = {"status": "drafting", "source": "https://youtu.be/abc", "game_index": 2,
               "actions": [{"sequence": 1, "side": "blue", "action_type": "ban",
                            "champion_id": "888", "confidence": 0.97, "locked_at": 1200.0}],
               "updated_at": "2026-10-18T12:00:00+00:00"}
    (home / "run" / "capture-state.json").write_text(json.dumps(written))
    monkeypatch.setattr(cap_api, "_pid_alive", lambda pid: True)
    data = client.get(f"{API}/state").json()
    assert data["status"] == "drafting" and data["game_index"] == 2
    assert data["actions"][0]["champion_id"] == "888"


def test_state_after_stop_keeps_the_last_draft(home, client, popen, killpg):
    client.post(f"{API}/start", json={"source": "https://youtu.be/abc"})
    (home / "run" / "capture-state.json").write_text(json.dumps(
        {"status": "drafting", "actions": [{"sequence": 1, "champion_id": "56"}], "updated_at": "t0"}))
    client.post(f"{API}/stop")
    data = client.get(f"{API}/state").json()
    assert data["status"] == "idle" and data["actions"] == [{"sequence": 1, "champion_id": "56"}]


def test_state_with_a_dead_worker_falls_back_to_idle(home, client, popen, monkeypatch):
    client.post(f"{API}/start", json={"source": "https://youtu.be/abc"})
    (home / "run" / "capture-state.json").write_text(json.dumps({"status": "drafting", "source": "https://youtu.be/abc"}))
    monkeypatch.setattr(cap_api, "_pid_alive", lambda pid: False)
    data = client.get(f"{API}/state").json()
    assert data["status"] == "idle" and CONTRACT_FIELDS <= set(data)


def test_state_keeps_an_error_even_with_a_dead_worker(home, client, monkeypatch):
    (home / "run" / "capture-state.json").write_text(json.dumps(
        {"status": "error", "error": "boom", "source": "https://youtu.be/abc"}))
    (home / "run" / "capture.pid").write_text("424242\n")
    monkeypatch.setattr(cap_api, "_pid_alive", lambda pid: False)
    data = client.get(f"{API}/state").json()
    assert data["status"] == "error" and data["error"] == "boom"


# ---- GET /events ------------------------------------------------------------------------------------------
def test_events_once_sends_one_state_block_and_closes(home, client, monkeypatch):
    (home / "run" / "capture.pid").write_text("424242\n")
    monkeypatch.setattr(cap_api, "_pid_alive", lambda pid: True)
    (home / "run" / "capture-state.json").write_text(json.dumps(
        {"status": "drafting", "source": "https://youtu.be/abc", "updated_at": "2026-10-18T12:00:00+00:00"}))
    r = client.get(f"{API}/events", params={"once": 1})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    body = r.text
    assert body.count("event: state") == 1
    payload = body.split("data: ", 1)[1].strip()
    assert json.loads(payload)["status"] == "drafting"


def test_events_once_without_a_state_sends_idle(home, client):
    r = client.get(f"{API}/events", params={"once": 1})
    assert r.status_code == 200
    payload = r.text.split("data: ", 1)[1].strip()
    data = json.loads(payload)
    assert data["status"] == "idle" and CONTRACT_FIELDS <= set(data)


class FakeRequest:
    """is_disconnected() flips to True after a few polls so the loop terminates."""

    def __init__(self, disconnect_after: int):
        self.polls = 0
        self.disconnect_after = disconnect_after

    async def is_disconnected(self):
        self.polls += 1
        return self.polls > self.disconnect_after


async def test_events_stream_emits_on_update_then_keeps_alive(home, monkeypatch):
    """Drive the SSE generator: state event, keep-alive, state event, disconnect."""
    monkeypatch.setattr(cap_api, "POLL_INTERVAL_S", 0.001)
    monkeypatch.setattr(cap_api, "KEEP_ALIVE_S", 0.002)
    monkeypatch.setattr(cap_api, "_pid_alive", lambda pid: True)
    (home / "run" / "capture.pid").write_text("424242\n")
    state_file = home / "run" / "capture-state.json"
    state_file.write_text(json.dumps({"status": "connecting", "updated_at": "t0"}))
    request = FakeRequest(disconnect_after=50)
    chunks = []
    async for chunk in cap_api.event_stream(request):
        chunks.append(chunk)
        if len(chunks) == 1:
            assert "connecting" in chunk
        if len(chunks) == 2 and "keep-alive" in chunk:
            state_file.write_text(json.dumps({"status": "drafting", "updated_at": "t1"}))
        if "drafting" in chunk:
            break
    joined = "".join(chunks)
    assert "event: state" in joined and "connecting" in joined
    assert ": keep-alive" in joined
    assert "drafting" in joined


async def test_events_stream_stops_when_the_client_disconnects(home, monkeypatch):
    monkeypatch.setattr(cap_api, "POLL_INTERVAL_S", 0.001)
    (home / "run" / "capture-state.json").write_text(json.dumps({"status": "idle", "updated_at": "t0"}))
    # disconnects right after the first state event: no keep-alive follows
    chunks = [chunk async for chunk in cap_api.event_stream(FakeRequest(disconnect_after=1))]
    assert len(chunks) == 1
    assert json.loads(chunks[0].split("data: ", 1)[1])["status"] == "idle"


def test_a_local_file_replays_in_real_time_and_a_stream_does_not(home, client, popen):
    replay = home / "capture" / "replay.mkv"
    replay.parent.mkdir(parents=True, exist_ok=True)
    replay.write_bytes(b"x")
    assert client.post(f"{API}/start", json={"source": str(replay)}).status_code == 200
    assert client.post(f"{API}/start", json={"source": "https://youtu.be/abc"}).status_code == 200
    assert "--realtime" in popen[0]["command"]
    assert "--realtime" not in popen[1]["command"]
