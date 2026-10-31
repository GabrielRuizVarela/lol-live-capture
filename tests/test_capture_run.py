"""Worker-loop tests for lol_live_capture.run: atomic writes, heartbeat,
recognition skipped off the draft screen, error and SIGTERM exits (fakes throughout)."""
from __future__ import annotations

import json
import signal
from pathlib import Path

from lol_live_capture import run as run_mod


class FakeFrame:
    """Plain object standing in for a numpy BGR frame."""


class FakeTracker:
    """Returns successive CaptureState dicts from observe(); a dict means "changed"."""

    def __init__(self, states):
        self._states = list(states)
        self.calls = []

    def observe(self, t, is_draft, reads):
        self.calls.append((t, is_draft, reads))
        index = min(len(self.calls) - 1, max(len(self._states) - 1, 0))
        return self._states[index] if self._states else {"status": "drafting"}


class FlagTracker:
    """observe() returns a plain changed flag; state comes from capture_state()."""

    def __init__(self, state):
        self._state = state
        self.calls = 0

    def observe(self, t, is_draft, reads):
        self.calls += 1
        return self.calls == 1  # first frame is always a change

    def capture_state(self):
        return self._state


def _frames(*ts):
    return [(t, FakeFrame()) for t in ts]


def _state_file(tmp_path) -> Path:
    return tmp_path / "run" / "capture-state.json"


def test_run_writes_state_atomically_and_skips_recognition_off_draft(tmp_path, monkeypatch):
    state_file = _state_file(tmp_path)
    replaced = []
    real_replace = run_mod.os.replace

    def spy_replace(src, dst):
        replaced.append((str(src), str(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(run_mod.os, "replace", spy_replace)
    drafts = [True, False, True, True]
    read_calls = []
    tracker = FakeTracker([
        {"status": "drafting", "game_index": 1, "actions": [{"sequence": 1}]},
        {"status": "drafting", "game_index": 1, "actions": [{"sequence": 1}]},
        {"status": "drafting", "game_index": 1, "actions": [{"sequence": 1}, {"sequence": 2}]},
        {"status": "drafting", "game_index": 1, "actions": [{"sequence": 1}, {"sequence": 2}]},
    ])
    now = [0.0]

    def clock():
        now[0] += 0.1
        return now[0]

    # consume the draft flag per frame regardless of recognition
    flags = list(drafts)

    def is_draft_queue(_frame):
        return flags.pop(0), 1.0

    def read_frame_fn(_frame):
        read_calls.append(True)
        return f"r{len(read_calls)}"

    rc = run_mod.run("https://youtu.be/abc", state_file, frames=_frames(10.0, 11.0, 12.0, 13.0),
                     read_frame=read_frame_fn, is_draft=is_draft_queue, tracker=tracker, clock=clock)
    assert rc == 0
    assert len(read_calls) == 3  # recognition skipped on the one non-draft frame
    assert [c[1] for c in tracker.calls] == [True, False, True, True]
    assert [c[2] for c in tracker.calls] == ["r1", None, "r2", "r3"]
    assert replaced and all(src.endswith(".tmp") for src, _ in replaced)
    assert not state_file.with_name(state_file.name + ".tmp").exists()
    state = json.loads(state_file.read_text())
    assert state["status"] == "drafting" and state["source"] == "https://youtu.be/abc"
    assert set(run_mod.STATE_FIELDS) <= set(state)
    assert state["stream_t"] == 13.0
    assert isinstance(state["frame_age_ms"], (int, float)) and state["frame_age_ms"] >= 0
    assert state["updated_at"]


def test_run_heartbeat_writes_without_changes(tmp_path):
    state_file = _state_file(tmp_path)
    now = [0.0]

    def clock():
        return now[0]

    def frames():
        for t in (1.0, 2.0, 3.0, 4.0, 5.0):
            now[0] += 1.0  # one wall-clock second per frame
            yield t, FakeFrame()

    tracker = FlagTracker({"status": "waiting_for_draft", "actions": []})
    writes = []
    original = run_mod.write_state

    def spy(path, state):
        writes.append(json.loads(json.dumps(state)))
        return original(path, state)

    run_mod.write_state = spy
    try:
        rc = run_mod.run("src.mp4", state_file, frames=frames(),
                         read_frame=lambda f: {}, is_draft=lambda f: (False, 0.0),
                         tracker=tracker, clock=clock)
    finally:
        run_mod.write_state = original
    assert rc == 0
    # connecting, then a heartbeat write every >= 2 s: at frames 3 and 5 (frames 2 and 4 are silent)
    assert [w["status"] for w in writes] == ["connecting", "waiting_for_draft", "waiting_for_draft", "waiting_for_draft"]
    assert writes[0]["updated_at"] != writes[1]["updated_at"]


def test_run_exception_writes_error_state(tmp_path):
    state_file = _state_file(tmp_path)

    def frames():
        yield 1.0, FakeFrame()
        raise RuntimeError("boom: stream died")

    tracker = FakeTracker([{"status": "drafting"}])
    rc = run_mod.run("https://twitch.tv/x", state_file, frames=frames(),
                     read_frame=lambda f: {}, is_draft=lambda f: (True, 1.0),
                     tracker=tracker, clock=lambda: 1000.0)
    assert rc == 1
    state = json.loads(state_file.read_text())
    assert state["status"] == "error" and "boom" in state["error"]
    assert state["source"] == "https://twitch.tv/x"
    assert set(run_mod.STATE_FIELDS) <= set(state)


def test_run_sigterm_writes_idle_and_exits_zero(tmp_path, monkeypatch):
    state_file = _state_file(tmp_path)
    captured = {}

    def fake_signal(sig, handler):
        captured["handler"] = handler
        return signal.SIG_DFL

    monkeypatch.setattr(run_mod.signal, "signal", fake_signal)
    tracker = FakeTracker([{"status": "drafting"}])

    def is_draft_fn(_frame):
        captured["handler"](signal.SIGTERM, None)  # deliver SIGTERM mid-loop
        return True, 1.0

    rc = run_mod.run("src.mp4", state_file, frames=_frames(1.0, 2.0, 3.0),
                     read_frame=lambda f: {}, is_draft=is_draft_fn,
                     tracker=tracker, clock=lambda: 500.0)
    assert rc == 0
    state = json.loads(state_file.read_text())
    assert state["status"] == "idle" and state["error"] is None
    assert len(tracker.calls) == 1  # the loop stopped before the next frame


def test_run_writes_connecting_before_the_first_frame(tmp_path):
    state_file = _state_file(tmp_path)
    seen = []
    original = run_mod.write_state

    def spy(path, state):
        seen.append(state["status"])
        return original(path, state)

    run_mod.write_state = spy
    try:
        run_mod.run("src.mp4", state_file, frames=[],
                    read_frame=lambda f: {}, is_draft=lambda f: (False, 0.0),
                    tracker=FakeTracker([]), clock=lambda: 1.0)
    finally:
        run_mod.write_state = original
    assert seen and seen[0] == "connecting"


def test_run_injects_tricodes_into_teams(tmp_path):
    state_file = _state_file(tmp_path)
    run_mod.run("src.mp4", state_file, frames=_frames(1.0),
                read_frame=lambda f: {}, is_draft=lambda f: (True, 1.0),
                tricodes=lambda f: {"blue": "KC", "red": "MVK"},
                tracker=FakeTracker([{"status": "drafting"}]), clock=lambda: 1.0)
    state = json.loads(state_file.read_text())
    assert state["teams"] == {"blue": {"tricode": "KC"}, "red": {"tricode": "MVK"}}


def test_run_stores_team_names_next_to_the_tricodes(tmp_path):
    state_file = _state_file(tmp_path)
    run_mod.run("src.mp4", state_file, frames=_frames(1.0),
                read_frame=lambda f: {}, is_draft=lambda f: (True, 1.0),
                tricodes=lambda f: {"blue": "KC", "red": "MVK"},
                tracker=FakeTracker([{"status": "drafting"}]),
                blue_team="Karmine Corp", red_team="MVK Esports", clock=lambda: 1.0)
    state = json.loads(state_file.read_text())
    assert state["teams"] == {
        "blue": {"tricode": "KC", "name": "Karmine Corp"},
        "red": {"tricode": "MVK", "name": "MVK Esports"},
    }


def test_run_reloads_the_teams_file_every_five_seconds(tmp_path):
    state_file = _state_file(tmp_path)
    teams_file = state_file.with_name("capture-teams.json")
    now = [0.0]

    def clock():
        return now[0]

    def frames():
        yield 1.0, FakeFrame()
        teams_file.write_text(json.dumps({
            "blue": {"id": "MVK Esports", "name": "MVK Esports"},
            "red": {"id": "Karmine Corp", "name": "Karmine Corp"},
        }))
        now[0] += 6.0
        yield 2.0, FakeFrame()

    run_mod.run("src.mp4", state_file, frames=frames(),
                read_frame=lambda f: {}, is_draft=lambda f: (True, 1.0),
                tracker=FakeTracker([{"status": "drafting"}]),
                blue_team="Karmine Corp", red_team="MVK Esports", clock=clock)
    state = json.loads(state_file.read_text())
    assert state["teams"]["blue"]["name"] == "MVK Esports"
    assert state["teams"]["red"]["name"] == "Karmine Corp"


def test_main_parses_args_and_uses_home_state_file(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVE_CAPTURE_HOME", str(tmp_path))
    seen = {}

    def fake_run(source, state_file, **kwargs):
        seen["source"] = source
        seen["state_file"] = Path(state_file)
        seen["fps"] = kwargs.get("fps")
        return 0

    monkeypatch.setattr(run_mod, "run", fake_run)
    assert run_mod.main(["--source", "https://youtu.be/z"]) == 0
    assert seen["source"] == "https://youtu.be/z"
    assert seen["state_file"] == tmp_path / "run" / "capture-state.json"
    assert seen["fps"] == 2.0


def test_main_parses_team_and_realtime_flags(tmp_path, monkeypatch):
    seen = {}

    def fake_run(source, state_file, **kwargs):
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(run_mod, "run", fake_run)
    assert run_mod.main(["--source", "clip.mp4", "--realtime", "--blue-team", "KC",
                         "--red-team", "MVK"]) == 0
    assert seen["realtime"] is True
    assert seen["blue_team"] == "KC" and seen["red_team"] == "MVK"
