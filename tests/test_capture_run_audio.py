"""Worker tests for the audio / LLM / lolesports wiring in live_capture.run.

Everything is faked: frames, audio chunks, the transcriber, the interpreter and
the lolesports confirmation, so no GPU, no network and no API key are needed.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from lol_live_capture import interpret as interpret_mod
from lol_live_capture import run as run_mod


@pytest.fixture(autouse=True)
def _no_llm_key(monkeypatch):
    monkeypatch.delenv("LIVE_LLM_API_KEY", raising=False)


class FakeFrame:
    """Plain object standing in for a numpy BGR frame."""


def seg(t: float, text: str, t1: float | None = None):
    return SimpleNamespace(t0=float(t), t1=float(t if t1 is None else t1),
                           text=text, avg_logprob=-0.2)


class FakeTranscriber:
    def __init__(self, text: str = "they lock in Ahri", delay: float = 0.0) -> None:
        self.text = text
        self.delay = delay
        self.calls: list[tuple[float, int]] = []

    def feed(self, t: float, pcm) -> list:
        self.calls.append((float(t), len(pcm)))
        if self.delay:
            time.sleep(self.delay)
        return [seg(t, self.text, float(t) + 0.5)]


class FakeTracker:
    def __init__(self, states) -> None:
        self._states = list(states)
        self.observations: list[tuple[float, bool]] = []
        self.mentions: list = []

    def observe(self, t, is_draft, reads):
        self.observations.append((float(t), bool(is_draft)))
        index = min(len(self.observations) - 1, max(len(self._states) - 1, 0))
        return self._states[index] if self._states else {"status": "drafting"}

    def state(self):
        return self._states[-1] if self._states else {}

    def observe_mentions(self, mentions):
        self.mentions.extend(mentions)


PCM = b"\x00\x00" * 1600


def _state_file(tmp_path) -> Path:
    return tmp_path / "run" / "capture-state.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_audio_chunks_reach_the_tracker_as_mentions(tmp_path):
    state_file = _state_file(tmp_path)
    tracker = FakeTracker([{"status": "drafting", "game_index": 1, "actions": [],
                            "teams": {"blue": {"tricode": "KC"}, "red": {"tricode": "MVK"}}}])
    transcriber = FakeTranscriber("they lock in Ahri")

    def items():
        yield "video", 10.0, FakeFrame()
        yield "audio", 9.0, PCM
        yield "audio", 10.0, PCM
        _wait_for(lambda: len(tracker.mentions) >= 2)
        yield "video", 11.0, FakeFrame()

    rc = run_mod.run("https://youtu.be/x", state_file, av=items(),
                     read_frame=lambda f: {}, is_draft=lambda f: (True, 1.0),
                     tracker=tracker, clock=lambda: 1000.0, transcriber=transcriber)
    assert rc == 0
    assert len(transcriber.calls) == 2
    assert [(m.champion_id, m.cue) for m in tracker.mentions] == [("103", "pick"), ("103", "pick")]
    state = _read(state_file)
    audio = state["audio"]
    assert audio["enabled"] is True and audio["llm"] is False
    assert audio["last_segment"] == "they lock in Ahri"
    assert audio["last_segment"] == audio["last_segment"][:120]
    assert isinstance(audio["lag_s"], (int, float)) and audio["lag_s"] >= 0


def test_a_tracker_without_observe_mentions_still_runs(tmp_path):
    state_file = _state_file(tmp_path)

    class OldTracker:
        def __init__(self):
            self.observations: list = []

        def observe(self, t, is_draft, reads):
            self.observations.append((float(t), bool(is_draft)))
            return {"status": "drafting", "actions": []}

        def state(self):
            return {"status": "drafting", "actions": []}

    tracker = OldTracker()

    def items():
        yield "audio", 9.0, PCM
        yield "video", 10.0, FakeFrame()
        time.sleep(0.1)
        yield "video", 11.0, FakeFrame()

    rc = run_mod.run("src.mp4", state_file, av=items(),
                     read_frame=lambda f: {}, is_draft=lambda f: (True, 1.0),
                     tracker=tracker, clock=lambda: 1000.0,
                     transcriber=FakeTranscriber())
    assert rc == 0
    assert len(tracker.observations) == 2


def test_llm_is_skipped_without_the_key(tmp_path, monkeypatch):
    state_file = _state_file(tmp_path)
    built: list = []

    class RecorderInterpreter:
        def __init__(self, *args, **kwargs) -> None:
            built.append(self)
            self.enabled = True
            self.calls: list = []

        def interpret(self, tail, context):
            self.calls.append((tail, context))
            return []

    monkeypatch.setattr(interpret_mod, "Interpreter", RecorderInterpreter)
    tracker = FakeTracker([{"status": "drafting", "actions": []}])

    def items():
        yield "audio", 9.0, PCM
        yield "video", 10.0, FakeFrame()
        time.sleep(0.1)
        yield "video", 11.0, FakeFrame()

    rc = run_mod.run("src.mp4", state_file, av=items(),
                     read_frame=lambda f: {}, is_draft=lambda f: (True, 1.0),
                     tracker=tracker, clock=lambda: 1000.0,
                     transcriber=FakeTranscriber("and there it is"))
    assert rc == 0
    assert built == []
    assert _read(state_file)["audio"]["llm"] is False


def test_llm_events_with_a_key_become_mentions(tmp_path, monkeypatch):
    state_file = _state_file(tmp_path)
    built: list = []

    class RecorderInterpreter:
        def __init__(self, *args, **kwargs) -> None:
            built.append(self)
            self.enabled = True
            self.calls: list = []

        def interpret(self, tail, context):
            self.calls.append({"tail": tail, "context": context})
            return [
                interpret_mod.Event("62", "blue", "ban", 0.9),
                interpret_mod.Event("103", None, "pick", 0.8),
                interpret_mod.Event("1", "red", "hover", 0.5),
                interpret_mod.Event("2", "red", "speculative", 0.4),
            ]

    monkeypatch.setenv("LIVE_LLM_API_KEY", "do-not-leak")
    monkeypatch.setattr(interpret_mod, "Interpreter", RecorderInterpreter)
    tracker = FakeTracker([{"status": "drafting", "game_index": 3, "actions": [],
                            "teams": {"blue": {"tricode": "KC"}, "red": {"tricode": "MVK"}}}])

    def items():
        yield "audio", 9.0, PCM
        yield "video", 10.0, FakeFrame()
        _wait_for(lambda: len(tracker.mentions) >= 2)
        yield "video", 11.0, FakeFrame()

    rc = run_mod.run("src.mp4", state_file, av=items(),
                     read_frame=lambda f: {}, is_draft=lambda f: (True, 1.0),
                     tracker=tracker, clock=lambda: 1000.0,
                     transcriber=FakeTranscriber("and there it is"))
    assert rc == 0
    assert len(built) == 1
    assert [(m.champion_id, m.cue) for m in tracker.mentions] == [("62", "ban"), ("103", "pick")]
    assert built[0].calls[0]["context"]["session"] and built[0].calls[0]["context"]["game_index"] == 3
    assert _read(state_file)["audio"]["llm"] is True


def test_lolesports_confirmation_only_after_the_draft_and_stops(tmp_path):
    state_file = _state_file(tmp_path)
    drafting = {"status": "drafting", "game_index": 1, "actions": [], "teams": {}}
    complete = {"status": "draft_complete", "game_index": 1, "actions": [], "teams": {}}
    tracker = FakeTracker([drafting, drafting] + [complete] * 10)
    calls: list[dict] = []

    def attach(state):
        calls.append({"status": state.get("status"), "actions": state.get("actions")})
        status = "confirmed" if len(calls) >= 3 else "partial"
        return {**state, "lolesports": {"status": status, "game_id": "g1"}}

    now = [0.0]

    def clock():
        return now[0]

    def items():
        for i in range(12):
            now[0] += 31.0
            yield float(i), FakeFrame()

    rc = run_mod.run("src.mp4", state_file, av=items(),
                     read_frame=lambda f: {}, is_draft=lambda f: (True, 1.0),
                     tracker=tracker, clock=clock, attach=attach, audio=False)
    assert rc == 0
    assert [c["status"] for c in calls] == ["draft_complete"] * 3
    assert len(calls) == 3
    assert _read(state_file)["lolesports"]["status"] == "confirmed"


def test_lolesports_stops_after_the_attempt_cap(tmp_path):
    state_file = _state_file(tmp_path)
    complete = {"status": "draft_complete", "game_index": 1, "actions": [], "teams": {}}
    tracker = FakeTracker([complete] * 60)
    calls: list = []

    def attach(state):
        calls.append(state)
        return {**state, "lolesports": {"status": "partial"}}

    now = [0.0]

    def items():
        for i in range(45):
            now[0] += 31.0
            yield float(i), FakeFrame()

    rc = run_mod.run("src.mp4", state_file, av=items(),
                     read_frame=lambda f: {}, is_draft=lambda f: (True, 1.0),
                     tracker=tracker, clock=lambda: now[0], attach=attach, audio=False)
    assert rc == 0
    assert len(calls) == run_mod.LOLESPORTS_ATTEMPTS == 30
    assert _read(state_file)["lolesports"]["status"] == "partial"


def test_lolesports_never_runs_before_the_draft_is_complete(tmp_path):
    state_file = _state_file(tmp_path)
    tracker = FakeTracker([{"status": "drafting", "actions": []}] * 5)
    calls: list = []

    def attach(state):
        calls.append(state)
        return {**state, "lolesports": {"status": "confirmed"}}

    now = [0.0]
    rc = run_mod.run("src.mp4", state_file,
                     av=((float(i), FakeFrame()) for i in range(5)),
                     read_frame=lambda f: {}, is_draft=lambda f: (True, 1.0),
                     tracker=tracker, clock=lambda: now[0], attach=attach, audio=False)
    assert rc == 0
    assert calls == []
    assert _read(state_file)["lolesports"] is None


def test_a_slow_transcriber_never_delays_the_frame_loop(tmp_path):
    state_file = _state_file(tmp_path)
    tracker = FakeTracker([{"status": "drafting", "actions": []}])
    transcriber = FakeTranscriber(delay=0.5)

    def items():
        for i in range(12):
            yield "audio", float(i), PCM
        for i in range(6):
            yield "video", 100.0 + i, FakeFrame()

    started = time.monotonic()
    rc = run_mod.run("src.mp4", state_file, av=items(),
                     read_frame=lambda f: {}, is_draft=lambda f: (True, 1.0),
                     tracker=tracker, clock=lambda: 1000.0, transcriber=transcriber)
    elapsed = time.monotonic() - started
    assert rc == 0
    assert len(tracker.observations) == 6
    assert elapsed < 2.5, f"frame loop waited on speech-to-text ({elapsed:.1f}s)"


def test_no_audio_reads_video_only(tmp_path):
    state_file = _state_file(tmp_path)

    def items():
        yield "video", 1.0, FakeFrame()

    rc = run_mod.run("src.mp4", state_file, av=items(), audio=False,
                     read_frame=lambda f: {}, is_draft=lambda f: (True, 1.0),
                     tracker=FakeTracker([{"status": "drafting", "actions": []}]),
                     clock=lambda: 1000.0)
    assert rc == 0
    assert _read(state_file)["audio"] == {"enabled": False, "llm": False,
                                          "last_segment": None, "lag_s": None}


def test_main_forwards_the_no_audio_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVE_CAPTURE_HOME", str(tmp_path))
    seen = {}

    def fake_run(source, state_file, **kwargs):
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(run_mod, "run", fake_run)
    assert run_mod.main(["--source", "https://youtu.be/z", "--no-audio"]) == 0
    assert seen["audio"] is False
    assert run_mod.main(["--source", "https://youtu.be/z"]) == 0
    assert seen["audio"] is True


def test_interpreter_payload_keeps_the_key_out_of_logs(caplog):
    calls: list = []

    def http(url, payload, headers, timeout):
        calls.append({"payload": payload, "headers": headers, "timeout": timeout})
        raise TimeoutError("slow")

    with caplog.at_level(logging.DEBUG):
        interp = interpret_mod.Interpreter(api_key="do-not-leak", http=http, timeout=6.0)
        assert interp.interpret("they ban Ahri", {"session": "s", "game_index": 1}) is None
    assert len(calls) == 1
    payload, headers, timeout = calls[0]["payload"], calls[0]["headers"], calls[0]["timeout"]
    assert payload["max_tokens"] == 400
    assert payload["response_format"] == {"type": "json_object"}
    assert timeout == 6.0
    assert headers["Authorization"] == "Bearer do-not-leak"
    assert "do-not-leak" not in json.dumps(payload)
    assert "do-not-leak" not in caplog.text


def test_lolesports_confirmation_gets_the_session_team_names(tmp_path):
    state_file = _state_file(tmp_path)
    complete = {"status": "draft_complete", "game_index": 1, "actions": [], "teams": None}
    tracker = FakeTracker([complete] * 2)
    seen: list = []

    def attach(state):
        seen.append(state.get("teams"))
        return {**state, "lolesports": {"status": "confirmed"}}

    rc = run_mod.run("src.mp4", state_file, av=((float(i), FakeFrame()) for i in range(2)),
                     read_frame=lambda f: {}, is_draft=lambda f: (True, 1.0),
                     tracker=tracker, clock=lambda: 0.0, attach=attach, audio=False,
                     blue_team="Karmine Corp", red_team="MVK Esports",
                     teams_file=tmp_path / "no-teams.json")
    assert rc == 0
    assert seen and seen[0]["blue"]["name"] == "Karmine Corp" and seen[0]["red"]["name"] == "MVK Esports"


def _fake_speech(monkeypatch, cuda):
    from lol_live_capture import speech as speech_mod
    made = []
    monkeypatch.setattr(speech_mod, "cuda_available", lambda: cuda)
    monkeypatch.setattr(speech_mod, "Transcriber", lambda **kw: made.append(kw) or object())
    return made


def test_transcriber_auto_needs_cuda(monkeypatch):
    made = _fake_speech(monkeypatch, cuda=False)
    monkeypatch.delenv("LIVE_WHISPER_DEVICE", raising=False)
    assert run_mod._resolve_transcriber(None) is None and made == []
    made = _fake_speech(monkeypatch, cuda=True)
    assert run_mod._resolve_transcriber(None) is not None
    assert made == [{"model": "large-v3-turbo", "device": "cuda"}]


def test_transcriber_cpu_is_opt_in_and_model_is_overridable(monkeypatch):
    made = _fake_speech(monkeypatch, cuda=False)
    monkeypatch.setenv("LIVE_WHISPER_DEVICE", "cpu")
    assert run_mod._resolve_transcriber(None) is not None
    monkeypatch.setenv("LIVE_WHISPER_MODEL", "base.en")
    run_mod._resolve_transcriber(None)
    assert made == [{"model": "small.en", "device": "cpu", "compute_type": "int8"},
                    {"model": "base.en", "device": "cpu", "compute_type": "int8"}]
