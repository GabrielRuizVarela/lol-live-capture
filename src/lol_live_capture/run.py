"""Capture worker process: pull broadcast frames, track the draft, write CaptureState.

Contract: docs/contract.md ("Run model"). The worker is started by
the HTTP API (api.py), runs in its own
process group and atomically rewrites ``$LIVE_CAPTURE_HOME/run/capture-state.json``
after every change and at least every ``HEARTBEAT_S`` seconds.

Video and audio are read from the **one** stream (``audio.open_av``), so a live
broadcast is only pulled once.  Speech-to-text, the LLM transcript interpreter
and the lolesports confirmation all run off the frame loop: slow work never
stalls frame reading.  ``stream.py`` / ``layout.py`` / ``recognize.py`` /
``tracker.py`` are imported lazily inside functions so the API (and this
module's tests) work without the optional capture dependencies; every
dependency is injectable for tests.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from . import paths

log = logging.getLogger(__name__)

HEARTBEAT_S = 2.0
STATE_FIELDS = ("status", "source", "layout", "updated_at", "stream_t", "frame_age_ms",
                "game_index", "teams", "actions", "hover", "warnings", "error",
                "suggestions")
AUDIO_QUEUE_MAX = 8
SEGMENT_CHARS = 120
LLM_TAIL_S = 30.0
LOLESPORTS_INTERVAL_S = 30.0
LOLESPORTS_ATTEMPTS = 30
THREAD_JOIN_S = 5.0
TEAMS_RELOAD_S = 5.0


def home_dir() -> Path:
    """$LIVE_CAPTURE_HOME, read at call time so tests can override it via env."""
    return paths.home_dir()


def default_state_file() -> Path:
    return home_dir() / "run" / "capture-state.json"


def refs_dir() -> Path:
    return paths.refs_dir()


def idle_state(source: str | None = None) -> dict[str, Any]:
    """A CaptureState with every contract field present, for missing/dead state."""
    return {"status": "idle", "source": source, "layout": None, "updated_at": None,
            "stream_t": None, "frame_age_ms": None, "game_index": 1,
            "teams": {"blue": {"tricode": None}, "red": {"tricode": None}},
            "actions": [], "hover": None, "warnings": [], "error": None,
            "suggestions": [], "lolesports": None, "audio": None}


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).isoformat()


def normalize_state(raw: Any, *, source: str | None, status: str | None = None,
                    error: str | None = None) -> dict[str, Any]:
    """Merge a tracker-produced state over the contract skeleton."""
    state = idle_state(source)
    if isinstance(raw, Mapping):
        state.update({k: raw[k] for k in STATE_FIELDS if k in raw})
    if status is not None:
        state["status"] = status
    if error is not None:
        state["error"] = error
    if not isinstance(state.get("teams"), Mapping):
        state["teams"] = {"blue": {"tricode": None}, "red": {"tricode": None}}
    else:
        teams = dict(state["teams"])
        for side in ("blue", "red"):
            value = teams.get(side)
            teams[side] = dict(value) if isinstance(value, Mapping) else {"tricode": None}
        state["teams"] = teams
    state["source"] = source
    return state


def write_state(path: Path, state: Mapping[str, Any]) -> None:
    """Atomic write: to ``path + '.tmp'`` then ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    os.replace(tmp, path)


def teams_file_for(state_file: str | os.PathLike) -> Path:
    """The side-car teams file: ``capture-teams.json`` next to ``capture-state.json``."""
    return Path(state_file).with_name("capture-teams.json")


def read_teams_file(path: Path) -> dict[str, str | None]:
    """Team names from the side-car file, or ``{blue: None, red: None}`` when absent/broken."""
    names: dict[str, str | None] = {"blue": None, "red": None}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return names
    if not isinstance(raw, Mapping):
        return names
    for side in ("blue", "red"):
        team = raw.get(side)
        name = team.get("name") if isinstance(team, Mapping) else None
        if isinstance(name, str) and name:
            names[side] = name
    return names


def apply_team_names(state: Mapping[str, Any], names: Mapping[str, str | None]) -> None:
    """Store the app's team names next to the broadcast tricodes (which are left untouched)."""
    teams = state.get("teams")
    if not isinstance(teams, dict):
        teams = {"blue": {"tricode": None}, "red": {"tricode": None}}
        state["teams"] = teams  # type: ignore[index]
    for side in ("blue", "red"):
        name = names.get(side)
        if not name:
            continue
        entry = teams.get(side)
        if not isinstance(entry, dict):
            entry = {"tricode": None}
            teams[side] = entry
        entry["name"] = name


def _resolve_read_frame(read_frame: Callable | None, champions_dir: Path | None) -> Callable | None:
    if read_frame is not None:
        return read_frame
    root = champions_dir  # None: the default champion art directory

    from . import recognize
    bank: list = []  # loaded once, on the first draft frame

    def _read(frame: Any) -> Any:
        if not bank:
            bank.append(recognize.ReferenceBank.load(root, refs_dir()))
        return recognize.read_frame(frame, bank[0])

    return _read


def _resolve_tracker(tracker: Any, source: str) -> Any:
    if tracker is not None:
        return tracker
    from . import layout as layout_mod
    from . import tracker as tracker_mod
    return tracker_mod.DraftTracker(source=source, layout=layout_mod.LAYOUT_ID)


def _resolve_is_draft(is_draft: Callable | None) -> Callable:
    if is_draft is not None:
        return is_draft
    from . import layout as layout_mod
    return layout_mod.is_draft_screen


def _resolve_av(frames: Any, av: Any, source: str, fps: float, audio_on: bool,
                realtime: bool = False) -> Iterator:
    """An iterator of ``("video"|"audio", t, payload)`` items for the capture loop.

    ``av`` and ``frames`` are injected by tests; a real run reads video *and*
    audio from one yt-dlp/ffmpeg through :func:`audio.open_av`, or video alone
    when audio is off.  ``realtime`` makes a local file play at 1x (ffmpeg -re).
    """
    if av is not None:
        return (item if len(item) == 3 else ("video", item[0], item[1]) for item in av)
    if frames is not None:
        return (("video", t, frame) for t, frame in frames)
    if audio_on:
        from . import audio as audio_mod
        return audio_mod.open_av(source, fps=fps, realtime=realtime)
    from . import stream
    return (("video", t, frame) for t, frame in stream.open_frames(source, fps=fps, realtime=realtime))


def _resolve_transcriber(transcriber: Any) -> Any:
    """The speech-to-text engine, or None (logged, video-only) when unavailable.

    ``LIVE_WHISPER_DEVICE`` is ``auto`` (default: on only with a CUDA device), ``cuda`` or ``cpu`` (opt-in: it can lag
    behind a live stream); ``LIVE_WHISPER_MODEL`` overrides the model (``large-v3-turbo`` on CUDA, ``small.en`` on CPU).
    """
    if transcriber is not None:
        return transcriber
    try:
        from . import speech
        device = os.environ.get("LIVE_WHISPER_DEVICE", "auto").lower()
        if device == "auto":
            device = "cuda" if speech.cuda_available() else ""
        if device == "cuda":
            return speech.Transcriber(model=os.environ.get("LIVE_WHISPER_MODEL") or "large-v3-turbo", device="cuda")
        if device == "cpu":
            return speech.Transcriber(model=os.environ.get("LIVE_WHISPER_MODEL") or "small.en", device="cpu",
                                      compute_type="int8")
        raise RuntimeError("no CUDA device available (set LIVE_WHISPER_DEVICE=cpu to try the CPU)")
    except Exception as error:  # noqa: BLE001 - audio is optional, video is not
        log.warning("speech-to-text unavailable (%s); continuing video-only", error)
        return None


def _resolve_interpreter(interpreter: Any) -> Any:
    """The LLM transcript interpreter, or None when LIVE_LLM_API_KEY is unset."""
    if interpreter is not None:
        return interpreter
    if not os.environ.get("LIVE_LLM_API_KEY"):
        return None
    try:
        from . import interpret
        return interpret.Interpreter()
    except Exception as error:  # noqa: BLE001 - the rule-based spotter still runs
        log.warning("LLM interpreter unavailable (%s); spotter only", error)
        return None


def _tracker_state(tracker: Any) -> Any:
    """The tracker's current CaptureState, whichever getter it happens to have."""
    getter = getattr(tracker, "capture_state", None) or getattr(tracker, "state", None)
    if callable(getter):
        try:
            return getter()
        except Exception:  # noqa: BLE001 - a state snapshot is best-effort
            return None
    return None


class _AudioPipeline:
    """Speech-to-text, spotting and LLM interpretation off the frame loop.

    Audio chunks go into a bounded queue (the oldest chunk is dropped, and
    logged, when full) so a slow transcriber can never stall frame reading.
    Mentions are handed to ``tracker.observe_mentions`` when the tracker has it
    (older trackers without it are supported: the mentions are then skipped).
    """

    def __init__(self, *, transcriber: Any, interpreter: Any, tracker: Any,
                 tracker_lock: threading.Lock, info: dict[str, Any],
                 dirty: dict[str, bool], clock: Callable[[], float],
                 stream_t: Callable[[], float | None],
                 state: Callable[[], Any]) -> None:
        from . import spotter  # stdlib only; lazy so importing run.py stays cheap

        self._spotter = spotter
        self._transcriber = transcriber
        self._interpreter = interpreter
        self._tracker = tracker
        self._lock = tracker_lock
        self._info = info
        self._dirty = dirty
        self._clock = clock
        self._stream_t = stream_t
        self._state = state
        self._q: queue.Queue = queue.Queue(maxsize=AUDIO_QUEUE_MAX)
        self._llm_q: queue.Queue = queue.Queue(maxsize=1)
        self._transcript: list[tuple[float, str]] = []
        self._closed = False
        self._dropped = 0
        self._missing_mentions = False
        self._session = str(int(self._clock()))
        self._threads: list[threading.Thread] = []

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        workers = [(self._audio_loop, "capture-speech")]
        if self._interpreter is not None:
            workers.append((self._llm_loop, "capture-llm"))
        for target, name in workers:
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)

    def close(self) -> None:
        self._closed = True
        for q in (self._q, self._llm_q):
            while True:
                try:
                    q.put_nowait(None)
                    break
                except queue.Full:
                    try:
                        q.get_nowait()
                    except queue.Empty:  # pragma: no cover - consumer drained it
                        break
        for thread in self._threads:
            thread.join(timeout=THREAD_JOIN_S)
            if thread.is_alive():  # pragma: no cover - only a stuck engine
                log.warning("capture thread %s did not stop", thread.name)

    def snapshot(self) -> dict[str, Any]:
        return dict(self._info)

    def take_dirty(self) -> bool:
        changed = self._dirty["value"]
        self._dirty["value"] = False
        return changed

    # -- producer (the frame loop; must never block) ------------------------
    def push(self, t: float, pcm: Any) -> None:
        if self._closed:
            return
        try:
            self._q.put_nowait((float(t), pcm))
        except queue.Full:
            try:
                self._q.get_nowait()
                self._dropped += 1
                log.warning("audio queue full: dropped the oldest chunk (%d dropped, stream t=%.1fs)",
                            self._dropped, t)
            except queue.Empty:  # pragma: no cover - consumer drained it
                pass
            try:
                self._q.put_nowait((float(t), pcm))
            except queue.Full:  # pragma: no cover - consumer drained it
                pass

    # -- workers ------------------------------------------------------------
    def _audio_loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            if self._closed:
                continue
            t, pcm = item
            try:
                segments = self._transcriber.feed(t, pcm)
            except Exception as error:  # noqa: BLE001 - one bad window must not kill audio
                log.warning("transcription failed: %s", error)
                continue
            if not segments:
                continue
            self._note(segments)
            self._spot(segments)
            if self._interpreter is not None:
                self._offer_llm(segments)

    def _note(self, segments: list[Any]) -> None:
        last = max(segments, key=lambda seg: getattr(seg, "t1", 0.0))
        text = str(getattr(last, "text", "") or "").strip()
        live = self._stream_t()
        t1 = getattr(last, "t1", None)
        lag = None if live is None or t1 is None else max(0.0, float(live) - float(t1))
        with self._lock:
            self._info["last_segment"] = text[:SEGMENT_CHARS]
            self._info["lag_s"] = None if lag is None else round(lag, 2)
            self._dirty["value"] = True
            ends = [float(getattr(seg, "t1", 0.0)) for seg in segments]
            cutoff = max(ends) - LLM_TAIL_S
            self._transcript = [(t, s) for t, s in self._transcript if t >= cutoff]
            self._transcript.extend(
                (float(getattr(seg, "t1", 0.0)), str(getattr(seg, "text", "") or "").strip())
                for seg in segments if str(getattr(seg, "text", "") or "").strip())

    def _spot(self, segments: list[Any]) -> None:
        try:
            mentions = self._spotter.spot_segments(segments)
        except Exception as error:  # noqa: BLE001 - spotting is best-effort
            log.warning("champion spotting failed: %s", error)
            return
        self._deliver(mentions)

    def _offer_llm(self, segments: list[Any]) -> None:
        last = max(segments, key=lambda seg: getattr(seg, "t1", 0.0))
        item = (float(getattr(last, "t1", 0.0)), segments)
        try:
            self._llm_q.put_nowait(item)
        except queue.Full:
            try:
                self._llm_q.get_nowait()
            except queue.Empty:  # pragma: no cover - consumer drained it
                pass
            try:
                self._llm_q.put_nowait(item)
            except queue.Full:  # pragma: no cover - consumer drained it
                pass

    def _llm_loop(self) -> None:
        while True:
            item = self._llm_q.get()
            if item is None:
                return
            if self._closed:
                continue
            base_t, segments = item
            tail = self._tail()
            try:
                events = self._interpreter.interpret(tail, self._context())
            except Exception as error:  # noqa: BLE001 - fallback is the spotter, already run
                log.warning("interpreter call failed: %s", error)
                continue
            mentions = []
            for event in events or []:
                if getattr(event, "action", None) not in ("ban", "pick"):
                    continue
                champion_id = getattr(event, "champion_id", None)
                if not champion_id:
                    continue
                mentions.append(self._spotter.Mention(
                    t=float(base_t), champion_id=str(champion_id),
                    cue=str(event.action), text=tail[:SEGMENT_CHARS],
                    score=float(getattr(event, "confidence", 0.0) or 0.0)))
            self._deliver(mentions)

    def _tail(self) -> str:
        if not self._transcript:
            return ""
        cutoff = self._transcript[-1][0] - LLM_TAIL_S
        return " ".join(text for t, text in self._transcript if t >= cutoff).strip()

    def _context(self) -> dict[str, Any]:
        with self._lock:
            state = _tracker_state(self._tracker) or {}
            session = self._session
        actions = state.get("actions") if isinstance(state, Mapping) else None
        used = [a.get("champion_id") for a in (actions or [])
                if isinstance(a, Mapping) and a.get("champion_id")]
        return {"session": session,
                "game_index": (state.get("game_index") if isinstance(state, Mapping) else None) or 1,
                "teams": state.get("teams") if isinstance(state, Mapping) else None,
                "used": used}

    def _deliver(self, mentions: list[Any]) -> None:
        if not mentions:
            return
        observe = getattr(self._tracker, "observe_mentions", None)
        if not callable(observe):
            if not self._missing_mentions:
                self._missing_mentions = True
                log.info("tracker has no observe_mentions; %d audio mentions ignored", len(mentions))
            return
        with self._lock:
            observe(mentions)
            self._dirty["value"] = True


def _outcome_state(outcome: Any, tracker: Any) -> tuple[bool, Any]:
    """ tracker.observe() returns either a CaptureState mapping, a (changed, state)
    tuple, or a plain truthy "changed" flag with the state on the tracker. """
    if isinstance(outcome, Mapping):
        return True, outcome
    if isinstance(outcome, tuple) and len(outcome) == 2:
        return bool(outcome[0]), outcome[1]
    getter = getattr(tracker, "capture_state", None) or getattr(tracker, "state", None)
    state = getter() if callable(getter) else None
    return bool(outcome), state


def _apply_tricodes(state: dict[str, Any], frame: Any, tricodes: Callable | None) -> None:
    if tricodes is None:
        return
    try:
        codes = tricodes(frame)
    except Exception:
        return
    if isinstance(codes, Mapping):
        for side in ("blue", "red"):
            if codes.get(side) is not None:
                state["teams"][side]["tricode"] = codes[side]


def run(source: str, state_file: str | os.PathLike, *, frames: Any = None,
        read_frame: Callable | None = None, is_draft: Callable | None = None,
        tricodes: Callable | None = None, tracker: Any = None,
        clock: Callable[[], float] = time.time, fps: float = 2.0,
        champions_dir: str | os.PathLike | None = None, av: Any = None,
        audio: bool = True, transcriber: Any = None, interpreter: Any = None,
        attach: Callable | None = None, blue_team: str | None = None,
        red_team: str | None = None, teams_file: str | os.PathLike | None = None,
        realtime: bool = False) -> int:
    """The real capture loop. Returns 0 on SIGTERM/idle exit, 1 on error.

    ``av`` yields ``("video"|"audio", t, payload)`` items (one read of the
    stream); ``frames`` stays video-only for callers that already have one.
    ``transcriber`` / ``interpreter`` / ``attach`` are the speech engine, the
    LLM transcript interpreter and ``lolesports.attach_confirmation``.
    """
    state_path = Path(state_file)
    root = Path(champions_dir) if champions_dir is not None else None
    teams_path = Path(teams_file) if teams_file is not None else teams_file_for(state_path)
    team_names: dict[str, str | None] = {"blue": blue_team, "red": red_team}
    last_teams_read: float | None = None

    def _refresh_teams(now: float, *, force: bool = False) -> None:
        nonlocal last_teams_read
        if not force and last_teams_read is not None and now - last_teams_read < TEAMS_RELOAD_S:
            return
        last_teams_read = now
        for side, name in read_teams_file(teams_path).items():
            if name:
                team_names[side] = name

    stopped = {"value": False}
    previous_handler = signal.getsignal(signal.SIGTERM)

    def _on_sigterm(_signum, _frame):
        stopped["value"] = True

    signal.signal(signal.SIGTERM, _on_sigterm)

    audio_info: dict[str, Any] = {"enabled": False, "llm": False,
                                  "last_segment": None, "lag_s": None}
    lolesports_result: dict[str, Any] | None = None
    pipe: _AudioPipeline | None = None

    def _write(state: Mapping[str, Any], last_frame_wall: float | None) -> None:
        payload = dict(state)
        apply_team_names(payload, team_names)
        payload["updated_at"] = _iso(clock())
        payload["frame_age_ms"] = None if last_frame_wall is None else round((clock() - last_frame_wall) * 1000, 1)
        payload["audio"] = pipe.snapshot() if pipe is not None else dict(audio_info)
        payload["lolesports"] = lolesports_result
        write_state(state_path, payload)

    last_frame_wall: float | None = None
    try:
        want_audio = bool(audio) and (av is not None or frames is None)
        transcriber_obj = _resolve_transcriber(transcriber) if want_audio else None
        if want_audio and transcriber_obj is None:
            want_audio = False
        interpreter_obj = _resolve_interpreter(interpreter)
        if interpreter_obj is not None and not getattr(interpreter_obj, "enabled", True):
            interpreter_obj = None
        audio_info["enabled"] = want_audio
        audio_info["llm"] = bool(want_audio and interpreter_obj is not None)

        frames_iter = _resolve_av(frames, av, source, fps, want_audio, realtime)
        is_draft_fn = _resolve_is_draft(is_draft)
        read_frame_fn = _resolve_read_frame(read_frame, root)
        tracker_obj = _resolve_tracker(tracker, source)
        tracker_lock = threading.Lock()
        dirty = {"value": False}
        stream_ref: dict[str, float | None] = {"t": None}

        def _state_snapshot() -> Any:
            with tracker_lock:
                return _tracker_state(tracker_obj)

        if want_audio:
            pipe = _AudioPipeline(transcriber=transcriber_obj, interpreter=interpreter_obj,
                                  tracker=tracker_obj, tracker_lock=tracker_lock,
                                  info=audio_info, dirty=dirty, clock=clock,
                                  stream_t=lambda: stream_ref["t"], state=_state_snapshot)
            pipe.start()
        _refresh_teams(clock(), force=True)
        _write(normalize_state(None, source=source, status="connecting"), None)
        last_write = clock()
        previous_raw: Any = None
        attach_fn = attach
        attach_tried = False
        attempts = 0
        last_attempt: float | None = None
        confirmed = False
        for item in frames_iter:
            if stopped["value"]:
                break
            kind, t, payload_item = item
            if kind == "audio":
                if pipe is not None:
                    pipe.push(t, payload_item)
                continue
            frame = payload_item
            last_frame_wall = clock()
            _refresh_teams(last_frame_wall)
            stream_ref["t"] = t
            draft, _score = is_draft_fn(frame)
            reads = read_frame_fn(frame) if draft else None
            with tracker_lock:
                outcome = tracker_obj.observe(t, bool(draft), reads)
            changed, raw = _outcome_state(outcome, tracker_obj)
            if raw is not None:
                changed = changed or raw != previous_raw
                previous_raw = raw
            state = normalize_state(raw, source=source)
            if state.get("stream_t") is None:
                state["stream_t"] = t
            if state.get("status") in (None, "idle", "connecting"):
                state["status"] = "drafting" if draft else "waiting_for_draft"
            _apply_tricodes(state, frame, tricodes)
            apply_team_names(state, team_names)  # lolesports matches by name when no tricode is read
            now = clock()
            if state.get("status") == "draft_complete" and not confirmed and attempts < LOLESPORTS_ATTEMPTS:
                if attach_fn is None and not attach_tried:
                    attach_tried = True
                    try:
                        from . import lolesports as lolesports_mod
                        attach_fn = lolesports_mod.attach_confirmation
                    except Exception as error:  # noqa: BLE001 - confirmation is optional
                        log.warning("lolesports unavailable (%s)", error)
                if attach_fn is not None and (last_attempt is None or now - last_attempt >= LOLESPORTS_INTERVAL_S):
                    attempts += 1
                    last_attempt = now
                    try:
                        result = attach_fn(state)
                    except Exception as error:  # noqa: BLE001 - network errors go to the log only
                        log.warning("lolesports confirmation failed: %s", error)
                        result = None
                    found = result.get("lolesports") if isinstance(result, Mapping) else None
                    if isinstance(found, Mapping):
                        lolesports_result = dict(found)
                        confirmed = lolesports_result.get("status") in ("confirmed", "mismatch")
                        log.info("lolesports attempt %d/%d: %s", attempts, LOLESPORTS_ATTEMPTS,
                                 lolesports_result.get("status"))
                        changed = True
            if pipe is not None and pipe.take_dirty():
                changed = True
            if changed or now - last_write >= HEARTBEAT_S:
                _write(state, last_frame_wall)
                last_write = now
        if stopped["value"]:
            # keep the last draft readable after Stop; a new start clears it
            _write(normalize_state(previous_raw, source=source, status="idle"), last_frame_wall)
        return 0
    except Exception as error:  # noqa: BLE001 - the worker must always report why it died
        _write(normalize_state(None, source=source, status="error", error=str(error)), last_frame_wall)
        return 1
    finally:
        if pipe is not None:
            pipe.close()
        signal.signal(signal.SIGTERM, previous_handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lol_live_capture.run",
                                     description="Live draft capture worker (docs/contract.md).")
    parser.add_argument("--source", required=True, help="stream URL or local video file")
    parser.add_argument("--state-file", default=None,
                        help="CaptureState output path (default: $LIVE_CAPTURE_HOME/run/capture-state.json)")
    parser.add_argument("--fps", type=float, default=2.0, help="frames per second pulled from the stream")
    parser.add_argument("--champions-dir", default=None,
                        help="champion art for the reference bank (default: $LIVE_CAPTURE_HOME/champions)")
    parser.add_argument("--no-audio", action="store_true",
                        help="read video only (no stream audio, no speech-to-text)")
    parser.add_argument("--blue-team", default=None, help="blue side team name (for lolesports)")
    parser.add_argument("--red-team", default=None, help="red side team name (for lolesports)")
    parser.add_argument("--realtime", action="store_true",
                        help="play a local file at 1x (ffmpeg -re), like a live stream")
    args = parser.parse_args(argv)
    state_file = Path(args.state_file) if args.state_file else default_state_file()
    return run(args.source, state_file, fps=args.fps, champions_dir=args.champions_dir,
               audio=not args.no_audio, blue_team=args.blue_team, red_team=args.red_team,
               realtime=args.realtime)


if __name__ == "__main__":
    sys.exit(main())
