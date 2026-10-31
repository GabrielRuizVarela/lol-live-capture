"""Local speech-to-text with faster-whisper.

:class:`Transcriber` is fed 16 kHz mono int16 chunks (from
:mod:`lol_live_capture.audio`) as they arrive.  It keeps a rolling
buffer and, once about ``window_s`` of audio is available, transcribes the
window with voice-activity detection and a hotword bias toward champion names.
Words that fall in the overlap with the previous window are dropped, so no
word is emitted twice.  The emitted :class:`Segment` carries the absolute
stream times and the average log-probability.
"""
from __future__ import annotations

import gc
import logging
import time
import traceback
from dataclasses import dataclass
from typing import Iterable

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class Segment:
    """A transcribed span in stream time."""

    t0: float
    t1: float
    text: str
    avg_logprob: float


def cuda_available() -> bool:
    try:
        import ctranslate2  # type: ignore

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


class Transcriber:
    """Rolling-window faster-whisper transcriber.

    ``model`` / ``device`` / ``compute_type`` are passed straight to
    ``faster_whisper.WhisperModel``.  ``hotwords`` defaults to the champion
    names and caster aliases from :mod:`spotter`.
    """

    def __init__(
        self,
        model: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "float16",
        *,
        window_s: float = 4.0,
        overlap_s: float = 1.0,
        language: str = "en",
        hotwords: Iterable[str] | None = None,
        initial_prompt: str | None = None,
        beam_size: int = 1,
        model_obj: object | None = None,
    ) -> None:
        if window_s <= 0 or overlap_s < 0 or overlap_s >= window_s:
            raise ValueError("require 0 <= overlap_s < window_s")
        # GPU memory is shared with other local models: on a CUDA failure, step down to int8 on the GPU,
        # then to a small English model on the GPU, then on the CPU (see _fallback).
        self._fallbacks: list[tuple[str, str, str]] = []
        if model_obj is None:
            self._fallbacks = [(model, "cuda", "int8_float16"), ("small.en", "cuda", "float16"),
                               ("small.en", "cpu", "int8")] if device == "cuda" else []
            model_obj = self._load(model, device, compute_type)
        self._model = model_obj
        self._window = int(round(window_s * SAMPLE_RATE))
        self._overlap = int(round(overlap_s * SAMPLE_RATE))
        self._language = language
        self._beam_size = beam_size
        if hotwords is None and initial_prompt is None:
            try:
                from .spotter import champion_hotwords

                hotwords = champion_hotwords()
            except Exception:  # pragma: no cover - bias is best-effort
                hotwords = None
        self._hotwords = " ".join(hotwords) if hotwords else None
        self._initial_prompt = initial_prompt

        self._buf = np.zeros(0, dtype=np.int16)
        self._buf_t0 = 0.0
        self._last_word_end = -1.0
        self.audio_seconds = 0.0
        self.wall_seconds = 0.0
        self.latencies: list[float] = []
        self._feed_marks: list[tuple[float, float]] = []
        self._started = time.perf_counter()

    # -- feeding -----------------------------------------------------------
    def feed(self, t: float, pcm: np.ndarray) -> list[Segment]:
        """Add a chunk of 16 kHz mono int16 audio starting at stream time ``t``."""
        pcm = np.asarray(pcm, dtype=np.int16)
        now = time.perf_counter()
        self.wall_seconds = now - self._started
        if self._buf.size == 0:
            self._buf_t0 = float(t)
        self._buf = np.concatenate([self._buf, pcm]) if self._buf.size else pcm.copy()
        self._feed_marks.append((float(t) + pcm.size / SAMPLE_RATE, now))
        self.audio_seconds += pcm.size / SAMPLE_RATE

        out: list[Segment] = []
        step = self._window - self._overlap
        while self._buf.size >= self._window:
            window = self._buf[: self._window]
            out.extend(self._transcribe_window(window, self._buf_t0, now))
            self._buf = self._buf[step:]
            self._buf_t0 += step / SAMPLE_RATE
        return out

    def flush(self) -> list[Segment]:
        """Transcribe whatever is left in the buffer (end of a stream)."""
        if self._buf.size < SAMPLE_RATE // 2:
            return []
        out = self._transcribe_window(self._buf, self._buf_t0, time.perf_counter())
        self._buf = np.zeros(0, dtype=np.int16)
        return out

    # -- internals ---------------------------------------------------------
    def _transcribe_window(
        self, window: np.ndarray, t0: float, now: float
    ) -> list[Segment]:
        audio = window.astype(np.float32) / 32768.0
        try:
            return self._run(audio, t0, now)
        except RuntimeError as error:
            traceback.clear_frames(error.__traceback__)  # its frames still hold the failed model
            if not self._fallback(error):
                raise
            return self._run(audio, t0, now)

    @staticmethod
    def _load(model: str, device: str, compute_type: str) -> object:
        from faster_whisper import WhisperModel

        return WhisperModel(model, device=device, compute_type=compute_type)

    def _fallback(self, error: Exception) -> bool:
        """Swap to the next lighter model after a CUDA failure; False when none is left."""
        message = str(error).lower()
        if "cuda" not in message and "out of memory" not in message and "device" not in message:
            return False
        self._model = None  # free the failed model's GPU memory before loading the next one
        gc.collect()
        while self._fallbacks:
            model, device, compute_type = self._fallbacks.pop(0)
            try:
                self._model = self._load(model, device, compute_type)
            except Exception as load_error:  # noqa: BLE001 - try the next, lighter option
                log.warning("speech model %s/%s/%s failed to load: %s", model, device, compute_type, load_error)
                continue
            log.warning("speech: %s; now using %s on %s (%s)", error, model, device, compute_type)
            return True
        return False

    def _run(self, audio: np.ndarray, t0: float, now: float) -> list[Segment]:
        segments, _info = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=self._beam_size,
            vad_filter=True,
            word_timestamps=True,
            condition_on_previous_text=False,
            initial_prompt=self._initial_prompt,
            hotwords=self._hotwords,
        )
        out: list[Segment] = []
        for seg in segments:
            words = [
                w for w in (seg.words or []) if t0 + w.end > self._last_word_end + 0.01
            ]
            if not words:
                continue
            text = "".join(w.word for w in words).strip()
            if not text:
                continue
            st = t0 + words[0].start
            en = t0 + words[-1].end
            self._last_word_end = max(self._last_word_end, en)
            out.append(Segment(st, en, text, float(seg.avg_logprob)))
            self.latencies.append(max(0.0, now - self._wall_at(en)))
        return out

    def _wall_at(self, stream_t: float) -> float:
        wall = self._started
        for end_t, w in self._feed_marks:
            if end_t >= stream_t:
                return w
            wall = w
        return wall

    # -- metrics -----------------------------------------------------------
    @property
    def real_time_factor(self) -> float:
        """Audio seconds processed per wall second (higher is faster than real time)."""
        if self.wall_seconds <= 0:
            return 0.0
        return self.audio_seconds / self.wall_seconds

    @property
    def mean_latency(self) -> float:
        if not self.latencies:
            return 0.0
        return float(np.mean(self.latencies))
