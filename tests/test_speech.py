"""Tests for the rolling-window transcriber (model injected, no GPU)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from lol_live_capture.speech import SAMPLE_RATE, Segment, Transcriber


@dataclass
class FakeWord:
    start: float
    end: float
    word: str


@dataclass
class FakeSegment:
    start: float
    end: float
    text: str
    words: list[FakeWord]
    avg_logprob: float = -0.2


class FakeModel:
    def __init__(self, words: list[FakeWord]) -> None:
        self.words = words
        self.calls = 0

    def transcribe(self, audio, **kwargs):  # noqa: ANN001, ANN003
        self.calls += 1
        return iter([FakeSegment(0.0, 4.0, "x", list(self.words))]), object()


def _transcriber(words: list[FakeWord]) -> tuple[Transcriber, FakeModel]:
    fake = FakeModel(words)
    tr = Transcriber(model_obj=fake, window_s=4.0, overlap_s=1.0, hotwords=["x"])
    return tr, fake


def test_feed_emits_segment_with_absolute_times() -> None:
    tr, _ = _transcriber([FakeWord(0.5, 1.5, " hello"), FakeWord(2.5, 3.5, " world")])
    out: list[Segment] = []
    for i in range(8):
        out += tr.feed(float(i), np.zeros(SAMPLE_RATE, dtype=np.int16))
    assert out
    assert out[0].t0 == pytest.approx(0.5)
    assert out[0].t1 == pytest.approx(3.5)
    assert out[0].text == "hello world"
    assert out[0].avg_logprob == pytest.approx(-0.2)


def test_overlap_deduplicates_words() -> None:
    # Words already covered by a previous window (absolute end <= last_word_end)
    # are dropped; only the genuinely new tail is emitted.
    tr, _ = _transcriber([FakeWord(0.0, 0.5, " old"), FakeWord(3.0, 3.5, " new")])
    tr._last_word_end = 2.0
    out: list[Segment] = []
    for i in range(4):
        out += tr.feed(float(i), np.zeros(SAMPLE_RATE, dtype=np.int16))
    assert len(out) == 1
    assert out[0].text == "new"
    assert out[0].t0 == pytest.approx(3.0)
    assert out[0].t1 == pytest.approx(3.5)


def test_metrics_track_progress() -> None:
    tr, _ = _transcriber([FakeWord(0.0, 1.0, " ok")])
    for i in range(8):
        tr.feed(float(i), np.zeros(SAMPLE_RATE, dtype=np.int16))
    assert tr.audio_seconds == pytest.approx(8.0)
    assert tr.real_time_factor > 0
    assert tr.latencies


def test_rejects_bad_window() -> None:
    with pytest.raises(ValueError):
        Transcriber(model_obj=object(), window_s=1.0, overlap_s=1.0)


def test_cuda_available_is_bool() -> None:
    from lol_live_capture.speech import cuda_available

    assert isinstance(cuda_available(), bool)
