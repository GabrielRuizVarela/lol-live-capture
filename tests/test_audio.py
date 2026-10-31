"""Tests for the audio / audio+video source."""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from lol_live_capture.audio import SAMPLE_RATE, open_audio, open_av
from lol_live_capture.stream import open_frames

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")


def _sine_wav(path: Path, seconds: float) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            str(path),
        ],
        check=True,
    )


def _av_mp4(path: Path, seconds: float) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=160x90:rate=10:duration={seconds}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
    )


def test_open_audio_chunks(tmp_path: Path) -> None:
    wav = tmp_path / "tone.wav"
    _sine_wav(wav, 3.0)
    chunks = list(open_audio(str(wav), chunk_s=1.0))
    assert [t for t, _ in chunks] == [0.0, 1.0, 2.0]
    assert all(pcm.dtype == np.int16 for _, pcm in chunks)
    assert all(len(pcm) == SAMPLE_RATE for _, pcm in chunks)


def test_open_audio_seek_and_trim(tmp_path: Path) -> None:
    wav = tmp_path / "tone.wav"
    _sine_wav(wav, 5.0)
    chunks = list(open_audio(str(wav), start=1.0, end=3.0, chunk_s=0.5))
    assert chunks[0][0] == 1.0
    assert len(chunks) == 4
    assert all(len(pcm) == SAMPLE_RATE // 2 for _, pcm in chunks)


def test_open_audio_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        list(open_audio("does-not-exist.m4a"))


def _elapsed(fn) -> float:
    started = time.monotonic()
    list(fn())
    return time.monotonic() - started


def test_realtime_plays_a_local_file_at_1x(tmp_path: Path) -> None:
    mp4 = tmp_path / "clip.mp4"
    _av_mp4(mp4, 3.0)
    fast = _elapsed(lambda: open_av(str(mp4), fps=4.0, width=160, height=90))
    slow = _elapsed(lambda: open_av(str(mp4), fps=4.0, width=160, height=90, realtime=True))
    fast_video = _elapsed(lambda: open_frames(str(mp4), fps=4.0, width=160, height=90))
    slow_video = _elapsed(lambda: open_frames(str(mp4), fps=4.0, width=160, height=90, realtime=True))
    assert slow >= 2.0 and slow_video >= 2.0
    assert slow > fast * 2 and slow_video > fast_video * 2


def test_open_av_reads_both(tmp_path: Path) -> None:
    mp4 = tmp_path / "av.mp4"
    _av_mp4(mp4, 4.0)
    items = list(open_av(str(mp4), fps=2.0, width=160, height=90, start=1, end=3))
    video = [i for i in items if i[0] == "video"]
    audio = [i for i in items if i[0] == "audio"]
    assert len(video) == 4
    assert len(audio) == 2
    assert video[0][2].shape == (90, 160, 3)
    assert audio[0][2].dtype == np.int16
    assert all(t >= 1.0 for _, t, _ in items)
