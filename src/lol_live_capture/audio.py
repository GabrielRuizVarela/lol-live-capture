"""Audio source: local files and http(s) broadcast URLs, decoded with ffmpeg.

Mirrors :mod:`stream` for audio: local files go straight through ffmpeg, an
``http(s)`` page URL is pulled with ``yt-dlp -o -`` and piped into ffmpeg.
Audio is delivered as 16 kHz mono signed 16-bit little-endian chunks aligned
with the stream timestamps.  :func:`open_av` reads video *and* audio from one
``yt-dlp`` and one ``ffmpeg`` so a live stream is only pulled once.
"""
from __future__ import annotations

import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import Iterator, Union

import numpy as np

from .stream import _is_http_source, _kill_process_group, _yt_dlp_binary

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2  # int16


def _ffmpeg_audio_args(chunk_s: float) -> list[str]:
    return [
        "-vn",
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
    ]


def _read_chunks(
    stream,
    t0: float,
    chunk_s: float,
) -> Iterator[tuple[float, np.ndarray]]:
    chunk_bytes = max(1, int(round(chunk_s * SAMPLE_RATE))) * SAMPLE_WIDTH
    t = t0
    try:
        while True:
            buf = stream.read(chunk_bytes)
            if not buf:
                break
            if len(buf) % SAMPLE_WIDTH:
                buf = buf[: len(buf) - (len(buf) % SAMPLE_WIDTH)]
            pcm = np.frombuffer(buf, dtype=np.int16).copy()
            yield t, pcm
            t += len(pcm) / SAMPLE_RATE
    finally:
        pass


def open_audio(
    source: str,
    *,
    start: float | None = None,
    end: float | None = None,
    chunk_s: float = 1.0,
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield ``(t_seconds, int16 mono 16 kHz chunk)`` from a broadcast source.

    ``source`` is a local audio/video file or an ``http(s)`` page URL (YouTube
    or Twitch).  ``start``/``end`` trim a local file; an ``http(s)`` source is
    read live from its beginning.  Each chunk holds ``round(chunk_s * 16000)``
    samples except possibly the last.
    """
    if chunk_s <= 0:
        raise ValueError("chunk_s must be positive")
    t0 = float(start or 0.0)

    if _is_http_source(source):
        yt_dlp_cmd = [
            _yt_dlp_binary(),
            "--js-runtimes",
            "node",
            "-f",
            "bestaudio/best",
            "-o",
            "-",
            source,
        ]
        ffmpeg_cmd = ["ffmpeg", "-v", "error", "-i", "pipe:0"]
        ffmpeg_cmd += _ffmpeg_audio_args(chunk_s) + ["pipe:1"]
        log.info("streaming audio %s via yt-dlp | ffmpeg", source)
        feeder = subprocess.Popen(
            yt_dlp_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        assert feeder.stdout is not None
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=feeder.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        feeder.stdout.close()

        def _chunks() -> Iterator[tuple[float, np.ndarray]]:
            assert process.stdout is not None
            try:
                yield from _read_chunks(process.stdout, t0, chunk_s)
            finally:
                _kill_process_group(process)
                _kill_process_group(feeder)

        return _chunks()

    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"audio not found: {source}")
    cmd = ["ffmpeg", "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{float(start):.3f}"]
    if end is not None:
        cmd += ["-to", f"{float(end):.3f}"]
    cmd += ["-i", str(path)] + _ffmpeg_audio_args(chunk_s) + ["pipe:1"]
    log.info("reading audio %s with ffmpeg", path)
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True
    )

    def _chunks_local() -> Iterator[tuple[float, np.ndarray]]:
        assert process.stdout is not None
        try:
            yield from _read_chunks(process.stdout, t0, chunk_s)
        finally:
            _kill_process_group(process)

    return _chunks_local()


# An item yielded by :func:`open_av`: ``("video", t, frame)`` or
# ``("audio", t, pcm)``.
AVItem = tuple[str, float, Union[np.ndarray, object]]


def open_av(
    source: str,
    *,
    fps: float = 2.0,
    width: int = 1280,
    height: int = 720,
    start: float | None = None,
    end: float | None = None,
    chunk_s: float = 1.0,
    realtime: bool = False,
) -> Iterator[AVItem]:
    """Yield video frames and audio chunks from one ffmpeg (and one yt-dlp).

    Items are tagged: ``("video", t, HxWx3 uint8 BGR)`` and
    ``("audio", t, int16 mono 16 kHz)``.  Audio is decoded through an extra
    ffmpeg output pipe passed with ``pass_fds``; a background thread drains it
    so the video pipe cannot deadlock.  ``source`` may be a local file or an
    ``http(s)`` page URL (then ``start``/``end`` are ignored).  ``realtime``
    plays a local file at 1x (ffmpeg ``-re``), like a live stream.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    if chunk_s <= 0:
        raise ValueError("chunk_s must be positive")
    t0 = float(start or 0.0)
    frame_bytes = width * height * 3
    dt = 1.0 / fps
    afd, awfd = os.pipe()
    base = [
        "ffmpeg",
        "-v",
        "error",
    ]
    if not _is_http_source(source) and start is not None:
        base += ["-ss", f"{float(start):.3f}"]
    if not _is_http_source(source) and end is not None:
        base += ["-to", f"{float(end):.3f}"]
    if realtime and not _is_http_source(source):
        base += ["-re"]
    base += (
        [
            "-i",
            "pipe:0" if _is_http_source(source) else str(Path(source)),
            "-map",
            "0:v:0",
            "-vf",
            f"fps={fps},scale={width}:{height}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
            "-map",
            "0:a:0",
        ]
        + _ffmpeg_audio_args(chunk_s)
        + [f"pipe:{awfd}"]
    )

    feeder = None
    if _is_http_source(source):
        yt_dlp_cmd = [
            _yt_dlp_binary(),
            "--js-runtimes",
            "node",
            "-f",
            "best[height<=720]/bestvideo[height<=720]",
            "-o",
            "-",
            source,
        ]
        feeder = subprocess.Popen(
            yt_dlp_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        assert feeder.stdout is not None

    log.info("opening av %s with one ffmpeg", source)
    process = subprocess.Popen(
        base,
        stdin=feeder.stdout if feeder is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        pass_fds=(awfd,),
        start_new_session=True,
    )
    if feeder is not None:
        feeder.stdout.close()
    os.close(awfd)

    audio_stream = os.fdopen(afd, "rb", closefd=True)
    audio_q: list[tuple[float, np.ndarray]] = []
    lock = threading.Lock()

    def _audio_reader() -> None:
        try:
            for item in _read_chunks(audio_stream, t0, chunk_s):
                with lock:
                    audio_q.append(item)
        except Exception:  # pragma: no cover - pipe teardown race
            pass

    thread = threading.Thread(target=_audio_reader, daemon=True)
    thread.start()

    def _av() -> Iterator[AVItem]:
        assert process.stdout is not None
        t = t0
        try:
            while True:
                buf = process.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                with lock:
                    pending = audio_q[:]
                    audio_q.clear()
                for at, pcm in pending:
                    yield "audio", at, pcm
                yield "video", t, np.frombuffer(buf, dtype=np.uint8).reshape(
                    (height, width, 3)
                ).copy()
                t += dt
        finally:
            _kill_process_group(process)
            if feeder is not None:
                _kill_process_group(feeder)
            thread.join(timeout=1.0)

    return _av()
