"""Frame source: local files and http(s) broadcast URLs, decoded with ffmpeg."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

import numpy as np

log = logging.getLogger(__name__)


def _yt_dlp_binary() -> str:
    sibling = Path(sys.executable).parent / "yt-dlp"
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling)
    found = shutil.which("yt-dlp")
    if found:
        return found
    return str(sibling)


def _is_http_source(source: str) -> bool:
    return urlparse(source).scheme in ("http", "https")


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        try:
            os.killpg(os.getpgid(process.pid), 9)
        except (ProcessLookupError, PermissionError, OSError):
            process.kill()
        process.wait()


def _read_frames(
    process: subprocess.Popen[bytes],
    t0: float,
    frame_bytes: int,
    fps: float,
    width: int,
    height: int,
) -> Iterator[tuple[float, np.ndarray]]:
    stream = process.stdout
    assert stream is not None
    dt = 1.0 / fps
    t = t0
    try:
        while True:
            buf = stream.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield t, np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 3)).copy()
            t += dt
    finally:
        _kill_process_group(process)


def open_frames(
    source: str,
    *,
    fps: float = 2.0,
    width: int = 1280,
    height: int = 720,
    start: float | None = None,
    end: float | None = None,
    realtime: bool = False,
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield ``(t_seconds, BGR frame)`` from a local video file or an http(s) page URL.

    Frames are pulled with ffmpeg at ``fps`` frames per second, resized to
    ``width`` x ``height``.  ``start``/``end`` trim a local file; an http(s)
    source is always read live from its beginning (``start``/``end`` are
    ignored).  ``realtime`` plays a local file at 1x (ffmpeg ``-re``), like a
    live stream.  Each yielded frame is a fresh ``uint8`` array of shape
    ``(height, width, 3)`` in BGR order.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    frame_bytes = width * height * 3
    seek = f"fps={fps},scale={width}:{height}"
    t0 = float(start or 0.0)

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
        ffmpeg_cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            "pipe:0",
            "-vf",
            seek,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ]
        log.info("streaming %s via yt-dlp | ffmpeg", source)
        feeder = subprocess.Popen(
            yt_dlp_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True
        )
        assert feeder.stdout is not None
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=feeder.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        feeder.stdout.close()  # ffmpeg owns the pipe; close our copy

        def _frames() -> Iterator[tuple[float, np.ndarray]]:
            try:
                yield from _read_frames(process, t0, frame_bytes, fps, width, height)
            finally:
                _kill_process_group(feeder)

        return _frames()

    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"video not found: {source}")
    ffmpeg_cmd = ["ffmpeg", "-v", "error"]
    if start is not None:
        ffmpeg_cmd += ["-ss", f"{float(start):.3f}"]
    if end is not None:
        ffmpeg_cmd += ["-to", f"{float(end):.3f}"]
    if realtime:
        ffmpeg_cmd += ["-re"]
    ffmpeg_cmd += ["-i", str(path), "-vf", seek, "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    log.info("reading %s with ffmpeg", path)
    process = subprocess.Popen(
        ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True
    )
    return _read_frames(process, t0, frame_bytes, fps, width, height)
