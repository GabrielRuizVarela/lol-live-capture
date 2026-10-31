# lol-live-capture

Reads a League of Legends **pick/ban draft from a broadcast stream while it happens** and serves it as JSON over an
HTTP API and a live web UI. It watches the champion-select overlay (computer vision), optionally listens to the
casters (speech-to-text, optionally an LLM) to settle look-alikes, and cross-checks the result with the lolesports
feed after the draft.

```
stream (YouTube / Twitch / file) ──▶ frames ──▶ slot reads ──▶ tracker ──▶ CaptureState ──▶ API + SSE ──▶ UI
                               └──▶ audio ──▶ whisper ──▶ champion mentions ─┘
```

Output is the 20 draft actions in order (`sequence`, `side`, `action_type`, `champion_id`, `confidence`, `via`), the
champion currently hovered, warnings, and a lolesports verdict. See [docs/contract.md](docs/contract.md) for the rules
and the exact schema.

> **Status: early.** The overlay layout is calibrated for the Worlds 2026 broadcast (`worlds-2026-1280x720-v1`). Other
> broadcasts need their own layout in `layout.py`. Image-only accuracy was 94.3% exact on 7 recorded drafts; treat
> `confidence` and `via` as the trust signal ([Measured accuracy](docs/contract.md#measured-accuracy)).

## Quick start

Needs Python 3.11+, `ffmpeg` on the PATH, and Node 20+ only if you rebuild the UI.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"           # add ".[audio]" for speech-to-text
lol-live-capture setup            # downloads champion art for the latest patch (~14 MB, ~1 min)
lol-live-capture serve            # http://127.0.0.1:8040
```

The UI is prebuilt only if you build it: `cd ui && npm install && npm run build` (writes into the Python package).
Open <http://127.0.0.1:8040>, paste a YouTube or Twitch address, press **Start reading**. `/?demo` runs the UI against
an invented draft with no stream.

Replay a saved broadcast: put the video under `~/.local/share/lol-live-capture/capture/` and paste its path.

Headless: `lol-live-capture run --source URL --state-file out.json` writes `CaptureState` to that file.

## API

All under `/api`; no authentication, so keep the default `127.0.0.1` bind (the API starts processes and reads files).

| | |
|---|---|
| `POST /capture/start` `{source, teams?}` | start the worker (YouTube/Twitch URL, or a file under the capture directory) |
| `POST /capture/stop` | stop it |
| `GET /capture/state` | current `CaptureState` |
| `GET /capture/events` | SSE, `event: state` on every change |
| `GET /champions`, `GET /champions/{ddragon_id}/{square\|loading}` | key → name and the downloaded art |
| `GET /health` | status and whether champion art is installed |

## Configuration

| Variable | Meaning |
|---|---|
| `LIVE_CAPTURE_HOME` | data directory (default `~/.local/share/lol-live-capture`) |
| `LIVE_CAPTURE_PYTHON` | interpreter for the worker, if it lives in its own venv |
| `LIVE_WHISPER_DEVICE` / `LIVE_WHISPER_MODEL` | `auto` (CUDA only), `cuda` or `cpu`; model override |
| `LIVE_LLM_API_KEY`, `LIVE_LLM_BASE_URL`, `LIVE_LLM_MODEL`, `LIVE_LLM_EXTRA_BODY` | optional LLM reading of the transcript (any OpenAI-compatible endpoint) |
| `LOLESPORTS_API_KEY` | override the lolesports public key |

## Development

```bash
pytest                      # backend; tests needing Riot footage or champion art skip without them
cd ui && npm test           # UI
cd ui && npm run dev        # UI on :5174, proxying /api to :8040
```

Recognizer and layout tests use broadcast stills kept out of the repository: put them in `tests/fixtures/private/`
(`s2520.jpg`, `holdout-*-band.jpg`) after running `setup`.

## License

MIT, see [LICENSE](LICENSE). Read [NOTICE.md](NOTICE.md) first: Riot's art, the unofficial lolesports API and stream
reading have terms of their own.
